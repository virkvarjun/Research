"""Hardware runtime scaffold for shadow, alarm-only, and intervention modes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from faact.backbone.base import BackbonePolicyWrapper
from faact.backbone.features import chunk_to_numpy, merge_feature_dicts
from failure_prediction.interfaces import InterventionPolicy, RiskScorer

from .config import HardwareExperimentConfig


class ObservationAdapter(Protocol):
    """Provides normalized hardware observations for the backbone wrapper."""

    def reset(self) -> None:
        ...

    def get_observation(self) -> dict[str, np.ndarray | dict[str, np.ndarray]] | None:
        ...


class HardwareRobot(Protocol):
    """Thin action sink for the physical robot."""

    def arm(self) -> bool:
        ...

    def stop(self) -> None:
        ...

    def execute_action(self, action: np.ndarray) -> None:
        ...


@dataclass
class StepRecord:
    step: int
    mode: str
    alarmed: bool
    accepted: bool
    reason: str
    risk_prob: float | None
    intervention_attempted: bool


class DryRunRobot:
    """Safe default robot sink that only logs intended actions."""

    def __init__(self) -> None:
        self.executed_actions: list[np.ndarray] = []

    def arm(self) -> bool:
        return True

    def stop(self) -> None:
        return None

    def execute_action(self, action: np.ndarray) -> None:
        self.executed_actions.append(np.asarray(action, dtype=np.float32).copy())


class HardwareRuntime:
    """Safety-first runtime for FAACT experiments on physical hardware."""

    def __init__(
        self,
        config: HardwareExperimentConfig,
        backbone: BackbonePolicyWrapper,
        observation_adapter: ObservationAdapter,
        robot: HardwareRobot,
        risk_scorer: RiskScorer | None = None,
        intervention_policy: InterventionPolicy | None = None,
    ) -> None:
        self.config = config
        self.backbone = backbone
        self.observation_adapter = observation_adapter
        self.robot = robot
        self.risk_scorer = risk_scorer
        self.intervention_policy = intervention_policy
        self.log_dir = Path(config.runtime.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.log_dir / "hardware_episode.jsonl"

    def _validate_action(self, action: np.ndarray) -> bool:
        max_abs = float(np.max(np.abs(action))) if action.size > 0 else 0.0
        return max_abs <= self.config.safety.max_abs_action_value

    def _write_record(self, record: StepRecord) -> None:
        with open(self.records_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def run_episode(self) -> dict[str, Any]:
        if not self.robot.arm():
            raise RuntimeError("Robot failed to arm")

        self.observation_adapter.reset()
        self.backbone.reset(task_spec=self.config.backbone.task_desc)

        current_chunk: np.ndarray | None = None
        current_features_raw: dict[str, np.ndarray] = {}
        chunk_step_idx = 0
        accepted_interventions = 0
        alarm_steps = 0

        for step in range(self.config.runtime.max_steps):
            raw_obs = self.observation_adapter.get_observation()
            if raw_obs is None:
                if self.config.safety.halt_on_missing_observation:
                    self.robot.stop()
                    raise RuntimeError("Missing hardware observation")
                break

            need_new_chunk = current_chunk is None or chunk_step_idx >= (
                self.config.backbone.replan_interval or self.backbone.chunk_size
            )
            if need_new_chunk:
                proposal = self.backbone.propose_chunk(
                    raw_obs,
                    context={"task": self.config.backbone.task_desc},
                    return_features=True,
                )
                current_chunk = chunk_to_numpy(proposal.actions)
                current_features_raw = (
                    dict(proposal.features.raw) if proposal.features and proposal.features.raw else {}
                )
                chunk_step_idx = 0

            runtime_features = merge_feature_dicts(
                current_features_raw,
                current_chunk,
                chunk_step_idx=chunk_step_idx,
            )
            risk_score = self.risk_scorer.predict_step(runtime_features) if self.risk_scorer else None
            decision = (
                self.intervention_policy.should_interrupt(
                    risk_score=risk_score,
                    step=step,
                    need_new_chunk=need_new_chunk,
                    accepted_interventions_so_far=accepted_interventions,
                )
                if self.intervention_policy and risk_score is not None
                else None
            )

            accepted = False
            alarmed = bool(decision.should_interrupt) if decision is not None else False
            reason = decision.reason if decision is not None else ""
            if alarmed:
                alarm_steps += 1

            # Hardware experiments start in shadow/alarm mode. Intervention is opt-in.
            if self.config.runtime.mode == "intervene" and alarmed:
                accepted = False
                reason = "intervention_not_implemented_yet"

            action = np.asarray(current_chunk[chunk_step_idx], dtype=np.float32)
            if not self._validate_action(action):
                self.robot.stop()
                if self.config.safety.halt_on_invalid_chunk:
                    raise RuntimeError("Action failed safety validation")
                break
            self.robot.execute_action(action)

            self._write_record(
                StepRecord(
                    step=step,
                    mode=self.config.runtime.mode,
                    alarmed=alarmed,
                    accepted=accepted,
                    reason=reason,
                    risk_prob=risk_score.prob if risk_score is not None else None,
                    intervention_attempted=alarmed,
                )
            )
            chunk_step_idx += 1

        self.robot.stop()
        return {
            "mode": self.config.runtime.mode,
            "log_dir": str(self.log_dir),
            "accepted_interventions": accepted_interventions,
            "alarm_steps": alarm_steps,
            "dry_run": self.config.robot.dry_run,
        }
