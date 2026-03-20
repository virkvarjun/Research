#!/usr/bin/env python3
"""Run FAACT hardware scaffold in alarm-only mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR))
sys.path.insert(0, str(RESEARCH_DIR / "faact"))

from faact.backbone.factory import make_backbone_wrapper
from faact_hardware.config import load_config
from faact_hardware.runtime import DryRunRobot, HardwareRuntime
from failure_prediction.runtime_components import (
    ThresholdInterventionPolicy,
    load_supervised_risk_runtime,
)


class DummyObservationAdapter:
    """Placeholder adapter until real cameras and state bridges are connected."""

    def reset(self) -> None:
        return None

    def get_observation(self) -> dict[str, np.ndarray] | None:
        return {
            "observation.state": np.zeros(14, dtype=np.float32),
            "pixels": {
                "top": np.zeros((224, 224, 3), dtype=np.uint8),
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FAACT hardware alarm-only mode")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.runtime.mode = "alarm_only"
    backbone = make_backbone_wrapper(
        config.backbone.policy_type,
        config.backbone.checkpoint,
        device=config.backbone.device,
        task_desc=config.backbone.task_desc,
    )
    risk_scorer = None
    intervention_policy = None
    if config.risk.enabled and config.risk.risk_model_ckpt:
        risk_scorer, _meta = load_supervised_risk_runtime(
            config.risk.risk_model_ckpt,
            config.backbone.device,
            feature_field=config.risk.risk_feature_field or None,
        )
        intervention_policy = ThresholdInterventionPolicy(
            config.risk.risk_threshold,
            cooldown_steps=config.risk.cooldown_steps,
            max_interventions_per_episode=config.risk.max_interventions_per_episode,
            boundary_only=config.risk.boundary_only_intervention,
        )
    runtime = HardwareRuntime(
        config=config,
        backbone=backbone,
        observation_adapter=DummyObservationAdapter(),
        robot=DryRunRobot(),
        risk_scorer=risk_scorer,
        intervention_policy=intervention_policy,
    )
    print(json.dumps(runtime.run_episode(), indent=2))


if __name__ == "__main__":
    main()
