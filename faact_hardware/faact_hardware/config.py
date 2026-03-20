"""Configuration helpers for FAACT hardware experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class HardwareRobotConfig:
    robot_type: str = "so101_follower"
    robot_port: str = "/dev/tty.usbmodem5AE60583121"
    robot_id: str = "so101_follower_1"
    teleop_type: str = "so101_leader"
    teleop_port: str = "/dev/tty.usbmodem5AE60798501"
    teleop_id: str = "so101_leader_1"
    dry_run: bool = True


@dataclass
class BackboneConfig:
    policy_type: str = "pi0"
    checkpoint: str = "lerobot/pi0_base"
    device: str = "cuda"
    task_desc: str = "transfer the cube to the target"
    replan_interval: int | None = None


@dataclass
class RiskConfig:
    enabled: bool = False
    risk_model_ckpt: str = ""
    risk_feature_field: str = ""
    risk_threshold: float = 0.55
    boundary_only_intervention: bool = True
    max_interventions_per_episode: int = 2
    cooldown_steps: int = 0


@dataclass
class RuntimeConfig:
    mode: str = "shadow"
    chunk_size: int = 50
    num_candidate_chunks: int = 8
    candidate_source: str = "hybrid"
    switch_margin: float = 0.02
    obs_noise_std: float = 0.05
    action_noise_std: float = 0.08
    action_noise_prefix_steps: int = 12
    min_candidate_l2_to_baseline: float = 1.0
    min_candidate_prefix_l2_to_baseline: float = 0.0
    max_candidate_tail_l2_to_baseline: float | None = None
    local_search_prefix_steps: int = 8
    max_steps: int = 300
    log_dir: str = "faact_hardware/runs/latest"


@dataclass
class SafetyConfig:
    require_human_armed: bool = True
    halt_on_missing_observation: bool = True
    halt_on_invalid_chunk: bool = True
    halt_on_runtime_error: bool = True
    max_abs_action_value: float = 1.5


@dataclass
class HardwareExperimentConfig:
    robot: HardwareRobotConfig = field(default_factory=HardwareRobotConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def load_config(path: str | Path) -> HardwareExperimentConfig:
    """Load a hardware experiment config from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return HardwareExperimentConfig(
        robot=HardwareRobotConfig(**_section(data, "robot")),
        backbone=BackboneConfig(**_section(data, "backbone")),
        risk=RiskConfig(**_section(data, "risk")),
        runtime=RuntimeConfig(**_section(data, "runtime")),
        safety=SafetyConfig(**_section(data, "safety")),
    )
