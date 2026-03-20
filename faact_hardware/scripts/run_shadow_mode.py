#!/usr/bin/env python3
"""Run FAACT hardware scaffold in shadow mode."""

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
    parser = argparse.ArgumentParser(description="Run FAACT hardware shadow mode")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.runtime.mode = "shadow"
    backbone = make_backbone_wrapper(
        config.backbone.policy_type,
        config.backbone.checkpoint,
        device=config.backbone.device,
        task_desc=config.backbone.task_desc,
    )
    runtime = HardwareRuntime(
        config=config,
        backbone=backbone,
        observation_adapter=DummyObservationAdapter(),
        robot=DryRunRobot(),
    )
    print(json.dumps(runtime.run_episode(), indent=2))


if __name__ == "__main__":
    main()
