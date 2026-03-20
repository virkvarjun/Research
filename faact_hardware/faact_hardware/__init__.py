"""Hardware-first FAACT experiment package."""

from .config import HardwareExperimentConfig, load_config, to_episode_runner_config
from .runtime import DryRunRobot, HardwareRuntime
from .so101_bridge import (
    LeRobotSO101FollowerRobot,
    LeRobotSO101ObservationAdapter,
    lerobot_available,
    make_so101_follower_robot,
    robot_observation_to_faact_obs,
)

__all__ = [
    "DryRunRobot",
    "HardwareExperimentConfig",
    "HardwareRuntime",
    "LeRobotSO101FollowerRobot",
    "LeRobotSO101ObservationAdapter",
    "lerobot_available",
    "load_config",
    "make_so101_follower_robot",
    "robot_observation_to_faact_obs",
    "to_episode_runner_config",
]
