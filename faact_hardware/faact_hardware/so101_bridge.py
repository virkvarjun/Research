"""LeRobot SO101 follower bridges for FAACT hardware (observations + actions)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import HardwareRobotConfig

if TYPE_CHECKING:
    from lerobot.robots.robot import Robot


def lerobot_available() -> bool:
    try:
        import lerobot.robots  # noqa: F401

        return True
    except ImportError:
        return False


def build_opencv_cameras(cameras_cfg: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build a ``cameras`` dict for ``SOFollowerRobotConfig`` from YAML-style nested dicts."""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    if not cameras_cfg:
        return {
            "top": OpenCVCameraConfig(
                index_or_path=0,
                fps=30,
                width=640,
                height=480,
            )
        }

    out: dict[str, Any] = {}
    for name, c in cameras_cfg.items():
        idx = c.get("index_or_path", 0)
        out[name] = OpenCVCameraConfig(
            index_or_path=idx,
            fps=int(c.get("fps", 30)),
            width=int(c.get("width", 640)),
            height=int(c.get("height", 480)),
        )
    return out


def make_so101_follower_robot(cfg: HardwareRobotConfig) -> Any:
    """Instantiate a connected ``SO101Follower`` from ``lerobot``."""
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.utils import make_robot_from_config

    cameras = build_opencv_cameras(dict(cfg.cameras))
    robot_cfg = SOFollowerRobotConfig(
        id=cfg.robot_id,
        port=cfg.robot_port,
        cameras=cameras,
        max_relative_target=cfg.max_relative_target,
        use_degrees=cfg.use_degrees,
    )
    robot = make_robot_from_config(robot_cfg)
    robot.connect(calibrate=cfg.connect_calibrate)
    return robot


def robot_observation_to_faact_obs(
    raw: dict[str, Any],
    joint_names: list[str],
    camera_keys: list[str],
) -> dict[str, Any]:
    """Map LeRobot ``RobotObservation`` to FAACT / ``PI0PolicyWrapper`` gym-style dict."""
    joint_positions = np.array([raw[f"{name}.pos"] for name in joint_names], dtype=np.float32)
    pixels = {key: np.asarray(raw[key]) for key in camera_keys if key in raw}
    return {"agent_pos": joint_positions, "pixels": pixels}


class LeRobotSO101ObservationAdapter:
    """Reads ``get_observation()`` from a LeRobot SO101 follower and returns FAACT-shaped dicts."""

    def __init__(self, robot: Any) -> None:
        self.robot = robot
        self._joint_names = list(robot.bus.motors.keys())
        self._camera_keys = list(robot.cameras.keys())

    def reset(self) -> None:
        return None

    def get_observation(self) -> dict[str, Any] | None:
        raw = self.robot.get_observation()
        return robot_observation_to_faact_obs(raw, self._joint_names, self._camera_keys)


class LeRobotSO101FollowerRobot:
    """Maps policy action vectors to ``RobotAction`` dicts and rate-limits ``send_action``."""

    def __init__(
        self,
        robot: Any,
        motor_names: list[str],
        *,
        dry_run: bool = True,
        control_hz: float | None = 10.0,
        max_abs_action_value: float = 1.5,
    ) -> None:
        self.robot = robot
        self._motor_names = motor_names
        self.dry_run = dry_run
        self.control_hz = control_hz
        self.max_abs_action_value = max_abs_action_value
        self._last_send_time: float | None = None
        self.executed_actions: list[np.ndarray] = []

    def arm(self) -> bool:
        return True

    def stop(self) -> None:
        return None

    def _sleep_rate_limit(self) -> None:
        if self.control_hz is None or self.control_hz <= 0:
            return
        min_dt = 1.0 / float(self.control_hz)
        now = time.perf_counter()
        if self._last_send_time is not None:
            elapsed = now - self._last_send_time
            if elapsed < min_dt:
                time.sleep(min_dt - elapsed)
        self._last_send_time = time.perf_counter()

    def execute_action(self, action: np.ndarray) -> None:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != len(self._motor_names):
            raise ValueError(
                f"Action dim {arr.size} does not match motor count {len(self._motor_names)}"
            )
        if np.any(~np.isfinite(arr)):
            raise ValueError("Non-finite action value")
        clipped = np.clip(arr, -self.max_abs_action_value, self.max_abs_action_value)
        self.executed_actions.append(clipped.copy())
        if self.dry_run:
            return
        self._sleep_rate_limit()
        robot_action = {f"{name}.pos": float(clipped[i]) for i, name in enumerate(self._motor_names)}
        self.robot.send_action(robot_action)
