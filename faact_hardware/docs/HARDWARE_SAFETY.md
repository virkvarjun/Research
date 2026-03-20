# Hardware Safety Notes

This folder is for supervised hardware research, not unattended deployment.

## Rules

- Start in `shadow` mode.
- Keep `dry_run=true` until observation and robot bridges are validated.
- Require a human operator present for every episode.
- Stop immediately if observations are missing, stale, or malformed.
- Stop immediately if any action chunk exceeds configured action limits.
- Do not enable automatic intervention on hardware until alarm-only logs look sensible on real rollouts.

## Recommended rollout order

1. Shadow mode with logging only.
2. Alarm-only mode with human review of alarm timing.
3. Intervention mode with one-step budget and manual arming.

## Implemented bridges (see `faact_hardware/so101_bridge.py`)

- LeRobot SO101 follower `get_observation` → FAACT `agent_pos` + `pixels`
- Policy action vector → `RobotAction` dict + rate limiting via `LeRobotSO101FollowerRobot`

## Still recommended

- Verified E-stop and watchdog for your lab setup
- Policy output scale check vs `safety.max_abs_action_value` and dataset normalization
