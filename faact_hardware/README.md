# FAACT Hardware

This folder is a separate hardware-focused workspace for testing whether the FAACT hypothesis translates from simulation to physical execution.

## Scope

This package is for:

- shadow-mode runs on the real robot
- alarm-only hardware evaluation
- gated intervention experiments with strict safety defaults

This package is not yet a production robot controller. It is a safety-first research scaffold around the existing FAACT and failure-prediction code.

## Layout

```text
faact_hardware/
├── configs/
├── docs/
├── faact_hardware/
└── scripts/
```

## Hardware plan

Phase 1: `shadow`

- read observations from hardware
- score the current chunk
- log alarms
- do not override the robot

Phase 2: `alarm_only`

- same as shadow mode
- confirm alarms line up with visible failure drift

Phase 3: `intervene`

- only allow chunk replacement when the runtime passes safety checks
- keep human supervision in the loop

## Safety defaults

- `dry_run=true` by default
- intervention budget capped per episode
- optional boundary-only intervention
- hard stop if robot state, observations, or candidate chunk validation fail

## Quick start

Install from the repo root:

```bash
cd faact_hardware
pip install -e .
```

Run the unified hardware runner (dummy observations, safe dry-run robot):

```bash
# From Research/: include the `faact_hardware/` dir (package root), `faact/`, and `.`
PYTHONPATH="faact_hardware:faact:." python faact_hardware/scripts/run_hardware_faact.py \
  --config faact_hardware/configs/so101_transfer_cube.yaml
```

SO101 with real cameras + follower (supervised; start with `robot.dry_run: true`):

```bash
PYTHONPATH="faact_hardware:faact:." python faact_hardware/scripts/run_hardware_faact.py \
  --config faact_hardware/configs/so101_transfer_cube.yaml --use-real-robot
```

See [docs/HARDWARE_ROLLOUT.md](docs/HARDWARE_ROLLOUT.md) for phased criteria.

Legacy entry points:

```bash
python scripts/run_shadow_mode.py --config configs/so101_transfer_cube.yaml
python scripts/run_alarm_eval.py --config configs/so101_transfer_cube.yaml
```

## USB ports (SO101)

Saved mapping for this setup: [docs/SO101_USB_PORTS.md](docs/SO101_USB_PORTS.md) (also noted in `configs/so101_transfer_cube.yaml`).

## Integration target

The intended backbone path is:

- `faact.backbone.*` for chunk proposals
- `failure_prediction.runtime_components` for supervised risk scoring
- physical robot observation + action bridges implemented behind the interfaces in `faact_hardware.runtime`

## Current status

This folder contains the hardware experiment scaffold, not a finished hardware deployment. The next step is wiring a real observation adapter and follower robot command bridge.
