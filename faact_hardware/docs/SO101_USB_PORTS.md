# SO101 USB ports (this machine)

Verified working with `lerobot-teleoperate` (arms match: follower vs leader).

| Role | Device |
|------|--------|
| **Follower** (FAACT `robot.robot_port`) | `/dev/tty.usbmodem5AE60583121` |
| **Leader** (teleop only; not used by `run_hardware_faact.py`) | `/dev/tty.usbmodem5AE60798501` |

**Teleop command (reference):**

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5AE60583121 \
  --robot.id=so101_follower_1 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5AE60798501 \
  --teleop.id=so101_leader_1 \
  --display_data=true
```

If macOS reassigns device nodes after replugging, re-run `ls /dev/tty.usbmodem*` and update this file + `configs/so101_transfer_cube.yaml`.
