#!/usr/bin/env bash
# Example driver for phased FAACT hardware runs (adjust paths and config).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}/faact_hardware:${ROOT}/faact:${ROOT}:${PYTHONPATH:-}"
CFG="${1:-${ROOT}/faact_hardware/configs/so101_transfer_cube.yaml}"

echo "=== Phase 1: shadow / dummy obs (no robot USB) ==="
python "${ROOT}/faact_hardware/scripts/run_hardware_faact.py" --config "${CFG}"

echo "=== Phase 2+: add --use-real-robot when SO101 + cameras are ready ==="
echo "python ${ROOT}/faact_hardware/scripts/run_hardware_faact.py --config ${CFG} --use-real-robot"
