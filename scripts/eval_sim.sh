#!/bin/bash
set -e

eval "$(/root/miniforge3/bin/conda shell.bash hook)"
conda activate lerobot
export MUJOCO_GL=egl

TASK=${1:-"transfer_cube"}
CHECKPOINT=${2:-"last"}
N_EPISODES=${3:-10}

case $TASK in
  transfer_cube)
    ENV_TYPE="aloha"
    ENV_TASK="AlohaTransferCube-v0"
    ;;
  insertion)
    ENV_TYPE="aloha"
    ENV_TASK="AlohaInsertion-v0"
    ;;
  *)
    echo "Unknown task: $TASK"
    echo "Available tasks: transfer_cube, insertion"
    exit 1
    ;;
esac

TRAIN_DIR="/root/Research/outputs/train/act_${TASK}"
POLICY_PATH="${TRAIN_DIR}/checkpoints/${CHECKPOINT}/pretrained_model"
OUTPUT_DIR="/root/Research/outputs/eval/act_${TASK}_${CHECKPOINT}"

if [ ! -d "$POLICY_PATH" ]; then
  echo "Error: Checkpoint not found at $POLICY_PATH"
  echo "Available checkpoints:"
  ls ${TRAIN_DIR}/checkpoints/ 2>/dev/null || echo "  No checkpoints found. Train first!"
  exit 1
fi

echo "================================================"
echo "  LeRobot ACT Evaluation"
echo "================================================"
echo "  Task:       $ENV_TASK"
echo "  Checkpoint: $CHECKPOINT"
echo "  Policy:     $POLICY_PATH"
echo "  Episodes:   $N_EPISODES"
echo "  Output:     $OUTPUT_DIR"
echo "================================================"

lerobot-eval \
  --policy.path=${POLICY_PATH} \
  --env.type=${ENV_TYPE} \
  --env.task=${ENV_TASK} \
  --eval.n_episodes=${N_EPISODES} \
  --eval.batch_size=5 \
  --output_dir=${OUTPUT_DIR} \
  --policy.device=cuda

echo ""
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}/"
echo "Videos saved to:  ${OUTPUT_DIR}/videos/"
