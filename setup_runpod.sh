#!/bin/bash
set -e

echo "=== [1/5] Cloning Research repo ==="
cd /root
if [ -d "Research" ]; then
    echo "Research dir already exists, removing for fresh clone..."
    rm -rf Research
fi
git clone --recurse-submodules https://github.com/virkvarjun/Research.git

echo "=== [2/5] Installing Miniforge ==="
if [ -f "/root/miniforge3/bin/conda" ]; then
    echo "Miniforge already installed, skipping..."
else
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p /root/miniforge3
    rm /tmp/miniforge.sh
fi

eval "$(/root/miniforge3/bin/conda shell.bash hook)"

echo "=== [3/5] Creating conda environment ==="
if conda env list | grep -q "lerobot"; then
    echo "lerobot env already exists, skipping..."
else
    conda create -y -n lerobot python=3.12
fi

conda activate lerobot

echo "=== [3.5/5] Installing system deps for headless MuJoCo ==="
apt-get update -qq && apt-get install -y -qq libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev xvfb > /dev/null 2>&1
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
export MUJOCO_GL=egl

echo "=== [4/5] Installing lerobot with hilserl ==="
cd /root/Research/lerobot
pip install -e ".[hilserl]"

echo "=== [5/5] Verifying installation ==="
python -c "
import sys
import torch
import numpy as np
import mujoco
import gymnasium
import gym_hil
import lerobot
import transformers

print('All imports successful!')
print('  Python:       ' + sys.version.split()[0])
print('  PyTorch:      ' + torch.__version__)
print('  CUDA:         ' + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print('  GPU:          ' + torch.cuda.get_device_name(0))
    print('  VRAM:         ' + str(round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)) + ' GB')
print('  MuJoCo:       ' + mujoco.__version__)
print('  Gymnasium:    ' + gymnasium.__version__)
print('  Transformers: ' + transformers.__version__)
print('  LeRobot:      ' + lerobot.__version__)
"

echo "=== SETUP COMPLETE ==="
