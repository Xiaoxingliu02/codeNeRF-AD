#!/bin/bash
#SBATCH --partition=gpu
#SBATCH -G, --gpus=1
#SBATCH --cpus-per-gpu=8
module add CUDA/11.3
nvcc -V
#CUDA_VISIBLE_DEVICES=0
nvidia-smi

python -u face_tracker.py