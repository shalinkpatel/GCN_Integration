#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH --nodes 1
#SBATCH -c 1
#SBATCH --time 36:00:00
#SBATCH --mem-per-cpu 20G
#SBATCH --job-name gcn_inference
#SBATCH --output run.log

export CUDA_LAUNCH_BLOCKING=1

experiment=$1

source ~/ml/bin/activate
python3 $experiment
