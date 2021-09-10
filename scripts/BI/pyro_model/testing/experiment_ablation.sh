#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH --nodes 1
#SBATCH -c 4
#SBATCH --time 48:00:00
#SBATCH --mem-per-cpu 10G
#SBATCH --job-name tunnel
#SBATCH --output ablation-log-%J.txt

source ~/ml/bin/activate
python3 experiment_ablation.py
