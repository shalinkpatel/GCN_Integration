#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH --nodes 1
#SBATCH -c 1
#SBATCH --time 48:00:00
#SBATCH --mem-per-cpu 20G
#SBATCH --job-name gcn_inference
#SBATCH --output job-log-%J.txt

source ~/ml/bin/activate
python syn.py
