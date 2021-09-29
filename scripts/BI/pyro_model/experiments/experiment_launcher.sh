#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH --nodes 1
#SBATCH -c 1
#SBATCH --time 48:00:00
#SBATCH --mem-per-cpu 20G
#SBATCH --job-name gcn_inference
#SBATCH --output experiment-log-%J.txt

experiment=$1

source ~/ml/bin/activate
python3 $experiment