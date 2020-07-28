#!/bin/bash
#SBATCH --nodes 1
#SBATCH -c 4
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time 01:00:00
#SBATCH --mem-per-cpu 10G
#SBATCH --job-name mlp

source ~/ml/bin/activate
python -u mlp.py --cell_line E114 --name A549
