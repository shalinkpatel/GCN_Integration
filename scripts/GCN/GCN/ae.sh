#!/bin/bash
#SBATCH --nodes 1
#SBATCH -c 4
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time 5:00:00
#SBATCH --mem-per-cpu 20G
#SBATCH --job-name ae
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shalin_patel@brown.edu

source ~/ml/bin/activate
python ae.py --cell_line E114 --name A549
