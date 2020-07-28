#!/bin/bash
#SBATCH --nodes 1
#SBATCH -c 4
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time 48:00:00
#SBATCH --mem-per-cpu 20G
#SBATCH --job-name HMExplain
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shalin_patel@brown.edu

source ~/ml/bin/activate
python -u NExplain.py --cell_line E096 --name HEPG2 --shuffle False

