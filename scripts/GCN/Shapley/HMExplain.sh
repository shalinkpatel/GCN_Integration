#!/bin/bash
#SBATCH --nodes 1
#SBATCH -c 4
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time 48:00:00
#SBATCH --mem-per-cpu 30G
#SBATCH --job-name HMExplain
#SBATCH --constraint=v100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shalin_patel@brown.edu

source ~/ml/bin/activate
python -u HMExplain.py --cell_line E118 --name HEPG2 --shuffle False
