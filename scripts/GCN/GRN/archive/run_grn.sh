#!/bin/bash
#SBATCH --nodes 1
#SBATCH -c 32
#SBATCH --time 48:00:00
#SBATCH --mem-per-cpu 6G
#SBATCH --job-name grn

source ~/ml/bin/activate
python run_grn.py