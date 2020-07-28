#!/bin/bash
#SBATCH --nodes 1
#SBATCH -c 16
#SBATCH -p batch
#SBATCH --time 48:00:00
#SBATCH --mem-per-cpu 6G
#SBATCH --job-name network
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shalin_patel@brown.edu

source ~/ml/bin/activate
hostname -i
python network.py --cell_line E071 --name Hippocampus
