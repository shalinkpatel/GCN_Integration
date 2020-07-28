#!/bin/bash
#SBATCH --nodes 1
#SBATCH -c 10
#SBATCH -p batch
#SBATCH --time 48:00:00
#SBATCH --mem-per-cpu 10G
#SBATCH --job-name network
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shalin_patel@brown.edu

source ~/ml/bin/activate
python network_analysis.py --cell_line E114 --name A549
