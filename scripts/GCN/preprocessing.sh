#!/bin/bash
#SBATCH --nodes 1
#SBATCH -c 4
#SBATCH -p batch
#SBATCH --time 48:00:00
#SBATCH --mem-per-cpu 15G
#SBATCH --job-name preprocssing
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shalin_patel@brown.edu

source ~/ml/bin/activate
python preprocessing.py --cell_line E118 --name HEPG2
