#!/bin/bash
#SBATCH -c 1
#SBATCH --mem-per-cpu=8G
#SBATCH -t 48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shalin_patel@brown.edu

source ~/ml/bin/activate

python HEPG2.py
