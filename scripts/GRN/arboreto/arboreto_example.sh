#!/bin/bash
#SBATCH -c 16
#SBATCH --mem-per-cpu=8G
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shalin_patel@brown.edu

source ~/ml/bin/activate

python arboreto_example.py
