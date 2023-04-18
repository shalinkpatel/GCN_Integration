#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --time 24:00:00
#SBATCH --mem-per-cpu 16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shalin_patel@brown.edu

export CUDA_LAUNCH_BLOCKING=1

experiment=$1
shift

source ~/ml/bin/activate
export PYTHONPATH="$PYTHONPATH:$pwd"
python3 -u -W ignore $experiment $@
