#!/bin/bash
#SBATCH -p bigmem
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --time 48:00:00
#SBATCH --mem-per-cpu 32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shalin_patel@brown.edu

export CUDA_LAUNCH_BLOCKING=1

experiment=$1
shift

source ~/ml/bin/activate
export PYTHONPATH="$PYTHONPATH:$pwd"
python3 $experiment $@
