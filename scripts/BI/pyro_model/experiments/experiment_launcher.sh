#!/bin/bash
#SBATCH -p bigmem
#SBATCH --nodes 1
#SBATCH -c 1
#SBATCH --time 48:00:00
#SBATCH --mem-per-cpu 128G

export CUDA_LAUNCH_BLOCKING=1

experiment=$1
shift

source ~/ml/bin/activate
python3 $experiment $@
