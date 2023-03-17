#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --time 96:00:00
#SBATCH --mem-per-cpu 8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shalin_patel@brown.edu

export CUDA_LAUNCH_BLOCKING=1

experiment=$1
shift

source ~/ml/bin/activate
export PYTHONPATH="$PYTHONPATH:$pwd"
python3 -u $experiment $@
