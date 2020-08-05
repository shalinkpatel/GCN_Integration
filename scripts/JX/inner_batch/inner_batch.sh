#!/bin/bash
#SBATCH --nodes 1
#SBATCH -c 4
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time 30:00:00
#SBATCH --mem-per-cpu 15G
#SBATCH --job-name inner_batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shalin_patel@brown.edu

source ~/ml/bin/activate
julia inner_batch.jl
