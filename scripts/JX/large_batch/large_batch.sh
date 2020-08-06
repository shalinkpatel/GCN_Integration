#!/bin/bash
#SBATCH --nodes 1
#SBATCH -c 2
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time 30:00:00
#SBATCH --mem-per-cpu 30G
#SBATCH --job-name large_batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shalin_patel@brown.edu

source ~/ml/bin/activate
julia large_batch.jl
