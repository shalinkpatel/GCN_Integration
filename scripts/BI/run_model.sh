#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH --nodes 1
#SBATCH -c 2
#SBATCH --time 20:00:00
#SBATCH --mem-per-cpu 20G
#SBATCH --job-name gcn_inference
#SBATCH --output job-log-%J.txt

source ~/ml/bin/activate
julia model_explore_cl.jl >> output1.log 2>&1
