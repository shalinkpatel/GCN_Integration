#!/bin/bash
#SBATCH --nodes 1
#SBATCH -c 4
#SBATCH -p gpu --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time 30:00:00
#SBATCH --mem-per-cpu 20G
#SBATCH --job-name gcn

source ~/ml/bin/activate
julia port_batched.jl