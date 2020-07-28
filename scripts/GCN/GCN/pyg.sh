#!/bin/bash
#SBATCH --nodes 1
#SBATCH -c 4
#SBATCH -p gpu --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time 02:00:00
#SBATCH --mem-per-cpu 8G
#SBATCH --job-name gcn

source ~/ml/bin/activate
python -u -W ignore pyg.py --cell_line E003 --name H1 --shuffle False --randomize False
