#!/bin/bash
#SBATCH --nodes 1
#SBATCH -c 4
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time 02:00:00
#SBATCH --mem-per-cpu 10G
#SBATCH --job-name attentive

source ~/ml/bin/activate
python -u attentive.py --cell_line E066 --name Adult_Liver
python -u DeepChrome/AttentiveChrome-PyTorch/v2PyTorch/train.py --cell_type E066
