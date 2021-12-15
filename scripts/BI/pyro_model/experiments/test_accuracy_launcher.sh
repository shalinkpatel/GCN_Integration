#!/bin/bash
#SBATCH -p batch
#SBATCH --nodes 1
#SBATCH -c 1
#SBATCH --time 36:00:00
#SBATCH --mem-per-cpu 80G
#SBATCH --job-name test_label_accuracy
#SBATCH --output test-acc.log

source ~/ml/bin/activate
python3 test_accuracy_labels.py