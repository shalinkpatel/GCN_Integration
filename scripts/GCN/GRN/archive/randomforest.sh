#!/bin/bash
#SBATCH -c 32
#SBATCH --mem-per-cpu=6G

module load R/3.4.3_mkl

R CMD BATCH randomforest.R
