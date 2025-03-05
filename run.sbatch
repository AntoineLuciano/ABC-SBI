#!/bin/sh
#SBATCH -n 2
#SBATCH -c 2
#SBATCH --job-name=Gauss_1D_Known_50bis
#SBATCH --output=./cluster/outs/%J.OUT-%x.out
#SBATCH --error=./cluster/errors/%J.ERROR-%x.out

echo Host `hostname`
## Command(s) to run (example):
python3 examples/Gauss-Gauss/Gauss_Gauss_1D_known_sigma/cluster_gauss_1D_known.py
