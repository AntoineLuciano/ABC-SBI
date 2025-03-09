#!/bin/sh
#SBATCH -n 2
#SBATCH -c 2
#SBATCH --job-name=Linear_Reg_25
#SBATCH --output=./cluster/outs/%J.OUT-%x.out
#SBATCH --error=./cluster/errors/%J.ERROR-%x.out

echo Host `hostname`
## Command(s) to run (example):

# python3 examples/Gauss-Gauss/Gauss_Gauss_1D_known_sigma/cluster_pattern_gauss.py 50
# python3 examples/Gauss-Gauss/Gauss_Gauss_Muli_known_sigma/cluster_pattern_multigauss.py 10
python3 examples/Linear-Reg/cluster_pattern_linear_regression.py 25