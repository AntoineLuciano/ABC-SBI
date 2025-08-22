#!/bin/bash

RESULTS_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results
ESTIMATOR_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results/estimators
SIMULATOR_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results/simulators
PLOT_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results/plots
MCMC_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results/mcmc
CONFIG_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/configs
XOBS_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/data/gauss_xobs.npy

mkdir -p $RESULTS_PATH $ESTIMATOR_PATH $SIMULATOR_PATH $PLOT_PATH $MCMC_PATH

N_CHAIN=100000
N_TUNE=100000
BURNIN=10000
eval "$(pyenv init -)"
pyenv activate jax_env

marginals=(4)
quantiles_epsilon=(1.0 0.5 0.1 0.05)


# for quantile in "${quantiles_epsilon[@]}"; do
#     for marginal in "${marginals[@]}"; do
    
#         echo -e "\n\n\n--- Processing MCMC for marginal $marginal, quantile $quantile ---\n\n\n"

#         abcnre run_mcmc \
#             $ESTIMATOR_PATH/estimator_marginal_${marginal}_quantile_${quantile}/estimator.yaml \
#             $MCMC_PATH/mcmc_marginal_${marginal}_quantile_${quantile}.npz \
#             --n-samples-chain $N_CHAIN \
#             --n-samples-tuning $N_TUNE \
#             --burnin $BURNIN \
#             --no-true \

#         echo -e "\n\n\n ----- MARGINAL $marginal, QUANTILE $quantile DONE! -----\n\n\n"
#     done
# done


echo -e "\n\n\n=== Computing the true posterior with MCMC ===\n\n\n"


abcnre run_mcmc \
            $ESTIMATOR_PATH/estimator_marginal_-1_quantile_1.0/estimator.yaml \
            $MCMC_PATH/mcmc_true.npz \
            --n-samples-chain $N_CHAIN \
            --n-samples-tuning $N_TUNE \
            --burnin $BURNIN \
            --true \
            --no-nre \
            --no-corrected-nre \


echo "=== Pipeline over ==="