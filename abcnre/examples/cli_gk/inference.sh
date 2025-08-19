#!/bin/bash

RESULTS_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gk/results
ESTIMATOR_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gk/results/estimators
SIMULATOR_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gk/results/simulators
PLOT_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gk/results/plots
MCMC_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gk/results/mcmc
CONFIG_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gk/configs
XOBS_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gk/data/gk_xobs.npy

mkdir -p $RESULTS_PATH $ESTIMATOR_PATH $SIMULATOR_PATH $PLOT_PATH $MCMC_PATH


eval "$(pyenv init -)"
pyenv activate jax_env

marginals=(-1)
quantiles_epsilon=(1.0 0.5 0.1)

true_theta_A=3.0
true_theta_B=1.0
true_theta_g=2.0
true_theta_k=0.5

for marginal in "${marginals[@]}"; do
    echo "=== Processing marginal: $marginal ==="
    
    abcnre create_simulator \
        g_and_k_default \
        $SIMULATOR_PATH/simulator_marginal_$marginal \
        --model_path $CONFIG_PATH/model_config.yaml \
        --marginal_of_interest $marginal \
        --quantile_distance 1.0 \
        --true_theta $true_theta_A $true_theta_B $true_theta_g $true_theta_k \
        --regressor_config_path $CONFIG_PATH/regressor_config.yaml \
        --observed_data_path $XOBS_PATH \
        --learn_summary_stats \

    if [ ! -f "$SIMULATOR_PATH/simulator_marginal_$marginal/simulator.yaml" ]; then
        echo "Erreur: Simulateur non créé pour marginal $marginal"
        continue
    fi

    for quantile in "${quantiles_epsilon[@]}"; do
        echo "  --- Processing quantile: $quantile ---"
        
        abcnre create_estimator \
            $SIMULATOR_PATH/simulator_marginal_$marginal/simulator.yaml \
            $ESTIMATOR_PATH/estimator_marginal_${marginal}_quantile_${quantile} \
            --quantile_distance $quantile \
            --classifier_config_path $CONFIG_PATH/classif_config.yaml

        if [ ! -f "$ESTIMATOR_PATH/estimator_marginal_${marginal}_quantile_${quantile}/estimator.yaml" ]; then
            echo "  Erreur: Estimateur non créé pour marginal $marginal, quantile $quantile"
            continue
        fi

        abcnre run_mcmc \
            $ESTIMATOR_PATH/estimator_marginal_${marginal}_quantile_${quantile}/estimator.yaml \
            $MCMC_PATH/mcmc_marginal_${marginal}_quantile_${quantile}.npz \
            --n-samples-chain 100000 \
            --n-samples-tuning 50000 \
            --burnin 5000 \
            --no-true

        echo "Completed marginal $marginal, quantile $quantile"
    done

done

abcnre run_mcmc \
            $ESTIMATOR_PATH/estimator_marginal_-1_quantile_1.0/estimator.yaml \
            $MCMC_PATH/mcmc_true.npz \
            --n-samples-chain 100000 \
            --n-samples-tuning 50000 \
            --burnin 5000 \
            --true \
            --no-nre \
            --no-corrected-nre \

echo "=== Pipeline over ==="
