#!/bin/bash

RESULTS_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results
ESTIMATOR_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results/estimators
SIMULATOR_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results/simulators
PLOT_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results/plots
MCMC_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results/mcmc
CONFIG_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/configs
XOBS_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/data/gauss_xobs.npy

mkdir -p $RESULTS_PATH $ESTIMATOR_PATH $SIMULATOR_PATH $PLOT_PATH $MCMC_PATH


eval "$(pyenv init -)"
pyenv activate jax_env

marginals=(-1 0 1 2 3 4)
quantiles_epsilon=(1.0 0.5 0.1 0.05)

for marginal in "${marginals[@]}"; do
    echo -e "\n\n\n=== Processing simulator for marginal: $marginal ===\n\n\n"

    abcnre create_simulator \
        gauss_10d_default \
        $SIMULATOR_PATH/simulator_marginal_$marginal \
        --model_path $CONFIG_PATH/model_config.yaml \
        --marginal_of_interest $marginal \
        --quantile_distance 1.0 \
        --regressor_config_path $CONFIG_PATH/regressor_config.yaml \
        --observed_data_path $XOBS_PATH \
        --learn_summary_stats \

    if [ ! -f "$SIMULATOR_PATH/simulator_marginal_$marginal/simulator.yaml" ]; then
        echo "Error: Simulator not created for marginal $marginal"
        continue
    fi
done

for quantile in "${quantiles_epsilon[@]}"; do
    for marginal in "${marginals[@]}"; do
        echo -e "\n\n\n--- Processing estimator for marginal $marginal, quantile $quantile ---\n\n\n"

        abcnre create_estimator \
            $SIMULATOR_PATH/simulator_marginal_$marginal/simulator.yaml \
            $ESTIMATOR_PATH/estimator_marginal_${marginal}_quantile_${quantile} \
            --quantile_distance $quantile \
            --classifier_config_path $CONFIG_PATH/classif_config.yaml

        if [ ! -f "$ESTIMATOR_PATH/estimator_marginal_${marginal}_quantile_${quantile}/estimator.yaml" ]; then
            echo "  Erreur: Estimateur non créé pour marginal $marginal, quantile $quantile"
            continue
        fi
        echo "  Estimator created successfully for marginal $marginal, quantile $quantile!"
        echo -e "\n\n\n--- Processing MCMC for marginal $marginal, quantile $quantile ---\n\n\n"

        abcnre run_mcmc \
            $ESTIMATOR_PATH/estimator_marginal_${marginal}_quantile_${quantile}/estimator.yaml \
            $MCMC_PATH/mcmc_marginal_${marginal}_quantile_${quantile}.npz \
            --n-samples-chain 100000 \
            --n-samples-tuning 100000 \
            --burnin 10000 \
            --no-true

        echo -e "\n\n\n ----- MARGINAL $marginal, QUANTILE $quantile DONE! -----\n\n\n"
    done
done


echo -e "\n\n\n=== Computing the true posterior with MCMC ===\n\n\n"


abcnre run_mcmc \
            $ESTIMATOR_PATH/estimator_marginal_-1_quantile_1.0/estimator.yaml \
            $MCMC_PATH/mcmc_true.npz \
            --n-samples-chain 100000 \
            --n-samples-tuning 100000 \
            --burnin 10000 \
            --true \
            --no-nre \
            --no-corrected-nre \


echo "=== Pipeline over ==="