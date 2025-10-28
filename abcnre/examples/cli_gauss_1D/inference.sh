#!/bin/bash


RESULTS_PATH=./results_deepsets_unnorm
ESTIMATOR_PATH=$RESULTS_PATH/estimators
SIMULATOR_PATH=$RESULTS_PATH/simulators
CONFIG_PATH=./configs
XOBS_PATH=./data/gauss1D_xobs.npy

mkdir -p $RESULTS_PATH $ESTIMATOR_PATH $SIMULATOR_PATH 


eval "$(pyenv init -)"
pyenv activate jax_env

marginals=(0)
quantiles_epsilon=(1.0 0.5 0.1 0.05 0.01)

for marginal in "${marginals[@]}"; do
    echo -e "\n\n\n=== Processing simulator for marginal: $marginal ===\n\n\n"

    abcnre create_simulator \
        gauss_10d_default \
        $SIMULATOR_PATH/simulator_marginal_$marginal \
        --model_path $CONFIG_PATH/model_config.yaml \
        --marginal_of_interest $marginal \
        --quantile_distance 1.0 \
        --observed_data_path $XOBS_PATH \

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
            --classifier_config_path $CONFIG_PATH/classif_config_deepsets.yaml

        if [ ! -f "$ESTIMATOR_PATH/estimator_marginal_${marginal}_quantile_${quantile}/estimator.yaml" ]; then
            echo "  Erreur: Estimateur non créé pour marginal $marginal, quantile $quantile"
            continue
        fi
        echo "  Estimator created successfully for marginal $marginal, quantile $quantile!"
       
        echo -e "\n\n\n ----- MARGINAL $marginal, QUANTILE $quantile DONE! -----\n\n\n"
    done
done




echo "=== Pipeline over ==="