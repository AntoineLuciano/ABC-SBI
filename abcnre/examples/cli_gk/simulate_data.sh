#!/bin/bash

RESULTS_PATH=./results
SIMULATOR_PATH=$RESULTS_PATH/simulators
CONFIG_PATH=./configs

mkdir -p  $SIMULATOR_PATH 


eval "$(pyenv init -)"
pyenv activate jax_env


true_theta_A=3.0
true_theta_B=1.0
true_theta_g=2.0
true_theta_k=0.5


abcnre create_simulator \
        g_and_k_default \
        $SIMULATOR_PATH/simulator \
        --model_path $CONFIG_PATH/model_config.yaml \
        --marginal_of_interest 0 \
        --quantile_distance 1.0 \
        --true_theta $true_theta_A $true_theta_B $true_theta_g $true_theta_k \
