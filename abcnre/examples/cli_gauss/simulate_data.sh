#!/bin/bash

RESULTS_PATH=./results_thesis
SIMULATOR_PATH=$RESULTS_PATH/simulators
CONFIG_PATH=./configs

mkdir -p  $SIMULATOR_PATH 


eval "$(pyenv init -)"
pyenv activate jax_env

mu_1=20.0
mu_2=15.0
mu_3=10.0
mu_4=5.0
mu_5=0.0

abcnre create_simulator \
        gauss_10d_default \
        $SIMULATOR_PATH/simulator \
        --model_path $CONFIG_PATH/model_config.yaml \
        --marginal_of_interest 0 \
        --quantile_distance 1.0 \
        --true_theta $mu_1 $mu_2 $mu_3 $mu_4 $mu_5
