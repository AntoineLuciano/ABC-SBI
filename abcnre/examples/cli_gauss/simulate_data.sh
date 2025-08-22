#!/bin/bash

RESULTS_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results
ESTIMATOR_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results/estimators
SIMULATOR_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results/simulators
PLOT_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results/plots
MCMC_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results/mcmc
CONFIG_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/configs
XOBS_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/data/gk_xobs.npy

mkdir -p $RESULTS_PATH $ESTIMATOR_PATH $SIMULATOR_PATH $PLOT_PATH $MCMC_PATH


eval "$(pyenv init -)"
pyenv activate jax_env

abcnre create_simulator \
        gauss_10d_default \
        $SIMULATOR_PATH/simulator \
        --model_path $CONFIG_PATH/model_config.yaml \
        --marginal_of_interest 0 \
        --quantile_distance 1.0 \
