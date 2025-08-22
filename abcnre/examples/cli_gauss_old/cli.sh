SIMULATOR_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results

ESTIMATOR_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results

MCMC_NPZ_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results/mcmc.npz

MCMC_FIG_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results/mcmc_fig.png

MCMC_NRE_OUT_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results/mcmc_nre_out.png

MCMC_CORRECT_NRE_OUT_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results/mcmc_correct_nre_out.png

MCMC_TRUE_OUT_PATH=/Users/antoineluciano/Documents/Recherche/ABC-SBI-1/abcnre/examples/cli_gauss/results/mcmc_true_out.png

# Activate the pyenv virtual environment
eval "$(pyenv init -)"
pyenv activate jax_env

# # Pass output_dir as required argument
abcnre create_simulator_and_estimator \
    --model_name gauss_gauss_10d_default \
    --classifier_config_template_name conditioned_deepset_default \
    --regressor_config_template_name deepset_default \
    --quantile_distance 1.0 \
    --output_dir "${ESTIMATOR_PATH}" \
    --delete_simulator_outputs

abcnre run_mcmc \
    "${ESTIMATOR_PATH}/estimator.yaml" ${ESTIMATOR_PATH} \
    --true \
    --burnin 5000 \
    --n-samples-tuning 20000 \
    --n-samples-chain 100000

abcnre plot_mcmc_posterior_comparison \
    ${MCMC_NPZ_PATH} \
    --save-path-1d ${MCMC_FIG_PATH}

abcnre plot_mcmc_output \
    ${MCMC_NPZ_PATH} \
    --nre \
    --save-path ${MCMC_NRE_OUT_PATH}

abcnre plot_mcmc_output \
    ${MCMC_NPZ_PATH} \
    --corrected-nre \
    --save-path ${MCMC_CORRECT_NRE_OUT_PATH}

abcnre plot_mcmc_output \
    ${MCMC_NPZ_PATH} \
    --true \
    --save-path ${MCMC_TRUE_OUT_PATH}
