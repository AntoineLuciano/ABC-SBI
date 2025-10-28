#!/bin/bash

RESULTS_PATH=./results_thesis
ESTIMATOR_PATH=$RESULTS_PATH/estimators
MCMC_PATH=$RESULTS_PATH/mcmc


mkdir -p $MCMC_PATH

N_CHAIN=100000
N_TUNE=100000
BURNIN=10000


marginals=()
quantiles_epsilon=()

for dir in "$ESTIMATOR_PATH"/estimator_marginal_*_quantile_*; do
    # Extract marginal and quantile from directory name
    basename=$(basename "$dir")
    marginal=$(echo "$basename" | sed -n 's/estimator_marginal_\([0-9]*\)_quantile_.*$/\1/p')
    quantile=$(echo "$basename" | sed -n 's/estimator_marginal_[0-9]*_quantile_\([0-9.]*\).*$/\1/p')
    marginals+=("$marginal")
    quantiles_epsilon+=("$quantile")
done

eval "$(pyenv init -)"
pyenv activate jax_env


# Remove duplicates
marginals=($(printf "%s\n" "${marginals[@]}" | sort -u))
marginals+=("-1")
quantiles_epsilon=($(printf "%s\n" "${quantiles_epsilon[@]}" | sort -u))


echo -e "Quantiles: ${quantiles_epsilon[*]}"
echo -e "Marginals: ${marginals[*]}"

marginals=(0)
quantiles_epsilon=(1.0 0.1 0.01 0.001)

for quantile in "${quantiles_epsilon[@]}"; do
    for marginal in "${marginals[@]}"; do
    
        echo -e "\n\n\n--- Processing MCMC for marginal $marginal, quantile $quantile ---\n\n\n"

        abcnre run_mcmc \
            $ESTIMATOR_PATH/estimator_marginal_${marginal}_quantile_${quantile}/estimator.yaml \
            $MCMC_PATH/mcmc_marginal_${marginal}_quantile_${quantile}.npz \
            --n-samples-chain $N_CHAIN \
            --n-samples-tuning $N_TUNE \
            --burnin $BURNIN \
            --no-true \

        echo -e "\n\n\n ----- MARGINAL $marginal, QUANTILE $quantile DONE! -----\n\n\n"
    done
done


echo -e "\n\n\n=== Computing the true posterior with MCMC ===\n\n\n"


abcnre run_mcmc \
            $ESTIMATOR_PATH/estimator_marginal_0_quantile_1.0/estimator.yaml \
            $MCMC_PATH/mcmc_true.npz \
            --n-samples-chain $N_CHAIN \
            --n-samples-tuning $N_TUNE \
            --burnin $BURNIN \
            --true \
            --no-nre \
            --no-corrected-nre \


echo "=== Pipeline over ==="