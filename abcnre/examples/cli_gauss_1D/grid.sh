#!/bin/bash


RESULTS_PATH=./results_deepsets_unnorm
ESTIMATOR_PATH=$RESULTS_PATH/estimators
SIMULATOR_PATH=$RESULTS_PATH/simulators
PLOT_PATH=$RESULTS_PATH/plots
GRID_PATH=$RESULTS_PATH/grids


mkdir -p $GRID_PATH

N_GRID=100000

eval "$(pyenv init -)"
pyenv activate jax_env
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


# Remove duplicates
marginals=($(printf "%s\n" "${marginals[@]}" | sort -u))
quantiles_epsilon=($(printf "%s\n" "${quantiles_epsilon[@]}" | sort -u))



echo -e "Quantiles: ${quantiles_epsilon[*]}"
echo -e "Marginals: ${marginals[*]}"


for quantile in "${quantiles_epsilon[@]}"; do
    for marginal in "${marginals[@]}"; do
    
        echo -e "\n\n\n--- Processing Grid for marginal $marginal, quantile $quantile ---\n\n\n"

        abcnre run_grid \
            $ESTIMATOR_PATH/estimator_marginal_${marginal}_quantile_${quantile}/estimator.yaml \
            $GRID_PATH/grid_marginal_${marginal}_quantile_${quantile}.npz \
            --n-grid $N_GRID \
            --no-true \

        echo -e "\n\n\n ----- MARGINAL $marginal, QUANTILE $quantile DONE! -----\n\n\n"
    done
done


echo -e "\n\n\n=== Computing the true posterior grid ===\n\n\n"


abcnre run_grid \
            $ESTIMATOR_PATH/estimator_marginal_0_quantile_1.0/estimator.yaml \
            $GRID_PATH/grid_true.npz \
            --n-grid $N_GRID \
            --true \
            --no-nre \
            --no-corrected-nre \


echo "=== Pipeline over ==="