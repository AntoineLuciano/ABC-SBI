#!/bin/bash


RESULTS_PATH=./results_thesis
ESTIMATOR_PATH=$RESULTS_PATH/estimators
SIMULATOR_PATH=$RESULTS_PATH/simulators
CONFIG_PATH=./configs
XOBS_PATH=./data/thesis.npy

mkdir -p $RESULTS_PATH $ESTIMATOR_PATH $SIMULATOR_PATH 


marginals=(0)
quantiles_epsilon=(0.01 0.001)

N_SIM=5000000
TRAIN_VAL=0.8

for quantile in "${quantiles_epsilon[@]}"; do
    echo -e "\n\n\n=== Using quantile: $quantile ===\n\n\n"
    cp $CONFIG_PATH/classif_config_mlp_thesis.yaml $CONFIG_PATH/classif_config_mlp_thesis_quantile_${quantile}.yaml

        if (( $(echo "$quantile < 1.0" | bc -l) )); then
            # Calculate training set size as TRAIN_VAL * N_SIM * quantile
            training_set_size=$(awk "BEGIN {printf \"%d\", $TRAIN_VAL * $N_SIM *$quantile * 2}")
 
            yq -y -i ".training.training_set_size = $training_set_size" $CONFIG_PATH/classif_config_mlp_thesis_quantile_${quantile}.yaml
            yq -y -i ".training.n_samples_per_epoch = $training_set_size" $CONFIG_PATH/classif_config_mlp_thesis_quantile_${quantile}.yaml

            yq -y -i ".training.use_presimulated_data = true" $CONFIG_PATH/classif_config_mlp_thesis_quantile_${quantile}.yaml

            validation_set_size=$(awk "BEGIN {printf \"%d\", (1 - $TRAIN_VAL) * $N_SIM * $quantile * 2}")
            yq -y -i ".training.validation_set_size = $validation_set_size" $CONFIG_PATH/classif_config_mlp_thesis_quantile_${quantile}.yaml
        else
            yq -y -i ".training.use_presimulated_data = false" $CONFIG_PATH/classif_config_mlp_thesis_quantile_${quantile}.yaml
        fi
done


eval "$(pyenv init -)"
pyenv activate jax_env



for marginal in "${marginals[@]}"; do
    echo -e "\n\n\n=== Processing simulator for marginal: $marginal ===\n\n\n"

    abcnre create_simulator \
        gauss_10d_default \
        $SIMULATOR_PATH/simulator_marginal_$marginal \
        --model_path $CONFIG_PATH/model_config.yaml \
        --marginal_of_interest $marginal \
        --quantile_distance 1.0 \
        --observed_data_path $XOBS_PATH \
        # --learn_summary_stats \
        # --regressor_config_path $CONFIG_PATH/regressor_config_mlp.yaml

        # --regressor_config_template_name mlp_regressor

    if [ ! -f "$SIMULATOR_PATH/simulator_marginal_$marginal/simulator.yaml" ]; then
        echo "Error: Simulator not created for marginal $marginal"
        continue
    fi
done

for quantile in "${quantiles_epsilon[@]}"; do

    echo -e "\n\n\n=== Using quantile: $quantile ===\n\n\n"
    for marginal in "${marginals[@]}"; do
        echo -e "\n\n\n--- Processing estimator for marginal $marginal, quantile $quantile ---\n\n\n"

        abcnre create_estimator \
            $SIMULATOR_PATH/simulator_marginal_$marginal/simulator.yaml \
            $ESTIMATOR_PATH/estimator_marginal_${marginal}_quantile_${quantile} \
            --quantile_distance $quantile \
            --classifier_config_path $CONFIG_PATH/classif_config_mlp_thesis_quantile_${quantile}.yaml

            # --classifier_config_template_name mlp_default 

        if [ ! -f "$ESTIMATOR_PATH/estimator_marginal_${marginal}_quantile_${quantile}/estimator.yaml" ]; then
            echo "  Erreur: Estimateur non créé pour marginal $marginal, quantile $quantile"
            continue
        fi
        echo "  Estimator created successfully for marginal $marginal, quantile $quantile!"
       
        echo -e "\n\n\n ----- MARGINAL $marginal, QUANTILE $quantile DONE! -----\n\n\n"
    done
done
echo "=== Pipeline over ==="