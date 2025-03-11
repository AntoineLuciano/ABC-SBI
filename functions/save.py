import pandas as pd
import pickle
import lzma
import numpy as np


def create_csv_for_a_dataset(
    ALPHAS, 
    EPSILONS, 
    TEST_ACCURACY,
    TRAIN_ACCURACY,
    TIME_SIMULATIONS,
    TIME_TRAINING,
    TIME_EVAL,
    METRICS, 
    TRUE_DATA,
    TRUE_THETA,
    PRIOR_ARGS, 
    MODEL_ARGS,
    NN_ARGS,
    INDEX_MARGINAL,
    file_name
):
    rows = []
    for alpha in ALPHAS:
        epsilon = EPSILONS[alpha]
        for method in METRICS[alpha].keys():
            for metric, values in METRICS[alpha][method].items():
                for value_metric in values:
                    rows.append({
                        "alpha": alpha, 
                        "epsilon": epsilon, 
                        "method": method, 
                        "metric": metric, 
                        "metric_value": value_metric,
                        **{k: v for k,v in MODEL_ARGS.items()},
                        **{k: v for k,v in PRIOR_ARGS.items()},
                        **{k: v for k,v in NN_ARGS.items()},
                        **{f"true_theta_{i+1}": TRUE_THETA[i] for i in range(len(TRUE_THETA))},
                        "index_marginal": INDEX_MARGINAL,
                        **{f"xobs_{i+1}": TRUE_DATA[i] for i in range(len(TRUE_DATA))}
                    })
        for metric,dico in zip(["TIME_SIMULATION", "TIME_TRAINING", "TIME_EVAL", "NN_TRAIN_ACCURACY", "NN_TEST_ACCURACY"], [TIME_SIMULATIONS, TIME_TRAINING, TIME_EVAL, TRAIN_ACCURACY, TEST_ACCURACY]):
            rows.append({
                "alpha": alpha, 
                "epsilon": epsilon, 
                "method": "ALL", 
                "metric": metric, 
                "metric_value": dico[alpha],
                **{k: v for k,v in MODEL_ARGS.items()},
                **{k: v for k,v in PRIOR_ARGS.items()},
                **{k: v for k,v in NN_ARGS.items()},
                **{f"true_theta_{i+1}": TRUE_THETA[i] for i in range(len(TRUE_THETA))},
                "index_marginal": INDEX_MARGINAL,
                **{f"xobs_{i+1}": TRUE_DATA[i] for i in range(len(TRUE_DATA))}
            })

            
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(file_name, index=False)
    print("CSV CREATED at {}".format(file_name))
    

# def create_pickle_for_a_dataset(ALPHAS, PARAMS, METRICS_ABC, METRICS_NRE, METRICS_CORRECTED_NRE, TRUE_DATA, TRUE_THETA, TIME_SIMULATIONS, TIME_TRAINING, TIME_EVAL, MODEL_ARGS, PRIOR_ARGS, NN_ARGS, THETAS_ABC, file_name):
#     dico = {
#         "ALPHAS": ALPHAS,
#         "PARAMS": PARAMS,
#         "METRICS_ABC": METRICS_ABC,
#         "METRICS_NRE": METRICS_NRE,
#         "METRICS_CORRECTED_NRE": METRICS_CORRECTED_NRE,
#         "TRUE_DATA": TRUE_DATA,
#         "TRUE_THETA": TRUE_THETA,
#         "TIME_SIMULATIONS": TIME_SIMULATIONS,
#         "TIME_TRAINING": TIME_TRAINING,
#         "TIME_EVAL": TIME_EVAL,
#         "MODEL_ARGS": MODEL_ARGS,
#         "PRIOR_ARGS": PRIOR_ARGS,
#         "NN_ARGS": NN_ARGS, 
#         "THETAS_ABC": THETAS_ABC
        
#         }
#     with lzma.open(file_name, "wb") as f:
#         pickle.dump(dico, f)
#     print("PICKLE CREATED at {}".format(file_name))
 