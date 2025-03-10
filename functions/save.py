import pandas as pd
import pickle
import lzma
import numpy as np


def create_csv_for_a_dataset(
    i_datasets,
    ALPHAS, 
    TEST_ACCURACY,
    TRAIN_ACCURACY,
    TEST_LOSSES,
    TRAIN_LOSSES,
    TIME_SIMULATIONS,
    TIME_TRAINING,
    TIME_EVAL,
    METRICS_ABC,
    METRICS_NRE,
    METRICS_CORRECTED_NRE,
    TRUE_DATA,
    TRUE_THETA,
    THETAS_ABC, 
    file_name
):
    df = pd.DataFrame()
    df["ALPHA"] = ALPHAS
    df["TRUE_DATA"] = [np.array(TRUE_DATA).reshape(-1)] * len(ALPHAS)
    df["TRUE_THETA"] = [np.array(TRUE_THETA).reshape(-1)] * len(ALPHAS)
    df["INDEX_MARGINAL"] = [0] * len(ALPHAS)
    df["TEST_ACCURACY"] = [TEST_ACCURACY[a] for a in ALPHAS]
    df["TRAIN_ACCURACY"] = [TRAIN_ACCURACY[a] for a in ALPHAS]
    df["TEST_LOSSES"] = [TEST_LOSSES[a] for a in ALPHAS]
    df["TRAIN_LOSSES"] = [TRAIN_LOSSES[a] for a in ALPHAS]
    df["TIME_SIMULATIONS"] = [TIME_SIMULATIONS[a] for a in ALPHAS]
    df["TIME_TRAINING"] = [TIME_TRAINING[a] for a in ALPHAS]
    df["TIME_EVAL"] = [TIME_EVAL[a] for a in ALPHAS]
    df["RANKSUMS_STAT_ABC"] = [METRICS_ABC[a]["RS_stat"] for a in ALPHAS]
    df["RANKSUMS_PVALUE_ABC"] = [METRICS_ABC[a]["RS_pvalue"] for a in ALPHAS]
    df["C2ST_ABC"] = [METRICS_ABC[a]["C2ST"] for a in ALPHAS]
    df["RANKSUMS_STAT_NRE"] = [METRICS_NRE[a]["RS_stat"] for a in ALPHAS]
    df["RANKSUMS_PVALUE_NRE"] = [METRICS_NRE[a]["RS_pvalue"] for a in ALPHAS]
    df["C2ST_NRE"] = [METRICS_NRE[a]["C2ST"] for a in ALPHAS]
    df["RANKSUMS_STAT_CORRECTED_NRE"] = [METRICS_CORRECTED_NRE[a]["RS_stat"] for a in ALPHAS]
    df["RANKSUMS_PVALUE_CORRECTED_NRE"] = [METRICS_CORRECTED_NRE[a]["RS_pvalue"] for a in ALPHAS]
    df["C2ST_CORRECTED_NRE"] = [METRICS_CORRECTED_NRE[a]["C2ST"] for a in ALPHAS]
    df["THETAS_ABC"] = [THETAS_ABC[a] for a in ALPHAS]
    df.to_csv(file_name)
    print("CSV CREATED at {}".format(file_name))
    

def create_pickle_for_a_dataset(ALPHAS, PARAMS, METRICS_ABC, METRICS_NRE, METRICS_CORRECTED_NRE, TRUE_DATA, TRUE_THETA, TIME_SIMULATIONS, TIME_TRAINING, TIME_EVAL, MODEL_ARGS, PRIOR_ARGS, NN_ARGS, THETAS_ABC, file_name):
    dico = {
        "ALPHAS": ALPHAS,
        "PARAMS": PARAMS,
        "METRICS_ABC": METRICS_ABC,
        "METRICS_NRE": METRICS_NRE,
        "METRICS_CORRECTED_NRE": METRICS_CORRECTED_NRE,
        "TRUE_DATA": TRUE_DATA,
        "TRUE_THETA": TRUE_THETA,
        "TIME_SIMULATIONS": TIME_SIMULATIONS,
        "TIME_TRAINING": TIME_TRAINING,
        "TIME_EVAL": TIME_EVAL,
        "MODEL_ARGS": MODEL_ARGS,
        "PRIOR_ARGS": PRIOR_ARGS,
        "NN_ARGS": NN_ARGS, 
        "THETAS_ABC": THETAS_ABC
        
        }
    with lzma.open(file_name, "wb") as f:
        pickle.dump(dico, f)
    print("PICKLE CREATED at {}".format(file_name))
    
    
