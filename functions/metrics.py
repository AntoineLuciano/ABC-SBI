

from functions.simulation import NRE_posterior_sample, NRE_corrected_posterior_sample
from jax import random
from sbibm.metrics import c2st
import torch 
import time 
import numpy as np
from scipy.stats import ranksums, gaussian_kde

def ranksumstest_stat(x,y):
    return ranksums(x,y).statistic

def ranksumstest_pvalue(x,y):
    return ranksums(x,y).pvalue

def c2stest(x,y):
    return c2st(torch.tensor(x)[:, None], torch.tensor(y)[:, None])

def evaluate_metrics(
    key, metrics_dico, TRUE_DATA, params, thetas_abc, PRIOR_DIST, N_GRID, N_SAMPLE, N_SAMPLES, true_posterior_sample, N_KDE
):
    print("Evaluation of the metrics...")
    METRICS = {"ABC": {metric: np.zeros(N_SAMPLES) for metric in metrics_dico.keys()}, "NRE": {metric: np.zeros(N_SAMPLES) for metric in metrics_dico.keys()}, "CORRECTED_NRE": {metric: np.zeros(N_SAMPLES) for metric in metrics_dico.keys()}}
    time_start = time.time()
    
    PRIOR_LOGPDF = lambda theta: PRIOR_DIST.logpdf(theta)
    key, key_kde = random.split(key)
    KDE_APPROX = gaussian_kde(thetas_abc[random.choice(key_kde, np.arange(len(thetas_abc)), (N_KDE,))])
    MIN_INIT, MAX_INIT = PRIOR_DIST.interval(0.999)
    
    key, key_true, key_nre, key_corrected_nre, key_abc = random.split(key, 5)
    samples_true = true_posterior_sample(key_true, TRUE_DATA, N_SAMPLE*N_SAMPLES)
    samples_nre = NRE_posterior_sample(key_nre, params, TRUE_DATA, PRIOR_LOGPDF, N_GRID, MIN_INIT, MAX_INIT, N_SAMPLE*N_SAMPLES)
    samples_corrected_nre = NRE_corrected_posterior_sample(key_corrected_nre, params, TRUE_DATA, KDE_APPROX, N_GRID, MIN_INIT, MAX_INIT, N_SAMPLE*N_SAMPLES)
    
    for j in range(N_SAMPLES):
        sample_true = samples_true[j*N_SAMPLE:(j+1)*N_SAMPLE]
        sample_nre = samples_nre[j*N_SAMPLE:(j+1)*N_SAMPLE]
        sample_corrected_nre = samples_corrected_nre[j*N_SAMPLE:(j+1)*N_SAMPLE]
        sample_abc = thetas_abc[random.choice(key_abc, np.arange(len(thetas_abc)), (N_SAMPLE,))]
        samples_test = {"ABC": sample_abc, "NRE": sample_nre, "CORRECTED_NRE": sample_corrected_nre}
        for method in ["ABC", "NRE", "CORRECTED_NRE"]:
            if np.isnan(samples_test[method]).any():
                print(f"NaN values in {method} samples")
                for metric in metrics_dico.keys():
                    METRICS[method][metric][j] = np.nan
            else:
                for metric in metrics_dico.keys():
                    METRICS[method][metric][j] = metrics_dico[metric](sample_true, samples_test[method])
    time_eval = time.time() - time_start
    
    return METRICS, time_eval