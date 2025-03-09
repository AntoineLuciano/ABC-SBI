

from functions.simulation import NRE_posterior_sample, NRE_corrected_posterior_sample
from jax import random
from sbibm.metrics import c2st
import torch 
import time 
import numpy as np
from scipy.stats import ranksums, gaussian_kde
def c2stest(x,y):
    return c2st(torch.tensor(x)[:, None], torch.tensor(y)[:, None])

def evaluate_metrics(
    key, TRUE_DATA, params, thetas_abc, PRIOR_DIST, N_GRID, N_SAMPLE, N_SAMPLES, true_posterior_sample, N_KDE
):
    print("Evaluation of the metrics...")
    METRICS_ABC_ij, METRICS_NRE_ij, METRICS_CORRECTED_NRE_ij = {"C2ST": np.zeros(N_SAMPLES), "RS_stat": np.zeros(N_SAMPLES), "RS_pvalue": np.zeros(N_SAMPLES)}, {"C2ST": np.zeros(N_SAMPLES), "RS_stat": np.zeros(N_SAMPLES), "RS_pvalue": np.zeros(N_SAMPLES)}, {"C2ST": np.zeros(N_SAMPLES), "RS_stat": np.zeros(N_SAMPLES), "RS_pvalue": np.zeros(N_SAMPLES)}
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
        if np.isnan(sample_abc).any():
            METRICS_ABC_ij["C2ST"][j], METRICS_ABC_ij["RS_pvalue"][j], METRICS_ABC_ij["RS_stat"][j] = 1.0, 0.0, -np.inf
        else:
            METRICS_ABC_ij["C2ST"][j] = c2stest(sample_true, sample_abc)
            METRICS_ABC_ij["RS_stat"][j], METRICS_ABC_ij["RS_pvalue"][j] = ranksums(sample_true, sample_abc)

        if np.isnan(sample_nre).any():
            METRICS_NRE_ij["C2ST"][j], METRICS_NRE_ij["RS_pvalue"][j], METRICS_NRE_ij["RS_stat"][j] = 1.0, 0.0, -np.inf
        else:
            METRICS_NRE_ij["C2ST"][j] = c2stest(sample_true, sample_nre)
            METRICS_NRE_ij["RS_stat"][j], METRICS_NRE_ij["RS_pvalue"][j] = ranksums(sample_true, sample_nre)

        if np.isnan(sample_corrected_nre).any():
            METRICS_CORRECTED_NRE_ij["C2ST"][j], METRICS_CORRECTED_NRE_ij["RS_pvalue"][j], METRICS_CORRECTED_NRE_ij["RS_stat"][j] = 1.0, 0.0, -np.inf
        else:
            METRICS_CORRECTED_NRE_ij["C2ST"][j] = c2stest(sample_true, sample_corrected_nre)
            METRICS_CORRECTED_NRE_ij["RS_stat"][j], METRICS_CORRECTED_NRE_ij["RS_pvalue"][j] =  ranksums(sample_true, sample_corrected_nre)
    return (
        METRICS_ABC_ij,
        METRICS_NRE_ij,
        METRICS_CORRECTED_NRE_ij,
        time.time() - time_start,
    )