import numpy as np
import matplotlib.pyplot as plt
from functions.simulation import NRE_corrected_posterior_pdf, NRE_posterior_pdf, find_grid_explorative
from scipy.stats import gaussian_kde
def plot_metric(alphas, abc, nre, corrected_nre, path_name = "", show = True, metric_name = "", title = ""):
    mean_abc = np.mean(abc, axis=1)
    std_abc = np.std(abc, axis=1)
    mean_nre = np.mean(nre, axis=1)
    std_nre = np.std(nre, axis=1)
    mean_corrected_nre = np.mean(corrected_nre, axis=1)
    std_corrected_nre = np.std(corrected_nre, axis=1)

     
    mean_abc = np.mean(abc, axis=1)
    std_abc = np.std(abc, axis=1)
    mean_nre = np.mean(nre, axis=1)
    std_nre = np.std(nre, axis=1)
    mean_corrected_nre = np.mean(corrected_nre, axis=1)
    std_corrected_nre = np.std(corrected_nre, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.errorbar(alphas, mean_abc, yerr=1.96*std_abc/np.sqrt(abc.shape[1]), label="ABC", fmt="o-", color="orange")
    ax.errorbar(alphas, mean_nre, yerr=1.96*std_nre/np.sqrt(nre.shape[1]), label="NRE", fmt="o-", color="red")
    ax.errorbar(alphas, mean_corrected_nre, yerr=1.96*std_corrected_nre/np.sqrt(corrected_nre.shape[1]), label="ABC-corrected NRE", fmt="o-", color="blue")
    ax.set_xlabel("$\\alpha$")
    if len(metric_name)>0: ax.set_ylabel(metric_name)
    ax.legend()
    ax.set_xscale("log")
    ax.invert_xaxis()
    if len(title)>0: ax.set_title(title)
    
    if len(path_name)>0:fig.savefig(path_name)
    if show: plt.show()
    plt.close(fig)
    print("FIGURE CREATED at {}".format(path_name))
    
    
def plot_metric_for_a_dataset(metric_name, ALPHAS, METRICS_ABC, METRICS_NRE, METRICS_CORRECTED_NRE,N_SAMPLES, PATH_NAME = "", show = True, title = ""):
    abc = np.array([[METRICS_ABC[a][metric_name][j] for j in range(N_SAMPLES)] for a in ALPHAS]).reshape(len(ALPHAS), -1)
    nre = np.array([[METRICS_NRE[a][metric_name][j] for j in range(N_SAMPLES)] for a in ALPHAS]).reshape(len(ALPHAS), -1)
    corrected_nre = np.array([[METRICS_CORRECTED_NRE[a][metric_name][j] for j in range(N_SAMPLES)] for a in ALPHAS]).reshape(len(ALPHAS), -1)
    plot_metric(ALPHAS, abc, nre, corrected_nre, PATH_NAME, show = show, metric_name = metric_name, title = title)
    

def plot_metric_for_many_datasets(metric_name, ALPHAS, METRICS_ABC, METRICS_NRE, METRICS_CORRECTED_NRE, N_SAMPLES, N_DATASETS, PATH_NAME = "", show = True, title = ""):
    abc = np.array([[METRICS_ABC[i][a][metric_name][j] for i in range(N_DATASETS) for j in range(N_SAMPLES)] for a in ALPHAS]).reshape(len(ALPHAS), -1)
    nre = np.array([[METRICS_NRE[i][a][metric_name][j] for i in range(N_DATASETS) for j in range(N_SAMPLES)] for a in ALPHAS]).reshape(len(ALPHAS), -1)
    corrected_nre = np.array([[METRICS_CORRECTED_NRE[i][a][metric_name][j] for i in range(N_DATASETS) for j in range(N_SAMPLES)] for a in ALPHAS]).reshape(len(ALPHAS), -1)
    
    plot_metric(ALPHAS, abc, nre, corrected_nre, PATH_NAME, show = show, metric_name = metric_name, title = title)
    
    
    
def plot_posterior_comparison(params, TRUE_DATA, thetas_abc, prior_dist, file_name="", show=True, N_GRID = 1000, true_posterior_pdf = None, N_KDE = 10000):
    prior_logpdf = lambda x: prior_dist.logpdf(x)
    kde_approx = gaussian_kde(thetas_abc[:N_KDE])
    min_init, max_init = prior_dist.interval(.999)
    grid_nre, pdf_nre = find_grid_explorative(lambda x: NRE_posterior_pdf(params, x, TRUE_DATA, prior_logpdf), N_GRID, N_GRID, min_init, max_init)
    grid_corrected_nre, pdf_corrected_nre = find_grid_explorative(lambda x: NRE_corrected_posterior_pdf(params, x, TRUE_DATA, kde_approx), N_GRID, N_GRID, min_init, max_init)
    grid_abc, pdf_abc = find_grid_explorative(lambda x: kde_approx.logpdf(x), N_GRID, N_GRID, min_init, max_init)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    ax.plot(grid_nre, pdf_nre/np.trapz(pdf_nre, grid_nre), label="NRE", color="red")
    ax.plot(grid_corrected_nre, pdf_corrected_nre/np.trapz(pdf_corrected_nre, grid_corrected_nre), label="Corrected NRE", color = "blue", linestyle="--")
    ax.plot(grid_abc, pdf_abc/np.trapz(pdf_abc, grid_abc), label="ABC", color = "orange")
    if true_posterior_pdf is not None:
        grid_true, pdf_true = find_grid_explorative(lambda x: true_posterior_pdf(x, TRUE_DATA), N_GRID, N_GRID, min_init, max_init)
        ax.plot(grid_true, pdf_true/np.trapz(pdf_true, grid_true), label="True", color = "green")
        min_true = np.min(grid_true)
        max_true = np.max(grid_true)
        ax.set_xlim(min_true-(max_true-min_true)*.2, max_true+(max_true-min_true)*.2)
    ax.legend()

    ax.set_title("Posterior comparison")
    if len(file_name)>0: fig.savefig(file_name)
    if show: plt.show()
    plt.close(fig)
    print("FIGURE CREATED at {}".format(file_name))
    