# ABC-SBI

This repository contains functions and examples for ABC-SBI.  
You can find an Overleaf project with additional documentation and details here: [Overleaf Project](https://www.overleaf.com/project/65fd81d35ac3b80aa8db4607).

---

## 📂 File Descriptions  

### 🔧 **Functions**  

#### `simulations.py`  
Contains functions related to Approximate Bayesian Computation (ABC):  
- **`ABC_epsilon`**: Returns samples $(\theta_i, z_i) \sim \pi_\epsilon(\theta, z \mid x_{\text{obs}})$.  
- **`get_dataset`**: Generates a dataset $X = (X_1, \dots, X_N)$, where $X_i = (\theta_i, z_i, y_i) such that:
  - If $y_i = 0$, then $(\theta_i, z_i) \sim \pi_\epsilon(\theta, z \mid x_{\text{obs}})$.  
  - If $y_i = 1$, then $(\theta_i', z_i) \sim \pi_\epsilon(\theta' \mid x_{\text{obs}}) \pi_\epsilon(z \mid x_{\text{obs}})$ (i.e., the two ABC marginals).  
- **`NRE_posterior_sample`**: Returns samples from $\hat{\pi}(\theta \mid x) \propto \pi(\theta) \hat{r}(x \mid \theta)$.  
- **`NRE_corrected_posterior_sample`**: Returns samples from $\hat{\pi}(\theta \mid x) \propto \hat \pi_\epsilon(\theta \mid x_{\text{obs}})\hat{r}(x \mid \theta)$.  

#### `training.py`  
- **`train_loop`**: Trains a neural network (NN) classifier to distinguish between the two models.  

#### `SBC.py`  
- **`SBC_epsilon`**: Performs ABC-SBC (Simulation-Based Calibration).  

#### `metrics.py`  
Contains various statistical tests and evaluation functions:  
- **`c2stest`**: Implements the Classifier 2-Sample Test (C2ST).  
- **`ranksums`**: Computes the Wilcoxon Rank-Sum Test.  
- **`evaluate_metrics.py`**: Evaluates the performance of the ABC method using a given dataset (`TRUE_DATA`), NN parameters (`params`), and ABC samples (`thetas_abc`). Outputs the results of the C2ST and Rank-Sum tests.  

#### `plots.py`  
- **`plot_metric`**: Generates plots comparing ABC-NRE with and without correction for different $\epsilon = q_\alpha (d(x, x_{\text{obs}}))$.
- **`plot_posterior_comparison`**: Generates a plot to compare the posterior from the NRE, the corrected NRE and the ABC pseudo-posterior.

#### `save.py`  
- Functions to save method outputs in CSV or pickle format.  

#### `cluster_pattern.py`  
- Provides a file execution pattern for running experiments on a computing cluster.  

---

## 📝 **Examples**  

### Available Examples  
- **Gauss-Gauss (1D, known $\sigma$)**  
- **Gauss-Gauss (Multidimensional, known $\sigma$)**  
- **Linear Regression**  
- **Logistic Regression** *(TO DO)*  
- **POTUS-full** *(TO DO)*  
- **POTUS-nat** *(TO DO)*  

For each example, run `cluster_pattern_example.py`. The outputs will be saved in the `clean_results` folder. For each hyperparameter the results are stored in `csv` and `pickles` folders and display in a `figures` folder. 

---

## 💻 **Cluster Outputs**  
Cluster-generated results are stored in the `cluster` directory.  
