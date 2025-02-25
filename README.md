# ABC-SBI

Functions and examples (see https://www.overleaf.com/project/65fd81d35ac3b80aa8db4607)


Files descriptions: 
- Functions:
    * `simulations.py`: `ABC_epsilon` functions that returns samples $(\theta_i,z_i)\sim \pi_\epsilon(\theta,z\mid x_{\text{obs}})$ and `get_dataset` function that returns a Dataset $X = ((\theta_i, z_i, y_i))_{i =1,\dots,N}$ such that if $y_i =0$, $(\theta_i,z_i)\sim \pi_\epsilon(\theta,z\mid x_{\text{obs}})$ and if $y_i = 1$, $(\theta_i,z_i)\sim \pi_\epsilon(\theta\mid x_{\text{obs}}) \pi_\epsilon(z\mid x_{\text{obs}})$ (the two ABC marginals)
    * `training.py`: `train_loop` that return the NN parameters for the classifier between the two models
    * `SBC.py`: `SBC_epsilon` function to perform ABC-SBC 

- Examples:
    * Gauss-Gauss:
    * Linear-Reg:
    * Logistic-Reg:
    * POTUS-full:
    * POTUS-nat:

- cluser: Cluster outputs
