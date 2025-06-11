import optax
from optax import tree_utils as otu
from optax.contrib import reduce_on_plateau
from flax import linen as nn
import time
from jax import value_and_grad, jit, random, vmap
import jax.numpy as jnp
from functions.simulation import get_dataset

import numpy as np

def train_loop(key, NN_ARGS, prior_simulator, data_simulator, discrepancy, true_data, X_train = None, y_train = None, X_test = None, y_test =  None, N_POINTS_TRAIN = 0, N_POINTS_TEST = 0, epsilon = jnp.inf, verbose = True, iid = True):
    
    num_epoch_max, num_layers, hidden_size, batch_size, num_batch, learning_rate, wdecay, patience, cooldown, factor, rtol, accumulation_size, learning_rate_min = NN_ARGS["N_EPOCH"], NN_ARGS["NUM_LAYERS"], NN_ARGS["HIDDEN_SIZE"], NN_ARGS["BATCH_SIZE"], NN_ARGS["NUM_BATCH"], NN_ARGS["LEARNING_RATE"], NN_ARGS["WDECAY"], NN_ARGS["PATIENCE"], NN_ARGS["COOLDOWN"], NN_ARGS["FACTOR"], NN_ARGS["RTOL"], NN_ARGS["ACCUMULATION_SIZE"], NN_ARGS["LEARNING_RATE_MIN"]    

    class MLP(nn.Module):
        """A simple multilayer perceptron model for image classification."""
        hidden_sizes= [hidden_size]* num_layers

        @nn.compact
        def __call__(self, x):
            
            for size in self.hidden_sizes:
                x = nn.Dense(features=size)(x)
                x = nn.elu(x)
            x = nn.Dense(features=2)(x)
            logratio = x[:,0]-x[:,1]
            x = nn.log_softmax(x)
            
            return x, logratio
    net = MLP()
    


    # class PhiConditioned(nn.Module):
    #     hidden_sizes: list

    #     @nn.compact
    #     def __call__(self, x_set, theta):  # x_set: (B, M, d), theta: (B, 1)
    #         B, M, d = x_set.shape
    #         theta_exp = jnp.repeat(theta[:, None, :], M, axis=1)  # (B, M, 1)
    #         h = jnp.concatenate([x_set, theta_exp], axis=-1)  # (B, M, d+1)
    #         for hsize in self.hidden_sizes:
    #             h = vmap(nn.Dense(hsize))(h)
    #             h = nn.elu(h)
    #         return h  # (B, M, hidden_size)
    # class MLP(nn.Module):
    #     hidden_sizes: list
    #     iid: bool
    #     M: int  # nombre de points
    #     d: int  # dimension de chaque x_i

    #     @nn.compact
    #     def __call__(self, x):  # x: (B, 1 + M*d)
    #         if self.iid:
    #             theta = x[:, 0:1]                          # (B, 1)
    #             x_flat = x[:, 1:]                          # (B, M*d)
    #             x_set = x_flat.reshape((-1, self.M, self.d))  # (B, M, d)
    #             phi_out = PhiConditioned(self.hidden_sizes)(x_set, theta)  # (B, M, h)
    #             pooled = jnp.sum(phi_out, axis=1)          # (B, h)
    #             h = jnp.concatenate([theta, pooled], axis=-1)
    #         else:
    #             h = x
    #             for size in self.hidden_sizes:
    #                 h = nn.Dense(size)(h)
    #                 h = nn.elu(h)

    #         h = nn.Dense(2)(h)
    #         logratio = h[:, 0] - h[:, 1]
    #         h = nn.log_softmax(h)
    #         return h, logratio

    # net = MLP(hidden_sizes=[hidden_size] * num_layers, iid=iid, d = 1, M = X_train.shape[1]-1)
    


    def get_train_batches(inputs, labels, batch_size, num_batch, key, true_data):
        if inputs is None:
            # print("Simulation of {} batches of size {}".format(num_batch, batch_size))
            for i in range(num_batch):
                inputs_batch, labels_batch, _, key = get_dataset(key, batch_size, prior_simulator, data_simulator, discrepancy, epsilon, true_data)
                yield inputs_batch, labels_batch
        else:
            permutation = random.permutation(key, len(inputs))
            if num_batch ==0 or num_batch> len(inputs)//batch_size: num_batch = len(inputs)//batch_size
            # print("Division in {} batches of size {}".format(num_batch, batch_size))
            
            for i in range(num_batch):
                yield inputs[permutation[i*batch_size:(i+1)*batch_size]], labels[permutation[i*batch_size:(i+1)*batch_size]]

    def dataset_stats(params, inputs, labels, batch_size, num_batch, key, true_data):
        """Computes loss and accuracy over the dataset `data_loader`."""
        all_accuracy = []
        all_loss = []
        for inputs_batch, labels_batch in get_train_batches(inputs, labels, batch_size, num_batch, key, true_data):
            batch_loss, batch_aux = loss_accuracy(params, inputs_batch, labels_batch)
            all_loss.append(batch_loss)
            all_accuracy.append(batch_aux["accuracy"])

        return {"loss": np.mean(all_loss), "accuracy": np.mean(all_accuracy)}

    @jit
    def loss_accuracy(params, inputs, labels):
        logits = net.apply({"params": params}, inputs)[0]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels= labels).mean()
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
        return loss, {"accuracy": accuracy}

    @jit
    def train_step(params, opt_state, inputs_batch, labels_batch):
        """Performs a one step update."""
        (value, aux), grad = value_and_grad(loss_accuracy, has_aux=True)(params, inputs_batch, labels_batch)
        updates, opt_state = opt.update(grad, opt_state, params, value=value)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value, aux

    def params_dict_to_array(params):
        res = []
        for key in params.keys():
            res.append((params[key]["kernel"].T, params[key]["bias"]))
        return res
       
       
    if X_test is None:
        X_test, y_test, _, key = get_dataset(key, N_POINTS_TEST, prior_simulator, data_simulator, discrepancy, epsilon, true_data)



    
    key, subkey = random.split(key)
    
    fake_data = jnp.ones((N_POINTS_TRAIN, X_test.shape[1]))
    params = net.init({"params": subkey}, fake_data)["params"]
    opt = optax.chain(optax.adamw(learning_rate, weight_decay=wdecay),
                      reduce_on_plateau(patience=patience, cooldown=cooldown, factor=factor, rtol=rtol, accumulation_size=accumulation_size))
    
    opt_state = opt.init(params)
    
    train_stats = dataset_stats(params, X_train, y_train, batch_size, num_batch, key, true_data)
    train_accuracy = [train_stats["accuracy"]]
    train_losses = [train_stats['loss']]

    test_stats = dataset_stats(params, X_test, y_test, batch_size, num_batch, key, true_data)
    test_accuracy = [test_stats["accuracy"]]
    test_losses = [test_stats["loss"]]

    lr_scale_history = []
    if verbose:print(f"Initial accuracy: {train_stats['accuracy']:.2%}, Initial test accuracy: {test_stats['accuracy']:.2%}\nTraining for {num_epoch_max} epochs...")
    for epoch in range(num_epoch_max):
        t_epoch = time.time()
        train_accuracy_epoch = []
        train_losses_epoch = []
        key, subkey = random.split(key)
        for inputs_batch, labels_batch in get_train_batches(X_train, y_train, batch_size, num_batch, subkey, true_data):
            params, opt_state, train_loss, train_aux = train_step(
                params, opt_state, inputs_batch, labels_batch
            )
            train_accuracy_epoch.append(train_aux["accuracy"])
            train_losses_epoch.append(train_loss)

        mean_train_accuracy = np.mean(train_accuracy_epoch)
        mean_train_loss = np.mean(train_losses_epoch)

        # fetch the scaling factor from the reduce_on_plateau transform
        lr_scale = otu.tree_get(opt_state, "scale")
        lr_scale_history.append(lr_scale)
        

        train_accuracy.append(mean_train_accuracy)
        train_losses.append(mean_train_loss)

        key, subkey = random.split(key)
        test_stats = dataset_stats(params, X_test, y_test, batch_size, num_batch, subkey, true_data)
        test_accuracy.append(test_stats["accuracy"])
        test_losses.append(test_stats["loss"])
    

        if verbose: print(
            f"Epoch {epoch + 1}/{num_epoch_max}, mean train accuracy:"
            f" {mean_train_accuracy:.2%}, mean test accuracy: {test_stats['accuracy']:.2%}, lr scale: {otu.tree_get(opt_state, 'scale')} in {time.time()-t_epoch:.2f} sec"
        )
        
        if learning_rate * lr_scale < learning_rate_min:
            if verbose: print("Learning rate reached {:.2E}, stopping training".format(learning_rate_min))
            break

    params = params_dict_to_array(params)

    return  params, train_accuracy, train_losses, test_accuracy, test_losses, key

