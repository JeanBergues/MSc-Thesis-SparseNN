import numpy as np
from time import perf_counter_ns
import keras as ks
import tensorflow as tf

import lassonet as lsn
import torch as pt

def estimate_starting_lambda(theta, W, M, starting_lambda = 1e-5, factor = 2, tol = 1e-6, max_iter_per_lambda = 10000, verbose=False, steps_back = 3):
    initial_theta = theta
    dense_W = W
    dense_theta = initial_theta
    l_test = starting_lambda

    # Apply the hier-prox until theta is completely zeroed out
    while not np.sum(np.abs(dense_theta)) == 0:
        dense_theta = initial_theta
        dense_W = W
        l_test = l_test * factor
        if verbose: print(f"Testing lambda={l_test}")

        for _ in range(max_iter_per_lambda):
            theta_new, W_new = hier_prox(dense_theta, dense_W, l_test, M)
            if np.max(np.abs(dense_theta - theta_new)) < tol: break # Check if the theta is still changing
            dense_theta = theta_new
            dense_W = W_new

    return l_test / (factor**steps_back)


def train_lasso_path(network, 
                     starting_lambda, 
                     X_train, 
                     X_val, 
                     y_train, 
                     y_val,
                     optimizer,
                     loss_func,
                     use_faster_fit = True,
                     use_faster_eval = True,
                     use_best_weights = True,
                     train_until_k = 0,
                     lr=1e-3, 
                     M=10, 
                     pm=2e-2, 
                     max_epochs_per_lambda = 100, 
                     patience = 10, 
                     verbose=False,
                     regressor=True,
                     save_best_network=False,
                     L1_penalty = True,
                     X_test = None,
                     y_test = None,
                     max_lambda = np.inf,
                     min_improvement = 0.99):
    
    minimized = False
    res_k = []
    res_theta = []
    res_val = []
    res_l = []
    res_oos = []
    l = starting_lambda / (1 + pm)
    k = X_train.shape[1]

    while k > train_until_k and l < max_lambda:
        l = (1 + pm) * l
        res_l.append(l)
        best_val_obj = np.inf
        e_since_best_val = 1
        train_time = 0
        prox_time = 0

        for b in range(max_epochs_per_lambda):
            start_train = perf_counter_ns()
            if use_faster_fit:
                with tf.GradientTape() as tape:
                    logits = network(X_train, training=True)
                    losses = loss_func(y_train, logits)
                gradients = tape.gradient(losses, network.trainable_weights)

                # Update theta and W using losses
                optimizer.apply_gradients(zip(gradients, network.trainable_weights))
            else:
                network.fit(X_train, y_train, verbose='0', epochs=1)

            train_time += perf_counter_ns() - start_train
            
            # Update using HIER-PROX
            start_prox = perf_counter_ns()
            theta_new, W_new = hier_prox(network.get_layer('skip_layer').get_weights()[0], network.get_layer('gw_layer').get_weights()[0], lr*l, M)

            new_skip_layer = network.get_layer('skip_layer').get_weights()
            new_skip_layer[0] = theta_new.reshape((-1, 1))
            network.get_layer('skip_layer').set_weights(new_skip_layer)

            new_gw_layer = network.get_layer('gw_layer').get_weights()
            new_gw_layer[0] = W_new
            network.get_layer('gw_layer').set_weights(new_gw_layer)

            prox_time += perf_counter_ns() - start_prox

            start_train = perf_counter_ns()

            if use_faster_eval:
                val_logits = network(X_val, training=False)
                val_obj = loss_func(y_val, val_logits)
            else:
                val_obj = network.evaluate(X_val, y_val, verbose='0') if regressor else network.evaluate(X_val, y_val, verbose='0')[0]
                if L1_penalty:
                    val_obj += l * np.sum(np.abs(theta_new))

            train_time += perf_counter_ns() - start_train

            e_since_best_val += 1
            if val_obj < best_val_obj * min_improvement:
                best_val_obj = val_obj
                best_weights = network.get_weights()
                e_since_best_val = 1
            
            if e_since_best_val == patience:
                if verbose: print(f"Ran for {b+1} epochs before early stopping.")
                if use_best_weights: network.set_weights(best_weights)
                break

            if b == max_epochs_per_lambda - 1: 
                if verbose: print(f"Ran for the full {max_epochs_per_lambda} epochs.")
                if use_best_weights: network.set_weights(best_weights)

        print(f"Training time (ms): {train_time // 1e6}")
        print(f"Proximal time (ms): {prox_time // 1e6}")

        last_theta = network.get_layer('skip_layer').get_weights()[0]
        k = np.shape(np.nonzero(last_theta))[1]

        res_k.append(k)
        res_theta.append(last_theta)

        val_acc = network.evaluate(X_val, y_val) if regressor else network.evaluate(X_val, y_val)[1]
        res_val.append(val_acc)

        if X_test is not None:
            test_acc = network.evaluate(X_test, y_test) if regressor else network.evaluate(X_test, y_test)[1]
            res_oos.append(test_acc)

        if len(res_val) > 1:
            if val_acc < res_val[-2] and save_best_network and k < X_train.shape[1] and False:
                print("Saved new best model") 
                network.save('best_network.keras')
                network.save_weights('best_network.weights.h5')
                minimized = True
        
        print(f"--------------------------------------------------------------------- K = {k}, lambda = {l:.3f}, MSE = {val_acc:.6f}, TEST MSE = {test_acc if X_test is not None else 0:.6f} \n\n")
    
    if train_until_k > 0 or max_lambda != np.inf:
        network.save('best_network.keras')
        network.save_weights('best_network.weights.h5')

    if X_test is not None:
        return (res_k, res_theta, res_val, res_l, res_oos, network)
    else:
        return (res_k, res_theta, res_val, res_l, minimized, network)


def hier_prox(theta: np.ndarray, W: np.ndarray, l: float, M: float, stable=True) -> tuple[np.ndarray, np.ndarray]:
    # Check if the shapes are correct
    theta = theta.ravel()
    d = theta.shape[0]
    K = W.shape[1]
    assert W.shape[0] == d

    # Sort the absolute values of W in decreasing order and create cumulative sum
    sorted_W = -np.sort(-np.abs(W))
    W_sum = np.cumsum(sorted_W, axis=1)

    # Calculate w_m vectorized
    m = np.arange(start=0, stop=K+1)
    padded_Wsum = np.concatenate([np.zeros((d, 1)), W_sum], axis=1)
    threshold = np.clip(np.repeat(np.abs(theta).reshape((-1, 1)), K+1, axis=1) + M * padded_Wsum - np.full_like(padded_Wsum, l), 0, np.inf)
    w_m = M / (1 + m * (M**2)) * threshold

    # Check for condition
    upper_bound = np.concatenate([np.repeat(np.inf, d).reshape((-1, 1)), sorted_W], axis=1)
    lower_bound = np.concatenate((sorted_W, np.zeros((d, 1))), axis=1)
    if stable:
        m_tilde_condition = np.logical_and(w_m >= lower_bound, w_m >= lower_bound)
    else:
        m_tilde_condition = np.logical_and(w_m <= upper_bound, w_m >= lower_bound)

    # Select only the first true value
    first_m_tilde = m_tilde_condition.cumsum(axis=1).cumsum(axis=1) == 1
    m_tilde = w_m[first_m_tilde]

    # Update the weights
    theta_out = (1/M) * np.sign(theta) * m_tilde
    W_out = np.sign(W) * np.minimum(np.abs(W), np.repeat(m_tilde.reshape((-1, 1)), K, axis=1))

    return (theta_out, W_out)


def paper_lassonet_results(Xt, Xv, yt, yv, K=(10,), verbose=0, pm=0.02, M=10, patiences=(100, 10), max_iters=(1000, 100), l_start="auto", use_custom_optimizer=False, regressor=True, tol=0.99):
    import lassonet as lsn
    from functools import partial
    import torch as pt
    if regressor:
        if use_custom_optimizer:
            lassoC = lsn.LassoNetRegressor(verbose=verbose, hidden_dims=K, path_multiplier=(1+pm), M=M,
                                        patience=patiences, n_iters=max_iters, random_state=1234, torch_seed=1234, lambda_start=l_start, backtrack=True, tol=tol,
                                        optim=(partial(pt.optim.Adam, lr=0.01), partial(pt.optim.SGD, lr=0.01, momentum=0.3)))
        else:
            lassoC = lsn.LassoNetRegressor(verbose=verbose, hidden_dims=K, path_multiplier=(1+pm), M=M,
                                        patience=patiences, n_iters=max_iters, random_state=1234, torch_seed=1234, lambda_start=l_start, backtrack=True, tol=tol)
            
    else:
       lassoC = lsn.LassoNetClassifier(verbose=verbose, hidden_dims=K, path_multiplier=(1+pm), M=M,
                                        patience=patiences, n_iters=max_iters, random_state=1234, torch_seed=1234, lambda_start=l_start, backtrack=True, tol=tol) 
    history = lassoC.path(Xt, yt, X_val=Xv, y_val=yv)

    res_k = np.zeros(len(history))
    res_val = np.zeros(len(history))
    res_l = np.zeros(len(history))
    res_theta = []

    for i, h in enumerate(history):
        res_k[i] = h.selected.sum()
        res_val[i] = h.val_loss
        res_l[i] = h.lambda_
        res_theta = h.selected.numpy()

    return (res_k, res_theta, res_val, res_l)


def paper_lassonet_mask(Xt, Xv, yt, yv, K=(10,), verbose=0, pm=0.02, M=10, patiences=(100, 10), max_iters=(1000, 100), l_start="auto", use_custom_optimizer=False, regressor=True, n_features=0):
    import lassonet as lsn
    from functools import partial
    import torch as pt
    if regressor:
        if use_custom_optimizer:
            lassoC = lsn.LassoNetRegressor(verbose=verbose, hidden_dims=K, path_multiplier=(1+pm), M=M,
                                        patience=patiences, n_iters=max_iters, random_state=1234, torch_seed=1234, lambda_start=l_start, backtrack=True, tol=1,
                                        optim=(partial(pt.optim.Adam, lr=0.01), partial(pt.optim.SGD, lr=0.01, momentum=0.9)))
        else:
            lassoC = lsn.LassoNetRegressor(verbose=verbose, hidden_dims=K, path_multiplier=(1+pm), M=M,
                                        patience=patiences, n_iters=max_iters, random_state=1234, torch_seed=1234, lambda_start=l_start, backtrack=True, tol=1)
            
    else:
       lassoC = lsn.LassoNetClassifier(verbose=verbose, hidden_dims=K, path_multiplier=(1+pm), M=M,
                                        patience=patiences, n_iters=max_iters, random_state=1234, torch_seed=1234, lambda_start=l_start, backtrack=True, tol=1) 
    history = lassoC.path(Xt, yt, X_val=Xv, y_val=yv)
    lowest_obj = np.inf

    for h in history:
        if h.val_loss < lowest_obj and h.selected.sum() < Xt.shape[1]:
            lowest_obj = h.val_loss
            obj_h = h.selected.data.numpy()
        if h.selected.sum() <= n_features and n_features > 0:
            frac_h = h.selected.data.numpy() if h.selected.sum() > 0 else backup_h
            return h.selected.data.numpy()
        else:
            backup_h = h.selected.data.numpy()

    if np.sum(obj_h) > 0:
        print("Minimized objective!")
        return obj_h
    else:
        print("Fractional features!")
        return frac_h


def return_LassoNet_results(dense, Xt, Xv, yt, yv, pm=0.02, activation='relu', M=10, max_iters=(1000, 100), patiences=(100, 10), 
                            print_lambda=False, print_path=False, a=1e-3, starting_lambda=None, mom=0.9, faster_fit=True, steps_back = 3, best_weights=True, regression=True,
                            Xtest = None, ytest=None, max_lambda=np.inf, min_improvement = 0.99):
    if starting_lambda == None:
        starting_lambda = estimate_starting_lambda(dense.get_layer('skip_layer').get_weights()[0], dense.get_layer('gw_layer').get_weights()[0], M, verbose=print_lambda, steps_back=steps_back) / a

    res_k, res_theta, res_val, res_l, res_oos, final_net = train_lasso_path(
        dense, starting_lambda, Xt, Xv, yt, yv, ks.optimizers.SGD(learning_rate=a, momentum=mom), ks.losses.MeanSquaredError() if regression else ks.losses.SparseCategoricalCrossentropy(from_logits=True), 
        train_until_k=0, use_faster_fit=faster_fit, lr=a, M=M, pm=pm, max_epochs_per_lambda=max_iters[1], use_best_weights=best_weights,
        patience=patiences[1], verbose=print_path, use_faster_eval=False, regressor=regression, X_test=Xtest, y_test=ytest, max_lambda=max_lambda, min_improvement=min_improvement)

    return (res_k, res_theta, res_val, res_l, res_oos)


def return_LassoNet_mask(dense, Xt, Xv, yt, yv, pm=0.02, activation='relu', M=10, max_iters=(1000, 100), patiences=(100, 10), 
                            print_lambda=False, print_path=False, a=1e-3, starting_lambda=None, mom=0.9, faster_fit=True, steps_back = 3, best_weights=True, regression=True, n_features=0, return_best_model=False):
    if starting_lambda == None:
        starting_lambda = estimate_starting_lambda(dense.get_layer('skip_layer').get_weights()[0], dense.get_layer('gw_layer').get_weights()[0], M, verbose=print_lambda, steps_back=steps_back) / a

    return_model = None
    res_k, res_theta, res_val, res_l, minimized = train_lasso_path(
        dense, starting_lambda, Xt, Xv, yt, yv, ks.optimizers.SGD(learning_rate=a, momentum=mom), ks.losses.MeanSquaredError() if regression else ks.losses.SparseCategoricalCrossentropy(from_logits=True), 
        train_until_k=n_features, use_faster_fit=faster_fit, lr=a, M=M, pm=pm, max_epochs_per_lambda=max_iters[1], use_best_weights=best_weights,
        patience=patiences[1], verbose=print_path, use_faster_eval=False, regressor=regression, save_best_network=return_best_model)
    
    best_theta = res_theta[-1]
    best_v = np.inf
    for i, v in enumerate(res_val):
        if v < best_v:
            best_v = v
            best_theta = res_theta[i]

    if return_best_model:
        return_model = ks.models.load_model('best_network.keras')
        return_model.load_weights('best_network.weights.h5')
        
    return (best_theta, return_model)