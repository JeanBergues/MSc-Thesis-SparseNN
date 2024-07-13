import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter_ns
import keras as ks
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as mt
import lassonet as lsn
import torch as pt

from train_nn import calc_investment_returns, return_MLP_skip_estimator

np.random.seed(1234)
tf.random.set_seed(1234)
ks.utils.set_random_seed(1234)
pt.manual_seed(1234)


def estimate_starting_lambda(theta, W, M, starting_lambda = 1e-3, factor = 2, tol = 1e-6, max_iter_per_lambda = 100000, verbose=False, divide_result = 8):
    initial_theta = theta
    dense_W = W
    dense_theta = initial_theta
    l_test = starting_lambda

    while not np.sum(np.abs(dense_theta)) == 0:
        dense_theta = initial_theta
        l_test = l_test * factor
        if verbose: print(f"Testing lambda={l_test}")

        for _ in range(max_iter_per_lambda):
            theta_new, _ = hier_prox(dense_theta, dense_W, l_test, M)
            if np.max(np.abs(dense_theta - theta_new)) < tol: break # Check if the theta is still changing
            dense_theta = theta_new

    return l_test / divide_result


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
                     verbose=False):
    
    res_k = []
    res_theta = []
    res_val = []
    res_l = []
    l = starting_lambda / (1 + pm)
    k = X_train.shape[1]

    while k > train_until_k:
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
            # val_obj = network.evaluate(X_val, y_val, verbose='0')[0]

            if use_faster_eval:
                val_logits = network(X_val, training=False)
                val_obj = loss_func(y_val, val_logits)
            else:
                val_obj = network.evaluate(X_val, y_val, verbose='0')

            train_time += perf_counter_ns() - start_train

            e_since_best_val += 1
            if val_obj < best_val_obj:
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

        val_acc = network.evaluate(X_val, y_val)
        res_val.append(val_acc)
        
        print(f"--------------------------------------------------------------------- K = {k}, lambda = {l:.1f}, MSE = {val_acc:.6f} \n\n")
    
    return (res_k, res_theta, res_val, res_l)


def hier_prox(theta: np.ndarray, W: np.ndarray, l: float, M: float) -> tuple[np.ndarray, np.ndarray]:
    # Assert correct sizes
    theta = theta.ravel()
    d = theta.shape[0]
    K = W.shape[1]
    assert W.shape[0] == d

    # Order the weights
    sorted_W = -np.sort(-np.abs(W))

    # Calculate w_m's
    W_sum = np.cumsum(sorted_W, axis=1)
    m = np.arange(start=1, stop=K+1)
    threshold = np.clip((np.repeat(np.abs(theta).reshape((-1, 1)), K, axis=1) + M * W_sum) - np.full_like(W_sum, l), 0, np.inf)
    w_m = (M * threshold) / (1 + m * (M**2)) 

    # Check for condition
    m_tilde_condition = np.logical_and(w_m <= sorted_W, w_m >= np.concatenate((sorted_W, np.zeros((d, 1))), axis=1)[:,1:])

    # Find the first true value per row
    m_tilde_first_only = np.zeros_like(m_tilde_condition, dtype=bool)
    idx = np.arange(len(m_tilde_condition)), m_tilde_condition.argmax(axis=1)
    m_tilde_first_only[idx] = m_tilde_condition[idx]

    # Set the first value of each row to true if all other values in the row are false
    set_first_true_array = np.full_like(m_tilde_first_only, False)
    set_first_true_array[:,-1] = np.sum(m_tilde_first_only, axis=1) < 1
    m_tilde_first_only = np.logical_or(m_tilde_first_only, set_first_true_array)
    m_tilde = w_m[m_tilde_first_only]

    # Calculate output
    theta_out = (1/M) * np.sign(theta) * m_tilde
    W_out = np.sign(W) * np.minimum(np.abs(W), np.repeat(m_tilde.reshape((-1, 1)), K, axis=1))

    return (theta_out, W_out)


def paper_lassonet_mask(Xt, Xv, yt, yv, K=(10,), verbose=0, pm=0.02, M=10, patiences=(100, 10), max_iters=(1000, 100), l_start="auto"):
    lassoC = lsn.LassoNetRegressor(verbose=verbose, hidden_dims=K, path_multiplier=(1+pm), M=M, patience=patiences, n_iters=max_iters, torch_seed=1234, lambda_start=l_start)
    history = lassoC.path(Xt, yt, X_val=Xv, y_val=yv)

    res_k = np.zeros(len(history))
    res_val = np.zeros(len(history))
    res_l = np.zeros(len(history))

    for i, h in enumerate(history):
        res_k[i] = h.selected.sum()
        res_val[i] = h.val_loss
        res_l[i] = h.lambda_

    return (res_k, res_val, res_l)


def return_LassoNet_mask(dense, Xt, Xv, yt, yv, K=[10], pm=0.02, activation='relu', M=10, max_iters=(1000, 100), patiences=(100, 10), print_lambda=False, print_path=False, a=1e-3):
    # dense.compile(optimizer=ks.optimizers.SGD(learning_rate=a, momentum=0.9), loss=ks.losses.MeanSquaredError())

    starting_lambda = estimate_starting_lambda(dense.get_layer('skip_layer').get_weights()[0], dense.get_layer('gw_layer').get_weights()[0], M, verbose=print_lambda, divide_result=10)

    res_k, res_theta, res_val, res_l = train_lasso_path(
        dense, starting_lambda, Xt, Xv, yt, yv, ks.optimizers.SGD(learning_rate=a, momentum=0.9), ks.losses.MeanSquaredError(), 
        train_until_k=0, use_faster_fit=True, lr=a, M=M, pm=pm, max_epochs_per_lambda=max_iters[1], use_best_weights=True,
        patience=patiences[1], verbose=print_path, use_faster_eval=False)

    return (res_k, res_val, res_l)


###############################################################################################################################################################################################

day_df = pd.read_csv(f'pct_btc_day.csv')
hour_df = pd.read_csv(f'pct_btc_hour.csv')

freq = 24

# raw_returns = day_df.close.pct_change(1)[1:].to_numpy()
open_returns =  day_df.open.to_numpy()
high_returns =  day_df.high.to_numpy()
low_returns =   day_df.low.to_numpy()
close_returns = day_df.close.to_numpy()
vol_returns =   day_df.volume.to_numpy()
volNot_returns =day_df.volumeNotional.to_numpy()
trades_returns =day_df.tradesDone.to_numpy()

open_h_returns =  hour_df.open.to_numpy()
high_h_returns =  hour_df.high.to_numpy()
low_h_returns =   hour_df.low.to_numpy()
close_h_returns = hour_df.close.to_numpy()
vol_h_returns =   hour_df.volume.to_numpy()
volNot_h_returns =hour_df.volumeNotional.to_numpy()
trades_h_returns =hour_df.tradesDone.to_numpy()

dlag_opt = [2]
use_hlag = [True]

for d_nlags in dlag_opt:
    for use_h in use_hlag:
        h_nlags = d_nlags if use_h else 0

        bound_lag = max(d_nlags, ((h_nlags-1)//freq + 1))
        y_raw = close_returns[bound_lag:].reshape(-1, 1)
        Xlist = np.arange(1, len(y_raw) + 1).reshape(-1, 1)
        if h_nlags > 0:
            for t_h in range(0, h_nlags):
                Xlist = np.concatenate(
                    [
                        Xlist,
                        open_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                        high_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                        low_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                        close_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                        vol_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                        volNot_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                        trades_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                    ], axis=1)
        if d_nlags > 0:
            for t in range(0, d_nlags):
                Xlist = np.concatenate(
                    [
                        Xlist,
                        open_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                        high_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                        low_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                        close_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                        vol_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                        volNot_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                        trades_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                    ], axis=1)

        Xlist = Xlist[:, 1:]
        X_pp = pp.MinMaxScaler().fit(Xlist)
        y_pp = pp.MinMaxScaler().fit(y_raw)
        Xvoortest = X_pp.transform(Xlist)
        yvoortest = y_pp.transform(y_raw)

        Xtrain, Xtest, ytrain, ytest = ms.train_test_split(Xvoortest, yvoortest, test_size=365, shuffle=False)
        print("Data has been fully transformed and split")

        n_repeats = 1
        ytest = y_pp.inverse_transform(ytest.reshape(1, -1)).ravel()

        best_K = [200, 100, 50, 20]

        Xt, Xv, yt, yv = ms.train_test_split(Xtrain, ytrain, test_size=120, shuffle=False)
        tXt = tf.convert_to_tensor(Xt)
        tXv = tf.convert_to_tensor(Xv)
        tyt = tf.convert_to_tensor(yt)
        tyv = tf.convert_to_tensor(yv)

        # Run for M variations
        HP_opts = [0.02]
        HP_results = []

        USE_PAPER_LASSONET = False
        if not USE_PAPER_LASSONET:
            initial_model = return_MLP_skip_estimator(tXt, tXv, tyt, tyv, k=Xt.shape[1], activation='tanh', K=best_K, verbose=1, patience=100, epochs=1000)
            initial_model.save('temp_network.keras')
            initial_model_best_weights = initial_model.get_weights()

        for m in HP_opts:
            np.random.seed(1234)
            tf.random.set_seed(1234)
            ks.utils.set_random_seed(1234)
            pt.manual_seed(1234)

            if USE_PAPER_LASSONET:
                res_k, res_val, res_l = paper_lassonet_mask(
                    Xt, Xv, yt, yv, K=tuple(best_K), verbose=2, pm=0.001, M=m, patiences=(100, 5), max_iters=(10000, 1000), l_start=200)
            else:
                # network = ks.models.load_model('temp_network.keras')
                # network.set_weights(initial_model_best_weights)
                # res_k, res_val, res_l = return_LassoNet_mask(
                #     network, Xt, Xv, yt, yv, K=best_K, pm=m, M=10, patiences=(100, 10), max_iters=(1000, 100), print_path=True, print_lambda=True)
                res_k, res_val, res_l = return_LassoNet_mask(
                    initial_model, tXt, tXv, tyt, tyv, K=best_K, pm=m, M=10, patiences=(100, 5), max_iters=(10000, 100), print_path=True, print_lambda=True)
            
            HP_results.append((res_k, res_val, res_l))

        # Plot selected features against mse
        fig = plt.figure(figsize=(10, 4))
        for m, res in zip(HP_opts, HP_results):
            fig = sns.lineplot(x=np.array(res[0]), y=np.array(res[1]), markers=True, drawstyle='steps-pre', size=10)
        
        # plt.legend(labels=[f"M={l}" for l in HP_opts])
        legd = fig.get_legend()
        for t, l in zip(legd.texts, HP_opts):
            t.set_text(f"M={l}")

        sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))
        plt.xlabel("selected features")
        plt.ylabel("mse")
        plt.show()

        # Plot selected features against lambda
        fig = plt.figure(figsize=(10, 4))
        for m, res in zip(HP_opts, HP_results):
            fig = sns.lineplot(x=np.array(res[2]), y=np.array(res[0]), markers=True, drawstyle='steps-pre', size=10)
        
        # plt.legend(labels=[f"M={l}" for l in HP_opts])
        legd = fig.get_legend()
        for t, l in zip(legd.texts, HP_opts):
            t.set_text(f"M={l}")

        sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))
        plt.xlabel(r'$\lambda$')
        plt.ylabel("selected features")
        plt.show()
        



        # Xtt, Xtv, ytt, ytv = ms.train_test_split(Xtrain, ytrain, test_size=30, shuffle=False)
        # final_mask = np.ravel(return_LassoNet_mask(Xtt, Xtv, ytt, ytv, K=best_K, activation='tanh', M=5, pm=0.02, epochs=2000, 
                                                #    patience=100, print_path=True, nfeat=0, plot=True, ppatience=10, pepochs=500) != 0)