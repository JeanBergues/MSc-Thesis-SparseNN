import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter_ns

np.random.seed(1234)
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(1234)

import keras as ks
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as mt
import lassonet as lsn

from train_nn import calc_investment_returns, return_MLP_skip_estimator


def estimate_starting_lambda(theta, W, M, starting_lambda = 1e-3, factor = 2, tol = 1e-6, max_iter_per_lambda = 1000, verbose=False, divide_result = 1):
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
    set_first_true_array[:,0] = np.sum(m_tilde_first_only, axis=1) < 1
    m_tilde_first_only = np.logical_or(m_tilde_first_only, set_first_true_array)
    m_tilde = w_m[m_tilde_first_only]

    # Calculate output
    theta_out = (1/M) * np.sign(theta) * m_tilde
    W_out = np.sign(W) * np.minimum(np.abs(W), np.repeat(m_tilde.reshape((-1, 1)), K, axis=1))

    return (theta_out, W_out)


def paper_lassonet_mask(X1, y1, K=(10,), verbose=0, n_features=0, pm=0.02, M=10, plot=False, test_size=60):
    Xt, Xv, yt, yv = ms.train_test_split(X1, y1, test_size=test_size, shuffle=False)
    lassoC = lsn.LassoNetRegressor(verbose=verbose, hidden_dims=K, path_multiplier=(1+pm), M=M, patience=(100, 10), n_iters=(1000, 100))
    history = lassoC.path(Xt, yt, X_val=Xv, y_val=yv)

    if plot:
        res_k = np.zeros(len(history))
        res_val = np.zeros(len(history))
        res_l = np.zeros(len(history))

        for i, h in enumerate(history):
            res_k[i] = h.selected.sum()
            res_val[i] = h.val_loss
            res_l[i] = h.lambda_

        
        sns.lineplot(x=np.array(res_k), y=np.array(res_val), markers=True)
        plt.title("VALIDATION PERFORMANCE")
        plt.show()

        sns.lineplot(x=np.array(res_l), y=np.array(res_k), markers=True)
        plt.title("SELECTED VS LAMBDA")
        plt.show()

    backup_h = 100
    for h in history:
        if h.selected.sum() <= n_features:
            return h.selected.data.numpy() if h.selected.sum() > 0 else backup_h
        else:
            backup_h = h.selected.sum()


def return_LassoNet_mask(Xt, Xv, yt, yv, K=[10], pm=0.02, activation='relu', M=10, epochs=500, patience=5, print_lambda=False, print_path=False, plot=False, a=1e-3, nfeat=0, ppatience=10, pepochs=100):
    dense = return_MLP_skip_estimator(Xt, Xv, yt, yv, activation=activation, K=K, verbose=1, patience=patience, epochs=epochs)
    # dense.compile(optimizer=ks.optimizers.SGD(learning_rate=a, momentum=0.9), loss=ks.losses.MeanSquaredError())

    starting_lambda = estimate_starting_lambda(dense.get_layer('skip_layer').get_weights()[0], dense.get_layer('gw_layer').get_weights()[0], M, verbose=print_lambda, divide_result=4)

    res_k, res_theta, res_val, res_l = train_lasso_path(
        dense, starting_lambda, Xt, Xv, yt, yv, ks.optimizers.SGD(learning_rate=a, momentum=0.9), ks.losses.MeanSquaredError(), 
        train_until_k=nfeat, use_faster_fit=True, lr=a, M=M, pm=pm, max_epochs_per_lambda=pepochs, use_best_weights=True,
        patience=ppatience, verbose=print_path, use_faster_eval=False)

    # Plot accuracies at all points of the lasso path
    if plot:
        sns.lineplot(x=np.array(res_k), y=np.array(res_val), markers=True)
        plt.title("VALIDATION PERFORMANCE")
        plt.show()

        sns.lineplot(x=np.array(res_l), y=np.array(res_k), markers=True)
        plt.title("K VS LAMBDA")
        plt.show()

    # index_first_non_full = np.argwhere(np.array(res_k) < res_k[0]).ravel()[0]
    # final_theta = res_theta[index_first_non_full:][np.argmin(np.array(res_val)[index_first_non_full:])]
    if nfeat == 0:
        # index_first_non_full = np.argwhere(np.array(res_k) < res_k[0]*0.8).ravel()[0]
        # final_theta = res_theta[index_first_non_full:][np.argmin(np.array(res_val)[index_first_non_full:])]
        final_theta = res_theta[np.argmin(np.array(res_val))]
    else:
        final_theta = res_theta[-1]
    theta_mask = np.ravel(final_theta != 0)
    print(f"Selected {np.sum(theta_mask)} features.")

    return theta_mask


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

dlag_opt = [3]
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

        best_mse = np.inf
        best_K = [200, 150, 100, 50, 20]

        K_opt = [
            # [5],
            # [20],
            # [50],
            # [100],
            # [200],

            # [20, 5],
            # [50, 20],
            # [100, 50],
            # [200, 100],

            # [50, 20, 5],
            # [100, 50, 20],
            # [200, 100, 50],

            # [100, 50, 20, 5],
            # [200, 100, 50, 20],

            # [100, 50, 20, 10, 5],
            [200, 100, 50, 20, 5],
            # [300, 200, 100, 50, 20],
            # [400, 300, 200, 100, 50],
        ]

        Xt, Xv, yt, yv = ms.train_test_split(Xtrain, ytrain, test_size=120, shuffle=False)
        yv = y_pp.inverse_transform(yv.reshape(1, -1)).ravel()

        # for K in K_opt:
        #     mses = np.zeros(5)
        #     for i in range(len(mses)):
        #         predictor = return_MLP_skip_estimator(Xt, yt, verbose=0, K=K, test_size=60, activation='tanh', epochs=20_000, patience=25, drop=0)
        #         ypred = predictor.predict(Xv).ravel()
        #         ypred = y_pp.inverse_transform(ypred.reshape(1, -1)).ravel()
        #         mse = mt.mean_squared_error(yv, ypred)
        #         mses[i] = mse

        #     print(f"Finished experiment")
        #     print(f"K = {K}")
        #     print(f"MSE: {np.mean(mses):.3f}")
        #     print(f"MSE SDEV: {np.std(mses):.3f}")

        #     if np.mean(mses) < best_mse:
        #         best_K = K
        #         best_mse = np.mean(mses)

        np.random.seed(1234)
        tf.random.set_seed(1234)

        # # Robustness of final model
        # final_results = np.zeros(5)
        # for i in range(5):
        #     nn = return_MLP_skip_estimator(Xtrain, ytrain, verbose=1, K=best_K, activation='tanh', epochs=20_000, patience=50, drop=0, test_size=30)
        #     test_f = nn.predict(Xtest).ravel()
        #     test_f = y_pp.inverse_transform(test_f.reshape(1, -1)).ravel()
        #     experiment_mse = mt.mean_squared_error(ytest, test_f)
        #     print(f"FINAL MSE: {experiment_mse:.3f}")
        #     final_results[i] = experiment_mse
        
        # print(np.mean(final_results))
        # print(np.std(final_results))

        # final_mask = np.ravel(paper_lassonet_mask(Xtrain, ytrain.ravel(), K=tuple(best_K), verbose=2, n_features=int(0.5*Xtrain.shape[1]), pm=0.005, M=0.5, plot=True) != 0)
        Xtt, Xtv, ytt, ytv = ms.train_test_split(Xtrain, ytrain, test_size=30, shuffle=False)
        final_mask = np.ravel(return_LassoNet_mask(Xtt, Xtv, ytt, ytv, K=best_K, activation='tanh', M=5, pm=0.02, epochs=2000, 
                                                   patience=100, print_path=True, nfeat=0, plot=True, ppatience=10, pepochs=500) != 0)

        1 / 0
        Xtm = Xtrain[:,final_mask]
        Xtt = Xtest[:,final_mask]
        Xtf = Xvoortest[:,final_mask]

        Xtt, Xtv, ytt, ytv = ms.train_test_split(Xtrain, ytrain, test_size=30, shuffle=False)
        final_predictor = return_MLP_skip_estimator(Xtt, Xtv, ytt, ytv, verbose=1, K=best_K, activation='tanh', epochs=20_000, patience=50, drop=0)
        test_forecast = final_predictor.predict(Xtt).ravel()
        test_forecast = y_pp.inverse_transform(test_forecast.reshape(1, -1)).ravel()
        full_forecast = final_predictor.predict(Xtf).ravel()
        full_forecast = y_pp.inverse_transform(full_forecast.reshape(1, -1)).ravel()

        print("FINAL RESULTS")
        print(f"BEST K = {best_K}")
        print(f"MSE = {best_mse:.3f}")

        print(f"FINAL MSE: {mt.mean_squared_error(ytest, test_forecast):.3f}")
        # print(f"FINAL RETURN: {calc_investment_returns(test_forecast, ytest, allow_empty=False, use_thresholds=False, ytrain=ytrain)[0]:.3f}")
        print(f"Only mean MSE: {mt.mean_squared_error(ytest, np.full_like(ytest, np.mean(ytrain))):.3f}")
        
        np.savetxt(f'final_forecasts/SKIPX_{d_nlags}_{h_nlags}_K', np.array(best_K))
        np.savetxt(f'final_forecasts/SKIPX_{d_nlags}_{h_nlags}_MSE', np.array([best_mse]))
        np.save(f'final_forecasts/SKIPX_test_{d_nlags}_{h_nlags}', test_forecast.ravel())
        np.save(f'final_forecasts/SKIPX_full_{d_nlags}_{h_nlags}', full_forecast.ravel())