import pandas as pd
import numpy as np
import keras as ks
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as mt
import sklearn.linear_model as sklm

from full_mylasso import train_lasso_path, train_dense_model, estimate_starting_lambda
import lassonet as lsn

tf.random.set_seed(1234)
tf.get_logger().setLevel('ERROR')
np.random.seed(1234)
rng = np.random.RandomState(1234)


def calc_investment_returns(forecast, real, allow_empty=False, start_val=1, trad_cost=0.0015):
    value = start_val
    path = np.zeros(len(real))
    prev_pos = 1

    lb = np.mean(forecast) - np.std(forecast)
    ub = np.mean(forecast) + np.std(forecast)

    for t, (f, r) in enumerate(zip(forecast, real)):
        pos = prev_pos
        if f == 0:
            pos = -1
        elif f == 1:
            pos = 1

        if pos != prev_pos: value = value * (1 - trad_cost)
        prev_pos = pos

        value = value * (1 + pos * r/100)
        path[t] = value

    return (value / start_val - 1, path)

def return_lassoCV_estimor(X1, y1, cv=5, verbose=0, time_series_split=True, max_iter=2000):
    lm = sklm.LassoCV(cv=ms.TimeSeriesSplit(n_splits=cv) if time_series_split else cv, verbose=verbose, fit_intercept=True, max_iter=max_iter, n_jobs=-1)
    lm.fit(X1, y1)
    return lm


def return_lassonet_estimor(X1, y1, K=[10], verbose=0, cv=3):
    nn = lsn.LassoNetRegressorCV(cv=cv, hidden_dims=K, verbose=verbose)
    nn.fit(X1, y1)
    return nn


def plot_paper_lassonet(X1, y1, K=(10,), verbose=0, n_features=0, pm=1.02, M=10):
    Xt, Xv, yt, yv = ms.train_test_split(X1, y1, test_size=0.1, shuffle=False)
    lassoC = lsn.LassoNetRegressor(verbose=verbose, hidden_dims=K, path_multiplier=pm, M=M)
    history = lassoC.path(Xt, yt, X_val=Xv, y_val=yv)

    res_k = np.zeros(len(history))
    res_val = np.zeros(len(history))

    for i, h in enumerate(history):
        res_k[i] = h.selected.sum()
        res_val[i] = h.val_loss

    sns.lineplot(x=np.array(res_k), y=np.array(res_val), markers=True)
    plt.title("VALIDATION PERFORMANCE")
    plt.show()

    for h in history:
        if h.selected.sum() <= n_features:
            return h.selected.data.numpy()


def return_MLP_skip_estimator(X1, y1, K=[10], activation='relu', epochs=500, patience=30, verbose=0, drop=0, shuff=False):
    Xt, Xv, yt, yv = ms.train_test_split(X1, y1, test_size=0.1, shuffle=shuff)
    return train_dense_model(Xt, Xv, yt, yv, 2, ks.optimizers.Adam(1e-3), ks.losses.SparseCategoricalCrossentropy(), ['accuracy'], activation=activation, neurons=K, verbose=verbose, patience=patience, epochs=epochs, drop=drop)


def return_LassoNet_mask(X1, y1, K=[10], activation='relu', M=10, epochs=500, patience=5, print_lambda=False, print_path=False, plot=False, a=1e-3, nfeat=0):
    Xt, Xv, yt, yv = ms.train_test_split(X1, y1, test_size=0.1, shuffle=False)
    dense = train_dense_model(Xt, Xv, yt, yv, 1, ks.optimizers.Adamax(1e-3), ks.losses.MeanSquaredError(), ['mse'], activation=activation, neurons=K, verbose=1, patience=patience, epochs=epochs)
    dense.compile(optimizer=ks.optimizers.SGD(learning_rate=a, momentum=0.9), loss=ks.losses.MeanSquaredError(), metrics=['mse'])

    starting_lambda = estimate_starting_lambda(dense.get_layer('skip_layer').get_weights()[0], dense.get_layer('gw_layer').get_weights()[0], M, verbose=print_lambda, divide_result=10)

    res_k, res_theta, res_val, res_isa, res_r = train_lasso_path(dense, starting_lambda, Xt, Xv, yt, yv, ks.optimizers.SGD(learning_rate=a, momentum=0.9), ks.losses.MeanSquaredError(), 
                                                          train_until_k=nfeat, use_faster_fit=False, lr=a, M=M, pm=0.02, max_epochs_per_lambda=100, use_best_weights=True,
                                                          patience=10, verbose=print_path, return_train=False, use_faster_eval=False, early_val_stop=False, use_invest_results=False)

    # Plot accuracies at all points of the lasso path
    if plot:
        sns.lineplot(x=np.array(res_k), y=np.array(res_r), markers=True)
        plt.title("VALIDATION RETURNS")
        plt.show()
        sns.lineplot(x=np.array(res_k), y=np.array(res_val), markers=True)
        plt.title("VALIDATION PERFORMANCE")
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


def return_MLP_estimator(X1, y1, K=[10], activation='relu', patience=30, epochs=500, verbose=0, metrics=['mse'], drop=0):
    Xt, Xv, yt, yv = ms.train_test_split(X1, y1, test_size=0.1, shuffle=False)
    inp = ks.layers.Input(shape=(Xt.shape[1],))
    dp = ks.layers.Dropout(drop)(inp)
    gw = ks.layers.Dense(units=K[0], activation=activation)(dp)

    if len(K) > 1:
        for K in K[1:]:
            gw = ks.layers.Dense(units=K, activation=activation)(gw)   

    output = ks.layers.Dense(units=1)(gw)

    # Implement early stopping
    early_stop = ks.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=patience,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0,
    )

    # Initial dense training
    nn = ks.models.Model(inputs=inp, outputs=output)
    nn.compile(optimizer=ks.optimizers.Adamax(1e-3), loss=ks.losses.MeanSquaredError(), metrics=metrics)
    nn.fit(Xt, yt, validation_data=(Xv, yv), epochs=epochs, callbacks=[early_stop], verbose=verbose)

    return nn


def return_MLP_drp(X1, y1, K=[10], activation='relu', epochs=500, verbose=0, optimizer=ks.optimizers.Adam(1e-3), loss_func=ks.losses.MeanSquaredError(), metrics=['mse'], drop=0):
    inp = ks.layers.Input(shape=(X1.shape[1],))
    dp = ks.layers.Dropout(drop)(inp)
    gw = ks.layers.Dense(units=K[0], activation=activation)(dp)

    if len(K) > 1:
        for K in K[1:]:
            gw = ks.layers.Dense(units=K, activation=activation)(gw)   

    output = ks.layers.Dense(units=1)(gw)

    # Initial dense training
    nn = ks.models.Model(inputs=inp, outputs=output)
    nn.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
    nn.fit(X1, y1, epochs=epochs, verbose=verbose)

    return nn

###############################################################################################################################################################################################

USE_OLD_DATA = False
extra = '_old' if USE_OLD_DATA else ''
day_df = pd.read_csv(f'agg_btc_day{extra}.csv', parse_dates=['date', 'ddate'])
hour_df = pd.read_csv(f'agg_btc_hour{extra}.csv', parse_dates=['date', 'ddate'])

d_nlags = 2
h_nlags = 1
freq = 12
bound_lag = max(d_nlags, ((h_nlags-1)//freq + 1))

# raw_returns = day_df.close.pct_change(1)[1:].to_numpy()
open_returns =  ((day_df.open[1:].to_numpy() - day_df.open[:-1].to_numpy()) / day_df.open[:-1].to_numpy()) * 100
high_returns =  ((day_df.high[1:].to_numpy() - day_df.high[:-1].to_numpy()) / day_df.high[:-1].to_numpy()) * 100
low_returns =   ((day_df.low[1:].to_numpy() - day_df.low[:-1].to_numpy()) / day_df.low[:-1].to_numpy()) * 100
close_returns = ((day_df.close[1:].to_numpy() - day_df.close[:-1].to_numpy()) / day_df.close[:-1].to_numpy()) * 100
vol_returns =   ((day_df.volume[1:].to_numpy() - day_df.volume[:-1].to_numpy()) / day_df.volume[:-1].to_numpy()) * 100
volNot_returns =((day_df.volumeNotional[1:].to_numpy() - day_df.volumeNotional[:-1].to_numpy()) / day_df.volumeNotional[:-1].to_numpy()) * 100
trades_returns =((day_df.tradesDone[1:].to_numpy() - day_df.tradesDone[:-1].to_numpy()) / day_df.tradesDone[:-1].to_numpy()) * 100

open_h_returns =  ((hour_df.open[1:].to_numpy() - hour_df.open[:-1].to_numpy()) / hour_df.open[:-1].to_numpy()) * 100
high_h_returns =  ((hour_df.high[1:].to_numpy() - hour_df.high[:-1].to_numpy()) / hour_df.high[:-1].to_numpy()) * 100
low_h_returns =   ((hour_df.low[1:].to_numpy() - hour_df.low[:-1].to_numpy()) / hour_df.low[:-1].to_numpy()) * 100
close_h_returns = ((hour_df.close[1:].to_numpy() - hour_df.close[:-1].to_numpy()) / hour_df.close[:-1].to_numpy()) * 100
vol_h_returns =   ((hour_df.volume[1:].to_numpy() - hour_df.volume[:-1].to_numpy()) / hour_df.volume[:-1].to_numpy()) * 100
volNot_h_returns =((hour_df.volumeNotional[1:].to_numpy() - hour_df.volumeNotional[:-1].to_numpy()) / hour_df.volumeNotional[:-1].to_numpy()) * 100
trades_h_returns =((hour_df.tradesDone[1:].to_numpy() - hour_df.tradesDone[:-1].to_numpy()) / hour_df.tradesDone[:-1].to_numpy()) * 100

y_raw = close_returns[bound_lag:].reshape(-1, 1) >= 0
Xlist = np.arange(1, len(y_raw) + 1).reshape(-1, 1)
if h_nlags > 0:
    for t_h in range(0, h_nlags):
        Xlist = np.concatenate(
            [
                Xlist,
                # open_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                # high_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                low_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                close_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                # vol_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                # volNot_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                # trades_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
            ], axis=1)
if d_nlags > 0:
    for t in range(0, d_nlags):
        Xlist = np.concatenate(
            [
                Xlist,
                # open_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                # high_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                low_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                close_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                # vol_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                # volNot_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                # trades_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
            ], axis=1)

Xlist = Xlist[:, 1:]
X_pp = pp.MinMaxScaler().fit(Xlist)
y_pp = pp.LabelEncoder().fit(y_raw.ravel())
Xvoortest = X_pp.transform(Xlist)
yvoortest = y_pp.transform(y_raw.ravel())

Xtrain, Xtest, ytrain, ytest = ms.train_test_split(Xvoortest, yvoortest, test_size=365, shuffle=False)
print("Data has been fully transformed and split")


# LASSO
# predictor = return_lassoCV_estimor(Xtrain, ytrain.ravel(), cv=5, max_iter=100_000)
# lasso_mask = np.ravel(predictor.coef_ != 0)
# n_selected = int(np.sum(lasso_mask))
# print(f"LASSO selected {n_selected} features")
# # print(f"LASSO selected {int(np.sum(lasso_mask[0:7*d_nlags]))} features from daily data")
# # print(f"LASSO selected {int(np.sum(lasso_mask[7*d_nlags : 7*d_nlags + 7*h_nlags]))} features from houry data")
# # print(f"LASSO selected {int(np.sum(lasso_mask[7*d_nlags + 7*h_nlags :]))} features from minty data")
# Xtrain = Xtrain[:,lasso_mask]
# Xtest = Xtest[:,lasso_mask]

# # # Paper LassoNet
# ln_mask = np.ravel(plot_paper_lassonet(Xtrain, ytrain.ravel(), K=(40, 20, 10), verbose=2, n_features=10, pm=1.01) != 0)
# n_selected = int(np.sum(ln_mask))
# print(f"LassoNet selected {n_selected} features")
# Xtrain = Xtrain[:,ln_mask]
# Xtest = Xtest[:,ln_mask]

# # My LassoNetw
# mask = return_LassoNet_mask(Xtrain, ytrain, K=[20, 10], activation='relu', epochs=20_000, patience=25, print_lambda=True, print_path=True, plot=True, nfeat=int(Xtrain.shape[1] * 0.15))
# print(f"LASSONET selected {int(np.sum(mask))} features")
# # print(f"LASSONET selected {int(np.sum(mask[0 : 7*d_nlags]))} features from daily data")
# # print(f"LASSONET selected {int(np.sum(mask[7*d_nlags : 7*d_nlags + 7*h_nlags]))} features from houry data")
# Xtrain = Xtrain[:,mask]
# Xtest = Xtest[:,mask]

n_repeats = 1
results = np.zeros(n_repeats)

for i in range(n_repeats):
    # predictor = return_lassoCV_estimor(Xtrain, ytrain.ravel(), cv=5, max_iter=5_000)
    predictor = return_MLP_skip_estimator(Xtrain, ytrain, verbose=1, K=[20, 5], activation='tanh', epochs=20_000, patience=20, drop=0, shuff=True)
    # lnr = lsn.LassoNetRegressorCV(cv=ms.TimeSeriesSplit(n_splits=5), hidden_dims=(50, 20), verbose=2, path_multiplier=1.01)
    # predictor = lnr.fit(Xtrain, ytrain)
    

    ypred = np.argmax(predictor.predict(Xtest), axis=1)
    print(f"Finished experiment {i+1}")
    print(f"Predicted true {np.sum(ypred) / len(ypred)}")
    print(f"ACC: {mt.accuracy_score(ytest, ypred):.4f}")
    print(calc_investment_returns(ypred, close_returns[-len(ytest):].ravel())[0])
    results[i] = mt.accuracy_score(ytest, ypred)

print(f"Ran {n_repeats} experiments:")
print(f"Average ACC: {np.mean(results):.4f}")
print(f"STD of ACC: {np.std(results):.4f}")
maj_acc = mt.mean_squared_error(ytest, np.full_like(ytest, True))
print(f"Only true ACC: {max(maj_acc, 1-maj_acc):.4f}")
# ytrain = y_pp.inverse_transform(ytrain.reshape(-1, 1)).ravel()
# print(f"Only mean MSE: {mt.mean_squared_error(ytest, np.full_like(ytest, np.mean(ytrain))):.6f}")
