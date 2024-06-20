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


def return_lassoCV_estimor(X1, y1, cv=5, verbose=0, time_series_split=True, max_iter=2000):
    lm = sklm.LassoCV(cv=ms.TimeSeriesSplit(n_splits=cv) if time_series_split else cv, verbose=verbose, fit_intercept=True, max_iter=max_iter, n_jobs=-1)
    lm.fit(X1, y1)
    return lm


def return_lassonet_estimor(X1, y1, K=[10], verbose=0, cv=3):
    nn = lsn.LassoNetRegressorCV(cv=cv, hidden_dims=K, verbose=verbose)
    nn.fit(X1, y1)
    return nn


def return_MLP_skip_estimator(X1, y1, K=[10], activation='relu', epochs=500, patience=30, verbose=0, drop=0, shuff=False):
    Xt, Xv, yt, yv = ms.train_test_split(X1, y1, test_size=0.125, shuffle=shuff)
    return train_dense_model(Xt, Xv, yt, yv, 1, ks.optimizers.Adamax(1e-3), ks.losses.MeanSquaredError(), ['mse'], activation=activation, neurons=K, verbose=verbose, patience=patience, epochs=epochs, drop=drop)


def return_LassoNet_mask(X1, y1, K=[10], activation='relu', M=10, epochs=500, patience=5, print_lambda=False, print_path=False, plot=False, a=1e-3, nfeat=0):
    Xt, Xv, yt, yv = ms.train_test_split(X1, y1, test_size=0.125, shuffle=False)
    dense = train_dense_model(Xt, Xv, yt, yv, 1, ks.optimizers.Adamax(1e-3), ks.losses.MeanSquaredError(), ['mse'], activation=activation, neurons=K, verbose=1, patience=patience, epochs=epochs)
    dense.compile(optimizer=ks.optimizers.SGD(learning_rate=a, momentum=0.9), loss=ks.losses.MeanSquaredError(), metrics=['mse'])

    starting_lambda = estimate_starting_lambda(dense.get_layer('skip_layer').get_weights()[0], dense.get_layer('gw_layer').get_weights()[0], M, verbose=print_lambda)

    res_k, res_theta, res_val, res_isa = train_lasso_path(dense, starting_lambda, Xt, Xv, yt, yv, ks.optimizers.SGD(learning_rate=a, momentum=0.9), ks.losses.MeanSquaredError(), 
                                                          train_until_k=nfeat, use_faster_fit=True, lr=a, M=M, pm=0.005, max_epochs_per_lambda=1000, use_best_weights=True,
                                                          patience=10, verbose=print_path, return_train=False, use_faster_eval=False)

    # Plot accuracies at all points of the lasso path
    if plot:
        # sns.lineplot(x=np.array(res_k), y=np.array(res_isa), markers=True)
        # plt.title("IN SAMPLE PERFORMANCE")
        # plt.show()
        sns.lineplot(x=np.array(res_k), y=np.array(res_val), markers=True)
        plt.title("VALIDATION PERFORMANCE")
        plt.show()

    # index_first_non_full = np.argwhere(np.array(res_k) < res_k[0]).ravel()[0]
    # final_theta = res_theta[index_first_non_full:][np.argmin(np.array(res_val)[index_first_non_full:])]
    # if nfeat != 0: final_theta = res_theta[-1]
    final_theta = res_theta[-1]
    theta_mask = np.ravel(final_theta != 0)
    print(f"Selected {np.sum(theta_mask)} features.")

    return theta_mask


def return_LassoNet_estimator(X1, y1, K=[10], activation='relu', M=10, epochs=500, patience=5, print_lambda=False, print_path=False, plot=False, a=1e-3, nfeat=0, drop=0):
    Xt, Xv, yt, yv = ms.train_test_split(X1, y1, test_size=0.125, shuffle=False, random_state=rng)
    dense = train_dense_model(Xt, Xv, yt, yv, 1, ks.optimizers.Adamax(1e-3), ks.losses.MeanSquaredError(), ['mse'], activation=activation, neurons=K, verbose=1, patience=patience, epochs=epochs)
    dense.compile(optimizer=ks.optimizers.SGD(learning_rate=a, momentum=0.9), loss=ks.losses.MeanSquaredError(), metrics=['mse'])

    starting_lambda = estimate_starting_lambda(dense.get_layer('skip_layer').get_weights()[0], dense.get_layer('gw_layer').get_weights()[0], M, verbose=print_lambda)

    res_k, res_theta, res_val, res_isa = train_lasso_path(dense, starting_lambda, Xt, Xv, yt, yv, ks.optimizers.SGD(learning_rate=a, momentum=0.9), ks.losses.MeanSquaredError(), 
                                                          train_until_k=nfeat, use_faster_fit=True, lr=a, M=M, pm=0.02, max_epochs_per_lambda=1000, use_best_weights=True,
                                                          patience=25, verbose=print_path, return_train=False, use_faster_eval=False)

    # Plot accuracies at all points of the lasso path
    if plot:
        # sns.lineplot(x=np.array(res_k), y=np.array(res_isa), markers=True)
        # plt.title("IN SAMPLE PERFORMANCE")
        # plt.show()
        sns.lineplot(x=np.array(res_k), y=np.array(res_val), markers=True)
        plt.title("VALIDATION PERFORMANCE")
        plt.show()

    final_theta = res_theta[np.argmin(res_val[res_k < res_k[0]])]
    if nfeat != 0: final_theta = res_theta[-1]
    theta_mask = np.ravel(final_theta != 0)
    print(f"Selected {np.sum(theta_mask)} features.")

    Xtf = Xt[:,theta_mask]
    Xvf = Xv[:,theta_mask]

    return_model = train_dense_model(Xtf, Xvf, yt, yv, 1, ks.optimizers.Adam(1e-3), ks.losses.MeanSquaredError(), ['mse'], activation=activation, neurons=K, verbose=1, patience=patience, epochs=epochs, drop=drop)

    return (return_model, theta_mask)


def return_MLP_estimator(X1, y1, K=[10], activation='relu', patience=30, epochs=500, verbose=0, optimizer=ks.optimizers.Adam(1e-3), loss_func=ks.losses.MeanSquaredError(), metrics=['mse'], drop=0):
    Xt, Xv, yt, yv = ms.train_test_split(X1, y1, test_size=0.125, shuffle=False, random_state=rng)
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
    nn.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
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

day_df = pd.read_csv('agg_btc_day.csv', parse_dates=['date', 'ddate'])
hour_df = pd.read_csv('agg_btc_hour.csv', parse_dates=['date', 'ddate'])
min_df = pd.read_csv('agg_btc_min.csv', parse_dates=['date', 'ddate'])

d_nlags = 6
h_nlags = 4
m_nlags = 2

y_raw = day_df.close.pct_change(1)[d_nlags:].to_numpy().reshape(-1, 1)

Xlist = np.zeros((len(y_raw), 7*(d_nlags + h_nlags + m_nlags)))
for i, day in enumerate(day_df.ddate[d_nlags:]):
    appending = []
    if d_nlags > 0: appending.append(day_df[day_df.ddate <= day].tail(d_nlags).iloc[:, 1:-2].to_numpy().ravel())
    if h_nlags > 0: appending.append(hour_df[hour_df.ddate <= day].tail(h_nlags).iloc[:, 1:-2].to_numpy().ravel())
    if m_nlags > 0: appending.append(min_df[min_df.ddate <= day].tail(m_nlags).iloc[:, 2:-1].to_numpy().ravel())
    
    Xlist[i] = np.concatenate(appending)

if True:
    X_pp = pp.MinMaxScaler().fit(Xlist)
    y_pp = pp.MinMaxScaler().fit(y_raw)

    X = X_pp.transform(Xlist)
    y = y_pp.transform(y_raw)
else:
    X = Xlist
    y = y_raw

Xtrain, Xtest, ytrain, ytest = ms.train_test_split(X, y, test_size=0.2, shuffle=False, random_state=rng)
print("Data has been fully transformed and split")

# # LASSO
# predictor = return_lassoCV_estimor(Xtrain, ytrain, cv=5, max_iter=1_000)
# lasso_mask = np.ravel(predictor.coef_ != 0)
# n_selected = int(np.sum(lasso_mask))
# print(f"LASSO selected {n_selected} features")
# print(f"LASSO selected {int(np.sum(lasso_mask[0:7*d_nlags]))} features from daily data")
# print(f"LASSO selected {int(np.sum(lasso_mask[7*d_nlags : 7*d_nlags + 7*h_nlags]))} features from houry data")
# print(f"LASSO selected {int(np.sum(lasso_mask[7*d_nlags + 7*h_nlags :]))} features from minty data")
# Xtrain = Xtrain[:,lasso_mask]
# Xtest = Xtest[:,lasso_mask]

# My LassoNet
mask = return_LassoNet_mask(Xtrain, ytrain, K=[20], activation='tanh', epochs=10_000, patience=100, print_lambda=True, print_path=True, plot=True, nfeat=0)
print(f"LASSONET selected {int(np.sum(mask))} features")
print(f"LASSONET selected {int(np.sum(mask[0 : 7*d_nlags]))} features from daily data")
print(f"LASSONET selected {int(np.sum(mask[7*d_nlags : 7*d_nlags + 7*h_nlags]))} features from houry data")
print(f"LASSONET selected {int(np.sum(mask[7*d_nlags + 7*h_nlags :]))} features from minty data")
Xtrain = Xtrain[:,mask]
Xtest = Xtest[:,mask]

# # Regular MLPs
n_repeats = 10
results = np.zeros(n_repeats)

for i in range(n_repeats):
    # predictor = return_lassoCV_estimor(Xtrain, ytrain, cv=5, max_iter=5_000)
    predictor = return_MLP_skip_estimator(Xtrain, ytrain, verbose=0, K=[20], activation='tanh', epochs=10000, patience=100, drop=0, shuff=False)

    ypred = predictor.predict(Xtest).ravel()
    print(f"Finished experiment {i+1}")
    print(f"MSE: {mt.mean_squared_error(ytest, ypred):.6f}")
    results[i] = mt.mean_squared_error(ytest, ypred)

    # x_axis = list(range(len(ytest.ravel())))
    # sns.lineplot(x=x_axis, y=ytest.ravel(), color='black')
    # sns.lineplot(x=x_axis, y=ypred.ravel(), color='red')
    # sns.lineplot(x=x_axis, y=ytest.ravel() - ypred.ravel(), color='blue')
    # plt.show()

print(f"Ran {n_repeats} experiments:")
print(f"Average MSE: {1000*np.mean(results):.2f}")
print(f"STD of MSE: {1000*np.std(results):.2f}")
print(f"Only mean MSE: {1000*mt.mean_squared_error(ytest, np.full_like(ytest, np.mean(ytrain))):.2f}")