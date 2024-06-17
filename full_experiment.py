import pandas as pd
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as mt
import numpy as np
import statsmodels as sts
import matplotlib.pyplot as plt
import seaborn as sns
import keras as ks
import tensorflow as tf
import itertools as itertools
import time as time

from full_mylasso import train_lasso_path, train_dense_model, hier_prox, estimate_starting_lambda


def return_MLP_estimator(X, y, K=[10], activation='relu', patience=30, epochs=500, verbose=0, optimizer=ks.optimizers.legacy.Adam(1e-3), loss_func=ks.losses.MeanSquaredError(), metrics=['mse']):
    Xt, Xv, yt, yv = ms.train_test_split(X, y, test_size=0.1, shuffle=False)
    inp = ks.layers.Input(shape=(Xt.shape[1],))
    gw = ks.layers.Dense(units=K[0], activation=activation)(inp)

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


def return_MLP_skip_estimator(X, y, K=[10], activation='relu', epochs=500, patience=30):
    Xt, Xv, yt, yv = ms.train_test_split(X, y, test_size=0.1, shuffle=False)
    return train_dense_model(Xt, Xv, yt, yv, 1, ks.optimizers.Adam(1e-3), ks.losses.MeanSquaredError(), ['mse'], activation=activation, neurons=K, verbose=0, patience=patience, epochs=epochs)


def return_LassoNet_estimator(X, y, K=[10], activation='relu', M=10):
    Xt, Xv, yt, yv = ms.train_test_split(X, y, test_size=0.1, shuffle=False)
    dense = train_dense_model(Xt, Xv, yt, yv, 1, ks.optimizers.Adam(1e-3), ks.losses.MeanSquaredError(), ['mse'], activation=activation, neurons=K, verbose=0, patience=30, epochs=500)


def main():
    # Read in the data
    full_data = pd.read_csv('btcusd_full.csv')
    dates = full_data.date
    full_data = full_data.drop(['date'], axis=1)
    full_data['target'] = np.log(full_data.close / full_data.open)
    print("Data has been fully loaded")
    
    n_lags = 1

    std_data = pp.StandardScaler().fit_transform(full_data)
    X = np.concatenate([std_data[lag:-(n_lags-lag),:-1] for lag in range(n_lags)], axis=1)
    y = std_data[n_lags:,-1]
    print(X.shape)
    print(y.shape)
    Xtrain, Xtest, ytrain, ytest = ms.train_test_split(X, y, test_size=0.2, shuffle=True)
    print("Data has been fully transformed and split")

    # Apply cross-validation for finding the best settings
    kf = ms.KFold(shuffle=True)
    T = ytrain.shape[0]
    pct_of_data_for_cv = 5 / 100
    indices_to_use = list(range(0, T, int(1/pct_of_data_for_cv)))
    print(f"Using {len(indices_to_use)} rows for cross-validation.")
    experiment_results = {}

    layer_sizes = [[2], [4], [8], [16], [32]]
    activations = ['relu', 'tanh', 'hard_sigmoid']

    # # For testing purposes
    # layer_sizes = [[1]]
    # activations = ['relu']

    best_set = (0, 'relu')
    best_mse_till_now = np.inf

    for n_ex, (K, act) in enumerate(itertools.product(layer_sizes, activations)):
        print(f"===== - ===== Testing {K} and {act}")
        mse_results = np.zeros(5)

        for i, (train_index, test_index) in enumerate(kf.split(Xtrain[indices_to_use], ytrain[indices_to_use])):
            start = time.perf_counter()
            XFtrain, yftrain = Xtrain[train_index], np.array(ytrain[train_index])
            XFtest, yftest = np.array(Xtrain[test_index]), np.array(ytrain[test_index])

            estimator = return_MLP_estimator(XFtrain, yftrain, K=K, activation=act)
            ypred = estimator.predict(XFtest, verbose='0')
            test_mse = mt.mean_squared_error(yftest, ypred)
            mse_results[i] = test_mse
            print(f"Training fold {i+1}/5 took {time.perf_counter() - start:.2f}s \t- mse: {test_mse:.4f}")

        mean_mse = np.mean(mse_results)
        print(f"MSE of {K} and {act}: {mean_mse}")
        experiment_results[n_ex] = f"K = {K}, \tact = {act}, \tmean mse = {mean_mse:.4f}"
        if mean_mse <= best_mse_till_now:
            best_mse_till_now = mean_mse
            best_set = (K, act)

    print()
    print("Cross validation results:")
    for ex, res in experiment_results.items():
        print(res)
    K, act = best_set

    return
    final_model = return_MLP_estimator(Xtrain, ytrain, K=K, activation=act, epochs=1000, patience=100)

    ypred = final_model.predict(Xtest, verbose='0')
    final_mse = mt.mean_squared_error(ytest, ypred)
    print(f"Final MSE using {K} and {act}: {final_mse}")

if __name__ == '__main__':
    tf.random.set_seed(1234)
    tf.get_logger().setLevel('ERROR')
    np.random.seed(1234)
    main()