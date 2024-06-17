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


def return_MLP_estimator(X, y, K=[10], activation='relu', patience=30, epochs=500, verbose=0, optimizer=ks.optimizers.Adam(1e-3), loss_func=ks.losses.MeanSquaredError(), metrics=['mse']):
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


def return_MLP_skip_estimator(X, y, K=[10], activation='relu', epochs=500, patience=30, verbose=0):
    Xt, Xv, yt, yv = ms.train_test_split(X, y, test_size=0.1, shuffle=False)
    return train_dense_model(Xt, Xv, yt, yv, 1, ks.optimizers.Adam(1e-3), ks.losses.MeanSquaredError(), ['mse'], activation=activation, neurons=K, verbose=verbose, patience=patience, epochs=epochs)


def return_LassoNet_estimator(X, y, K=[10], activation='relu', M=10, epochs=500, patience=5, print_lambda=False, print_path=False, plot=False, a=1e-3):
    Xt, Xv, yt, yv = ms.train_test_split(X, y, test_size=0.1, shuffle=True)
    dense = train_dense_model(Xt, Xv, yt, yv, 1, ks.optimizers.Adam(1e-3), ks.losses.MeanSquaredError(), ['mse'], activation=activation, neurons=K, verbose=1, patience=patience, epochs=epochs)
    dense.compile(optimizer=ks.optimizers.SGD(learning_rate=a, momentum=0.9), loss=ks.losses.MeanSquaredError(), metrics=['mse'])

    starting_lambda = estimate_starting_lambda(dense.get_layer('skip_layer').get_weights()[0], dense.get_layer('gw_layer').get_weights()[0], M, verbose=print_lambda)

    res_k, res_theta, res_val, res_isa = train_lasso_path(dense, starting_lambda, Xt, Xv, yt, yv, ks.optimizers.SGD(learning_rate=a, momentum=0.9), ks.losses.MeanSquaredError(), 
                                                          train_until_k=0, use_faster_fit=False, lr=a, M=M, pm=0.02, max_epochs_per_lambda=100, use_best_weights=True,
                                                          patience=10, verbose=print_path, return_train=False, use_faster_eval=False)

    # Plot accuracies at all points of the lasso path
    if plot:
        # sns.lineplot(x=np.array(res_k), y=np.array(res_isa), markers=True)
        # plt.title("IN SAMPLE PERFORMANCE")
        # plt.show()
        sns.lineplot(x=np.array(res_k), y=np.array(res_val), markers=True)
        plt.title("VALIDATION PERFORMANCE")
        plt.show()

    final_theta = res_theta[np.argmin(res_val)]
    theta_mask = np.ravel(final_theta != 0)
    print(f"Selected {np.sum(theta_mask)} features.")

    Xtf = Xt[:,theta_mask]
    Xvf = Xv[:,theta_mask]

    return_model = train_dense_model(Xtf, Xvf, yt, yv, 1, ks.optimizers.Adam(1e-3), ks.losses.MeanSquaredError(), ['mse'], activation=activation, neurons=K, verbose=1, patience=patience, epochs=epochs)

    return (return_model, theta_mask)


def main():
    APPLY_CV            = False
    TRAIN_FULL_MODEL    = True

    # Read in the data
    full_data = pd.read_csv('FD_btc_data_hourly.csv')
    dates = full_data.date
    full_data = full_data.drop(['date'], axis=1)
    print("Data has been fully loaded")
    
    n_lags = 1

    std_data = pp.StandardScaler().fit_transform(full_data)
    X = np.concatenate([std_data[lag:-(n_lags-lag),:] for lag in range(n_lags)], axis=1)
    y = std_data[n_lags:,-2]
    print(X.shape)
    print(y.shape)
    Xtrain, Xtest, ytrain, ytest = ms.train_test_split(X, y, test_size=0.2, shuffle=False)
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

    best_set = ([30], 'relu')
    best_mse_till_now = np.inf

    if APPLY_CV:
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

    if TRAIN_FULL_MODEL:
        K, act = best_set
        final_model, mask = return_LassoNet_estimator(Xtrain, ytrain, K=K, activation=act, epochs=500, patience=30, print_lambda=True, print_path=True, plot=True)
        Xtest = Xtest[:,mask]
        # final_model = return_MLP_skip_estimator(Xtrain, ytrain, K=K, activation=act, epochs=500, patience=30, verbose=1)

        ypred = final_model.predict(Xtest, verbose='0')
        final_mse = mt.mean_squared_error(ytest, ypred)
        print(f"Final MSE using {K} and {act} Vanilla MLP: 2.1329")
        print(f"Final MSE using {K} and {act} Skipped MLP: 2.1168")
        print(f"Final MSE using {K} and {act} LassoNetted: 2.0961")

if __name__ == '__main__':
    tf.random.set_seed(1234)
    tf.get_logger().setLevel('ERROR')
    np.random.seed(1234)
    main()