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


def return_MLP_estimator(X, y, K=[10], activation='relu'):
    Xt, Xv, yt, yv = ms.train_test_split(X, y, test_size=0.1, shuffle=False)
    return train_dense_model(Xt, Xv, yt, yv, 1, ks.optimizers.Adam(1e-3), ks.losses.MeanSquaredError(), ['mse'], activation=activation, neurons=K, verbose=0, patience=10, epochs=100)


def main():
    # Read in the data
    full_data = pd.read_csv('btcusd_full.csv', nrows=10000)
    dates = full_data.date
    full_data = full_data.drop(['date'], axis=1)
    full_data['target'] = np.log(full_data.close / full_data.open)
    print("Data has been fully loaded")
    
    std_data = pp.StandardScaler().fit_transform(full_data)
    X = std_data[:-1,:-1]
    y = std_data[1:,-1]
    Xtrain, Xtest, ytrain, ytest = ms.train_test_split(X, y, test_size=0.2, shuffle=False)
    print("Data has been fully transformed and split")

    # Apply cross-validation for finding the best settings
    kf = ms.KFold(shuffle=False)
    experiment_results = {}
    layer_sizes = [2, 4, 8, 16]
    activations = ['relu']

    for n_ex, (K, act) in enumerate(itertools.product(layer_sizes, activations)):
        print(f"===== Testing {K} and {act}")
        mse_results = np.zeros(5)

        for i, (train_index, test_index) in enumerate(kf.split(Xtrain, ytrain)):
            print(f"Performing fold {i+1}/5")
            start = time.perf_counter()
            XFtrain, yftrain = Xtrain[train_index], np.array(ytrain[train_index])
            XFtest, yftest = np.array(Xtrain[test_index]), np.array(ytrain[test_index])

            estimator = return_MLP_estimator(XFtrain, yftrain)
            ypred = estimator.predict(XFtest, verbose='0')
            test_mse = mt.mean_squared_error(yftest, ypred)
            mse_results[i] = test_mse
            print(f"Training fold {i+1}/5 took {time.perf_counter() - start:.2f}s - mse: {test_mse:.4f}")

        print(f"MSE of {K} and {act}: {np.mean(mse_results)}")
        experiment_results[n_ex] = f"K = {K}, act = {act}, mean mse = {np.mean(mse_results):.4f}"

    print(experiment_results)

if __name__ == '__main__':
    tf.random.set_seed(1234)
    tf.get_logger().setLevel('ERROR')
    np.random.seed(1234)
    main()