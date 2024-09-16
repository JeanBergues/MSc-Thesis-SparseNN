import numpy as np
import tensorflow as tf
import keras as ks
import sklearn.model_selection as ms
import sklearn.metrics as mt
import matplotlib.pyplot as plt
import seaborn as sns

from network_definitions import return_MLP_skip_estimator, return_MLP_estimator

def generate_nonlinear_dataset(N = 100, f = lambda x: (0.7*np.abs(x)) / (np.abs(x) + 2)):
    series = np.zeros(N)
    for i in range(1, N):
        series[i] = f(series[i-1]) + np.random.standard_normal()
    return series

def main():
    NX = 10
    N = 10000
    K = [10]

    non_skip_results = np.zeros(NX)
    skip_results = np.zeros(NX)

    for ex in range(NX):
        print(f"Starting experiment number {ex}")
        data = generate_nonlinear_dataset(N=N)
        X, y = data[1:].reshape(-1, 1), data[:-1]
        Xtrainfull, Xtest, ytrainfull, ytest = ms.train_test_split(X, y, test_size=0.2, shuffle=False)
        Xtrain, Xval, ytrain, yval = ms.train_test_split(Xtrainfull, ytrainfull, test_size=0.1, shuffle=False)

        non_skip_predictor = return_MLP_estimator(Xtrain, Xval, ytrain, yval, K=K, verbose=0)
        skip_predictor = return_MLP_skip_estimator(Xtrain, Xval, ytrain, yval, K=K, verbose=0)

        non_skip_prediction = non_skip_predictor.predict(Xtest)
        skip_prediction = skip_predictor.predict(Xtest)

        non_skip_results[ex] = mt.mean_squared_error(ytest, non_skip_prediction)
        skip_results[ex] = mt.mean_squared_error(ytest, skip_prediction)

    # print(non_skip_results)
    # print(skip_results)

    print(np.mean(non_skip_results))
    print(np.std(non_skip_results))
    print(np.mean(skip_results))
    print(np.std(skip_results))


if __name__ == '__main__':
    np.random.seed(1234)
    tf.random.set_seed(1234)
    ks.utils.set_random_seed(1234)
    main()