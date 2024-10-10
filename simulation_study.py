import numpy as np
import tensorflow as tf
import keras as ks
import sklearn.model_selection as ms
import sklearn.metrics as mt
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tsa.stattools as tsa

from network_definitions import return_MLP_skip_estimator, return_MLP_estimator
from lassonet_implementation import return_LassoNet_results, paper_lassonet_results, train_lasso_path, estimate_starting_lambda

def generate_nonlinear_dataset(N = 100, K = [10], features=5, remove_lags_skip = [], remove_lags_gw = []):
    series = np.ones(N)

    inp = ks.layers.Input(shape=(features,))

    skip = ks.layers.Dense(units=1, activation='linear', use_bias=False,
                           name='skip_layer', kernel_initializer="random_normal")(inp)
    gw = ks.layers.Dense(units=K[0], activation="relu", name='gw_layer', kernel_initializer="random_normal")(inp) 
    last_node = ks.layers.Dense(units=1)(gw)
    output = ks.layers.Add()([skip, last_node])

    dgp = ks.models.Model(inputs=inp, outputs=output)
    skip_weights = dgp.get_layer('skip_layer').get_weights()[0]
    gw_weights = dgp.get_layer('gw_layer').get_weights()[0]
    gw_bias = dgp.get_layer('gw_layer').get_weights()[1]

    for lag in remove_lags_skip:
        skip_weights[lag] = 0

    for lag in remove_lags_gw:
        gw_weights[lag] = np.zeros(K[0])
        gw_bias[lag] = 0

    new_skip_layer = dgp.get_layer('skip_layer').get_weights()
    new_skip_layer[0] = skip_weights.reshape((-1, 1))
    dgp.get_layer('skip_layer').set_weights(new_skip_layer)
    print(dgp.get_layer('skip_layer').get_weights())

    new_gw_layer = dgp.get_layer('gw_layer').get_weights()
    new_gw_layer[0] = gw_weights
    new_gw_layer[1] = gw_bias
    dgp.get_layer('gw_layer').set_weights(new_gw_layer)
    print(dgp.get_layer('gw_layer').get_weights())

    for t in range(features, N):
        X_input = np.array([series[t-i-1] for i in range(features)]).reshape(1, -1)
        #series[t] = dgp.predict(X_input, verbose=0) + np.random.normal()
        series[t] = dgp(X_input, training=False) + np.random.normal()

    return series

def main():
    # Experiment setup
    # [2, 3, 4, 5, 6, 7, 8, 9]
    NX = 1
    N = 1000
    include_xlags = 10
    remove_skip_lags = [2, 3, 4, 5, 6, 7, 8, 9]
    remove_gw_lags = [2, 3, 4, 5, 6, 7, 8, 9]
    K = [10]
    alpha = 0.01

    # Begin experiment
    all_features = set(range(include_xlags))
    LR_features = all_features - set(remove_skip_lags)
    NLR_features = all_features - set(remove_gw_lags)
    R_features = all_features - set(remove_skip_lags).intersection(set(remove_gw_lags))
    I_features = set(remove_skip_lags).intersection(set(remove_gw_lags))

    LR_pct = np.zeros(NX)
    NLR_pct = np.zeros(NX)
    R_pct = np.zeros(NX)
    I_pct = np.zeros(NX)

    for ex in range(NX):
        print(f"Starting experiment number {ex}")
        stationary = False
        while not stationary:
            data = generate_nonlinear_dataset(N=N, K=K, features=include_xlags, remove_lags_skip=remove_skip_lags, remove_lags_gw=remove_gw_lags)
            p = tsa.adfuller(data)[1]
            if p < 0.05:
                stationary = True
                print("Finished generating data.")
            else:
                print("Failed test: retrying")
        
        y = data[include_xlags:].reshape(-1, 1)
        Xlist = np.arange(1, len(y) + 1).reshape(-1, 1)
        for t_h in range(0, include_xlags):
            Xlist = np.concatenate(
                [
                    Xlist,
                    data[(include_xlags-1-t_h):(-1-t_h)].reshape(-1, 1),
                ], axis=1)
        X = Xlist[:, 1:]

        Xt, Xv, yt, yv = ms.train_test_split(X, y, test_size=0.1, shuffle=False)
        skip_predictor = return_MLP_skip_estimator(Xt, Xv, yt, yv, K=K, verbose=0, lr=alpha)

        starting_lambda = estimate_starting_lambda(skip_predictor.get_layer('skip_layer').get_weights()[0], skip_predictor.get_layer('gw_layer').get_weights()[0], 10, verbose=True, steps_back=3) / alpha
        lassonet_results = train_lasso_path(skip_predictor, starting_lambda, Xt, Xv, yt, yv, ks.optimizers.SGD(learning_rate=alpha, momentum=0.9), ks.losses.MeanSquaredError(), lr=alpha, verbose=True, min_improvement=0.995, patience=5, use_best_weights=True, use_faster_eval=False)

        k_array = np.array(lassonet_results[0])
        v_array = np.array(lassonet_results[2])

        best_theta = lassonet_results[1][np.argmin(v_array)]
        selected_features = set(list(np.argwhere(np.array(best_theta).ravel() != 0).ravel()))
        print(selected_features)

        unique_k = np.unique(k_array)
        lowest_values = [np.min(v_array[k_array == k]) for k in unique_k]
        sns.lineplot(x=unique_k, y=lowest_values)

        plt.show()
        # Add oos performance metric

        LR_pct[ex] = 1 - len(LR_features.difference(selected_features)) / len(LR_features)      if len(LR_features) > 0 else 1
        NLR_pct[ex] = 1 - len(NLR_features.difference(selected_features)) / len(NLR_features)   if len(NLR_features) > 0 else 1
        R_pct[ex] = 1 - len(R_features.difference(selected_features)) / len(R_features)         if len(R_features) > 0 else 1
        I_pct[ex] = len(I_features.difference(selected_features)) / len(I_features)             if len(I_features) > 0 else 1

    print(f"Correct LR: {np.mean(LR_pct)}")
    print(f"Correct NLR: {np.mean(NLR_pct)}")
    print(f"Correct R: {np.mean(R_pct)}")
    print(f"Correct I: {np.mean(I_pct)}")


if __name__ == '__main__':
    seed = 1235
    np.random.seed(seed)
    tf.random.set_seed(seed)
    ks.utils.set_random_seed(seed)
    main()