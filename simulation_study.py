import numpy as np
import tensorflow as tf
import keras as ks
import sklearn.model_selection as ms
import sklearn.metrics as mt
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tsa.stattools as tsa
from time import perf_counter_ns

from network_definitions import return_MLP_skip_estimator, return_MLP_estimator
from lassonet_implementation import return_LassoNet_results, paper_lassonet_results, train_lasso_path, estimate_starting_lambda

def generate_nonlinear_dataset(N = 100, K = [10], features=5, remove_lags_skip = [], remove_lags_gw = []):
    series = np.ones(N) * 0.001

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

    new_gw_layer = dgp.get_layer('gw_layer').get_weights()
    new_gw_layer[0] = gw_weights
    new_gw_layer[1] = gw_bias
    dgp.get_layer('gw_layer').set_weights(new_gw_layer)

    for t in range(features, N):
        X_input = np.array([series[t-i-1] for i in range(features)]).reshape(1, -1)
        # series[t] = dgp.predict(X_input, verbose=0) + 0.5*np.random.normal()
        series[t] = dgp(X_input, training=False) + 0.5*np.random.normal()

    return series

def main():
    # Experiment setup
    # [2, 3, 4, 5, 6, 7, 8, 9]
    TEST_NAME = 'simulation_results/T1500_R2I8'
    NX = 100
    N = 1500
    include_xlags = 10
    remove_skip_lags = [2, 3, 4, 5, 6, 7, 8, 9]
    remove_gw_lags = [2, 3, 4, 5, 6, 7, 8, 9]
    K = [10]
    alpha = 0.001

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
    dense_mses = np.zeros(NX)
    mlp_mses = np.zeros(NX)
    L1_mses = np.zeros(NX)
    sparse_mses = np.zeros(NX)
    pct_improvements = np.zeros(NX)
    pct_mlp_improvements = np.zeros(NX)
    pct_L1_improvements = np.zeros(NX)
    n_features_chosen = np.zeros(NX)
    n_times_improved = np.zeros(NX)
    n_times_mlp_improved = np.zeros(NX)
    n_times_L1_improved = np.zeros(NX)

    for ex in range(NX):
        print(f"Starting experiment number {ex}")
        start_time = perf_counter_ns()
        stationary = False
        while not stationary:
            data = generate_nonlinear_dataset(N=N, K=K, features=include_xlags, remove_lags_skip=remove_skip_lags, remove_lags_gw=remove_gw_lags)
            # sns.lineplot(data)
            # plt.show()
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

        Xtrain, Xtest, ytrain, ytest = ms.train_test_split(X, y, test_size= N // 10, shuffle=False)
        Xt, Xv, yt, yv = ms.train_test_split(Xtrain, ytrain, test_size=N // 10, shuffle=False)

        mlp_predictor = return_MLP_estimator(Xt, Xv, yt, yv, K=K, verbose=0, lr=alpha)
        mlp_mse = mt.mean_squared_error(ytest, mlp_predictor.predict(Xtest).ravel())
        L1_skip_predictor = return_MLP_skip_estimator(Xt, Xv, yt, yv, K=K, verbose=0, lr=alpha, use_L1=True)
        L1_mse = mt.mean_squared_error(ytest, L1_skip_predictor.predict(Xtest).ravel())
        skip_predictor = return_MLP_skip_estimator(Xt, Xv, yt, yv, K=K, verbose=0, lr=alpha)
        dense_mse = mt.mean_squared_error(ytest, skip_predictor.predict(Xtest).ravel())

        starting_lambda = estimate_starting_lambda(skip_predictor.get_layer('skip_layer').get_weights()[0], skip_predictor.get_layer('gw_layer').get_weights()[0], 10, verbose=False, steps_back=2) / alpha
        lassonet_results = train_lasso_path(skip_predictor, starting_lambda, Xt, Xv, yt, yv, ks.optimizers.SGD(learning_rate=alpha, momentum=0.9), ks.losses.MeanSquaredError(), lr=alpha, verbose=True, min_improvement=0.99, patience=5, max_epochs_per_lambda=100,
                                            use_best_weights=True, use_faster_eval=False, X_test=Xtest, y_test=ytest, pm=0.01)

        # k_array = np.array(lassonet_results[0])
        v_array = np.array(lassonet_results[2])
        t_array = np.array(lassonet_results[4])

        final_model_index = np.argmin(v_array)
        best_theta = lassonet_results[1][final_model_index]
        selected_features = set(list(np.argwhere(np.array(best_theta).ravel() != 0).ravel()))
        print(selected_features)
        end_time = perf_counter_ns()

        # unique_k = np.unique(k_array)
        # lowest_values = [np.min(v_array[k_array == k]) for k in unique_k]
        # lowest_test_values = [np.min(t_array[k_array == k]) for k in unique_k]

        # sns.lineplot(x=unique_k, y=lowest_values)
        # sns.lineplot(x=unique_k, y=lowest_test_values)
        # plt.show()

        LR_pct[ex]          = 1 - len(LR_features.difference(selected_features)) / len(LR_features)      if len(LR_features) > 0 else 1
        NLR_pct[ex]         = 1 - len(NLR_features.difference(selected_features)) / len(NLR_features)   if len(NLR_features) > 0 else 1
        R_pct[ex]           = 1 - len(R_features.difference(selected_features)) / len(R_features)         if len(R_features) > 0 else 1
        I_pct[ex]           = len(I_features.difference(selected_features)) / len(I_features)             if len(I_features) > 0 else 1
        dense_mses[ex]      = dense_mse
        mlp_mses[ex]        = mlp_mse
        L1_mses[ex]         = L1_mse
        sparse_mses[ex]     = t_array[final_model_index]
        pct_improvements[ex]= (t_array[final_model_index] / dense_mse - 1) * 100
        pct_mlp_improvements[ex]= (t_array[final_model_index] / mlp_mse - 1) * 100
        pct_L1_improvements[ex]= (t_array[final_model_index] / L1_mse - 1) * 100
        n_features_chosen[ex] = len(selected_features)
        n_times_improved[ex] = 1 if dense_mse > t_array[final_model_index] else 0
        n_times_mlp_improved[ex] = 1 if mlp_mse > t_array[final_model_index] else 0
        n_times_L1_improved[ex] = 1 if L1_mse > t_array[final_model_index] else 0

        print(f"Finished experiment number {ex}")
        print(f"MMSE = {mlp_mse:.4f}, LMSE = {L1_mse:.4f}, DMSE = {dense_mse:.4f}, \tFMSE = {t_array[final_model_index]:.4f}, \tPCT = {(t_array[final_model_index] / dense_mse - 1) * 100:.4f}, \tNF = {n_features_chosen[ex]}")
        print(f"LR = {LR_pct[ex]:.3f}, \tNLR = {NLR_pct[ex]:.3f}, \tR = {R_pct[ex]:.3f}, \tI = {I_pct[ex]:.3f}, \tMPCT = {(t_array[final_model_index] / mlp_mse - 1) * 100:.4f}, \tLPCT = {(t_array[final_model_index] / L1_mse - 1) * 100:.4f}")
        print("Mean results:")
        print(f"MMSE = {np.mean(mlp_mses):.4f}, LMSE = {np.mean(L1_mses):.4f}, DMSE = {np.mean(dense_mses):.4f}, \tFMSE = {np.mean(sparse_mses):.4f}, \tPCT = {np.mean(pct_improvements):.4f}, \tNF = {np.mean(n_features_chosen)}")
        print(f"LR = {np.mean(LR_pct):.3f}, \tNLR = {np.mean(NLR_pct):.3f}, \tR = {np.mean(R_pct):.3f}, \tI = {np.mean(I_pct):.3f}, \tNI = {np.mean(n_times_improved):.3f}, \tNMI = {np.mean(n_times_mlp_improved):.3f}, \tNLI = {np.mean(n_times_L1_improved):.3f}")
        print(f"Experiment took {(end_time - start_time) // 1e9} seconds, \tMPCT = {np.mean(pct_mlp_improvements):.4f}, \tLPCT = {np.mean(pct_L1_improvements):.4f}")
        print()
        np.save(TEST_NAME, np.vstack((LR_pct, NLR_pct, R_pct, I_pct, dense_mses, mlp_mses, L1_mses, sparse_mses, pct_improvements, pct_mlp_improvements, pct_L1_improvements, n_features_chosen, n_times_improved, n_times_mlp_improved, n_times_L1_improved)))
    

if __name__ == '__main__':
    seed = 1234
    np.random.seed(seed)
    tf.random.set_seed(seed)
    ks.utils.set_random_seed(seed)
    main()