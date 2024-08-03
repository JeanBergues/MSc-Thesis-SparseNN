import pandas as pd
import numpy as np
import tensorflow as tf
import keras as ks
import sklearn.model_selection as ms
import sklearn.metrics as mt

from network_definitions import return_MLP_estimator, return_MLP_skip_estimator
from data_loader import load_AR_data, load_data_with_X, scale_data
from investing_simulation import plot_returns, calc_investment_returns
from lassonet_implementation import paper_lassonet_mask, return_LassoNet_mask, estimate_starting_lambda, train_lasso_path

###############################################################################################################################################################################################

def main():
    day_df = pd.read_csv(f'pct_btc_day.csv')
    hour_df = pd.read_csv(f'pct_btc_hour.csv')

    # Define the experiment parameters
    dlag_opt = [1, 2]
    hlag_opt = [0, 1, 2, 3]

    dlag_opt = [2]
    hlag_opt = [0]

    K_opt = [
        [5],
        [10],
        [20],
        [30],
        [50]
    ]
    
    USE_X = True
    USE_SKIP = True
    VALIDATE_LAYER = False
    DEFAULT_K = [50]

    activation      = 'relu'
    n_cv_reps       = 5
    cv_patience     = 100
    n_fm_reps       = 5
    fm_patience     = 100
    learning_rate   = 0.01
    es_tolerance    = 0
    dropout         = 0
    use_l1_penalty  = False
    M = 10

    EXPERIMENT_NAME = "final_LN_forecasts/"
    EXPERIMENT_NAME += "LN_SNN_" if USE_SKIP else "LN_NN_"
    EXPERIMENT_NAME += "X_" if USE_X else ""

    LOAD_BACKUP = True

    # Begin the training
    for d_nlags in dlag_opt:
        for h_nlags in hlag_opt:
            np.random.seed(1234)
            tf.random.set_seed(1234)
            ks.utils.set_random_seed(1234)
            EXPERIMENT_NAME += f"{d_nlags}_{h_nlags}"

            if USE_X:
                X_raw, y_raw = load_data_with_X(day_df, hour_df, d_nlags, h_nlags)
            else:
                X_raw, y_raw = load_AR_data(day_df, hour_df, d_nlags, h_nlags)
            X_scaled, y_scaled, X_pp, y_pp = scale_data(X_raw, y_raw)

            Xtrainfull, Xtestfull, ytrain, ytest = ms.train_test_split(X_scaled, y_scaled, test_size=365, shuffle=False)
            print("Data has been fully transformed and split")

            np.random.seed(1234)
            tf.random.set_seed(1234)
            ks.utils.set_random_seed(1234)
            tscv = ms.TimeSeriesSplit(n_cv_reps)
            Xtt, Xtv, ytt, ytv = ms.train_test_split(Xtrainfull, ytrain, test_size=30, shuffle=False)
            full_dense = return_MLP_skip_estimator(Xtt, Xtv, ytt, ytv, verbose=0, K=DEFAULT_K, activation=activation, epochs=20_000, patience=cv_patience, drop=dropout, use_L1=use_l1_penalty, es_tol=es_tolerance, lr=learning_rate)
            cv_starting_lambda = estimate_starting_lambda(full_dense.get_layer('skip_layer').get_weights()[0], full_dense.get_layer('gw_layer').get_weights()[0], M, verbose=True, steps_back=3) / learning_rate

            if not LOAD_BACKUP:
                val_mse_paths = []
                val_lambda_paths = []

                for i, (train_index, test_index) in enumerate(tscv.split(Xtrainfull)):
                    Xt, Xv, yt, yv = Xtrainfull[train_index], Xtrainfull[test_index], ytrain[train_index], ytrain[test_index]
                    Xtt, Xtv, ytt, ytv = ms.train_test_split(Xt, yt, test_size=30, shuffle=False)
                    #yval = y_pp.inverse_transform(yv.reshape(1, -1)).ravel()

                    dense = return_MLP_skip_estimator(Xtt, Xtv, ytt, ytv, verbose=0, K=DEFAULT_K, activation=activation, epochs=20_000, patience=cv_patience, drop=dropout, use_L1=use_l1_penalty, es_tol=es_tolerance, lr=learning_rate)
                    res_k, res_theta, res_val, res_l, res_oos, final_net = train_lasso_path(
                        dense, cv_starting_lambda, Xt, Xtv, yt, ytv, ks.optimizers.SGD(learning_rate=learning_rate, momentum=0.9), ks.losses.MeanSquaredError(), 
                        train_until_k=0, use_faster_fit=True, lr=learning_rate, M=M, pm=0.02, max_epochs_per_lambda=100, use_best_weights=True,
                        patience=5, verbose=True, use_faster_eval=False, regressor=True, X_test=Xv, y_test=yv)

                    val_mse_paths.append(res_oos)
                    val_lambda_paths.append(res_l)
                    np.save(f'{EXPERIMENT_NAME}_FOLD{i}', np.array(res_oos).ravel())

                longest_path_length = max(map(len, val_mse_paths))
                longest_lambda_path = max(val_lambda_paths, key=len)
                cv_mses = np.array([path+[path[-1]]*(longest_path_length-len(path)) for path in val_mse_paths])
                np.save(f'{EXPERIMENT_NAME}_ALLFOLDS', cv_mses)
                np.save(f'{EXPERIMENT_NAME}_CV_LAMBDA', np.array(longest_lambda_path))
            else:
                cv_mses = np.load(f'{EXPERIMENT_NAME}_ALLFOLDS.npy')
                longest_lambda_path = [cv_starting_lambda * 1.02 ** p for p in range(cv_mses.shape[1])]

            mean_cv_mse = np.mean(cv_mses, axis=0)
            cv_selected_lambda = longest_lambda_path[np.argmin(mean_cv_mse)]
            
            Xt, Xv, yt, yv = ms.train_test_split(Xtrainfull, ytrain, test_size=30, shuffle=False)
            # mask = paper_lassonet_mask(Xt, Xv, yt, yv, K=tuple(DEFAULT_K), verbose=2, pm=0.02, M=20, patiences=(100, 10), max_iters=(10000, 100), l_start='auto', n_features=0, use_custom_optimizer=True)
            res_k, res_theta, res_val, res_l, res_oos, final_net = train_lasso_path(
                    full_dense, cv_starting_lambda, Xt, Xv, yt, yv, ks.optimizers.SGD(learning_rate=learning_rate, momentum=0.9), ks.losses.MeanSquaredError(), 
                    train_until_k=0, use_faster_fit=True, lr=learning_rate, M=M, pm=0.02, max_epochs_per_lambda=100, use_best_weights=True,
                    patience=10, verbose=True, use_faster_eval=False, regressor=True, X_test=Xv, y_test=yv, max_lambda=cv_selected_lambda)
            
            np.save(f'{EXPERIMENT_NAME}_OOS', np.array(res_oos).ravel())
            np.save(f'{EXPERIMENT_NAME}_K', np.array(res_k).ravel())

            ytest = y_pp.inverse_transform(ytest.reshape(1, -1)).ravel()

            test_f = final_net.predict(Xtestfull).ravel()
            train_f = final_net.predict(Xtrainfull).ravel()
            test_f = y_pp.inverse_transform(test_f.reshape(1, -1)).ravel()
            train_f = y_pp.inverse_transform(train_f.reshape(1, -1)).ravel()

            experiment_mse = mt.mean_squared_error(ytest, test_f)
            np.save(f'{EXPERIMENT_NAME}_LN_FORECAST', test_f.ravel())
            np.save(f'{EXPERIMENT_NAME}_LN_TRAIN_FORECAST', train_f.ravel())
            print(f"LASSONET MODEL ACHIEVED MSE: {experiment_mse}")

if __name__ == '__main__':
    main()