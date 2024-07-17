import pandas as pd
import numpy as np
import tensorflow as tf
import keras as ks
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as mt

from network_definitions import return_MLP_estimator, return_MLP_skip_estimator
from data_loader import load_AR_data, load_data_with_X, scale_data

###############################################################################################################################################################################################

def main():
    day_df = pd.read_csv(f'pct_btc_day.csv')
    hour_df = pd.read_csv(f'pct_btc_hour.csv')

    dlag_opt = [1]
    hlag_opt = [1]

    K_opt = [
        [5],
        [10],
        [20],
        [30],
        [40],
        [50],
    ]
    
    USE_X = False
    USE_SKIP = False
    VALIDATE_LAYER = False
    DEFAULT_K = [5]

    activation      = 'tanh'
    n_cv_reps       = 5
    cv_patience     = 100
    n_fm_reps       = 5
    fm_patience     = 100
    learning_rate   = 0.1
    es_tolerance    = 0
    dropout         = 0
    use_l1_penalty  = False

    # Begin the training
    for d_nlags in dlag_opt:
        for h_nlags in hlag_opt:
            np.random.seed(1234)
            tf.random.set_seed(1234)
            ks.utils.set_random_seed(1234)
            EXPERIMENT_NAME = f"final_forecasts/LR1_NN_{d_nlags}_{h_nlags}"

            if USE_X:
                X_raw, y_raw = load_data_with_X(day_df, hour_df, d_nlags, h_nlags)
            else:
                X_raw, y_raw = load_AR_data(day_df, hour_df, d_nlags, h_nlags)
            X_scaled, y_scaled, X_pp, y_pp = scale_data(X_raw, y_raw)

            Xtrain, Xtest, ytrain, ytest = ms.train_test_split(X_scaled, y_scaled, test_size=365, shuffle=False)
            print("Data has been fully transformed and split")

            ytest = y_pp.inverse_transform(ytest.reshape(1, -1)).ravel()

            best_val_mse = np.inf
            best_val_mse_sd = 0
            best_K = DEFAULT_K

            # Select layer size using validation set
            if VALIDATE_LAYER:
                Xt, Xv, yt, yv = ms.train_test_split(Xtrain, ytrain, test_size=120, shuffle=False)
                yval = y_pp.inverse_transform(yv.reshape(1, -1)).ravel()

                for K in K_opt:
                    np.random.seed(1234)
                    tf.random.set_seed(1234)
                    ks.utils.set_random_seed(1234)

                    mses = np.zeros(n_cv_reps)
                    for i in range(len(mses)):
                        if USE_SKIP:
                            predictor = return_MLP_estimator(Xt, Xv, yt, yv, verbose=0, K=K, activation=activation, epochs=20_000, patience=cv_patience, drop=dropout, use_L1=use_l1_penalty, es_tol=es_tolerance, lr=learning_rate)
                        else:
                            predictor = return_MLP_skip_estimator(Xt, Xv, yt, yv, verbose=0, K=K, activation=activation, epochs=20_000, patience=cv_patience, drop=dropout, use_L1=use_l1_penalty, es_tol=es_tolerance, lr=learning_rate)
                        ypred = predictor.predict(Xv).ravel()
                        ypred = y_pp.inverse_transform(ypred.reshape(1, -1)).ravel()
                        mse = mt.mean_squared_error(yval, ypred)
                        mses[i] = mse

                    print(f"Finished experiment")
                    print(f"K = {K}")
                    print(f"MSE: {np.mean(mses):.3f}")
                    print(f"MSE SDEV: {np.std(mses):.3f}")

                    if np.mean(mses) < best_val_mse:
                        best_K = K
                        best_val_mse = np.mean(mses)
                        best_val_mse_sd = np.std(mses)

                print("VALIDATION RESULTS")
                print(f"BEST K = {best_K}")
                print(f"MSE = {best_val_mse:.3f}")
                print(f"SD = {best_val_mse_sd:.3f}")
                np.savetxt(f'{EXPERIMENT_NAME}_VAL_K', np.array(best_K))
                np.savetxt(f'{EXPERIMENT_NAME}_VAL_STATS', np.array([best_val_mse, best_val_mse_sd]))

            np.random.seed(1234)
            tf.random.set_seed(1234)
            ks.utils.set_random_seed(1234)

            # Select final model based on small validation set
            Xt, Xv, yt, yv = ms.train_test_split(Xtrain, ytrain, test_size=30, shuffle=False)
            yval = y_pp.inverse_transform(yv.reshape(1, -1)).ravel()
            yvaltrain = y_pp.inverse_transform(ytrain.reshape(1, -1)).ravel()

            best_final_val_mse = np.inf
            best_final_mse = np.inf
            best_test_mse = np.inf

            # Robustness of final model
            final_results = np.zeros(n_fm_reps)
            for i in range(n_fm_reps):
                if USE_SKIP:
                    nn = return_MLP_estimator(Xt, Xv, yt, yv, verbose=0, K=best_K, activation=activation, epochs=20_000, patience=fm_patience, drop=dropout, use_L1=use_l1_penalty, es_tol=es_tolerance, lr=learning_rate)
                else:
                    nn = return_MLP_skip_estimator(Xt, Xv, yt, yv, verbose=0, K=best_K, activation=activation, epochs=20_000, patience=fm_patience, drop=dropout, use_L1=use_l1_penalty, es_tol=es_tolerance, lr=learning_rate)

                test_f = nn.predict(Xtest).ravel()
                test_f = y_pp.inverse_transform(test_f.reshape(1, -1)).ravel()
                experiment_mse = mt.mean_squared_error(ytest, test_f)
                final_results[i] = experiment_mse

                val_f = nn.predict(Xv).ravel()
                val_f = y_pp.inverse_transform(val_f.reshape(1, -1)).ravel()
                val_mse = mt.mean_squared_error(yval, val_f)

                train_f = nn.predict(Xtrain).ravel()
                train_f = y_pp.inverse_transform(train_f.reshape(1, -1)).ravel()
                train_mse = mt.mean_squared_error(yvaltrain, train_f)

                print(f"TEST MSE: {experiment_mse:.3f}")
                print(f"VAL MSE: {val_mse:.3f}")
                print(f"TRAIN MSE: {train_mse:.3f}")
                if val_mse < best_final_val_mse:
                    best_final_val_mse = val_mse
                    best_final_mse = experiment_mse
                    test_forecast = test_f
                if experiment_mse < best_test_mse:
                    best_test_mse = experiment_mse
                    best_test_forecast = test_f
            
            print("FINAL RESULTS")
            print(f"AVERAGE TEST MSE = {np.mean(final_results):.3f}")
            print(f"AVERAGE TEST SD = {np.std(final_results):.3f}")
            print(f"BEST SELECTED MSE = {mt.mean_squared_error(ytest, test_forecast):.3f}")
            print(f"BEST TEST MSE = {mt.mean_squared_error(ytest, best_test_forecast):.3f}")
            print(f"Only mean MSE = {mt.mean_squared_error(ytest, np.full_like(ytest, np.mean(ytrain))):.3f}")
            
            np.savetxt(f'{EXPERIMENT_NAME}_TEST_STATS', np.array([np.mean(final_results), np.std(final_results), best_final_mse, best_final_val_mse, best_test_mse]))
            np.save(f'{EXPERIMENT_NAME}_FORECAST', test_forecast.ravel())
            np.save(f'{EXPERIMENT_NAME}_TEST_FORECAST', test_forecast.ravel())

if __name__ == '__main__':
    main()