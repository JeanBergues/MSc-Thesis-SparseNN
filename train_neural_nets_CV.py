import pandas as pd
import numpy as np
import tensorflow as tf
import keras as ks
import sklearn.model_selection as ms
import sklearn.metrics as mt

from network_definitions import return_MLP_estimator, return_MLP_skip_estimator
from data_loader import load_AR_data, load_data_with_X, scale_data, load_data_with_XLOG
from investing_simulation import plot_returns, calc_investment_returns

###############################################################################################################################################################################################

def main(using_X):
    USE_LOG = False
    day_df = pd.read_csv(f'com_btc_day.csv') if USE_LOG else pd.read_csv(f'pct_btc_day.csv')
    hour_df = pd.read_csv(f'com_btc_hour.csv') if USE_LOG else pd.read_csv(f'pct_btc_hour.csv')

    # Define the experiment parameters

    lag_opt = [(1, 0), (2, 0), (1, 24), (2, 48)]

    K_opt = [
        [10],
        [50],
        [100],
        [10, 5],
        [50, 10],
        [100, 20]
    ]
    
    USE_X = using_X
    USE_SKIP = True
    VALIDATE_LAYER = True
    MASK = None
    DEFAULT_K = []

    activation      = 'tanh'
    n_cv_reps       = 5
    cv_patience     = 100
    n_fm_reps       = 10
    fm_patience     = 100
    learning_rate   = 0.01
    es_tolerance    = 0
    dropout         = 0
    use_l1_penalty  = True
    n_days_es       = 30

    BASE_EXPERIMENT_NAME = "final_forecasts/TEST_"
    BASE_EXPERIMENT_NAME += "SNN_" if USE_SKIP else "NN_"
    BASE_EXPERIMENT_NAME += "X_" if USE_X else ""
    BASE_EXPERIMENT_NAME += "LOG_" if USE_LOG else ""
    BASE_EXPERIMENT_NAME += f"D{dropout:.2f}_" if dropout > 0 else ""
    BASE_EXPERIMENT_NAME += f"MASKED_" if MASK is not None else ""

    PLOT_FINAL_FORECASTS = False

    # Begin the training
    for d_nlags, h_nlags in lag_opt:
        np.random.seed(1234)
        tf.random.set_seed(1234)
        ks.utils.set_random_seed(1234)
        EXPERIMENT_NAME = BASE_EXPERIMENT_NAME + f"{d_nlags}_{h_nlags}"
        print(f"STARTING {EXPERIMENT_NAME}")

        if USE_X:
            if USE_LOG:
                X_raw, y_raw = load_data_with_XLOG(day_df, hour_df, d_nlags, h_nlags)
            else:
                X_raw, y_raw = load_data_with_X(day_df, hour_df, d_nlags, h_nlags)
        else:
            X_raw, y_raw = load_AR_data(day_df, hour_df, d_nlags, h_nlags)
        X_scaled, y_scaled, X_pp, y_pp = scale_data(X_raw, y_raw)

        Xtrain, Xtest, ytrain, ytest = ms.train_test_split(X_scaled, y_scaled, test_size=365, shuffle=False)
        if MASK is not None:
            Xtrain = Xtrain[:,MASK]
            Xtest = Xtest[:,MASK]
        print("Data has been fully transformed and split")

        ytest = y_pp.inverse_transform(ytest.reshape(1, -1)).ravel()

        best_val_mse = np.inf
        best_val_mse_sd = 0
        best_K = DEFAULT_K

        # Select layer size using validation set
        if VALIDATE_LAYER:
            for K in K_opt:
                np.random.seed(1234)
                tf.random.set_seed(1234)
                ks.utils.set_random_seed(1234)
                tscv = ms.TimeSeriesSplit(n_cv_reps)

                mses = np.zeros(n_cv_reps)
                for i, (train_index, test_index) in enumerate(tscv.split(Xtrain)):
                    Xt, Xv, yt, yv = Xtrain[train_index], Xtrain[test_index], ytrain[train_index], ytrain[test_index]
                    Xtt, Xtv, ytt, ytv = ms.train_test_split(Xt, yt, test_size=n_days_es, shuffle=False)
                    yval = y_pp.inverse_transform(yv.reshape(1, -1)).ravel()

                    if not USE_SKIP:
                        predictor = return_MLP_estimator(Xtt, Xtv, ytt, ytv, verbose=0, K=K, activation=activation, epochs=20_000, patience=cv_patience, drop=dropout, use_L1=use_l1_penalty, es_tol=es_tolerance, lr=learning_rate)
                    else:
                        predictor = return_MLP_skip_estimator(Xtt, Xtv, ytt, ytv, verbose=0, K=K, activation=activation, epochs=20_000, patience=cv_patience, drop=dropout, use_L1=use_l1_penalty, es_tol=es_tolerance, lr=learning_rate, skip_bias=True)
                    ypred = predictor.predict(Xv).ravel()
                    ypred = y_pp.inverse_transform(ypred.reshape(1, -1)).ravel()
                    mse = mt.mean_squared_error(yval, ypred)
                    mses[i] = mse

                print(f"Finished experiment")
                print(f"K = {K}")
                print(f"MSE: {np.mean(mses):.3f}")
                # print(f"MSE SDEV: {np.std(mses):.3f}")

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
        Xt, Xv, yt, yv = ms.train_test_split(Xtrain, ytrain, test_size=n_days_es, shuffle=False)
        yval = y_pp.inverse_transform(yv.reshape(1, -1)).ravel()
        yvaltrain = y_pp.inverse_transform(ytrain.reshape(1, -1)).ravel()

        best_final_val_mse = np.inf
        best_final_mse = np.inf
        best_test_mse = np.inf
        best_test_return = -np.inf

        # Robustness of final model
        final_results = np.zeros(n_fm_reps)
        for i in range(n_fm_reps):
            if not USE_SKIP:
                nn = return_MLP_estimator(Xt, Xv, yt, yv, verbose=0, K=best_K, activation=activation, epochs=20_000, patience=fm_patience, drop=dropout, use_L1=use_l1_penalty, es_tol=es_tolerance, lr=learning_rate)
            else:
                nn = return_MLP_skip_estimator(Xt, Xv, yt, yv, verbose=0, K=best_K, activation=activation, epochs=20_000, patience=fm_patience, drop=dropout, use_L1=use_l1_penalty, es_tol=es_tolerance, lr=learning_rate)

            print(nn.get_layer('skip_layer').get_weights()[0])
            print(nn.get_layer('gw_layer').get_weights()[0])
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

            f_return = calc_investment_returns(val_f, yval, yvaltrain, allow_empty=True, trad_cost=0, use_thresholds=False)[0]

            print(f"TEST MSE: {experiment_mse:.3f}")
            print(f"VAL MSE: {val_mse:.3f}")
            print(f"TRAIN MSE: {train_mse:.3f}")
            print(f"RETURN: {f_return:.3f}")

            if train_mse < best_final_val_mse:
                best_final_val_mse = train_mse
                best_final_mse = experiment_mse
                test_forecast = test_f
                train_forecast = train_f
            if experiment_mse < best_test_mse:
                best_test_mse = experiment_mse
                best_test_forecast = test_f
            if f_return > best_test_return:
                best_test_return = f_return
                best_return_forecast = test_f

        final_returns = calc_investment_returns(best_return_forecast, ytest, yvaltrain, allow_empty=True, trad_cost=0, use_thresholds=False)[0]
        val_returns = calc_investment_returns(test_forecast, ytest, yvaltrain, allow_empty=True, trad_cost=0, use_thresholds=False)[0]
        print("")
        print("FINAL RESULTS")
        print(f"AVERAGE TEST MSE = {np.mean(final_results):.3f}")
        print(f"AVERAGE TEST SD = {np.std(final_results):.3f}")
        print(f"VAL SELECTED MSE = {mt.mean_squared_error(ytest, test_forecast):.3f}")
        print(f"BEST TEST MSE = {mt.mean_squared_error(ytest, best_test_forecast):.3f}")
        print(f"RETURN OF BEST RETURNING MODEL = {final_returns:.3f}")
        print(f"MSE OF BEST RETURNING MODEL = {mt.mean_squared_error(ytest, best_return_forecast):.3f}")
        print(f"Only mean MSE = {mt.mean_squared_error(ytest, np.full_like(ytest, np.mean(yvaltrain))):.3f}")
        
        np.savetxt(f'{EXPERIMENT_NAME}_TEST_STATS', np.array([np.mean(final_results), np.std(final_results), best_final_mse, best_final_val_mse, best_test_mse, val_returns, final_returns]))
        np.save(f'{EXPERIMENT_NAME}_FORECAST', test_forecast.ravel())
        np.save(f'{EXPERIMENT_NAME}_TRAIN_FORECAST', train_forecast.ravel())
        # np.save(f'{EXPERIMENT_NAME}_TEST_FORECAST', best_test_forecast.ravel())
        # np.save(f'{EXPERIMENT_NAME}_RETURN_FORECAST', best_return_forecast.ravel())

        if PLOT_FINAL_FORECASTS:
            plot_returns(ytest, yvaltrain, forecasts=[test_forecast, best_test_forecast, best_return_forecast], names=["TEST", "VAL", "RETURN"], strat=False)
            plot_returns(ytest, yvaltrain, forecasts=[test_forecast, best_test_forecast, best_return_forecast], names=["TEST", "VAL", "RETURN"], strat=True)

if __name__ == '__main__':
    main(False)
    main(True)