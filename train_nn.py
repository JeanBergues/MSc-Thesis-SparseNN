import pandas as pd
import numpy as np
import tensorflow as tf
import keras as ks
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as mt

np.random.seed(1234)
tf.random.set_seed(1234)
ks.utils.set_random_seed(1234)


def calc_investment_returns(forecast, real, ytrain, allow_empty=False, start_val=1, trad_cost=0.001, use_thresholds=True):
    value = start_val
    path = np.zeros(len(real))
    prev_pos = 1
    mean_f = 0

    if use_thresholds:
        last_seen = list(ytrain.ravel()[-14:])

    for t, (f, r) in enumerate(zip(forecast, real)):
        pos = prev_pos
        if use_thresholds:
            seen = np.array(last_seen[-7:])
            lb = np.mean(seen) - np.std(seen)
            ub = np.mean(seen) + np.std(seen)
            last_seen.append(r[0])

            if f < lb:
                pos = -1
            elif f > ub:
                pos = 1
            else:
                pos = 0 if allow_empty else prev_pos
        else:
            if f < mean_f:
                pos = -1
            elif f > mean_f:
                pos = 1
            else:
                pos = 0 if allow_empty else prev_pos

        if pos != prev_pos: value = value * (1 - trad_cost)
        prev_pos = pos

        value = value * (1 + pos * r[0]/100)
        path[t] = value

    return (value / start_val - 1, path)


def return_MLP_skip_estimator(Xt, Xv, yt, yv, k, K=[10], activation='relu', epochs=500, patience=30, verbose=0):
    inp = ks.layers.Input(shape=(k,))
    # skip = ks.layers.Dense(units=1, activation='linear', use_bias=True, name='skip_layer')(inp)
    skip = ks.layers.Dense(units=1, activation='linear', use_bias=False, kernel_regularizer=ks.regularizers.L1(), name='skip_layer')(inp)
    gw = ks.layers.Dense(units=K[0], activation=activation, name='gw_layer')(inp)
    if len(K) > 1:
        for k in K[1:]:
            gw = ks.layers.Dense(units=k, activation=activation)(gw)   

    merge = ks.layers.Concatenate()([skip, gw])
    output = ks.layers.Dense(units=1)(merge)

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
    nn.compile(optimizer=ks.optimizers.Adam(1e-3), loss=ks.losses.MeanSquaredError())
    nn.fit(Xt, yt, validation_data=(Xv, yv), epochs=epochs, callbacks=[early_stop], verbose=verbose)

    return nn


###############################################################################################################################################################################################

def main():
    day_df = pd.read_csv(f'pct_btc_day.csv')
    hour_df = pd.read_csv(f'pct_btc_hour.csv')

    freq = 24

    # raw_returns = day_df.close.pct_change(1)[1:].to_numpy()
    open_returns =  day_df.open.to_numpy()
    high_returns =  day_df.high.to_numpy()
    low_returns =   day_df.low.to_numpy()
    close_returns = day_df.close.to_numpy()
    vol_returns =   day_df.volume.to_numpy()
    volNot_returns =day_df.volumeNotional.to_numpy()
    trades_returns =day_df.tradesDone.to_numpy()

    open_h_returns =  hour_df.open.to_numpy()
    high_h_returns =  hour_df.high.to_numpy()
    low_h_returns =   hour_df.low.to_numpy()
    close_h_returns = hour_df.close.to_numpy()
    vol_h_returns =   hour_df.volume.to_numpy()
    volNot_h_returns =hour_df.volumeNotional.to_numpy()
    trades_h_returns =hour_df.tradesDone.to_numpy()

    dlag_opt = [1, 2]
    use_hlag = [0, 1, 2, 3, 4, 5]

    # dlag_opt = [2]
    # use_hlag = [2]

    for d_nlags in dlag_opt:
        for h_nlags in use_hlag:
            EXPERIMENT_NAME = f"final_forecasts/SKIPXV_{d_nlags}_{h_nlags}"

            bound_lag = max(d_nlags, ((h_nlags-1)//freq + 1))
            y_raw = close_returns[bound_lag:].reshape(-1, 1)
            Xlist = np.arange(1, len(y_raw) + 1).reshape(-1, 1)
            if h_nlags > 0:
                for t_h in range(0, h_nlags):
                    Xlist = np.concatenate(
                        [
                            # [(bound_lag*freq-1-t_h):(-1-t_h):freq]
                            # [(bound_lag*freq-t_h):(-t_h):freq]
                            Xlist,
                            open_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                            high_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                            low_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                            close_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                            vol_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                            volNot_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                            trades_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                        ], axis=1)
            if d_nlags > 0:
                for t in range(0, d_nlags):
                    Xlist = np.concatenate(
                        [
                            Xlist,
                            open_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                            high_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                            low_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                            close_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                            vol_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                            volNot_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                            trades_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                        ], axis=1)

            Xlist = Xlist[:, 1:]
            X_pp = pp.MinMaxScaler().fit(Xlist)
            y_pp = pp.MinMaxScaler().fit(y_raw)
            Xvoortest = X_pp.transform(Xlist)
            yvoortest = y_pp.transform(y_raw)

            Xtrain, Xtest, ytrain, ytest = ms.train_test_split(Xvoortest, yvoortest, test_size=365, shuffle=False)
            print("Data has been fully transformed and split")

            n_repeats = 1
            ytest = y_pp.inverse_transform(ytest.reshape(1, -1)).ravel()

            best_val_mse = np.inf
            best_val_mse_sd = 0
            best_K = [200, 100, 50, 20, 5]

            K_opt = [
                [100, 50, 20, 10],
                [200, 100, 50, 20],
                [100, 50, 20, 10, 5],
                [200, 100, 50, 20, 5],
                [100, 75, 50, 20, 10, 5],
                [200, 100, 50, 20, 10, 5],
            ]

            # Select layer size using validation set
            if True:
                Xt, Xv, yt, yv = ms.train_test_split(Xtrain, ytrain, test_size=120, shuffle=False)
                Xtt, Xtv, ytt, ytv = ms.train_test_split(Xt, yt, test_size=30, shuffle=False)
                yval = y_pp.inverse_transform(yv.reshape(1, -1)).ravel()
                tXtt = tf.convert_to_tensor(Xtt)
                tXtv = tf.convert_to_tensor(Xtv)
                tytt = tf.convert_to_tensor(ytt)
                tytv = tf.convert_to_tensor(ytv)

                for K in K_opt:
                    mses = np.zeros(10)
                    for i in range(len(mses)):
                        predictor = return_MLP_skip_estimator(tXtt, tXtv, tytt, tytv, Xt.shape[1], verbose=0, K=K, activation='tanh', epochs=20_000, patience=50)
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

            # Select final model based on small validation set
            Xt, Xv, yt, yv = ms.train_test_split(Xtrain, ytrain, test_size=30, shuffle=False)
            yval = y_pp.inverse_transform(yv.reshape(1, -1)).ravel()

            best_final_mse = np.inf

            # Robustness of final model
            n_tests = 10
            final_results = np.zeros(n_tests)
            for i in range(n_tests):
                nn = return_MLP_skip_estimator(Xt, Xv, yt, yv, Xt.shape[1], verbose=0, K=best_K, activation='tanh', epochs=20_000, patience=50)
                test_f = nn.predict(Xtest).ravel()
                test_f = y_pp.inverse_transform(test_f.reshape(1, -1)).ravel()
                experiment_mse = mt.mean_squared_error(ytest, test_f)
                print(f"FINAL MSE: {experiment_mse:.3f}")
                final_results[i] = experiment_mse

                val_f = nn.predict(Xv).ravel()
                val_f = y_pp.inverse_transform(val_f.reshape(1, -1)).ravel()
                val_mse = mt.mean_squared_error(yval, val_f)
                if val_mse < best_final_mse:
                    best_final_mse = val_mse
                    test_forecast = test_f
            
            print("FINAL RESULTS")
            print(f"AVERAGE TEST MSE = {np.mean(final_results):.3f}")
            print(f"AVERAGE TEST SD = {np.std(final_results):.3f}")
            print(f"BEST TEST MSE = {mt.mean_squared_error(ytest, test_forecast):.3f}")
            print(f"Only mean MSE = {mt.mean_squared_error(ytest, np.full_like(ytest, np.mean(ytrain))):.3f}")
            
            np.savetxt(f'{EXPERIMENT_NAME}_TEST_STATS', np.array([np.mean(final_results), np.std(final_results), best_final_mse]))
            np.save(f'{EXPERIMENT_NAME}_FORECAST', test_forecast.ravel())

if __name__ == '__main__':
    main()