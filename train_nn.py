import pandas as pd
import numpy as np
import tensorflow as tf

np.random.seed(1234)
tf.random.set_seed(1234)

import keras as ks
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as mt


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


def return_MLP_skip_estimator(Xt, Xv, yt, yv, K=[10], activation='relu', epochs=500, patience=30, verbose=0, drop=0):
    inp = ks.layers.Input(shape=(Xt.shape[1],))
    # skip = ks.layers.Dense(units=1, activation='linear', use_bias=True, name='skip_layer')(inp)
    skip = ks.layers.Dense(units=1, activation='linear', use_bias=True, kernel_regularizer=ks.regularizers.L1(), name='skip_layer')(inp)
    dp = ks.layers.Dropout(drop)(inp)
    gw = ks.layers.Dense(units=K[0], activation=activation, name='gw_layer')(dp)
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
    nn.compile(optimizer=ks.optimizers.Adam(), loss=ks.losses.MeanSquaredError())
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

    dlag_opt = [1]
    use_hlag = [True]

    for d_nlags in dlag_opt:
        for use_h in use_hlag:
            h_nlags = d_nlags if use_h else 0

            bound_lag = max(d_nlags, ((h_nlags-1)//freq + 1))
            y_raw = close_returns[bound_lag:].reshape(-1, 1)
            Xlist = np.arange(1, len(y_raw) + 1).reshape(-1, 1)
            if h_nlags > 0:
                for t_h in range(0, h_nlags):
                    Xlist = np.concatenate(
                        [
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

            best_mse = np.inf
            best_K = [200, 150, 100, 50, 20]

            K_opt = [
                # [5],
                # [20],
                # [50],
                # [100],
                # [200],

                # [20, 5],
                # [50, 20],
                # [100, 50],
                # [200, 100],

                # [50, 20, 5],
                # [100, 50, 20],
                # [200, 100, 50],

                # [100, 50, 20, 5],
                # [200, 100, 50, 20],

                # [100, 50, 20, 10, 5],
                [200, 100, 50, 20, 5],
                # [300, 200, 100, 50, 20],
                # [400, 300, 200, 100, 50],
            ]

            Xt, Xv, yt, yv = ms.train_test_split(Xtrain, ytrain, test_size=120, shuffle=False)
            yv = y_pp.inverse_transform(yv.reshape(1, -1)).ravel()

            # for K in K_opt:
            #     mses = np.zeros(5)
            #     for i in range(len(mses)):
            #         predictor = return_MLP_skip_estimator(Xt, yt, verbose=0, K=K, test_size=60, activation='tanh', epochs=20_000, patience=25, drop=0)
            #         ypred = predictor.predict(Xv).ravel()
            #         ypred = y_pp.inverse_transform(ypred.reshape(1, -1)).ravel()
            #         mse = mt.mean_squared_error(yv, ypred)
            #         mses[i] = mse

            #     print(f"Finished experiment")
            #     print(f"K = {K}")
            #     print(f"MSE: {np.mean(mses):.3f}")
            #     print(f"MSE SDEV: {np.std(mses):.3f}")

            #     if np.mean(mses) < best_mse:
            #         best_K = K
            #         best_mse = np.mean(mses)

            np.random.seed(1234)
            tf.random.set_seed(1234)
            Xt, Xv, yt, yv = ms.train_test_split(Xtrain, ytrain, test_size=30, shuffle=False)

            # Robustness of final model
            final_results = np.zeros(5)
            for i in range(5):
                nn = return_MLP_skip_estimator(Xt, Xv, yt, yv, verbose=1, K=best_K, activation='tanh', epochs=20_000, patience=50, drop=0)
                test_f = nn.predict(Xtest).ravel()
                test_f = y_pp.inverse_transform(test_f.reshape(1, -1)).ravel()
                experiment_mse = mt.mean_squared_error(ytest, test_f)
                print(f"FINAL MSE: {experiment_mse:.3f}")
                final_results[i] = experiment_mse
            
            print(np.mean(final_results))
            print(np.std(final_results))

            1/0

            Xtt, Xtv, ytt, ytv = ms.train_test_split(Xtrain, ytrain, test_size=30, shuffle=False)
            final_predictor = return_MLP_skip_estimator(Xtt, Xtv, ytt, ytv, verbose=1, K=best_K, activation='tanh', epochs=20_000, patience=50, drop=0)
            test_forecast = final_predictor.predict(Xtest).ravel()
            test_forecast = y_pp.inverse_transform(test_forecast.reshape(1, -1)).ravel()
            full_forecast = final_predictor.predict(Xvoortest).ravel()
            full_forecast = y_pp.inverse_transform(full_forecast.reshape(1, -1)).ravel()

            print("FINAL RESULTS")
            print(f"BEST K = {best_K}")
            print(f"MSE = {best_mse:.3f}")

            print(f"FINAL MSE: {mt.mean_squared_error(ytest, test_forecast):.3f}")
            # print(f"FINAL RETURN: {calc_investment_returns(test_forecast, ytest, allow_empty=False, use_thresholds=False, ytrain=ytrain)[0]:.3f}")
            print(f"Only mean MSE: {mt.mean_squared_error(ytest, np.full_like(ytest, np.mean(ytrain))):.3f}")
            
            np.savetxt(f'final_forecasts/SKIPX_{d_nlags}_{h_nlags}_K', np.array(best_K))
            np.savetxt(f'final_forecasts/SKIPX_{d_nlags}_{h_nlags}_MSE', np.array([best_mse]))
            np.save(f'final_forecasts/SKIPX_test_{d_nlags}_{h_nlags}', test_forecast.ravel())
            np.save(f'final_forecasts/SKIPX_full_{d_nlags}_{h_nlags}', full_forecast.ravel())

if __name__ == '__main__':
    main()