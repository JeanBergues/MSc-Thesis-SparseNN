import numpy as np
import pandas as pd
import sklearn.preprocessing as pp

def load_data_with_X(day_df, hour_df, d_nlags, h_nlags, freq=24):
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

    return(Xlist, y_raw)

def load_data_with_XLOG(day_df, hour_df, d_nlags, h_nlags, freq=24):
    open_returns =  day_df.open.to_numpy()
    high_returns =  day_df.high.to_numpy()
    low_returns =   day_df.low.to_numpy()
    close_returns = day_df.close.to_numpy()
    vol_returns =   day_df.volume.to_numpy()
    volNot_returns =day_df.volumeNotional.to_numpy()
    trades_returns =day_df.tradesDone.to_numpy()

    log_open_returns =  day_df.log_open.to_numpy()
    log_high_returns =  day_df.log_high.to_numpy()
    log_low_returns =   day_df.log_low.to_numpy()
    log_close_returns = day_df.log_close.to_numpy()
    log_vol_returns =   day_df.log_volume.to_numpy()
    log_volNot_returns =day_df.log_volumeNotional.to_numpy()
    log_trades_returns =day_df.log_tradesDone.to_numpy()

    open_h_returns =  hour_df.open.to_numpy()
    high_h_returns =  hour_df.high.to_numpy()
    low_h_returns =   hour_df.low.to_numpy()
    close_h_returns = hour_df.close.to_numpy()
    vol_h_returns =   hour_df.volume.to_numpy()
    volNot_h_returns =hour_df.volumeNotional.to_numpy()
    trades_h_returns =hour_df.tradesDone.to_numpy()

    log_open_h_returns =  hour_df.log_open.to_numpy()
    log_high_h_returns =  hour_df.log_high.to_numpy()
    log_low_h_returns =   hour_df.log_low.to_numpy()
    log_close_h_returns = hour_df.log_close.to_numpy()
    log_vol_h_returns =   hour_df.log_volume.to_numpy()
    log_volNot_h_returns =hour_df.log_volumeNotional.to_numpy()
    log_trades_h_returns =hour_df.log_tradesDone.to_numpy()

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

                    log_open_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                    log_high_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                    log_low_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                    log_close_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                    log_vol_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                    log_volNot_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                    log_trades_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
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

                    log_open_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                    log_high_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                    log_low_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                    log_close_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                    log_vol_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                    log_volNot_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                    log_trades_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                ], axis=1)

    Xlist = Xlist[:, 1:]

    return(Xlist, y_raw)


def load_AR_data(day_df, hour_df, d_nlags, h_nlags, freq=24):
    close_returns = day_df.close.to_numpy()
    close_h_returns = hour_df.close.to_numpy()

    bound_lag = max(d_nlags, ((h_nlags-1)//freq + 1))
    y_raw = close_returns[bound_lag:].reshape(-1, 1)
    Xlist = np.arange(1, len(y_raw) + 1).reshape(-1, 1)
    if h_nlags > 0:
        for t_h in range(0, h_nlags):
            Xlist = np.concatenate(
                [
                    Xlist,
                    close_h_returns[(bound_lag*freq-1-t_h):(-1-t_h):freq].reshape(-1, 1),
                ], axis=1)
    if d_nlags > 0:
        for t in range(0, d_nlags):
            Xlist = np.concatenate(
                [
                    Xlist,
                    close_returns[bound_lag-1-t:-1-t].reshape(-1, 1),
                ], axis=1)

    Xlist = Xlist[:, 1:]

    return(Xlist, y_raw)


def scale_data(X, y):
    X_pp = pp.MinMaxScaler().fit(X)
    y_pp = pp.MinMaxScaler().fit(y)
    Xt = X_pp.transform(X)
    yt = y_pp.transform(y)
    return (Xt, yt, X_pp, y_pp)

