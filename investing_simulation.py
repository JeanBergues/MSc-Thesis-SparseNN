import pandas as pd
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.linear_model as sklm
import sklearn.metrics as mt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as itertools
import time as time
import lassonet as ln
import dieboldmariano as dm

def calc_investment_returns_with_Sharpe(forecast_train, forecast, real, vol_train, vol_test, use_quantiles=False, start_val=1, trad_cost=0.001, alpha=0.05):
    value = start_val
    path = np.zeros(len(real))
    prev_pos = 1 * np.sign(forecast[0])

    train_sharp_ratios = forecast_train / np.sqrt(vol_train[-len(forecast_train):])
    uqtl = np.quantile(train_sharp_ratios, 1-alpha) if use_quantiles else 0
    lqtl = np.quantile(train_sharp_ratios, alpha) if use_quantiles else 0

    for t, (f, r) in enumerate(zip(forecast, real)):
        sharpe = f / np.sqrt(vol_test[t])

        pos = prev_pos
        if sharpe < lqtl:
            pos = -1
        elif sharpe > uqtl:
            pos = 1

        if pos != prev_pos: value = value * (1 - trad_cost)
        prev_pos = pos

        value = value * (1 + pos * r/100)
        path[t] = value

    return (value / start_val - 1, path)

def calc_investment_returns(forecast, real, ytrain, allow_empty=False, start_val=1, trad_cost=0.001, use_thresholds=True):
    value = start_val
    path = np.zeros(len(real))
    prev_pos = 0
    mean_f = 0

    if use_thresholds:
        last_seen = [0] * 7

    for t, (f, r) in enumerate(zip(forecast, real)):
        pos = prev_pos
        if use_thresholds:
            seen = np.array(last_seen[-7:])
            lb = 0 - np.std(seen)
            ub = 0 + np.std(seen)
            last_seen.append(f)

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

        value = value * (1 + pos * r/100)
        path[t] = value

    return (value / start_val - 1, path)

def plot_returns(ytrue, ytrain, forecasts=[], names=[], strat=False):
    paths_to_plot = {}
    leg = names
    leg += ['Long', 'Short']
    
    for n, m in zip(names, forecasts):
        fc = m.ravel()[-365:]
        fret, investing_results = calc_investment_returns(fc, ytrue, trad_cost=0.001, allow_empty=True, use_thresholds=strat, ytrain=ytrain)
        paths_to_plot[n] = investing_results

    hret, holding_results = calc_investment_returns(np.ones_like(ytrue).ravel(), ytrue, trad_cost=0, use_thresholds=False, ytrain=ytrain)
    sret, shorting_results = calc_investment_returns(-1*np.ones_like(ytrue).ravel(), ytrue, trad_cost=0, use_thresholds=False, ytrain=ytrain)

    x_axis = list(range(len(ytrue.ravel())))
    fig = plt.figure(figsize=(8, 3))

    # sns.lineplot(x=x_axis, y=optimal_results, color='green')
    for path in paths_to_plot.values():
        fig =sns.lineplot(x=x_axis, y=path - 1, size=1)

    fig = sns.lineplot(x=x_axis, y=holding_results - 1, color='black', size=5)
    fig = sns.lineplot(x=x_axis, y=shorting_results - 1, linestyle='dashed', color='black', size=5)

    legd = fig.get_legend()
    for t, l in zip(legd.texts, leg):
        t.set_text(l)

    sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))
    plt.xlabel("Days")
    plt.ylabel("Cumulative returns")
    plt.show()


def main():
    full_data = pd.read_csv(f'pct_btc_day.csv', usecols=['close'])
    close_prices = full_data.close.to_numpy().ravel()
    y_raw = close_prices

    # dates = full_data.date
    ytrain, ytest = ms.train_test_split(y_raw, test_size=365, shuffle=False)
    print("Data has been fully loaded")

    best_model_test_np = [
        'final_forecasts/CV_SNN_1_0_FORECAST',
        'final_forecasts/CV_SNN_2_0_FORECAST',
        'final_forecasts/CV_SNN_1_24_FORECAST',
        'final_forecasts/CV_SNN_2_48_FORECAST',
        'final_LN_forecasts/DLN_SNN_7_24_LN_FORECAST',
    ]

    best_model_train_np = [
        'final_forecasts/CV_SNN_1_0_TRAIN_FORECAST',
        'final_forecasts/CV_SNN_2_0_TRAIN_FORECAST',
        'final_forecasts/CV_SNN_1_24_TRAIN_FORECAST',
        'final_forecasts/CV_SNN_2_48_TRAIN_FORECAST',
        'final_LN_forecasts/DLN_SNN_7_24_LN_TRAIN_FORECAST',
    ]

    best_model_test_txt = [
        'final_R_forecasts/MIDAS_test',
        'final_R_forecasts/garch_test',
        'final_R_forecasts/arima_day_test',
    ]

    best_model_train_txt = [
        'final_R_forecasts/MIDAS_train',
        'final_R_forecasts/garch_train',
        'final_R_forecasts/arima_day_train',
    ]

    paths_to_plot = {}
    series_to_test = {}
    WITH_SHARPE = True
    USE_QUANTILES = True

    train_vol = np.loadtxt(f'final_R_forecasts/garch_train_vol.txt').ravel() if WITH_SHARPE else np.ones_like(ytrain)
    test_vol = np.loadtxt(f'final_R_forecasts/roll_garch_vol.txt').ravel() if WITH_SHARPE else np.ones_like(ytest)

    for mtest, mtrain in zip(best_model_test_np, best_model_train_np):
        fc = np.load(f'{mtest}.npy').ravel()[-365:]
        tfc = np.load(f'{mtrain}.npy').ravel()
        print(f"Examining {mtest}")
        print(f"MSE: {mt.mean_squared_error(ytest, fc):.3f}")
        print(f"MAPE: {mt.mean_absolute_percentage_error(ytest, fc):.3f}")
        print(f"%Correct: {np.sum(np.sign(ytest) == np.sign(fc)) / len(fc) * 100:.3f}%")
        fret, investing_results = calc_investment_returns_with_Sharpe(tfc, fc, ytest, train_vol, test_vol, trad_cost=0.001, use_quantiles=USE_QUANTILES)
        print(f"RETURN: {fret*100:.2f}")
        paths_to_plot[mtest] = investing_results
        series_to_test[mtest] = fc

    for mtest, mtrain in zip(best_model_test_txt, best_model_train_txt):
        fc = np.loadtxt(f'{mtest}.txt').ravel()[-365:]
        tfc = np.loadtxt(f'{mtrain}.txt').ravel()
        print(f"Examining {mtest}")
        print(f"MSE: {mt.mean_squared_error(ytest, fc):.3f}")
        print(f"MAPE: {mt.mean_absolute_percentage_error(ytest, fc):.3f}")
        print(f"%Correct: {np.sum(np.sign(ytest) == np.sign(fc)) / len(fc) * 100:.3f}%")
        fret, investing_results = calc_investment_returns_with_Sharpe(tfc, fc, ytest, train_vol, test_vol, trad_cost=0.001, use_quantiles=USE_QUANTILES)
        print(f"RETURN: {fret*100:.2f}")
        paths_to_plot[mtest] = investing_results
        series_to_test[mtest] = fc

    hret, holding_results = calc_investment_returns(np.ones_like(ytest).ravel(), ytest, None, trad_cost=0, use_thresholds=False)
    sret, shorting_results = calc_investment_returns(-1*np.ones_like(ytest).ravel(), ytest, None, trad_cost=0, use_thresholds=False)
    print(f"Only mean MSE: {mt.mean_squared_error(ytest, np.full_like(ytest, np.mean(ytrain))):.3f}")
    print(f"Only mean MAPE: {mt.mean_absolute_percentage_error(ytest, np.full_like(ytest, np.mean(ytrain))):.3f}")
    print(f"Only mean %Correct: {np.sum(np.sign(ytest) == np.sign(np.full_like(ytest, np.mean(ytrain)))) / len(ytest) * 100:.3f}%")

    y_prev = y_raw[-366:-1]
    print(f"Repeat MSE: {mt.mean_squared_error(ytest, y_prev):.3f}")
    print(f"Repeat MAPE: {mt.mean_absolute_percentage_error(ytest, y_prev):.3f}")
    print(f"Repeat %Correct: {np.sum(np.sign(ytest) == np.sign(y_prev)) / len(ytest) * 100:.3f}%")

    # benchmark = np.load(f'skipx_forc/SKIPXA_day_test_1_1.npy')[-365:]
    # for k, v in series_to_test.items():
    #     print(k)
    #     print(dm.dm_test(ytest, benchmark, v))

    leg = ['NN(1, 0)', 'NN(2, 0)', 'NN(1, 24)', 'NN(2, 48)', 'LassoNet', 'MIDAS', 'GARCH', 'ARIMA', 'Long', 'Short']

    x_axis = list(range(len(ytest.ravel())))
    fig = plt.figure(figsize=(16, 6))

    # sns.lineplot(x=x_axis, y=optimal_results, color='green')
    for path in paths_to_plot.values():
        fig =sns.lineplot(x=x_axis, y=path - 1, size=1, linewidth=2)

    fig = sns.lineplot(x=x_axis, y=holding_results - 1, color='black', size=1, linewidth=2)
    fig = sns.lineplot(x=x_axis, y=shorting_results - 1, linestyle='dashed', color='black', size=1, linewidth=2)

    legd = fig.get_legend()
    for t, l in zip(legd.texts, leg):
        t.set_text(l)

    sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))
    plt.xlabel("Days")
    plt.ylabel("Cumulative returns")

    # plt.savefig('plots/STRAT.eps', format='eps')
    plt.show()

if __name__ == '__main__':
    main()