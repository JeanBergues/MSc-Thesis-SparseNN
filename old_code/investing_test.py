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

full_data = pd.read_csv(f'agg_btc_day.csv', usecols=['close'])
close_prices = full_data.close.to_numpy().ravel()
y_raw = ((close_prices[1:] - close_prices[:-1]) / close_prices[:-1]).reshape(-1, 1) * 100
# dates = full_data.date
ytrain, ytest = ms.train_test_split(y_raw, test_size=365, shuffle=False)
print("Data has been fully loaded")

best_model_np = [
    'forecasts/SKIP_day_test_7_0',
    'forecasts/SKIP_day_test_7_7',
    'skipx_forc/SKIPXA_day_test_3_0',
    # 'skipx_forc/SKIPXA_day_test_1_1',
    'lasso_forc/elasso_day_test_2_2',
    'lasso_forc/elasso_day_test_3_3',
    'lasso_forc/lasso_day_test_7_7',
]

best_model_txt = [
    'txt_forecast/amidas_day_m_test',
    'txt_forecast/amidasX_day_m_test',
    'txt_forecast/garch_day_test',
    'txt_forecast/arima_day_test',
    'txt_forecast/arimaX_hour_test',
]

paths_to_plot = {}
series_to_test = {}

for m in best_model_np:
    fc = np.load(f'{m}.npy')[-365:]
    print(f"Examining {m}")
    print(f"MSE: {mt.mean_squared_error(ytest, fc):.3f}")
    fret, investing_results = calc_investment_returns(fc, ytest, trad_cost=0.001, allow_empty=True, use_thresholds=True, ytrain=ytrain)
    print(f"RETURN: {fret*100:.2f}")
    paths_to_plot[m] = investing_results
    series_to_test[m] = fc

for m in best_model_txt:
    fc = np.loadtxt(f'{m}.txt', delimiter=' ').ravel()[-365:]
    print(f"Examining {m}")
    print(f"MSE: {mt.mean_squared_error(ytest, fc):.3f}")
    fret, investing_results = calc_investment_returns(fc, ytest, trad_cost=0.001, allow_empty=True, use_thresholds=True, ytrain=ytrain)
    print(f"RETURN: {fret*100:.2f}")
    paths_to_plot[m] = investing_results
    series_to_test[m] = fc

# Hype based strategy

hret, holding_results = calc_investment_returns(np.ones_like(ytest).ravel(), ytest, trad_cost=0, use_thresholds=False, ytrain=ytrain)
sret, shorting_results = calc_investment_returns(-1*np.ones_like(ytest).ravel(), ytest, trad_cost=0, use_thresholds=False, ytrain=ytrain)
oret, optimal_results = calc_investment_returns(ytest, ytest, trad_cost=0, use_thresholds=False, ytrain=ytrain)
print(f"Only mean MSE: {mt.mean_squared_error(ytest, np.full_like(ytest, np.mean(ytrain))):.3f}")

benchmark = np.load(f'skipx_forc/SKIPXA_day_test_1_1.npy')[-365:]
for k, v in series_to_test.items():
    print(k)
    print(dm.dm_test(ytest, benchmark, v))

1 / 0
leg = ['ANN(l, 0)', 'ANN(l, l)', 'ANNX(l, 0)', 'ANNX(l, l)', 
            'LassoNet(2,2)', 'LassoNet(7,7)', 'MIDAS', 'MIDASX', 'GARCH', 'ARIMA', 'ARIMAX', 'Long', 'Short']

x_axis = list(range(len(ytest.ravel())))
fig = plt.figure(figsize=(16, 6))

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

# plt.savefig('plots/stepret.eps', format='eps')
plt.show()

# x_axis = list(range(len(ytest.ravel())))
# sns.lineplot(x=x_axis, y=ytest.ravel(), color='black')
# sns.lineplot(x=x_axis, y=forecast.ravel(), color='blue')
# plt.show()