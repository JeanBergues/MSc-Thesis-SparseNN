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

def calc_investment_returns(forecast, real, allow_empty=False, start_val=1, trad_cost=0.001, use_thresholds=True):
    value = start_val
    path = np.zeros(len(real))
    prev_pos = 1
    mean_f = 0

    lb = np.mean(forecast) - np.std(forecast)
    ub = np.mean(forecast) + np.std(forecast)

    for t, (f, r) in enumerate(zip(forecast, real)):
        pos = prev_pos
        if use_thresholds:
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
    'skipx_forc/SKIPX_day_test_3_0',
    'skipx_forc/SKIPX_day_test_1_1',
    'lasso_forc/lasso_day_test_2_2',
    'lasso_forc/lasso_day_test_7_7',
]

best_model_txt = [
    
]

for m in best_model_np:
    fc = np.load(f'{m}.npy')[-365:]
    print(f"Examining {m}")
    print(f"MSE: {mt.mean_squared_error(ytest, fc):.3f}")
    fret, investing_results = calc_investment_returns(fc, ytest, trad_cost=0.001, use_thresholds=False)
    print(f"RETURN: {fret.ravel()[0]*100:.2f}")

# Hype based strategy

hret, holding_results = calc_investment_returns(np.ones_like(ytest), ytest, trad_cost=0, use_thresholds=False)
sret, shorting_results = calc_investment_returns(-1*np.ones_like(ytest), ytest, trad_cost=0, use_thresholds=False)
oret, optimal_results = calc_investment_returns(ytest, ytest, trad_cost=0, use_thresholds=False)
print(f"Only mean MSE: {mt.mean_squared_error(ytest, np.full_like(ytest, np.mean(ytrain))):.3f}")
1/0

x_axis = list(range(len(ytest.ravel())))
sns.lineplot(x=x_axis, y=holding_results - 1, color='black', size=1.5)
sns.lineplot(x=x_axis, y=shorting_results - 1, linestyle='dashed', color='black', size=1.5)
# sns.lineplot(x=x_axis, y=optimal_results, color='green')
sns.lineplot(x=x_axis, y=investing_results - 1, color='blue', size=1)
plt.xlabel("Days")
plt.ylabel("Cumulative returns")
plt.show()

# x_axis = list(range(len(ytest.ravel())))
# sns.lineplot(x=x_axis, y=ytest.ravel(), color='black')
# sns.lineplot(x=x_axis, y=forecast.ravel(), color='blue')
# plt.show()