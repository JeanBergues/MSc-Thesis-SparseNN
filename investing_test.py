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

def calc_investment_returns(forecast, real, allow_empty=False, start_val=1, trad_cost=0, use_thresholds=True):
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

        value = value * (1 + pos * r/100)
        path[t] = value

    return (value / start_val - 1, path)

full_data = pd.read_csv(f'agg_btc_day.csv', usecols=['close'])
close_prices = full_data.close.to_numpy().ravel()
y_raw = ((close_prices[1:] - close_prices[:-1]) / close_prices[:-1]).reshape(-1, 1) * 100
# dates = full_data.date
ytrain, ytest = ms.train_test_split(y_raw, test_size=365, shuffle=False)
print("Data has been fully loaded")

use_forecast = 'lasso_day_test_5_5'
forecast = np.load(f'forecasts/{use_forecast}.npy')

print(len(forecast))
print(len(ytest))
assert len(forecast) == len(ytest)

# Hype based strategy
fret, investing_results = calc_investment_returns(forecast, ytest, trad_cost=0, use_thresholds=False)
hret, holding_results = calc_investment_returns(np.ones_like(ytest), ytest, trad_cost=0, use_thresholds=False)
sret, shorting_results = calc_investment_returns(-1*np.ones_like(ytest), ytest, trad_cost=0, use_thresholds=False)
oret, optimal_results = calc_investment_returns(ytest, ytest, trad_cost=0, use_thresholds=False)

x_axis = list(range(len(ytest.ravel())))
sns.lineplot(x=x_axis, y=holding_results, color='black')
sns.lineplot(x=x_axis, y=shorting_results, color='blue')
# sns.lineplot(x=x_axis, y=optimal_results, color='green')
sns.lineplot(x=x_axis, y=investing_results, color='red')
plt.show()

x_axis = list(range(len(ytest.ravel())))
sns.lineplot(x=x_axis, y=ytest.ravel(), color='black')
sns.lineplot(x=x_axis, y=forecast.ravel(), color='blue')
plt.show()