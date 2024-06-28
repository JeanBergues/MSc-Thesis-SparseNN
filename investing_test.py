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

full_data = pd.read_csv(f'agg_btc_day.csv', usecols=['open', 'close'])
close_prices = full_data.close.to_numpy().ravel()
open_prices = full_data.open.to_numpy().ravel()
y_raw = ((close_prices[1:] - close_prices[:-1]) / close_prices[:-1]).reshape(-1, 1)
# dates = full_data.date
ytrain, ytest = ms.train_test_split(y_raw, test_size=0.2, shuffle=False)
print("Data has been fully loaded")

use_forecast = 'garch'
forecast = np.load(f'forecasts/{use_forecast}.npy')
invest = 1
hold = 1
ytest = ytest[1:]
investing_results = np.zeros(len(ytest))
holding_results = np.zeros(len(ytest))
last_strat = 1

# Hype based strategy
for t in range(len(ytest)):
    strat = 1
    if forecast[t] < 0:
        strat = -1
    invest = invest * (1 + strat * ytest[t])
    investing_results[t] = invest
    hold = hold * (1 + ytest[t])
    holding_results[t] = hold

x_axis = list(range(len(ytest.ravel())))
sns.lineplot(x=x_axis, y=holding_results, color='black')
sns.lineplot(x=x_axis, y=investing_results, color='red')
plt.show()