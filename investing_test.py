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

full_data = pd.read_csv('FD_btc_data_hourly.csv')
dates = full_data.date
full_data = full_data.drop(['date'], axis=1)
print("Data has been fully loaded")

hold_return = (full_data.iloc[-1].close_55 / full_data.iloc[0].open_0 - 1) * 100
starting_funds = 10_000
total_funds = 10_000
hype_data = [0]
hype_horizon = 5
current_strat = 0

# Hype based strategy
for row in full_data.itertuples():
    print(f"Actualized return this hour: {row.return_hourly*100:.2f}%")
    print()
    total_funds = total_funds * (1 + current_strat * row.return_hourly)

    # When to bet
    if row.totalTrades_hour > np.mean(hype_data) + np.std(hype_data) / np.sqrt(len(hype_data)):
        current_strat = 1
    # When to short
    elif row.totalTrades_hour < np.mean(hype_data) - np.std(hype_data) / np.sqrt(len(hype_data)):
        current_strat = -1
    # When to stay
    else:
        current_strat = 0

    if len(hype_data) < hype_horizon:
        hype_data = hype_data + [row.totalTrades_hour]
    else:
        hype_data = hype_data[1:] + [row.totalTrades_hour]

    print(f"Hype of {int(row.totalTrades_hour)} against average of {int(np.mean(hype_data))}: selected strat: {current_strat}")
    print(f"Total funds = {total_funds:.0f}. Total return = {(total_funds / starting_funds - 1) * 100:.2f}%")

print(f"Hold from start returns: {hold_return:.2f}%")