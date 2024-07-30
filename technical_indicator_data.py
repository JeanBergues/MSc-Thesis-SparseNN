import pandas as pd
import numpy as np

rdf = pd.read_csv(f'level_btc_day.csv', parse_dates=['date', 'daydate'], index_col=0)

window = 14
n = 10
dict_list = []
C = rdf.close.to_numpy()
H = rdf.high.to_numpy()
L = rdf.low.to_numpy()
print(rdf.iloc[0:window])

for t, row in enumerate(rdf.itertuples()):
    if t + 1 < window: continue
    new_dict = {}

    # Calculate Moving Average
    new_dict['MA'] = np.sum(C[(t-window+1):t+1]) / window

    # Calculate Weighted Moving Average
    wma_range = np.arange(1, window+1, step=1)
    new_dict['WMA'] = np.sum(wma_range * C[(t-window+1):t+1]) / np.sum(wma_range)

    # Momentum
    new_dict['Mom'] = C[t] - C[t-n]

    # Stochastic K%
    lows = L[t-n+1:t+1]
    highs = H[t-n+1:t+1]
    LL = np.min(lows)
    HH = np.max(highs)
    new_dict['StochK'] = (C[t] - LL) / (HH - LL) * 100

    # LW
    new_dict['LW'] = (H[n] - C[t]) / (H[n] - L[n]) * 100

    # A/D
    new_dict['AD'] = (H[t] - C[t-1]) / (H[t] - L[t])

    # Label
    new_dict['Movement'] = (C[t] - C[t-1]) / C[t-1] * 100

    dict_list.append(new_dict)
    if t % 100 == 0: print(f"T = {t}")

df = pd.DataFrame(dict_list)
print(df)
df.to_csv('TI_btc_data.csv')