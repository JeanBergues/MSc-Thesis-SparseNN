import pandas as pd
import numpy as np


full_df = pd.read_csv('it_btc_data.csv', parse_dates=['date'])
frequency = 'h'

full_df['filter_date'] = full_df.date.dt.floor(freq=frequency)

full_df.drop(['date'], axis=1, inplace=True)
unique_freqs = full_df.filter_date.unique()
print(f"Parsing {len(unique_freqs)} dates.")

new_df_list = []

for i, t in enumerate(unique_freqs):
    tdt = full_df[full_df.filter_date == t].drop(['filter_date'], axis=1)
    means = pd.DataFrame(tdt.mean()).T
    means['open'] = tdt.iloc[0].open
    means['high'] = tdt.high.max()
    means['low'] = tdt.low.min()
    means['close'] = tdt.iloc[-1].close
    means['volume'] = tdt.volume.sum()
    means['volumeNotional'] = tdt.volumeNotional.sum()
    means['tradesDone'] = tdt.tradesDone.sum()

    means['date'] = [t]
    means['ddate'] = [t.floor(freq='d')]
    new_df_list.append(means)
    if i % 100 == 0: print(f"Finished date {i}")

new_df = pd.concat(new_df_list).reset_index().drop(['index'], axis=1)
new_df.to_csv("agg_btc_hour.csv")
