import pandas as pd

full_df = pd.read_csv('btcusd_full.csv', parse_dates=['date'], nrows=1000)
frequency = 'h'

full_df['filter_date'] = full_df.date.dt.floor(freq=frequency)
full_df.drop(['date'], axis=1, inplace=True)
unique_freqs = full_df.filter_date.unique()

new_df_list = []

for t in unique_freqs:
    tdt = full_df[full_df.filter_date == t].drop(['filter_date'], axis=1)
    means = pd.DataFrame(tdt.mean()).T
    means['date'] = [t]
    new_df_list.append(means)

new_df = pd.concat(new_df_list).reset_index().drop(['index'], axis=1)
print(new_df)

new_df.to_csv("agg_btc_hour.csv")