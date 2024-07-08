import pandas as pd
import numpy as np

minute_interval = 5
minutes_per_row = 5

# Read in data
data = pd.read_csv('btcusd_full.csv', parse_dates=['date'], index_col=['date'])

start_time = data.index[0]
end_time = data.index[-1]

date_range = pd.date_range(start_time, end_time, freq='5min')
dict_list = []

last_valid_t = 0
for t in date_range:
    try:
        t_df = data.loc[t]
        last_valid_t = t
    except KeyError:
        # print(f"{t}")
        floored_date = t.floor('d')
        new_dict = {}
        new_dict['ddate'] = floored_date
        new_dict['date'] = t
        dict_list.append(new_dict)
        # data.loc[t] = data.loc[last_valid_t].copy()

df = pd.DataFrame(dict_list)
df['yearmonth'] = df.apply(lambda x: f"{x.ddate.year}-{x.ddate.month:02}", axis=1)
print(df.yearmonth.value_counts().sort_index().to_latex())
# new_df = data.sort_index()
# new_df.to_csv('it_btc_data.csv')