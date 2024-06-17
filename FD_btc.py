import pandas as pd
import numpy as np

data = pd.read_csv('btcusd_full.csv')
value_col_names = data.columns[1:]

data_old = data.iloc[0:-1]
data_new = data.iloc[1:]

new_df = pd.DataFrame()
new_df['date'] = data['date'].iloc[1:]

for col in value_col_names:
    old_vals = data_old[col].to_numpy()
    new_vals = data_new[col].to_numpy()
    new_df[col] = data_new[col]
    new_df[col + '_ad'] = new_vals - old_vals
    new_df[col + '_rd'] = np.log(new_vals / old_vals)

print(new_df)
new_df.to_csv("FD_btc_full_data.csv")