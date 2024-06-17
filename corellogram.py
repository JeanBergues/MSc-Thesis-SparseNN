import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

full_data = pd.read_csv('btcusd_full.csv')
dates = full_data.date
print(full_data.shape)
full_data['return'] = full_data.close / full_data.open - 1
base_data = full_data.drop(['date'], axis=1).diff(1).diff(1).iloc[:-2].copy()
base_data['returnNext'] = full_data['return'].iloc[2:].copy()
print(base_data.shape)
print("Data has been fully loaded")

sns.pairplot(base_data)
plt.savefig('full_sd_corr.png')
