import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('btcusd_full.csv')
# date_col = pd.to_datetime(data.date)
date_col = range(len(data.date))
data = data.drop(['date'], axis=1)
print("Done reading in data")

data = data.diff(2)
data_plot_type = "SD"
fig = plt.figure(figsize=(15, 10))

for i, col in enumerate(data.columns):
    ax = plt.subplot(2, 4, i+1)
    sns.lineplot(x=date_col, y=data[col], ax=ax)
    ax.set_title(f"{data_plot_type} of {col}")
    print(f"Finished subplot {i+1}")

plt.show()