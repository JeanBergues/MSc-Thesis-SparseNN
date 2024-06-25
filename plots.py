import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('agg_btc_min.csv', parse_dates=['date'])

close_prices = data.close.to_numpy().ravel()
open_prices = data.open.to_numpy().ravel()


y_raw = ((close_prices[1:] - close_prices[:-1]) / close_prices[:-1]).reshape(-1, 1)
y_raw2 = np.log(close_prices[1:]) - np.log(close_prices[:-1])


fig = plt.figure(figsize=(15, 3))
x_axis = list(range(len(y_raw.ravel())))
x_axis = data.date[1:]
sns.lineplot(x=x_axis, y=y_raw2.ravel())
# sns.lineplot(x=x_axis, y=y_raw2.ravel(), color='red')
# sns.lineplot(x=x_axis, y=y_raw.ravel() - y_raw2.ravel(), color='blue')
plt.show()