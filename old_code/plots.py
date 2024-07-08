import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.graphics.tsaplots as tsp
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

data = pd.read_csv('agg_btc_day.csv', parse_dates=['date', 'ddate'])
hata = pd.read_csv('agg_btc_hour.csv', parse_dates=['date', 'ddate'])

fd = data.drop(['date', 'ddate'], axis=1).iloc[:, 1:].pct_change(1)
fh = hata.drop(['date', 'ddate'], axis=1).iloc[:, 1:].pct_change(1)

# print(data.date.to_numpy()[np.argmin(fd.close)])

y = fd.close.values.ravel()[1:]
print(jarque_bera(y))

1 / 0
f = tsp.plot_pacf(y, lags=30, title='')
plt.xlabel("lags")
plt.savefig('plots/pacf_day')
1/0

fig = plt.figure(figsize=(15, 20), )
x_axis = data.date[1:]
h_axis = hata.date[1:]
for i, col in enumerate(fd.columns):
    ax = plt.subplot(7, 1, i+1)
    sns.lineplot(x=x_axis, y=fd[col], ax=ax)
    sns.lineplot(x=h_axis, y=fh[col], ax=ax, size=1)
    plt.axvline(x_axis.to_numpy()[-364], color='black', linestyle='--')
    plt.axvline(x_axis.to_numpy()[-484], color='black', linestyle='--')
    ax.set_title(f"returns of {col}")
    ax.set_ylabel(f"% difference")
    ax.legend_.remove()
    if i == 6: ax.set_xlabel(f"date")
    print(f"Finished subplot {i+1}")

plt.subplots_adjust(hspace=1)
plt.savefig('plots/dh_rets.eps', format='eps')