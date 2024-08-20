import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.graphics.tsaplots as tsp

fullmdata = pd.read_csv('imputed_btc_data.csv', parse_dates=['date'])
mdata = fullmdata.drop(['date'], axis=1)
hdata = pd.read_csv('pct_btc_hour.csv')
ddata = pd.read_csv('pct_btc_day.csv')

for fd in [mdata, hdata, ddata]:
    df = fd.describe().drop(['count']).T
    df['skew'] = fd.skew(axis=0)
    df['kurt'] = fd.kurtosis(axis=0)
    print(df.to_latex(float_format="%.3f", escape=True))    

f = tsp.plot_pacf(ddata.close, lags=30, title='')
plt.xlabel("lags")
plt.savefig('plots/pacf_day')
# plt.show()

f = tsp.plot_pacf(hdata.close, lags=30, title='')
plt.xlabel("lags")
plt.savefig('plots/pacf_hour')
# plt.show()

start_time = fullmdata.date.iloc[0]
end_time = fullmdata.date.iloc[-1]
ddate_range = pd.date_range(start_time, end_time, freq='d')[1:]
hdate_range = pd.date_range(start_time, end_time, freq='h')[1:]

fig = plt.figure(figsize=(15, 20), )
for i, col in enumerate(ddata.columns):
    ax = plt.subplot(7, 1, i+1)
    sns.lineplot(x=ddate_range, y=ddata[col], ax=ax)
    sns.lineplot(x=hdate_range, y=hdata[col], ax=ax, size=1)
    plt.axvline(ddate_range.to_numpy()[-364], color='black', linestyle='--')
    #plt.axvline(ddate_range.to_numpy()[-484], color='black', linestyle='--')
    ax.set_title(f"returns of {col}")
    ax.set_ylabel(f"% difference")
    ax.legend_.remove()
    if i == 6: ax.set_xlabel(f"date")
    print(f"Finished subplot {i+1}")

plt.subplots_adjust(hspace=1)
plt.savefig('plots/dh_rets_noval.eps', format='eps')
# plt.show()

fig = plt.figure(figsize=(15, 20), )
for i, col in enumerate(mdata.columns):
    ax = plt.subplot(7, 1, i+1)
    sns.lineplot(x=fullmdata.date, y=mdata[col], ax=ax)
    plt.axvline(ddate_range.to_numpy()[-364], color='black', linestyle='--')
    #plt.axvline(ddate_range.to_numpy()[-484], color='black', linestyle='--')
    ax.set_title(f"level of {col}")
    ax.set_ylabel(f"US$" if col not in ['volume', 'tradesDone'] else f"amount")
    if i == 6: ax.set_xlabel(f"date")
    print(f"Finished subplot {i+1}")

plt.subplots_adjust(hspace=1)
plt.savefig('plots/level_plots_noval.eps', format='eps')
# plt.show()