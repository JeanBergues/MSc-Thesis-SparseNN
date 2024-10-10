import pandas as pd
import statsmodels.tsa.stattools as tsa
import statsmodels.stats.stattools as stats

hdata = pd.read_csv('pct_btc_hour.csv')
ddata = pd.read_csv('pct_btc_day.csv').iloc[-365:]

DAY = True
if DAY:
    df = ddata
else:
    df = hdata

for var in df:
    print(f"\nCalculating statistics for {var} ({'daily' if DAY else 'hourly'} frequency)")
    y = df[var].to_numpy()
    print(f"Dickey-Fuller p-value: {tsa.adfuller(y)}")
    # print(f"KPSS p-value: {tsa.kpss(y)[1]:.5f}")
    # print(f"ZivotAndrews p-value: {tsa.zivot_andrews(y, regression='ct')}")
    # print(f"JarqueBera p-value: {stats.jarque_bera(y)[1]:.5f}")
    # acf = tsa.acf(y)[0]
    # print(f"LJung Box: {tsa.q_stat(acf, len(y))[1]}")