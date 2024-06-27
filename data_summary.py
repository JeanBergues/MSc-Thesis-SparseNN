import pandas as pd

# data = pd.read_csv('agg_btc_day.csv', parse_dates=['date', 'ddate'])
data = pd.read_csv('btcusd_full.csv', nrows=20000)
midpoints = (data.open + data.close) / 2
data['variance'] = (data.high - midpoints) + (midpoints - data.low)
# print(variance)
# 1/0
# returns = data.pct_change(1)[1:] * 100
# print(len(returns))
data.open = data.open / 1e3
data.low = data.low / 1e3
data.high = data.high / 1e3
data.close = data.close / 1e3

data.volume = data.volume / 1e4
data.volumeNotional = data.volumeNotional / 1e8
data.tradesDone = data.tradesDone / 1e5

# in_between = returns.describe().drop(['count']).to_latex(float_format="%.3f", escape=True)
# print(in_between)
in_between = data.describe().drop(['count']).T.to_latex(float_format="%.2f", escape=True).replace('e+', '$\\times10^{')
print(in_between)
# final_string = ""
# skip_first = False
# for part in in_between.split('{'):
#     if part[-1] == '^':
#         if skip_first:
#             final_string = final_string + '{' + str(int(part[0:2])) + '}$' + part[2:]
#         else:
#             final_string = final_string + '{' + part
#             skip_first = True
#     else:
#         final_string = final_string + '{' + part

# print(final_string[1:])