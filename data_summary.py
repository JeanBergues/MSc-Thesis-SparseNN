import pandas as pd

data = pd.read_csv('agg_btc_hour.csv', parse_dates=['date', 'ddate']).drop(['date', 'ddate'], axis=1).iloc[:, 1:]
# fd = data.diff()[:,1:] / data[:-1] * 100
fd = data.pct_change(1)
# data = pd.read_csv('agg_btc_day.csv', parse_dates=['date'])

# in_between = returns.describe().drop(['count']).to_latex(float_format="%.3f", escape=True)
# print(in_between)
df = fd.describe().drop(['count']).T
df['skew'] = fd.skew(axis=0)
df['kurt'] = fd.kurtosis(axis=0)
print(len(fd['open']))

print(df.to_latex(float_format="%.3f", escape=True))
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