import arch.unitroot
import arch.univariate
import arch.univariate.distribution
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import arch as arch

import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as mt
import sklearn.linear_model as sklm

###############################################################################################################################################################################################

USE_OLD_DATA = False
extra = '_old' if USE_OLD_DATA else ''
day_df = pd.read_csv(f'agg_btc_min{extra}.csv', usecols=['close'])
# old_df = pd.read_csv(f'agg_btc_day_old.csv', parse_dates=['date', 'ddate'])
# hour_df = pd.read_csv(f'agg_btc_hour{extra}.csv', parse_dates=['date', 'ddate'])
# min_df = pd.read_csv(f'agg_btc_min{extra}.csv', parse_dates=['date', 'ddate', 'hdate'])

close_prices = day_df.close.to_numpy().ravel()
y_raw = ((close_prices[1:] - close_prices[:-1]) / close_prices[:-1]).reshape(-1, 1)
yvoortest = y_raw * 100

print(arch.unitroot.ADF(yvoortest).summary())
print(arch.unitroot.PhillipsPerron(yvoortest).summary())
print(arch.unitroot.KPSS(yvoortest).summary())

# y_raw = np.clip(y_raw, np.mean(y_raw) - np.std(y_raw), np.mean(y_raw) + np.std(y_raw))
# y_pp = pp.StandardScaler().fit(y_raw)
# yvoortest = y_pp.transform(y_raw)

ytrain, ytest = ms.train_test_split(yvoortest, test_size=0.2, shuffle=False)

# use_x_cols = ['open']
# X = day_df[use_x_cols[0]].pct_change().to_numpy()[:-1].reshape(-1, 1)
# if len(use_x_cols) > 1:
#     for col in use_x_cols[1:]:
#         X = np.concatenate([X, day_df[col].pct_change().to_numpy()[:-1].reshape(-1, 1)], axis=1)

# for col in use_x_cols:
#         X = np.concatenate([X, day_df[col].to_numpy()[:-1].reshape(-1, 1)], axis=1)   

# X_pp = pp.StandardScaler().fit(X)
# X = X_pp.transform(X)

# print(Xtest[:3][:,3])
# print(yvoortest[:3])
print("Data has been fully transformed and split")

# # Regular MLPs 0.604
options_q = [1]
options_o = [1]
options_p = [1]
options_l = [1]
# ytest = y_pp.inverse_transform(ytest.reshape(1, -1)).ravel()
best_pred = np.zeros_like(ytest)
lowest_mse = np.inf
best_config = ""

for l in options_l:
    for p in options_p:
        for q in options_q:
            for o in options_o:
                model = arch.univariate.ARCHInMean(y=yvoortest, lags=l, volatility=arch.univariate.GARCH(p=p, o=o, q=q), form='log', distribution=arch.univariate.distribution.StudentsT())
                # model = arch.arch_model(y=yvoortest, x=X, mean='constant', lags=l, vol='GARCH', p=p, o=o, q=q)
                predictor = model.fit(last_obs=len(ytrain), disp=False)
                lm_test_res = predictor.arch_lm_test()
                print(lm_test_res.null)
                print(lm_test_res.pval)
                # predictor = model.fit(disp=False)
                print(predictor.summary())

                ypred = predictor.forecast(start=len(ytrain)).mean.to_numpy()
                # ypred = y_pp.inverse_transform(np.array(ypred.mean.to_numpy()).reshape(-1, 1)).ravel()
                mse = mt.mean_squared_error(ytest, ypred)
                print(f"Finished experiment l={l}, p={p}, o={o}, q={q}")
                print(f"MSE: {mse:.6f}")

                if mse < lowest_mse:
                    best_config = f"l={l}, p={p}, o={o}, q={q}"
                    lowest_mse = mse
                    best_pred = ypred

                # x_axis = list(range(len(ytest.ravel())))
                # sns.lineplot(x=x_axis, y=ytest.ravel(), color='black')
                # sns.lineplot(x=x_axis, y=ypred.ravel(), color='red')
                # # sns.lineplot(x=x_axis, y=ytest.ravel() - ypred.ravel(), color='blue')
                # plt.show()

print(f"Best config: {best_config} with MSE = {lowest_mse:.6f}")
# ytrain = y_pp.inverse_transform(ytrain.reshape(-1, 1)).ravel()
print(f"Only mean MSE: {mt.mean_squared_error(ytest, np.full_like(ytest, np.mean(ytrain))):.3f}")

x_axis = list(range(len(ytest.ravel())))
sns.lineplot(x=x_axis, y=ytest.ravel(), color='black')
sns.lineplot(x=x_axis, y=best_pred.ravel(), color='red')
# y_old = old_df.close.pct_change()[1:].to_numpy().reshape(-1, 1)
# _, yold = ms.train_test_split(y_old, test_size=0.2, shuffle=False)
# sns.lineplot(x=x_axis, y=yold.ravel(), color='red')
# sns.lineplot(x=x_axis, y=ytest.ravel() - ypred.ravel(), color='blue')
plt.show()