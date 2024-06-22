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

day_df = pd.read_csv('agg_btc_day.csv', usecols=['close'])

y_raw = day_df.close.pct_change()[1:].to_numpy().reshape(-1, 1)
y_pp = pp.StandardScaler().fit(y_raw)
yvoortest = y_pp.transform(y_raw)

ytrain, ytest = ms.train_test_split(yvoortest, test_size=0.2, shuffle=False)
# print(Xtest[:3][:,3])
# print(yvoortest[:3])
print("Data has been fully transformed and split")

# # Regular MLPs 0.604
options_q = [0, 1, 2, 3]
options_o = [0, 1, 2, 3]
options_p = [1, 2, 3]
options_l = [0, 1, 2, 3]
ytest = y_pp.inverse_transform(ytest.reshape(1, -1)).ravel()
lowest_mse = np.inf
best_config = ""

for l in options_l:
    for p in options_p:
        for q in options_q:
            for o in options_o:
                # predictor = return_lassoCV_estimor(Xtrain, ytrain.ravel(), cv=5, max_iter=5_000)
                model = arch.arch_model(y=yvoortest, mean='AR', lags=l, vol='GARCH', p=p, o=o, q=q)
                predictor = model.fit(last_obs=len(ytrain), disp=False)

                ypred = predictor.forecast(start=len(ytrain))
                ypred = y_pp.inverse_transform(np.array(ypred.mean.to_numpy()).reshape(-1, 1)).ravel()
                mse = mt.mean_squared_error(ytest, ypred)
                print(f"Finished experiment l={l}, p={p}, o={o}, q={q}")
                print(f"MSE: {mse:.6f}")

                if mse < lowest_mse:
                    best_config = f"l={l}, p={p}, o={o}, q={q}"
                    lowest_mse = mse

                # x_axis = list(range(len(ytest.ravel())))
                # sns.lineplot(x=x_axis, y=ytest.ravel(), color='black')
                # sns.lineplot(x=x_axis, y=ypred.ravel(), color='red')
                # # sns.lineplot(x=x_axis, y=ytest.ravel() - ypred.ravel(), color='blue')
                # plt.show()

print(f"Best config: {best_config} with MSE = {lowest_mse:.6f}")

# print(f"Ran {n_repeats} experiments:")
# print(f"Average MSE: {1000*np.mean(results):.3f}")
# print(f"STD of MSE: {1000*np.std(results):.3f}")
# ytrain = y_pp.inverse_transform(ytrain.reshape(-1, 1)).ravel()
# print(f"Only mean MSE: {1000*mt.mean_squared_error(ytest, np.full_like(ytest, np.mean(ytrain))):.3f}")