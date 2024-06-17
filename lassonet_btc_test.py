import pandas as pd
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as mt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as itertools
import time as time
import lassonet as ln


full_data = pd.read_csv('btc_data_hourly.csv')
dates = full_data.date
full_data = full_data.drop(['date'], axis=1)
print("Data has been fully loaded")

n_lags = 1

std_data = pp.StandardScaler().fit_transform(full_data)
X = np.concatenate([std_data[lag:-(n_lags-lag),:-1] for lag in range(n_lags)], axis=1)
y = std_data[n_lags:,-1]
print(X.shape)
print(y.shape)
Xtrain, Xtest, ytrain, ytest = ms.train_test_split(X, y, test_size=0.2, shuffle=True)
print("Data has been fully transformed and split")

regressor = ln.LassoNetRegressor(verbose=2, hidden_dims=(50,5))
regressor.fit(Xtrain, ytrain)
ypred = regressor.predict(Xtest)
print(f"LassoNet got mse of: {mt.mean_squared_error(ytest, ypred):.4f}")



