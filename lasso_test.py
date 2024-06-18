import pandas as pd
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.linear_model as sklm
import sklearn.metrics as mt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as itertools
import time as time
import lassonet as ln
import keras as ks

# Read in the data
full_data = pd.read_csv('FD_btc_data_hourly.csv', nrows=22000)
dates = full_data.date
full_data = full_data.drop(['date'], axis=1)
print("Data has been fully loaded")

n_lags = 8

std_data = pp.StandardScaler().fit_transform(full_data)
X = np.concatenate([std_data[lag:-(n_lags-lag),:].copy() for lag in range(n_lags)], axis=1)
y = std_data[n_lags:,-2].copy()
print(X.shape)
print(y.shape)
Xtrain, Xtest, ytrain, ytest = ms.train_test_split(X, y, test_size=0.2, shuffle=False)
print("Data has been fully transformed and split")

# full_data = pd.read_csv('superconductivity_train.csv')
# X = pp.StandardScaler().fit_transform(full_data.drop([full_data.columns[-1]], axis=1))
# y = pp.StandardScaler().fit_transform(full_data[full_data.columns[-1]].to_numpy().reshape(-1, 1))
# print(X.shape)
# print(y.shape)
# Xtrain, Xtest, ytrain, ytest = ms.train_test_split(X, y, test_size=0.2, shuffle=True)
# print("Data has been fully transformed and split")

regressor = sklm.LassoCV(cv=2)
fm = regressor.fit(Xtrain, ytrain)

regressor = ln.LassoNetRegressorCV(cv=2, verbose=2, hidden_dims=(30,))
regressor.fit(Xtrain, ytrain)

# Xt, Xv, yt, yv = ms.train_test_split(Xtrain, ytrain, test_size=0.1, shuffle=True)
# inp = ks.layers.Input(shape=(Xt.shape[1],))
# gw = ks.layers.Dense(units=20, activation='relu')(inp)
# sl = ks.layers.Dense(units=4, activation='relu')(gw)
# output = ks.layers.Dense(units=1)(inp)

# # Implement early stopping
# early_stop = ks.callbacks.EarlyStopping(
#     monitor="val_loss",
#     min_delta=0,
#     patience=50,
#     verbose=0,
#     mode="auto",
#     baseline=None,

#     restore_best_weights=True,
#     start_from_epoch=0,
# )

# # Initial dense training
# regressor = ks.models.Model(inputs=inp, outputs=output)
# regressor.compile(optimizer=ks.optimizers.Adam(), loss=ks.losses.MeanSquaredError(), metrics=['mse'])
# regressor.fit(Xt, yt, validation_data=(Xv, yv), epochs=1000, callbacks=[early_stop], verbose=1)

ypred = regressor.predict(Xtest).ravel()
print(f"MSE: {mt.mean_squared_error(ytest, ypred):.6f}")
print(f"Only mean yields: {mt.mean_squared_error(ytest, np.full_like(ypred, np.mean(ytrain))):.6f}")
x_axis = list(range(len(ytest.ravel())))
sns.lineplot(x=x_axis, y=ytest.ravel(), color='black')
sns.lineplot(x=x_axis, y=ypred.ravel(), color='red')
sns.lineplot(x=x_axis, y=ytest.ravel() - ypred.ravel(), color='blue')
plt.show()
