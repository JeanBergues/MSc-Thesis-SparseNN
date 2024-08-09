import pandas as pd
import numpy as np
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as mt
import sklearn.linear_model as sklm

from data_loader import load_AR_data, load_data_with_X, scale_data

np.random.seed(1234)


###############################################################################################################################################################################################

def main():
    day_df = pd.read_csv(f'pct_btc_day.csv')
    hour_df = pd.read_csv(f'pct_btc_hour.csv')

    # Define the experiment parameters

    dlag_opt = [7]
    hlag_opt = [0]
    
    USE_X = False
    USE_SKIP = True

    EXPERIMENT_NAME = "final_forecasts/"
    EXPERIMENT_NAME += "SLASSO_" if USE_SKIP else "LASSO_"
    EXPERIMENT_NAME += "X_" if USE_X else ""

    # Begin the training
    for d_nlags in dlag_opt:
        for h_nlags in hlag_opt:
            np.random.seed(1234)
            EXPERIMENT_NAME += f"{d_nlags}_{h_nlags}"

            if USE_X:
                X_raw, y_raw = load_data_with_X(day_df, hour_df, d_nlags, h_nlags)
            else:
                X_raw, y_raw = load_AR_data(day_df, hour_df, d_nlags, h_nlags)
            X_scaled, y_scaled, X_pp, y_pp = scale_data(X_raw, y_raw)

            Xtrain, Xtest, ytrain, ytest = ms.train_test_split(X_scaled, y_scaled, test_size=365, shuffle=False)
            print("Data has been fully transformed and split")

            lm = sklm.LassoCV(cv=ms.TimeSeriesSplit(5), verbose=1, fit_intercept=True, max_iter=100_000, n_jobs=-1)
            lm.fit(Xtrain, ytrain)

            lasso_mask = np.ravel(lm.coef_ != 0)
            print(lasso_mask)
            n_selected = int(np.sum(lasso_mask))
            print(n_selected)

            test_forecast = lm.predict(Xtest)
            test_forecast = y_pp.inverse_transform(test_forecast.reshape(1, -1)).ravel()
            ytest = y_pp.inverse_transform(ytest.reshape(1, -1)).ravel()
            ytrain = y_pp.inverse_transform(ytrain.reshape(1, -1)).ravel()

            print(f"BEST TEST MSE = {mt.mean_squared_error(ytest, test_forecast):.3f}")
            print(f"Only mean MSE = {mt.mean_squared_error(ytest, np.full_like(ytest, np.mean(ytrain))):.3f}")

            Xtrain = Xtrain[:,lasso_mask]
            Xtest = Xtest[:,lasso_mask]
            

if __name__ == '__main__':
    main()