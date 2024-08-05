import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import keras as ks
import sklearn.model_selection as ms
import torch as pt

np.random.seed(1234)
tf.random.set_seed(1234)
ks.utils.set_random_seed(1234)
pt.manual_seed(1234)

from data_loader import load_AR_data, load_data_with_X, scale_data
from lassonet_implementation import return_LassoNet_results, paper_lassonet_results
from network_definitions import return_MLP_skip_estimator

def results_plot(HP_opts, HP_results, use=(0, 1), title=r"$B$", name="", show=True, log_scale=False):
    fig = plt.figure(figsize=(6, 3))
    labels = ["selected features", "MSE", r"$\lambda$"]
    for m, res in zip(HP_opts, HP_results):
        fig = sns.lineplot(x=np.array(res[use[0]]), y=np.array(res[use[1]]), drawstyle='steps-pre', size=10)
    
    legd = fig.get_legend()
    for t, l in zip(legd.texts, HP_opts):
        t.set_text(title + f"={l}")

    sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))
    if log_scale: plt.xscale('log')
    plt.xlabel(labels[use[0]])
    plt.ylabel(labels[use[1]])
    if show:
        plt.show()
    else:
        plt.savefig(f'plots/{name}.eps', format='eps', bbox_inches='tight')


###############################################################################################################################################################################################

day_df = pd.read_csv(f'pct_btc_day.csv')
hour_df = pd.read_csv(f'pct_btc_hour.csv')

freq = 24

dlag_opt = [2]
use_hlag = [0]

best_K = [100]

USE_X = True
USE_PAPER_LASSONET = False
SHOW = True

for d_nlags in dlag_opt:
    for h_nlags in use_hlag:
        np.random.seed(1234)
        tf.random.set_seed(1234)
        ks.utils.set_random_seed(1234)

        if USE_X:
            X_raw, y_raw = load_data_with_X(day_df, hour_df, d_nlags, h_nlags)
        else:
            X_raw, y_raw = load_AR_data(day_df, hour_df, d_nlags, h_nlags)
        X_scaled, y_scaled, X_pp, y_pp = scale_data(X_raw, y_raw)

        Xtrain, Xtest, ytrain, ytest = ms.train_test_split(X_scaled, y_scaled, test_size=365, shuffle=False)
        print("Data has been fully transformed and split")

        n_repeats = 1
        # Xt, Xv, yt, yv = Xtrain, Xtest, ytrain, ytest
        #ytest = y_pp.inverse_transform(ytest.reshape(1, -1)).ravel()

        # best_K = [200, 100]
        Xt, Xv, yt, yv = ms.train_test_split(Xtrain, ytrain, test_size=30, shuffle=False)

        # Run for M variations
        HP_opts = [10]
        HP_results = []
        EXPERIMENT_NAME = "NEW_CRIT_TEST_M_K100-20"
        
        if not USE_PAPER_LASSONET:
            initial_model = return_MLP_skip_estimator(Xt, Xv, yt, yv, activation='tanh', K=best_K, verbose=1, patience=100, epochs=1000, drop=0, lr=0.01)
            initial_model.save('temp_network.keras')
            initial_model_best_weights = initial_model.get_weights()
            initial_model.save_weights('temp_weights.weights.h5')

        for hp in HP_opts:
            np.random.seed(1234)
            tf.random.set_seed(1234)
            ks.utils.set_random_seed(1234)
            pt.manual_seed(1234)

            if USE_PAPER_LASSONET:
                res_k, _, res_val, res_l = paper_lassonet_results( 
                    Xt, Xv, yt, yv, K=tuple(best_K), verbose=2, pm=0.02, M=hp, patiences=(100, 10), max_iters=(10000, 1000), l_start='auto', use_custom_optimizer=True)
            else:
                network = ks.models.load_model('temp_network.keras')
                network.set_weights(initial_model_best_weights)
                network.compile(optimizer=ks.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss=ks.losses.MeanSquaredError())
                network.load_weights('temp_weights.weights.h5')
                res_k, _, res_val, res_l, res_oos = return_LassoNet_results(
                    network, Xt, Xv, yt, yv, pm=0.02, M=hp, patiences=(100, 10), max_iters=(1000, 100), print_path=True, print_lambda=True, starting_lambda=None, a=0.01, Xtest=Xtest, ytest=ytest, min_improvement=0.99, steps_back=2)
                # res_k, res_val, res_l = return_LassoNet_mask(
                #     initial_model, tXt, tXv, tyt, tyv, K=best_K, pm=hp, M=10, patiences=(100, 10), max_iters=(10000, 1000), print_path=True, print_lambda=True, starting_lambda=13)
            
            HP_results.append((res_k, res_oos, res_l))

        # Plot selected features against mse
        results_plot(HP_opts, HP_results, use=(0, 1), title=r"$M$", name=f"{EXPERIMENT_NAME}_KMSE", show=SHOW)
        results_plot(HP_opts, HP_results, use=(2, 0), title=r"$M$", name=f"{EXPERIMENT_NAME}_LK", show=SHOW, log_scale=True)
        results_plot(HP_opts, HP_results, use=(2, 1), title=r"$M$", name=f"{EXPERIMENT_NAME}_LMSE", show=SHOW, log_scale=True)