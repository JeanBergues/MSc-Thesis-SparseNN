import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter_ns
import keras as ks
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as mt
import lassonet as lsn
import torch as pt
from functools import partial

np.random.seed(1234)
tf.random.set_seed(1234)
ks.utils.set_random_seed(1234)
pt.manual_seed(1234)

from data_loader import load_AR_data, load_data_with_X, scale_data
from lassonet_implementation import return_LassoNet_results, paper_lassonet_results
from network_definitions import return_MLP_skip_estimator

def results_plot(HP_opts, HP_results, use=(0, 1), title=r"$B$", name="", show=True):
    fig = plt.figure(figsize=(6, 3))
    labels = ["selected features", "MSE", r"$\lambda$"]
    for m, res in zip(HP_opts, HP_results):
        fig = sns.lineplot(x=np.array(res[use[0]]), y=np.array(res[use[1]]), drawstyle='steps-pre', size=10)
    
    # plt.legend(labels=[f"M={l}" for l in HP_opts])
    legd = fig.get_legend()
    for t, l in zip(legd.texts, HP_opts):
        t.set_text(title + f"={l}")

    sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))
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

# raw_returns = day_df.close.pct_change(1)[1:].to_numpy()
open_returns =  day_df.open.to_numpy()
high_returns =  day_df.high.to_numpy()
low_returns =   day_df.low.to_numpy()
close_returns = day_df.close.to_numpy()
vol_returns =   day_df.volume.to_numpy()
volNot_returns =day_df.volumeNotional.to_numpy()
trades_returns =day_df.tradesDone.to_numpy()

open_h_returns =  hour_df.open.to_numpy()
high_h_returns =  hour_df.high.to_numpy()
low_h_returns =   hour_df.low.to_numpy()
close_h_returns = hour_df.close.to_numpy()
vol_h_returns =   hour_df.volume.to_numpy()
volNot_h_returns =hour_df.volumeNotional.to_numpy()
trades_h_returns =hour_df.tradesDone.to_numpy()

dlag_opt = [7]
use_hlag = [24]

best_K = [200, 100, 50]

USE_X = False
USE_PAPER_LASSONET = True
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
        ytest = y_pp.inverse_transform(ytest.reshape(1, -1)).ravel()

        # best_K = [200, 100]

        Xt, Xv, yt, yv = ms.train_test_split(Xtrain, ytrain, test_size=120, shuffle=False)
        tXt = tf.convert_to_tensor(Xt)
        tXv = tf.convert_to_tensor(Xv)
        tyt = tf.convert_to_tensor(yt)
        tyv = tf.convert_to_tensor(yv)

        # Run for M variations
        HP_opts = [100]
        HP_results = []
        EXPERIMENT_NAME = "LN_B"

        
        if not USE_PAPER_LASSONET:
            initial_model = return_MLP_skip_estimator(tXt, tXv, tyt, tyv, ksize=Xt.shape[1], activation='relu', K=best_K, verbose=1, patience=100, epochs=1000, drop=0, lr=0.01)
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
                    Xt, Xv, yt, yv, K=tuple(best_K), verbose=2, pm=0.02, M=20, patiences=(100, 10), max_iters=(10000, 100), l_start='auto', use_custom_optimizer=True)
            else:
                network = ks.models.load_model('temp_network.keras')
                network.set_weights(initial_model_best_weights)
                network.compile(optimizer=ks.optimizers.SGD(learning_rate=0.01, momentum=0.3), loss=ks.losses.MeanSquaredError())
                network.load_weights('temp_weights.weights.h5')
                res_k, _, res_val, res_l = return_LassoNet_results(
                    initial_model, Xt, Xv, yt, yv, K=best_K, pm=0.02, M=20, patiences=(100, 10), max_iters=(1000, 100), print_path=True, print_lambda=True, starting_lambda=None, a=0.1)
                # res_k, res_val, res_l = return_LassoNet_mask(
                #     initial_model, tXt, tXv, tyt, tyv, K=best_K, pm=hp, M=10, patiences=(100, 10), max_iters=(10000, 1000), print_path=True, print_lambda=True, starting_lambda=13)
            
            HP_results.append((res_k, res_val, res_l))

        # Plot selected features against mse
        results_plot(HP_opts, HP_results, use=(0, 1), title=r"$B$", name=f"{EXPERIMENT_NAME}_KMSE", show=SHOW)
        results_plot(HP_opts, HP_results, use=(2, 0), title=r"$B$", name=f"{EXPERIMENT_NAME}_LK", show=SHOW)
        results_plot(HP_opts, HP_results, use=(2, 1), title=r"$B$", name=f"{EXPERIMENT_NAME}_LMSE", show=SHOW)

        # fig = plt.figure(figsize=(6, 3))
        # for m, res in zip(HP_opts, HP_results):
        #     fig = sns.lineplot(x=np.array(res[0]), y=np.array(res[1]), drawstyle='steps-pre', size=10)
        
        # # plt.legend(labels=[f"M={l}" for l in HP_opts])
        # legd = fig.get_legend()
        # for t, l in zip(legd.texts, HP_opts):
        #     t.set_text(r"$B$" + f"={l}")

        # sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))
        # plt.xlabel("selected features")
        # plt.ylabel("mse")
        # # plt.savefig(f'plots/{EXPERIMENT_NAME}_KMSE.eps', format='eps', bbox_inches='tight')
        # # plt.savefig(f'plots/{EXPERIMENT_NAME}_KMSE.png', format='png', bbox_inches='tight')
        # plt.show()

        # # Plot selected features against lambda
        # fig = plt.figure(figsize=(6, 3))
        # for m, res in zip(HP_opts, HP_results):
        #     fig = sns.lineplot(x=np.array(res[2]), y=np.array(res[0]), drawstyle='steps-pre', size=10)
        
        # # plt.legend(labels=[f"M={l}" for l in HP_opts])
        # legd = fig.get_legend()
        # for t, l in zip(legd.texts, HP_opts):
        #     t.set_text(r"$B$" + f"={l}")

        # sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))
        # plt.xlabel(r'$\lambda$')
        # plt.ylabel("selected features")
        # # plt.savefig(f'plots/{EXPERIMENT_NAME}_LK.eps', format='eps', bbox_inches='tight')
        # # plt.savefig(f'plots/{EXPERIMENT_NAME}_LK.png', format='png', bbox_inches='tight')
        # plt.show()

        # # Plot mse against lambda
        # fig = plt.figure(figsize=(6, 3))
        # for m, res in zip(HP_opts, HP_results):
        #     fig = sns.lineplot(x=np.array(res[2]), y=np.array(res[1]), drawstyle='steps-pre', size=10)
        
        # # plt.legend(labels=[f"M={l}" for l in HP_opts])
        # legd = fig.get_legend()
        # for t, l in zip(legd.texts, HP_opts):
        #     t.set_text(r"$B$" + f"={l}")

        # sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))
        # plt.xlabel(r'$\lambda$')
        # plt.ylabel("mse")
        # # plt.savefig(f'plots/{EXPERIMENT_NAME}_LMSE.eps', format='eps', bbox_inches='tight')
        # # plt.savefig(f'plots/{EXPERIMENT_NAME}_LMSE.png', format='png', bbox_inches='tight')
        # plt.show()