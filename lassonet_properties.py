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

from data_loader import load_AR_data, load_data_with_X, scale_data, load_data_with_XLOG
from lassonet_implementation import return_LassoNet_results, paper_lassonet_results
from network_definitions import return_MLP_skip_estimator

def results_plot(HP_opts, HP_results, use=(0, 1), title=r"$B$", name="", show=True, log_scale=False, format='eps', old_style=False, use_lowest_value=False):
    fig = plt.figure(figsize=(6, 3))
    labels = ["selected features", "MSE", r"$\lambda$"]
    for m, res in zip(HP_opts, HP_results):
        if old_style:
            fig = sns.lineplot(x=np.array(res[use[0]]), y=np.array(res[use[1]]), drawstyle='steps-pre', size=10)
        else:
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
        plt.savefig(f'lnprop_plots/NEWPROP_{name}.{format}', format=format, bbox_inches='tight')


###############################################################################################################################################################################################

day_df = pd.read_csv(f'pct_btc_day.csv')
hour_df = pd.read_csv(f'pct_btc_hour.csv')

freq = 24

dlag_opt = [2]
use_hlag = [0]
best_K = [50]

USE_X = True
USE_PAPER_LASSONET = False
SHOW = True
changing_hp = r"$\epsilon$"

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
        HP_opts = [0.02, 0.01, 0.005, 0.001]
        # HP_opts = [10]
        HP_results = []
        HP_new = []
        HP_val_results = []
        HP_val_new = []
        EXPERIMENT_NAME = "XSV_M10_K[50]_L[2_0]_B1000_pmVAR"
        
        if not USE_PAPER_LASSONET:
            initial_model = return_MLP_skip_estimator(Xt, Xv, yt, yv, activation='tanh', K=best_K, verbose=1, patience=100, epochs=20_000, drop=0, lr=0.01)
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
                    network, Xt, Xv, yt, yv, pm=hp, M=10, patiences=(100, 10), max_iters=(1000, 1000), print_path=True, print_lambda=True, starting_lambda=None, a=0.01, Xtest=Xtest, ytest=ytest, min_improvement=0.99, steps_back=5)
            
            HP_results.append((res_k, res_oos, res_l))
            HP_val_results.append((res_k, res_val, res_l))

            unique_k = np.unique(np.array(res_k))
            lowest_l_values = [np.min(np.array(res_l)[np.array(res_k) == k]) for k in unique_k]
            lowest_test_values = [np.min(np.array(res_oos)[np.array(res_k) == k]) for k in unique_k]
            lowest_val_values = [np.min(np.array(res_val)[np.array(res_k) == k]) for k in unique_k]

            HP_new.append((unique_k, lowest_test_values, lowest_l_values))
            HP_val_new.append((unique_k, lowest_val_values, lowest_l_values))

        # Plot selected features against mse
        # results_plot(HP_opts, HP_results, use=(0, 1), title=changing_hp, name=f"{EXPERIMENT_NAME}_KMSE", show=True)
        # results_plot(HP_opts, HP_results, use=(0, 1), title=changing_hp, name=f"{EXPERIMENT_NAME}_KMSE", show=False)
        # results_plot(HP_opts, HP_results, use=(2, 0), title=changing_hp, name=f"{EXPERIMENT_NAME}_LK", show=True, log_scale=True)
        # results_plot(HP_opts, HP_results, use=(2, 0), title=changing_hp, name=f"{EXPERIMENT_NAME}_LK", show=False, log_scale=True)
        # results_plot(HP_opts, HP_results, use=(2, 1), title=changing_hp, name=f"{EXPERIMENT_NAME}_LMSE", show=True, log_scale=True)
        # results_plot(HP_opts, HP_results, use=(2, 1), title=changing_hp, name=f"{EXPERIMENT_NAME}_LMSE", show=False, log_scale=True)

        # Fully old style
        results_plot(HP_opts, HP_results, use=(0, 1), title=changing_hp, name=f"{EXPERIMENT_NAME}_KMSE_OT", show=False, old_style=True)
        results_plot(HP_opts, HP_results, use=(2, 0), title=changing_hp, name=f"{EXPERIMENT_NAME}_LK_OT", show=False, log_scale=True, old_style=True)
        results_plot(HP_opts, HP_results, use=(2, 1), title=changing_hp, name=f"{EXPERIMENT_NAME}_LMSE_OT", show=False, log_scale=True, old_style=True)
        results_plot(HP_opts, HP_results, use=(0, 1), title=changing_hp, name=f"{EXPERIMENT_NAME}_KMSE_OT", show=False, old_style=True, format='png')
        results_plot(HP_opts, HP_results, use=(2, 0), title=changing_hp, name=f"{EXPERIMENT_NAME}_LK_OT", show=False, log_scale=True, old_style=True, format='png')
        results_plot(HP_opts, HP_results, use=(2, 1), title=changing_hp, name=f"{EXPERIMENT_NAME}_LMSE_OT", show=False, log_scale=True, old_style=True, format='png')
        # New data new style
        results_plot(HP_opts, HP_new, use=(0, 1), title=changing_hp, name=f"{EXPERIMENT_NAME}_KMSE_NT", show=False, old_style=False)
        results_plot(HP_opts, HP_new, use=(0, 1), title=changing_hp, name=f"{EXPERIMENT_NAME}_KMSE_NT", show=False, old_style=False, format='png')

        # Fully old style val
        results_plot(HP_opts, HP_val_results, use=(0, 1), title=changing_hp, name=f"{EXPERIMENT_NAME}_KMSE_OV", show=False, old_style=True)
        results_plot(HP_opts, HP_val_results, use=(2, 0), title=changing_hp, name=f"{EXPERIMENT_NAME}_LK_OV", show=False, log_scale=True, old_style=True)
        results_plot(HP_opts, HP_val_results, use=(2, 1), title=changing_hp, name=f"{EXPERIMENT_NAME}_LMSE_OV", show=False, log_scale=True, old_style=True)
        results_plot(HP_opts, HP_val_results, use=(0, 1), title=changing_hp, name=f"{EXPERIMENT_NAME}_KMSE_OV", show=False, old_style=True, format='png')
        results_plot(HP_opts, HP_val_results, use=(2, 0), title=changing_hp, name=f"{EXPERIMENT_NAME}_LK_OV", show=False, log_scale=True, old_style=True, format='png')
        results_plot(HP_opts, HP_val_results, use=(2, 1), title=changing_hp, name=f"{EXPERIMENT_NAME}_LMSE_OV", show=False, log_scale=True, old_style=True, format='png')
        # New data new style val
        results_plot(HP_opts, HP_val_new, use=(0, 1), title=changing_hp, name=f"{EXPERIMENT_NAME}_KMSE_NV", show=False, old_style=False)
        results_plot(HP_opts, HP_val_new, use=(0, 1), title=changing_hp, name=f"{EXPERIMENT_NAME}_KMSE_NV", show=False, old_style=False, format='png')