import sklearn.datasets as skdata
import sklearn.impute as imp
import sklearn.preprocessing as pp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import lassonet as lsn
import keras as ks
import numpy as np
import tensorflow as tf

from lassonet_implementation import return_LassoNet_results, paper_lassonet_results, estimate_starting_lambda, train_lasso_path
from network_definitions import return_MLP_skip_classifier

CALCULATE = True
dataset = "isolet"
dataset = "miceprotein"

if CALCULATE:
    n_classes = 8 if dataset == "miceprotein" else 26
    X_full, y_full = skdata.fetch_openml(name=dataset, return_X_y=True)
    print(X_full.shape)
    print("Loaded data.")

    X_full = imp.SimpleImputer().fit_transform(X_full)
    X_full = pp.StandardScaler().fit_transform(X_full)

    y_full = pp.LabelEncoder().fit_transform(y_full)
    print("Cleaned data.")

    X_ftrain, X_test, y_ftrain, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=1234)
    X_train, X_val, y_train, y_val = train_test_split(X_ftrain, y_ftrain, test_size=0.1, random_state=2345)
    print("Split data.")

    K = 100 if dataset == "miceprotein" else 300
    M = 10
    pm= 0.02
    a = 0.001

    np.random.seed(1234)
    tf.random.set_seed(1234)
    ks.utils.set_random_seed(1234)

    # paper_LN = lsn.LassoNetClassifier(verbose=2, hidden_dims=(K,), path_multiplier=(1+pm), M=M, random_state=1234, torch_seed=1234, lambda_start=6, backtrack=True, tol=0.99, n_iters=(5000, 100))
    # paper_LN_path = paper_LN.path(X_train, y_train, X_val=X_val, y_val=y_val, return_state_dicts=True)
    # lsn.plot_path(paper_LN, paper_LN_path, X_test=X_test, y_test=y_test)
    # plt.savefig(f'plots/paper_{dataset}.eps', format='eps', bbox_inches='tight')
    # lsn.plot_path(paper_LN, paper_LN_path, X_test=X_test, y_test=y_test)
    # plt.savefig(f'plots/paper_{dataset}.png', format='png', bbox_inches='tight')

    dense = return_MLP_skip_classifier(X_train, X_val, y_train, y_val, n_classes, K=[K], epochs=5000, verbose=1, lr=a, activation='relu')
    # starting_lambda = estimate_starting_lambda(dense.get_layer('skip_layer').get_weights()[0], dense.get_layer('gw_layer').get_weights()[0], M, verbose=True, steps_back=3) / 0.001

    dense.compile(optimizer=ks.optimizers.SGD(learning_rate=a, momentum=0.9), loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    res_k, res_theta, res_val, res_l, oos, final_net = train_lasso_path(
        dense, 6, X_train, X_val, y_train, y_val, ks.optimizers.SGD(learning_rate=a, momentum=0.9), ks.losses.SparseCategoricalCrossentropy(from_logits=True), 
        train_until_k=0, use_faster_fit=True, lr=a, M=M, pm=pm, max_epochs_per_lambda=100, use_best_weights=True,
        patience=10, verbose=True, use_faster_eval=False, regressor=False, X_test=X_test, y_test=y_test, min_improvement=0.99)

    # np.save(f'paper_reproduction/{dataset}_oos', np.array(oos).ravel())
    # np.save(f'paper_reproduction/{dataset}_res_l', np.array(res_l).ravel())
    # np.save(f'paper_reproduction/{dataset}_res_k', np.array(res_k).ravel())

else:
    oos = np.load(f'paper_reproduction/{dataset}_oos.npy')
    l = np.load(f'paper_reproduction/{dataset}_res_l.npy')
    k = np.load(f'paper_reproduction/{dataset}_res_k.npy')

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 15),)
    #KMSE
    ax = plt.subplot(3, 1, 1)
    # sns.lineplot(x=k, y=oos, ax=ax, marker="o", markersize=3, linewidth=1)
    sns.lineplot(x=k, y=oos, ax=ax)
    ax.set_xlabel(f"selected features")
    ax.set_ylabel(f"accuracy")

    #LMSE
    ax = plt.subplot(3, 1, 2)
    # sns.lineplot(x=l, y=oos, ax=ax, marker="o", markersize=3, linewidth=1)
    sns.lineplot(x=l, y=oos, ax=ax)
    ax.set_xscale('log')
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(f"accuracy")

    #LK
    ax = plt.subplot(3, 1, 3)
    # sns.lineplot(x=l, y=k, ax=ax, marker="o", markersize=3, linewidth=1)
    sns.lineplot(x=l, y=k, ax=ax)
    ax.set_xscale('log')
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(f"selected features")

    plt.savefig(f'plots/myln_{dataset}.eps', format='eps', bbox_inches='tight')
    # plt.savefig(f'plots/myln_{dataset}.png', format='png', bbox_inches='tight')
    # plt.show()