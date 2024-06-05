import numpy as np
import sklearn.datasets as skdata
import sklearn.impute as imp
import sklearn.preprocessing as pp
import sklearn.metrics as met
import tensorflow as tf
import keras as ks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from time import perf_counter_ns

def train_dense_model(X_train, X_val, y_train, y_val, output_size, include_bias=True, neurons=100, patience=100, epochs=1000):
    inp = ks.layers.Input(shape=(X_train.shape[1],))
    skip = ks.layers.Dense(units=1, activation='linear', use_bias=include_bias, name='skip_layer')(inp)
    gw = ks.layers.Dense(units=neurons, activation='relu', name='gw_layer')(inp)
    # rnn = ks.layers.Dense(units=10)(gw)
    merge = ks.layers.Concatenate()([skip, gw])
    output = ks.layers.Dense(units=output_size)(merge)

    # Implement early stopping
    early_stop = ks.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=patience,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
    )

    # Initial dense training
    nn = ks.models.Model(inputs=inp, outputs=output)
    nn.compile(optimizer=ks.optimizers.Adam(), loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    nn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[early_stop])

    return nn


def split_data(X, y, test_frac=0.2, val_frac=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac)
    X_trainv, X_val, y_trainv, y_val = train_test_split(X_train, y_train, test_size=val_frac/(1-test_frac))

    return (X_trainv, X_val, X_test, y_trainv, y_val, y_test)


def hier_prox(theta: np.ndarray, W: np.ndarray, l: float, M: float) -> tuple[np.ndarray, np.ndarray]:
    # Assert correct sizes
    theta = theta.ravel()
    d = theta.shape[0]
    K = W.shape[1]
    assert W.shape[0] == d

    # Order the weights
    sorted_W = -np.sort(-np.abs(W))

    # Calculate w_m's
    W_sum = np.cumsum(sorted_W, axis=1)
    m = np.arange(start=1, stop=K+1)
    threshold = np.clip(np.repeat(np.abs(theta).reshape((-1, 1)), K, axis=1) + M * W_sum - np.full_like(W_sum, l), 0, np.inf)
    w_m = M / (1 + m * (M**2)) * threshold

    # Check for condition
    m_tilde_condition = np.logical_and(w_m <= sorted_W, w_m >= np.concatenate((sorted_W, np.zeros((d, 1))), axis=1)[:,1:])

    # Find the first true value per row
    m_tilde_first_only = np.zeros_like(m_tilde_condition, dtype=bool)
    idx = np.arange(len(m_tilde_condition)), m_tilde_condition.argmax(axis=1)
    m_tilde_first_only[idx] = m_tilde_condition[idx]

    # Set the first value of each row to true if all other values in the row are false
    set_first_true_array = np.full_like(m_tilde_first_only, False)
    set_first_true_array[:,0] = np.sum(m_tilde_first_only, axis=1) < 1
    m_tilde_first_only = np.logical_or(m_tilde_first_only, set_first_true_array)
    m_tilde = w_m[m_tilde_first_only]

    # Calculate output
    theta_out = (1/M) * np.sign(theta) * m_tilde
    W_out = np.sign(W) * np.minimum(np.abs(W), np.repeat(m_tilde.reshape((-1, 1)), K, axis=1))

    return (theta_out, W_out)


def estimate_starting_lambda(theta, W, M, starting_lambda = 1e-6, factor = 2, tol = 1e-5, max_iter_per_lambda = 10000, verbose=False):
    initial_theta = theta
    dense_W = W
    dense_theta = initial_theta
    l_test = starting_lambda

    while not np.sum(dense_theta) == 0:
        dense_theta = initial_theta
        l_test = l_test * factor
        if verbose: print(f"Testing lambda={l_test}")

        for _ in range(max_iter_per_lambda):
            theta_new, _ = hier_prox(dense_theta, dense_W, l_test, M)
            if np.max(np.abs(dense_theta - theta_new)) < tol: break # Check if the theta is still changing
            dense_theta = theta_new

    return l_test / 10


def train_lasso_path(network, 
                     starting_lambda, 
                     X_train, 
                     X_val, 
                     y_train, 
                     y_val, 
                     return_train = False,
                     train_until_k = 0,
                     lr=1e-3, 
                     M=10, 
                     pm=2e-2, 
                     max_epochs_per_lambda = 100, 
                     patience = 10, 
                     verbose=False):
    
    res_k = []
    res_theta = []
    res_isa = []
    res_val = []
    l = starting_lambda / (1 + pm)
    k = X_train.shape[1]

    while k > train_until_k:
        l = (1 + pm) * l if k >= train_until_k else (1 - pm) * l
        best_val_obj = np.inf
        e_since_best_val = 1
        train_time = 0
        prox_time = 0

        for b in range(max_epochs_per_lambda):
            start_train = perf_counter_ns()
            # network.fit(X_train, y_train, verbose='0', epochs=1)
            # Compute gradient of loss
            with tf.GradientTape() as tape:
                logits = network(X_train, training=True)
                losses = ks.losses.SparseCategoricalCrossentropy(from_logits=True)(y_train, logits)
            gradients = tape.gradient(losses, network.trainable_weights)

            # Update theta and W using losses
            ks.optimizers.SGD(learning_rate=1e-3, momentum=0.9).apply_gradients(zip(gradients, network.trainable_weights))
            train_time += perf_counter_ns() - start_train
            
            # Update using HIER-PROX
            start_prox = perf_counter_ns()
            theta_new, W_new = hier_prox(network.get_layer('skip_layer').get_weights()[0], network.get_layer('gw_layer').get_weights()[0], lr*l, M)

            new_skip_layer = network.get_layer('skip_layer').get_weights()
            new_skip_layer[0] = theta_new.reshape((-1, 1))
            network.get_layer('skip_layer').set_weights(new_skip_layer)

            new_gw_layer = network.get_layer('gw_layer').get_weights()
            new_gw_layer[0] = W_new
            network.get_layer('gw_layer').set_weights(new_gw_layer)

            prox_time += perf_counter_ns() - start_prox

            start_train = perf_counter_ns()
            # val_obj = network.evaluate(X_val, y_val, verbose='0')[0]

            val_logits = network(X_val, training=False)
            val_obj = ks.losses.SparseCategoricalCrossentropy(from_logits=True)(y_val, val_logits)

            train_time += perf_counter_ns() - start_train

            e_since_best_val += 1
            if val_obj < best_val_obj:
                best_val_obj = val_obj
                e_since_best_val = 1
            
            if e_since_best_val == patience:
                if verbose: print(f"Ran for {b+1} epochs before early stopping.")
                break

            if b == max_epochs_per_lambda - 1 and verbose: print(f"Ran for the full {max_epochs_per_lambda} epochs.")

        print(f"Training time (ms): {train_time // 1e6}")
        print(f"Proximal time (ms): {prox_time // 1e6}")

        last_theta = network.get_layer('skip_layer').get_weights()[0]
        k = np.shape(np.nonzero(last_theta))[1]
        res_k.append(k)
        res_theta.append(last_theta)

        if return_train: res_isa.append(network.evaluate(X_train, y_train, verbose='0')[1])
        val_acc = network.evaluate(X_val, y_val)[1]
        res_val.append(val_acc)
        print(f"--------------------------------------------------------------------- K = {k}, lambda = {l:.3f}, accuracy = {val_acc:.3f} \n\n")
    
    return (res_k, res_theta, res_val, res_isa)

def main() -> None:
    # Dataset parameters
    dataset = "isolet"
    n_classes = 26
    calculate_out_of_sample_accuracy = True
    train_frac, val_frac, test_frac = 0.7, 0.1, 0.2

    # Dense network parameters
    bias = False
    layer_size = 100 #int((2/3) * 617)
    max_epochs = 1000
    dense_patience = 100

    # Sparse algorithm parameters
    n_features = 50
    starting_lambda = 3.5
    estimate_lambda = True
    sparse_patience = 10
    print_lambda_estimation = True
    print_sparsification = True

    B = 100
    M = 10
    a = 1e-3
    e = 0.02

    # Load in the data
    X_full, y_full = skdata.fetch_openml(name=dataset, return_X_y=True)
    print("Loaded data.")

    X_full = imp.SimpleImputer().fit_transform(X_full)
    X_full = pp.StandardScaler().fit_transform(X_full)
    y_full = pp.LabelEncoder().fit_transform(y_full)
    print("Cleaned data.")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_full, y_full, test_frac=test_frac, val_frac=val_frac)
    print("Split data.")

    nn = train_dense_model(X_train, X_val, y_train, y_val, n_classes, neurons=layer_size, include_bias=bias, patience=dense_patience, epochs=max_epochs)
    if calculate_out_of_sample_accuracy: fm_result = nn.evaluate(X_test, y_test)

    nn.compile(optimizer=ks.optimizers.SGD(learning_rate=1e-3, momentum=0.9), loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    if estimate_lambda: starting_lambda = estimate_starting_lambda(nn.get_layer('skip_layer').get_weights()[0], nn.get_layer('gw_layer').get_weights()[0], M, verbose=print_lambda_estimation)

    res_k, res_theta, res_val, res_isa = train_lasso_path(nn, starting_lambda, X_train, X_val, y_train, y_val, train_until_k=n_features, lr=a, M=M, pm=e, max_epochs_per_lambda=B, patience=sparse_patience, verbose=print_sparsification)

    if calculate_out_of_sample_accuracy:
        final_theta = res_theta[-1] if res_theta[-1].shape[0] == n_features else res_theta[-2]
        theta_mask = np.ravel(final_theta != 0)

        X_train_f = X_train[:,theta_mask]
        X_val_f = X_val[:,theta_mask]
        X_test_f = X_test[:,theta_mask]


        final_nn = train_dense_model(X_train_f, X_val_f, y_train, y_val, n_classes, neurons=layer_size, include_bias=bias, patience=dense_patience, epochs=max_epochs)

        fs_result = final_nn.evaluate(X_test_f, y_test)
        print(f"Final accuracy for the full model: {fm_result[1]:.3f}")
        print(f"Final accuracy for the LassoNet selected features: {fs_result[1]:.3f}")


if __name__ == '__main__':
    np.random.seed(1234)
    main()