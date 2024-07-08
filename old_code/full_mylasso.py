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

def train_dense_model(X_train, X_val, y_train, y_val, output_size, optimizer, loss_func, metrics, activation='relu', include_bias=True, neurons=[100], patience=100, epochs=1000, verbose=0, drop=0):
    inp = ks.layers.Input(shape=(X_train.shape[1],))
    skip = ks.layers.Dense(units=1, activation='linear', use_bias=include_bias, kernel_regularizer=ks.regularizers.L1L2(), name='skip_layer')(inp)
    dp = ks.layers.Dropout(drop)(inp)
    gw = ks.layers.Dense(units=neurons[0], activation=activation, name='gw_layer')(dp)

    if len(neurons) > 1:
        for K in neurons[1:]:
            gw = ks.layers.Dense(units=K, activation=activation)(gw)   

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
        restore_best_weights=True,
        start_from_epoch=0,
    )

    # Initial dense training
    nn = ks.models.Model(inputs=inp, outputs=output)
    nn.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
    nn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[early_stop], verbose=verbose)

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
    threshold = np.clip((np.repeat(np.abs(theta).reshape((-1, 1)), K, axis=1) + M * W_sum) - np.full_like(W_sum, l), 0, np.inf)
    w_m = (M * threshold) / (1 + m * (M**2)) 

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


def estimate_starting_lambda(theta, W, M, starting_lambda = 1e-3, factor = 2, tol = 1e-6, max_iter_per_lambda = 10000, verbose=False, divide_result = 1):
    initial_theta = theta
    dense_W = W
    dense_theta = initial_theta
    l_test = starting_lambda

    while not np.sum(np.abs(dense_theta)) == 0:
        dense_theta = initial_theta
        l_test = l_test * factor
        if verbose: print(f"Testing lambda={l_test}")

        for _ in range(max_iter_per_lambda):
            theta_new, _ = hier_prox(dense_theta, dense_W, l_test, M)
            if np.max(np.abs(dense_theta - theta_new)) < tol: break # Check if the theta is still changing
            dense_theta = theta_new

    return l_test / divide_result


def calc_investment_returns(forecast, real, allow_empty=False, start_val=1, trad_cost=0.0015):
    value = start_val
    path = np.zeros(len(real))
    prev_pos = 1

    lb = np.mean(forecast) - np.std(forecast)
    ub = np.mean(forecast) + np.std(forecast)

    for t, (f, r) in enumerate(zip(forecast, real)):
        pos = prev_pos
        if f < lb:
            pos = 1
        elif f > ub:
            pos = -1
        else:
            pos = 0 if allow_empty else prev_pos

        if pos != prev_pos: value = value * (1 - trad_cost)
        prev_pos = pos

        value = value * (1 + pos * r/100)
        path[t] = value

    return (value / start_val - 1, path)


def train_lasso_path(network, 
                     starting_lambda, 
                     X_train, 
                     X_val, 
                     y_train, 
                     y_val,
                     optimizer,
                     loss_func,
                     return_train = False,
                     use_faster_fit = True,
                     use_faster_eval = True,
                     use_best_weights = True,
                     train_until_k = 0,
                     lr=1e-3, 
                     M=10, 
                     pm=2e-2, 
                     max_epochs_per_lambda = 100, 
                     patience = 10, 
                     verbose=False,
                     early_val_stop=False,
                     use_invest_results=False):
    
    res_k = []
    res_theta = []
    res_isa = []
    res_val = []
    res_r = []
    l = starting_lambda / (1 + pm)
    k = X_train.shape[1]
    prev_obj = 0

    while k > train_until_k:
        l = (1 + pm) * l
        best_val_obj = np.inf
        e_since_best_val = 1
        train_time = 0
        prox_time = 0

        for b in range(max_epochs_per_lambda):
            start_train = perf_counter_ns()
            if use_faster_fit:
                with tf.GradientTape() as tape:
                    logits = network(X_train, training=True)
                    losses = loss_func(y_train, logits)
                gradients = tape.gradient(losses, network.trainable_weights)

                # Update theta and W using losses
                optimizer.apply_gradients(zip(gradients, network.trainable_weights))
            else:
                network.fit(X_train, y_train, verbose='0', epochs=1)

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

            if use_faster_eval:
                val_logits = network(X_val, training=False)
                val_obj = loss_func(y_val, val_logits)
            else:
                val_obj = network.evaluate(X_val, y_val, verbose='0')[1]

            train_time += perf_counter_ns() - start_train

            e_since_best_val += 1
            if val_obj < best_val_obj:
                best_val_obj = val_obj
                best_weights = network.get_weights()
                e_since_best_val = 1
            
            if e_since_best_val == patience:
                if verbose: print(f"Ran for {b+1} epochs before early stopping.")
                if use_best_weights: network.set_weights(best_weights)
                break

            if b == max_epochs_per_lambda - 1: 
                if verbose: print(f"Ran for the full {max_epochs_per_lambda} epochs.")
                if use_best_weights: network.set_weights(best_weights)

        print(f"Training time (ms): {train_time // 1e6}")
        print(f"Proximal time (ms): {prox_time // 1e6}")

        last_theta = network.get_layer('skip_layer').get_weights()[0]
        k = np.shape(np.nonzero(last_theta))[1]

        res_k.append(k)
        res_theta.append(last_theta)

        if return_train: res_isa.append(network.evaluate(X_train, y_train, verbose='0')[1])

        val_acc = network.evaluate(X_val, y_val)[1]
        res_val.append(val_acc)

        forecast = network.predict(X_val)
        val_r, _ = calc_investment_returns(forecast, y_val, trad_cost=0)
        res_r.append(val_r[0])
        
        print(f"--------------------------------------------------------------------- K = {k}, lambda = {l:.1f}, MSE = {val_acc:.6f}, r = {val_r[0]:.3f} \n\n")

        if k <  X_train.shape[1]/2 and val_acc < 0.5 * prev_obj and early_val_stop: 
            break
        else:
            prev_obj = val_acc  
    
    return (res_k, res_theta, res_val, res_isa, res_r)

def main() -> None:
    USE_BTC_DATA = False

    if USE_BTC_DATA:
        print("Hoi")
    else:
        # Dataset parameters
        dataset = "miceprotein"
        n_classes = 8
        calculate_out_of_sample_accuracy = False
        train_frac, val_frac, test_frac = 0.7, 0.1, 0.2

        # Load in the data
        X_full, y_full = skdata.fetch_openml(name=dataset, return_X_y=True)
        print("Loaded data.")

        X_full = imp.SimpleImputer().fit_transform(X_full)
        X_full = pp.StandardScaler().fit_transform(X_full)
        y_full = pp.LabelEncoder().fit_transform(y_full)
        print("Cleaned data.")

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_full, y_full, test_frac=test_frac, val_frac=val_frac)
        print("Split data.")

    # Dense network parameters
    bias            = False
    layer_size      = [10, 5] # int((3/3) * X_train.shape[1])
    max_epochs      = 1000
    dense_patience  = 100
    dense_opt       = ks.optimizers.Adam()
    loss            = ks.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics         = ['accuracy']

    # Sparse algorithm parameters
    estimate_lambda         = True
    fast_fit                = True
    fast_eval               = False
    use_best_weights        = False
    print_lambda_estimation = True
    print_sparsification    = True
    plot_lambda_path        = True

    n_features      = 0
    starting_lambda = 6.5
    sparse_patience = 10
    B               = 100
    M               = 10
    a               = 1e-3
    e               = 0.02

    sparse_opt = ks.optimizers.SGD(learning_rate=a, momentum=0.9)

    # Train the dense model
    nn = train_dense_model(X_train, X_val, y_train, y_val, n_classes, dense_opt, loss, metrics, neurons=layer_size, include_bias=bias, patience=dense_patience, epochs=max_epochs)
    if calculate_out_of_sample_accuracy: fm_result = nn.evaluate(X_test, y_test)

    # Recompile model for regularization path
    nn.compile(optimizer=sparse_opt, loss=loss, metrics=metrics)

    # Estime the starting value for lambda (if enabled)
    if estimate_lambda: starting_lambda = estimate_starting_lambda(nn.get_layer('skip_layer').get_weights()[0], nn.get_layer('gw_layer').get_weights()[0], M, verbose=print_lambda_estimation)

    # Train the LassoNet over the lambda path
    res_k, res_theta, res_val, res_isa = train_lasso_path(nn, starting_lambda, X_train, X_val, y_train, y_val, sparse_opt, loss, 
                                                          train_until_k=n_features, use_faster_fit=fast_fit, lr=a, M=M, pm=e, max_epochs_per_lambda=B, use_best_weights=use_best_weights,
                                                          patience=sparse_patience, verbose=print_sparsification, return_train=plot_lambda_path, use_faster_eval=fast_eval)

    # Plot accuracies at all points of the lasso path
    if plot_lambda_path:
        sns.lineplot(x=np.array(res_k), y=np.array(res_isa), markers=True)
        plt.title("IN SAMPLE PERFORMANCE")
        plt.show()
        sns.lineplot(x=np.array(res_k), y=np.array(res_val), markers=True)
        plt.title("VALIDATION PERFORMANCE")
        plt.show()

    # Retrain the model using only the selected features
    if calculate_out_of_sample_accuracy:
        final_theta = res_theta[-1] if np.sum(np.ravel(res_theta[-1] != 0)) == n_features else res_theta[-2]
        theta_mask = np.ravel(final_theta != 0)
        print(f"Selected {np.sum(theta_mask)} features.")

        X_train_f = X_train[:,theta_mask]
        X_val_f = X_val[:,theta_mask]
        X_test_f = X_test[:,theta_mask]

        final_nn = train_dense_model(X_train_f, X_val_f, y_train, y_val, n_classes, dense_opt, loss, metrics, neurons=layer_size, include_bias=bias, patience=dense_patience, epochs=max_epochs)
        print(X_train_f.shape)
        print(final_nn.get_layer('skip_layer').get_weights()[0].shape)

        fs_result = final_nn.evaluate(X_test_f, y_test)
        print(f"Final accuracy for the full model: {fm_result[1]:.3f}")
        print(f"Final accuracy for the LassoNet selected features: {fs_result[1]:.3f}")


if __name__ == '__main__':
    np.random.seed(1234)
    main()