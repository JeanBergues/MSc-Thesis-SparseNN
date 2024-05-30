import numpy as np
import sklearn.datasets as skdata
import sklearn.impute as imp
import sklearn.preprocessing as pp
import keras as ks
from sklearn.model_selection import train_test_split
import my_hierprox
import matplotlib.pyplot as plt
import seaborn as sns


def main() -> None:
    X_full, y_full = skdata.fetch_openml(name="isolet", return_X_y=True)
    print(X_full.shape)
    print("Loaded data.")

    # X_full = imp.SimpleImputer().fit_transform(X_full)
    X_full = pp.StandardScaler().fit_transform(X_full)

    y_full = pp.LabelEncoder().fit_transform(y_full)
    print("Cleaned data.")

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2)
    X_trainv, X_val, y_trainv, y_val = train_test_split(X_train, y_train, test_size=0.125)
    print("Split data.")

    # Starting model, optimized with Adam
    inp = ks.layers.Input(shape=(X_train.shape[1],))
    skip = ks.layers.Dense(units=1, activation='linear', use_bias=False, kernel_regularizer='l1_l2')(inp)
    gw = ks.layers.Dense(units=100, activation='relu')(inp)
    merge = ks.layers.Concatenate()([skip, gw])
    output = ks.layers.Dense(units=26)(merge)

    # Implement early stopping
    early_stop = ks.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=100,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
    )

    # Initial dense training
    nn = ks.models.Model(inputs=inp, outputs=output)
    nn.compile(optimizer=ks.optimizers.Adam(), loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    nn.fit(X_trainv, y_trainv, validation_data=(X_val, y_val), epochs=1000, callbacks=[early_stop])

    # Recompile to SGD solver for regularization path
    nn.compile(optimizer=ks.optimizers.SGD(learning_rate=1e-3, momentum=0.9), loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    # Start Algorithm 1
    res_k = []
    res_acc = []
    res_isa = []
    B = 100
    M = 10
    eps = 0.02
    alpha = 1e-3
    k = X_train.shape[1]

    # Estimate when the model starts to sparsify
    l_test = 1e-6
    factor = 2
    tolerance = 1e-5
    dense_weights = nn.get_weights()
    dense_theta = dense_weights[0]
    dense_W = dense_weights[1]

    while not np.linalg.norm(dense_theta, ord=2) == 0:
        l_test = l_test * factor
        print(f"Testing lambda={l_test}")
        for _ in range(10000):
            theta_new, W_new = my_hierprox.hier_prox(dense_theta, dense_W, l_test, 10)
            if np.max(np.abs(dense_theta - theta_new)) < tolerance: break
            dense_theta = theta_new

    # l_test = l_test / factor
    print(f"Sparsify started at {l_test}")
    l = (l_test / eps) / 10
    # l = 6.5
    
    while k > 0:
        l = (1 + eps) * l
        best_val_obj = np.inf
        e_since_best_val = 0

        for b in range(B):
            # Compute gradient of loss
            # with tf.GradientTape() as tape:
            #     logits = nn(X_trainv, training=True)
            #     losses = loss_function(y_trainv, logits)
            # gradients = tape.gradient(losses, nn.trainable_weights)

            # # Update theta and W using losses
            # optimizer.apply_gradients(zip(gradients, nn.trainable_weights))
            # metric.update_state(y_trainv, logits)
            nn.fit(X_trainv, y_trainv, verbose='0')

            # Update using HIER-PROX
            weights = nn.get_weights()
            theta_new, W_new = my_hierprox.hier_prox(weights[0], weights[1], alpha*l, M)
            weights[0] = theta_new
            weights[1] = W_new
            nn.set_weights(weights)

            val_obj = nn.evaluate(X_val, y_val, verbose='0')[0]
            e_since_best_val += 1
            if val_obj < best_val_obj:
                best_val_obj = val_obj
                e_since_best_val = 0
            
            if e_since_best_val == 10:
                print(f"Ran for {b+1} epochs before early stopping.")
                break

        k = np.shape(np.nonzero(weights[0]))[1]
        res_k.append(k)
        res_acc.append(nn.evaluate(X_test, y_test)[1])
        res_isa.append(nn.evaluate(X_trainv, y_trainv)[1])
        print(f"\n\n --------------------------------------------------------------------- K = {k}, lambda = {l}, accuracy = {nn.evaluate(X_val, y_val)[1]}")
        # print(np.nonzero(weights[0]))
        # print()
        # print(weights[0])

    sns.lineplot(x=res_k, y=res_acc)
    plt.show()
    sns.lineplot(x=res_k, y=res_isa)
    plt.show()

if __name__ == '__main__':
    main()