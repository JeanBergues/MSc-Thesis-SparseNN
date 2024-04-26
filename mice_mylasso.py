import numpy as np
import sklearn.datasets as skdata
import sklearn.impute as imp
import sklearn.preprocessing as pp
import sklearn.metrics as met
import keras as ks
import tensorflow as tf
from sklearn.model_selection import train_test_split
import my_hierprox


def main() -> None:
    X_full, y_full = skdata.fetch_openml(name="miceprotein", return_X_y=True)
    print(X_full.shape)
    print("Loaded data.")

    X_full = imp.SimpleImputer().fit_transform(X_full)
    X_full = pp.MinMaxScaler().fit_transform(X_full)

    y_full = pp.LabelEncoder().fit_transform(y_full)
    print("Cleaned data.")

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2)
    print("Split data.")

    # Starting model, optimized with Adam
    inp = ks.layers.Input(shape=(X_train.shape[1],))
    skip = ks.layers.Dense(units=1, activation='linear')(inp)
    gw = ks.layers.Dense(units=10, activation='relu')(inp)
    merge = ks.layers.Concatenate()([skip, gw])
    output = ks.layers.Dense(units=8)(merge)

    # Initial dense training
    nn = ks.models.Model(inputs=inp, outputs=output)
    nn.compile(optimizer='adam', loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    nn.fit(X_train, y_train, epochs=1)

    # Recompile to SGD solver for regularization path
    nn.compile(optimizer='sgd', loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    # Start Algorithm 1
    B = 10
    M = 10
    eps = 0.2
    alpha = 0.01
    k = X_train.shape[1]
    l = 1
    
    while k > 0:
        l = (1 + eps) * l
        for b in range(B):
            nn.fit(X_train, y_train, epochs=1)
            weights = nn.get_weights()
            theta_new, W_new = my_hierprox.hier_prox(weights[0], weights[2], alpha*l, M)
            weights[0] = theta_new
            weights[2] = W_new
            nn.set_weights(weights)
        k = np.shape(np.nonzero(weights[0]))[1]
        print(f"\n\n K = {k}")
        print(np.nonzero(weights[0]))
        print()
        print(weights[0])

if __name__ == '__main__':
    main()