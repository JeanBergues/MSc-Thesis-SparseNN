import sklearn.datasets as skdata
import sklearn.impute as imp
import sklearn.preprocessing as pp
from sklearn.model_selection import train_test_split
import hisel as hl
import keras as ks


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


def main() -> None:
    X_full, y_full = skdata.fetch_openml(name="isolet", return_X_y=True)
    print(X_full.shape)
    print("Loaded data.")

    X_full = imp.SimpleImputer().fit_transform(X_full)
    X_full = pp.MinMaxScaler().fit_transform(X_full)

    y_full = pp.LabelEncoder().fit_transform(y_full)
    print("Cleaned data.")

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, shuffle=False)
    X_trainv, X_val, y_trainv, y_val = train_test_split(X_train, y_train, test_size=0.125, shuffle=False)
    print("Split data.")

    selector = hl.select.HSICSelector(X_train, y_train.reshape((-1, 1)), xfeattype=hl.select.FeatureType.CONT, yfeattype=hl.select.FeatureType.DISCR)
    selected_features = selector.select(50, batch_size=1000, number_of_epochs=5, return_index=True)

    print('\n\n##########################################################')
    print(
        f'The following features are relevant for the prediction of y:')
    print(f'{selected_features}')



    X_train_f = X_train[:,selected_features]
    X_val_f = X_val[:,selected_features]
    X_test_f = X_test[:,selected_features]


    final_nn = train_dense_model(X_train_f, X_val_f, y_train, y_val, 26, neurons=int(617 * 1/3), include_bias=False, patience=100, epochs=1000)

    fs_result = final_nn.evaluate(X_test_f, y_test)
    # print(f"Final accuracy for the full model: {fm_result[1]:.3f}")
    print(f"Final accuracy for the HSIC selected features: {fs_result[1]:.3f}")

if __name__ == '__main__':
    main()