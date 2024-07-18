import keras as ks

def return_MLP_estimator(Xt, Xv, yt, yv, K=[100], activation='relu', epochs=20_000, patience=100, verbose=0, drop=0, use_L1=False, es_tol=0, lr=1e-1):
    inp = ks.layers.Input(shape=(Xt.shape[1],))

    if use_L1:
        gw = ks.layers.Dense(units=K[0], activation=activation, kernel_regularizer=ks.regularizers.L1(), name='gw_layer')(inp)
    else:
        gw = ks.layers.Dense(units=K[0], activation=activation, name='gw_layer')(inp)
    gw = ks.layers.Dropout(drop)(gw)

    if len(K) > 1:
        for k in K[1:]:
            gw = ks.layers.Dense(units=k, activation=activation)(gw)   

    output = ks.layers.Dense(units=1)(gw)

    # Implement early stopping
    early_stop = ks.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=es_tol,
        patience=patience,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        # start_from_epoch=0,
    )

    # Initial dense training
    nn = ks.models.Model(inputs=inp, outputs=output)
    nn.compile(optimizer=ks.optimizers.Adam(lr), loss=ks.losses.MeanSquaredError())
    nn.fit(Xt, yt, validation_data=(Xv, yv), epochs=epochs, callbacks=[early_stop], verbose=verbose)

    return nn


def return_MLP_skip_estimator(Xt, Xv, yt, yv, K=[100], activation='relu', epochs=20_000, patience=100, verbose=0, drop=0, use_L1=False, es_tol=0, lr=1e-1):
    inp = ks.layers.Input(shape=(Xt.shape[1],))

    if use_L1:
        skip = ks.layers.Dense(units=1, activation='linear', use_bias=False, kernel_regularizer=ks.regularizers.L1(), name='skip_layer')(inp)
    else:
        skip = ks.layers.Dense(units=1, activation='linear', use_bias=False, name='skip_layer')(inp)

    gw = ks.layers.Dropout(drop)(inp)
    gw = ks.layers.Dense(units=K[0], activation=activation, name='gw_layer')(inp)

    if len(K) > 1:
        for k in K[1:]:
            gw = ks.layers.Dropout(drop)(gw)
            gw = ks.layers.Dense(units=k, activation=activation)(gw)   

    last_node = ks.layers.Dense(units=1)(gw)
    output = ks.layers.Add()([skip, last_node])

    # Implement early stopping
    early_stop = ks.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=es_tol,
        patience=patience,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        # start_from_epoch=0,
    )

    # Initial dense training
    nn = ks.models.Model(inputs=inp, outputs=output)
    nn.compile(optimizer=ks.optimizers.Adam(lr), loss=ks.losses.MeanSquaredError())
    nn.fit(Xt, yt, validation_data=(Xv, yv), epochs=epochs, callbacks=[early_stop], verbose=verbose)

    return nn

def return_MLP_skip_classifier(Xt, Xv, yt, yv, classes, K=[100], activation='relu', epochs=20_000, patience=100, verbose=0, drop=0, use_L1=False, es_tol=0, lr=1e-1):
    inp = ks.layers.Input(shape=(Xt.shape[1],))

    if use_L1:
        skip = ks.layers.Dense(units=1, activation='linear', use_bias=False, kernel_regularizer=ks.regularizers.L1(), name='skip_layer')(inp)
    else:
        skip = ks.layers.Dense(units=1, activation='linear', use_bias=False, name='skip_layer')(inp)

    gw = ks.layers.Dense(units=K[0], activation=activation, name='gw_layer')(inp)
    # gw = ks.layers.Dropout(drop)(gw)

    if len(K) > 1:
        for k in K[1:]:
            gw = ks.layers.Dense(units=k, activation=activation)(gw)   

    last_node = ks.layers.Dense(units=classes)(gw)
    output = ks.layers.Add()([skip, last_node])

    # Implement early stopping
    early_stop = ks.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=es_tol,
        patience=patience,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        # start_from_epoch=0,
    )

    # Initial dense training
    nn = ks.models.Model(inputs=inp, outputs=output)
    nn.compile(optimizer=ks.optimizers.Adam(lr), loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    nn.fit(Xt, yt, validation_data=(Xv, yv), epochs=epochs, callbacks=[early_stop], verbose=verbose)

    return nn