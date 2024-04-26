import numpy as np
import sklearn.datasets as skdata
import sklearn.impute as imp
import sklearn.preprocessing as pp
import sklearn.metrics as met
import keras as ks
import tensorflow as tf
from sklearn.model_selection import train_test_split


def main() -> None:
    # Read in the data
    # data = pd.read_csv('data/preprocessed_mice.csv', header=0)
    # df = data.drop(['MouseID', 'Genotype', 'Treatment', 'Behavior'], axis=1)
    
    # Split into test and train
    # X_full = df.drop(['class'], axis=1).to_numpy()
    # y_full = df['class'].to_numpy()

    X_full, y_full = skdata.fetch_openml(name="miceprotein", return_X_y=True)
    print(X_full.shape)
    print("Loaded data.")

    X_full = imp.SimpleImputer().fit_transform(X_full)
    X_full = pp.MinMaxScaler().fit_transform(X_full)

    y_full = pp.LabelEncoder().fit_transform(y_full)
    print("Cleaned data.")

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2)
    print("Split data.")

    d = np.shape(X_full)[1]
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.125)
    options = [d//3, (2*d)//3, d, (4*d)//3]
    best_accuracy = 0
    best_size = 0
    for s in options:
        nn = ks.models.Sequential()
        nn.add(ks.layers.Input(shape=(X_t.shape[1],)))
        nn.add(ks.layers.Dense(units=s, activation='relu'))
        nn.add(ks.layers.Dense(units=8))

        nn.compile(optimizer = 'adam', loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        nn.fit(X_t, y_t, epochs=100)
        test_loss, test_acc = nn.evaluate(X_v,  y_v, verbose=2)
        if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_size = s
                nonval_class = nn

    best_class = nonval_class
    print(f"Layers: {best_size}")

    prediction = np.argmax(best_class.predict(X_test), axis=1)
    print(met.accuracy_score(y_test, prediction))

    # Doing the tensorflow
    # model = lasso.LassoNetClassifier()
    # path = model.path(X_train, y_train)
    # print(f"Score: {model.score(X_test, y_test)}")
    # lasso.plot_path(model, path, X_test, y_test)
    # plt.savefig("miceprotein.png")

if __name__ == '__main__':
    main()