import numpy as np
import sklearn.datasets as skdata
import sklearn.impute as imp
import sklearn.preprocessing as pp
import sklearn.metrics as met
import matplotlib.pyplot as plt
import keras as ks
import tensorflow as tf
from sklearn.model_selection import train_test_split
import lassonet as ln


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

    lassoC = ln.LassoNetClassifier()
    path = lassoC.path(X_train, y_train)
    ln.plot_path(lassoC, path, X_test, y_test)
    plt.show()
    print("Finished training.")

    prediction = np.argmax(lassoC.predict(X_test), axis=1)
    print(met.accuracy_score(y_test, prediction))

    # Doing the tensorflow
    # model = lasso.LassoNetClassifier()
    # path = model.path(X_train, y_train)
    # print(f"Score: {model.score(X_test, y_test)}")
    # lasso.plot_path(model, path, X_test, y_test)
    # plt.savefig("miceprotein.png")

if __name__ == '__main__':
    main()