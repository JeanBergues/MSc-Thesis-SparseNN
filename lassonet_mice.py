import numpy as np
import sklearn.datasets as skdata
import sklearn.impute as imp
import sklearn.preprocessing as pp
import sklearn.metrics as met
import matplotlib.pyplot as plt
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
    X_trainv, X_val, y_trainv, y_val = train_test_split(X_train, y_train, test_size=0.125)
    print("Split data.")

    lassoC = ln.LassoNetClassifierCV(verbose=2, hidden_dims=(77,))
    cls = lassoC.fit(X_train, y_train)
    # path = lassoC.path(X_trainv, y_trainv, X_val=X_val, y_val=y_val)
    # ln.plot_path(lassoC, path, X_test, y_test)
    plt.show()
    print("Finished training.")

    prediction = cls.predict(X_test)
    print(met.accuracy_score(y_test, prediction))

    # Doing the tensorflow
    # model = lasso.LassoNetClassifier()
    # path = model.path(X_train, y_train)
    # print(f"Score: {model.score(X_test, y_test)}")
    # lasso.plot_path(model, path, X_test, y_test)
    # plt.savefig("miceprotein.png")

if __name__ == '__main__':
    main()