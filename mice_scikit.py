import numpy as np
import pandas as pd
import sklearn.datasets as skdata
import sklearn.impute as imp
import sklearn.preprocessing as pp
import sklearn.neural_network as nn
import sklearn.model_selection as msel
import sklearn.metrics as metrics
import lassonet as lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main() -> None:
    # Read in the data
    # data = pd.read_csv('data/preprocessed_mice.csv', header=0)
    # df = data.drop(['MouseID', 'Genotype', 'Treatment', 'Behavior'], axis=1)
    
    # Split into test and train
    # X_full = df.drop(['class'], axis=1).to_numpy()
    # y_full = df['class'].to_numpy()

    X_full, y_full = skdata.fetch_openml(name="miceprotein", return_X_y=True)
    print("Loaded data.")

    X_full = imp.SimpleImputer().fit_transform(X_full)
    X_full = pp.MinMaxScaler().fit_transform(X_full)

    y_full = pp.LabelEncoder().fit_transform(y_full)
    print("Cleaned data.")

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2)
    print("Split data.")

    d = np.shape(X_full)[1]
    options = [d//3, (2*d)//3, d, (4*d)//3]
    gsearch_model = msel.GridSearchCV(nn.MLPClassifier(solver='adam', activation='relu', validation_fraction=0.125), param_grid={'hidden_layer_sizes': options, 'random_state': [12345], 'max_iter': [500]}, n_jobs=-1, refit=True, verbose=3)
    gmodel = gsearch_model.fit(X_train, y_train)

    best_class = gmodel.best_estimator_
    print(f"Layers: {gmodel.best_params_['hidden_layer_sizes']}")

    prediction = best_class.predict(X_test)
    print(metrics.accuracy_score(y_test, prediction))

    # Doing the tensorflow
    # model = lasso.LassoNetClassifier()
    # path = model.path(X_train, y_train)
    # print(f"Score: {model.score(X_test, y_test)}")
    # lasso.plot_path(model, path, X_test, y_test)
    # plt.savefig("miceprotein.png")

if __name__ == '__main__':
    main()