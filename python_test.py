import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def main() -> None:
    # Read in the data
    data = pd.read_csv('data/preprocessed_mice.csv', header=0)
    df = data.drop(['MouseID', 'Genotype', 'Treatment', 'Behavior'], axis=1)
    
    # Split into test and train
    X_full = df.drop(['class'], axis=1).to_numpy()
    y_full = df['class'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2)

    # Doing the tensorflow
    d = np.shape(X_train)[1]
    
    nn_model = ks.Sequential(
        ks.layers.Dense(25, activation='relu')
    )

    nn_model.compile(optimizer='adam')

if __name__ == '__main__':
    main()