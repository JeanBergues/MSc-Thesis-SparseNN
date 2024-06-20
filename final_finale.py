import pandas as pd
import numpy as np
import keras as ks
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as mt
import sklearn.linear_model as sklm

from full_mylasso import train_lasso_path, train_dense_model, estimate_starting_lambda
import lassonet as lsn

tf.random.set_seed(1234)
tf.get_logger().setLevel('ERROR')
np.random.seed(1234)
rng = np.random.RandomState(1234)

def return_skip_estimator()