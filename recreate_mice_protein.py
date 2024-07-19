import sklearn.datasets as skdata
import sklearn.impute as imp
import sklearn.preprocessing as pp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from lassonet_implementation import return_LassoNet_results, paper_lassonet_results
from network_definitions import return_MLP_skip_classifier

X_full, y_full = skdata.fetch_openml(name="miceprotein", return_X_y=True)
print(X_full.shape)
print("Loaded data.")

X_full = imp.SimpleImputer().fit_transform(X_full)
X_full = pp.StandardScaler().fit_transform(X_full)

y_full = pp.LabelEncoder().fit_transform(y_full)
print("Cleaned data.")

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2)
print("Split data.")

# res_k, res_theta, res_val, res_l = paper_lassonet_results(X_train, X_test, y_train, y_test, K=(77 // 3,), regressor=False, verbose=2)
initial_network = return_MLP_skip_classifier(X_train, X_test, y_train, y_test, 8, K=[100], epochs=1000, verbose=1, lr=0.001, activation='relu')
res_k, res_theta, res_val, res_l = return_LassoNet_results(initial_network, X_train, X_test, y_train, y_test, print_lambda=True, print_path=True, starting_lambda=None, regression=False)

sns.lineplot(x=res_k, y=res_val)
plt.show()