import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

experiment_name = "LN_SNN_X_2_0"
folds = np.load(f"final_LN_forecasts/{experiment_name}_ALLFOLDS.npy")
l_path = np.load(f"final_LN_forecasts/{experiment_name}_CV_LAMBDA.npy")

sns.set_style("whitegrid")
fig = plt.figure()
for fold in folds:
    sns.lineplot(x=l_path, y=fold, color='tab:blue', linewidth=0.5, alpha=0.8)

sns.lineplot(x=l_path, y=np.mean(folds, axis=0), color='black', linewidth=1)
plt.xlabel(r'$\lambda$')
plt.ylabel(f'mse')

# plt.savefig(f'plots/LassoNetCV_{experiment_name}.eps', format='eps', bbox_inches='tight')
# plt.savefig(f'plots/LassoNetCV_{experiment_name}.png', format='png', bbox_inches='tight')
plt.show()