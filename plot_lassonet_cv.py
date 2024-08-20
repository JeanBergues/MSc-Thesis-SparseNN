import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

experiment_name = "M10000LN_SNN_[100, 20]_X_2_48"
folds = np.load(f"final_LN_forecasts/{experiment_name}_ALLFOLDS.npy")
l_path = np.load(f"final_LN_forecasts/{experiment_name}_CV_LAMBDA.npy")

sns.set_style("whitegrid")
fig = plt.figure()
for fold in folds:
    fig = sns.lineplot(x=l_path, y=fold, linewidth=0.5, alpha=0.8, size=0.5)

fig = sns.lineplot(x=l_path, y=np.mean(folds, axis=0), color='black', linewidth=2, size=1)
plt.xlabel(r'$\lambda$')
plt.ylabel(f'mse')

legd = fig.get_legend()
for t, l in zip(legd.texts, ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Mean']):
    t.set_text(l)

sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))

# plt.savefig(f'plots/LassoNetCV_{experiment_name}.eps', format='eps', bbox_inches='tight')
# plt.savefig(f'plots/LassoNetCV_{experiment_name}.png', format='png', bbox_inches='tight')
plt.show()