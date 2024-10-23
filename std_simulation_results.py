import numpy as np

EXPERIMENT_NAME = "R2NL8"
res = np.load(f"simulation_results/OFFICIAL_{EXPERIMENT_NAME}.npy")

deviations = np.std(res, axis=1) / np.sqrt(100)
names = ["LR_pct", "NLR_pct", "R_pct", "I_pct", "dense_mses", "mlp_mses", 
                                      "L1_mses", "sparse_mses", "pct_improvements", "pct_mlp_improvements", "pct_L1_improvements", "n_features_chosen", 
                                      "n_times_improved", "n_times_mlp_improved", "n_times_L1_improved"]
names = ["LR_pct", "NLR_pct", "R_pct", "I_pct", "dense_mses", "sparse_mses", "pct_improvements", "n_features_chosen", "n_times_improved"]
                                      


for n, d in zip(names, deviations):
    print(f"{n} \t\t\t= {d:.3f}")

feat = res[7]
pct_L = res[0] * 2

pct_NL = (feat - pct_L) / 8
print(np.mean(pct_NL))
print(np.std(pct_NL) / np.sqrt(100))
