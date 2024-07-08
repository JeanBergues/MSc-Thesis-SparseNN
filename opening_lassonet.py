import numpy as np

mask = np.load('lasso_forc/lasso_day_7_7_mask.npy')
print(np.sum(mask))
for i in range(len(mask) // 7):
    print(mask[i*7 : (i + 1)*7]*1)