import numpy as np

k_list = np.array([5, 5, 5, 4, 4, 3, 2, 2, 2, 1])
results =  np.array([9, 8, 9, 7, 7, 3, 2, 3, 4, 5])
unique_k = np.unique(k_list)
print(unique_k)
print(k_list == 5)

x = [np.min(results[k_list == k]) for k in unique_k]
print(x)