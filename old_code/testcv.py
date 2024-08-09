import numpy as np

res1 = [3, 3, 3, 3, 3, 2.5, 2.3, 2.1, 1.9, 1.7, 1.5, 1, 0.5]
res2 = [3, 3, 3, 3, 3, 2.6, 2.3, 2.0, 1.9, 1.6, 1.6, 1.4, 1.2, 1.0, 20]
res3 = [3, 3, 3, 3, 3, 2.5, 2.3, 2.1, 1.9, 1, 0.5]
res4 = [3, 3, 3, 2.5, 2.1, 2.1, 1.9, 1.8, 1.5, 1, 0.7, 0.5]

full_res = []
full_res.append(res1)
full_res.append(res2)
full_res.append(res3)
full_res.append(res4)

longest_path_length = max(map(len, full_res))
longest_path = max(full_res, key=len)
l_path = np.array([1 * (1.02**i) for i in range(longest_path_length)])
y = np.array([path+[path[-1]]*(longest_path_length-len(path)) for path in full_res])
means = np.mean(y, axis=0)
final_lambda = l_path[np.argmin(means)]
print(l_path)
print(l_path[np.argmin(means)])