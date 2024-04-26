import numpy as np
import numba as nb
import time as tm

# @nb.njit()
def soft_threshold(x: float, labda: float) -> float:
    return np.sign(x) * max(abs(x) - labda, 0)

# @nb.njit()
def hier_prox(theta: np.ndarray, W: np.ndarray, l: float, M: float) -> tuple[np.ndarray, np.ndarray]:
    # Notation
    d = theta.shape[0]
    K = W.shape[1]
    assert W.shape[0] == d

    # Allocate space for output
    theta_out = np.zeros_like(theta)
    W_out = np.zeros_like(W)

    # Perform the algorithm
    for j, Wj in enumerate(W):
        sorted_Wj = np.flip(np.sort(np.abs(Wj))) 
        wm = np.array([M/(1 + (m+1)*M**2) * soft_threshold(np.abs(theta[j]) + M * np.sum(sorted_Wj[:(m+1)]), l) for m in range(K)])

        for m in range(K+1):
            if m == 0 and sorted_Wj[0] <= M/(1 + m*M**2) * soft_threshold(np.abs(theta[j]), l):
                wm_tilde = M/(1 + m*M**2) * soft_threshold(np.abs(theta[j]), l)
                break
            elif m == K and wm[K-1] <= sorted_Wj[K-1]:
                wm_tilde = wm[K-1]
                break
            elif sorted_Wj[m] <= wm[m-1] <= sorted_Wj[m-1]:
                wm_tilde = wm[m-1]
                break

        theta_out[j] = (1/M) * np.sign(theta[j]) * wm_tilde
        W_out[j] = np.sign(Wj) * np.minimum(np.repeat(wm_tilde, K), np.abs(Wj))

    return (theta_out, W_out)

if __name__ == '__main__':
    np.set_printoptions(precision=3, linewidth=300, floatmode='maxprec')
    d_test = 77
    K_test = 100

    start = tm.perf_counter()
    x = hier_prox(np.ones(d_test), np.random.standard_normal((d_test, K_test)), 50, 10)
    end = tm.perf_counter()
    print(f"First run took {end - start:.6f} ns.")

    start = tm.perf_counter()
    x = hier_prox(np.ones(d_test), np.random.standard_normal((d_test, K_test)), 50, 10)
    end = tm.perf_counter()
    print(f"Secon run took {end - start:.6f} ns.")

    start = tm.perf_counter()
    x = hier_prox(np.ones(d_test), np.random.standard_normal((d_test, K_test)), 50, 10)
    end = tm.perf_counter()
    print(f"Third run took {end - start:.6f} ns.")