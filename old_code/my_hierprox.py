import numpy as np
# import numba as nb
import time as tm

# @nb.njit()
def soft_threshold(x: float, labda: float) -> float:
    return np.sign(x) * np.maximum(np.abs(x) - labda, np.zeros_like(x))

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
        wm = np.array([M/(1 + (m+1)*(M**2)) * soft_threshold(np.abs(theta[j]) + M * np.sum(sorted_Wj[:(m+1)]), l) for m in range(K)])

        for m in range(K+1):
            if m == 0 and sorted_Wj[0] <= M/(1 + m*M**2) * soft_threshold(np.abs(theta[j]), l):
                wm_tilde = M * soft_threshold(np.abs(theta[j]), l)
                break
            elif m == K:
                wm_tilde = wm[K-1]
                break
            elif sorted_Wj[m] <= wm[m-1] <= sorted_Wj[m-1]:
                wm_tilde = wm[m-1]
                break

        theta_out[j] = (1/M) * np.sign(theta[j]) * wm_tilde
        W_out[j] = np.sign(Wj) * np.minimum(np.repeat(wm_tilde, K), np.abs(Wj))

    return (theta_out, W_out)

def alt_hier_prox(theta: np.ndarray, W: np.ndarray, l: float, M: float) -> tuple[np.ndarray, np.ndarray]:
    # Notation
    d = theta.shape[0]
    K = W.shape[1]
    assert W.shape[0] == d

    # Allocate space for output
    theta_out = np.zeros_like(theta)
    W_out = np.zeros_like(W)

    # Perform the algorithm
    for j in range(d):
        theta_j = theta[j]
        Wj = W[j]
        sorted_Wj = -np.sort(-np.abs(Wj))

        wm = np.zeros(K + 1)
        for m in range(0, K+1):
            if m == 0:
                wm[m] = M / (1 + m*(M**2)) * soft_threshold(np.abs(theta_j), l)
            else:
                wm[m] = M / (1 + m*(M**2)) * soft_threshold(np.abs(theta_j) + M * np.sum(sorted_Wj[0:m]), l)
        
        upperbound = np.concatenate([[np.inf], sorted_Wj], axis=0)
        lowerbound = np.concatenate([sorted_Wj, [0]], axis=0)

        for m in range(0, K+1):
            if lowerbound[m] <= wm[m] <= upperbound[m]:
                wm_tilde = wm[m]
                break

        theta_out[j] = (1/M) * np.sign(theta_j) * wm_tilde
        W_out[j] = np.sign(Wj) * np.minimum(np.repeat(wm_tilde, K), np.abs(Wj))

    print(theta_out)
    return (theta_out, W_out)

# @nb.jit
def vec_hier_prox(theta: np.ndarray, W: np.ndarray, l: float, M: float) -> tuple[np.ndarray, np.ndarray]:
    # Notation
    theta = theta.ravel()
    d = theta.shape[0]
    K = W.shape[1]
    assert W.shape[0] == d

    # sorted_W = np.flip(np.sort(np.abs(W)), axis=1)
    sorted_W = -np.sort(-np.abs(W))
    W_sum = np.cumsum(sorted_W, axis=1)

    m = np.arange(start=0, stop=K+1)
    padded_Wsum = np.concatenate([np.zeros((d, 1)), W_sum], axis=1)
    threshold = np.clip(np.repeat(np.abs(theta).reshape((-1, 1)), K+1, axis=1) -  M * padded_Wsum - np.full_like(padded_Wsum, l), 0, np.inf)
    w_m = M / (1 + m * (M**2)) * threshold

    # Check for condition
    upper_bound = np.concatenate([np.repeat(np.inf, d).reshape((-1, 1)), sorted_W], axis=1)
    lower_bound = np.concatenate((sorted_W, np.zeros((d, 1))), axis=1)
    m_tilde_condition = np.logical_and(w_m <= upper_bound, w_m >= lower_bound)

    first_m_tilde = m_tilde_condition.cumsum(axis=1).cumsum(axis=1) == 1
    m_tilde = w_m[first_m_tilde]

    theta_out = (1/M) * np.sign(theta) * m_tilde
    W_out = np.sign(W) * np.minimum(np.abs(W), np.repeat(m_tilde.reshape((-1, 1)), K, axis=1))

    print(theta_out)
    return (theta_out, W_out)

def old_hier_prox(theta: np.ndarray, W: np.ndarray, l: float, M: float) -> tuple[np.ndarray, np.ndarray]:
    # Assert correct sizes
    theta = theta.ravel()
    d = theta.shape[0]
    K = W.shape[1]
    assert W.shape[0] == d

    # Order the weights
    sorted_W = -np.sort(-np.abs(W))

    # Calculate w_m's
    W_sum = np.cumsum(sorted_W, axis=1)
    m = np.arange(start=1, stop=K+1)
    threshold = np.clip((np.repeat(np.abs(theta).reshape((-1, 1)), K, axis=1) + M * W_sum) - np.full_like(W_sum, l), 0, np.inf)
    w_m = (M * threshold) / (1 + m * (M**2))

    # Check for condition
    m_tilde_condition = np.logical_and(w_m <= sorted_W, w_m >= np.concatenate((sorted_W, np.zeros((d, 1))), axis=1)[:,1:])

    # Find the first true value per row
    m_tilde_first_only = np.zeros_like(m_tilde_condition, dtype=bool)
    idx = np.arange(len(m_tilde_condition)), m_tilde_condition.argmax(axis=1)
    m_tilde_first_only[idx] = m_tilde_condition[idx]

    # Set the first value of each row to true if all other values in the row are false
    set_first_true_array = np.full_like(m_tilde_first_only, False)
    set_first_true_array[:,0] = np.sum(m_tilde_first_only, axis=1) < 1
    m_tilde_first_only = np.logical_or(m_tilde_first_only, set_first_true_array)
    m_tilde = w_m[m_tilde_first_only]

    # Calculate output
    theta_out = (1/M) * np.sign(theta) * m_tilde
    W_out = np.sign(W) * np.minimum(np.abs(W), np.repeat(m_tilde.reshape((-1, 1)), K, axis=1))

    print(W_out)
    return (theta_out, W_out)

if __name__ == '__main__':
    np.set_printoptions(precision=3, linewidth=300, floatmode='maxprec')
    np.random.seed(1234)
    d_test = 5
    K_test = 10

    test_U = np.random.standard_normal((d_test, K_test))

    # start = tm.perf_counter()
    # x = hier_prox(np.ones(d_test), test_U, 50, 10)
    # # print(x[0])
    # end = tm.perf_counter()
    # print(f"First run took {end - start:.6f} ns.")

    start = tm.perf_counter()
    x = alt_hier_prox(np.array([1, .1, .2, .4, 2]).reshape((-1, 1)), test_U, 11, 10)
    # print(x[0])
    end = tm.perf_counter()
    print(f"First run took {end - start:.6f} ns.")

    start = tm.perf_counter_ns()
    x = vec_hier_prox(np.array([1, .1, .2, .4, 2]).reshape((-1, 1)), test_U, 11, 10)
    end = tm.perf_counter_ns()
    print(f"First run took {end - start} ns.")

    start = tm.perf_counter_ns()
    x = old_hier_prox(np.array([1, .1, .2, .4, 2]).reshape((-1, 1)), test_U, 11, 10)
    end = tm.perf_counter_ns()
    print(f"First run took {end - start} ns.")

    # start = tm.perf_counter()
    # x = vec_hier_prox(np.ones(d_test).reshape((-1, 1)), test_U, 0.05, 10)
    # end = tm.perf_counter()
    # print(f"First run took {end - start:.6f} ns.")

    # start = tm.perf_counter()
    # x = vec_hier_prox(np.ones(d_test).reshape((-1, 1)), test_U, 0.05, 10)
    # end = tm.perf_counter()
    # print(f"First run took {end - start:.6f} ns.")
    print(np.linalg.norm(np.array([1, -2, 3, -4, 5]).reshape((-1, 1)), ord=2))