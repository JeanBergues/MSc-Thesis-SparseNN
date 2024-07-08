import numpy as np
def hier_prox(theta: np.ndarray, W: np.ndarray, l: float, M: float) -> tuple[np.ndarray, np.ndarray]:
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

    # Set the first value of each row to true if all other values in the row are false. Should in theory never happen
    set_first_true_array = np.full_like(m_tilde_first_only, False)
    set_first_true_array[:,0] = np.sum(m_tilde_first_only, axis=1) < 1
    m_tilde_first_only = np.logical_or(m_tilde_first_only, set_first_true_array)
    m_tilde = w_m[m_tilde_first_only]

    # Calculate output
    theta_out = (1/M) * np.sign(theta) * m_tilde
    W_out = np.sign(W) * np.minimum(np.abs(W), np.repeat(m_tilde.reshape((-1, 1)), K, axis=1))

    return (theta_out, W_out)