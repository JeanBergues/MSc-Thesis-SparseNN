import numpy as np
def hier_prox(theta: np.ndarray, W: np.ndarray, l: float, M: float) -> tuple[np.ndarray, np.ndarray]:
    # Check if the shapes are correct
    theta = theta.ravel()
    d = theta.shape[0]
    K = W.shape[1]
    assert W.shape[0] == d

    # Sort the absolute values of W in decreasing order and create cumulative sum in order to vectorize operations
    sorted_W = -np.sort(-np.abs(W))
    W_sum = np.cumsum(sorted_W, axis=1)

    # Calculate w_m per theta
    m = np.arange(start=0, stop=K+1)
    padded_Wsum = np.concatenate([np.zeros((d, 1)), W_sum], axis=1)
    threshold = np.clip(np.repeat(np.abs(theta).reshape((-1, 1)), K+1, axis=1) + M * padded_Wsum - np.full_like(padded_Wsum, l), 0, np.inf)
    w_m = M / (1 + m * (M**2)) * threshold

    # Check for the two conditions
    upper_bound = np.concatenate([np.repeat(np.inf, d).reshape((-1, 1)), sorted_W], axis=1)
    lower_bound = np.concatenate((sorted_W, np.zeros((d, 1))), axis=1)
    m_tilde_condition = np.logical_and(w_m <= upper_bound, w_m >= lower_bound)

    # Select only the first true value for each theta
    first_m_tilde = m_tilde_condition.cumsum(axis=1).cumsum(axis=1) == 1
    m_tilde = w_m[first_m_tilde]

    # Calculate the final weights
    theta_out = (1/M) * np.sign(theta) * m_tilde
    W_out = np.sign(W) * np.minimum(np.abs(W), np.repeat(m_tilde.reshape((-1, 1)), K, axis=1))

    return (theta_out, W_out)