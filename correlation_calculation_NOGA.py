
import numpy as np
def corrcoef_safe(W):
    """
    Correlation where zero-variance columns are handled, then regularize to SPD.
    W: (window_size, n_nodes) with variables in columns
    returns: (n_nodes, n_nodes) SPD correlation matrix
    """
    C = np.corrcoef(W, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    zero_var = (np.std(W, axis=0, ddof=1) == 0)
    for j, is_zero in enumerate(zero_var):
        if is_zero:
            C[j, :]=0
            C[:, j]=0
            C[j, j] =1

    # eigenvalues >=0 enforcement
    w, V = np.linalg.eigh(C) #eigenvalues, eigenvectors
    w = np.maximum(w, 1e-6) #making sure all eigenvalues are positive
    C = (V * w) @ V.T #Eigen decomposition
    return C

def corrcoef_per_timewindow(X, timestamps, window_duration, overlap=0.5):
    """
    Compute correlation matrices in sliding time windows using corrcoef_safe.

    Parameters:
    X : np.ndarray (T, n_cells)
        Data matrix with firing rates.T - time points.
    timestamps : np.ndarray (T,)
        sorted array of timestamps corresponding to rows of X.
    window_duration : float
        Sliding window duration in seconds.
    overlap : float
        Fractional overlap between windows (e.g., 0.5 for 50%).

    Returns:
    C_all : np.ndarray (n_windows, n_cells, n_cells)
        Correlation matrices per window.
    window_starts : np.ndarray (n_windows,)
        Start times of each window.
    """

    T, n_cells = X.shape
    hop = window_duration *(1-overlap)
    t_start =timestamps.iloc[0]
    t_end = timestamps.iloc[-1] #kast timestamp in dataset
    window_starts = []
    window_ends=[]
    Corr_list = []
    current_start = t_start
    while current_start+window_duration <= t_end:
        current_end = current_start + window_duration
        in_window = np.where((timestamps >= current_start) & (timestamps < current_end))[0] #indices of timestamps within the window
        if len(in_window)>1:
            W = X[in_window,:] #window dataframe
            corr_mat = corrcoef_safe(W)
        else:
            corr_mat = np.zeros((n_cells, n_cells))

        Corr_list.append(corr_mat)
        window_starts.append(current_start)
        window_ends.append(current_end)

        current_start += hop
    for _ in tqdm(range(len(window_starts)), desc="Sliding time windows"):
        pass
    C_all = np.stack(Corr_list, axis=0)
    window_starts = np.array(window_starts)
    window_ends=np.array(window_ends)
    return C_all, window_starts,window_ends