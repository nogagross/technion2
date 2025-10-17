import numpy as np
from scipy.linalg import svd
import warnings
from sklearn.neighbors import NearestNeighbors
#from pyriemann.utils.mean import mean_riemann
from tqdm import tqdm
#import torch

#region
def sym_pos_def_dist(A, B, p=2):
    eig = np.linalg.eigvals(np.linalg.inv(A) @ B)
    if p == 1:
        dist = np.sum(np.abs(np.log(eig)))
    else:
        dist = np.sum(np.abs(np.log(eig)) ** p) ** (1 / p)
    return dist
#endregion
# ------- GPU version of sym_pos_def_dist function ----------

# uses Cholesky decompusition of SPD matrix A=LL^T, where L is lower triangular
#
# def _device(dev=None):
#     return dev or ('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# def _as_torch(x, device=None, dtype=torch.float64):
#     return torch.as_tensor(x, dtype=dtype, device=_device(device))
#
# def GPU_sym_pos_def_dist(A, B, p=2,device=None):
#     A_t = _as_torch(A, device) # converts to torch tensor
#     B_t = _as_torch(B, device)
#     Y = torch.linalg.solve(A_t, B_t)  # A^{-1} B
#     lam = torch.linalg.eigvals(Y).real #compute eigenvalues of A^{-1}B
#     v = torch.log(lam).abs() # log of the euigenvalues
#     d = torch.linalg.vector_norm(v, ord=p)  # p is the order of the norm
#     return d.item()
# #region
def sym_pos_semi_def_dist(A, B, r, k=1):
    sym = lambda M: (M + M.T) / 2
    A = sym(A)
    B = sym(B)

    eig_A, vec_A = np.linalg.eig(A)
    eig_B, vec_B = np.linalg.eig(B)

    # keep the eigenvectors of the r largest eigenvalues
    vec_A = vec_A[:, np.argsort(eig_A)[-r:]]
    vec_B = vec_B[:, np.argsort(eig_B)[-r:]]

    # numpy returns V transposed compared to matlab
    try:
        OA, S, OB = np.linalg.svd(vec_A.T @ vec_B)
    except:
        OA, S, OB = svd(vec_A.T @ vec_B, lapack_driver='gesvd')
    if np.any(abs(S) > 1.):
        # if not np.allclose(abs(S[abs(S) > 1.]), 1.):
        #     print(f"SVD yields S {S[abs(S) > 1.]}")
        S[S > 1.] = 1.
        S[S < -1.] = -1.
    vTheta = np.arccos(S)
    UA = vec_A @ OA
    UB = vec_B @ OB.T
    RA = sym(UA.T @ A @ UA)
    RB = sym(UB.T @ B @ UB)
    dU = np.linalg.norm(vTheta)
    dR = sym_pos_def_dist(RA, RB)
    d = np.sqrt(dU ** 2 + k * dR ** 2)
    return d
#endregion
# # ------- GPU version of sym_pos_semi_def_dist function ----------
#
# def GPU_sym_pos_semi_def_dist(A, B, r, k=1, p=2, device=None):
#     A_t = _as_torch(A, device) # convert to torch tensor
#     B_t = _as_torch(B, device)
#     n = A_t.shape[-1]
#     r = int(r)
#     # top r eigenpairs
#     evalA, vecA = torch.linalg.eigh(A_t)          # (n,), (n,n)
#     evalB, vecB = torch.linalg.eigh(B_t)
#     idx = torch.arange(n - r, n, device=A_t.device) #indx vector for top r eigenpairs
#     lamA = evalA.index_select(0, idx) # (r,). pick top r eigenvalues
#     lamB = evalB.index_select(0, idx) #pick top r eigenvalues
#     VA   = vecA.index_select(1, idx) # (n,r). top r eigenvectors
#     VB   = vecB.index_select(1, idx) #top r eigenvectors
#
#     # principal angles via SVD(VA^T VB)
#     C = VA.mT @ VB                                 # (r,r)
#     U, S, Vh = torch.linalg.svd(C, full_matrices=False)
#     theta = torch.arccos(S.clamp(max=1.0))         # Principal angles ( arccos(singular values) )
#     dU = torch.linalg.vector_norm(theta, ord=2)    # same as np.linalg.norm
#
#     # restrict A,B to the matched r-dim subspaces:
#     V  = Vh.mT
#     # restrict A and B to the r dimensional bases:
#     RA = U.mT @ torch.diag(lamA)@ U               # (r,r) SPD
#     RB = V.mT @ torch.diag(lamB)@ V
#
#     # SPD part with your p-norm on log-eigs
#     dR_val = GPU_sym_pos_def_dist(RA, RB, p=p, device=A_t.device)  # SPD distance on the restricted r×r matrices. float
#     dR = torch.as_tensor(dR_val, dtype=A_t.dtype, device=A_t.device) #convert the float to torch tensor
#
#     d = torch.sqrt(dU**2 +(k*dR)**2) #Combines the two parts of semi definite distence: the angle part and the SPD eigenvalue part
#     return d.item()
#
#
# #region
def _riemannian_dist(corrs, eigval_bound=0.01):
# r: smallest rank
    r = np.min(np.sum(np.linalg.eigvals(corrs) > eigval_bound, axis=1))
    dR = np.zeros((len(corrs), len(corrs)))

    n = len(corrs)
    total_pairs = n * (n - 1) // 2  # total number of (i,j) pairs

    with tqdm(total=total_pairs, desc="Computing distances") as pbar:
        for i, corr_i in enumerate(corrs):
            for j, corr_j in enumerate(corrs[i + 1:]):
                dR[i + j + 1, i] = sym_pos_semi_def_dist(corr_i, corr_j, r)
                dR[i, i + j + 1] = dR[i + j + 1, i]
                pbar.update(1)
    return dR
# #
# #endregion
# ------- GPU version of sym_pos_semi_def_dist function ----------
#
# def GPU_riemannian_dist(corrs, eigval_bound=0.01,device=None):
#
#     X = _as_torch(corrs, device)  # converts to python tensor shape (K,n,n)
#     K, n, _ = X.shape #K-num of matrices, n-size of correlation matrix (nxn)
#     evals_all, vecs_all = torch.linalg.eigh(X)  # eigenvalue of size (K, n), and eigenvectors of size(K, n, n)
#     r = int((evals_all > eigval_bound).sum(dim=-1).min().item()) #rank of the matrices- how many eigenvalues are above the boundary. choose the min rank out of all K matrices.
#     r = max(r, 1)
#     idx = torch.arange(n - r, n, device=X.device) # index vector for top r eigenpairs
#     lam_all = evals_all.index_select(-1, idx)  # r largest eigenvalues (K, r)
#     V_all = vecs_all.index_select(-1, idx)  # (K, n, r)
#     # Pairwise distances
#     dists = torch.zeros((K, K), dtype=X.dtype, device=X.device) #initialize distance matrix
#     total_pairs = K *(K - 1)// 2 #number of unique pairs (k choose 2)
#     with tqdm(total=total_pairs, desc="Riemannian distances (GPU)") as pbar:
#         for i in range(K): #iterate over matrix
#             lamA, VA = lam_all[i], V_all[i] #top r eigenvalues and eigenvectors of matrix i
#             for j in range(i + 1, K): #iterate over matrices j>i
#                 lamB, VB = lam_all[j], V_all[j] #top-r eigenpairs for matrix j
#
#                 # principal angles via SVD
#                 C = VA.mT @ VB
#                 U, S, Vh = torch.linalg.svd(C, full_matrices=False)
#                 theta = torch.arccos(S.clamp(max=1.0))
#                 dU = torch.linalg.vector_norm(theta, ord=2)
#
#                 # restrict A,B to matched r dim principal subspaces
#                 V = Vh.mT
#                 RA = U.mT @ torch.diag(lamA) @ U  # (r, r)
#                 RB = V.mT @ torch.diag(lamB) @ V
#
#                 # SPD part (log-eig p-norm) on r×r blocks
#                 dR_val = GPU_sym_pos_def_dist(RA, RB, p=2, device=X.device)  # float
#                 dR = torch.as_tensor(dR_val, dtype=X.dtype, device=X.device)
#
#                 d = torch.sqrt(dU**2 + dR**2)
#
#                 dists[i, j] = dists[j, i] = d
#                 pbar.update(1)
#
#     return dists.cpu().numpy()


def _get_kernel_riemannian(all_distances, sigma_cutoff=9999999, eps=2):
    closest_distances = np.sort(all_distances)[:, :sigma_cutoff]
    sigma = eps * np.median(closest_distances)
    kernel = np.exp(- (all_distances / (np.sqrt(2) * sigma)) ** 2)
    kernel = (kernel + kernel.T) / 2
    kernel = _make_row_stochastic(kernel)
    return kernel


def _make_row_stochastic(kernel):
    'This does not strictly mean row-stochastic,but normalizes the kernel'
    column_sum = np.sum(kernel, axis=0)
    row_stochastic_kernel = np.einsum("i, j, ij -> ij",
                                      1 / np.sqrt(column_sum),
                                      1 / np.sqrt(column_sum),
                                      kernel)
    return row_stochastic_kernel


def _regularize_by_median_sv(correlations, signal, window_length=0,
                             subsampling=0):
    if subsampling > 0:
        length = signal.shape[-1]
        midpoints = np.linspace(window_length // 2,
                                length - window_length // 2,
                                subsampling)
        midpoints = list(map(int, midpoints))
        u, s, v = np.linalg.svd(signal[..., midpoints])
    else:
        u, s, v = np.linalg.svd(signal)
    regularizer = np.median(s, axis=1)[:, None, None] * np.eye(
        correlations.shape[-1])[None, :, :]
    return correlations + regularizer[:, None, :, :]


def _regularize_by_smallest_ev(correlations, eps=1e-3):
    eig = np.linalg.eigvals(correlations)
    for idx, e in enumerate(eig):
        if any(e.flatten() < 0):
            eps = eps - np.min(e.flatten())
            regularizer = eps * np.eye(correlations.shape[-1])
            correlations[idx] = regularizer[None, :, :] + correlations[idx]
    return correlations


def mse_distance_matrix(matrix_list):
    n = len(matrix_list)
    dist_matrix = np.zeros((n, n))

    for i in tqdm(range(n), desc="Computing MSE distances"):
        for j in range(i, n):  # symmetric, so compute only upper triangle
            mse = np.mean((matrix_list[i] - matrix_list[j])**2)
            dist_matrix[i, j] = mse
            dist_matrix[j, i] = mse  # mirror the value

    return dist_matrix


def _get_kernel_euclidean(X, scale_k):
    """
     Computes a kernel using Mean Squared Error (MSE) as the distance metric
     between rows of X. Uses locally scaled RBF kernel, and returns
     the kernel and the pairwise MSE distances of nearest neighbors.
     """

    distances = mse_distance_matrix(X)

    # Step 3: Locally scaled RBF kernel
    sigma = np.median(distances, axis=1) + 1e-8  # avoid divide-by-zero
    nonvanishing_entries = np.exp(- (distances / sigma[:, None]) ** 2)

    # Step 4: Construct sparse kernel matrix
    kernel = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            kernel[i, j] = nonvanishing_entries[i, j]

    # Step 5: Symmetrize and normalize
    kernel = (kernel + kernel.T) / 2
    kernel = _make_row_stochastic(kernel)

    return kernel, distances

#region
def get_diffusion_embedding(correlations, window_length, scale_k=20,
                            signal=None, subsampling=0, mode='riemannian'):
    """
    :param
    correlations: (Bx)KxNxN K correlation matrices. Will be carried out
    over all first dimensions
    :param
    scale_k: number of nearest neighbors to use for evaluating the scale
    :param
    tol: tolerance when iteration to get Riemannian mean converged
    :param
    maxiter: when to stop Riemannian mean algorithm
    :param
    vector_input: to use diffusion embedding onto vectors, not correlation
    matrices. Only used to test functionality. Use with care.
    :return:
    """

    if ((correlations.ndim == 3 and mode != 'vector_input') or
            (correlations.ndim == 2 and mode == 'vector_input')):
        correlations = np.array([correlations])
    elif ((correlations.ndim == 4 and mode != 'vector_input') or
          (correlations.ndim == 3 and mode == 'vector_input')):
        pass
    else:
        raise ValueError(f"correlations must be shape (Bx)KxNxN but is "
                         f"{correlations.shape}")

    if window_length < correlations.shape[-1] and mode != 'vector_input':
        warnings.warn("Small window_length. Regularizing correlations.")
        if signal is not None:
            if subsampling > 0:
                correlations = _regularize_by_median_sv(
                    correlations, signal, window_length, subsampling)
            else:
                correlations = _regularize_by_median_sv(correlations, signal)
        else:
            correlations = _regularize_by_smallest_ev(correlations)

    distances = []
    diffusion_representations = []
    for corrs in correlations:
        if mode == 'riemannian':
            dists = _riemannian_dist(corrs)
            distances.append(dists)
            kernel = _get_kernel_riemannian(dists)
        elif mode == 'euclidean':
            kernel, dists = _get_kernel_euclidean(
                corrs.reshape(corrs.shape[:-2] + (-1,)), scale_k)
            distances.append(dists)
        else:
            raise ValueError(f'{mode=}')

        # TODO check eigenvalues - do they give away reconstruction error?
        eig, vec = np.linalg.eigh(kernel)
        sort_idx = eig.argsort()[-2::-1]
        vec = vec.T[sort_idx]
        vec = eig[sort_idx, None] * vec

        diffusion_representations.append(vec)

    return np.array(diffusion_representations), np.array(distances)
#endregion

# ---- GPU version of get_diffusion_embedding function ----

# def GPU_get_diffusion_embedding(correlations, window_length, scale_k=20,
#                             signal=None, subsampling=0, mode='riemannian'):
#     """
#     :param
#     correlations: (Bx)KxNxN K correlation matrices. Will be carried out
#     over all first dimensions
#     :param
#     scale_k: number of nearest neighbors to use for evaluating the scale
#     :param
#     tol: tolerance when iteration to get Riemannian mean converged
#     :param
#     maxiter: when to stop Riemannian mean algorithm
#     :param
#     vector_input: to use diffusion embedding onto vectors, not correlation
#     matrices. Only used to test functionality. Use with care.
#     :return:
#     """
#
#     if ((correlations.ndim == 3 and mode != 'vector_input') or
#             (correlations.ndim == 2 and mode == 'vector_input')):
#         correlations = np.array([correlations])
#     elif ((correlations.ndim == 4 and mode != 'vector_input') or
#           (correlations.ndim == 3 and mode == 'vector_input')):
#         pass
#     else:
#         raise ValueError(f"correlations must be shape (Bx)KxNxN but is "
#                          f"{correlations.shape}")
#
#     if window_length < correlations.shape[-1] and mode != 'vector_input':
#         warnings.warn("Small window_length. Regularizing correlations.")
#         if signal is not None:
#             if subsampling > 0:
#                 correlations = _regularize_by_median_sv(
#                     correlations, signal, window_length, subsampling)
#             else:
#                 correlations = _regularize_by_median_sv(correlations, signal)
#         else:
#             correlations = _regularize_by_smallest_ev(correlations)
#
#     distances = []
#     diffusion_representations = []
#     for corrs in correlations:
#         if mode == 'riemannian':
#             dists = GPU_riemannian_dist(corrs)
#             distances.append(dists)
#             kernel = _get_kernel_riemannian(dists)
#         elif mode == 'euclidean':
#             kernel, dists = _get_kernel_euclidean(
#                 corrs.reshape(corrs.shape[:-2] + (-1,)), scale_k)
#             distances.append(dists)
#         else:
#             raise ValueError(f'{mode=}')
#
#         # TODO check eigenvalues - do they give away reconstruction error?
#         eig, vec = np.linalg.eigh(kernel)
#         sort_idx = eig.argsort()[-2::-1]
#         vec = vec.T[sort_idx]
#         vec = eig[sort_idx, None] * vec
#
#         diffusion_representations.append(vec)
#
#     return np.array(diffusion_representations), np.array(distances)
#
#


# def get_mean_matrices(corr_matrices,  window_size=20, overlap=5):
#     corrs = np.array(corr_matrices)  # shape: (N, C, C)
#
#     step_size = window_size - 2 * overlap
#
#     mean_riemanns = []
#
#     for start in range(0, len(corrs) - window_size + 1, step_size):
#         window = corrs[start:start + window_size]
#         riemann_mean = mean_riemann(window)
#         mean_riemanns.append(riemann_mean)
#
#     mean_riemanns = np.array(mean_riemanns)
#     return mean_riemanns