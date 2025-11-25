from typing import Literal
import numpy as np
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

Policy = Literal['all', 'labeled_to_unlabeled_only']

def _local_scale(dists_row: np.ndarray) -> float:
    nz = dists_row[dists_row > 0.0]
    if nz.size == 0:
        return 1.0
    return float(np.median(nz))

def build_label_aware_knn_laplacian(
    Z: np.ndarray,
    labels: np.ndarray,
    k: int = 15,
    upweight_anom_edges_gamma: float = 1.0,
    row_normalize: bool = True,
    device: str = 'cpu',
    policy: Policy = 'labeled_to_unlabeled_only',
    oversample_factor: int = 3,
) -> torch.Tensor:
    n = Z.shape[0]
    K = min(max(k * max(1, oversample_factor) + 1, k + 1), n)
    nn = NearestNeighbors(n_neighbors=K, algorithm='auto', metric='euclidean')
    nn.fit(Z)
    dists, idxs = nn.kneighbors(Z, return_distance=True)

    sigmas = np.array([_local_scale(d[1:]) for d in dists], dtype=np.float64)
    if np.any(sigmas == 0.0):
        nonz = sigmas[sigmas > 0.0]
        sigmas[sigmas == 0.0] = np.median(nonz) if nonz.size > 0 else 1.0

    rows, cols, vals = [], [], []
    for i in range(n):
        neigh = idxs[i, 1:]
        di = dists[i, 1:]
        yi = labels[i]

        chosen_idx = []
        chosen_dist = []

        if policy == 'all' or yi == 0:
            chosen_idx = neigh[:k]
            chosen_dist = di[:k]
        else:
            for j, dij in zip(neigh, di):
                if labels[j] == 0:
                    chosen_idx.append(j); chosen_dist.append(dij)
                    if len(chosen_idx) >= k:
                        break

        for j, dist_ij in zip(chosen_idx, chosen_dist):
            s = sigmas[i] * sigmas[j]
            w = float(np.exp(- (dist_ij ** 2) / (s if s > 0 else 1.0)))
            if upweight_anom_edges_gamma > 1.0 and (yi == -1 or labels[j] == -1):
                w *= upweight_anom_edges_gamma
            rows.append(i); cols.append(j); vals.append(w)

    W = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)

    if row_normalize:
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums[row_sums == 0.0] = 1.0
        P = sparse.diags(1.0 / row_sums) @ W
    else:
        P = W.tocsr()

    Pt = P.transpose().tocsr()
    M = 0.5 * (P.tocsr() + Pt)

    I = sparse.identity(n, format='csr', dtype=np.float64)
    L = (I - M).tocsr().tocoo()

    indices = np.vstack((L.row, L.col))
    indices_t = torch.tensor(indices, dtype=torch.long, device=device)
    values_t = torch.tensor(L.data, dtype=torch.float32, device=device)
    L_torch = torch.sparse_coo_tensor(indices_t, values_t, size=(n, n)).coalesce()
    return L_torch
