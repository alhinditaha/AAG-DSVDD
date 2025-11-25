from typing import Tuple
import numpy as np
import torch
from sklearn.datasets import make_blobs

def synthetic_blobs(n_norm=1000, n_anom=120, n_unl=400, seed=42) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)
    Xn, _ = make_blobs(n_samples=n_norm, centers=[(0,0)], cluster_std=1.0, random_state=seed)
    Xa, _ = make_blobs(n_samples=n_anom, centers=[(6,6)], cluster_std=0.8, random_state=seed+1)
    Xu, _ = make_blobs(n_samples=n_unl, centers=[(2.5,2.5)], cluster_std=1.6, random_state=seed+2)
    X = np.vstack([Xn, Xa, Xu]).astype(np.float32)
    y = np.concatenate([np.ones(n_norm, dtype=np.int64), -np.ones(n_anom, dtype=np.int64), np.zeros(n_unl, dtype=np.int64)])
    perm = rng.permutation(X.shape[0])
    X = X[perm]; y = y[perm]
    return torch.from_numpy(X), torch.from_numpy(y)

def split_indices(y_all: torch.Tensor):
    idx_norm = (y_all == 1).nonzero(as_tuple=True)[0]
    idx_anom = (y_all == -1).nonzero(as_tuple=True)[0]
    idx_unl  = (y_all == 0).nonzero(as_tuple=True)[0]
    return idx_norm, idx_anom, idx_unl

def load_csv_dataset(path: str, label_col: str = 'label'):
    import pandas as pd
    df = pd.read_csv(path)
    y = df[label_col].astype(int).to_numpy()
    X = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    import torch
    return torch.from_numpy(X), torch.from_numpy(y)
