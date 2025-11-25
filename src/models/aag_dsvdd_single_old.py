# srs/methods/aag_dsvdd_single.py
# -----------------------------------------------------------------------------
# Anomaly-Aware Graph-based Semi-Supervised Deep SVDD (single-file implementation)
#
# Combines:
#   - Bias-free MLP encoder (no BN, no biases) to avoid center collapse
#   - Soft-boundary Deep SVDD hinges on LABELED data:
#       * Labeled normals (+1):   hinge(d^2 - R^2)^p   (kept INSIDE)
#       * Labeled anomalies (-1): hinge(m + R^2 - d^2)^p scaled by Ω (pushed OUTSIDE)
#   - Graph Laplacian regularization on UNLABELED geometry (kNN over embeddings)
#       * Edge policy: labeled → unlabeled only; unlabeled → both
#   - NEW (DeepSAD-style) UNLABELED pull-in term:
#       * η_unl * mean_{y=0} d^2  (optional capping for robustness)
#
# Total batch loss (per-epoch also includes a full-batch graph step):
#   L = R^2
#       + C_slack * [ mean_{y=+1} hinge(d^2 - R^2)^p  +  Ω * mean_{y=-1} hinge(m + R^2 - d^2)^p ]
#       + η_unl * mean_{y=0} ρ(d^2)          # ρ(x)=x or min(x, R^2 + cap_offset) if cap enabled
#   Full-batch graph step (separate optimization step each epoch):
#       L_graph = λ_u * ( (R^2 - d^2)ᵀ L (R^2 - d^2) )  ≡ λ_u * d^2ᵀ L d^2  (since L·1 = 0)
#
# Label convention:
#   +1 = labeled normal (used in labeled-hinge)
#    0 = unlabeled      (used in graph and unlabeled pull-in)
#   -1 = labeled anomaly (used in labeled-hinge)
#
# API:
#   cfg = TrainConfig(in_dim=..., ...)
#   trainer = AAGDeepSVDDTrainer(cfg)
#   trainer.fit(X_all, y_all)            # tensors
#   scores = trainer.score(X_test)       # d^2 - R^2
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import NearestNeighbors
from scipy import sparse


# ============================== Encoder & Utils ==============================

class BiasFreeMLP(nn.Module):
    """Bias-free MLP. Avoid batch-norm to prevent center collapse."""
    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...] = (128, 64), out_dim: int = 32, act=nn.ReLU):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h, bias=False))  # <-- bias=False
            layers.append(act())
            prev = h
        layers.append(nn.Linear(prev, out_dim, bias=False))  # <-- bias=False
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def squared_distance_to_center(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Return d^2(x) = ||z - c||^2 for each row z in Z."""
    return ((z - c) ** 2).sum(dim=1)


# ============================ Graph Construction =============================

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
    """
    kNN graph on embeddings Z with label-aware edge policy:
      - if policy == 'labeled_to_unlabeled_only':
            labeled nodes connect only to unlabeled;
            unlabeled connect to both labeled and unlabeled.
      - if policy == 'all': standard kNN.
    Edge weights use a node-adaptive Gaussian with local σ_i via median neighbor distance.
    Returns combinatorial Laplacian L as torch.sparse_coo (float32) on `device`.
    """
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


# =============================== Training Core ===============================

@dataclass
class TrainConfig:
    # Encoder
    in_dim: int
    hidden: Tuple[int, ...] = (128, 64)
    out_dim: int = 32

    # Hinge settings
    p: int = 2                 # 1 = linear hinge, 2 = squared hinge
    nu: float = 0.1            # soft-boundary target fraction outside (for R^2 update)
    Omega: float = 2.0         # anomaly hinge weight multiplier
    margin_m: float = 1.0      # anomaly margin

    # Graph
    lambda_u: float = 0.1      # weight for graph Laplacian term (full-batch step)
    k: int = 15
    gamma_anom_edges: float = 1.0  # upweight edges touching anomalies in graph (>=1)
    graph_refresh: int = 2      # epochs between graph rebuilds (on current embeddings)

    # NEW: DeepSAD-style unlabeled pull-in
    eta_unl: float = 1.0        # weight for unlabeled mean d^2 pull-in
    cap_unlabeled: bool = False # if True, cap unlabeled d^2 by (R^2 + cap_offset)
    cap_offset: float = 0.5     # add-on to R^2 for the cap

    # Optimization
    wd: float = 1e-4
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 256
    warmup_epochs: int = 2

    # System
    device: str = 'cpu'
    print_every: int = 1


def _quantile_R2(d2_normals: torch.Tensor, q: float) -> float:
    return float(torch.quantile(d2_normals.detach(), q))


def _hinge(x: torch.Tensor, p: int) -> torch.Tensor:
    h = torch.clamp(x, min=0.0)
    return h * h if p == 2 else h


class AAGDeepSVDDTrainer:
    """
    Anomaly-Aware Graph-based Semi-Supervised Deep SVDD (AAG-DSVDD)
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.model = BiasFreeMLP(cfg.in_dim, cfg.hidden, cfg.out_dim).to(cfg.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        self.c: Optional[torch.Tensor] = None
        self.R2: float = 0.0
        self.L: Optional[torch.Tensor] = None  # torch.sparse_coo Laplacian

    # ----------------------------- internal utils -----------------------------

    def _warmup_and_set_center(self, X_norm: torch.Tensor):
        # Warmup on normals to stabilize encoder
        if self.cfg.warmup_epochs > 0:
            dl = DataLoader(TensorDataset(X_norm), batch_size=self.cfg.batch_size, shuffle=True)
            for _ in range(self.cfg.warmup_epochs):
                for (xb,) in dl:
                    xb = xb.to(self.cfg.device)
                    z = self.model(xb)
                    loss = (z ** 2).mean()
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
        # Center as mean embedding of normals
        with torch.no_grad():
            Z = self.model(X_norm.to(self.cfg.device))
            c = Z.mean(dim=0)
            # Small epsilon push to avoid exactly-zero coords (optional)
            eps = 1e-1
            c[(c.abs() < eps) & (c < 0)] = -eps
            c[(c.abs() < eps) & (c >= 0)] = eps
            self.c = c.detach()

    def _build_graph(self, X_all: torch.Tensor, y_all: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            Z = self.model(X_all.to(self.cfg.device)).cpu().numpy()
        labels = y_all.cpu().numpy().astype(np.int32)
        self.L = build_label_aware_knn_laplacian(
            Z, labels, k=self.cfg.k,
            upweight_anom_edges_gamma=self.cfg.gamma_anom_edges,
            row_normalize=True, device=self.cfg.device,
            policy='labeled_to_unlabeled_only', oversample_factor=3
        )

    def _graph_step(self, X_all: torch.Tensor) -> float:
        """Full-batch graph regularization step (separate optimizer step)."""
        if self.L is None or self.cfg.lambda_u <= 0.0:
            return 0.0
        self.model.train()
        z = self.model(X_all.to(self.cfg.device))
        d2 = squared_distance_to_center(z, self.c.to(self.cfg.device))
        # f = R^2 - d^2  →  fᵀLf == d^2ᵀ L d^2 since L·1=0
        Ld2 = torch.sparse.mm(self.L, d2.unsqueeze(1)).squeeze(1)
        L_graph = self.cfg.lambda_u * torch.sum(d2 * Ld2)
        self.opt.zero_grad()
        L_graph.backward()
        self.opt.step()
        return float(L_graph.detach().cpu())

    # ------------------------------- training ---------------------------------

    def _train_epoch(self, dl: DataLoader, y_all: torch.Tensor) -> float:
        self.model.train()
        total = 0.0
        device = self.cfg.device

        for (xb, idxb) in dl:
            xb = xb.to(device)
            idxb = idxb.to(device)
            yb = y_all[idxb]

            z = self.model(xb)
            d2 = squared_distance_to_center(z, self.c.to(device))

            # Masks
            mask_n = (yb == 1)     # labeled normals
            mask_a = (yb == -1)    # labeled anomalies
            mask_u = (yb == 0)     # unlabeled

            # Labeled hinges
            L_norm = _hinge(d2[mask_n] - self.R2, self.cfg.p).mean() / max(self.cfg.nu, 1e-8) if mask_n.any() else torch.tensor(0.0, device=device)
            L_anom = self.cfg.Omega * _hinge(self.cfg.margin_m + self.R2 - d2[mask_a], self.cfg.p).mean() if mask_a.any() else torch.tensor(0.0, device=device)

            # NEW: DeepSAD-style unlabeled pull-in (with optional cap)
            if mask_u.any():
                if self.cfg.cap_unlabeled:
                    tau = torch.tensor(self.R2 + float(self.cfg.cap_offset), dtype=xb.dtype, device=device)
                    L_unl = torch.minimum(d2[mask_u], tau).mean()
                else:
                    L_unl = d2[mask_u].mean()
            else:
                L_unl = torch.tensor(0.0, device=device)

            # Compose batch loss (no graph term here; it has its own full-batch step)
            loss = (self.R2
                    + L_norm
                    + L_anom
                    + self.cfg.eta_unl * L_unl)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            total += float(loss.detach().cpu())

        return total

    def fit(self, X_all: torch.Tensor, y_all: torch.Tensor):
        """
        X_all: (n, d) float tensor
        y_all: (n,) long tensor in {+1 labeled normal, 0 unlabeled, -1 labeled anomaly}
        """
        X_all = X_all.to(self.cfg.device)
        y_all = y_all.to(self.cfg.device)

        # Initialize from labeled normals only
        idx_normals = (y_all == 1).nonzero(as_tuple=True)[0]
        if idx_normals.numel() == 0:
            raise ValueError("Need at least one labeled normal (+1) to initialize center.")
        self._warmup_and_set_center(X_all[idx_normals])

        # Initialize R^2 from normals' (1 - nu) quantile
        with torch.no_grad():
            z_norm = self.model(X_all[idx_normals])
            d2_norm = squared_distance_to_center(z_norm, self.c.to(self.cfg.device))
            self.R2 = _quantile_R2(d2_norm, q=1.0 - self.cfg.nu)

        # Epoch loop
        n = X_all.shape[0]
        dl = DataLoader(TensorDataset(X_all, torch.arange(n)), batch_size=self.cfg.batch_size, shuffle=True)

        for epoch in range(1, self.cfg.epochs + 1):
            # Rebuild graph on current embeddings
            if (epoch - 1) % max(1, self.cfg.graph_refresh) == 0:
                self._build_graph(X_all, y_all)

            # Labeled-hinge + unlabeled pull-in batches
            base_loss = self._train_epoch(dl, y_all)

            # Update R^2 from labeled normals
            with torch.no_grad():
                z_norm = self.model(X_all[idx_normals])
                d2_norm = squared_distance_to_center(z_norm, self.c.to(self.cfg.device))
                self.R2 = _quantile_R2(d2_norm, q=1.0 - self.cfg.nu)

            # Full-batch graph regularization step
            graph_loss = self._graph_step(X_all)

            if self.cfg.print_every and (epoch % self.cfg.print_every == 0):
                print(f"Epoch {epoch:03d} | base={base_loss:.4f} | graph={graph_loss:.4f} | R2={self.R2:.6f}")

    # -------------------------------- scoring --------------------------------

    @torch.no_grad()
    def score(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns anomaly scores s(x) = d^2(x) - R^2  (positive ⇒ outside ⇒ anomalous).
        """
        self.model.eval()
        z = self.model(X.to(self.cfg.device))
        d2 = squared_distance_to_center(z, self.c.to(self.cfg.device))
        return d2 - self.R2


# ================================ Smoke test =================================
if __name__ == "__main__":
    # Minimal sanity check on synthetic data (not an experiment)
    torch.manual_seed(0)
    n_p, n_u, n_n, d = 200, 150, 40, 4
    X_p = torch.randn(n_p, d) * 0.8 + 0.0     # labeled normals
    X_u = torch.randn(n_u, d) * 0.9 + 0.5     # unlabeled (mixed)
    X_n = torch.randn(n_n, d) * 0.7 + 3.5     # labeled anomalies
    X = torch.vstack([X_p, X_u, X_n]).float()
    y = torch.tensor([1]*n_p + [0]*n_u + [-1]*n_n, dtype=torch.long)

    cfg = TrainConfig(
        in_dim=d, hidden=(64, 32), out_dim=16,
        p=2, nu=0.1, Omega=2.0, margin_m=1.0,
        lambda_u=0.1, k=15, gamma_anom_edges=1.0, graph_refresh=2,
        eta_unl=1.0, cap_unlabeled=True, cap_offset=0.5,
        wd=1e-4, lr=1e-3, epochs=5, batch_size=64, warmup_epochs=1,
        device='cpu', print_every=1
    )
    trainer = AAGDeepSVDDTrainer(cfg)
    trainer.fit(X, y)
    s = trainer.score(X)
    print("scores:", s[:10].cpu().numpy().round(4))
