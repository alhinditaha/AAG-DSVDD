# srs/baselines/graph_svdd_paper_qp.py
# -----------------------------------------------------------------------------
# Graph-based Semi-Supervised SVDD — EXACT QP (kernel, no anomalies in training)
#
# This implementation follows the paper "Graph-based semi-supervised Support Vector
# Data Description for novelty detection" using a convex Quadratic Program (QP).
#
# Key points (paper-faithful):
#   • Variables are the SVDD dual coefficients α over **labeled normals only**.
#   • Center in RKHS: a = Σ_{i∈L} α_i Φ(x_i), with constraints:
#         α ≥ 0,  1ᵀα = 1,  α_i ≤ C_box   (C_box corresponds to the slack weight)
#   • Unlabeled points are used **only** in the graph Laplacian regularizer term.
#   • Labeled anomalies are **ignored** (excluded from training), as requested.
#   • Graph is built in INPUT space: kNN, **single global** Gaussian bandwidth σ_graph,
#     symmetrized W, combinatorial Laplacian L = D − W, and the smoothness is on d².
#
# Formulation solved (convex QP in α):
#
#   Let indices split as L (labeled normals, size n_L) and U (unlabeled, size n_U).
#   Define
#       K_LL ∈ ℝ^{n_L×n_L}  (kernel among L),
#       K_allL ∈ ℝ^{(n_L+n_U)×n_L}  (kernel between all used points and L),
#       c_all = diag(K_all),  c_L = diag(K_LL),
#       A = -2 * K_allL.
#   Also build Laplacian L ∈ ℝ^{(n_L+n_U)×(n_L+n_U)} (symmetric PSD).
#
#   The objective becomes:
#       minimize_α   αᵀ [ K_LL + 4*C_graph * (K_allLᵀ L K_allL) ] α
#                     + [ -c_Lᵀ - 4*C_graph * (c_allᵀ L K_allL) ] α   + const
#   subject to       1ᵀ α = 1,   0 ≤ α ≤ C_box
#
# After solving for α, compute:
#       s = αᵀ K_LL α,
#       d²_L = c_L − 2 K_LL α + s,
#   and set the radius R² as the mean d² over "boundary" support vectors {i∈L : 0 < α_i < C_box},
#   or fall back to the median if the strict set is empty.
#
# Scoring new points x:
#       d²(x) = k(x,x) − 2 k(x, X_L) α + s
#       score(x) = d²(x) − R²   (positive ⇒ outside ⇒ anomalous)
#
# Dependencies: numpy, scipy, scikit-learn, cvxpy
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import numpy as np
import cvxpy as cp
from sklearn.neighbors import NearestNeighbors
from scipy import sparse


# ------------------------------ Configuration -------------------------------

@dataclass
class GraphSVDDPaperQPConfig:
    # Kernel (RBF) bandwidth; if None → median heuristic
    sigma_kernel: Optional[float] = None

    # Graph parameters (INPUT space, paper-faithful)
    k: int = 15
    sigma_graph: Optional[float] = None
    edge_policy: Literal["labeled_to_unlabeled_only", "all"] = "labeled_to_unlabeled_only"

    # Box upper bound for α (maps to slack weight C in SVDD dual)
    C_box: float = 1.0

    # Graph regularizer weight
    C_graph: float = 0.1

    # Solver
    solver: str = "OSQP"   # options: "OSQP", "SCS", "ECOS" (depending on your cvxpy install)
    solver_kwargs: dict = None

    # Tolerance when selecting boundary SVs for R²
    sv_tol: float = 1e-6

    # If no boundary SVs exist, use "median" or "mean" over d²_L to set R²
    fallback_radius: Literal["median", "mean"] = "median"


# ------------------------------- Utilities ----------------------------------

def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    # ||x||^2 + ||y||^2 − 2 x·y  (safe, nonnegative)
    x2 = np.sum(X * X, axis=1, keepdims=True)
    sq = x2 + x2.T - 2.0 * (X @ X.T)
    np.maximum(sq, 0.0, out=sq)
    return sq


def _rbf_kernel_from_sq(sq: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-sq / (2.0 * sigma * sigma))


def _median_heuristic_sigma(X: np.ndarray, k: int = 7) -> float:
    # median distance to the k-th neighbor as a robust global bandwidth
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)))
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    kth = dists[:, -1]
    s = float(np.median(kth[kth > 0])) if np.any(kth > 0) else float(np.mean(kth))
    return max(s, 1e-6)


def _build_graph_laplacian_input(
    X: np.ndarray, y: np.ndarray, k: int, sigma_graph: float, policy: str
) -> sparse.csr_matrix:
    """
    Build symmetric kNN graph in INPUT space with a global σ_graph.
    Uses ONLY y in {+1 (labeled normal), 0 (unlabeled)}.
    Edge policy:
      - 'labeled_to_unlabeled_only': labeled normals connect only to unlabeled; unlabeled connect to both
      - 'all': every node connects to its k nearest neighbors
    If not enough unlabeled neighbors exist for a labeled node, the remainder is filled by nearest neighbors.
    """
    n = X.shape[0]
    Koversample = min(max(k * 3 + 1, k + 1), n)

    nn = NearestNeighbors(n_neighbors=Koversample)
    nn.fit(X)
    dists, idxs = nn.kneighbors(X, return_distance=True)

    rows, cols, vals = [], [], []
    two_sig2 = 2.0 * sigma_graph * sigma_graph

    for i in range(n):
        yi = int(y[i])  #+1 or 0 only (anomalies already dropped)
        neigh = idxs[i, 1:]  # drop self
        di = dists[i, 1:]

        chosen_idx, chosen_dist = [], []

        if policy == "labeled_to_unlabeled_only" and yi == 1:
            # labeled normals → only unlabeled targets
            for j, dij in zip(neigh, di):
                if y[j] == 0:
                    chosen_idx.append(j); chosen_dist.append(dij)
                    if len(chosen_idx) >= k: break
            # fallback if not enough unlabeled
            if len(chosen_idx) < k:
                for j, dij in zip(neigh, di):
                    if j not in chosen_idx:
                        chosen_idx.append(j); chosen_dist.append(dij)
                        if len(chosen_idx) >= k: break
        else:
            # unlabeled or policy "all" → first k neighbors
            chosen_idx = neigh[:k]; chosen_dist = di[:k]

        for j, dij in zip(chosen_idx, chosen_dist):
            w = float(np.exp(-(dij * dij) / two_sig2))
            if w > 0.0:
                rows.append(i); cols.append(j); vals.append(w)

    # Directed kNN → symmetrize W
    W = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64).tocsr()
    W = 0.5 * (W + W.T)

    # Combinatorial Laplacian L = D − W
    d = np.asarray(W.sum(axis=1)).reshape(-1)
    L = sparse.diags(d, format="csr") - W
    return L


# ------------------------------- Main Class ----------------------------------

class GraphSVDDPaperQPNoAnom:
    """
    EXACT QP solver for Graph-based Semi-Supervised SVDD (kernel, unlabeled + labeled normals).
    Labeled anomalies are dropped (not used in the QP), per your comparison protocol.

    API:
        fit(X, y)   where y ∈ {+1 (labeled normal), 0 (unlabeled), −1 (anomaly → ignored)}
        score(Xnew) returns s(x) = d²(x) − R²  (positive ⇒ anomalous)
    """

    def __init__(self, cfg: GraphSVDDPaperQPConfig):
        self.cfg = cfg

        # Trained state
        self.alpha: Optional[np.ndarray] = None   # (n_L,)
        self.R2: Optional[float] = None
        self.X_L: Optional[np.ndarray] = None     # (n_L, d)
        self.K_LL: Optional[np.ndarray] = None    # (n_L, n_L)
        self.sigma_kernel_: Optional[float] = None
        self.sigma_graph_: Optional[float] = None

        # Cache for fast scoring
        self._alpha_K_alpha: Optional[float] = None

    # ------------------------------ Fitting ---------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the QP:
          • Keeps only y ∈ {+1, 0}; drops y == −1 (anomalies).
          • α is defined ONLY over the labeled-normal set L.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        # Drop labeled anomalies
        keep = (y != -1)
        X_used = X[keep]
        y_used = y[keep]

        # Split indices
        idx_L = np.where(y_used == 1)[0]
        idx_U = np.where(y_used == 0)[0]
        if idx_L.size == 0:
            raise ValueError("Need at least one labeled normal (+1).")

        X_L = X_used[idx_L]                        # (n_L, d)
        X_all = X_used                             # (n_L + n_U, d)

        n_L = X_L.shape[0]
        n_all = X_all.shape[0]

        # --- Kernel(s) ---
        # Kernel bandwidth for SVDD
        sigma_k = self.cfg.sigma_kernel or _median_heuristic_sigma(X_L)
        self.sigma_kernel_ = sigma_k

        # Gram blocks
        sq_LL = _pairwise_sq_dists(X_L)
        K_LL = _rbf_kernel_from_sq(sq_LL, sigma_k)                 # (n_L, n_L)

        sq_all = _pairwise_sq_dists(X_all)
        # We need diag over all points (for graph's linear term), and K_allL for A
        K_all_all_diag = np.diag(_rbf_kernel_from_sq(sq_all, sigma_k))  # (n_all,)
        # Cross-kernel K_allL: between ALL used points and L
        # Compute via distances between X_all and X_L
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
        X_all_2 = np.sum(X_all * X_all, axis=1, keepdims=True)
        X_L_2 = np.sum(X_L * X_L, axis=1, keepdims=True)
        sq_allL = X_all_2 + X_L_2.T - 2.0 * (X_all @ X_L.T)
        np.maximum(sq_allL, 0.0, out=sq_allL)
        K_allL = _rbf_kernel_from_sq(sq_allL, sigma_k)             # (n_all, n_L)

        c_all = K_all_all_diag                                    # vector length n_all
        c_L = np.diag(K_LL).copy()                                # vector length n_L

        # --- Graph (INPUT space, global σ_graph) ---
        sigma_g = self.cfg.sigma_graph or _median_heuristic_sigma(X_all, k=self.cfg.k)
        self.sigma_graph_ = sigma_g
        L = _build_graph_laplacian_input(
            X_all, y_used, k=self.cfg.k, sigma_graph=sigma_g, policy=self.cfg.edge_policy
        )  # csr (n_all, n_all)

        # --- Build the QP objective:  αᵀ Q α + bᵀ α + const ---
        # A = -2 * K_allL  →  Aᵀ L A = 4 * K_allLᵀ L K_allL
        # Linear: 2 * c_allᵀ L A = -4 * (c_allᵀ L K_allL)
        Q_graph = 4.0 * (K_allL.T @ (L @ K_allL))                 # (n_L, n_L)
        b_graph = -4.0 * ((L @ c_all).T @ K_allL)                 # (n_L,)

        Q = K_LL + self.cfg.C_graph * Q_graph
        b = (-c_L) + self.cfg.C_graph * b_graph

        # Symmetrize Q to be safe
        Q = 0.5 * (Q + Q.T)

        # --- Solve QP with cvxpy ---
        alpha = cp.Variable(n_L)
        P = cp.psd_wrap(Q)  # ensure symmetric
        objective = 0.5 * cp.quad_form(alpha, P) + b @ alpha

        constraints = [
            cp.sum(alpha) == 1.0,
            alpha >= 0.0,
            alpha <= self.cfg.C_box
        ]

        prob = cp.Problem(cp.Minimize(objective), constraints)
        solver_kwargs = self.cfg.solver_kwargs or {}
        prob.solve(solver=self.cfg.solver, **solver_kwargs)

        if alpha.value is None:
            raise RuntimeError(f"CVXPY failed to solve the QP (status={prob.status}). "
                               f"Try a different solver or adjust regularization.")

        alpha_opt = np.asarray(alpha.value, dtype=np.float64)

        # --- Compute R² from boundary support vectors on L ---
        s = float(alpha_opt @ (K_LL @ alpha_opt))                 # αᵀ K_LL α
        d2_L = c_L - 2.0 * (K_LL @ alpha_opt) + s                 # (n_L,)

        tol = self.cfg.sv_tol
        mask_boundary = (alpha_opt > tol) & (alpha_opt < self.cfg.C_box - tol)
        if np.any(mask_boundary):
            R2 = float(np.mean(d2_L[mask_boundary]))
        else:
            # fallback: median or mean over labeled normals
            if self.cfg.fallback_radius == "mean":
                R2 = float(np.mean(d2_L))
            else:
                R2 = float(np.median(d2_L))

        # --- Save state for scoring ---
        self.alpha = alpha_opt
        self.R2 = R2
        self.X_L = X_L
        self.K_LL = K_LL
        self._alpha_K_alpha = s

    # ------------------------------- Scoring --------------------------------

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns s(x) = d²(x) − R²  (positive ⇒ more anomalous)
        """
        if self.alpha is None or self.R2 is None:
            raise RuntimeError("Call fit() before score().")

        X = np.asarray(X, dtype=np.float64)
        # k(x,x)
        x2 = np.sum(X * X, axis=1, keepdims=True)
        l2 = np.sum(self.X_L * self.X_L, axis=1, keepdims=True)
        sq_xL = x2 + l2.T - 2.0 * (X @ self.X_L.T)
        np.maximum(sq_xL, 0.0, out=sq_xL)

        k_xL = _rbf_kernel_from_sq(sq_xL, self.sigma_kernel_)     # (m, n_L)

        # k(x,x) — using the same RBF bandwidth
        # Diagonal of RBF for a vector with itself is 1.0 (since ||x-x||^2=0).
        # Keep explicit in case you swap kernels later.
        k_xx = np.ones(X.shape[0], dtype=np.float64)

        v = k_xL @ self.alpha                                     # (m,)
        d2 = k_xx - 2.0 * v + self._alpha_K_alpha
        return d2 - self.R2


# ----------------------------- Convenience alias -----------------------------

def build_paper_qp_noanom(
    sigma_kernel: Optional[float] = None,
    k: int = 15,
    sigma_graph: Optional[float] = None,
    edge_policy: str = "labeled_to_unlabeled_only",
    C_box: float = 1.0,
    C_graph: float = 0.1,
    solver: str = "OSQP",
    solver_kwargs: Optional[dict] = None,
    sv_tol: float = 1e-6,
    fallback_radius: str = "median",
) -> Tuple[GraphSVDDPaperQPNoAnom, GraphSVDDPaperQPConfig]:
    cfg = GraphSVDDPaperQPConfig(
        sigma_kernel=sigma_kernel, k=k, sigma_graph=sigma_graph, edge_policy=edge_policy,
        C_box=C_box, C_graph=C_graph, solver=solver, solver_kwargs=solver_kwargs or {},
        sv_tol=sv_tol, fallback_radius=fallback_radius
    )
    return GraphSVDDPaperQPNoAnom(cfg), cfg
