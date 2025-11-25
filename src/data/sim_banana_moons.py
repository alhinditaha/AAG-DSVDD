# srs/data/sim_banana_moons.py
# ----------------------------------------------------------------------
# Synthetic generators for Banana & Two-Moons with customizable
# distributions and semi-supervised labeling.
#
# Train labels:  +1 = labeled normal, 0 = unlabeled (mix), -1 = labeled anomaly
# Test labels:    0 = normal (GT),    1 = anomaly (GT)
#
# You can specify *counts* or *fractions* of labeled/unlabeled/test
# for normals and anomalies. If any fraction is provided, you must also
# provide total_norm/total_anom; counts will be computed from fractions.
# Otherwise, explicit counts are used as-is.
# ----------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------- utils --------------------------------

def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else 42)

def rotate_points(X: np.ndarray, degrees: float) -> np.ndarray:
    if degrees == 0.0:
        return X.copy()
    theta = np.deg2rad(degrees)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], dtype=X.dtype)
    return (X @ R.T).astype(X.dtype, copy=False)

def sample_mog(n: int,
               means: np.ndarray,
               covs: np.ndarray,
               weights: Optional[np.ndarray] = None,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Mixture-of-Gaussians sampler (2D)."""
    rng = _rng(None) if rng is None else rng
    k = len(means)
    if weights is None:
        weights = np.ones(k) / k
    else:
        weights = np.asarray(weights, float)
        weights = weights / weights.sum()
    comp = rng.choice(k, size=n, p=weights)
    X = np.zeros((n, 2), float)
    for i in range(k):
        idx = (comp == i)
        if not np.any(idx):
            continue
        X[idx] = rng.multivariate_normal(mean=means[i], cov=covs[i], size=idx.sum())
    return X

def standardize(X: np.ndarray, mean: Optional[np.ndarray]=None, std: Optional[np.ndarray]=None):
    if mean is None: mean = X.mean(axis=0)
    if std  is None: std  = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std

# ------------------------- labeling helpers ----------------------------

@dataclass
class LabelingConfig:
    """
    You can use explicit counts OR provide fractions with totals.

    If any fraction is not None, you must set total_norm and total_anom.
    Fractions are in [0,1]. If f_test_* is None, test counts are the
    leftover after labeled + unlabeled.

    Counts (used directly if no fractions provided):
      n_lab_norm, n_unl_norm, n_test_norm
      n_lab_anom, n_unl_anom, n_test_anom
    """
    # Totals (required when using fractions)
    total_norm: Optional[int] = None
    total_anom: Optional[int] = None

    # Fractions (optional)
    f_lab_norm: Optional[float] = None
    f_unl_norm: Optional[float] = None
    f_test_norm: Optional[float] = None

    f_lab_anom: Optional[float] = None
    f_unl_anom: Optional[float] = None
    f_test_anom: Optional[float] = None

    # Counts (used if all fractions are None)
    n_lab_norm: int = 80
    n_unl_norm: int = 1200
    n_test_norm: int = 1000

    n_lab_anom: int = 40
    n_unl_anom: int = 300
    n_test_anom: int = 300

    # Standardization & seed
    standardize_all: bool = True
    seed: Optional[int] = 123

def _resolve_labeling_counts(L: LabelingConfig) -> Tuple[Dict[str,int], int, int]:
    """
    Returns (counts_dict, total_norm, total_anom)
    counts_dict keys: n_lab_norm, n_unl_norm, n_test_norm, n_lab_anom, n_unl_anom, n_test_anom
    """
    use_fracs = any(v is not None for v in
                    (L.f_lab_norm, L.f_unl_norm, L.f_test_norm,
                     L.f_lab_anom, L.f_unl_anom, L.f_test_anom))
    if not use_fracs:
        tot_norm = L.n_lab_norm + L.n_unl_norm + L.n_test_norm
        tot_anom = L.n_lab_anom + L.n_unl_anom + L.n_test_anom
        counts = dict(
            n_lab_norm=L.n_lab_norm, n_unl_norm=L.n_unl_norm, n_test_norm=L.n_test_norm,
            n_lab_anom=L.n_lab_anom, n_unl_anom=L.n_unl_anom, n_test_anom=L.n_test_anom,
        )
        return counts, tot_norm, tot_anom

    # Fractions path
    if L.total_norm is None or L.total_anom is None:
        raise ValueError("When using fractions you must set total_norm and total_anom.")

    def _three_from_fracs(total: int, f_lab: Optional[float], f_unl: Optional[float], f_test: Optional[float]):
        def _to_int(x: float) -> int:
            return int(np.floor(x + 1e-9))
        # default test to leftover
        f_lab = 0.0 if f_lab is None else max(0.0, min(1.0, f_lab))
        f_unl = 0.0 if f_unl is None else max(0.0, min(1.0, f_unl))
        if f_test is None:
            f_test = max(0.0, 1.0 - f_lab - f_unl)
        else:
            f_test = max(0.0, min(1.0, f_test))

        n_lab = _to_int(total * f_lab)
        n_unl = _to_int(total * f_unl)
        n_test = total - n_lab - n_unl  # ensure exact total

        # If explicit f_test was provided, try to match it by adjusting unlabeled
        desired_n_test = _to_int(total * f_test)
        if desired_n_test != n_test:
            n_test = desired_n_test
            n_unl  = total - n_lab - n_test

        if min(n_lab, n_unl, n_test) < 0:
            raise ValueError("Negative split sizes after rounding; check fractions.")
        return n_lab, n_unl, n_test

    n_lab_norm, n_unl_norm, n_test_norm = _three_from_fracs(
        L.total_norm, L.f_lab_norm, L.f_unl_norm, L.f_test_norm
    )
    n_lab_anom, n_unl_anom, n_test_anom = _three_from_fracs(
        L.total_anom, L.f_lab_anom, L.f_unl_anom, L.f_test_anom
    )

    counts = dict(
        n_lab_norm=n_lab_norm, n_unl_norm=n_unl_norm, n_test_norm=n_test_norm,
        n_lab_anom=n_lab_anom, n_unl_anom=n_unl_anom, n_test_anom=n_test_anom,
    )
    return counts, L.total_norm, L.total_anom

def split_and_label(Xn: np.ndarray, Xa: np.ndarray,
                    n_lab_norm: int,
                    n_lab_anom: int,
                    n_unl_norm: int,
                    n_unl_anom: int,
                    rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build semi-supervised TRAIN set with +1/0/-1 labels.
    """
    idx_n = rng.permutation(len(Xn))
    idx_a = rng.permutation(len(Xa))

    X_lab_n = Xn[idx_n[:n_lab_norm]]
    X_unl_n = Xn[idx_n[n_lab_norm:n_lab_norm+n_unl_norm]]

    X_lab_a = Xa[idx_a[:n_lab_anom]]
    X_unl_a = Xa[idx_a[n_lab_anom:n_lab_anom+n_unl_anom]]

    X_train = np.vstack([X_lab_n, X_lab_a, X_unl_n, X_unl_a]).astype(np.float32, copy=False)
    y_train = np.concatenate([
        np.full(len(X_lab_n), +1, int),   # labeled normals
        np.full(len(X_lab_a), -1, int),   # labeled anomalies
        np.full(len(X_unl_n),  0, int),   # unlabeled normals
        np.full(len(X_unl_a),  0, int),   # unlabeled anomalies
    ])

    p = rng.permutation(len(X_train))
    return X_train[p], y_train[p]

def plot_train(X_train: np.ndarray, y_train: np.ndarray, title: str = ""):
    """Scatter plot of TRAIN set colored by semi-supervised labels."""
    mask_p = (y_train == +1)
    mask_n = (y_train == -1)
    mask_u = (y_train ==  0)
    plt.figure(figsize=(6.5, 5.5))
    if mask_u.any(): plt.scatter(X_train[mask_u,0], X_train[mask_u,1], s=16, alpha=0.6, label="Unlabeled (0)")
    if mask_p.any(): plt.scatter(X_train[mask_p,0], X_train[mask_p,1], s=28, marker='o', label="Labeled normal (+1)")
    if mask_n.any(): plt.scatter(X_train[mask_n,0], X_train[mask_n,1], s=28, marker='x', label="Labeled anomaly (-1)")
    plt.gca().set_aspect('equal', 'box')
    plt.legend()
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.tight_layout()
    plt.show()

# --------------------------- Banana generator --------------------------

@dataclass
class BananaCfg:
    """
    Friedman-style banana (normals) + two anomaly lobes on the convex side.
    Both lobes are considered ONE anomaly class.

    Parameters mirror your attached banana.py.
    """
    # Friedman banana params
    b: float = 0.2        # curvature
    s1: float = 2.0       # std for u1 ~ N(0, s1)
    s2: float = 1.5       # std for u2 ~ N(0, s2)
    rotate_deg: float = 90.0

    # Anomaly lobes
    anom_split: float = 0.5   # fraction of anomalies in lobe-1 (rest in lobe-2)
    mu_a1: Tuple[float, float] = (0.0, 4.9)
    mu_a2: Tuple[float, float] = (0.0, -4.0)
    cov_a1: Tuple[float, float] = (0.04, 0.81)  # diagonal (var_x, var_y)
    cov_a2: Tuple[float, float] = (0.04, 0.81)  # diagonal (var_x, var_y)

def _banana_normals(n: int, cfg: BananaCfg, rng: np.random.Generator) -> np.ndarray:
    """Normals: Friedman banana, then rotate."""
    u1 = rng.normal(0.0, cfg.s1, size=n)
    u2 = rng.normal(0.0, cfg.s2, size=n)
    X = np.stack([u1, u2 + cfg.b * (u1**2 - cfg.s1**2)], axis=1)  # center by s1**2
    X = rotate_points(X.astype(np.float32, copy=False), cfg.rotate_deg)
    return X

def _banana_anomalies(n: int, cfg: BananaCfg, rng: np.random.Generator) -> np.ndarray:
    """Anomalies: two diagonal Gaussians (single class), then rotate."""
    # clamp to [0,1] for safety
    split = float(np.clip(cfg.anom_split, 0.0, 1.0))
    n1 = int(round(split * n))
    n2 = n - n1
    cov1 = np.array([[cfg.cov_a1[0], 0.0], [0.0, cfg.cov_a1[1]]], dtype=float)
    cov2 = np.array([[cfg.cov_a2[0], 0.0], [0.0, cfg.cov_a2[1]]], dtype=float)
    X1 = rng.multivariate_normal(cfg.mu_a1, cov1, size=n1)
    X2 = rng.multivariate_normal(cfg.mu_a2, cov2, size=n2)
    X = np.vstack([X1, X2]).astype(np.float32, copy=False)
    X = rotate_points(X, cfg.rotate_deg)
    return X

def make_banana_dataset(labeling: LabelingConfig,
                        banana_cfg: BananaCfg) -> Dict[str, np.ndarray]:
    rng = _rng(labeling.seed)
    counts, tot_norm, tot_anom = _resolve_labeling_counts(labeling)

    # Generate pools using derived totals
    Xn_pool = _banana_normals(tot_norm, banana_cfg, rng)
    Xa_pool = _banana_anomalies(tot_anom, banana_cfg, rng)

    # Standardize globally (recommended for MLPs)
    if labeling.standardize_all:
        allX = np.vstack([Xn_pool, Xa_pool]).astype(np.float32, copy=False)
        allX_std, mean, std = standardize(allX)
        allX_std = allX_std.astype(np.float32, copy=False)
        Xn_pool = allX_std[:len(Xn_pool)]
        Xa_pool = allX_std[len(Xn_pool):]
    else:
        mean = np.zeros(2, dtype=float); std = np.ones(2, dtype=float)

    # Build TRAIN (semi-supervised)
    X_train, y_train = split_and_label(
        Xn_pool[:counts['n_lab_norm'] + counts['n_unl_norm']],
        Xa_pool[:counts['n_lab_anom'] + counts['n_unl_anom']],
        counts['n_lab_norm'], counts['n_lab_anom'],
        counts['n_unl_norm'], counts['n_unl_anom'], rng
    )

    # Build TEST (pure GT 0/1)
    rem_n = Xn_pool[counts['n_lab_norm'] + counts['n_unl_norm'] :
                    counts['n_lab_norm'] + counts['n_unl_norm'] + counts['n_test_norm']]
    rem_a = Xa_pool[counts['n_lab_anom'] + counts['n_unl_anom'] :
                    counts['n_lab_anom'] + counts['n_unl_anom'] + counts['n_test_anom']]
    X_test = np.vstack([rem_n, rem_a]).astype(np.float32, copy=False)
    y_test = np.concatenate([np.zeros(len(rem_n), int), np.ones(len(rem_a), int)])
    p = rng.permutation(len(X_test))
    X_test, y_test = X_test[p], y_test[p]

    return dict(
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        mean=mean, std=std,
        counts=counts
    )

# --------------------------- Two-moons generator -----------------------

@dataclass
class MoonsConfig:
    n_total_per_moon: int = 2000
    noise: float = 0.08           # isotropic Gaussian noise for moons
    gap_shift: float = 0.0        # vertical shift of the lower moon (bridge control)
    scale: float = 1.0            # scale both axes
    rotate_deg: float = 0.0       # rotate moons for variety

    # Anomaly options:
    #   - "gap_blob": dense blob in the gap/bridge
    #   - "global_mog": mixture of Gaussians far away
    anomaly_mode: str = "gap_blob"
    gap_blob_mean: Tuple[float, float] = (1.0, 0.25)
    gap_blob_cov: Tuple[float, float, float, float] = (0.12, 0.0, 0.0, 0.12)  # 2x2

    mog_means: Optional[np.ndarray] = None
    mog_covs: Optional[np.ndarray] = None
    mog_weights: Optional[np.ndarray] = None

def _make_clean_moons(n_per_moon: int, rng: np.random.Generator) -> np.ndarray:
    # Upper moon: angle in [0, pi]
    t1 = rng.uniform(0, np.pi, size=n_per_moon)
    x1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    # Lower moon: angle in [0, pi] then shifted right and down
    t2 = rng.uniform(0, np.pi, size=n_per_moon)
    x2 = np.stack([1 - np.cos(t2), -np.sin(t2) - 0.5], axis=1)
    return np.vstack([x1, x2])

def sample_moons_normals(n: int, cfg: MoonsConfig, rng: np.random.Generator) -> np.ndarray:
    # ensure n is even-ish
    n1 = n // 2
    n2 = n - n1
    base = _make_clean_moons(max(n1, n2), rng)
    X = np.vstack([base[:n1], base[:max(n1, n2)][-n2:]])
    # noise, scale, rotate, shift lower moon
    X = X + rng.normal(0.0, cfg.noise, size=X.shape)
    X = (X * cfg.scale).astype(np.float32, copy=False)
    X = rotate_points(X, cfg.rotate_deg)
    # shift lower half (second moon) by gap_shift on y
    X[n1:, 1] += cfg.gap_shift
    return X.astype(np.float32, copy=False)

def sample_moons_anomalies(n: int, cfg: MoonsConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.anomaly_mode == "gap_blob":
        m = np.array(cfg.gap_blob_mean, float)
        a,b,c,d = cfg.gap_blob_cov
        C = np.array([[a,b],[c,d]], float)
        X = rng.multivariate_normal(m, C, size=n).astype(np.float32, copy=False)
        return X
    # default global MOG if provided, else a simple far blob
    if cfg.mog_means is not None and cfg.mog_covs is not None:
        return sample_mog(n, cfg.mog_means, cfg.mog_covs, cfg.mog_weights, rng).astype(np.float32, copy=False)
    means = np.array([[2.5,  2.0],
                      [-2.0, 2.2]], float)
    covs  = np.array([[[0.3,0.0],[0.0,0.3]],
                      [[0.3,0.0],[0.0,0.3]]], float)
    return sample_mog(n, means, covs, None, rng).astype(np.float32, copy=False)

def make_moons_dataset(labeling: LabelingConfig,
                       moons_cfg: MoonsConfig) -> Dict[str, np.ndarray]:
    rng = _rng(labeling.seed)
    counts, tot_norm, tot_anom = _resolve_labeling_counts(labeling)

    # Generate pools
    Xn_pool = sample_moons_normals(tot_norm, moons_cfg, rng)
    Xa_pool = sample_moons_anomalies(tot_anom, moons_cfg, rng)

    # Standardize globally
    if labeling.standardize_all:
        allX = np.vstack([Xn_pool, Xa_pool]).astype(np.float32, copy=False)
        allX_std, mean, std = standardize(allX)
        allX_std = allX_std.astype(np.float32, copy=False)
        Xn_pool = allX_std[:len(Xn_pool)]
        Xa_pool = allX_std[len(Xn_pool):]
    else:
        mean = np.zeros(2, dtype=float); std = np.ones(2, dtype=float)

    # Build TRAIN (semi-supervised)
    X_train, y_train = split_and_label(
        Xn_pool[:counts['n_lab_norm'] + counts['n_unl_norm']],
        Xa_pool[:counts['n_lab_anom'] + counts['n_unl_anom']],
        counts['n_lab_norm'], counts['n_lab_anom'],
        counts['n_unl_norm'], counts['n_unl_anom'], rng
    )

    # Build TEST (pure GT 0/1)
    rem_n = Xn_pool[counts['n_lab_norm'] + counts['n_unl_norm'] :
                    counts['n_lab_norm'] + counts['n_unl_norm'] + counts['n_test_norm']]
    rem_a = Xa_pool[counts['n_lab_anom'] + counts['n_unl_anom'] :
                    counts['n_lab_anom'] + counts['n_unl_anom'] + counts['n_test_anom']]
    X_test = np.vstack([rem_n, rem_a]).astype(np.float32, copy=False)
    y_test = np.concatenate([np.zeros(len(rem_n), int), np.ones(len(rem_a), int)])
    p = rng.permutation(len(X_test))
    X_test, y_test = X_test[p], y_test[p]

    return dict(
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        mean=mean, std=std,
        counts=counts
    )

# ------------------------------ demo / CLI ----------------------------

if __name__ == "__main__":
    # Example A: BANANA with FRACTIONS
    lab_frac = LabelingConfig(
        total_norm=1800, total_anom=600,
        f_lab_norm=0.05, f_unl_norm=0.60,  # -> test_norm = 0.35
        f_lab_anom=0.10, f_unl_anom=0.40,  # -> test_anom = 0.50
        standardize_all=True, seed=123
    )
    banana_cfg = BananaCfg(
        b=0.2, s1=2.0, s2=1.5, rotate_deg=90.0,
        anom_split=0.5, mu_a1=(0.0, 4.9), mu_a2=(0.0, -4.0),
        cov_a1=(0.04, 0.81), cov_a2=(0.04, 0.81),
    )
    banana = make_banana_dataset(lab_frac, banana_cfg)
    plot_train(banana["X_train"], banana["y_train"], title="Banana – Training (fractions)")

    # Example B: MOONS with explicit COUNTS
    lab_counts = LabelingConfig(
        n_lab_norm=80, n_unl_norm=1200, n_test_norm=1000,
        n_lab_anom=40, n_unl_anom=300,  n_test_anom=300,
        standardize_all=True, seed=321
    )
    moons_cfg = MoonsConfig(
        noise=0.08, gap_shift=0.10, scale=1.0, rotate_deg=10.0,
        anomaly_mode="gap_blob",
        gap_blob_mean=(1.0, 0.15),
        gap_blob_cov=(0.10, 0.0, 0.0, 0.10),
    )
    moons = make_moons_dataset(lab_counts, moons_cfg)
    plot_train(moons["X_train"], moons["y_train"], title="Two Moons – Training (counts)")
