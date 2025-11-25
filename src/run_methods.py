# run_methods_simple.py
# ----------------------------------------------------------------------
# Runner for: AAG-DSVDD (proposed), DeepSVDD, DeepSAD, OC-SVM, SVDD-RBF
# Datasets: banana or moons (2D), with tweakable generators.
# Metrics: F1, Recall, Precision, Accuracy, and TP/FP/TN/FN (overall).
# Plots: decision boundary (score contour; dashed boundary line) + TEST pts.
# CLI: full control of data + model params; CSV append logging.
# ----------------------------------------------------------------------

import os, csv, argparse, types, inspect
from dataclasses import dataclass
from typing import Tuple, Dict, Callable, List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, accuracy_score,
)
    # comment to keep: metrics used
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons


# ============================== CLI helpers ==============================

def _parse_int_tuple(s: str) -> Tuple[int, ...]:
    s = s.strip()
    if not s:
        return tuple()
    return tuple(int(x) for x in s.split(','))

def _parse_pair(s: str) -> tuple[float, float]:
    t = tuple(float(x) for x in s.split(','))
    if len(t) != 2:
        raise ValueError(f'Expected "a,b" but got: {s}')
    return t

def _ensure_dir(p: str):
    if p:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

def append_csv_row(csv_path: str, fieldnames: List[str], row: Dict):
    _ensure_dir(csv_path)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ============================== dataset makers ==============================

# --- Moons (simple) ---

@dataclass
class MoonsCfg:
    n_norm: int = 600
    n_anom: int = 240
    noise: float = 0.08
    gap: float = 0.5
    spread: float = 0.6
    seed: int = 0

def make_moons_data(cfg: MoonsCfg):
    rng = np.random.default_rng(cfg.seed)
    Xn, _ = make_moons(n_samples=cfg.n_norm, noise=cfg.noise, random_state=cfg.seed)
    Xn = Xn.astype(np.float32)
    xa = rng.normal(loc=(0.5, 1.5 + cfg.gap), scale=cfg.spread, size=(cfg.n_anom, 2)).astype(np.float32)
    X = np.vstack([Xn, xa])
    y = np.hstack([np.zeros(len(Xn), dtype=int), np.ones(len(xa), dtype=int)])
    return X, y

# --- Banana (Friedman banana + two anomaly lobes) ---

@dataclass
class BananaAdvCfg:
    n_norm: int = 600
    n_anom: int = 240
    anom_split: float = 0.5
    b: float = 0.2
    s1: float = 2.0
    s2: float = 1.5
    rotate_deg: float = 90.0
    mu_a1: Tuple[float, float] = (0.0, 4.9)
    mu_a2: Tuple[float, float] = (0.0, -4.0)
    cov_a1: Tuple[float, float] = (0.04, 0.81)
    cov_a2: Tuple[float, float] = (0.04, 0.81)
    seed: int = 0

def _rotate_points(X: np.ndarray, degrees: float) -> np.ndarray:
    if degrees == 0.0:
        return X.astype(np.float32)
    theta = np.deg2rad(degrees)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], dtype=float)
    return (X @ R.T).astype(np.float32)

def make_banana_data(cfg: BananaAdvCfg):
    rng = np.random.default_rng(cfg.seed)
    # normals
    u1 = rng.normal(0.0, cfg.s1, size=cfg.n_norm)
    u2 = rng.normal(0.0, cfg.s2, size=cfg.n_norm)
    Xn = np.stack([u1, u2 + cfg.b * (u1**2 - cfg.s1**2)], axis=1).astype(np.float32)
    Xn = _rotate_points(Xn, cfg.rotate_deg)
    # anomalies
    n1 = int(round(cfg.anom_split * cfg.n_anom)); n2 = cfg.n_anom - n1
    C1 = np.array([[cfg.cov_a1[0], 0.0], [0.0, cfg.cov_a1[1]]], dtype=float)
    C2 = np.array([[cfg.cov_a2[0], 0.0], [0.0, cfg.cov_a2[1]]], dtype=float)
    X1 = rng.multivariate_normal(mean=cfg.mu_a1, cov=C1, size=n1).astype(np.float32)
    X2 = rng.multivariate_normal(mean=cfg.mu_a2, cov=C2, size=n2).astype(np.float32)
    Xa = np.vstack([X1, X2])
    Xa = _rotate_points(Xa, cfg.rotate_deg)
    X = np.vstack([Xn, Xa]).astype(np.float32)
    y = np.hstack([np.zeros(len(Xn), dtype=int), np.ones(len(Xa), dtype=int)])
    return X, y


# ========================== semi-supervised splits ==========================

def split_for_semi_supervised(
    X, y, test_size=0.3, val_size=0.2, seed=0,
    label_frac_anom=1.0,      # fraction of train anomalies labeled (-1)
    label_frac_norm=1.0,      # fraction of train normals labeled (+1)
    label_normals_subset_only=False
):
    """
    Returns: (train_X,y_lab), (val_X,y_val), (test_X,y_test)
    y_lab in {+1 labeled normal, 0 unlabeled, -1 labeled anomaly}
    y_val/test in {0 normal, 1 anomaly} for scoring/metrics.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=val_size, random_state=seed, stratify=y_tr)

    rng = np.random.default_rng(seed)
    y_lab = np.zeros_like(y_tr, dtype=int)

    norm_idx = np.where(y_tr == 0)[0]
    eff_norm_frac = 0.0 if (label_normals_subset_only and label_frac_norm >= 1.0) else float(np.clip(label_frac_norm, 0.0, 1.0))
    n_lab_norm = int(round(eff_norm_frac * len(norm_idx)))
    if n_lab_norm > 0:
        sel_n = rng.choice(norm_idx, size=n_lab_norm, replace=False)
        y_lab[sel_n] = +1

    anom_idx = np.where(y_tr == 1)[0]
    eff_anom_frac = float(np.clip(label_frac_anom, 0.0, 1.0))
    n_lab_anom = int(round(eff_anom_frac * len(anom_idx)))
    if n_lab_anom > 0:
        sel_a = rng.choice(anom_idx, size=n_lab_anom, replace=False)
        y_lab[sel_a] = -1

    return (X_tr, y_lab), (X_va, y_va), (X_te, y_te)


# ============================== scoring helpers ==============================

def pick_threshold_max_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    uniq = np.unique(scores)
    cands = np.unique(np.concatenate([uniq, np.array([0.0], dtype=scores.dtype)]))
    best_f1, best_thr = -1.0, 0.0
    for thr in cands:
        yhat = (scores > thr).astype(int)
        f1 = f1_score(y_true, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)}

def svdd_like_scores_from_net(model, X: np.ndarray, device='cpu') -> np.ndarray:
    """
    Works for AAG-DSVDD / DeepSVDD / DeepSAD-style models exposing:
      - .model (preferred) or .net (fallback) or .network
      - .trainer or self with attributes: c (center), R or R2 (radius or radius^2)
    Computes s(x) = d^2 - R^2 (falls back to d^2 if R/R2 missing).
    """
    device = torch.device(device if (device=='cuda' and torch.cuda.is_available()) else 'cpu')

    # Prefer trained .model, then .net, then .network
    net = None
    maybe = getattr(model, 'model', None)
    if isinstance(maybe, torch.nn.Module):
        net = maybe
    if net is None:
        maybe = getattr(model, 'net', None)
        if isinstance(maybe, torch.nn.Module):
            net = maybe
    if net is None and hasattr(model, 'network'):
        net = getattr(model, 'network')

    if net is None:
        raise RuntimeError("No network found on model (expected .model/.net/.network).")

    net = net.to(device).eval()

    holder = getattr(model, 'trainer', model)
    c_src = getattr(holder, 'c', None)
    if isinstance(c_src, torch.Tensor):
        c = c_src.detach().clone().to(device).float()
    else:
        # infer last linear out_features or fallback to 2
        try:
            last = [m for m in net.modules() if isinstance(m, torch.nn.Linear)][-1]
            out_dim = last.out_features
        except Exception:
            out_dim = 2
        c = torch.zeros(out_dim, device=device, dtype=torch.float32) if c_src is None else torch.as_tensor(c_src, dtype=torch.float32, device=device)

    if hasattr(holder, 'R2'):
        R2 = torch.tensor(float(getattr(holder, 'R2')), dtype=torch.float32, device=device)
    else:
        R_src = getattr(holder, 'R', 0.0)
        R2 = (torch.as_tensor(R_src, dtype=torch.float32, device=device) ** 2)

    out = []
    bs = 4096
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            xb = torch.from_numpy(X[i:i+bs].astype(np.float32)).to(device)
            z = net(xb)
            d2 = torch.sum((z - c) ** 2, dim=1)
            s = d2 - R2
            out.append(s.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


# ============================== plotting ==============================

def plot_decision_boundary(score_fn: Callable[[np.ndarray], np.ndarray],
                           splits: Tuple[Tuple[np.ndarray, np.ndarray],
                                         Tuple[np.ndarray, np.ndarray],
                                         Tuple[np.ndarray, np.ndarray]],
                           thr: float,
                           title: str,
                           savepath: str,
                           grid_res: int = 300,
                           show_unlabeled: bool = False):
    (Xtr, ytr_lab), (Xva, yva), (Xte, yte) = splits
    Xall = np.vstack([Xtr, Xva, Xte])
    pad = 0.6
    xlim = (Xall[:, 0].min() - pad, Xall[:, 0].max() + pad)
    ylim = (Xall[:, 1].min() - pad, Xall[:, 1].max() + pad)
    xs = np.linspace(*xlim, grid_res)
    ys = np.linspace(*ylim, grid_res)
    XX, YY = np.meshgrid(xs, ys)
    P = np.stack([XX.ravel(), YY.ravel()], axis=1)
    S = score_fn(P).reshape(XX.shape)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=600)
    cs = ax.contourf(XX, YY, S, levels=40, cmap='cividis')
    cbar = fig.colorbar(cs); cbar.set_label('Anomaly score s(x)')
    ax.contour(XX, YY, S, levels=[thr], colors='white', linestyles='dashed', linewidths=2)

    # --- Distinct observation styling ---
    # Train labeled normals (+1)
    idx_p = (ytr_lab == 1)
    # Train labeled anomalies (-1)
    idx_n = (ytr_lab == -1)
    # Train unlabeled (0)
    idx_u = (ytr_lab == 0)

    # Training points
    ax.scatter(Xtr[idx_p,0], Xtr[idx_p,1], s=20, c='#1b9e77', marker='o', label='Train Normal (+1)')
    if show_unlabeled:
        ax.scatter(Xtr[idx_u,0], Xtr[idx_u,1], s=18, c='#7570b3', marker='D', label='Train Unlabeled (0)')
    # Always show labeled anomalies
    ax.scatter(Xtr[idx_n,0], Xtr[idx_n,1], s=22, c='#d95f02', marker='^', label='Train Labeled Anom (-1)')

    # Test points
    ax.scatter(Xte[yte==0,0], Xte[yte==0,1], s=24, c='#666666', marker='s', label='Test Normal')
    ax.scatter(Xte[yte==1,0], Xte[yte==1,1], s=26, c='#e7298a', marker='v', label='Test Anomaly')

    ax.legend(framealpha=0.9, fontsize=12)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel('x1', fontsize=16); ax.set_ylabel('x2', fontsize=16)
    ax.set_title(title, fontsize=18)
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(savepath); plt.close(fig)
    
def plot_training_data(Xtr: np.ndarray,
                       ytr_lab: np.ndarray,
                       title: str,
                       savepath: str,
                       show_unlabeled: bool = True):
    """
    Simple scatter plot of the TRAIN split only, using the same
    styling and font sizes as the decision-boundary plot.
    """
    pad = 0.6
    xlim = (Xtr[:, 0].min() - pad, Xtr[:, 0].max() + pad)
    ylim = (Xtr[:, 1].min() - pad, Xtr[:, 1].max() + pad)

    # Label masks in the semi-supervised scheme
    idx_p = (ytr_lab == 1)   # labeled normals
    idx_n = (ytr_lab == -1)  # labeled anomalies
    idx_u = (ytr_lab == 0)   # unlabeled

    fig, ax = plt.subplots(figsize=(7, 6), dpi=600)

    # Match colors/markers to plot_decision_boundary
    if np.any(idx_p):
        ax.scatter(Xtr[idx_p, 0], Xtr[idx_p, 1],
                   s=20, c='#1b9e77', marker='o', label='Train Normal (+1)')
    if show_unlabeled and np.any(idx_u):
        ax.scatter(Xtr[idx_u, 0], Xtr[idx_u, 1],
                   s=18, c='#7570b3', marker='D', label='Train Unlabeled (0)')
    if np.any(idx_n):
        ax.scatter(Xtr[idx_n, 0], Xtr[idx_n, 1],
                   s=22, c='#d95f02', marker='^', label='Train Labeled Anom (-1)')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('x1', fontsize=16)
    ax.set_ylabel('x2', fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.legend(framealpha=0.9, fontsize=12)

    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)



# ============================== trainers (adapters) ==============================

def _ensure_mlp(model, hidden_dims: Tuple[int, ...], rep_dim: int, bias: bool = False):
    """
    Only build an MLP if the baseline does NOT already provide one.
    - If model exposes a trained `.model` (preferred) or `.net`, do NOTHING.
    - Else, try model.set_network(...).
    - Else, attach a minimal MLP at `model.net`.
    """
    if isinstance(getattr(model, 'model', None), torch.nn.Module):
        return
    if isinstance(getattr(model, 'net', None), torch.nn.Module):
        return

    if hasattr(model, 'set_network'):
        try:
            model.set_network('mlp', input_dim=2,
                              hidden_dims=(hidden_dims if hidden_dims else (64, 32)),
                              rep_dim=rep_dim, bias=bias)
            return
        except Exception:
            pass

    # Fallback: attach a simple MLP
    hd = hidden_dims if hidden_dims else (64, 32)
    layers, in_dim = [], 2
    for h in hd:
        layers += [torch.nn.Linear(in_dim, h, bias=True), torch.nn.ReLU()]
        in_dim = h
    layers += [torch.nn.Linear(in_dim, rep_dim, bias=True)]
    model.net = torch.nn.Sequential(*layers)


def train_aag_dsvdd(Xtr, ytr_lab, args):
    from models.aag_dsvdd import AAGDeepSVDDTrainer, TrainConfig
    hidden = _parse_int_tuple(args.aag_hidden)
    cfg = TrainConfig(
        in_dim=2, hidden=hidden if hidden else (128, 64),
        out_dim=args.aag_out_dim,
        p=args.aag_p, nu=args.aag_nu, Omega=args.aag_Omega, margin_m=args.aag_margin,
        lambda_u=args.aag_lambda_u, k=args.aag_k, gamma_anom_edges=args.aag_gamma_edges,
        graph_refresh=args.aag_graph_refresh,
        eta_unl=args.aag_eta_unl, cap_unlabeled=args.aag_cap_unlabeled,
        cap_offset=args.aag_cap_offset,
        wd=args.aag_wd, lr=args.aag_lr, epochs=args.aag_epochs,
        batch_size=args.aag_batch, warmup_epochs=args.aag_warmup,
        device=args.device, print_every=args.print_every
    )
    trainer = AAGDeepSVDDTrainer(cfg)
    trainer.fit(torch.from_numpy(Xtr.astype(np.float32)),
                torch.from_numpy(ytr_lab.astype(np.int64)))

    class Wrap:
        def __init__(self, trainer):
            self.trainer = trainer
            self.model = trainer.model      # expose as .model
            self.net = trainer.model        # alias as .net for scorer compatibility
    return Wrap(trainer)


# ----- DeepSVDD & DeepSAD now use their dataclass configs directly -----

def train_deepsvdd(Xtr, ytr_lab, args):
    from baselines.deep_svdd import DeepSVDD, DeepSVDDConfig
    hidden = _parse_int_tuple(args.dsvdd_hidden) or (64, 32)
    cfg = DeepSVDDConfig(
        in_dim=2,
        hidden=hidden,
        out_dim=args.dsvdd_rep_dim,
        objective="soft-boundary",
        nu=args.dsvdd_nu,
        hinge_power=2,
        epochs=args.dsvdd_epochs,
        batch_size=args.dsvdd_batch,
        lr=args.dsvdd_lr,
        weight_decay=args.dsvdd_wd,
        warmup_epochs=args.dsvdd_warmup,
        device=args.device,
        print_every=args.print_every,
        r2_update_every=1,
        r2_start_epoch=1,
        center_eps=args.center_eps,
    )
    model = DeepSVDD(cfg)
    model.fit(torch.from_numpy(Xtr.astype(np.float32)),
              torch.from_numpy(ytr_lab.astype(np.int64)))
    return model

def train_deepsad(Xtr, ytr_lab, args):
    from baselines.deep_sad import DeepSAD, DeepSADConfig
    hidden = _parse_int_tuple(args.dsad_hidden) or (64, 32)
    cfg = DeepSADConfig(
        in_dim=2,
        hidden=hidden,
        out_dim=args.dsad_rep_dim,
        eta=args.dsad_nu,      # interpret --dsad-nu as η (labeled-term weight)
        eps_inv=1e-6,
        epochs=args.dsad_epochs,
        batch_size=args.dsad_batch,
        lr=args.dsad_lr,
        weight_decay=args.dsad_wd,
        warmup_epochs=args.dsad_warmup,
        device=args.device,
        print_every=args.print_every,
        center_eps=args.center_eps,
    )
    model = DeepSAD(cfg)
    model.fit(torch.from_numpy(Xtr.astype(np.float32)),
              torch.from_numpy(ytr_lab.astype(np.int64)))
    return model


def train_ocsvm(Xtr, ytr_lab, args):
    from baselines.ocsvm import OCSVMWrapper
    oc = OCSVMWrapper(nu=args.ocsvm_nu, kernel='rbf', gamma=args.ocsvm_gamma)
    Xn = Xtr[ytr_lab == 1]
    oc.train((np.asarray(Xn), np.zeros(len(Xn), dtype=int)))
    return oc

def train_ksvdd(Xtr, ytr_lab, args):
    from baselines.svdd_rbf import KernelSVDD
    Xn = Xtr[ytr_lab == 1]
    model = KernelSVDD(nu=args.ksvdd_nu, gamma=args.ksvdd_gamma)
    try:
        class _XYTrainWrapper:
            def __init__(self, X, y): self.train_set = (np.asarray(X), np.asarray(y))
        model.fit_on_dataset(_XYTrainWrapper(Xn, np.zeros(len(Xn), int)))
    except Exception:
        model.fit(Xn)
    return model


# ============================== main experiment ==============================

RESULT_FIELDS = [
    'dataset','seed','model',
    'F1','Recall','Precision','Accuracy',
    'TP','FP','TN','FN',
    'thr'
]

def run_one(args, dataset: str, seed: int):
    np.random.seed(seed); torch.manual_seed(seed)

    if dataset == 'banana':
        cfg = BananaAdvCfg(
            n_norm=args.n_norm,
            n_anom=args.n_anom,
            anom_split=args.anom_split,
            b=args.b, s1=args.s1, s2=args.s2,
            rotate_deg=args.rotate_deg,
            mu_a1=_parse_pair(args.mu_a1),
            mu_a2=_parse_pair(args.mu_a2),
            cov_a1=_parse_pair(args.cov_a1),
            cov_a2=_parse_pair(args.cov_a2),
            seed=seed
        )
        X, y = make_banana_data(cfg)
    elif dataset == 'moons':
        X, y = make_moons_data(MoonsCfg(
            n_norm=args.n_norm, n_anom=args.n_anom,
            noise=args.noise, gap=args.gap, spread=args.spread, seed=seed
        ))
    else:
        raise ValueError("dataset must be 'banana' or 'moons'")

    (Xtr, ytr_lab), (Xva, yva), (Xte, yte) = split_for_semi_supervised(
        X, y,
        test_size=args.test_size, val_size=args.val_size, seed=seed,
        label_frac_anom=args.label_frac_anom,
        label_frac_norm=args.label_frac_norm,
        label_normals_subset_only=args.label_normals_subset_only
    )
    
    if args.plot:
        plot_training_data(
            Xtr,
            ytr_lab,
            title="Training Dataset",
            savepath=os.path.join(args.out_dir, f"{dataset}_seed{seed}_train_split.png"),
            show_unlabeled=True,
        )
    

    models: List[Tuple[str, object, Callable[[np.ndarray], np.ndarray]]] = []

    if args.run_aag:
        aag = train_aag_dsvdd(Xtr, ytr_lab, args)
        models.append(("AAG-DSVDD", aag, lambda X_: svdd_like_scores_from_net(aag, X_, device=args.device)))

    if args.run_deepsvdd:
        try:
            dsvdd = train_deepsvdd(Xtr, ytr_lab, args)
            models.append(("DeepSVDD", dsvdd, lambda X_: svdd_like_scores_from_net(dsvdd, X_, device=args.device)))
        except Exception as e:
            print(f"[warn] DeepSVDD failed: {e}")

    if args.run_deepsad:
        try:
            dsad = train_deepsad(Xtr, ytr_lab, args)
            models.append(("DeepSAD", dsad, lambda X_: svdd_like_scores_from_net(dsad, X_, device=args.device)))
        except Exception as e:
            print(f"[warn] DeepSAD failed: {e}")

    if args.run_ocsvm:
        try:
            oc = train_ocsvm(Xtr, ytr_lab, args)
            models.append(("OC-SVM", oc, lambda X_: (-oc.model.decision_function(np.asarray(X_))).ravel()))
        except Exception as e:
            print(f"[warn] OC-SVM failed: {e}")

    if args.run_ksvdd:
        try:
            ksvdd = train_ksvdd(Xtr, ytr_lab, args)
            if hasattr(ksvdd, 'decision_function'):
                models.append(("SVDD-RBF", ksvdd, lambda X_: ksvdd.decision_function(np.asarray(X_)).ravel()))
            else:
                models.append(("SVDD-RBF", ksvdd, lambda X_: ksvdd.score(np.asarray(X_)).ravel()))
        except Exception as e:
            print(f"[warn] SVDD-RBF failed: {e}")

    os.makedirs(args.out_dir, exist_ok=True)
    for name, model, score_fn in models:
        s_val = score_fn(Xva)
        assert len(s_val) == len(yva), f"s_val ({len(s_val)}) and yva ({len(yva)}) must match!"
        thr = pick_threshold_max_f1(yva, s_val)

        s_test = score_fn(Xte)
        yhat = (s_test > thr).astype(int)

        f1  = f1_score(yte, yhat, zero_division=0)
        rec = recall_score(yte, yhat, zero_division=0)
        prc = precision_score(yte, yhat, zero_division=0)
        acc = accuracy_score(yte, yhat)
        counts = confusion_counts(yte, yhat)

        row = {
            'dataset': dataset, 'seed': seed, 'model': name,
            'F1': float(f1), 'Recall': float(rec), 'Precision': float(prc), 'Accuracy': float(acc),
            **counts, 'thr': float(thr),
        }
        append_csv_row(args.csv_path, RESULT_FIELDS, row)

        if args.plot:
            plot_decision_boundary(
                score_fn,
                splits=((Xtr, ytr_lab), (Xva, yva), (Xte, yte)),
                thr=thr,
                title=f"{name}",
                savepath=os.path.join(args.out_dir, f"{dataset}_seed{seed}_{name.replace(' ','_').replace('/','-')}.png"),
                grid_res=args.grid_res,
                show_unlabeled=(name in ("AAG-DSVDD", "DeepSAD"))
            )

        print(f"[{name}] {dataset} s{seed} | F1={f1:.3f} Rec={rec:.3f} Prec={prc:.3f} Acc={acc:.3f} "
              f"| TP={counts['TP']} FP={counts['FP']} TN={counts['TN']} FN={counts['FN']} | thr={thr:.4f}")

def main():
    p = argparse.ArgumentParser(description="Run AAG-DSVDD and baselines on 2D datasets with CSV logging.")
    # General
    p.add_argument('--dataset', type=str, default='banana', choices=['banana','moons'])
    p.add_argument('--seeds', type=str, default='0')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--out-dir', type=str, default='plots')
    p.add_argument('--csv-path', type=str, default='logs/results_simple.csv')
    p.add_argument('--plot', action='store_true')
    p.add_argument('--grid-res', type=int, default=300)
    p.add_argument('--print-every', type=int, default=1)

    # Splits / labeling
    p.add_argument('--test-size', type=float, default=0.30)
    p.add_argument('--val-size', type=float, default=0.20)
    p.add_argument('--label-frac-anom', type=float, default=1.0, dest='label_frac_anom')
    p.add_argument('--label-frac-norm', type=float, default=1.0, dest='label_frac_norm')
    p.add_argument('--label-normals-subset-only', action='store_true')

    # Dataset shape/scale (both)
    p.add_argument('--n-norm', type=int, default=600)
    p.add_argument('--n-anom', type=int, default=240)
    p.add_argument('--noise', type=float, default=0.06)

    # Banana (accept both short and namespaced flags)
    p.add_argument('--b', type=float, default=0.2, dest='b')
    p.add_argument('--banana-b', type=float, dest='b')
    p.add_argument('--s1', type=float, default=2.0, dest='s1')
    p.add_argument('--banana-s1', type=float, dest='s1')
    p.add_argument('--s2', type=float, default=1.5, dest='s2')
    p.add_argument('--banana-s2', type=float, dest='s2')
    p.add_argument('--rotate-deg', type=float, default=90.0, dest='rotate_deg')
    p.add_argument('--banana-rotate-deg', type=float, dest='rotate_deg')
    p.add_argument('--anom-split', type=float, default=0.5, dest='anom_split')
    p.add_argument('--banana-anom-split', type=float, dest='anom_split')
    p.add_argument('--mu-a1', type=str, default='0.0,4.9', dest='mu_a1')
    p.add_argument('--banana-mu-a1', type=str, dest='mu_a1')
    p.add_argument('--mu-a2', type=str, default='0.0,-4.0', dest='mu_a2')
    p.add_argument('--banana-mu-a2', type=str, dest='mu_a2')
    p.add_argument('--cov-a1', type=str, default='0.04,0.81', dest='cov_a1')
    p.add_argument('--banana-cov-a1', type=str, dest='cov_a1')
    p.add_argument('--cov-a2', type=str, default='0.04,0.81', dest='cov_a2')
    p.add_argument('--banana-cov-a2', type=str, dest='cov_a2')

    # Legacy banana args (ignored but kept for compat)
    p.add_argument('--bend', type=float, default=0.2)
    p.add_argument('--scale-x', type=float, default=1.0)
    p.add_argument('--scale-y', type=float, default=1.0)

    # Moons (also accept namespaced)
    p.add_argument('--gap', type=float, default=0.5, dest='gap')
    p.add_argument('--moons-gap', type=float, dest='gap')
    p.add_argument('--spread', type=float, default=0.6, dest='spread')
    p.add_argument('--moons-spread', type=float, dest='spread')
    p.add_argument('--moons-noise', type=float, dest='noise')

    # Model toggles
    p.add_argument('--run-aag', action='store_true')
    p.add_argument('--run-deepsvdd', action='store_true')
    p.add_argument('--run-deepsad', action='store_true')
    p.add_argument('--run-ocsvm', action='store_true')
    p.add_argument('--run-ksvdd', action='store_true')

    # AAG-DSVDD params
    p.add_argument('--aag-hidden', type=str, default='128,64')
    p.add_argument('--aag-out-dim', type=int, default=16)
    p.add_argument('--aag-p', type=int, default=2)
    p.add_argument('--aag-nu', type=float, default=0.1)
    p.add_argument('--aag-Omega', type=float, default=2.0)
    p.add_argument('--aag-margin', type=float, default=1.0)
    p.add_argument('--aag-lambda-u', type=float, default=0.1)
    p.add_argument('--aag-k', type=int, default=15)
    p.add_argument('--aag-gamma-edges', type=float, default=1.0)
    p.add_argument('--aag-graph-refresh', type=int, default=2)
    p.add_argument('--aag-eta-unl', type=float, default=1.0)
    p.add_argument('--aag-cap-unlabeled', action='store_true')
    p.add_argument('--aag-cap-offset', type=float, default=0.5)
    p.add_argument('--aag-lr', type=float, default=1e-3)
    p.add_argument('--aag-wd', type=float, default=1e-4)
    p.add_argument('--aag-epochs', type=int, default=50)
    p.add_argument('--aag-batch', type=int, default=128)
    p.add_argument('--aag-warmup', type=int, default=2)

    # DeepSVDD params
    p.add_argument('--dsvdd-hidden', type=str, default='64,32')
    p.add_argument('--dsvdd-rep-dim', type=int, default=16)
    p.add_argument('--dsvdd-nu', type=float, default=0.1)
    p.add_argument('--dsvdd-lr', type=float, default=1e-3)
    p.add_argument('--dsvdd-wd', type=float, default=1e-6)
    p.add_argument('--dsvdd-epochs', type=int, default=50)
    p.add_argument('--dsvdd-batch', type=int, default=128)
    p.add_argument('--dsvdd-warmup', type=int, default=0)

    # DeepSAD params (accept --dsad-nu for runner; mapped to eta)
    p.add_argument('--dsad-hidden', type=str, default='64,32')
    p.add_argument('--dsad-rep-dim', type=int, default=16)
    p.add_argument('--dsad-nu', type=float, default=0.1)   # interpreted as eta (labeled-term weight)
    p.add_argument('--dsad-lr', type=float, default=1e-3)
    p.add_argument('--dsad-wd', type=float, default=1e-6)
    p.add_argument('--dsad-epochs', type=int, default=50)
    p.add_argument('--dsad-batch', type=int, default=128)
    p.add_argument('--dsad-warmup', type=int, default=0)

    # OC-SVM params
    p.add_argument('--ocsvm-nu', type=float, default=0.05)
    p.add_argument('--ocsvm-gamma', type=str, default='scale')

    # Kernel SVDD params
    p.add_argument('--ksvdd-nu', type=float, default=0.05)
    p.add_argument('--ksvdd-gamma', type=float, default=0.2)

    # Shared Deep cfg bits some baselines require explicitly
    p.add_argument('--center-eps', type=float, default=1e-3, dest='center_eps')

    args = p.parse_args()

    for seed in [int(s) for s in args.seeds.split(',') if s.strip()]:
        run_one(args, args.dataset, seed)

if __name__ == "__main__":
    main()
