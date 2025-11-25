# srs/baselines/deep_svdd.py
# -----------------------------------------------------------------------------
# Deep SVDD baseline (Ruff et al., 2018) — trains **only on labeled normals (+1)**
#
# Label convention expected by this baseline:
#   +1 = labeled normal  (used for training)
#    0 = unlabeled       (ignored during training)
#   -1 = anomaly         (ignored during training)
#
# Variants supported:
#   - "one-class":       minimize mean d^2(z, c) over labeled normals
#   - "soft-boundary":   R^2 + (1/nu) * mean( max(0, d^2(z, c) - R^2) ) over labeled normals
#
# Design choices aligned with the paper:
#   - Bias-free MLP encoder (no biases, no batch norm)
#   - Center c is fixed after initialization on labeled normals
#   - Optional squared-hinge (hinge_power=2) for smoother gradients (default = linear hinge)
#   - R^2 updated via (1 - nu)-quantile of d^2 on labeled normals (soft-boundary only)
#
# Usage example:
#   cfg = DeepSVDDConfig(in_dim=2, hidden=(64,32), out_dim=16, objective="soft-boundary",
#                        nu=0.1, epochs=50, device="cuda")
#   model = DeepSVDD(cfg)
#   model.fit(X_train, y_train)        # tensors; y in {+1, 0, -1}
#   scores = model.score(X_test)       # d^2 - R^2 (soft-boundary) or d^2 (one-class)
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------- Model utilities -------------------------------

class BiasFreeMLP(nn.Module):
    """Bias-free MLP used in Deep SVDD (no batch norm, no biases)."""
    def __init__(self, in_dim: int, hidden: Tuple[int, ...] = (128, 64), out_dim: int = 32, act=nn.ReLU):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h, bias=False), act()]
            d = h
        layers += [nn.Linear(d, out_dim, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def squared_distance_to_center(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Compute squared L2 distance to center c for each row in z."""
    return ((z - c) ** 2).sum(dim=1)


# --------------------------------- Config ------------------------------------

@dataclass
class DeepSVDDConfig:
    # Encoder
    in_dim: int
    hidden: Tuple[int, ...] = (128, 64)
    out_dim: int = 32

    # Objective
    objective: str = "soft-boundary"   # "one-class" or "soft-boundary"
    nu: float = 0.1                    # used only for soft-boundary
    hinge_power: int = 1               # 1 = linear hinge (paper); 2 = squared hinge

    # Optimization
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-6
    warmup_epochs: int = 2             # short warmup to stabilize encoder at start
    device: str = "cpu"
    print_every: int = 1               # epoch print frequency

    # Soft-boundary radius update
    r2_update_every: int = 1           # epochs between R^2 updates
    r2_start_epoch: int = 1            # start updating after this epoch
    center_eps: float = 1e-1           # epsilon to avoid exactly-zero center coords


# --------------------------------- Trainer -----------------------------------

class DeepSVDD:
    """
    Deep SVDD baseline that trains **only on labeled normals (+1)** and ignores
    unlabeled (0) and anomalies (-1) during training.
    """

    def __init__(self, cfg: DeepSVDDConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = BiasFreeMLP(cfg.in_dim, cfg.hidden, cfg.out_dim).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        self.c: Optional[torch.Tensor] = None  # center (on device)
        self.R2: float = 0.0                   # radius^2 (for soft-boundary)

    # --------------------------- internal helpers ---------------------------

    def _hinge(self, x: torch.Tensor) -> torch.Tensor:
        """Linear or squared hinge."""
        h = torch.clamp(x, min=0.0)
        return h * h if self.cfg.hinge_power == 2 else h

    @torch.no_grad()
    def _init_center(self, X_norm: torch.Tensor) -> torch.Tensor:
        """
        Initialize center c as mean of embeddings of labeled normals.
        Apply epsilon adjustment to avoid exactly-zero coordinates (per Ruff et al.).
        """
        self.model.eval()
        z = self.model(X_norm.to(self.device))
        c = z.mean(dim=0)
        eps = self.cfg.center_eps
        # push near-zeros away from 0 in a sign-consistent way
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c >= 0)] = eps
        return c.detach()

    @torch.no_grad()
    def _update_r2(self, X_norm: torch.Tensor):
        """Set R^2 to the (1 - nu)-quantile of d^2 over labeled normals."""
        self.model.eval()
        z = self.model(X_norm.to(self.device))
        d2 = squared_distance_to_center(z, self.c)
        q = max(0.0, min(1.0, 1.0 - float(self.cfg.nu)))
        self.R2 = float(torch.quantile(d2, q))

    # ---------------------------------- API ----------------------------------

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Train Deep SVDD using ONLY labeled normals (+1).
        Args:
            X: (n, d) float tensor
            y: (n,)   long tensor with values in {+1, 0, -1}
        """
        X = X.to(self.device)
        y = y.to(self.device)

        # Select labeled normals only
        idx_norm = (y == 1).nonzero(as_tuple=True)[0]
        if idx_norm.numel() == 0:
            raise ValueError("DeepSVDD: need at least one labeled normal (+1).")
        X_norm = X[idx_norm]

        # Warmup to stabilize embeddings before center init
        if self.cfg.warmup_epochs > 0:
            dl_warm = DataLoader(TensorDataset(X_norm), batch_size=self.cfg.batch_size, shuffle=True)
            for _ in range(self.cfg.warmup_epochs):
                for (xb,) in dl_warm:
                    xb = xb.to(self.device)
                    z = self.model(xb)
                    loss = (z ** 2).mean()
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

        # Initialize fixed center c
        self.c = self._init_center(X_norm)

        # Initialize radius for soft-boundary
        if self.cfg.objective.lower().startswith("soft"):
            self._update_r2(X_norm)

        # Main training loop
        dl = DataLoader(TensorDataset(X, y), batch_size=self.cfg.batch_size, shuffle=True)
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                # Only labeled normals contribute to loss
                mask = (yb == 1)
                if not mask.any():
                    continue

                z = self.model(xb)
                d2 = squared_distance_to_center(z, self.c)

                if self.cfg.objective.lower().startswith("soft"):
                    # R^2 is treated as a constant within the step; updated periodically
                    R2_t = torch.tensor(self.R2, dtype=xb.dtype, device=xb.device, requires_grad=False)
                    loss_norm = self._hinge(d2[mask] - R2_t).mean() / max(self.cfg.nu, 1e-8)
                    loss = R2_t + loss_norm
                else:
                    # One-class variant: minimize mean distance for normals
                    loss = d2[mask].mean()

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                epoch_loss += float(loss.detach().cpu())

            # Periodic R^2 update (soft-boundary)
            if self.cfg.objective.lower().startswith("soft"):
                if epoch >= self.cfg.r2_start_epoch and (epoch % max(1, self.cfg.r2_update_every) == 0):
                    self._update_r2(X_norm)

            if self.cfg.print_every and (epoch % self.cfg.print_every == 0):
                if self.cfg.objective.lower().startswith("soft"):
                    print(f"[DeepSVDD] epoch {epoch:03d}  loss={epoch_loss:.6f}  R2={self.R2:.6f}")
                else:
                    print(f"[DeepSVDD] epoch {epoch:03d}  loss={epoch_loss:.6f}")

    @torch.no_grad()
    def score(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return anomaly scores for X:
          - one-class:      d^2(x, c)
          - soft-boundary:  d^2(x, c) - R^2
        """
        if self.c is None:
            raise RuntimeError("DeepSVDD: model not fitted. Call fit() first.")
        self.model.eval()
        X = X.to(self.device)
        z = self.model(X)
        d2 = squared_distance_to_center(z, self.c)
        return d2 - self.R2 if self.cfg.objective.lower().startswith("soft") else d2


# ------------------------------- Smoke test ----------------------------------
if __name__ == "__main__":
    # Minimal sanity check on random data (do not use as a real experiment)
    torch.manual_seed(0)
    n_norm, n_unl, n_anom, d = 200, 100, 40, 4
    X_norm = torch.randn(n_norm, d) * 0.9 + 0.0
    X_unl  = torch.randn(n_unl,  d) * 1.2 + 2.5
    X_anom = torch.randn(n_anom, d) * 0.7 + 5.0
    X = torch.vstack([X_norm, X_unl, X_anom]).float()
    y = torch.tensor([1]*n_norm + [0]*n_unl + [-1]*n_anom, dtype=torch.long)

    cfg = DeepSVDDConfig(
        in_dim=d, hidden=(64, 32), out_dim=16,
        objective="soft-boundary", nu=0.1, hinge_power=1,
        epochs=10, batch_size=64, lr=1e-3, device="cpu"
    )
    model = DeepSVDD(cfg)
    model.fit(X, y)
    scores = model.score(X)
    print("scores shape:", tuple(scores.shape))
