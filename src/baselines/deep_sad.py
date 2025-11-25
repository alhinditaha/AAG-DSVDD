# srs/baselines/deep_sad.py
# -----------------------------------------------------------------------------
# Deep SAD (Ruff et al., ICLR 2020) baseline
# Uses: labeled normals (+1), labeled anomalies (-1), and unlabeled (0)
#
# Objective (mini-batch estimate):
#   Let U = {i: y_i = 0}, P = {j: y_j = +1}, N = {k: y_k = -1}.
#   z = φ(x; W), c fixed after init (excluding anomalies).
#
#   L_batch = ( sum_{i∈U} ||z_i - c||^2  +  η * [ sum_{j∈P} ||z_j - c||^2
#                                               + sum_{k∈N} 1/(||z_k - c||^2 + ε) ] )
#             / max(1, |U| + |P| + |N|)
#
# Notes:
# - Bias-free MLP (no BN, no biases) to avoid collapse, center is fixed after init.
# - Weight decay handled by optimizer (λ).
# - Anomaly score: s(x) = ||φ(x) - c|| (Euclidean).
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------- Model utilities -------------------------------

class BiasFreeMLP(nn.Module):
    """Bias-free MLP (no BN, no biases) as recommended for Deep SVDD/SAD."""
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


def sqdist_to_center(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """||z - c||^2 per row."""
    return ((z - c) ** 2).sum(dim=1)


# --------------------------------- Config ------------------------------------

@dataclass
class DeepSADConfig:
    # Encoder
    in_dim: int
    hidden: Tuple[int, ...] = (128, 64)
    out_dim: int = 32

    # Loss mixing
    eta: float = 1.0        # weight for labeled term
    eps_inv: float = 1e-6   # ε for inverse distance term (stability)

    # Optimization
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-6
    warmup_epochs: int = 2
    device: str = "cpu"
    print_every: int = 1

    # Center init
    center_eps: float = 1e-1  # push near-zero coords away from 0 to avoid collapse


# --------------------------------- Trainer -----------------------------------

class DeepSAD:
    """
    Deep SAD baseline:
      - Unlabeled (y==0):       squared distance to center
      - Labeled normals (y==+1): squared distance to center
      - Labeled anomalies (y==-1): inverse squared distance to center
    Center c is fixed after initialization (computed on all non-anomalous points).
    """

    def __init__(self, cfg: DeepSADConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = BiasFreeMLP(cfg.in_dim, cfg.hidden, cfg.out_dim).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        self.c: Optional[torch.Tensor] = None  # center on device

    # --------------------------- internal helpers ---------------------------

    @torch.no_grad()
    def _init_center(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Initialize center c as mean of embeddings over non-anomalous data (y != -1).
        If labeled normals exist, they dominate; otherwise unlabeled are used.
        """
        self.model.eval()
        mask_non_anom = (y != -1)
        if not mask_non_anom.any():
            raise ValueError("DeepSAD: need at least one non-anomalous sample (y != -1) to init center.")
        z = self.model(X[mask_non_anom].to(self.device))
        c = z.mean(dim=0)
        eps = self.cfg.center_eps
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c >= 0)] = eps
        return c.detach()

    # ---------------------------------- API ----------------------------------

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Args:
            X: (n, d) float tensor
            y: (n,) long tensor with values in {+1 (labeled normal), 0 (unlabeled), -1 (labeled anomaly)}
        """
        X = X.to(self.device)
        y = y.to(self.device)

        # Warmup on all non-anomalous points (stabilize embeddings)
        mask_non_anom = (y != -1)
        if self.cfg.warmup_epochs > 0 and mask_non_anom.any():
            dl_warm = DataLoader(TensorDataset(X[mask_non_anom]), batch_size=self.cfg.batch_size, shuffle=True)
            for _ in range(self.cfg.warmup_epochs):
                for (xb,) in dl_warm:
                    z = self.model(xb)
                    loss = (z ** 2).mean()
                    self.opt.zero_grad(); loss.backward(); self.opt.step()

        # Fix center c (exclude anomalies)
        self.c = self._init_center(X, y)

        # Main training loop
        dl = DataLoader(TensorDataset(X, y), batch_size=self.cfg.batch_size, shuffle=True)
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for xb, yb in dl:
                xb = xb.to(self.device); yb = yb.to(self.device)
                z = self.model(xb)
                d2 = sqdist_to_center(z, self.c)

                mask_u = (yb == 0)
                mask_p = (yb == 1)
                mask_n = (yb == -1)

                # Per-set sums (0 if empty)
                term_u = d2[mask_u].sum() if mask_u.any() else torch.tensor(0.0, device=xb.device)
                term_p = d2[mask_p].sum() if mask_p.any() else torch.tensor(0.0, device=xb.device)
                if mask_n.any():
                    term_n = torch.sum(1.0 / (d2[mask_n] + self.cfg.eps_inv))
                else:
                    term_n = torch.tensor(0.0, device=xb.device)

                count = int(mask_u.sum() + mask_p.sum() + mask_n.sum())
                if count == 0:
                    continue

                loss = (term_u + self.cfg.eta * (term_p + term_n)) / float(count)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                epoch_loss += float(loss.detach().cpu())

            if self.cfg.print_every and (epoch % self.cfg.print_every == 0):
                print(f"[DeepSAD] epoch {epoch:03d}  loss={epoch_loss:.6f}")

    @torch.no_grad()
    def score(self, X: torch.Tensor) -> torch.Tensor:
        """
        Anomaly score s(x) = ||φ(x) - c||  (higher = more anomalous)
        """
        if self.c is None:
            raise RuntimeError("DeepSAD not fitted. Call fit() first.")
        self.model.eval()
        X = X.to(self.device)
        z = self.model(X)
        d2 = sqdist_to_center(z, self.c)
        return torch.sqrt(torch.clamp(d2, min=0.0))


# ------------------------------- Smoke test ----------------------------------
if __name__ == "__main__":
    # Minimal sanity check on synthetic blobs (not an experiment)
    torch.manual_seed(0)
    n_p, n_u, n_n, d = 200, 150, 40, 4
    X_p = torch.randn(n_p, d) * 0.8 + 0.0     # labeled normals
    X_u = torch.randn(n_u, d) * 0.9 + 0.2     # unlabeled (mostly normal-ish)
    X_n = torch.randn(n_n, d) * 0.7 + 4.5     # labeled anomalies (far)
    X = torch.vstack([X_p, X_u, X_n]).float()
    y = torch.tensor([1]*n_p + [0]*n_u + [-1]*n_n, dtype=torch.long)

    cfg = DeepSADConfig(in_dim=d, hidden=(64, 32), out_dim=16,
                        eta=1.0, eps_inv=1e-6, epochs=10, batch_size=64, lr=1e-3, device="cpu")
    model = DeepSAD(cfg)
    model.fit(X, y)
    s = model.score(X)
    print("scores shape:", tuple(s.shape), " | mean score:", float(s.mean()))
