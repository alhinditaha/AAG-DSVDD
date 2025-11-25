from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .model import BiasFreeMLP, squared_distance_to_center
from .graph import build_label_aware_knn_laplacian

@dataclass
class TrainConfig:
    in_dim: int
    hidden: Tuple[int, ...] = (128, 64)
    out_dim: int = 32
    p: int = 2
    nu: float = 0.1
    Omega: float = 2.0
    margin_m: float = 1.0
    lambda_u: float = 0.1
    wd: float = 1e-4
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 256
    warmup_epochs: int = 2
    graph_refresh: int = 2
    k: int = 15
    gamma_anom_edges: float = 1.0
    device: str = 'cpu'

def _quantile_R2(d2_normals: torch.Tensor, q: float) -> float:
    return float(torch.quantile(d2_normals.detach(), q))

def _hinge(x: torch.Tensor, p: int) -> torch.Tensor:
    h = torch.clamp(x, min=0.0)
    return h*h if p == 2 else h

class SRGraphDeepSVDDTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.model = BiasFreeMLP(cfg.in_dim, cfg.hidden, cfg.out_dim).to(cfg.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        self.c: Optional[torch.Tensor] = None
        self.R2: float = 0.0
        self.L: Optional[torch.Tensor] = None

    def _warmup_and_set_center(self, X_norm: torch.Tensor):
        if self.cfg.warmup_epochs > 0:
            dl = DataLoader(TensorDataset(X_norm), batch_size=self.cfg.batch_size, shuffle=True)
            for _ in range(self.cfg.warmup_epochs):
                for (xb,) in dl:
                    xb = xb.to(self.cfg.device)
                    z = self.model(xb)
                    loss = (z**2).mean()
                    self.opt.zero_grad(); loss.backward(); self.opt.step()
        with torch.no_grad():
            Z = self.model(X_norm.to(self.cfg.device))
            self.c = Z.mean(dim=0).detach()

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
        if self.L is None or self.cfg.lambda_u <= 0.0:
            return 0.0
        self.model.train()
        z = self.model(X_all.to(self.cfg.device))
        d2 = squared_distance_to_center(z, self.c.to(self.cfg.device))
        Ld2 = torch.sparse.mm(self.L, d2.unsqueeze(1)).squeeze(1)
        L_graph = self.cfg.lambda_u * torch.sum(d2 * Ld2)
        self.opt.zero_grad(); L_graph.backward(); self.opt.step()
        return float(L_graph.detach().cpu())

    def _train_epoch(self, dl: DataLoader, y_batch: torch.Tensor) -> float:
        self.model.train()
        total = 0.0
        for (xb, idxb) in dl:
            xb = xb.to(self.cfg.device); idxb = idxb.to(self.cfg.device)
            yb = y_batch[idxb]
            z = self.model(xb)
            d2 = squared_distance_to_center(z, self.c.to(self.cfg.device))

            mask_n = (yb == 1); mask_a = (yb == -1)
            L_norm = _hinge(d2[mask_n] - self.R2, self.cfg.p).mean() / self.cfg.nu if mask_n.any() else 0.0
            L_anom = self.cfg.Omega * _hinge(self.cfg.margin_m + self.R2 - d2[mask_a], self.cfg.p).mean() if mask_a.any() else 0.0

            loss = self.R2 + (L_norm if isinstance(L_norm, torch.Tensor) else torch.tensor(L_norm, device=self.cfg.device)) \
                           + (L_anom if isinstance(L_anom, torch.Tensor) else torch.tensor(L_anom, device=self.cfg.device))
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            total += float(loss.detach().cpu())
        return total

    def fit(self, X_all: torch.Tensor, y_all: torch.Tensor, idx_normals: torch.Tensor):
        self._warmup_and_set_center(X_all[idx_normals].to(self.cfg.device))
        with torch.no_grad():
            z_norm = self.model(X_all[idx_normals].to(self.cfg.device))
            d2_norm = squared_distance_to_center(z_norm, self.c.to(self.cfg.device))
            self.R2 = _quantile_R2(d2_norm, q=1.0 - self.cfg.nu)

        n = X_all.shape[0]
        dl = DataLoader(TensorDataset(X_all, torch.arange(n)), batch_size=self.cfg.batch_size, shuffle=True)

        for epoch in range(1, self.cfg.epochs+1):
            if (epoch-1) % max(1, self.cfg.graph_refresh) == 0:
                self._build_graph(X_all, y_all)

            hinge_loss = self._train_epoch(dl, y_all)

            with torch.no_grad():
                z_norm = self.model(X_all[idx_normals].to(self.cfg.device))
                d2_norm = squared_distance_to_center(z_norm, self.c.to(self.cfg.device))
                self.R2 = _quantile_R2(d2_norm, q=1.0 - self.cfg.nu)

            graph_loss = self._graph_step(X_all)
            print(f"Epoch {epoch:03d} | hinge={hinge_loss:.4f} | graph={graph_loss:.4f} | R2={self.R2:.4f}")

    @torch.no_grad()
    def score(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        z = self.model(X.to(self.cfg.device))
        d2 = squared_distance_to_center(z, self.c.to(self.cfg.device))
        return d2 - self.R2
