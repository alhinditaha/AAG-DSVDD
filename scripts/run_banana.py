import argparse, numpy as np, torch
from sr_graph_dsvdd.datasets.banana import BananaParams, make_banana_data
from sr_graph_dsvdd.train import SRGraphDeepSVDDTrainer, TrainConfig

def to_triplet_labels(y_raw, unlabeled_frac=0.3, seed=0):
    """
    Map y_raw: 0=normal, 1=anomaly -> triplet labels {+1 normal, -1 anomaly, 0 unlabeled}.
    A random fraction of points becomes unlabeled (sampled from both normals and anomalies).
    """
    rng = np.random.default_rng(seed)
    n = y_raw.shape[0]
    y_trip = np.where(y_raw==0, 1, -1).astype(int)  # normals -> +1, anomalies -> -1
    if unlabeled_frac > 0.0:
        m = int(unlabeled_frac * n)
        idx_unl = rng.choice(n, size=m, replace=False)
        y_trip[idx_unl] = 0
    return y_trip

def main():
    ap = argparse.ArgumentParser()
    # Banana params
    ap.add_argument("--n-norm", type=int, default=700)
    ap.add_argument("--n-anom1", type=int, default=150)
    ap.add_argument("--n-anom2", type=int, default=150)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--b", type=float, default=0.2)
    ap.add_argument("--s1", type=float, default=2.0)
    ap.add_argument("--s2", type=float, default=1.5)
    ap.add_argument("--rotate-deg", type=float, default=90.0)
    ap.add_argument("--mu-a1-x", type=float, default=0.0)
    ap.add_argument("--mu-a1-y", type=float, default=4.9)
    ap.add_argument("--mu-a2-x", type=float, default=0.0)
    ap.add_argument("--mu-a2-y", type=float, default=-4.0)
    ap.add_argument("--cov-a1-x", type=float, default=0.04)
    ap.add_argument("--cov-a1-y", type=float, default=0.81)
    ap.add_argument("--cov-a2-x", type=float, default=0.04)
    ap.add_argument("--cov-a2-y", type=float, default=0.81)

    # Labeling
    ap.add_argument("--unlabeled-frac", type=float, default=0.30, help="fraction of all points to treat as unlabeled")

    # Training params (subset of existing flags)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--graph-refresh", type=int, default=2)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--nu", type=float, default=0.1)
    ap.add_argument("--Omega", type=float, default=2.0)
    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--lambda-u", type=float, default=0.1, dest="lambda_u")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--gamma-anom-edges", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    params = BananaParams(
        n_norm=args.n_norm, n_anom1=args.n_anom1, n_anom2=args.n_anom2, seed=args.seed,
        b=args.b, s1=args.s1, s2=args.s2, rotate_deg=args.rotate_deg,
        mu_a1=(args.mu_a1_x, args.mu_a1_y), mu_a2=(args.mu_a2_x, args.mu_a2_y),
        cov_a1=(args.cov_a1_x, args.cov_a1_y), cov_a2=(args.cov_a2_x, args.cov_a2_y)
    )
    X, y_raw, g = make_banana_data(params)
    y_trip = to_triplet_labels(y_raw, unlabeled_frac=args.unlabeled_frac, seed=args.seed)

    # Indices
    y_torch = torch.from_numpy(y_trip).long()
    X_torch = torch.from_numpy(X).float()
    idx_norm = (y_torch == 1).nonzero(as_tuple=True)[0]

    cfg = TrainConfig(
        in_dim=X_torch.shape[1],
        hidden=(128,64), out_dim=16,
        p=args.p, nu=args.nu, Omega=args.Omega, margin_m=args.m,
        lambda_u=args.lambda_u, wd=args.wd, lr=args.lr,
        epochs=args.epochs, batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs, graph_refresh=args.graph_refresh,
        k=args.k, gamma_anom_edges=args.gamma_anom_edges, device=args.device
    )
    trainer = SRGraphDeepSVDDTrainer(cfg)
    trainer.fit(X_torch, y_torch, idx_norm)
    scores = trainer.score(X_torch).cpu().numpy()

    # Print summary
    print('Banana dataset run complete. Shapes:', X.shape, y_trip.shape)
    print('Scores (first 10):', scores[:10])

if __name__ == "__main__":
    main()
