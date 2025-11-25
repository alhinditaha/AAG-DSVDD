import argparse, torch
from sr_graph_dsvdd.data import synthetic_blobs, split_indices
from sr_graph_dsvdd.train import SRGraphDeepSVDDTrainer, TrainConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--graph-refresh', type=int, default=2)
    ap.add_argument('--k', type=int, default=15)
    ap.add_argument('--p', type=int, default=2)
    ap.add_argument('--nu', type=float, default=0.1)
    ap.add_argument('--Omega', type=float, default=2.0)
    ap.add_argument('--m', type=float, default=1.0)
    ap.add_argument('--lambda-u', type=float, default=0.1, dest='lambda_u')
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--warmup-epochs', type=int, default=2)
    ap.add_argument('--gamma-anom-edges', type=float, default=1.0)
    ap.add_argument('--device', type=str, default='cpu')
    args = ap.parse_args()

    X, y = synthetic_blobs()
    idx_norm, idx_anom, idx_unl = split_indices(y)

    cfg = TrainConfig(
        in_dim=X.shape[1], hidden=(128,64), out_dim=16,
        p=args.p, nu=args.nu, Omega=args.Omega, margin_m=args.m,
        lambda_u=args.lambda_u, wd=args.wd, lr=args.lr,
        epochs=args.epochs, batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs, graph_refresh=args.graph_refresh,
        k=args.k, gamma_anom_edges=args.gamma_anom_edges, device=args.device
    )

    trainer = SRGraphDeepSVDDTrainer(cfg)
    trainer.fit(X.float(), y.long(), idx_norm)
    scores = trainer.score(X.float()).cpu().numpy()
    print('Scores (first 10):', scores[:10])

if __name__ == '__main__':
    main()
