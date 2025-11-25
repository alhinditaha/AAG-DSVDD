import torch
from sr_graph_dsvdd.data import synthetic_blobs, split_indices
from sr_graph_dsvdd.train import SRGraphDeepSVDDTrainer, TrainConfig

def test_smoke():
    X, y = synthetic_blobs(n_norm=200, n_anom=40, n_unl=80, seed=0)
    idx_norm, idx_anom, idx_unl = split_indices(y)
    cfg = TrainConfig(in_dim=X.shape[1], epochs=2, batch_size=64, lambda_u=0.05, k=10)
    trainer = SRGraphDeepSVDDTrainer(cfg)
    trainer.fit(X.float(), y.long(), idx_norm)
    s = trainer.score(X.float())
    assert s.shape[0] == X.shape[0]
