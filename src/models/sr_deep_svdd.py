
import json
from src.optim.sr_deep_svdd_trainer import SRDeepSVDDTrainer
from src.networks.main import build_network

class SRDeepSVDD(object):
    def __init__(self, objective='soft-boundary', nu=0.1, severity_weights=None, margin_per_group=None):
        self.objective=objective; self.nu=nu
        self.R=0.0; self.c=None; self.net=None; self.trainer=None
        self.severity_weights=severity_weights or {1:1.0,2:1.0}
        self.margin_per_group=margin_per_group or {1:0.0,2:0.0}
        self.results={'train_time':None,'test_auc':None,'test_time':None,'test_scores':None}
    def set_network(self, net_name, **net_kwargs):
        self.net=build_network(net_name, **net_kwargs)
    def train(self, dataset, optimizer_name='adamw', lr=1e-3, n_epochs=50, lr_milestones=(), batch_size=128, weight_decay=1e-6, device='cuda'):
        self.trainer=SRDeepSVDDTrainer(R=self.R, c=self.c, nu=self.nu, severity_weights=self.severity_weights, margin_per_group=self.margin_per_group, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay, device=device)
        self.net=self.trainer.train(dataset, self.net); self.R=float(self.trainer.R.detach().cpu().numpy()); self.c=self.trainer.c.detach().cpu().numpy().tolist(); self.results['train_time']=self.trainer.train_time
    def test(self, dataset, device='cuda'):
        self.trainer.test(dataset, self.net); self.results['test_auc']=self.trainer.test_auc; self.results['test_time']=self.trainer.test_time; self.results['test_scores']=self.trainer.test_scores
