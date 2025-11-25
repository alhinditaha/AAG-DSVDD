
import json
from src.optim.deepSVDD_trainer import DeepSVDDTrainer
from src.networks.main import build_network

class DeepSVDD(object):
    def __init__(self, objective='soft-boundary', nu=0.1, hinge_power=2):
        self.objective=objective; self.nu=nu; self.hinge_power=hinge_power
        self.R=0.0; self.c=None; self.net=None; self.trainer=None
        self.results={'train_time':None,'test_auc':None,'test_time':None,'test_scores':None}
    def set_network(self, net_name, **net_kwargs):
        self.net=build_network(net_name, **net_kwargs)
    def train(self, dataset, optimizer_name='adamw', lr=1e-3, n_epochs=50, lr_milestones=(), batch_size=128, weight_decay=1e-6, device='cuda'):
        self.trainer=DeepSVDDTrainer('soft-boundary', self.R, self.c, self.nu, optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device)
        self.trainer.hinge_power=self.hinge_power
        self.net=self.trainer.train(dataset, self.net); self.R=float(self.trainer.R.detach().cpu().numpy()); self.c=self.trainer.c.detach().cpu().numpy().tolist(); self.results['train_time']=self.trainer.train_time
    def test(self, dataset, device='cuda'):
        self.trainer.test(dataset, self.net); self.results['test_auc']=self.trainer.test_auc; self.results['test_time']=self.trainer.test_time; self.results['test_scores']=self.trainer.test_scores
    def save_results(self, export_json):
        with open(export_json,'w') as fp: json.dump(self.results, fp)
