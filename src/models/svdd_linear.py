
import json
from src.optim.svdd_linear_trainer import LinearSVDDTrainer

class LinearSVDD(object):
    def __init__(self, objective='soft-boundary', nu=0.1, hinge_power=2):
        self.objective=objective; self.nu=nu; self.hinge_power=hinge_power
        self.trainer=None; self.results={'train_time':None,'test_auc':None}
    def train(self, dataset, lr=1e-2, n_epochs=300, batch_size=256, device='cpu'):
        self.trainer=LinearSVDDTrainer(self.objective, R=0.0, a=None, nu=self.nu, lr=lr, n_epochs=n_epochs, batch_size=batch_size, device=device, hinge_power=self.hinge_power)
        self.trainer.train(dataset); self.results['train_time']=self.trainer.train_time
    def test(self, dataset):
        self.trainer.test(dataset); self.results['test_auc']=self.trainer.test_auc
