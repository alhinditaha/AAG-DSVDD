
import time, torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score

class LinearSVDDTrainer(object):
    def __init__(self, objective='soft-boundary', R=0.0, a=None, nu=0.1, lr=1e-2, n_epochs=200, batch_size=256, device='cpu', hinge_power=2):
        self.objective=objective; self.R=torch.tensor(R); self.a=None if a is None else torch.tensor(a)
        self.nu=nu; self.lr=lr; self.n_epochs=n_epochs; self.batch_size=batch_size; self.device=torch.device(device if torch.cuda.is_available() else 'cpu'); self.hinge_power=hinge_power
        self.train_time=None; self.test_auc=None
    def train(self, dataset):
        X=dataset.train_set.X.to(self.device); y=dataset.train_set.y.to(self.device).long()
        if self.a is None: self.a = X[y==0].mean(dim=0).detach().clone().requires_grad_(True)
        else: self.a=self.a.to(self.device).requires_grad_(True)
        self.R=self.R.to(self.device); self.R.requires_grad=False
        opt=optim.Adam([self.a], lr=self.lr); st=time.time()
        for _ in range(self.n_epochs):
            idx=(y==0).nonzero(as_tuple=False).squeeze(1)
            if idx.numel()==0: break
            sel=idx[torch.randperm(idx.numel(), device=self.device)[:min(self.batch_size, idx.numel())]]
            xb=X[sel]; d2=torch.sum((xb-self.a)**2, dim=1)
            if self.objective=='one-class': loss=d2.mean()
            else:
                viol=torch.clamp(d2 - self.R**2, min=0.0); loss = (self.R**2) + (1.0/max(self.nu,1e-8))*((viol**self.hinge_power).mean())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad():
                d2_all=torch.sum((X[y==0]-self.a)**2, dim=1)
                self.R=torch.quantile(d2_all, 1.0 - self.nu)
        self.train_time=time.time()-st; return self.a.detach()
    @torch.no_grad()
    def test(self, dataset):
        X=dataset.test_set.X.to(self.device); y=dataset.test_set.y.to(self.device).long()
        d2=torch.sum((X-self.a)**2, dim=1); s = d2 if self.objective=='one-class' else (d2 - (self.R**2))
        try: self.test_auc=float(roc_auc_score(y.cpu().numpy(), s.cpu().numpy()))
        except Exception: self.test_auc=None
        return s.cpu().numpy()
