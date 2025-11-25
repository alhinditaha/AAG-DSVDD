
import time, torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

class SRDeepSVDDTrainer(object):
    def __init__(self, R=0.0, c=None, nu=0.1, severity_weights=None, margin_per_group=None,
                 optimizer_name='adamw', lr=1e-3, n_epochs=50, lr_milestones=(),
                 batch_size=128, weight_decay=1e-6, device='cuda', n_jobs_dataloader=0):
        self.device=torch.device(device if torch.cuda.is_available() else 'cpu')
        self.R=torch.tensor(R, device=self.device, dtype=torch.float32, requires_grad=False)
        self.c=torch.tensor(c, device=self.device, dtype=torch.float32) if c is not None else None
        self.nu=nu; self.optimizer_name=optimizer_name; self.lr=lr; self.n_epochs=n_epochs
        self.lr_milestones=set(lr_milestones or []); self.batch_size=batch_size; self.weight_decay=weight_decay
        self.n_jobs_dataloader=n_jobs_dataloader
        self.train_time=None; self.test_time=None; self.test_auc=None; self.test_scores=None
        self.sev = severity_weights or {1:1.0, 2:1.0}
        self.margin = margin_per_group or {1:0.0, 2:0.0}

    @torch.no_grad()
    def init_center_c(self, net, loader, eps: float = 0.1):
        net.eval(); n_samples, c_sum=0, None
        for batch in loader:
            x,y=batch['x'].to(self.device), batch['y'].to(self.device)
            z=net(x); z_n=z[y==0]
            if z_n.numel()==0: continue
            c_sum = z_n.sum(dim=0) if c_sum is None else c_sum + z_n.sum(dim=0)
            n_samples += z_n.size(0)
        c = c_sum / max(n_samples,1)
        c[c.abs()<eps]=eps*torch.sign(c[c.abs()<eps]+1e-12)
        return c.detach()

    def get_optimizer(self, params):
        n=self.optimizer_name.lower()
        if n=='adamw': return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if n=='adam': return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if n=='sgd': return optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        raise ValueError('Unknown optimizer')

    def train(self, dataset, net):
        net=net.to(self.device)
        loader=DataLoader(dataset.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs_dataloader, drop_last=False)
        if self.c is None: self.c=self.init_center_c(net, loader)
        self.R=torch.tensor(float(self.R), device=self.device, requires_grad=False)
        opt=self.get_optimizer(net.parameters())
        sch=optim.lr_scheduler.MultiStepLR(opt, milestones=sorted(list(self.lr_milestones)), gamma=0.1)
        st=time.time()
        for _ in range(1, self.n_epochs+1):
            net.train()
            for batch in loader:
                x=batch['x'].to(self.device); y=batch['y'].to(self.device).long(); g=batch['g'].to(self.device).long()
                z=net(x); d2=torch.sum((z-self.c)**2, dim=1)
                viol=torch.clamp(d2[y==0]-self.R**2, min=0.0)
                loss_norm = (viol**2).mean() if viol.numel()>0 else torch.tensor(0.0, device=self.device)
                loss = (self.R**2) + (1.0/max(self.nu,1e-8))*loss_norm
                for grp, w in self.sev.items():
                    mask = (y==1) & (g==grp)
                    if mask.any():
                        m = float(self.margin.get(int(grp), 0.0))
                        viol_a = torch.clamp(m + self.R**2 - d2[mask], min=0.0)
                        loss = loss + float(w) * (viol_a**2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            sch.step()
            self.R=self.update_radius(net, dataset, q=1.0-self.nu)
        self.train_time=time.time()-st; return net

    @torch.no_grad()
    def update_radius(self, net, dataset, q=0.95):
        loader=DataLoader(dataset.train_set, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs_dataloader)
        net.eval(); d2_list=[]
        for batch in loader:
            x,y=batch['x'].to(self.device), batch['y'].to(self.device).long()
            if (y==0).any():
                z=net(x[y==0]); d2=torch.sum((z-self.c)**2, dim=1); d2_list.append(d2)
        if len(d2_list)==0: return self.R
        d2_all=torch.cat(d2_list, dim=0); return torch.quantile(d2_all, q).detach()

    @torch.no_grad()
    def test(self, dataset, net):
        net=net.to(self.device)
        loader=DataLoader(dataset.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs_dataloader)
        st=time.time(); scores=[]; ys=[]
        for batch in loader:
            x=batch['x'].to(self.device); y=batch['y'].to(self.device).long()
            z=net(x); d2=torch.sum((z-self.c)**2, dim=1); s=(d2 - (self.R**2))
            scores.append(s.detach().cpu()); ys.append(y.detach().cpu())
        import torch as _t
        scores=_t.cat(scores, dim=0).numpy(); ys=_t.cat(ys, dim=0).numpy()
        self.test_scores=scores.tolist()
        try: self.test_auc=float(roc_auc_score(ys, scores))
        except Exception: self.test_auc=None
        self.test_time=time.time()-st
