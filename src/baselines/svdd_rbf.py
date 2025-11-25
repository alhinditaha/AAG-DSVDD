import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import roc_auc_score

class KernelSVDD:
    def __init__(self, nu=0.05, gamma=0.2):
        self.nu=float(nu); self.gamma=float(gamma)
        self.Xn=None
        self.alpha=None
        self.results = {}

    def fit(self, Xn):
        self.Xn=np.asarray(Xn, dtype=np.float64)
        N=self.Xn.shape[0]
        # Simple SVDD dual with uniform alphas under box constraints; here
        # we use a ν-approximation: center is implicit in feature space.
        # For demonstration, take alpha = 1/N (center-of-mass in feature space).
        self.alpha=np.full(N, 1.0/N, dtype=np.float64)
        # Radius^2 approximated as max distance among training normals to center
        K = rbf_kernel(self.Xn, self.Xn, gamma=self.gamma)
        # center dot-products and distances
        aKa = self.alpha @ K @ self.alpha
        d2 = np.maximum(0.0, np.diag(K) - 2*(K @ self.alpha) + aKa)
        self.R2 = float(np.quantile(d2, 1.0 - self.nu))
        return self

    def fit_on_dataset(self, dataset):
        # Expect BaseADDataset-like or (X,y)
        if isinstance(dataset, tuple) and len(dataset)==2:
            X, y = dataset
        elif hasattr(dataset, 'train'):
            X, y = dataset.train.X, dataset.train.y
        else:
            X, y = dataset.train_set
        return self.fit(X[y==0])

    def decision_function(self, X):
        X=np.asarray(X, dtype=np.float64)
        Kxz = rbf_kernel(X, self.Xn, gamma=self.gamma)
        Kzz = rbf_kernel(self.Xn, self.Xn, gamma=self.gamma)
        aKa = self.alpha @ Kzz @ self.alpha
        d2 = np.maximum(0.0, np.ones(X.shape[0]) - 2*(Kxz @ self.alpha) + aKa)  # diag(Kxx)=1 for RBF on itself
        # Score = d2 - R2, consistent with SVDD-like sign
        return d2 - self.R2

    def test_on_dataset(self, dataset):
        if hasattr(dataset, 'test'):
            X, y = dataset.test.X, dataset.test.y
        else:
            X, y = dataset
        scores = self.decision_function(X)
        self.results['test_scores'] = scores
        try:
            self.results['test_auc'] = float(roc_auc_score(y, scores))
        except Exception:
            self.results['test_auc'] = float('nan')
        return scores
