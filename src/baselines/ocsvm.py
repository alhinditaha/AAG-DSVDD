import numpy as np
from sklearn.svm import OneClassSVM

class OCSVMWrapper:
    def __init__(self, nu=0.05, kernel='rbf', gamma='scale'):
        self.nu=nu; self.kernel=kernel; self.gamma=gamma
        self.model=None
        self.results = {}

    def _xy_from_dataset(self, dataset):
        # Accept BaseADDataset, DictDataset, or raw (X,y) tuple
        if isinstance(dataset, tuple) and len(dataset)==2:
            X, y = dataset
            return np.asarray(X), np.asarray(y, int)
        # BaseADDataset-like
        if hasattr(dataset, 'train') and dataset.train is not None:
            split = dataset.train
        elif hasattr(dataset, 'train_set'):
            split = dataset.train_set
        else:
            split = dataset

        # DictDataset-like
        if isinstance(split, tuple) and len(split)==2:
            X, y = split
        elif hasattr(split, 'X') and hasattr(split, 'y'):
            X, y = split.X, split.y
        elif callable(split):  # old style: a function returning (X,y)
            X, y = split()
        else:
            raise ValueError("Unsupported dataset type for OCSVMWrapper.")
        return np.asarray(X), np.asarray(y, int)

    def train(self, dataset):
        X, y = self._xy_from_dataset(dataset)
        # Fit on normals only (y==0)
        m = (y==0)
        if isinstance(self.gamma, str):
            gamma = self.gamma  # 'scale' or 'auto'
        else:
            gamma = float(self.gamma)
        self.model = OneClassSVM(nu=float(self.nu), kernel=self.kernel, gamma=gamma).fit(X[m])
        return self

    def test(self, dataset):
        X, y = self._xy_from_dataset(dataset)
        scores = -self.model.decision_function(X).ravel()
        self.results['test_scores'] = scores
        return scores
