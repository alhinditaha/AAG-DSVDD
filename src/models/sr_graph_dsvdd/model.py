import torch
import torch.nn as nn

class BiasFreeMLP(nn.Module):
    """Bias-free MLP. Avoid batch-norm to prevent center collapse."""
    def __init__(self, in_dim: int, hidden_dims=(128, 64), out_dim: int = 32, act=nn.ReLU):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h, bias=False))
            layers.append(act())
            prev = h
        layers.append(nn.Linear(prev, out_dim, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def squared_distance_to_center(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Return d^2(x) = ||z - c||^2 for each row z in Z."""
    return ((z - c) ** 2).sum(dim=1)
