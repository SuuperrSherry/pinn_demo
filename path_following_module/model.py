# model.py
import torch
import torch.nn as nn
from typing import Tuple

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, layers: int, act: str):
        super().__init__()
        acts = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}
        activation = acts.get(act, nn.Tanh())
        net = []
        last = in_dim
        for _ in range(layers):
            net += [nn.Linear(last, hidden), activation]
            last = hidden
        net += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*net)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class PINN(nn.Module):
    def __init__(self, nx: int, np_: int, hidden: int, layers: int, act: str):
        super().__init__()
        self.nx, self.np = nx, np_
        self.mlp = MLP(1, nx + np_, hidden=hidden, layers=layers, act=act)
        self.log_sigma = nn.ParameterDict({k: nn.Parameter(torch.zeros(1)) for k in ["phys","arc","ic","dir","smooth"]})

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.mlp(s)
        x = y[:, :self.nx]
        p = y[:, self.nx:self.nx+self.np]
        return x, p

    @staticmethod
    def first_derivative(y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        D = y.shape[1]
        grads = [torch.autograd.grad(y[:,k].sum(), s, create_graph=True, retain_graph=True)[0] for k in range(D)]
        return torch.cat(grads, dim=1)

    @staticmethod
    def second_derivative(y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        D = y.shape[1]
        d2 = []
        for k in range(D):
            dy = torch.autograd.grad(y[:,k].sum(), s, create_graph=True, retain_graph=True)[0]
            d2y = torch.autograd.grad(dy.sum(), s, create_graph=True, retain_graph=True)[0]
            d2.append(d2y)
        return torch.cat(d2, dim=1)
