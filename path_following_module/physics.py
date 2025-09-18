# physics.py
import torch
import numpy as np
from typing import Callable, Dict, Tuple

# ========= 系统定义 =========
# 我们统一用 NX=1 或 2；NP=1；Case1/2 为 1D 平衡，可嵌入 2D（第二维为 -x2 稳定化）

# Case1：Saddle-node（抛物线） F(x,p)=x^2 + p = 0
def F_case1_1d(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    # x:[B,1], p:[B,1] -> [B,1]
    return x**2 + p

def F_case1_2d_embed(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    # x:[B,2] -> [B,2]; 第一维是抛物线，第二维衰减到 0
    x1 = x[:, 0:1]; x2 = x[:, 1:2]
    f1 = x1**2 + p
    f2 = -x2
    return torch.cat([f1, f2], dim=1)

# Case2：Transcritical xdot = r x - x^2 -> 平衡 F1 = x(r - x) = r x - x^2
def F_case2_1d(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return p * x - x**2

def F_case2_2d_embed(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    x1 = x[:, 0:1]; x2 = x[:, 1:2]
    f1 = p * x1 - x1**2
    f2 = -x2
    return torch.cat([f1, f2], dim=1)

# Case3：Hopf 幅值流形（做法A）：F(r, μ)= μ r - r^3 = 0
def F_case3_amp(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    # x 为幅值 r ≥0，p 为 μ
    return p * x - x**3

# ========= 通用雅可比 =========
def jac_x(F: Callable, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    Fval = F(x, p)  # [B,NX]
    nx = Fval.shape[1]
    rows = []
    for i in range(nx):
        gi = torch.autograd.grad(Fval[:, i].sum(), x, create_graph=True, retain_graph=True)[0]
        rows.append(gi.unsqueeze(1))  # [B,1,nx]
    return torch.cat(rows, dim=1)     # [B,nx,nx]

def jac_p(F: Callable, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    Fval = F(x, p)
    cols = []
    for i in range(Fval.shape[1]):
        gi = torch.autograd.grad(Fval[:, i].sum(), p, create_graph=True, retain_graph=True)[0]
        cols.append(gi.unsqueeze(-1))  # [B,1,1] (NP=1)
    Jp = torch.cat(cols, dim=-1)      # [B,1,nx]
    return Jp.transpose(-1, -2)       # [B,nx,1]

# ========= 理论解（用于画对比与误差）=========
def theory_case1(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # x = ±sqrt(-p) for p<=0；p>0 无解（NaN）
    p = np.asarray(p)
    xabs = np.sqrt(np.clip(-p, 0.0, None))
    return -xabs, xabs  # (stable, unstable) for 1D dynamics (f_x=2x)

def theory_case2(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 分支 x=0 与 x=p；稳定性因 r-2x
    p = np.asarray(p)
    return np.zeros_like(p), p

def theory_case3_amp(p: np.ndarray) -> np.ndarray:
    # r = sqrt(max(μ,0))? 注意本范式 F=μ r - r^3=0 -> r=0 或 r=±sqrt(μ)
    # 幅值取非负支：r = sqrt(max(μ,0))
    mu = np.asarray(p)
    return np.sqrt(np.clip(mu, 0.0, None))

# ========= 注册表 =========
SYSTEMS: Dict[str, Dict] = {
    # name : {F, nx, np, theory_fn(s)}
    "case1_1d": {"F": F_case1_1d, "nx": 1, "np": 1, "theory": theory_case1},
    "case1_2d": {"F": F_case1_2d_embed, "nx": 2, "np": 1, "theory": theory_case1},
    "case2_1d": {"F": F_case2_1d, "nx": 1, "np": 1, "theory": theory_case2},
    "case2_2d": {"F": F_case2_2d_embed, "nx": 2, "np": 1, "theory": theory_case2},
    "case3_amp": {"F": F_case3_amp, "nx": 1, "np": 1, "theory": theory_case3_amp},
}

def get_system(name: str):
    meta = SYSTEMS[name]
    return meta["F"], meta["nx"], meta["np"], meta["theory"]
