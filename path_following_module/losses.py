# losses.py
import torch
from typing import Dict, Optional
from torch.autograd import grad
from config import Config
from model import PINN


# ---------- 基础项 ----------
def physics_residual(F, x, p) -> torch.Tensor:
    r = F(x, p)                      # [B, nx]
    return (r**2).sum(dim=1).mean()

def arc_length_from_dyds(dyds: torch.Tensor, target_speed: float = 1.0) -> torch.Tensor:
    spd = torch.linalg.norm(dyds, dim=1)          # [B]
    return ((spd - target_speed) ** 2).mean()

def smoothness(d2yds2: torch.Tensor) -> torch.Tensor:
    return (d2yds2**2).sum(dim=1).mean()

def initial_conditions(y_s0: torch.Tensor, y0: torch.Tensor,
                       t_s0: Optional[torch.Tensor] = None,
                       t0: Optional[torch.Tensor] = None) -> torch.Tensor:
    Lp = ((y_s0 - y0) ** 2).sum(dim=1).mean()
    Lt = 0.0
    if (t_s0 is not None) and (t0 is not None):
        Lt = ((t_s0 - t0) ** 2).sum(dim=1).mean()
    return Lp + Lt

# ---------- 方向约束 ----------
def _sorted_idx(s: torch.Tensor):
    return s.squeeze(-1).argsort()

def dir_cosine(dyds: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    idx = _sorted_idx(s); t = dyds.index_select(0, idx)
    t1, t0 = t[1:], t[:-1]
    t1 = t1 / (t1.norm(dim=1, keepdim=True) + 1e-8)
    t0 = t0 / (t0.norm(dim=1, keepdim=True) + 1e-8)
    return (1 - (t1*t0).sum(dim=1)).mean()

def dir_forward(y: torch.Tensor, dyds: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    idx = _sorted_idx(s)
    ys = y.index_select(0, idx); ts = dyds.index_select(0, idx)
    dy = ys[1:] - ys[:-1]; t = ts[:-1]
    dy = dy / (dy.norm(dim=1, keepdim=True) + 1e-8)
    t  = t  / (t.norm(dim=1, keepdim=True)  + 1e-8)
    return (1 - (dy*t).sum(dim=1)).mean()

def dir_global(dyds: torch.Tensor, nx: int, p_idx: int = 0, margin: float = 0.0) -> torch.Tensor:
    # 对参数维 dp/ds 做“前向”约束（≥ margin）；nx 为 x 的维度，用来定位 p 的通道
    dp = dyds[:, nx + p_idx]
    if margin > 0:
        return torch.relu(margin - dp).mean()
    else:
        return torch.relu(-dp).mean()

# ---------- Kendall 组合 ----------
def combine_kendall(losses: Dict[str, torch.Tensor], alphas: Dict[str, float], log_sigma) -> torch.Tensor:
    total = 0.0
    for k, L in losses.items():
        if L is None: 
            continue
        s_k = log_sigma[k]              # nn.Parameter
        a_k = alphas.get(k, 1.0)        # α 缩放
        total = total + a_k * (0.5 * torch.exp(-s_k) * L + 0.5 * s_k)
    return total

# ---------- 统一组装：compute_loss ----------
def compute_loss(model, s: torch.Tensor, y0: torch.Tensor, physics_fn, **kw):
    """
    组装总损失：
      L = Kendall( phys, arc, ic, smooth, dir )
    其中 dir = w_cos * L_cos + w_fwd * L_fwd + w_g * L_global
    - 读 Config.DIR_WEIGHTS 来组合方向项
    - 读 Config.ALPHA / Config.USE_KENDALL 控制加权方式
    """
    device = Config.DEVICE
    s = s.to(device).requires_grad_(True)
    x, p = model(s)                                # x:[B,nx], p:[B,np]
    y = torch.cat([x, p], dim=1)                   # y:[B, nx+np]

    # 基础项
    L_phys = physics_residual(physics_fn, x, p)

    dyds   = PINN.first_derivative(y, s) 
    L_arc = arc_length_from_dyds(dyds, target_speed=1.0)

    # s=0 的输出用于 IC
    y_s0 = model(torch.zeros(1, 1, device=device))
    y_s0_cat = torch.cat([y_s0[0], y_s0[1]], dim=1) if isinstance(y_s0, tuple) else y_s0
    L_ic = initial_conditions(y_s0_cat, y0)

    d2yds2 = PINN.second_derivative(y, s)
    L_smooth = smoothness(d2yds2)

    # 方向项 = 三项组合
    w = getattr(Config, "DIR_WEIGHTS", {"cos":0.0, "forward":0.0, "global":0.0})
    L_dir_cos = dir_cosine(dyds, s) if w.get("cos", 0.0) > 0 else None
    L_dir_fwd = dir_forward(y, dyds, s) if w.get("forward", 0.0) > 0 else None
    L_dir_glb = dir_global(dyds, nx=x.shape[1],
                           p_idx=getattr(Config, "DIR_GLOBAL_PARAM_IDX", 0),
                           margin=getattr(Config, "DIR_GLOBAL_MARGIN", 0.0)) if w.get("global", 0.0) > 0 else None

    # 将三个方向子项线性组合成一个 dir 原子项，再交给 Kendall/α
    L_dir_raw = None
    if any(v > 0 for v in w.values()):
        parts = []
        if L_dir_cos is not None: parts.append(w["cos"] * L_dir_cos)
        if L_dir_fwd is not None: parts.append(w["forward"] * L_dir_fwd)
        if L_dir_glb is not None: parts.append(w["global"] * L_dir_glb)
        L_dir_raw = sum(parts)

    losses = {
        "phys": L_phys,
        "arc": L_arc,
        "ic": L_ic,
        "smooth": L_smooth,
        "dir": L_dir_raw
    }


    # 总损失：Kendall 或 直接加权和
    if getattr(Config, "USE_KENDALL", False):
        # 从 model.log_sigma[...] 读取每项的 log_sigma
        log_sigma = {k: model.log_sigma[k] for k in losses.keys()}
        total = combine_kendall(losses, getattr(Config, "ALPHA", {}), log_sigma)
    else:
        alphas = getattr(Config, "ALPHA", {})
        total = sum(alphas.get(k, 1.0) * (L if L is not None else 0.0) for k, L in losses.items())


    # losses.py（增加一个小工具）
    def _as_tensor(x, device):
        """把 float/None 统一成 tensor（标量）"""
        if x is None:
            return torch.zeros((), device=device)
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x, dtype=torch.float32, device=device)
        # —— 组合总损失（示意，保持你原来的公式）——
    # 先把分量放进 dict
    losses = {
        "phys": L_phys,
        "arc":  L_arc,
        "ic":   L_ic,
        "smooth": L_smooth if 'L_smooth' in locals() else 0.0,
        "dir":  L_dir if 'L_dir' in locals() else 0.0,
    }

    # 统一成 tensor，避免 float 与 tensor 混算/日志时报错
    device = y.device if 'y' in locals() else s.device
    for k, v in list(losses.items()):
        losses[k] = _as_tensor(v, device)

    # 你的总损失（Kendall/α 等）保持不变，这里示例：
    total = combine_kendall(losses, alphas=Config.ALPHA, log_sigma=model.log_sigma)
    # 若你还有其它加项，也同样用 _as_tensor 包一下再相加

    # —— 返回日志用的标量（避免 float.detach 报错）——
    comps = {}
    for k, L in losses.items():
        # 此处 L 一定是 tensor 了
        comps[k] = float(L.detach().cpu().item())
    comps["total"] = float(total.detach().cpu().item())

    return total, comps