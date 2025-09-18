# bifurcation.py
import torch
from typing import Callable, Dict
from physics import jac_x, jac_p
from model import PINN

def compute_indicators(F: Callable, x: torch.Tensor, p: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    计算残差范数、Jx 的最小奇异值、tau=||w^T F_p||。
    外层可能在 no_grad，这里强制开启梯度，并把 x,p 作为叶子张量重新构图。
    """
    with torch.enable_grad():
        xg = x.detach().requires_grad_(True)
        pg = p.detach().requires_grad_(True)

        res = torch.linalg.norm(F(xg, pg), dim=1)               # [B]
        Jx  = jac_x(F, xg, pg)                                   # [B,nx,nx]
        Jp  = jac_p(F, xg, pg)                                   # [B,nx,np]

        U, S, Vh = torch.linalg.svd(Jx, full_matrices=False)     # batched SVD
        w = U[..., :, -1]                                        # 最小奇异向量
        proj = torch.matmul(w.unsqueeze(1), Jp).squeeze(1)       # [B,np]
        tau = torch.linalg.norm(proj, dim=1)                     # [B]
        sigma_min = S[..., -1]                                    # [B]

    return {"res": res.detach(), "sigma_min": sigma_min.detach(), "tau": tau.detach()}

def compute_stability(F: Callable, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    稳定性：Jx 的特征值实部最大值 < 0 视为稳定。返回 bool 向量 [B]。
    """
    with torch.enable_grad():
        xg = x.detach().requires_grad_(True)
        pg = p.detach().requires_grad_(True)
        Jx = jac_x(F, xg, pg)
        evals = torch.linalg.eigvals(Jx)                         # [B,nx]
        stable = (evals.real.max(dim=1).values < 0)
    return stable.detach()

def eval_geom_terms(net, s_eval: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    弧长一致性误差、曲率、相邻切向量余弦。
    """
    with torch.enable_grad():
        s = s_eval.detach().clone().requires_grad_(True)
        x, p = net(s)
        y = torch.cat([x, p], dim=1)

        dyds = PINN.first_derivative(y, s)
        d2   = PINN.second_derivative(y, s)

        speed   = torch.linalg.norm(dyds, dim=1)
        arc_err = (speed - 1.0) ** 2
        curv    = torch.linalg.norm(d2, dim=1)

        t = dyds / (speed.unsqueeze(-1) + 1e-8)
        tangent_cos = torch.ones_like(speed)
        tangent_cos[1:] = (t[1:] * t[:-1]).sum(dim=1)

    return {
        "arc_err": arc_err.detach(),
        "curv": curv.detach(),
        "tangent_cos": tangent_cos.detach()
    }

class SaddleNodeDetector:
    """连续 window 次满足阈值条件才触发（去抖）"""
    def __init__(self, eps_r=1e-4, eps_sigma=1e-3, eps_tau=1e-3, window: int = 5):
        from collections import deque
        self.eps_r, self.eps_sigma, self.eps_tau = eps_r, eps_sigma, eps_tau
        self.buf = deque(maxlen=window)

    def update(self, ind: Dict[str, torch.Tensor]) -> bool:
        cond = ((ind["res"] < self.eps_r) &
                (ind["sigma_min"] < self.eps_sigma) &
                (ind["tau"] > self.eps_tau)).any().item()
        self.buf.append(bool(cond))
        return len(self.buf) == self.buf.maxlen and all(self.buf)

def export_branch_csv(F: Callable, net, s_eval: torch.Tensor, out_csv: str,
                      nx: int, np_: int, detector: SaddleNodeDetector):
    """
    导出用于画图/评估的 CSV。
    列：s, x_0, (x_1), p_0, res, sigma_min, tau, stable, bif_flag, arc_err, curv, tangent_cos
    """
    import os, csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # 前向（无需创建计算图）
    x, p = net(s_eval)

    # 指标、几何项、稳定性
    ind  = compute_indicators(F, x, p)
    geom = eval_geom_terms(net, s_eval)
    stab = compute_stability(F, x, p)

    cond = ((ind['res'] < detector.eps_r) &
            (ind['sigma_min'] < detector.eps_sigma) &
            (ind['tau'] > detector.eps_tau)).to(torch.int32)

    # 落盘前搬到 CPU
    s_c   = s_eval.detach().cpu()
    x_c   = x.detach().cpu()
    p_c   = p.detach().cpu()
    res_c = ind['res'].detach().cpu()
    sig_c = ind['sigma_min'].detach().cpu()
    tau_c = ind['tau'].detach().cpu()
    arc_c = geom['arc_err'].detach().cpu()
    curv_c= geom['curv'].detach().cpu()
    tcos_c= geom['tangent_cos'].detach().cpu()
    st_c  = stab.detach().cpu().to(torch.int32)

    # 列名统一：x_0,(x_1),p_0
    cols = ["s", "x_0"] + (["x_1"] if nx > 1 else []) + ["p_0",
            "res","sigma_min","tau","stable","bif_flag","arc_err","curv","tangent_cos"]

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        B = s_eval.shape[0]
        for i in range(B):
            row = [float(s_c[i].item()),
                   float(x_c[i,0].item())]
            if nx > 1:
                row.append(float(x_c[i,1].item()))
            row += [float(p_c[i,0].item()),
                    float(res_c[i].item()),
                    float(sig_c[i].item()),
                    float(tau_c[i].item()),
                    int(st_c[i].item()),
                    int(cond[i].item()),
                    float(arc_c[i].item()),
                    float(curv_c[i].item()),
                    float(tcos_c[i].item())]
            w.writerow(row)
