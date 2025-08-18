from __future__ import annotations
import torch
from typing import Dict, List
from physics import physics_equation


def _ensure_tensor(v: torch.Tensor) -> torch.Tensor:
    return v if isinstance(v, torch.Tensor) else torch.tensor(v)


def evaluate_path_with_derivs(model, s_grid: torch.Tensor, device: str) -> Dict[str, torch.Tensor]:
    s = s_grid.to(device).view(-1, 1).detach().clone().requires_grad_(True)
    x, p = model(s)
    dxds = torch.autograd.grad(x, s, torch.ones_like(x), create_graph=True)[0]
    dpds = torch.autograd.grad(p, s, torch.ones_like(p), create_graph=True)[0]
    F = physics_equation(x, p)
    Fx = torch.autograd.grad(F, x, torch.ones_like(F), create_graph=True, retain_graph=True)[0]
    Fp = torch.autograd.grad(F, p, torch.ones_like(F), create_graph=True, retain_graph=False)[0]
    return {"s": s, "x": x, "p": p, "dxds": dxds, "dpds": dpds, "F": F, "Fx": Fx, "Fp": Fp}


def detect_folds(data: Dict[str, torch.Tensor], tol_F: float = 1e-3, tol_Fx: float = 1e-3, tol_dx: float = 1e-3) -> List[int]:
    s = data["s"].squeeze()
    dxds = data["dxds"].squeeze()
    dpds = data["dpds"].squeeze()
    F = data["F"].squeeze()
    Fx = data["Fx"].squeeze()

    # turning wrt parameter p: dp/ds changes sign (or small) and F_x ~ 0 (fold for F(x,p)=0) and F ~ 0
    sign = torch.sign(dpds)
    zc = (sign[1:] * sign[:-1] <= 0) | (dpds[1:].abs() < tol_dx) | (dpds[:-1].abs() < tol_dx)
    cand = torch.nonzero(zc).squeeze()
    if cand.ndim == 0:
        cand = cand.unsqueeze(0)

    idxs: List[int] = []
    for i in cand.tolist():
        i = int(i)
        if i < 0 or i >= s.numel():
            continue
        if (F[i].abs() < tol_F) and (Fx[i].abs() < tol_Fx):
            idxs.append(i)
        # fallback: vertical tangent in x(s): |dx/ds| ≈ 0
        elif (F[i].abs() < tol_F) and (dxds[i].abs() < tol_dx):
            idxs.append(i)
    # unique & sorted
    return sorted(set(idxs))


def analyze_bifurcations(model, s_grid: torch.Tensor, device: str,
                          tol_F: float = 1e-3, tol_Fx: float = 1e-3, tol_dx: float = 1e-3) -> Dict:
    data = evaluate_path_with_derivs(model, s_grid, device)
    fold_idx = detect_folds(data, tol_F=tol_F, tol_Fx=tol_Fx, tol_dx=tol_dx)

    x = data["x"].detach().cpu().squeeze()
    p = data["p"].detach().cpu().squeeze()
    Fx = data["Fx"].detach().cpu().squeeze()
    F = data["F"].detach().cpu().squeeze()

    # stability for saddle-node: f_x = 2x; stable if f_x < 0 ⇔ x < 0
    stable_mask = (x < 0)

    points = []
    for i in fold_idx:
        stab = "stable" if stable_mask[i].item() else "unstable"
        points.append({
            "i": int(i),
            "s": float(data["s"][i].item()),
            "x": float(x[i].item()),
            "p": float(p[i].item()),
            "Fx": float(Fx[i].item()),
            "F": float(F[i].item()),
            "stability": stab,
        })
    return {"fold_indices": fold_idx, "points": points, "stable_mask": stable_mask.tolist()}