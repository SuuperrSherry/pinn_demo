import torch
from torch.autograd import grad
from config import Config
from strategies import STRATEGY_MAP


def physics_loss(x: torch.Tensor, p: torch.Tensor, physics_fn) -> torch.Tensor:
    return torch.mean((physics_fn(x, p)) ** 2)


def arc_length_loss(dx: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
    return torch.mean((dx ** 2 + dp ** 2 - 1) ** 2)


def ic_loss(x0: torch.Tensor, p0: torch.Tensor, x_ic: torch.Tensor, p_ic: torch.Tensor) -> torch.Tensor:
    return torch.mean((x0 - x_ic) ** 2 + (p0 - p_ic) ** 2)


def p_nonpos_loss(p: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.relu(p) ** 2)


def p_monotonic_loss(dp: torch.Tensor, margin: float) -> torch.Tensor:
    return torch.mean(torch.relu(margin - dp) ** 2)


def _prev_no_wrap(v: torch.Tensor) -> torch.Tensor:
    # pad the first element with itself to avoid circular wrap-around
    return torch.cat([v[:1].detach(), v[:-1].detach()], dim=0)


def direction_penalty(x: torch.Tensor, p: torch.Tensor, s: torch.Tensor, dx: torch.Tensor, dp: torch.Tensor,
                      strategy: str) -> torch.Tensor:
    if strategy is None:
        return torch.tensor(0.0, device=s.device)
    fn = STRATEGY_MAP.get(strategy)
    if fn is None:
        return torch.tensor(0.0, device=s.device)

    if strategy in ("l2", "cosine"):
        dx_last, dp_last = _prev_no_wrap(dx), _prev_no_wrap(dp)
        # skip the first item to avoid degenerate pair
        return fn(dx[1:], dp[1:], dx_last[1:], dp_last[1:])
    elif strategy == "smooth":
        return fn(dx, dp, s)
    elif strategy == "global_forward":
        g = Config.INIT_DIRECTION.to(s.device).view(1, 2)
        return fn(dx, dp, global_direction=g)
    elif strategy == "adaptive":
        return fn(dx, dp)
    elif strategy == "forward_consistency":
        return fn(x, p, dx, dp)
    else:
        return torch.tensor(0.0, device=s.device)


def _penalty_weight(strategy: str) -> float:
    # used only when AUTO_DIRECTION_WEIGHT=False
    if strategy in ("l2", "cosine"):
        return Config.DIRECTION_PENALTY_WEIGHT
    if strategy == "smooth":
        return Config.SMOOTH_PENALTY_WEIGHT
    if strategy == "global_forward":
        return Config.GLOBAL_PENALTY_WEIGHT
    if strategy == "adaptive":
        return Config.ADAPTIVE_PENALTY_WEIGHT
    if strategy == "forward_consistency":
        return Config.FORWARD_CONSISTENCY_WEIGHT
    return 0.0


def compute_loss(model, s: torch.Tensor, x_ic: torch.Tensor, p_ic: torch.Tensor,
                 physics_fn, strategy: str = None, **kw):
    device = Config.DEVICE
    s = s.to(device).requires_grad_(True)
    x, p = model(s)

    # base terms
    L_phys = physics_loss(x, p, physics_fn)
    dx = grad(x, s, torch.ones_like(x), create_graph=True)[0]
    dp = grad(p, s, torch.ones_like(p), create_graph=True)[0]
    L_arc = arc_length_loss(dx, dp)
    x0, p0 = model(torch.zeros(1, 1, device=device))
    L_ic = ic_loss(x0, p0, x_ic.to(device), p_ic.to(device))

    # direction strategy
    L_dir = direction_penalty(x, p, s, dx, dp, strategy)

    # extra constraints for saddle-node tracking (keep the path on p<=0 and monotone toward fold)
    L_p_nonpos = p_nonpos_loss(p)
    L_p_mono = p_monotonic_loss(dp, margin=Config.P_MONO_MARGIN)

    # physics weighting: fixed or uncertainty-weighted
    if getattr(Config, 'USE_FIXED_PHYS_WEIGHT', False):
        phys_term = Config.PHYS_FIXED_WEIGHT * L_phys
        w_phys_scalar = float(Config.PHYS_FIXED_WEIGHT)
        phys_mode = 'fixed'
    else:
        phys_term = 0.5 * torch.exp(-model.log_sigma_physics) * L_phys + model.log_sigma_physics
        w_phys_scalar = float(0.5 * torch.exp(-model.log_sigma_physics).detach().cpu().item())
        phys_mode = 'auto'

    # direction term: auto or fixed
    if strategy is not None and L_dir.requires_grad:
        if getattr(Config, 'AUTO_DIRECTION_WEIGHT', False):
            w_dir_eff = 0.5 * torch.exp(-model.log_sigma_dir)
            dir_term = w_dir_eff * L_dir + model.log_sigma_dir
            dir_mode = 'auto'
            w_dir_scalar = float(w_dir_eff.detach().cpu().item())
        else:
            w_dir = _penalty_weight(strategy)
            dir_term = w_dir * L_dir
            dir_mode = 'fixed'
            w_dir_scalar = float(w_dir)
    else:
        dir_term = torch.tensor(0.0, device=device)
        dir_mode = 'none'
        w_dir_scalar = 0.0

    total = (
        phys_term
        + 0.5 * torch.exp(-model.log_sigma_arc) * L_arc + model.log_sigma_arc
        + 0.5 * torch.exp(-model.log_sigma_ic) * L_ic + model.log_sigma_ic
        + dir_term
        + Config.P_NONPOS_WEIGHT * L_p_nonpos
        + Config.P_MONOTONIC_WEIGHT * L_p_mono
    )

    comps = {
        'total': float(total.detach().cpu().item()),
        'physics': float(L_phys.detach().cpu().item()),
        'arc': float(L_arc.detach().cpu().item()),
        'ic': float(L_ic.detach().cpu().item()),
        'direction': float(L_dir.detach().cpu().item()) if strategy is not None else 0.0,
        'w_direction': w_dir_scalar,
        'dir_weight_mode': dir_mode,
        'w_physics': w_phys_scalar,
        'phys_weight_mode': phys_mode,
        'p_nonpos': float(L_p_nonpos.detach().cpu().item()),
        'p_mono': float(L_p_mono.detach().cpu().item()),
    }
    return total, comps
    