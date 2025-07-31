import torch
import numpy as np
from strategies import DirectionStrategies
from torch.autograd import grad
from config import Config
from strategies import STRATEGY_MAP



def compute_derivatives(x, p, s_points):
    """Compute the derivatives dx/ds and dp/ds using autograd."""
    dx_ds = grad(x, s_points, torch.ones_like(x), create_graph=True)[0]
    dp_ds = grad(p, s_points, torch.ones_like(p), create_graph=True)[0]
    return dx_ds, dp_ds

def compute_loss(model, s_points, x_ic, p_ic, s_ic, physics_fn, 
                 direction_strategy=None, **strategy_kwargs):
    # compute the model output
    s_points.requires_grad_(True)
    x_pred, p_pred = model(s_points)
    dx_ds, dp_ds = compute_derivatives(x_pred, p_pred, s_points)
    
    # physics loss
    physics_residual = physics_fn(x_pred, p_pred)
    loss_physics = torch.mean(physics_residual ** 2)
    loss_arclength = torch.mean((dx_ds**2 + dp_ds**2 - 1)**2)
    
    # initial condition loss
    s_ic_tensor = torch.tensor([[s_ic]], dtype=torch.float32, requires_grad=True)
    x_start, p_start = model(s_ic_tensor)
    loss_ic = torch.mean((x_start - x_ic)**2 + (p_start - p_ic)**2)
    
    # total loss
    total_loss = (
        0.5 * torch.exp(-model.log_sigma_physics) * loss_physics + model.log_sigma_physics +
        0.5 * torch.exp(-model.log_sigma_arc) * loss_arclength + model.log_sigma_arc +
        0.5 * torch.exp(-model.log_sigma_ic) * loss_ic + model.log_sigma_ic
    )
    
    # apply direction strategy if specified
    if direction_strategy and direction_strategy in STRATEGY_MAP:
        strategy_fn = STRATEGY_MAP[direction_strategy]
        
        if direction_strategy in ['l2', 'cosine']:
            # L2 or cosine penalty for direction consistency
            # dx_ds and dp_ds are already computed, for simplification
            if len(dx_ds) > 1:
                dx_last = dx_ds[:-1].detach()
                dp_last = dp_ds[:-1].detach()
                dx_current = dx_ds[1:]
                dp_current = dp_ds[1:]
                penalty = strategy_fn(dx_current, dp_current, dx_last, dp_last)
                total_loss += Config.DIRECTION_PENALTY_WEIGHT * penalty
            
        elif direction_strategy == 'smooth':
            penalty = strategy_fn(dx_ds, dp_ds, s_points)
            total_loss += Config.SMOOTH_PENALTY_WEIGHT * penalty
            
        elif direction_strategy == 'global_forward':
            penalty = strategy_fn(dx_ds, dp_ds, strategy_kwargs.get('global_direction'))
            total_loss += Config.GLOBAL_PENALTY_WEIGHT * penalty

        elif direction_strategy == 'adaptive':
            penalty = strategy_fn(dx_ds, dp_ds)
            total_loss += Config.ADAPTIVE_PENALTY_WEIGHT * penalty

        elif direction_strategy == 'forward_consistency':
            penalty = strategy_fn(x_pred, p_pred, dx_ds, dp_ds)
            total_loss += Config.FORWARD_CONSISTENCY_WEIGHT * penalty

            
    if 'init_direction' in strategy_kwargs and strategy_kwargs['init_direction'] is not None:
        init_dir = strategy_kwargs['init_direction']
        tangent_0 = torch.cat([dx_ds[0], dp_ds[0]], dim=0)
        dir_unit = init_dir / (torch.norm(init_dir) + 1e-8)
        tan_unit = tangent_0 / (torch.norm(tangent_0) + 1e-8)
        cosine_sim = torch.sum(tan_unit * dir_unit)
        direction_penalty = 1 - cosine_sim  # 惩罚夹角
        total_loss += Config.DIRECTION_PENALTY_WEIGHT * direction_penalty

    
    return total_loss, loss_physics, loss_arclength, loss_ic