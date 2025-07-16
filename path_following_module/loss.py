import torch
from config import Config
from torch.autograd import grad

def compute_direction_penalty(dx_ds, dp_ds, dx_last, dp_last):
    """
    Compute the penalty that encourages the derivative (dx/ds, dp/ds)
    to stay close to the previous segment's direction.
    """
    return torch.mean((dx_ds - dx_last)**2 + (dp_ds - dp_last)**2)


def compute_derivatives(x, p, s_points):
    dx_ds = grad(x, s_points, torch.ones_like(x), create_graph=True)[0]
    dp_ds = grad(p, s_points, torch.ones_like(p), create_graph=True)[0]
    return dx_ds, dp_ds

def compute_loss(model, s_points, x_ic, p_ic, s_ic, physics_fn):
    s_points.requires_grad_(True)
    x_pred, p_pred = model(s_points)
    dx_ds, dp_ds = compute_derivatives(x_pred, p_pred, s_points)
    physics_residual = physics_fn(x_pred, p_pred)
    loss_physics = torch.mean(physics_residual ** 2)
    loss_arclength = torch.mean((dx_ds**2 + dp_ds**2 - 1)**2)
    s_ic_tensor = torch.tensor([[s_ic]], dtype=torch.float32, requires_grad=True)
    x_start, p_start = model(s_ic_tensor)
    loss_ic = torch.mean((x_start - x_ic)**2 + (p_start - p_ic)**2)
    # monotonic_penalty = torch.mean(torch.relu(dp_ds)) ** 2
    total_loss = (
        0.5 * torch.exp(-model.log_sigma_physics) * loss_physics + model.log_sigma_physics +
        0.5 * torch.exp(-model.log_sigma_arc) * loss_arclength + model.log_sigma_arc +
        0.5 * torch.exp(-model.log_sigma_ic) * loss_ic + model.log_sigma_ic 
        # +Config.MONOTONIC_PENALTY_WEIGHT * monotonic_penalty
    )
    d2x_ds2 = grad(dx_ds, s_points, torch.ones_like(dx_ds), create_graph=True)[0]
    d2p_ds2 = grad(dp_ds, s_points, torch.ones_like(dp_ds), create_graph=True)[0]
    loss_smooth = torch.mean(d2x_ds2**2 + d2p_ds2**2)
    total_loss += 0.1 * loss_smooth

    return total_loss, loss_physics, loss_arclength, loss_ic