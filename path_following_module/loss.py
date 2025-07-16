import torch
from config import Config
from torch.autograd import grad

def compute_derivatives(x, p, s):
    dx_ds = grad(x, s, torch.ones_like(x), create_graph=True)[0]
    dp_ds = grad(p, s, torch.ones_like(p), create_graph=True)[0]
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
    monotonic_penalty = torch.mean(torch.relu(dp_ds)) ** 2
    total_loss = (
        0.5 * torch.exp(-model.log_sigma_physics) * loss_physics + model.log_sigma_physics +
        0.5 * torch.exp(-model.log_sigma_arc) * loss_arclength + model.log_sigma_arc +
        0.5 * torch.exp(-model.log_sigma_ic) * loss_ic + model.log_sigma_ic +
        Config.MONOTONIC_PENALTY_WEIGHT * monotonic_penalty
    )
    return total_loss, loss_physics, loss_arclength, loss_ic