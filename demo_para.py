import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

torch.manual_seed(123)
np.random.seed(123)

class PINN_ArcLength(nn.Module):
    """
    Physics Informed Neural Network for Arc-Length Parameterization
    Input: s (landa)
    Output: [x(s), p(s)] 
    """
    def __init__(self, hidden_layers=4, hidden_neurons=50):
        super(PINN_ArcLength, self).__init__()
        
        layers = []
        layers.append(nn.Linear(1, hidden_neurons))  # Input layer: s -> hidden
        
        for i in range(hidden_layers):
            layers.append(nn.Tanh()) 
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
        
        layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_neurons, 2))  # Output layer: hidden -> [x, p]
        
        self.network = nn.Sequential(*layers)
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, s):
        output = self.network(s)
        x = output[:, 0:1]  # x
        p = output[:, 1:2]  # p
        return x, p

def physics_equation(x, p):
    """
    F(x, p) = x² + p = 0
    """
    return x**2 + p

def compute_derivatives(x, p, s):
    """
    dx/ds, dp/ds
    """
    dx_ds = grad(x, s, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    dp_ds = grad(p, s, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    return dx_ds, dp_ds

def compute_loss(model, s_points, x0, p0):
    """
    
    Args:
        model: PINN
        s_points: LANDA
        x0, p0: START POINT
    
    Returns:
        total_loss: 
        loss_components
    """
    s_points.requires_grad_(True)
    
   
    x_pred, p_pred = model(s_points)
    
   
    dx_ds, dp_ds = compute_derivatives(x_pred, p_pred, s_points)
    
    physics_residual = physics_equation(x_pred, p_pred)
    loss_physics = torch.mean(physics_residual**2)
    
    arclength_residual = dx_ds**2 + dp_ds**2 - 1
    loss_arclength = torch.mean(arclength_residual**2)
    

    # Loss
    lambda_physics = 1.0      
    lambda_arclength = 1.0    

    
    total_loss = (lambda_physics * loss_physics + 
                  lambda_arclength * loss_arclength)
    
   
    loss_components = {
        'total': total_loss.item(),
        'physics': loss_physics.item(),
        'arclength': loss_arclength.item()
    }
    
    return total_loss, loss_components