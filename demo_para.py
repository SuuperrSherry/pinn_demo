import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

# Set random seeds
torch.manual_seed(123)
np.random.seed(123)

# Define the Physics-Informed Neural Network (PINN)
class PINN_ArcLength(nn.Module):
    def __init__(self, hidden_layers=4, hidden_neurons=50):
        super(PINN_ArcLength, self).__init__()
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_neurons, 2))  # Outputs: [x, p]
        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.init_weights()

        # Learnable log-variance for automatic loss weighting
        self.log_sigma_physics = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_arc = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_ic = nn.Parameter(torch.tensor(0.0))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, s):
        output = self.network(s)
        x = output[:, 0:1]
        p = output[:, 1:2]
        return x, p

# Define the physics equation: F(x, p) = x² + p = 0
# Original: return x**2 + p
def physics_equation(x, p):
    return x - p**2  # now x = p^2


# Compute dx/ds and dp/ds using autograd
def compute_derivatives(x, p, s):
    dx_ds = grad(x, s, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    dp_ds = grad(p, s, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    return dx_ds, dp_ds

# Compute the total loss
def compute_loss(model, s_points, x0, p0):
    s_points.requires_grad_(True)
    x_pred, p_pred = model(s_points)
    dx_ds, dp_ds = compute_derivatives(x_pred, p_pred, s_points)

    # Physics loss: x² + p = 0
    physics_residual = physics_equation(x_pred, p_pred)
    loss_physics = torch.mean(physics_residual ** 2)

    # Arc length constraint: (dx/ds)² + (dp/ds)² = 1
    arclength_residual = dx_ds ** 2 + dp_ds ** 2 - 1
    loss_arclength = torch.mean(arclength_residual ** 2)

    # Initial condition loss
    x_start, p_start = model(torch.tensor([[0.0]], dtype=torch.float32))
    loss_ic = (x_start - x0) ** 2 + (p_start - p0) ** 2
    loss_ic = torch.mean(loss_ic)

    # Combine loss with learnable log weights
    total_loss = (
        0.5 * torch.exp(-model.log_sigma_physics) * loss_physics + model.log_sigma_physics +
        0.5 * torch.exp(-model.log_sigma_arc) * loss_arclength + model.log_sigma_arc +
        0.5 * torch.exp(-model.log_sigma_ic) * loss_ic + model.log_sigma_ic
    )
    return total_loss


# Initialize model and optimizer
model = PINN_ArcLength()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define arc-length parameter (s or lambda)
s_points = torch.linspace(-2, 2, 200).view(-1, 1)  # range from -2 to 2
x0 = torch.tensor([[0.0]])
p0 = torch.tensor([[0.0]])

# Training loop
for step in range(3000):
    optimizer.zero_grad()
    loss = compute_loss(model, s_points, x0, p0)
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(f"Step {step} - Loss: {loss.item():.6f}")

# Predict
x_pred, p_pred = model(s_points)
x_pred = x_pred.detach().numpy()
p_pred = p_pred.detach().numpy()

# For comparison: true x = p^2
p_true = np.linspace(-1.1, 1.1, 200)
x_true = p_true**2

# Plot predicted vs true path
plt.figure(figsize=(8, 6))
plt.plot(x_pred, p_pred, 'b-', label='Predicted Path')
plt.plot(x_true, p_true, 'r--', label='True Path: x = p²')
plt.xlabel("x")
plt.ylabel("p")
plt.title("PINN: Predicted vs True Path")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print learned loss weights
sigma_physics = torch.exp(model.log_sigma_physics).item()
sigma_arc = torch.exp(model.log_sigma_arc).item()
sigma_ic = torch.exp(model.log_sigma_ic).item()

print("\nLearned loss weights (w):")
print(f"  w_physics     = {sigma_physics:.6f}")
print(f"  w_arc_length  = {sigma_arc:.6f}")
print(f"  w_initial     = {sigma_ic:.6f}")
