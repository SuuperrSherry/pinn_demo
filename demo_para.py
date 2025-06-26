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

# Segment training function
def segment_train(model, optimizer, s_start, x0, p0, step_size=0.2, segment_len=50, steps_per_segment=500):
    s_segment = torch.linspace(s_start, s_start + step_size * (segment_len - 1), segment_len).view(-1, 1)
    s_segment.requires_grad_(True)

    for step in range(steps_per_segment):
        optimizer.zero_grad()
        loss = compute_loss(model, s_segment, x0, p0)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Segment Start s={s_start:.3f}, Step {step}, Loss={loss.item():.6f}")

    with torch.no_grad():
        x_last, p_last = model(s_segment[-1:])
    return s_segment.detach(), x_last.detach(), p_last.detach()

# Adapt step size based on Euclidean distance between segment endpoints
def adaptive_step(x_prev, p_prev, x_new, p_new, base=1e-2, max_step=0.3):
    delta = torch.sqrt((x_new - x_prev) ** 2 + (p_new - p_prev) ** 2)
    return min(float(delta.item()) + base, max_step)

# Initialize model and optimizer
model = PINN_ArcLength()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Generate true path for comparison
x0 = torch.tensor([[0.0]])
p0 = torch.tensor([[0.0]])
s_start = 0.0
segments = 10
step_size = 0.1
all_x, all_p = [], []

# Train segments iteratively
for seg in range(segments):
    print(f"\n>> Training Segment {seg+1}")
    s_segment, x0, p0 = segment_train(model, optimizer, s_start, x0, p0, step_size=step_size)
    all_x.append(x0.item())
    all_p.append(p0.item())

    if seg > 0:
        step_size = adaptive_step(torch.tensor([[all_x[-2]]]), torch.tensor([[all_p[-2]]]), x0, p0)

    s_start = s_segment[-1].item()

# Plot predicted vs true path
# Plot final predicted vs true path
plt.figure(figsize=(8, 6))
plt.plot(all_x, all_p, 'bo-', label='Predicted Path')
p_true = np.linspace(-1.1, 1.1, 200)
x_true = p_true**2
plt.plot(x_true, p_true, 'r--', label='True Path: x = p²')
plt.xlabel("x")
plt.ylabel("p")
plt.title("PINN: Segmented Adaptive Training Path")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
