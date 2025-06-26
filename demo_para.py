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
    def __init__(self, hidden_layers=3, hidden_neurons=32):
        super(PINN_ArcLength, self).__init__()
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_neurons, 2))  # Output: x and p
        self.network = nn.Sequential(*layers)

        self.init_weights()

        # Learnable log variances for dynamic loss weights
        self.log_sigma_physics = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_arc = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_ic = nn.Parameter(torch.tensor(0.0))

    def init_weights(self): 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, s):
        out = self.network(s)
        x = out[:, 0:1]
        p = out[:, 1:2]
        return x, p

 
# Physics equation: x = p²
def physics_equation(x, p):
    return x - p**2

# Compute derivatives
def compute_derivatives(x, p, s):
    dx_ds = grad(x, s, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    dp_ds = grad(p, s, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    return dx_ds, dp_ds

# Improved loss function with better weighting
def compute_loss(model, s_points, x_ic, p_ic, s_ic=0.0):
    s_points.requires_grad_(True)
    x_pred, p_pred = model(s_points)
    dx_ds, dp_ds = compute_derivatives(x_pred, p_pred, s_points)

    # Physics constraint
    physics_residual = physics_equation(x_pred, p_pred)
    loss_physics = torch.mean(physics_residual ** 2)

    # Arc length constraint
    arclength_residual = dx_ds ** 2 + dp_ds ** 2 - 1
    loss_arclength = torch.mean(arclength_residual ** 2)

    # Initial condition
    s_ic_tensor = torch.tensor([[s_ic]], dtype=torch.float32, requires_grad=True)
    x_start, p_start = model(s_ic_tensor)
    loss_ic = torch.mean((x_start - x_ic)**2 + (p_start - p_ic)**2)

    # Dynamic weighting
    total_loss = (
        0.5 * torch.exp(-model.log_sigma_physics) * loss_physics + model.log_sigma_physics +
        0.5 * torch.exp(-model.log_sigma_arc) * loss_arclength + model.log_sigma_arc +
        0.5 * torch.exp(-model.log_sigma_ic) * loss_ic + model.log_sigma_ic
    )

    return total_loss, loss_physics, loss_arclength, loss_ic


# Segment training with local coordinate system
def train_segment(s_start, s_end, x_start, p_start, num_points=20, epochs=1000):
    # Create a new model for each segment to avoid catastrophic forgetting
    model = PINN_ArcLength()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training points in the segment
    s_train = torch.linspace(s_start, s_end, num_points).view(-1, 1)
    
    best_loss = float('inf')
    patience = 200
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss, loss_physics, loss_arclength, loss_ic = compute_loss(
            model, s_train, x_start, p_start, s_start
        )
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}, "
                  f"Physics: {loss_physics.item():.6f}, "
                  f"Arc: {loss_arclength.item():.6f}, "
                  f"IC: {loss_ic.item():.6f}")
        
        # Early stopping
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return model

# Path following with adaptive step size
def path_following(start_point, total_length=2.0, base_step=0.2, max_segments=20):
    x_current, p_current = start_point
    s_current = 0.0
    
    # Storage for results
    path_x = [x_current]
    path_p = [p_current]
    path_s = [s_current]
    
    print(f"Starting path following from ({x_current:.3f}, {p_current:.3f})")
    
    for segment in range(max_segments):
        if s_current >= total_length:
            break
            
        # Determine segment length
        s_end = min(s_current + base_step, total_length)
        
        print(f"\n=== Segment {segment + 1} ===")
        print(f"s: {s_current:.3f} -> {s_end:.3f}")
        print(f"Starting point: ({x_current:.3f}, {p_current:.3f})")
        
        # Train segment
        model = train_segment(
            s_current, s_end, 
            torch.tensor([[x_current]]), torch.tensor([[p_current]])
        )
        
        # Get trajectory for this segment
        s_segment = torch.linspace(s_current, s_end, 50).view(-1, 1)
        with torch.no_grad():
            x_segment, p_segment = model(s_segment)
        
        with torch.no_grad():
            s0 = torch.tensor([[s_current]], dtype=torch.float32)
            x0_pred, p0_pred = model(s0)
            print(f"Predicted start point (model): ({x0_pred.item():.4f}, {p0_pred.item():.4f})")
            print(f"Target start point (truth):   ({x_current:.4f}, {p_current:.4f})")
        
        # Add points to path (skip first point to avoid duplication)
        path_x.extend(x_segment[1:].squeeze().tolist())
        path_p.extend(p_segment[1:].squeeze().tolist())
        path_s.extend(s_segment[1:].squeeze().tolist())
        
        # Update current position
        x_current = x_segment[-1].item()
        p_current = p_segment[-1].item()
        s_current = s_end
        
        print(f"End point: ({x_current:.3f}, {p_current:.3f})")
        
        # Adaptive step size based on curvature (optional enhancement)
        # For parabola x = p², curvature is higher near p = 0
        curvature_factor = 1.0 / (1.0 + abs(p_current))
        base_step = max(0.1, min(0.3, 0.2 * curvature_factor))
    
    return path_x, path_p, path_s

# Run path following
print("Starting improved path following...")

# Better starting point - avoid the problematic (0,0)
start_x, start_p = 0.25, 0.5  # Point (0.25, 0.5) is on the curve x = p²

path_x, path_p, path_s = path_following(
    (start_x, start_p), 
    total_length=3.0, 
    base_step=0.25,
    max_segments=15
)

# Generate true path for comparison
p_true = np.linspace(-1.5, 1.5, 300)
x_true = p_true**2

# Plotting
plt.figure(figsize=(12, 8))

# Main plot
plt.subplot(2, 2, 1)
plt.plot(x_true, p_true, 'r-', linewidth=2, label='True Path: x = p²')
plt.plot(path_x, path_p, 'bo-', markersize=4, linewidth=1, label='PINN Predicted Path')
plt.plot(start_x, start_p, 'go', markersize=8, label='Start Point')
plt.xlabel('x')
plt.ylabel('p')
plt.title('Path Following Results')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Error analysis
plt.subplot(2, 2, 2)
# Compute error for predicted points
errors = []
for x_pred, p_pred in zip(path_x, path_p):
    error = abs(x_pred - p_pred**2)
    errors.append(error)

plt.plot(path_s, errors, 'b-', linewidth=2)
plt.xlabel('Arc length s')
plt.ylabel('Physics constraint error |x - p²|')
plt.title('Physics Constraint Error Along Path')
plt.grid(True)
plt.yscale('log')

# Path in parameter space
plt.subplot(2, 2, 3)
plt.plot(path_s, path_x, 'b-', label='x(s)')
plt.plot(path_s, path_p, 'r-', label='p(s)')
plt.xlabel('Arc length s')
plt.ylabel('Value')
plt.title('Path Components vs Arc Length')
plt.legend()
plt.grid(True)

# Zoom in on starting region
plt.subplot(2, 2, 4)
plt.plot(x_true, p_true, 'r-', linewidth=2, label='True Path')
plt.plot(path_x[:30], path_p[:30], 'bo-', markersize=6, label='PINN Path (first 30 points)')
plt.plot(start_x, start_p, 'go', markersize=10, label='Start Point')
plt.xlabel('x')
plt.ylabel('p')
plt.title('Zoom: Starting Region')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print statistics
print(f"\nPath Following Statistics:")
print(f"Total points generated: {len(path_x)}")
print(f"Arc length covered: {path_s[-1]:.3f}")
print(f"Mean physics error: {np.mean(errors):.6f}")
print(f"Max physics error: {np.max(errors):.6f}")
print(f"Final point: ({path_x[-1]:.3f}, {path_p[-1]:.3f})")