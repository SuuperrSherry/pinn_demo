import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
from typing import Tuple, List, Optional


# Configuration - Tunable Parameters (MUST report in paper)
class Config:
    """
    Configuration class containing all tunable hyperparameters
    """
    # Network Architecture Parameters 
    HIDDEN_LAYERS = 3           # Number of hidden layers
    HIDDEN_NEURONS = 32         # Number of neurons per layer
    
    # Training Parameters 
    LEARNING_RATE = 1e-3        # Learning rate for Adam optimizer
    MAX_EPOCHS = 1000           # Maximum training epochs
    PRINT_INTERVAL = 100        # Print interval for training progress
    
    # Path Following Parameters 
    TOTAL_LENGTH = 3.0          # Total arc length to follow
    BASE_STEP_SIZE = 0.25       # Base step size for segments
    MAX_SEGMENTS = 15           # Maximum number of segments
    POINTS_PER_SEGMENT = 40     # Number of training points per segment
    
    # Loss Function Weights 
    MONOTONIC_PENALTY_WEIGHT = 1.0  # Weight for monotonic penalty
    
    # Numerical Stability Parameters
    MIN_STEP_SIZE = 0.05        # Minimum adaptive step size
    MAX_STEP_SIZE = 0.3         # Maximum adaptive step size
    
    # Random Seeds
    TORCH_SEED = 123
    NUMPY_SEED = 123

def set_random_seeds(torch_seed: int = Config.TORCH_SEED, 
                    numpy_seed: int = Config.NUMPY_SEED):
    """Set random seeds for reproducibility"""
    torch.manual_seed(torch_seed)
    np.random.seed(numpy_seed)

# =============================================================================
# Physics-Informed Neural Network Definition
# =============================================================================
class PINN_ArcLength(nn.Module):
    """
    Physics-Informed Neural Network for arc-length parameterization
    
    Architecture: Input(1) -> Hidden Layers -> Output(2)
    Activation: Tanh (commonly used in PINNs - cite relevant papers)
    """
    
    def __init__(self, hidden_layers: int = Config.HIDDEN_LAYERS, 
                 hidden_neurons: int = Config.HIDDEN_NEURONS):
        super(PINN_ArcLength, self).__init__()
        
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_neurons, 2))  # Output: x and p
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
        
        # Learnable log variances for adaptive loss weighting
        self.log_sigma_physics = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_arc = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_ic = nn.Parameter(torch.tensor(0.0))
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        output = self.network(s)
        x = output[:, 0:1]
        p = output[:, 1:2]
        return x, p

# Physics Equations and Derivatives

def physics_equation(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Physics constraint equation: x = p²
    """
    return x - p**2

def compute_derivatives(x: torch.Tensor, p: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute first-order derivatives using automatic differentiation
    Returns: dx/ds, dp/ds
    """
    dx_ds = grad(x, s, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    dp_ds = grad(p, s, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    return dx_ds, dp_ds


# Loss Function with Multi-objective Optimization
def compute_loss(model, s_points, x_ic, p_ic, s_ic, physics_fn):
    s_points.requires_grad_(True)
    x_pred, p_pred = model(s_points)
    dx_ds, dp_ds = compute_derivatives(x_pred, p_pred, s_points)
    
    # Now use passed-in physics function
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


# Training Function

def train_segment(s_start, s_end, x_start, p_start, physics_fn,
                  num_points=Config.POINTS_PER_SEGMENT, max_epochs=Config.MAX_EPOCHS):
    model = PINN_ArcLength()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    s_train = torch.linspace(s_start, s_end, num_points).view(-1, 1)
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        total_loss, loss_physics, loss_arclength, loss_ic = compute_loss(
            model, s_train, x_start, p_start, s_start, physics_fn=physics_fn
        )
        total_loss.backward()
        optimizer.step()
        
        if epoch % Config.PRINT_INTERVAL == 0:
            print(f"Epoch {epoch:4d}, Total: {total_loss.item():.6f}, "
                  f"Physics: {loss_physics.item():.6f}, "
                  f"Arc: {loss_arclength.item():.6f}, IC: {loss_ic.item():.6f}")
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    return model


# Path Following Algorithm with Adaptive Step Size
def path_following(start_point, physics_fn,
                  total_length=Config.TOTAL_LENGTH,
                  base_step=Config.BASE_STEP_SIZE,
                  max_segments=Config.MAX_SEGMENTS):
    x_current, p_current = start_point
    s_current = 0.0
    path_x, path_p, path_s = [x_current], [p_current], [s_current]
    
    print(f"Starting path following from ({x_current:.3f}, {p_current:.3f})")
    for segment in range(max_segments):
        if s_current >= total_length:
            break
        s_end = min(s_current + base_step, total_length)
        
        print(f"\n=== Segment {segment + 1} ===")
        print(f"s: {s_current:.3f} -> {s_end:.3f}")
        print(f"Starting point: ({x_current:.3f}, {p_current:.3f})")
        
        model = train_segment(s_current, s_end,
                              torch.tensor([[x_current]]), torch.tensor([[p_current]]),
                              physics_fn=physics_fn)
        
        s_segment = torch.linspace(s_current, s_end, 50).view(-1, 1)
        with torch.no_grad():
            x_segment, p_segment = model(s_segment)
        
        with torch.no_grad():
            s0 = torch.tensor([[s_current]], dtype=torch.float32)
            x0_pred, p0_pred = model(s0)
            print(f"Predicted start: ({x0_pred.item():.4f}, {p0_pred.item():.4f}), "
                  f"Target start: ({x_current:.4f}, {p_current:.4f})")
        
        path_x.extend(x_segment[1:].squeeze().tolist())
        path_p.extend(p_segment[1:].squeeze().tolist())
        path_s.extend(s_segment[1:].squeeze().tolist())
        
        x_current, p_current, s_current = x_segment[-1].item(), p_segment[-1].item(), s_end
        curvature_factor = 1.0 / (1.0 + abs(p_current))
        base_step = max(Config.MIN_STEP_SIZE, 
                       min(Config.MAX_STEP_SIZE, Config.BASE_STEP_SIZE * curvature_factor))
    return path_x, path_p, path_s


# =============================================================================
# Visualization and Analysis Functions
# =============================================================================
def plot_results(path_x: List[float], path_p: List[float], path_s: List[float],
                start_point: Tuple[float, float]):
    """
    Create comprehensive visualization of results
    """



    # Generate analytical solution for comparison
    p_true = np.linspace(-1.5, 1.5, 300)
    x_true = p_true**2
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Main trajectory plot
    axes[0, 0].plot(x_true, p_true, 'r-', linewidth=2, label='True Path: x = p²')
    axes[0, 0].plot(path_x, path_p, 'bo-', markersize=4, linewidth=1, 
                   label='PINN Predicted Path')
    axes[0, 0].plot(start_point[0], start_point[1], 'go', markersize=8, 
                   label='Start Point')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('p')
    axes[0, 0].set_title('Path Following Results')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].axis('equal')
    
    # Physics constraint error analysis
    errors = [abs(x - p**2) for x, p in zip(path_x, path_p)]
    axes[0, 1].plot(path_s, errors, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Arc length s')
    axes[0, 1].set_ylabel('Physics constraint error |x - p²|')
    axes[0, 1].set_title('Physics Constraint Error Along Path')
    axes[0, 1].grid(True)
    axes[0, 1].set_yscale('log')
    
    # Path components vs arc length
    axes[1, 0].plot(path_s, path_x, 'b-', label='x(s)')
    axes[1, 0].plot(path_s, path_p, 'r-', label='p(s)')
    axes[1, 0].set_xlabel('Arc length s')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Path Components vs Arc Length')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Zoom in on starting region
    n_zoom = min(30, len(path_x))
    axes[1, 1].plot(x_true, p_true, 'r-', linewidth=2, label='True Path')
    axes[1, 1].plot(path_x[:n_zoom], path_p[:n_zoom], 'bo-', markersize=6, 
                   label=f'PINN Path (first {n_zoom} points)')
    axes[1, 1].plot(start_point[0], start_point[1], 'go', markersize=10, 
                   label='Start Point')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('p')
    axes[1, 1].set_title('Zoom: Starting Region')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return errors

def print_statistics(path_x: List[float], path_p: List[float], path_s: List[float], 
                    errors: List[float]):
    """
    Print comprehensive statistics for the paper
    
    IMPORTANT: Include these statistics in your results section
    """
    print(f"\n{'='*50}")
    print(f"PATH FOLLOWING STATISTICS")
    print(f"{'='*50}")
    print(f"Total points generated: {len(path_x)}")
    print(f"Arc length covered: {path_s[-1]:.3f}")
    print(f"Mean physics error: {np.mean(errors):.6f}")
    print(f"Std physics error: {np.std(errors):.6f}")
    print(f"Max physics error: {np.max(errors):.6f}")
    print(f"Min physics error: {np.min(errors):.6f}")
    print(f"Final point: ({path_x[-1]:.3f}, {path_p[-1]:.3f})")
    print(f"{'='*50}")

# =============================================================================
# Main Execution
# =============================================================================
def main():
    """
    Main execution function
    """
    # Set random seeds for reproducibility
    set_random_seeds()
    
    print("Starting improved PINN path following...")
    print(f"Configuration: {Config.HIDDEN_LAYERS} layers, {Config.HIDDEN_NEURONS} neurons")
    
    # Choose starting point (avoid problematic (0,0))
    # This choice should be justified in the paper
    start_x, start_p = 0.25, 0.5  # Point (0.25, 0.5) is on the curve x = p²
    
    # Execute path following
    path_x, path_p, path_s = path_following(
        (start_x, start_p), 
        total_length=Config.TOTAL_LENGTH, 
        base_step=Config.BASE_STEP_SIZE,
        max_segments=Config.MAX_SEGMENTS
    )
    
    # Visualize results
    errors = plot_results(path_x, path_p, path_s, (start_x, start_p))
    
    # Print statistics
    print_statistics(path_x, path_p, path_s, errors)

if __name__ == "__main__":
    main()

# =============================================================================
# IMPORTANT NOTES FOR RESEARCH PAPER
# =============================================================================
"""
PARAMETERS TO REPORT IN PAPER:
1. Network architecture: 3 hidden layers, 32 neurons per layer
2. Activation function: Tanh
3. Optimizer: Adam with learning rate 1e-3
4. Training parameters: Max 1000 epochs, early stopping patience 200
5. Segmentation: 40 collocation points per segment, max 15 segments
6. Arc length: Total length 3.0, base step size 0.25
7. Random seeds: torch=123, numpy=123

METHODS TO CITE:
1. Xavier initialization: Glorot & Bengio (2010)
2. Adaptive loss weighting: Kendall & Gal (2017) - "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
3. Physics-Informed Neural Networks: Raissi et al. (2019)
4. Arc-length parameterization: Numerical continuation literature
5. Early stopping: General deep learning literature

KEY CONTRIBUTIONS TO HIGHLIGHT:
1. Segmented training approach to avoid catastrophic forgetting
2. Adaptive loss weighting using learnable uncertainties
3. Arc-length parameterization for curve following
4. Monotonic penalty for directional constraints

ABLATION STUDIES TO CONSIDER:
1. Effect of number of segments
2. Effect of network architecture
3. Effect of different loss weightings
4. Comparison with and without monotonic penalty
"""