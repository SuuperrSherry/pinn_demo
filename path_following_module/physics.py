import torch
import numpy as np

def physics_equation(x, p):
    """ x**2 + p**2 - 1 = 0 """
    return x - p**2

def physics_fn_inv(p):
    """Inverse function for the physics equation to get x from p."""
    return np.sqrt(1 - p**2)

def get_theoretical_ellipse(num_points=1000):
    """Generate theoretical ellipse points for the physics equation."""
    p_theory = np.linspace(-1, 1, num_points)
    x_theory = physics_fn_inv(p_theory)
    return x_theory, p_theory