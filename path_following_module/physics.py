import torch
import numpy as np


def physics_equation(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Steady-state residual for **saddle-node** ODE: xdot = r + x^2, with r ≡ p.
    Equilibria satisfy F(x,p) = x^2 + p = 0.
    """
    return x**2 + p


def physics_fn_inv(p: np.ndarray) -> np.ndarray:
    """Given parameter p (r), return |x| branch: x = sqrt(-p) for p<=0, NaN otherwise."""
    p = np.asarray(p)
    x = np.sqrt(np.clip(-p, 0.0, None))
    return x


def get_theoretical_saddle_node(num_points: int = 1000, p_min: float = -2.0, p_max: float = 0.0):
    """Generate theoretical equilibrium branches for saddle-node: x = ±sqrt(-p), p≤0.
    Returns a dict with arrays: p, x_plus (unstable), x_minus (stable), stable_mask.
    Stability: f_x=2x < 0 ⇒ x<0 stable; x>0 unstable.
    """
    p_theory = np.linspace(p_min, p_max, num_points)
    x_abs = physics_fn_inv(p_theory)
    x_plus = x_abs          # unstable
    x_minus = -x_abs        # stable
    stable_mask = p_theory <= 0
    return {"p": p_theory, "x_plus": x_plus, "x_minus": x_minus, "stable_mask": stable_mask}
