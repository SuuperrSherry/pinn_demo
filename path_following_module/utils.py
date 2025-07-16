import numpy as np

def auto_start_point(physics_fn_inv, p_range=(-1,1), num_samples=100):
    """
    Automatically find a valid start point (x0, p0) on the curve by sampling p.
    """
    for p0 in np.linspace(*p_range, num_samples):
        try:
            x0 = physics_fn_inv(p0)
            if np.isfinite(x0) and np.imag(x0) == 0:
                return (np.real(x0), p0)
        except Exception:
            continue
    raise RuntimeError("Could not find a valid start point on the curve.")

