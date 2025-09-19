import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})

# 1. Saddle-node bifurcation
def plot_saddle_node():
    r = np.linspace(-1, 1, 400)
    x_pos = np.sqrt(np.clip(-r, 0, None))
    x_neg = -x_pos

    plt.figure()
    plt.plot(r, x_pos, 'b-', label="Stable branch")
    plt.plot(r, x_neg, 'r--', label="Unstable branch")
    plt.axvline(0, color="k", linestyle=":", label="Bifurcation point")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$x$")
    plt.title("Saddle-node bifurcation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_sn_branch_small.png", dpi=300)
    plt.close()

# 2. Transcritical bifurcation
def plot_transcritical():
    r = np.linspace(-2, 2, 400)
    plt.figure()
    plt.plot(r, np.zeros_like(r), "r--", label="Unstable branch")
    plt.plot(r, r, "b-", label="Stable branch")
    plt.axvline(0, color="k", linestyle=":", label="Bifurcation point")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$x$")
    plt.title("Transcritical bifurcation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_transcritical_branch_small.png", dpi=300)
    plt.close()

def plot_hopf_amplitude_corrected(
    mu_min: float = -1.0,
    mu_max: float = 1.0,
    n: int = 400,
    savepath: str = "fig_hopf_amplitude.png",
) -> None:
    """
    Corrected Hopf amplitude diagram matching the normal form description.
    """
    mu = np.linspace(mu_min, mu_max, n)
    amp = np.sqrt(np.clip(mu, 0.0, None))
    
    plt.figure(figsize=(8, 6))
    
    # Equilibrium (unstable for μ>0, stable for μ<0)
    mu_neg = mu[mu <= 0]
    mu_pos = mu[mu > 0]
    
    plt.plot(mu_neg, np.zeros_like(mu_neg), 'k-', linewidth=2, label="Equilibrium (stable)")
    plt.plot(mu_pos, np.zeros_like(mu_pos), 'k--', linewidth=2, label="Equilibrium (unstable)")
    
    # Stable limit cycle amplitude (only for μ>0)
    mu_positive = mu[mu > 0]
    amp_positive = amp[mu > 0]
    plt.plot(mu_positive, amp_positive, 'red', linewidth=2, label="Stable limit cycle")
    plt.plot(mu_positive, -amp_positive, 'red', linewidth=2)  # Symmetric branch
    
    # Hopf bifurcation point
    plt.axvline(0.0, color='blue', linestyle=':', linewidth=2, label="Hopf point")
    
    plt.title("Hopf bifurcation (amplitude growth)")
    plt.xlabel(r"$\mu$")
    plt.ylabel("Amplitude")
    plt.xlim(mu_min, mu_max)
    plt.ylim(-1.1, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()

def plot_hopf_phase_corrected(
    mu0: float = 0.1,
    omega: float = 1.0,
    r0: float = 0.05,
    t_end: float = 25.0,
    n: int = 1000,
    savepath: str = "fig_hopf_phase.png",
) -> None:
    """
    Corrected Hopf phase portrait with proper spiral direction and labeling.
    """
    assert mu0 > 0.0, "Choose μ0>0 to show a stable limit cycle."
    r_star = np.sqrt(mu0)
    
    # Stable limit cycle
    theta = np.linspace(0, 2 * np.pi, 800)
    x_lim = r_star * np.cos(theta)
    y_lim = r_star * np.sin(theta)
    
    # Trajectory spiraling outward TO the stable limit cycle
    t = np.linspace(0.0, t_end, n)
    exp_term = np.exp(2 * mu0 * t)
    r_traj = (r0 * np.sqrt(exp_term)) / np.sqrt(1 + (r0**2) * (exp_term - 1))
    phi = omega * t
    x_traj = r_traj * np.cos(phi)
    y_traj = r_traj * np.sin(phi)
    
    plt.figure(figsize=(8, 6))
    
    # Use consistent colors
    plt.plot(x_lim, y_lim, color='blue', linewidth=2, label="Stable limit cycle")
    plt.plot(x_traj, y_traj, color='red', linestyle='--', linewidth=2, alpha=0.7, 
             label="Trajectory to limit cycle")
    plt.plot([0], [0], 'ko', markersize=6, label="Unstable equilibrium")
    
    # Add some inner unstable spirals for completeness
    for r_init in [r_star * 1.2, r_star * 1.5]:
        if r_init > r_star:
            # Trajectory spiraling inward from outside
            r_in = r_star + (r_init - r_star) * np.exp(-2 * mu0 * t)
            x_in = r_in * np.cos(-omega * t)  # Reverse direction
            y_in = r_in * np.sin(-omega * t)
            plt.plot(x_in, y_in, color='orange', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.title(rf"Hopf bifurcation phase portrait ($\mu={mu0}$, $\omega={omega}$)")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.gca().set_aspect("equal", adjustable="box")
    lim = 1.3 * r_star
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()

# 5. Cusp bifurcation parameter set
def plot_cusp_set():
    a = np.linspace(-2, 2, 400)
    b = np.linspace(-2, 2, 400)
    A, B = np.meshgrid(a, b)
    disc = (A/3)**3 + (B/2)**2

    plt.figure()
    plt.contour(A, B, disc, levels=[0], colors="k")
    plt.xlabel(r"$a$")
    plt.ylabel(r"$b$")
    plt.title("Cusp bifurcation set")
    plt.tight_layout()
    plt.savefig("fig_cusp_set_small.png", dpi=300)
    plt.close()

# 6. Cusp bifurcation slice
def plot_cusp_slice(b=0.5):
    a = np.linspace(-2, 2, 400)
    roots = []
    for ai in a:
        coeffs = [1, 0, ai, b]
        r = np.roots(coeffs)
        real_r = r[np.isreal(r)].real
        roots.append(real_r)

    plt.figure()
    for i in range(max(len(r) for r in roots)):
        branch = [r[i] if len(r) > i else np.nan for r in roots]
        plt.plot(a, branch, "b-")
    plt.xlabel(r"$a$")
    plt.ylabel(r"$x$")
    plt.title(f"Cusp slice at b={b}")
    plt.tight_layout()
    plt.savefig("fig_cusp_slice_small.png", dpi=300)
    plt.close()

# 7. Allen–Cahn potential
def plot_allencahn_potential():
    x = np.linspace(-2, 2, 400)
    V = 0.25*x**4 - 0.5*x**2

    plt.figure()
    plt.plot(x, V, "b-")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$V(x)$")
    plt.title("Allen–Cahn potential")
    plt.tight_layout()
    plt.savefig("fig_allencahn_potential_small.png", dpi=300)
    plt.close()

# 8. Allen–Cahn steady profile
def plot_allencahn_profile():
    x = np.linspace(-5, 5, 400)
    profile = np.tanh(x/np.sqrt(2))

    plt.figure()
    plt.plot(x, profile, "b-")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x)$")
    plt.title("Allen–Cahn steady state profile")
    plt.tight_layout()
    plt.savefig("fig_allencahn_profile_small.png", dpi=300)
    plt.close()

# continuation_comparison.py
# Compare natural parameter continuation vs pseudo-arclength on F(x, r) = x^3 - x + r = 0  (=> r = -x^3 + x)

import numpy as np
import matplotlib.pyplot as plt

# --------- S-shaped equilibrium curve ---------
x_curve = np.linspace(-1.5, 1.5, 600)
r_curve = -x_curve**3 + x_curve

# Fold points: dr/dx = -3x^2 + 1 = 0  ->  x = ±1/sqrt(3)
xf = 1 / np.sqrt(3)
fold_x = np.array([-xf, xf])
fold_r = -fold_x**3 + fold_x

# --------- Natural parameter continuation (conceptual demo) ---------
# Step uniformly in r and try to follow the upper branch (largest real root).
def real_roots_cubic(r):
    # Solve x^3 - x + r = 0
    coeffs = [1.0, 0.0, -1.0, r]
    roots = np.roots(coeffs)
    real = roots[np.isclose(roots.imag, 0, atol=1e-10)].real
    return np.sort(real)

r_steps = np.linspace(-0.6, 0.6, 50)  # try to cross the upper fold
x_nat, r_nat = [], []
for r in r_steps:
    roots = real_roots_cubic(r)
    if roots.size == 0:
        break
    x_sel = roots.max()           # try to stick to the upper branch
    x_nat.append(x_sel)
    r_nat.append(r)
    # (Visualization choice) stop near the upper fold to show stalling
    if r > fold_r[1] - 1e-3:
        break

x_nat = np.array(x_nat)
r_nat = np.array(r_nat)

# --------- Pseudo-arclength path (conceptual) ---------
# Sample points roughly uniformly in arclength along the curve (x_curve, r_curve).
pts = np.column_stack([x_curve, r_curve])
ds = np.sqrt(np.diff(pts[:, 0])**2 + np.diff(pts[:, 1])**2)
s = np.concatenate([[0.0], np.cumsum(ds)])
s_samples = np.linspace(s.min(), s.max(), 60)
x_arc = np.interp(s_samples, s, x_curve)
r_arc = np.interp(s_samples, s, r_curve)

# --------- Plot ---------
plt.figure(figsize=(7, 5), dpi=160)

# Full equilibrium curve
plt.plot(r_curve, x_curve, linewidth=1.6, label=r"Equilibrium curve $F(x,r)=0$")

# Fold points
plt.plot(fold_r, fold_x, "o", ms=6, label="Folds")

# Natural parameter continuation path
plt.plot(r_nat, x_nat, "-s", linewidth=1.2, ms=4, label="Natural parameter continuation")

# Pseudo-arclength sampling path
plt.plot(r_arc, x_arc, "--.", linewidth=1.0, ms=3, label="Pseudo-arclength path")

plt.xlabel(r"Parameter $r$")
plt.ylabel(r"State $x$")
plt.title("Continuation across folds: natural parameter vs pseudo-arclength")
plt.legend(loc="best", frameon=False)
plt.grid(True, linewidth=0.4, alpha=0.4)
plt.tight_layout()

# Save figure
out_path = "continuation_comparison.png"   # change this path/name as you like
plt.savefig(out_path, bbox_inches="tight", dpi=300)
print(f"Saved figure to: {out_path}")


# 一次性调用生成所有图
if __name__ == "__main__":
    plot_saddle_node()
    plot_transcritical()
    plot_cusp_set()
    plot_cusp_slice()
    plot_allencahn_potential()
    plot_allencahn_profile()
    plot_hopf_amplitude_corrected()
    plot_hopf_phase_corrected()
