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

# 3. Hopf bifurcation amplitude
def plot_hopf_amplitude():
    mu = np.linspace(-1, 1, 400)
    amp = np.sqrt(np.clip(mu, 0, None))

    plt.figure()
    plt.plot(mu, np.zeros_like(mu), "r--", label="Equilibrium (unstable)")
    plt.plot(mu, amp, "b-", label="Stable limit cycle")
    plt.plot(mu, -amp, "b-")
    plt.axvline(0, color="k", linestyle=":", label="Hopf point")
    plt.xlabel(r"$\mu$")
    plt.ylabel("Amplitude")
    plt.title("Hopf bifurcation (amplitude growth)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_hopf_amplitude_small.png", dpi=300)
    plt.close()

# 4. Hopf phase portrait
def plot_hopf_phase():
    theta = np.linspace(0, 2*np.pi, 200)
    circle = np.exp(1j*theta)
    spiral = (1 - np.exp(-0.5*theta)) * np.exp(1j*theta)

    plt.figure()
    plt.plot(circle.real, circle.imag, "b-", label="Stable limit cycle")
    plt.plot(spiral.real, spiral.imag, "r--", label="Unstable spiral")
    plt.plot(0, 0, "ko", label="Equilibrium")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title("Hopf bifurcation phase portrait")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("fig_hopf_phase_small.png", dpi=300)
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

# 一次性调用生成所有图
if __name__ == "__main__":
    plot_saddle_node()
    plot_transcritical()
    plot_hopf_amplitude()
    plot_hopf_phase()
    plot_cusp_set()
    plot_cusp_slice()
    plot_allencahn_potential()
    plot_allencahn_profile()
