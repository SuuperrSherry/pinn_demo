# path: figures_ch2_corrected.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# -------------------- Global style (keep your sizes/colors) --------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.unicode_minus": False,
    "lines.linewidth": 2.0,
})

# -------------------- Small helpers --------------------
def draw_branch(ax, x, y, stable: bool, label: str | None = None):
    """稳定=蓝实线；不稳定=红虚线（保持你的配色语义）。"""
    if stable:
        ax.plot(x, y, "b-", label=label)
    else:
        ax.plot(x, y, "r--", label=label)

def _real_roots_cubic(a3: float, a2: float, a1: float, a0: float) -> np.ndarray:
    roots = np.roots([a3, a2, a1, a0])
    real = roots[np.isclose(roots.imag, 0.0, atol=1e-10)].real
    return np.sort(real)

# -------------------- Fig. 2.1 Saddle-node --------------------
def plot_saddle_node(clear: bool = True, savepath: str = "fig_sn_branch_small.png"):
    """
    x' = r + x^2
    仅 r<=0 存在平衡：x = ±sqrt(-r)。
    视觉消歧：r>0 区域明确标注“无平衡解”，在 (0,0) 标注非双曲折点。
    """
    rmin, rmax = -1.0, 1.0
    r = np.linspace(rmin, rmax, 1000)
    mask = r <= 0.0
    r_leq0 = r[mask]
    x_pos =  np.sqrt(-r_leq0)   # 不稳定
    x_neg = -np.sqrt(-r_leq0)   # 稳定

    fig, ax = plt.subplots()

    # 稳定=蓝实线；不稳定=红虚线（保持你的配色语义）
    ax.plot(r_leq0, x_neg, "b-",  label="Stable branch")
    ax.plot(r_leq0, x_pos, "r--", label="Unstable branch")

    # 右侧无解区域：淡填充 + 斜线孵化，强提示“没有分支”
    ax.axvspan(0.0, rmax, alpha=0.08, hatch="//", edgecolor="none")
    # 标注无解
    yspan = float(max(abs(x_neg.min()), abs(x_pos.max())))

    # 分岔点(0,0)：非双曲折点（避免读者以为是普通交点）
    ax.plot(0.0, 0.0, "ko", ms=4, label="Fold (nonhyperbolic)")
    ax.axvline(0.0, color="k", linestyle=":", linewidth=1.2)

    # 轴与标题
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$x$")
    ax.set_title("Saddle-node")

    # 轴范围适度留白，避免“延伸错觉”
    ax.set_xlim(rmin, rmax)
    ax.set_ylim(-1.1*yspan, 1.1*yspan)

    # 图例：只显示一次
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")

    fig.tight_layout()
    fig.savefig(savepath, dpi=300)
    plt.close(fig)
# -------------------- Fig. 2.2 Transcritical --------------------
def plot_transcritical(savepath: str = "fig_transcritical_branch_small.png"):
    # Model: x' = r x - x^2 = x(r-x)
    r = np.linspace(-2.0, 2.0, 800)
    r_neg = r[r < 0]
    r_pos = r[r > 0]
    r_zle = r[r <= 0]
    r_gze = r[r >= 0]

    plt.figure()
    # x = 0 branch: stable for r<0; unstable for r>=0
    if r_neg.size:
        draw_branch(plt.gca(), r_neg, np.zeros_like(r_neg), True,  "x=0 (stable for r<0)")
    if r_gze.size:
        draw_branch(plt.gca(), r_gze, np.zeros_like(r_gze), False, "x=0 (unstable for r≥0)")
    # x = r branch: unstable for r<=0; stable for r>0
    if r_zle.size:
        draw_branch(plt.gca(), r_zle, r_zle, False, "x=r (unstable for r≤0)")
    if r_pos.size:
        draw_branch(plt.gca(), r_pos, r_pos, True,  "x=r (stable for r>0)")

    plt.axvline(0, color="k", linestyle=":", label="Bifurcation point")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$x$")
    plt.title("Transcritical")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()

# -------------------- Fig. 2.5 Cusp set (parameter plane) --------------------
def plot_cusp_set(savepath: str = "fig_cusp_set_small.png"):
    """
    规范形：0 = x^3 + r1 x + r2
    判别式 Δ = -4 r1^3 - 27 r2^2 = 0
    等价曲线： (r1/3)^3 + (r2/2)^2 = 0  （仅尺度不同）
    """
    r1 = np.linspace(-2.0, 2.0, 600)
    r2 = np.linspace(-2.0, 2.0, 600)
    R1, R2 = np.meshgrid(r1, r2)
    disc = (R1/3.0)**3 + (R2/2.0)**2

    plt.figure()
    plt.contour(R1, R2, disc, levels=[0.0], colors="k")
    plt.xlabel(r"$r_1$")
    plt.ylabel(r"$r_2$")
    plt.title("Cusp set (in $(r_1,r_2)$)")
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()

# -------------------- Fig. 2.6 Cusp slice (stability shown) --------------------
def plot_cusp_slice(r2: float = 0.5, savepath: str = "fig_cusp_slice_small.png"):
    """
    切片：0 = x^3 + r1 x + r2 （固定 r2）
    稳定性来自 1D ODE：x' = - (x^3 + r1 x + r2)
      f'(x*) = - (3 x*^2 + r1)
      稳定 <=> 3 x*^2 + r1 > 0
    """
    r1 = np.linspace(-2.0, 2.0, 900)
    branches: List[Dict[str, list]] = []
    prev_roots: np.ndarray | None = None
    thr = 0.25  # 连续性阈值（可微调）

    for r1i in r1:
        roots = _real_roots_cubic(1.0, 0.0, r1i, r2)        # x^3 + r1 x + r2 = 0
        st = [(3.0*xi*xi + r1i) > 0.0 for xi in roots]      # True=stable
        if prev_roots is None:
            for xi, si in zip(roots, st):
                branches.append({"r1":[r1i], "x":[xi], "st":[si]})
        else:
            used = [False]*len(roots)
            # 续接已有分支
            for br in branches:
                last_x = br["x"][-1]
                if roots.size:
                    idx = int(np.argmin([abs(xi-last_x) if not used[j] else 1e9 for j, xi in enumerate(roots)]))
                    if not used[idx] and abs(roots[idx]-last_x) < thr:
                        br["r1"].append(r1i); br["x"].append(roots[idx]); br["st"].append(st[idx]); used[idx] = True
            # 折点处可能出现新分支
            for j, xi in enumerate(roots):
                if not used[j]:
                    branches.append({"r1":[r1i], "x":[xi], "st":[st[j]]})
        prev_roots = roots

    plt.figure()
    ax = plt.gca()
    # 分段按稳定性绘制
    for br in branches:
        r1_arr = np.asarray(br["r1"]); x_arr = np.asarray(br["x"]); st_arr = np.asarray(br["st"], dtype=bool)
        if r1_arr.size < 2:
            continue
        change = np.where(st_arr[1:] != st_arr[:-1])[0] + 1
        cuts = np.r_[0, change, len(r1_arr)]
        for s, e in zip(cuts[:-1], cuts[1:]):
            if e - s < 2: 
                continue
            draw_branch(ax, r1_arr[s:e], x_arr[s:e], bool(st_arr[s]))

    # 图例（示意稳定/不稳定）
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color='b', linestyle='-',  label="Stable"),
        Line2D([0],[0], color='r', linestyle='--', label="Unstable"),
    ]
    ax.legend(handles=handles, loc="best")

    plt.xlabel(r"$r_1$")
    plt.ylabel(r"$x$")
    plt.title(f"Cusp slice at $r_2={r2}$")
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()

# -------------------- Hopf (kept, style-aligned) --------------------
def plot_hopf_amplitude_corrected(
    mu_min: float = -1.0,
    mu_max: float = 1.0,
    n: int = 400,
    savepath: str = "fig_hopf_amplitude.png",
) -> None:
    mu = np.linspace(mu_min, mu_max, n)
    amp = np.sqrt(np.clip(mu, 0.0, None))
    plt.figure(figsize=(8, 6))
    mu_neg = mu[mu <= 0]; mu_pos = mu[mu > 0]
    plt.plot(mu_neg, np.zeros_like(mu_neg), 'k-',  linewidth=2, label="Equilibrium (stable)")
    plt.plot(mu_pos, np.zeros_like(mu_pos), 'k--', linewidth=2, label="Equilibrium (unstable)")
    mu_positive = mu[mu > 0]; amp_positive = amp[mu > 0]
    plt.plot(mu_positive,  amp_positive, 'blue', linewidth=2, label="Stable limit cycle")
    plt.plot(mu_positive, -amp_positive, 'blue', linewidth=2)
    plt.axvline(0.0, color='k', linestyle=':', linewidth=1.5, label="Hopf point")
    plt.title("Hopf bifurcation (amplitude growth)")
    plt.xlabel(r"$\mu$"); plt.ylabel("Amplitude")
    plt.xlim(mu_min, mu_max); plt.ylim(-1.1, 1.1)
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(savepath, dpi=300); plt.close()

def plot_hopf_phase_corrected(
    mu0: float = 0.1,
    omega: float = 1.0,
    r0: float = 0.05,
    t_end: float = 25.0,
    n: int = 1000,
    savepath: str = "fig_hopf_phase.png",
) -> None:
    assert mu0 > 0.0, "Choose μ0>0 to show a stable limit cycle."
    r_star = np.sqrt(mu0)
    theta = np.linspace(0, 2*np.pi, 800)
    x_lim = r_star*np.cos(theta); y_lim = r_star*np.sin(theta)
    t = np.linspace(0.0, t_end, n)
    exp_term = np.exp(2*mu0*t)
    r_traj = (r0*np.sqrt(exp_term))/np.sqrt(1+(r0**2)*(exp_term-1))
    phi = omega*t
    x_traj = r_traj*np.cos(phi); y_traj = r_traj*np.sin(phi)

    plt.figure(figsize=(8, 6))
    plt.plot(x_lim, y_lim, color='blue', linewidth=2, label="Stable limit cycle")
    plt.plot(x_traj, y_traj, color='red', linestyle='--', linewidth=2, alpha=0.8, label="Trajectory to limit cycle")
    plt.plot([0],[0], 'ko', ms=5, label="Unstable equilibrium")
    plt.gca().set_aspect("equal", adjustable="box")
    lim = 1.3*r_star; plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    plt.title(rf"Hopf bifurcation phase portrait ($\mu={mu0}$, $\omega={omega}$)")
    plt.xlabel(r"$x$"); plt.ylabel(r"$y$")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(savepath, dpi=300); plt.close()

# -------------------- Allen–Cahn (unchanged) --------------------
def plot_allencahn_potential(savepath: str = "fig_allencahn_potential_small.png"):
    x = np.linspace(-2, 2, 400)
    V = 0.25*x**4 - 0.5*x**2
    plt.figure(); plt.plot(x, V, "b-")
    plt.xlabel(r"$x$"); plt.ylabel(r"$V(x)$"); plt.title("Allen–Cahn potential")
    plt.tight_layout(); plt.savefig(savepath, dpi=300); plt.close()

def plot_allencahn_profile(savepath: str = "fig_allencahn_profile_small.png"):
    x = np.linspace(-5, 5, 400)
    profile = np.tanh(x/np.sqrt(2))
    plt.figure(); plt.plot(x, profile, "b-")
    plt.xlabel(r"$x$"); plt.ylabel(r"$u(x)$"); plt.title("Allen–Cahn steady state profile")
    plt.tight_layout(); plt.savefig(savepath, dpi=300); plt.close()

# -------------------- Continuation comparison (kept) --------------------
def continuation_comparison(savepath: str = "continuation_comparison.png"):
    # F(x,r) = x^3 - x + r = 0  -> r = -x^3 + x
    x_curve = np.linspace(-1.5, 1.5, 600)
    r_curve = -x_curve**3 + x_curve
    xf = 1/np.sqrt(3); fold_x = np.array([-xf, xf]); fold_r = -fold_x**3 + fold_x

    def real_roots_cubic(r):
        coeffs = [1.0, 0.0, -1.0, r]
        roots = np.roots(coeffs)
        real = roots[np.isclose(roots.imag, 0, atol=1e-10)].real
        return np.sort(real)

    r_steps = np.linspace(-0.6, 0.6, 50)
    x_nat, r_nat = [], []
    for r in r_steps:
        roots = real_roots_cubic(r)
        if roots.size == 0: break
        x_nat.append(roots.max()); r_nat.append(r)
        if r > fold_r[1] - 1e-3: break

    x_nat = np.array(x_nat); r_nat = np.array(r_nat)
    pts = np.column_stack([x_curve, r_curve])
    ds = np.sqrt(np.diff(pts[:,0])**2 + np.diff(pts[:,1])**2)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s_samples = np.linspace(s.min(), s.max(), 60)
    x_arc = np.interp(s_samples, s, x_curve)
    r_arc = np.interp(s_samples, s, r_curve)

    plt.figure(figsize=(7,5), dpi=160)
    plt.plot(r_curve, x_curve, linewidth=1.6, label=r"Equilibrium curve $F(x,r)=0$")
    plt.plot(fold_r, fold_x, "o", ms=6, label="Folds")
    plt.plot(r_nat, x_nat, "-s", linewidth=1.2, ms=4, label="Natural parameter continuation")
    plt.plot(r_arc, x_arc, "--.", linewidth=1.0, ms=3, label="Pseudo-arclength path")
    plt.xlabel(r"Parameter $r$"); plt.ylabel(r"State $x$")
    plt.title("Continuation across folds: natural parameter vs pseudo-arclength")
    plt.legend(loc="best", frameon=False)
    plt.grid(True, linewidth=0.4, alpha=0.4)
    plt.tight_layout(); plt.savefig(savepath, bbox_inches="tight", dpi=300); plt.close()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(8, 6))
ax.axis("off")

# Boxes
box_style = dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black", lw=1.2)
loss_style = dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black", lw=1.2)

ax.text(0.5, 0.9, "Arc-length parameter s", ha="center", va="center", bbox=box_style, fontsize=11)
ax.text(0.5, 0.75, "PINN", ha="center", va="center", bbox=box_style, fontsize=12, fontweight="bold")

# Loss terms
losses = ["Physics Residual", "Arc-length Constraint", "Initial Condition",
          "Smoothness", "Directional Consistency"]
y_positions = [0.6, 0.5, 0.4, 0.3, 0.2]
for loss, y in zip(losses, y_positions):
    ax.text(0.2, y, loss, ha="center", va="center", bbox=loss_style, fontsize=10)
    ax.annotate("", xy=(0.5, 0.72), xytext=(0.28, y+0.02), arrowprops=dict(arrowstyle="->"))

# Total Loss
ax.text(0.8, 0.45, "Total Loss", ha="center", va="center", bbox=box_style, fontsize=11)
ax.annotate("", xy=(0.72, 0.45), xytext=(0.5, 0.72), arrowprops=dict(arrowstyle="->"))

# Optimizer
ax.text(0.8, 0.3, "Optimizer", ha="center", va="center", bbox=box_style, fontsize=11)
ax.annotate("", xy=(0.8, 0.38), xytext=(0.8, 0.42), arrowprops=dict(arrowstyle="->"))

# Output
ax.text(0.5, 0.1, "Equilibrium solutions (x(s), p(s))", ha="center", va="center", bbox=box_style, fontsize=11)
ax.annotate("", xy=(0.5, 0.15), xytext=(0.5, 0.7), arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
plt.savefig("framework_pinn.pdf")  # 推荐 pdf 给 LaTeX 用
plt.show()


# -------------------- Batch --------------------
if __name__ == "__main__":
    plot_saddle_node()
    plot_transcritical()
    plot_cusp_set()
    plot_cusp_slice(r2=0.5)
    plot_allencahn_potential()
    plot_allencahn_profile()
    plot_hopf_amplitude_corrected()
    plot_hopf_phase_corrected()
    continuation_comparison()
