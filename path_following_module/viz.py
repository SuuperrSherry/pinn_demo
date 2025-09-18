# viz.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import Config

def _style(cfg: Config):
    plt.rcParams.update({
        "figure.dpi": cfg.FIG_DPI, "font.size": cfg.FONT_SIZE,
        "axes.labelsize": cfg.FONT_SIZE, "axes.titlesize": cfg.FONT_SIZE,
        "legend.fontsize": cfg.FONT_SIZE-1, "lines.linewidth": cfg.LINE_WIDTH
    })

def _pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found. Available: {list(df.columns)}")

class Visualizer:
    def __init__(self, cfg: Config, out_dir: str = "assets"):
        self.cfg = cfg; self.out_dir = out_dir; _style(cfg)

    # ---------- 通用 ----------
    def plot_losses(self, history, out_path: str, title="Training losses"):
        it = np.arange(1, len(history)+1)
        keys = ["total","phys","arc","ic","smooth","dir"]
        plt.figure(figsize=(7.0,5.0))
        for k in keys:
            if len(history) > 0 and (k in history[0]):
                plt.semilogy(it, [h[k] for h in history], label=k)
        plt.xlabel("iteration"); plt.ylabel("loss"); plt.title(title)
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(out_path); plt.close()

    def plot_residual_vs_s(self, csv_path: str, out_path: str):
        df = pd.read_csv(csv_path)
        s_col = _pick_col(df, ["s"])
        r_col = _pick_col(df, ["res"])
        plt.figure(figsize=(6.6,5.0))
        plt.semilogy(df[s_col].values, np.abs(df[r_col].values), "-")
        plt.xlabel("s"); plt.ylabel("|F(x,p)|"); plt.title("Physics residual vs s")
        plt.tight_layout(); plt.savefig(out_path); plt.close()

    def plot_arcerr_vs_s(self, csv_path: str, out_path: str):
        df = pd.read_csv(csv_path)
        s_col = _pick_col(df, ["s"])
        if "arc_err" not in df.columns:
            raise KeyError("Column 'arc_err' not in CSV.")
        plt.figure(figsize=(6.6,5.0))
        plt.plot(df[s_col].values, df["arc_err"].values, "-")
        plt.xlabel("s"); plt.ylabel("((||y'||-1))^2"); plt.title("Arc-length consistency vs s")
        plt.tight_layout(); plt.savefig(out_path); plt.close()

    # ---------- Case 1：论文版 p-x 图 ----------
    def fig_case1_px_publication(self, csv_path: str, out_path: str, theory_fn, eps_r: float = 1e-3):
        df = pd.read_csv(csv_path)
        p_col = _pick_col(df, ["p","p1","p_0","p0"])
        x_col = _pick_col(df, ["x","x1","x_0","x0"])
        stab_col = "stable" if "stable" in df.columns else None

        p = df[p_col].values
        x = df[x_col].values
        stable_pred = df[stab_col].astype(bool).values if stab_col else (x < 0)

        # 理论两支 + 最近误差
        x_stable, x_unstable = theory_fn(p)
        err_near = np.minimum(np.abs(x - x_stable), np.abs(x - x_unstable))

        # —— 折点定位（更稳）——
        # 1) 限制在 |F|<eps_r（若有）再找 x 的零点
        mask = np.ones_like(x, dtype=bool)
        if "res" in df.columns:
            mask &= (np.abs(df["res"].values) < eps_r)

        idxs = np.where((np.sign(x[1:]) * np.sign(x[:-1]) <= 0) & mask[1:] & mask[:-1])[0]
        if len(idxs) > 0:
            # 线性插值估计 x=0 时的 p*
            i = idxs[np.argmin(np.abs(x[idxs]))]  # 任选最靠近零的一对
            x0, x1 = x[i], x[i+1]
            p0, p1 = p[i], p[i+1]
            if abs(x1 - x0) > 1e-12:
                p_star = p0 + (0 - x0) * (p1 - p0) / (x1 - x0)
            else:
                p_star = p0
            x_star = 0.0
        else:
            # 2) 退回：在 |F|<eps_r (若有) 的集合里选 σ_min 最小 / 或 |x| 最小
            if "sigma_min" in df.columns:
                cand = np.where(mask)[0]
                j = cand[np.argmin(df["sigma_min"].values[cand])] if cand.size > 0 else np.argmin(df["sigma_min"].values)
            else:
                cand = np.where(mask)[0]
                j = cand[np.argmin(np.abs(x[cand]))] if cand.size > 0 else np.argmin(np.abs(x))
            p_star, x_star = p[j], x[j]

        # —— 绘图 —— 
        plt.figure(figsize=(6.6,5.0))
        mask_th = p <= 0
        plt.plot(p[mask_th], x_stable[mask_th], "k-", alpha=0.7, label="theory stable")
        plt.plot(p[mask_th], x_unstable[mask_th], "k--", alpha=0.7, label="theory unstable")

        s = 0
        for e in range(1, len(x)+1):
            if e == len(x) or stable_pred[e] != stable_pred[e-1]:
                ls = "-" if stable_pred[e-1] else "--"
                plt.plot(p[s:e], x[s:e], ls, color="#1f77b4", label="PINN" if s == 0 else None)
                s = e
    
        plt.fill_between(p, x-err_near, x+err_near, color="#1f77b4", alpha=0.18, label="|error| band")

        # 折点圆圈（x_star=0 时会正好落在 (0,0)）
        plt.scatter([p_star], [x_star], s=self.cfg.MARKER_SIZE, edgecolors='k',
                    facecolors='none', zorder=3, label="bifurcation")
        plt.axvline(0.0, color="0.7", lw=1, ls=":")

        plt.xlabel("p"); plt.ylabel("x")
        plt.title("Fig.4.1(a) Learned (x,p) vs theory (stable solid / unstable dashed)")
        plt.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=self.cfg.FIG_DPI); plt.close()

    def report_case1_metrics(self, csv_path: str, theory_fn, out_txt: str,
                             max_res_ok=1e-3, mae_ok=1e-2, mean_arc_ok=5e-3):
        """把 Case1 的三项指标写到 txt：max|F|、MAE（最近理论支）、平均弧长偏差。"""
        df = pd.read_csv(csv_path)
        p_col = _pick_col(df, ["p","p1","p_0","p0"])
        x_col = _pick_col(df, ["x","x1","x_0","x0"])
        p = df[p_col].values
        x = df[x_col].values
        res = df["res"].values if "res" in df.columns else np.nan*np.ones_like(p)
        arc = df["arc_err"].values if "arc_err" in df.columns else np.nan*np.ones_like(p)

        x_stable, x_unstable = theory_fn(p)
        mae = np.mean(np.minimum(np.abs(x - x_stable), np.abs(x - x_unstable)))
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(f"max|F| = {np.nanmax(np.abs(res)):.3e}  (target ≤ {max_res_ok:.1e})\n")
            f.write(f"MAE(x_pred, x_true_near) = {mae:.3e}  (target ≤ {mae_ok:.1e})\n")
            f.write(f"mean arc-length error = {np.nanmean(arc):.3e}  (target ≤ {mean_arc_ok:.1e})\n")

    # ---------- Case 2 ----------
    def fig_case2_tangent_cos(self, csv_path: str, out_path: str):
        df = pd.read_csv(csv_path)
        s_col = _pick_col(df, ["s"])
        if "tangent_cos" not in df.columns:
            raise KeyError("Column 'tangent_cos' not in CSV.")
        plt.figure(figsize=(6.6,5.0))
        plt.plot(df[s_col].values, df["tangent_cos"].values, "-")
        plt.xlabel("s"); plt.ylabel("cos(t_i, t_{i-1})"); plt.title("Fig.4.2(b) Tangent consistency vs s")
        plt.tight_layout(); plt.savefig(out_path); plt.close()

    def fig_case2_px_with_two_branches(self, csv_path: str, out_path: str, theory_fn,
                                       inset_range=(-0.2,0.2,-0.2,0.2)):
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
        df = pd.read_csv(csv_path)
        p_col = _pick_col(df, ["p","p1","p_0","p0"])
        x_col = _pick_col(df, ["x","x1","x_0","x0"])
        stab_col = _pick_col(df, ["stable"])

        p = df[p_col].values
        x = df[x_col].values
        stab = df[stab_col].astype(bool).values
        x0, xr = theory_fn(p)

        plt.figure(figsize=(6.8,5.0))
        s = 0
        for e in range(1, len(x)+1):
            if e == len(x) or stab[e] != stab[e-1]:
                ls = "-" if stab[e-1] else "--"
                plt.plot(p[s:e], x[s:e], ls, color="#1f77b4")
                s = e
        plt.plot(p, x0, "k-", alpha=0.7, label="x=0")
        plt.plot(p, xr, "k--", alpha=0.7, label="x=r")

        ax = plt.gca()
        axins = inset_axes(ax, width="50%", height="50%", loc="upper left")
        axins.plot(p, x, "-", color="#1f77b4")
        axins.plot(p, x0, "k-"); axins.plot(p, xr, "k--")
        x1, x2, y1, y2 = inset_range
        axins.set_xlim(x1, x2); axins.set_ylim(y1, y2)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        plt.xlabel("r"); plt.ylabel("x"); plt.title("Fig.4.2(a) Two branches with stability / inset")
        plt.legend(); plt.tight_layout(); plt.savefig(out_path); plt.close()

    # ---------- Case 3 ----------
    def fig_case3_r_mu(self, csv_path: str, out_path: str, theory_fn):
        df = pd.read_csv(csv_path)
        p_col = _pick_col(df, ["p","p1","p_0","p0"])
        x_col = _pick_col(df, ["x","x1","x_0","x0"])
        mu = df[p_col].values
        r  = df[x_col].values
        r_true = theory_fn(mu)
        err = np.abs(r - r_true)
        plt.figure(figsize=(6.6,5.0))
        plt.plot(mu, r_true, "k--", alpha=0.7, label="theory r(μ)")
        plt.plot(mu, r, "-", label="PINN")
        plt.fill_between(mu, r-err, r+err, alpha=0.2, label="|error| band")
        plt.xlabel("μ"); plt.ylabel("r"); plt.title("Fig.4.3(a) r(μ) learned vs theory")
        plt.legend(); plt.tight_layout(); plt.savefig(out_path); plt.close()

    def fig_case3_curvature(self, csv_path: str, out_path: str):
        df = pd.read_csv(csv_path)
        s_col = _pick_col(df, ["s"])
        if "curv" not in df.columns:
            raise KeyError("Column 'curv' not in CSV.")
        plt.figure(figsize=(6.6,5.0))
        plt.plot(df[s_col].values, df["curv"].values, "-")
        plt.xlabel("s"); plt.ylabel("||y''||"); plt.title("Fig.4.3(b) Curvature distribution")
        plt.tight_layout(); plt.savefig(out_path); plt.close()
