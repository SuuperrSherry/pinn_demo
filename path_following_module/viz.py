# viz.py — minimal, paper-ready
import os
from typing import Callable, Tuple, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import Config

# ---------------- Basic style & IO ----------------
class PlotStyle:
    def __init__(self, config: Config):
        self.config = config
        plt.rcParams.update({
            "figure.dpi": self.config.FIG_DPI,
            "font.size": self.config.FONT_SIZE,
            "axes.labelsize": self.config.FONT_SIZE,
            "axes.titlesize": self.config.FONT_SIZE,
            "legend.fontsize": self.config.FONT_SIZE - 1,
            "lines.linewidth": self.config.LINE_WIDTH,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.autolayout": True,
        })

class DataLoader:
    @staticmethod
    def load_branch_data(csv_path: str) -> pd.DataFrame:
        return pd.read_csv(csv_path)

    @staticmethod
    def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
        for c in candidates:
            if c in df.columns:
                return c
        raise KeyError(f"None of {candidates} in {list(df.columns)}")

# ---------------- Visualizer ----------------
class Visualizer:
    def __init__(self, config: Config, output_dir: str = "assets"):
        self.config = config
        self.output_dir = output_dir
        self.style = PlotStyle(config)

    # ===== Common small plots =====
    def plot_training_curves(self, history: List[Dict], output_path: str, title: str = "Training losses"):
        if not history:
            return
        it = np.arange(1, len(history) + 1)
        loss_names = ["total", "physics", "arc_length", "initial_condition", "smoothness", "direction"]
        plt.figure(figsize=(7.0, 5.0))
        for name in loss_names:
            if name in history[0]:
                vals = [h[name] for h in history]
                plt.semilogy(it, vals, label=name)
        plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.title(title)
        plt.legend(); plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=self.config.FIG_DPI); plt.close()

    def plot_residual_vs_s(self, csv_path: str, output_path: str):
        df = DataLoader.load_branch_data(csv_path)
        s_col = DataLoader.find_column(df, ["s"])
        r_col = DataLoader.find_column(df, ["res", "residual"])
        plt.figure(figsize=(6.6, 5.0))
        plt.semilogy(df[s_col], np.abs(df[r_col]), "-")
        plt.xlabel("s"); plt.ylabel("|F(x,p)|"); plt.title("Physics residual vs s")
        plt.tight_layout(); os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=self.config.FIG_DPI); plt.close()

    def plot_arc_length_error_vs_s(self, csv_path: str, output_path: str):
        df = DataLoader.load_branch_data(csv_path)
        s_col = DataLoader.find_column(df, ["s"])
        if "arc_err" not in df.columns:
            raise KeyError("Column 'arc_err' not found")
        plt.figure(figsize=(6.6, 5.0))
        plt.plot(df[s_col], df["arc_err"], "-")
        plt.xlabel("s"); plt.ylabel("(||y'|| - 1)²"); plt.title("Arc-length consistency vs s")
        plt.tight_layout(); os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=self.config.FIG_DPI); plt.close()

    # ===== Case 1: Saddle-node =====
    def plot_case1_bifurcation_diagram(self, csv_path: str, output_path: str,
                                       theory_fn: Callable, eps_residual: float = 1e-3):
        df = DataLoader.load_branch_data(csv_path)
        p_col = DataLoader.find_column(df, ["p", "p_0", "p0"])
        x_col = DataLoader.find_column(df, ["x", "x_0", "x0"])
        p, x = df[p_col].values, df[x_col].values
        stability = df["stable"].astype(bool).values if "stable" in df.columns else (x < 0)

        # theory
        x_stable_theory, x_unstable_theory = theory_fn(p)
        err_band = np.minimum(np.abs(x - x_stable_theory), np.abs(x - x_unstable_theory))

        # detect bifurcation (x crosses 0 under small residual)
        valid = np.ones_like(x, dtype=bool)
        if "res" in df.columns:
            valid &= (np.abs(df["res"].values) < eps_residual)
        idx = np.where((np.sign(x[1:]) * np.sign(x[:-1]) <= 0) & valid[1:] & valid[:-1])[0]
        if len(idx) > 0:
            i = idx[np.argmin(np.abs(x[idx]))]
            x0, x1, p0, p1 = x[i], x[i+1], p[i], p[i+1]
            bif_p = p0 + (0 - x0) * (p1 - p0) / (x1 - x0 + 1e-12)
            bif_x = 0.0
        else:
            j = np.argmin(np.abs(x))
            bif_p, bif_x = p[j], x[j]

        # plot
        plt.figure(figsize=(6.6, 5.0))
        mask = p <= 0
        plt.plot(p[mask], x_stable_theory[mask], "k-", alpha=0.7, label="theory stable")
        plt.plot(p[mask], x_unstable_theory[mask], "k--", alpha=0.7, label="theory unstable")
        self._plot_stability_segments(p, x, stability)
        plt.fill_between(p, x - err_band, x + err_band, alpha=0.18, label="|error| band")
        plt.scatter([bif_p], [bif_x], s=self.config.MARKER_SIZE, edgecolors='k',
                    facecolors='none', zorder=3, label="bifurcation")
        plt.axvline(0.0, color="0.7", linestyle=":", linewidth=1)
        plt.xlabel("p"); plt.ylabel("x")
        plt.title("Fig.4.1(a) Learned (x,p) vs theory")
        plt.legend(); plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=self.config.FIG_DPI); plt.close()

    def _plot_stability_segments(self, p: np.ndarray, x: np.ndarray, st: np.ndarray):
        start = 0
        while start < len(x):
            end = start + 1
            while end < len(x) and st[end] == st[start]:
                end += 1
            ls = "-" if st[start] else "--"
            lbl = "PINN" if start == 0 else None
            plt.plot(p[start:end], x[start:end], ls, label=lbl)
            start = end

    def generate_case1_metrics_report(self, csv_path: str, theory_fn: Callable, output_path: str,
                                      max_residual_threshold: float = 1e-3,
                                      mae_threshold: float = 1e-2,
                                      arc_error_threshold: float = 5e-3):
        df = DataLoader.load_branch_data(csv_path)
        p_col = DataLoader.find_column(df, ["p", "p_0", "p0"])
        x_col = DataLoader.find_column(df, ["x", "x_0", "x0"])
        p, x = df[p_col].values, df[x_col].values
        res = df["res"].values if "res" in df.columns else np.full_like(p, np.nan)
        arc = df["arc_err"].values if "arc_err" in df.columns else np.full_like(p, np.nan)
        x_s, x_u = theory_fn(p)
        mae = np.mean(np.minimum(np.abs(x - x_s), np.abs(x - x_u)))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"max|F| = {np.nanmax(np.abs(res)):.3e}  (target ≤ {max_residual_threshold:.1e})\n")
            f.write(f"MAE(x_pred, x_true_near) = {mae:.3e}  (target ≤ {mae_threshold:.1e})\n")
            f.write(f"mean arc-length error = {np.nanmean(arc):.3e}  (target ≤ {arc_error_threshold:.1e})\n")

    # ===== Case 2: single paper-ready outlet (overview + zoom) =====
    def plot_case2_master(
        self,
        csv_paths: List[str],
        output_path: str,
        theory_fn: Callable,   # 兼容签名，不实际依赖
        zoom_range: Tuple[float, float, float, float] = (-0.3, 0.3, -0.3, 0.3),
        title: str = "Case 2: Transcritical bifurcation (stable=solid, unstable=dashed)",
    ) -> None:
        # ------- 收集数据 -------
        branches, p_min, p_max = [], None, None
        for path in csv_paths:
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)

            # 兼容列名
            if "p_0" in df.columns: p = df["p_0"].values
            elif "p" in df.columns:  p = df["p"].values
            else: raise KeyError(f"{path} 缺少列 p_0/p")

            if "x_0" in df.columns: x = df["x_0"].values
            elif "x" in df.columns:  x = df["x"].values
            else: raise KeyError(f"{path} 缺少列 x_0/x")

            if "stable" in df.columns:
                st = df["stable"].astype(bool).values
            else:
                st = (p - 2.0 * x) < 0.0  # 跨临界兜底

            branches.append({"p": p, "x": x, "st": st, "label": os.path.basename(path), "df": df})
            p_min = p.min() if p_min is None else min(p_min, p.min())
            p_max = p.max() if p_max is None else max(p_max, p.max())

        if not branches:
            raise FileNotFoundError("No valid CSV for Case2 master plot")

        # 理论曲线
        p_lo = p_min if p_min is not None else -2.0
        p_hi = p_max if p_max is not None else  2.0
        p_theory = np.linspace(p_lo, p_hi, 600)
        mask_neg = p_theory < 0
        mask_pos = ~mask_neg

        def _draw(ax, focus: bool):
            # 理论：x=0
            ax.plot(p_theory[mask_neg], np.zeros_like(p_theory[mask_neg]), "k-",  lw=2.0, alpha=0.8, label="theory x=0 (stable)")
            ax.plot(p_theory[mask_pos], np.zeros_like(p_theory[mask_pos]), "k--", lw=2.0, alpha=0.8, label="theory x=0 (unstable)")
            # 理论：x=p
            ax.plot(p_theory[mask_pos], p_theory[mask_pos], "k-",  lw=2.0, alpha=0.8, label="theory x=p (stable)")
            ax.plot(p_theory[mask_neg], p_theory[mask_neg], "k--", lw=2.0, alpha=0.8, label="theory x=p (unstable)")

            # PINN（按稳定性分段线型）
            palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
            for i, br in enumerate(branches):
                p, x, st = br["p"], br["x"], br["st"]
                c = palette[i % len(palette)]
                start, first_label = 0, f"PINN #{i+1}"
                while start < len(p):
                    end = start + 1
                    while end < len(p) and st[end] == st[start]:
                        end += 1
                    ls = "-" if st[start] else "--"
                    lbl = first_label; first_label = None
                    ax.plot(p[start:end], x[start:end], ls, color=c, lw=2.0, label=lbl)
                    start = end

            # (0,0)
            ax.scatter([0.0], [0.0], s=self.config.MARKER_SIZE, edgecolors="red",
                       facecolors="none", linewidth=2, zorder=5, label="Bifurcation (0,0)")
            ax.axvline(0.0, color="0.7", lw=1.0, alpha=0.6)

            if focus:
                x1, x2, y1, y2 = zoom_range
                ax.set_xlim(x1, x2)
                ax.set_ylim(y1, y2)
            else:
                ax.set_xlim(p_lo, p_hi)
                # 自动 y 范围：用所有分支的 2%~98% 分位，避免离群点
                all_x = np.concatenate([br["x"] for br in branches])
                y_lo, y_hi = np.quantile(all_x, [0.02, 0.98])
                if not np.isfinite(y_lo) or not np.isfinite(y_hi) or (y_hi - y_lo) < 1e-10:
                    y_lo, y_hi = float(np.min(all_x)), float(np.max(all_x))
                pad = 0.08 * max(1e-8, (y_hi - y_lo))
                ax.set_ylim(y_lo - pad, y_hi + pad)

            ax.grid(True, alpha=0.25)
            ax.set_xlabel("p"); ax.set_ylabel("x")

        # ------- 主图（全范围） -------
        plt.figure(figsize=(9.0, 5.8))
        ax = plt.gca()
        _draw(ax, focus=False)
        ax.set_title(title, pad=8)
        ax.legend(loc="best", frameon=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIG_DPI, bbox_inches="tight")
        plt.close()

        # ------- 放大图（zoom_range） -------
        zoom_path = os.path.splitext(output_path)[0] + "_zoom.png"
        plt.figure(figsize=(7.2, 5.4))
        axz = plt.gca()
        _draw(axz, focus=True)
        axz.set_title(title + " — zoomed", pad=8)

        # 放大图里加个指标小卡片（用第一条分支即可）
        df0 = branches[0]["df"]
        p0 = df0["p_0"].values if "p_0" in df0.columns else df0["p"].values
        x0 = df0["x_0"].values if "x_0" in df0.columns else df0["x"].values
        err_near = np.minimum(np.abs(x0), np.abs(x0 - p0))
        win = np.abs(p0) <= 0.2
        if win.any():
            pw, xw = p0[win], x0[win]
            p_at_x0 = float(pw[np.argmin(np.abs(xw))])
            x_at_p0 = float(xw[np.argmin(np.abs(xw - pw))])
        else:
            p_at_x0 = float(p0[np.argmin(np.abs(x0))])
            x_at_p0 = float(x0[np.argmin(np.abs(x0 - p0))])
        txt = f"MAE to nearest: {np.mean(err_near):.2e}\n" \
              f"p* error: {abs(p_at_x0):.2e}\n" \
              f"x* error: {abs(x_at_p0):.2e}"
        if "res" in df0.columns: txt += f"\nmax |F|: {df0['res'].abs().max():.2e}"
        if "arc_err" in df0.columns: txt += f"\nmean arc err: {np.nanmean(df0['arc_err'].values):.2e}"
        axz.text(0.98, 0.02, txt, transform=axz.transAxes, ha="right", va="bottom",
                 fontsize=self.config.FONT_SIZE - 1,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.95))
        axz.legend(loc="best", frameon=True)
        plt.tight_layout()
        plt.savefig(zoom_path, dpi=self.config.FIG_DPI, bbox_inches="tight")
        plt.close()

    # ===== Case 3: Hopf amplitude =====
    def plot_case3_amplitude_curve(self, csv_path: str, output_path: str, theory_fn: Callable):
        df = DataLoader.load_branch_data(csv_path)
        p_col = DataLoader.find_column(df, ["p", "p_0", "p0"])
        x_col = DataLoader.find_column(df, ["x", "x_0", "x0"])
        mu, r = df[p_col].values, df[x_col].values
        r_th = theory_fn(mu)
        err = np.abs(r - r_th)
        plt.figure(figsize=(6.6, 5.0))
        plt.plot(mu, r_th, "k--", alpha=0.7, label="theory r(μ)")
        plt.plot(mu, r, "-", label="PINN")
        plt.fill_between(mu, r - err, r + err, alpha=0.2, label="|error| band")
        plt.xlabel("μ"); plt.ylabel("r"); plt.title("Fig.4.3(a) r(μ) learned vs theory")
        plt.legend(); plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=self.config.FIG_DPI); plt.close()

    def plot_case3_curvature_distribution(self, csv_path: str, output_path: str):
        df = DataLoader.load_branch_data(csv_path)
        s_col = DataLoader.find_column(df, ["s"])
        if "curv" not in df.columns:
            raise KeyError("Column 'curv' not found")
        plt.figure(figsize=(6.6, 5.0))
        plt.plot(df[s_col], df["curv"], "-")
        plt.xlabel("s"); plt.ylabel("||y''||"); plt.title("Fig.4.3(b) Curvature distribution")
        plt.tight_layout(); os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=self.config.FIG_DPI); plt.close()

    # ===== Compatibility aliases =====
    def plot_losses(self, history: List[Dict], output_path: str, title: str = "Training losses"):
        self.plot_training_curves(history, output_path, title)

    def fig_case1_px_publication(self, csv_path: str, output_path: str, theory_fn: Callable, eps_r: float = 1e-3):
        self.plot_case1_bifurcation_diagram(csv_path, output_path, theory_fn, eps_r)

    def report_case1_metrics(self, csv_path: str, theory_fn: Callable, output_path: str,
                             max_res_ok: float = 1e-3, mae_ok: float = 1e-2, mean_arc_ok: float = 5e-3):
        self.generate_case1_metrics_report(csv_path, theory_fn, output_path, max_res_ok, mae_ok, mean_arc_ok)

    def fig_case3_r_mu(self, csv_path: str, output_path: str, theory_fn: Callable):
        self.plot_case3_amplitude_curve(csv_path, output_path, theory_fn)

    def fig_case3_curvature(self, csv_path: str, output_path: str):
        self.plot_case3_curvature_distribution(csv_path, output_path)

    def plot_arcerr_vs_s(self, csv_path: str, output_path: str):
        self.plot_arc_length_error_vs_s(csv_path, output_path)
