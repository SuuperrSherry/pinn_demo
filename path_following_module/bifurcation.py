# bifurcation.py  — drop-in replacement
# ------------------------------------------------------------
# - General exporter for branch CSV (used by trainer)
# - Lightweight bifurcation detector
# - Minimal Case2 metrics (works with/without theory_fn)
# - Backward-compatible helpers used by older main/viz
# ------------------------------------------------------------

from __future__ import annotations
import os
import json
from typing import Dict, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import torch


# =========================
# Utils
# =========================
def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.clamp(torch.sum(x * x, dim=dim), min=eps))


def _infer_px_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """找列名（对 x/p 的多种命名做兼容）"""
    def find(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None
    p_col = find(["p_0", "p", "p1", "p0"])
    x0_col = find(["x_0", "x", "x1", "x0"])
    if p_col is None or x0_col is None:
        raise KeyError(f"CSV缺少 x/p 列，已找到列: {list(df.columns)}")
    return x0_col, p_col


# =========================
# Detector (lightweight)
# =========================
class SaddleNodeDetector:
    """
    轻量级分岔检测器：
    - 在导出 CSV 时给出“疑似分岔邻域”标记。
    - 阈值可配置，尽量通用。
    """

    def __init__(
        self,
        residual_threshold: float = 1e-3,
        sigma_threshold: float = 1e-2,
        tau_threshold: float = 1e-2,
        debounce_window: int = 5,
    ):
        self.residual_threshold = float(residual_threshold)
        self.sigma_threshold = float(sigma_threshold)
        self.tau_threshold = float(tau_threshold)
        self.debounce_window = int(debounce_window)

    def detect_flags(
        self,
        p: np.ndarray,
        x: np.ndarray,
        residual: np.ndarray,
        sigma_min: Optional[np.ndarray] = None,
        tau: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        返回 0/1 标记：1 表示疑似分岔邻域。
        逻辑（尽量一般化）：
        - 残差较小（在“解流形附近”）
        - 若有 sigma_min：其较小（奇异、条件数高）
        - 若有 tau：其接近 0
        - 另外在 |p| 的小窗内更容易命中（对 transcritical 友好）
        """
        res_ok = np.abs(residual) < max(self.residual_threshold, 1e-12)

        cond_list = [res_ok]
        if sigma_min is not None:
            cond_list.append((sigma_min >= 0) & (sigma_min < max(self.sigma_threshold, 1e-12)))
        if tau is not None:
            cond_list.append(np.abs(tau) < max(self.tau_threshold, 1e-12))

        # 软偏置到 p≈0（适合 Case2；其它 case 也只是弱偏置，不会强制）
        p_bias = np.abs(p) < 0.4
        cond_list.append(p_bias)

        mask = np.logical_and.reduce(cond_list).astype(np.uint8)

        # 简单去抖（可选：scipy 不在则跳过）
        if self.debounce_window > 1 and mask.any():
            try:
                from scipy.ndimage import binary_dilation  # optional
                mask = binary_dilation(mask.astype(bool), iterations=max(1, self.debounce_window // 2)).astype(np.uint8)
            except Exception:
                pass

        return mask


# =========================
# Exporter (trainer 使用)
# =========================
class BifurcationExporter:
    """
    把 (s -> (x,p)) 曲线及若干派生量导出为 CSV，供可视化与评估。
    尽量使用“可用即导”的策略：有则算；没有就 NaN。
    """

    @staticmethod
    def _compute_derivatives(model, s_eval: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 y' 和 y''，其中 y = concat[x, p]。
        注意：需要打开 autograd。
        """
        s = s_eval.clone().detach().requires_grad_(True)
        x, p = model(s)
        y = torch.cat([x, p], dim=1)  # [N, nx+np]

        # 一阶
        grads = []
        for i in range(y.shape[1]):
            gi = torch.autograd.grad(y[:, i].sum(), s, create_graph=True, retain_graph=True)[0]
            grads.append(gi)
        dy_ds = torch.cat(grads, dim=1)  # [N, nx+np]

        # 二阶
        d2 = []
        for i in range(dy_ds.shape[1]):
            gi = torch.autograd.grad(dy_ds[:, i].sum(), s, create_graph=True, retain_graph=True)[0]
            d2.append(gi)
        d2y_ds2 = torch.cat(d2, dim=1)  # [N, nx+np]
        return dy_ds, d2y_ds2

    @staticmethod
    def _maybe_sigma_min(physics_fn, x: torch.Tensor, p: torch.Tensor) -> Optional[torch.Tensor]:
        """
        可选：计算 dF/dx 的最小奇异值（如果 physics.compute_jacobian_x 可用）。
        """
        try:
            from physics import compute_jacobian_x  # 可选依赖
            J = compute_jacobian_x(physics_fn, x.detach().requires_grad_(True), p.detach().requires_grad_(True))
            sv = torch.linalg.svdvals(J)  # [N, nx]
            sigma_min = sv[..., -1]
            return sigma_min
        except Exception:
            return None

    @staticmethod
    def export_to_csv(
        physics_fn,
        model,
        s_eval: torch.Tensor,
        output_path: str,
        detector: Optional[SaddleNodeDetector] = None,
    ) -> None:
        """
        导出列（尽量通用）：
        - s, x_0, x_1...(可变), p_0, p_1...(可变)
        - res（||F||）
        - sigma_min（可选）
        - tau（保留字段：目前置为 NaN）
        - arc_err = (||y'|| - 1)^2
        - curv = ||y''||
        - tangent_cos（相邻切向量夹角余弦，第一点=1）
        - stable（不强制，若需可在后处理里补）
        - bif_flag（由 detector 给出，0/1）
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # ---- BEGIN 修改：纯前向数值可 no_grad，加速 ----
        with torch.no_grad():
            x_val, p_val = model(s_eval)  # 用于导出的数值（不需要梯度）
            F = physics_fn(x_val, p_val)
            if not torch.is_tensor(F):
                F = torch.as_tensor(F)
            res = _safe_norm(F, dim=1)
        # ---- END 修改 ----

        # ---- BEGIN 修改：所有“需要梯度的量”显式开启 enable_grad ----
        with torch.enable_grad():
            dy_ds, d2y_ds2 = BifurcationExporter._compute_derivatives(model, s_eval)
            sigma_min = BifurcationExporter._maybe_sigma_min(physics_fn, x_val, p_val)
        # ---- END 修改 ----

        # 几何量
        yprime = dy_ds
        y2 = d2y_ds2
        speed = _safe_norm(yprime, dim=1)
        arc_err = (speed - 1.0) ** 2
        curv = _safe_norm(y2, dim=1)

        t = torch.nn.functional.normalize(yprime, dim=1)
        tc = torch.ones_like(speed)
        if t.shape[0] >= 2:
            dot = (t[1:] * t[:-1]).sum(dim=1).clamp(-1, 1)
            tc = torch.cat([torch.ones(1, device=t.device), dot])

        # 组织表格（注意用 x_val/p_val）
        data = {"s": _to_numpy(s_eval.view(-1))}
        for i in range(x_val.shape[1]):
            data[f"x_{i}"] = _to_numpy(x_val[:, i])
        for j in range(p_val.shape[1]):
            data[f"p_{j}"] = _to_numpy(p_val[:, j])

        data["res"] = _to_numpy(res)
        data["arc_err"] = _to_numpy(arc_err)
        data["curv"] = _to_numpy(curv)
        data["tangent_cos"] = _to_numpy(tc)
        data["sigma_min"] = _to_numpy(sigma_min) if sigma_min is not None else np.full_like(data["res"], np.nan)
        data["tau"] = np.full_like(data["res"], np.nan)

        # 分岔标记
        if detector is not None:
            x_for_flag = data["x_0"]
            p_for_flag = data["p_0"]
            sig = np.asarray(data["sigma_min"]) if np.isfinite(data["sigma_min"]).any() else None
            flags = detector.detect_flags(
                p=np.asarray(p_for_flag),
                x=np.asarray(x_for_flag),
                residual=np.asarray(data["res"]),
                sigma_min=sig,
                tau=None,
            )
            data["bif_flag"] = flags.astype(np.uint8)
        else:
            data["bif_flag"] = np.zeros_like(data["res"], dtype=np.uint8)

        pd.DataFrame(data).to_csv(output_path, index=False)


# =========================
# Case2: metrics（无/有 theory 均支持）
# =========================
def _load_case2_csv(csv_path: str) -> pd.DataFrame:
    """统一列名，兼容 p/x/res/arc_err 的不同命名。"""
    df = pd.read_csv(csv_path)

    # p
    if "p" not in df.columns:
        if "p_0" in df.columns:
            df["p"] = df["p_0"]
        elif "p0" in df.columns:
            df["p"] = df["p0"]
        else:
            raise KeyError("CSV must contain 'p' or 'p_0' column.")

    # x
    if "x" not in df.columns:
        if "x_0" in df.columns:
            df["x"] = df["x_0"]
        elif "x0" in df.columns:
            df["x"] = df["x0"]
        else:
            raise KeyError("CSV must contain 'x' or 'x_0' column.")

    # residual
    if "res" not in df.columns and "residual" in df.columns:
        df["res"] = df["residual"]

    # arc_err
    if "arc_err" not in df.columns:
        df["arc_err"] = np.nan

    return df


def compute_case2_evidence_metrics(csv_path: str) -> Dict[str, float]:
    """
    **无需 theory_fn** 的 Case2 证据性指标：
      - MAE 到最近理论分支：min(|x|, |x-p|)
      - 分区 MAE：p<0 向 x=0；p>0 向 x=p
      - 分岔定位误差：在 |p|<=0.2 的窗口内估计 p*、x*
      - 残差统计：p50/p90/max；弧长误差均值
      - 稳定性准确率（若 CSV 含 'stable'）
    """
    df = _load_case2_csv(csv_path)

    p = df["p"].to_numpy()
    x = df["x"].to_numpy()
    n = len(df)

    err_x0 = np.abs(x)          # 与 x=0 的距离
    err_xp = np.abs(x - p)      # 与 x=p 的距离
    err_near = np.minimum(err_x0, err_xp)

    # 分区 MAE
    mask_neg = p < 0
    mask_pos = p > 0
    mae_neg_to_x0 = float(np.mean(err_x0[mask_neg])) if mask_neg.any() else np.nan
    mae_pos_to_xp = float(np.mean(err_xp[mask_pos])) if mask_pos.any() else np.nan

    mae_to_nearest = float(np.mean(err_near))
    max_err_to_theory = float(np.max(err_near))

    # 分岔定位（近零窗口）
    win = np.abs(p) <= 0.2
    if win.any():
        pw, xw = p[win], x[win]
        j0 = int(np.argmin(np.abs(xw)))        # x≈0 -> 取 p*
        j1 = int(np.argmin(np.abs(xw - pw)))   # x≈p -> 取 x*
        p_at_x0 = float(pw[j0])
        x_at_p0 = float(xw[j1])
    else:
        j0 = int(np.argmin(np.abs(x)))
        j1 = int(np.argmin(np.abs(x - p)))
        p_at_x0 = float(p[j0])
        x_at_p0 = float(x[j1])

    bifurcation_p_error = abs(p_at_x0)         # 期望 ≈ 0
    bifurcation_x_error = abs(x_at_p0 - 0.0)   # 期望 ≈ 0
    bifurcation_distance = float(np.hypot(bifurcation_p_error, bifurcation_x_error))

    # 残差/弧长统计
    if "res" in df.columns:
        res = np.abs(df["res"].to_numpy())
        residual_p50 = float(np.percentile(res, 50))
        residual_p90 = float(np.percentile(res, 90))
        residual_max = float(np.max(res))
        residual_mean = float(np.mean(res))
        residual_median = float(np.median(res))
    else:
        residual_p50 = residual_p90 = residual_max = residual_mean = residual_median = np.nan

    arc_mean = float(np.nanmean(df["arc_err"].to_numpy())) if "arc_err" in df.columns else np.nan
    arc_max = float(np.nanmax(df["arc_err"].to_numpy())) if "arc_err" in df.columns else np.nan

    # 稳定性准确率（可选）
    stability_accuracy = np.nan
    if "stable" in df.columns:
        stable_pred = df["stable"].astype(bool).to_numpy()
        nearer_is_x0 = err_x0 <= err_xp
        # 理论稳定性：x=0 在 p<0 稳定；x=p 在 p>0 稳定
        theory_stable = (nearer_is_x0 & (p < 0)) | ((~nearer_is_x0) & (p > 0))
        stability_accuracy = float((stable_pred == theory_stable).mean())

    out = {
        "n_points": float(n),
        "p_min": float(np.min(p)),
        "p_max": float(np.max(p)),
        "p_range": float(np.max(p) - np.min(p)),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),

        "mae_to_nearest_theory": mae_to_nearest,
        "max_error_to_theory": max_err_to_theory,
        "mae_x0_branch_negative_p": mae_neg_to_x0,
        "mae_xp_branch_positive_p": mae_pos_to_xp,

        "bifurcation_p_error": bifurcation_p_error,
        "bifurcation_x_error": bifurcation_x_error,
        "bifurcation_distance": bifurcation_distance,

        "mean_residual": residual_mean,
        "median_residual": residual_median,
        "residual_p50": residual_p50,
        "residual_p90": residual_p90,
        "max_residual": residual_max,

        "mean_arc_error": arc_mean,
        "max_arc_error": arc_max,

        "stability_accuracy": stability_accuracy,
        "mean_tangent_cos": float(df["tangent_cos"].mean()) if "tangent_cos" in df.columns else np.nan,
        "min_tangent_cos": float(df["tangent_cos"].min()) if "tangent_cos" in df.columns else np.nan,
    }
    return out


def compute_case2_metrics(csv_path: str, theory_fn: Optional[Callable] = None) -> Dict[str, float]:
    """
    兼容旧接口：若提供 theory_fn，则使用其生成的 (stable/unstable) 理论分支进行误差评估；
    否则退化为 evidence 版本（min(|x|, |x-p|) 近似）。
    """
    if theory_fn is None:
        return compute_case2_evidence_metrics(csv_path)

    df = _load_case2_csv(csv_path)
    p = df["p"].to_numpy()
    x = df["x"].to_numpy()

    # 理论两分支（跨临界常见实现：返回 stable, unstable）
    try:
        x_stable, x_unstable = theory_fn(p)
        err_near = np.minimum(np.abs(x - x_stable), np.abs(x - x_unstable))
        mae_to_nearest = float(np.mean(err_near))
        max_err_to_theory = float(np.max(err_near))
    except Exception:
        # 若 theory_fn 形态不符，就退回 evidence
        return compute_case2_evidence_metrics(csv_path)

    # 复用 evidence 其它统计
    base = compute_case2_evidence_metrics(csv_path)
    base["mae_to_nearest_theory"] = mae_to_nearest
    base["max_error_to_theory"] = max_err_to_theory
    return base


def write_case2_metrics(
    csv_path: str,
    out_txt: str,
    out_json: Optional[str] = None,
    theory_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    写指标（txt + 可选 json）。兼容旧调用：
      - 旧：write_case2_metrics(csv, theory_fn, txt, json)
      - 新：write_case2_metrics(csv, txt, json=None, theory_fn=None)
    """
    # if user passed old positional order (csv, theory_fn, txt, json)
    if callable(out_txt):  # 用户把 theory_fn 传在第二位
        # 重新解释参数
        _theory_fn = out_txt  # type: ignore
        _out_txt = out_json   # type: ignore
        _out_json = None
        metrics = compute_case2_metrics(csv_path, _theory_fn)  # type: ignore
        if _out_txt is None:
            raise ValueError("Old-style call detected but no txt path provided.")
        os.makedirs(os.path.dirname(_out_txt), exist_ok=True)
        with open(_out_txt, "w", encoding="utf-8") as f:
            f.write("Case2 Method Metrics\n")
            f.write("=" * 50 + "\n")
            for k, v in metrics.items():
                if isinstance(v, (int, np.integer)):
                    f.write(f"{k}: {int(v)}\n")
                else:
                    try:
                        f.write(f"{k}: {float(v):.3e}\n")
                    except Exception:
                        f.write(f"{k}: {v}\n")
        return metrics

    # 新式/推荐写法
    metrics = compute_case2_metrics(csv_path, theory_fn=theory_fn)
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Case2 Method Metrics\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            if isinstance(v, (int, np.integer)):
                f.write(f"{k}: {int(v)}\n")
            else:
                try:
                    f.write(f"{k}: {float(v):.3e}\n")
                except Exception:
                    f.write(f"{k}: {v}\n")
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    return metrics


# =========================
# Backward-compat helper
# =========================
def annotate_stability_and_bifurcation(csv_path: str,
                                       physics_fn: Optional[Callable] = None,
                                       residual_thresh: float = 1e-3,
                                       sigma_thresh: float = 1e-2,
                                       tau_thresh: float = 5e-3) -> str:
    """
    离线标注：给 CSV 添加/补全
      - stable（若无则用 transcritical 规则兜底：p-2x<0 稳定）
      - bif_flag（标记离 (0,0) 最近的“分叉点”）
    保存为同目录 *_ann.csv，返回新路径。
    """
    df = pd.read_csv(csv_path)
    x_col, p_col = _infer_px_columns(df)

    # 1) 稳定性（已有 stable 列则保留；否则兜底规则）
    if "stable" not in df.columns:
        p = df[p_col].to_numpy()
        x = df[x_col].to_numpy()
        st = (p - 2.0 * x) < 0.0
        df["stable"] = st.astype(int)

    # 2) 分叉点（在残差较小样本中找离原点最近的点）
    x = df[x_col].to_numpy()
    p = df[p_col].to_numpy()
    if "res" in df.columns:
        mask = np.abs(df["res"].to_numpy()) < residual_thresh
        sel = np.argmin(x[mask]**2 + p[mask]**2) if mask.any() else np.argmin(x**2 + p**2)
        idx = (np.where(mask)[0][sel] if mask.any() else sel)
    else:
        idx = int(np.argmin(x**2 + p**2))
    df["bif_flag"] = 0
    df.loc[idx, "bif_flag"] = 1

    # 保存
    base, ext = os.path.splitext(csv_path)
    out_csv = f"{base}_ann{ext}"
    df.to_csv(out_csv, index=False)
    return out_csv
