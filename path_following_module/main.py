# main.py  —— 精简版：保留 Case1 & Case3，Case2 用一键论文流程
import os
from typing import List
from train import PINNTrainer
from viz import Visualizer
from physics import get_system
from bifurcation import annotate_stability_and_bifurcation, write_case2_metrics
from config import Config, set_random_seeds

# ---------- Case 1 ----------
def run_case1_with_direction():
    """Case1: 鞍节点分叉（开启方向约束）"""
    print("Running Case1 with direction constraints...")
    physics_fn, nx, np_, theory_fn = get_system("case1_1d")

    cfg = Config()
    cfg.setup_case1(use_direction=True)

    trainer = PINNTrainer(cfg, physics_fn)
    csv_path, history = trainer.train()

    viz = Visualizer(cfg, output_dir=cfg.OUTPUT_DIR)
    outdir = os.path.join(cfg.OUTPUT_DIR, "figs_case1")
    os.makedirs(outdir, exist_ok=True)

    viz.fig_case1_px_publication(csv_path, os.path.join(outdir, "Fig4_1a_px_vs_theory_pub.png"), theory_fn)
    viz.plot_residual_vs_s(csv_path, os.path.join(outdir, "Fig4_1b_residual_vs_s.png"))
    viz.plot_arc_length_error_vs_s(csv_path, os.path.join(outdir, "Fig4_1c_arc_vs_s.png"))
    viz.plot_training_curves(history, os.path.join(outdir, "Fig4_1d_training_curves.png"),
                             title="Fig.4.1(d) Training losses")
    viz.generate_case1_metrics_report(csv_path, theory_fn, os.path.join(outdir, "Fig4_1_metrics.txt"))
    print(f"[Case1] outputs -> {outdir}")
    return csv_path, history

def run_case1_baseline():
    """Case1: 鞍节点分叉（baseline，无方向约束）"""
    print("Running Case1 baseline (no direction)...")
    physics_fn, nx, np_, theory_fn = get_system("case1_1d")

    cfg = Config()
    cfg.setup_case1(use_direction=False)
    cfg.S_MAX = 7.0

    trainer = PINNTrainer(cfg, physics_fn)
    csv_path, history = trainer.train()

    viz = Visualizer(cfg, output_dir=cfg.OUTPUT_DIR)
    outdir = os.path.join(cfg.OUTPUT_DIR, "figs_case1_baseline")
    os.makedirs(outdir, exist_ok=True)

    viz.fig_case1_px_publication(csv_path, os.path.join(outdir, "Fig4_1a_px_vs_theory_pub_baseline.png"), theory_fn)
    viz.plot_residual_vs_s(csv_path, os.path.join(outdir, "Fig4_1b_residual_vs_s.png"))
    viz.plot_arc_length_error_vs_s(csv_path, os.path.join(outdir, "Fig4_1c_arc_vs_s.png"))
    viz.plot_training_curves(history, os.path.join(outdir, "Fig4_1d_training_curves.png"),
                             title="Fig.4.1(d) Training losses (baseline)")
    viz.generate_case1_metrics_report(csv_path, theory_fn, os.path.join(outdir, "Fig4_1_metrics.txt"))
    print(f"[Case1-baseline] outputs -> {outdir}")
    return csv_path, history

# ---------- Case 2 (paper-ready) ----------
def run_case2_paper_minimal(
    system: str = "case2_2d",
    residual_thresh: float = 1e-3,
    sigma_thresh: float = 1e-2,
    tau_thresh: float = 5e-3,
):
    """
    一键生成 Case2 论文用结果（训练 -> 稳定性/分叉打标 -> 论文图 -> 指标）。
    - 保持 Case1/Case3 不受影响
    - 图与表直接落盘，不弹窗

    Args:
        system: "case2_2d" 或 "case2_1d"
        residual_thresh/sigma_thresh/tau_thresh: 分叉检测阈值（离线判别阶段）

    Returns:
        dict: { "csv": [csv_paths...], "ann_csv": 标注后的CSV, "fig": 论文主图PNG,
                "loss": 训练曲线PNG, "metrics": (txt,json) }
    """
    import os
    import matplotlib
    matplotlib.use("Agg", force=True)  # 无GUI环境更稳
    from physics import get_system
    from config import Config
    from train import PINNTrainer
    from viz import Visualizer
    # 用精简过的 bifurcation 工具（你已替换）
    from bifurcation import (
        annotate_stability_and_bifurcation,
        write_case2_metrics,
    )

    print("\n=== Case2 (paper-minimal) ===")
    print(f"[1/4] Load system: {system}")
    physics_fn, nx, np_, theory_fn = get_system(system)

    print("[2/4] Train a single branch...")
    cfg = Config()
    cfg.setup_case2()  # 使用你现有的 case2 默认训练配置（不影响 case1/3）
    trainer = PINNTrainer(cfg, physics_fn)
    csv_path, history = trainer.train()  # assets/tables/branch.csv

    print("[3/4] Offline annotate stability/bifurcation on branch.csv ...")
    ann_csv = annotate_stability_and_bifurcation(
        csv_path,
        physics_fn,
        residual_thresh=residual_thresh,
        sigma_thresh=sigma_thresh,
        tau_thresh=tau_thresh,
    )

    # 如果你手上已有第二条分支（例如之前保存的 branch2_xp.csv），会自动并入论文大图
    csv_paths = [ann_csv]
    maybe_branch2 = os.path.join(cfg.OUTPUT_DIR, "tables", "branch2_xp.csv")
    if os.path.exists(maybe_branch2):
        csv_paths.append(maybe_branch2)

    print("[4/4] Make paper-ready figure & metrics ...")
    vis = Visualizer(cfg, output_dir=cfg.OUTPUT_DIR)
    fig_dir = os.path.join(cfg.OUTPUT_DIR, "figs_case2_paper")
    os.makedirs(fig_dir, exist_ok=True)

    paper_png = os.path.join(fig_dir, "case2_master.png")
    loss_png = os.path.join(fig_dir, "case2_training_curves.png")
    metrics_txt = os.path.join(fig_dir, "case2_metrics.txt")
    metrics_json = os.path.join(fig_dir, "case2_metrics.json")

    # 论文主图（理论稳/不稳 实/虚线 + PINN 稳/不稳 实/虚线；自带原点 inset）
    vis.plot_case2_master(csv_paths, paper_png, theory_fn)

    # 训练损失图
    vis.plot_training_curves(history, loss_png, title="Case 2 Training Losses")

    # 指标（含 bifurcation_x/p误差、MAE、残差、弧长误差等；若给了 theory_fn 会额外算对理论的误差）
    write_case2_metrics(csv_path, metrics_txt, metrics_json)

    print(f"[OK] CSV (annotated): {ann_csv}")
    print(f"[OK] Figure:          {paper_png}")
    print(f"[OK] Loss curves:     {loss_png}")
    print(f"[OK] Metrics:         {metrics_txt} / {metrics_json}")

    return {
        "csv": csv_paths,
        "ann_csv": ann_csv,
        "fig": paper_png,
        "loss": loss_png,
        "metrics": (metrics_txt, metrics_json),
    }

# ---------- Case 3 ----------
def run_case3():
    """Case3: Hopf 幅值流形"""
    print("Running Case3 (Hopf amplitude manifold)...")
    physics_fn, nx, np_, theory_fn = get_system("case3_amp")

    cfg = Config()
    cfg.setup_case3()

    trainer = PINNTrainer(cfg, physics_fn)
    csv_path, history = trainer.train()

    viz = Visualizer(cfg, output_dir=cfg.OUTPUT_DIR)
    outdir = os.path.join(cfg.OUTPUT_DIR, "figs_case3")
    os.makedirs(outdir, exist_ok=True)

    viz.fig_case3_r_mu(csv_path, os.path.join(outdir, "Fig4_3a_r_of_mu.png"), theory_fn)
    viz.fig_case3_curvature(csv_path, os.path.join(outdir, "Fig4_3b_curvature.png"))
    viz.plot_training_curves(history, os.path.join(outdir, "Fig4_3d_training_curves.png"),
                             title="Fig.4.3(d) Training losses")
    print(f"[Case3] outputs -> {outdir}")
    return csv_path, history

# ---------- 主入口 ----------
def main():
    os.makedirs(os.path.join("assets", "tables"), exist_ok=True)
    print("=== PINN Bifurcation Analysis (C1, C2, C3) ===")

    # Case1：方向 + baseline（可按需注释掉其中一个）
    '''
    try:
        run_case1_with_direction()
    except Exception as e:
        print(f"[WARN] Case1 (direction) failed: {e}")
    try:
        run_case1_baseline()
    except Exception as e:
        print(f"[WARN] Case1 (baseline) failed: {e}")
    '''
    # Case2：论文一键流程
    try:
        run_case2_paper_minimal(system="case2_2d")
    except Exception as e:
        print(f"[WARN] Case2 (paper) failed: {e}")

    # Case3
    try:
        run_case3()

    except Exception as e:
        print(f"[WARN] Case3 failed: {e}")

    print("=== All done. Check assets/figs_case*/ for outputs. ===")

if __name__ == "__main__":
    main()
