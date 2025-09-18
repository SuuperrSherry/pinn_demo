# main.py
import os
from config import Config
from train import PINNTrainer
from physics import get_system
from viz import Visualizer

def run_case1_with_dir():
    # Case1：开启方向约束（cos+forward+global），Kendall自适应 + α缩放
    F, nx, np_, theory = get_system("case1_1d")   # 或 "case1_2d"
    cfg = Config()
    cfg.NX, cfg.NP = nx, np_
    cfg.S_MAX = 10.0
    # 起点放在流形上更稳：x0=2 => p0=-4（x^2 + p = 0）
    cfg.Y0 = [2.0, -4.0]

    # —— 方向约束：开启并给出合理权重 —— #
    cfg.DIR_WEIGHTS = {"cos": 0.0, "forward": 0.00, "global": 0.50}
    cfg.DIR_GLOBAL_PARAM_IDX = 0.0
    cfg.DIR_GLOBAL_MARGIN = 1e-5  # 鼓励 dp/ds ≥ 1e-3

    # —— Kendall 自适应 + α 缩放（注意把 dir 的 α 设为 >0 才会生效）—— #
    cfg.USE_KENDALL = True
    cfg.ALPHA = {"phys": 2.0, "arc": 1.0, "ic": 1.0, "smooth": 0.2, "dir": 0.5}


    # —— 可选：启用自适应采样（注意 warmup 期不更新）—— #
    cfg.USE_ADAPTIVE = False
    cfg.ADAPTIVE_WARMUP_ITERS = 800
    cfg.ADAPTIVE_UPDATE_EVERY = 200
    cfg.ADAPTIVE_GRID_SIZE = 256
    cfg.ADAPTIVE_SCORE = "res"      # 或 "sigma"
    cfg.ADAPTIVE_MIX = 0.5          # 与均匀混合
    cfg.ADAPTIVE_TEMP = 0.5
    cfg.BATCH_POINTS = cfg.POINTS_PER_SEGMENT = 80


    trainer = PINNTrainer(cfg, F)
    csv_path, history = trainer.train()

    vis = Visualizer(cfg, out_dir="assets")
    os.makedirs("assets/figs_case1", exist_ok=True)

    # 论文版出图：稳定=实线 / 不稳定=虚线，自动圈出分叉点，误差带
    vis.fig_case1_px_publication(csv_path, "assets/figs_case1/Fig4_1a_px_vs_theory_pub.png", theory)
    vis.plot_residual_vs_s(csv_path, "assets/figs_case1/Fig4_1b_residual_vs_s.png")
    vis.plot_arcerr_vs_s(csv_path, "assets/figs_case1/Fig4_1c_arc_vs_s.png")
    vis.plot_losses(history, "assets/figs_case1/Fig4_1d_training_curves.png", title="Fig.4.1(d) Training losses")
    # 指标文本（max|F| / MAE / 平均弧长偏差）
    vis.report_case1_metrics(csv_path, theory, "assets/figs_case1/Fig4_1_metrics.txt")

def run_case1_baseline():
    # 可选：无方向约束基线，便于对比
    F, nx, np_, theory = get_system("case1_1d")
    cfg = Config()
    cfg.NX, cfg.NP = nx, np_
    cfg.S_MAX = 7.0
    cfg.Y0 = [2.0, -4.0]
    cfg.DIR_WEIGHTS = {"cos": 0.0, "forward": 0.0, "global": 0.0}  # 关闭方向约束
    cfg.USE_KENDALL = True
    cfg.ALPHA = {"phys": 1.5, "arc": 1.0, "ic": 1.0, "smooth": 0.2, "dir": 0.0}

    trainer = PINNTrainer(cfg, F)
    csv_path, history = trainer.train()

    vis = Visualizer(cfg, out_dir="assets")
    os.makedirs("assets/figs_case1_baseline", exist_ok=True)
    vis.fig_case1_px_publication(csv_path, "assets/figs_case1_baseline/Fig4_1a_px_vs_theory_pub_baseline.png", theory)
    vis.plot_residual_vs_s(csv_path, "assets/figs_case1_baseline/Fig4_1b_residual_vs_s.png")
    vis.plot_arcerr_vs_s(csv_path, "assets/figs_case1_baseline/Fig4_1c_arc_vs_s.png")
    vis.plot_losses(history, "assets/figs_case1_baseline/Fig4_1d_training_curves.png", title="Fig.4.1(d) Training losses (baseline)")
    vis.report_case1_metrics(csv_path, theory, "assets/figs_case1_baseline/Fig4_1_metrics.txt")


def run_case2():
    # Case2：启用 cosine 方向约束；其他同 Case1
    F, nx, np_, theory = get_system("case2_2d")   # 2D embed 更利于稳定性标签
    cfg = Config(); cfg.NX, cfg.NP = nx, np_
    cfg.S_MAX = 7.0; cfg.Y0 = [0.0, 0.0,] + [0.0] if nx==2 else [0.0, 0.0]  # x=0, p<0 起点可在训练初期自动修正
    cfg.DIR_WEIGHTS = {"cos":0.5,"forward":0.5,"global":0.0}
    cfg.USE_KENDALL = True; cfg.ALPHA = {"phys":1.0,"arc":1.0,"ic":1.0,"smooth":0.2,"dir":0.3}
    trainer = PINNTrainer(cfg, F)
    csv_path, hist = trainer.train()
    vis = Visualizer(cfg, out_dir="assets")
    os.makedirs("assets/figs_case2", exist_ok=True)
    vis.fig_case2_px_with_two_branches(csv_path, "assets/figs_case2/Fig4_2a_two_branches.png", theory, inset_range=(-0.5,0.5,-0.5,0.5))
    vis.fig_case2_tangent_cos(csv_path, "assets/figs_case2/Fig4_2b_tangent_cos.png")
    vis.plot_losses(hist, "assets/figs_case2/Fig4_2d_training_curves.png", title="Fig.4.2(d) Training losses (proxy to natural param.)")

def run_case3():
    # Case3：Hopf 幅值流形（做法A），适当提高平滑与方向
    F, nx, np_, theory = get_system("case3_amp")
    cfg = Config(); cfg.NX, cfg.NP = nx, np_
    cfg.S_MAX = 6.0; cfg.Y0 = [0.0, 0.0]  # r=0, μ=0 起点
    cfg.DIR_WEIGHTS = {"cos":0.3,"forward":0.4,"global":0.1}
    cfg.USE_KENDALL = True; cfg.ALPHA = {"phys":1.0,"arc":1.0,"ic":1.0,"smooth":0.5,"dir":0.3}
    trainer = PINNTrainer(cfg, F)
    csv_path, hist = trainer.train()
    vis = Visualizer(cfg, out_dir="assets")
    os.makedirs("assets/figs_case3", exist_ok=True)
    vis.fig_case3_r_mu(csv_path, "assets/figs_case3/Fig4_3a_r_of_mu.png", theory)
    vis.fig_case3_curvature(csv_path, "assets/figs_case3/Fig4_3b_curvature.png")
    vis.plot_losses(hist, "assets/figs_case3/Fig4_3d_training_curves.png", title="Fig.4.3(d) Training losses")

if __name__ == "__main__":
    os.makedirs("assets/tables", exist_ok=True)
    # 只跑带方向的版本
    run_case1_with_dir()
    # 如果需要和基线对比，再开下面一行：
    run_case1_baseline()
    print("Done. See assets/figs_case1/ for figures and metrics.")