# viz.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Callable, Optional, Tuple, Dict, List
from config import Config

class PlotStyle:
    """统一的绘图样式管理"""
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_matplotlib_style()
    
    def _setup_matplotlib_style(self):
        """设置matplotlib全局样式"""
        plt.rcParams.update({
            "figure.dpi": self.config.FIG_DPI,
            "font.size": self.config.FONT_SIZE,
            "axes.labelsize": self.config.FONT_SIZE,
            "axes.titlesize": self.config.FONT_SIZE,
            "legend.fontsize": self.config.FONT_SIZE - 1,
            "lines.linewidth": self.config.LINE_WIDTH,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.autolayout": True
        })

class DataLoader:
    """CSV数据加载和预处理"""
    
    @staticmethod
    def load_branch_data(csv_path: str) -> pd.DataFrame:
        """加载分支数据CSV"""
        return pd.read_csv(csv_path)
    
    @staticmethod
    def find_column(df: pd.DataFrame, candidates: list) -> str:
        """在DataFrame中查找合适的列名"""
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")

class Visualizer:
    """可视化器 - 生成论文级别的图表"""
    
    def __init__(self, config: Config, output_dir: str = "assets"):
        self.config = config
        self.output_dir = output_dir
        self.style = PlotStyle(config)
    
    # ========= 通用绘图函数 =========
    
    def plot_training_curves(self, history: list, output_path: str, 
                           title: str = "Training losses"):
        """绘制训练曲线"""
        if not history:
            return
            
        iterations = np.arange(1, len(history) + 1)
        loss_names = ["total", "physics", "arc_length", "initial_condition", "smoothness", "direction"]
        
        plt.figure(figsize=(7.0, 5.0))
        
        for loss_name in loss_names:
            if loss_name in history[0]:
                values = [h[loss_name] for h in history]
                plt.semilogy(iterations, values, label=loss_name)
        
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIG_DPI)
        plt.close()
    
    def plot_residual_vs_s(self, csv_path: str, output_path: str):
        """物理残差随弧长参数变化"""
        df = DataLoader.load_branch_data(csv_path)
        s_col = DataLoader.find_column(df, ["s"])
        residual_col = DataLoader.find_column(df, ["res", "residual"])
        
        plt.figure(figsize=(6.6, 5.0))
        plt.semilogy(df[s_col], np.abs(df[residual_col]), "-", color="#1f77b4")
        plt.xlabel("s")
        plt.ylabel("|F(x,p)|")
        plt.title("Physics residual vs s")
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIG_DPI)
        plt.close()
    
    def plot_arc_length_error_vs_s(self, csv_path: str, output_path: str):
        """弧长一致性误差"""
        df = DataLoader.load_branch_data(csv_path)
        s_col = DataLoader.find_column(df, ["s"])
        
        if "arc_err" not in df.columns:
            raise KeyError("Column 'arc_err' not found in CSV")
        
        plt.figure(figsize=(6.6, 5.0))
        plt.plot(df[s_col], df["arc_err"], "-", color="#1f77b4")
        plt.xlabel("s")
        plt.ylabel("(||y'|| - 1)²")
        plt.title("Arc-length consistency vs s")
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIG_DPI)
        plt.close()
    
    # ========= Case 1: 鞍节点分叉 =========
    
    def plot_case1_bifurcation_diagram(self, csv_path: str, output_path: str, 
                                     theory_fn: Callable, eps_residual: float = 1e-3):
        """
        Case1论文级p-x图：稳定=实线，不稳定=虚线，误差带，分叉点标记
        保持与原代码完全一致的图表效果
        """
        df = DataLoader.load_branch_data(csv_path)
        
        # 数据提取
        p_col = DataLoader.find_column(df, ["p", "p1", "p_0", "p0"])
        x_col = DataLoader.find_column(df, ["x", "x1", "x_0", "x0"])
        stability_col = "stable" if "stable" in df.columns else None
        
        p_data = df[p_col].values
        x_data = df[x_col].values
        stability = df[stability_col].astype(bool).values if stability_col else (x_data < 0)
        
        # 理论解
        x_stable_theory, x_unstable_theory = theory_fn(p_data)
        error_envelope = np.minimum(
            np.abs(x_data - x_stable_theory),
            np.abs(x_data - x_unstable_theory)
        )
        
        # 分叉点检测
        bifurcation_p, bifurcation_x = self._detect_bifurcation_point(
            df, p_data, x_data, eps_residual
        )
        
        # 绘图
        plt.figure(figsize=(6.6, 5.0))
        
        # 理论曲线
        theory_mask = p_data <= 0
        plt.plot(p_data[theory_mask], x_stable_theory[theory_mask], 
                "k-", alpha=0.7, label="theory stable")
        plt.plot(p_data[theory_mask], x_unstable_theory[theory_mask], 
                "k--", alpha=0.7, label="theory unstable")
        
        # PINN预测（按稳定性分段绘制）
        self._plot_stability_segments(p_data, x_data, stability, color="#1f77b4")
        
        # 误差带
        plt.fill_between(p_data, x_data - error_envelope, x_data + error_envelope,
                        color="#1f77b4", alpha=0.18, label="|error| band")
        
        # 分叉点标记
        plt.scatter([bifurcation_p], [bifurcation_x], 
                   s=self.config.MARKER_SIZE, edgecolors='k',
                   facecolors='none', zorder=3, label="bifurcation")
        
        # 参考线
        plt.axvline(0.0, color="0.7", linestyle=":", linewidth=1)
        
        plt.xlabel("p")
        plt.ylabel("x")
        plt.title("Fig.4.1(a) Learned (x,p) vs theory (stable solid / unstable dashed)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIG_DPI)
        plt.close()
    
    def _detect_bifurcation_point(self, df: pd.DataFrame, p_data: np.ndarray, 
                                x_data: np.ndarray, eps_residual: float) -> Tuple[float, float]:
        """检测分叉点位置"""
        # 方法1：寻找x的零点交叉
        valid_mask = np.ones_like(x_data, dtype=bool)
        if "res" in df.columns:
            valid_mask &= (np.abs(df["res"].values) < eps_residual)
        
        zero_crossing_indices = np.where(
            (np.sign(x_data[1:]) * np.sign(x_data[:-1]) <= 0) & 
            valid_mask[1:] & valid_mask[:-1]
        )[0]
        
        if len(zero_crossing_indices) > 0:
            # 线性插值找零点
            i = zero_crossing_indices[np.argmin(np.abs(x_data[zero_crossing_indices]))]
            x0, x1 = x_data[i], x_data[i + 1]
            p0, p1 = p_data[i], p_data[i + 1]
            
            if abs(x1 - x0) > 1e-12:
                bifurcation_p = p0 + (0 - x0) * (p1 - p0) / (x1 - x0)
                bifurcation_x = 0.0
            else:
                bifurcation_p, bifurcation_x = p0, x0
        else:
            # 方法2：选择最小奇异值最小的点
            if "sigma_min" in df.columns:
                candidates = np.where(valid_mask)[0]
                if len(candidates) > 0:
                    j = candidates[np.argmin(df["sigma_min"].values[candidates])]
                else:
                    j = np.argmin(df["sigma_min"].values)
            else:
                candidates = np.where(valid_mask)[0]
                if len(candidates) > 0:
                    j = candidates[np.argmin(np.abs(x_data[candidates]))]
                else:
                    j = np.argmin(np.abs(x_data))
            
            bifurcation_p, bifurcation_x = p_data[j], x_data[j]
        
        return bifurcation_p, bifurcation_x
    
    def _plot_stability_segments(self, p_data: np.ndarray, x_data: np.ndarray, 
                               stability: np.ndarray, color: str = "#1f77b4"):
        """按稳定性分段绘制曲线"""
        start_idx = 0
        
        for end_idx in range(1, len(x_data) + 1):
            if end_idx == len(x_data) or stability[end_idx] != stability[end_idx - 1]:
                # 绘制当前段
                linestyle = "-" if stability[end_idx - 1] else "--"
                label = "PINN" if start_idx == 0 else None
                
                plt.plot(p_data[start_idx:end_idx], x_data[start_idx:end_idx], 
                        linestyle, color=color, label=label)
                start_idx = end_idx
    
    def generate_case1_metrics_report(self, csv_path: str, theory_fn: Callable, 
                                    output_path: str, max_residual_threshold: float = 1e-3,
                                    mae_threshold: float = 1e-2, 
                                    arc_error_threshold: float = 5e-3):
        """生成Case1的指标报告"""
        df = DataLoader.load_branch_data(csv_path)
        
        p_col = DataLoader.find_column(df, ["p", "p1", "p_0", "p0"])
        x_col = DataLoader.find_column(df, ["x", "x1", "x_0", "x0"])
        
        p_data = df[p_col].values
        x_data = df[x_col].values
        residual_data = df["res"].values if "res" in df.columns else np.full_like(p_data, np.nan)
        arc_error_data = df["arc_err"].values if "arc_err" in df.columns else np.full_like(p_data, np.nan)
        
        # 计算指标
        x_stable_theory, x_unstable_theory = theory_fn(p_data)
        mae = np.mean(np.minimum(
            np.abs(x_data - x_stable_theory),
            np.abs(x_data - x_unstable_theory)
        ))
        
        max_residual = np.nanmax(np.abs(residual_data))
        mean_arc_error = np.nanmean(arc_error_data)
        
        # 写入报告
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"max|F| = {max_residual:.3e}  (target ≤ {max_residual_threshold:.1e})\n")
            f.write(f"MAE(x_pred, x_true_near) = {mae:.3e}  (target ≤ {mae_threshold:.1e})\n")
            f.write(f"mean arc-length error = {mean_arc_error:.3e}  (target ≤ {arc_error_threshold:.1e})\n")
    
    # ========= Case 2: 跨临界分叉 =========
    
    def plot_case2_two_branches(self, csv_path: str, output_path: str, theory_fn: Callable,
                              inset_range: Tuple[float, float, float, float] = (-0.2, 0.2, -0.2, 0.2)):
        """Case2双分支图with插图 - 修复版本"""
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
        
        df = DataLoader.load_branch_data(csv_path)
        p_col = DataLoader.find_column(df, ["p", "p_0", "p1"])
        x_col = DataLoader.find_column(df, ["x", "x_0", "x1"])
        stability_col = DataLoader.find_column(df, ["stable"])
        
        p_data = df[p_col].values
        x_data = df[x_col].values
        stability = df[stability_col].astype(bool).values
        
        # 理论解 - 跨临界分叉的两个分支
        p_theory = np.linspace(p_data.min(), p_data.max(), 200)
        x_branch1 = np.zeros_like(p_theory)  # x = 0 分支
        x_branch2 = p_theory.copy()          # x = p 分支
        
        plt.figure(figsize=(6.8, 5.0))
        
        # PINN学到的轨迹（按稳定性标记）
        self._plot_stability_segments(p_data, x_data, stability, color="#1f77b4")
        
        # 理论分支
        plt.plot(p_theory, x_branch1, "k-", alpha=0.7, linewidth=2, label="Theory: x=0")
        plt.plot(p_theory, x_branch2, "k--", alpha=0.7, linewidth=2, label="Theory: x=p")
        
        # 插图显示原点附近的细节
        ax_main = plt.gca()
        axins = inset_axes(ax_main, width="40%", height="40%", loc="upper left")
        
        # 在插图中绘制相同内容但放大原点区域
        axins.plot(p_data, x_data, "-", color="#1f77b4", linewidth=1.5)
        axins.plot(p_theory, x_branch1, "k-", alpha=0.7, linewidth=1.5)
        axins.plot(p_theory, x_branch2, "k--", alpha=0.7, linewidth=1.5)
        
        # 设置插图范围聚焦原点
        x1, x2, y1, y2 = inset_range
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.grid(True, alpha=0.3)
        mark_inset(ax_main, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        
        # 标记分叉点
        plt.scatter([0], [0], s=self.config.MARKER_SIZE, 
                   edgecolors='red', facecolors='none', 
                   zorder=5, linewidth=2, label="Bifurcation point")
        
        plt.xlabel("p (parameter)")
        plt.ylabel("x (state)")
        plt.title("Fig.4.2(a) Transcritical bifurcation: Learned trajectory vs theoretical branches")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIG_DPI)
        plt.close()
    
    def plot_case2_tangent_consistency(self, csv_path: str, output_path: str):
        """Case2切向量一致性"""
        df = DataLoader.load_branch_data(csv_path)
        s_col = DataLoader.find_column(df, ["s"])
        
        if "tangent_cos" not in df.columns:
            raise KeyError("Column 'tangent_cos' not found in CSV")
        
        plt.figure(figsize=(6.6, 5.0))
        plt.plot(df[s_col], df["tangent_cos"], "-", color="#1f77b4")
        plt.xlabel("s")
        plt.ylabel("cos(t_i, t_{i-1})")
        plt.title("Fig.4.2(b) Tangent consistency vs s")
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIG_DPI)
        plt.close()
    def plot_case2_autospawn_results(self, branch_results: List[Dict], 
                                output_path: str, theory_fn: Callable):
        """绘制Case2自动派生结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
        # 子图1: 分岔图主结果
        ax1 = axes[0, 0]
    
        # 理论分支
        p_theory = np.linspace(-1.5, 1.5, 300)
        ax1.plot(p_theory, np.zeros_like(p_theory), 'k--', alpha=0.3, label='Theory: x=0')
        ax1.plot(p_theory, p_theory, 'k--', alpha=0.3, label='Theory: x=p')
    
        # 绘制每个分支
        colors = ['blue', 'red', 'green']
        for i, branch_data in enumerate(branch_results):
            df = pd.read_csv(branch_data['csv_path'])
            p_data = df['p_0'].values
            x_data = df['x_0'].values
        
            label = branch_data['name'].capitalize()
            ax1.plot(p_data, x_data, '-', color=colors[i], 
                    label=label, linewidth=2, alpha=0.8)
        
            # 标记初始点
            y0 = branch_data['initial_condition']
            ax1.plot(y0[-1], y0[0], 'o', color=colors[i], markersize=8)
    
        # 标记分岔点
        ax1.plot(0, 0, 'ko', markersize=10, label='Bifurcation')
    
        ax1.set_xlabel('Parameter p')
        ax1.set_ylabel('State x')
        ax1.set_title('Transcritical Bifurcation with Auto-spawn')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.0, 1.5)
    
        # 子图2: 物理残差
        ax2 = axes[0, 1]
        for i, branch_data in enumerate(branch_results):
            df = pd.read_csv(branch_data['csv_path'])
            s_data = df['s'].values
            res_data = df['res'].values if 'res' in df.columns else df['residual'].values
        
            ax2.semilogy(s_data, np.abs(res_data), '-', 
                        color=colors[i], label=branch_data['name'].capitalize())
    
        ax2.set_xlabel('Arc length s')
        ax2.set_ylabel('Physics residual')
        ax2.set_title('Residual Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
        # 子图3: 切向量一致性
        ax3 = axes[1, 0]
        for i, branch_data in enumerate(branch_results):
            df = pd.read_csv(branch_data['csv_path'])
            if 'tangent_cos' in df.columns:
                s_data = df['s'].values
                cos_data = df['tangent_cos'].values
            
                ax3.plot(s_data, cos_data, '-', color=colors[i], 
                        label=branch_data['name'].capitalize(), alpha=0.7)
    
        ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.3, label='Threshold')
        ax3.set_xlabel('Arc length s')
        ax3.set_ylabel('cos(t_i, t_{i-1})')
        ax3.set_title('Tangent Consistency (Spawn Trigger)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
        # 子图4: 性能指标表格
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
    
        # 计算指标
        metrics_data = []
        for branch_data in enumerate(branch_results):
            df = pd.read_csv(branch_data['csv_path'])
            metrics = {
                'Branch': branch_data['name'].capitalize(),
                'Points': len(df),
                'Max |F|': f"{df['res'].max():.2e}" if 'res' in df.columns else 'N/A',
                'Mean arc err': f"{df['arc_err'].mean():.2e}" if 'arc_err' in df.columns else 'N/A'
            }
            metrics_data.append(list(metrics.values()))
    
        table = ax4.table(cellText=metrics_data,
                          colLabels=['Branch', 'Points', 'Max |F|', 'Mean arc err'],
                          loc='center',
                          cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax4.set_title('Performance Metrics')
    
        plt.suptitle('Case 2: Transcritical Bifurcation with Auto-spawn Framework', 
                fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIG_DPI)
        plt.close() 
    # ========= Case 3: Hopf分叉 =========
    
    def plot_case3_amplitude_curve(self, csv_path: str, output_path: str, theory_fn: Callable):
        """Case3幅值曲线r(μ)"""
        df = DataLoader.load_branch_data(csv_path)
        p_col = DataLoader.find_column(df, ["p", "p1", "p_0", "p0"])
        x_col = DataLoader.find_column(df, ["x", "x1", "x_0", "x0"])
        
        mu_data = df[p_col].values
        r_data = df[x_col].values
        r_theory = theory_fn(mu_data)
        
        error = np.abs(r_data - r_theory)
        
        plt.figure(figsize=(6.6, 5.0))
        
        # 理论曲线和PINN预测
        plt.plot(mu_data, r_theory, "k--", alpha=0.7, label="theory r(μ)")
        plt.plot(mu_data, r_data, "-", color="#1f77b4", label="PINN")
        
        # 误差带
        plt.fill_between(mu_data, r_data - error, r_data + error,
                        alpha=0.2, color="#1f77b4", label="|error| band")
        
        plt.xlabel("μ")
        plt.ylabel("r")
        plt.title("Fig.4.3(a) r(μ) learned vs theory")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIG_DPI)
        plt.close()
    
    def plot_case3_curvature_distribution(self, csv_path: str, output_path: str):
        """Case3曲率分布"""
        df = DataLoader.load_branch_data(csv_path)
        s_col = DataLoader.find_column(df, ["s"])
        
        if "curv" not in df.columns:
            raise KeyError("Column 'curv' not found in CSV")
        
        plt.figure(figsize=(6.6, 5.0))
        plt.plot(df[s_col], df["curv"], "-", color="#1f77b4")
        plt.xlabel("s")
        plt.ylabel("||y''||")
        plt.title("Fig.4.3(b) Curvature distribution")
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIG_DPI)
        plt.close()
    
    # ========= 兼容性别名 =========
    # 保持与原代码的接口兼容性
    
    def plot_losses(self, history: list, output_path: str, title: str = "Training losses"):
        """兼容性别名"""
        self.plot_training_curves(history, output_path, title)
    
    def fig_case1_px_publication(self, csv_path: str, output_path: str, theory_fn: Callable, 
                               eps_r: float = 1e-3):
        """兼容性别名"""
        self.plot_case1_bifurcation_diagram(csv_path, output_path, theory_fn, eps_r)
    
    def report_case1_metrics(self, csv_path: str, theory_fn: Callable, output_path: str,
                           max_res_ok: float = 1e-3, mae_ok: float = 1e-2, 
                           mean_arc_ok: float = 5e-3):
        """兼容性别名"""
        self.generate_case1_metrics_report(csv_path, theory_fn, output_path, 
                                         max_res_ok, mae_ok, mean_arc_ok)
    
    def fig_case2_px_with_two_branches(self, csv_path: str, output_path: str, theory_fn: Callable,
                                     inset_range: Tuple[float, float, float, float] = (-0.2, 0.2, -0.2, 0.2)):
        """兼容性别名"""
        self.plot_case2_two_branches(csv_path, output_path, theory_fn, inset_range)
    
    def fig_case2_tangent_cos(self, csv_path: str, output_path: str):
        """兼容性别名"""
        self.plot_case2_tangent_consistency(csv_path, output_path)
    
    def fig_case3_r_mu(self, csv_path: str, output_path: str, theory_fn: Callable):
        """兼容性别名"""
        self.plot_case3_amplitude_curve(csv_path, output_path, theory_fn)
    
    def fig_case3_curvature(self, csv_path: str, output_path: str):
        """兼容性别名"""
        self.plot_case3_curvature_distribution(csv_path, output_path)
    
    def plot_arcerr_vs_s(self, csv_path: str, output_path: str):
        """兼容性别名"""
        self.plot_arc_length_error_vs_s(csv_path, output_path)
    
    # ========= 新增：方法对比和分析功能 =========
    
    def plot_method_comparison(self, results_dict: Dict, output_path: str):
        """绘制方法对比图"""
        methods = list(results_dict.keys())
        metrics = ["max_residual", "mean_branch_distance", "stability_accuracy"]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            if all(metric in results_dict[method]["metrics"] for method in methods):
                values = [results_dict[method]["metrics"][metric] for method in methods]
                axes[i].bar(methods, values)
                axes[i].set_title(metric.replace("_", " ").title())
                if 'residual' in metric:
                    axes[i].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIG_DPI)
        plt.close()
    
    def plot_convergence_analysis(self, history_list: list, labels: list, output_path: str):
        """绘制多个方法的收敛对比"""
        plt.figure(figsize=(10, 6))
        
        for history, label in zip(history_list, labels):
            iterations = range(len(history))
            total_losses = [h["total"] for h in history]
            plt.semilogy(iterations, total_losses, label=label, linewidth=2)
        
        plt.xlabel("Iteration")
        plt.ylabel("Total Loss")
        plt.title("Training Convergence Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIG_DPI)
        plt.close()