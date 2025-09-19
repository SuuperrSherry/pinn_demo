# main.py
import os
from typing import Dict
from config import Config
from train import PINNTrainer
from physics import get_system
from viz import Visualizer
from auto_spawn import AutoSpawnDetector

def run_case1_with_direction():
    """
    Case1: 鞍节点分叉 - 开启方向约束版本
    论文Fig.4.1系列图表
    """
    print("Running Case1 with direction constraints...")
    
    # 获取物理系统
    physics_fn, nx, np_, theory_fn = get_system("case1_1d")
    
    # 配置参数
    config = Config()
    config.setup_case1(use_direction=True)
    
    # 训练
    trainer = PINNTrainer(config, physics_fn)
    csv_path, training_history = trainer.train()
    
    # 生成图表
    visualizer = Visualizer(config, output_dir="assets")
    output_dir = "assets/figs_case1"
    os.makedirs(output_dir, exist_ok=True)
    
    # 论文级图表
    visualizer.fig_case1_px_publication(
        csv_path, f"{output_dir}/Fig4_1a_px_vs_theory_pub.png", theory_fn
    )
    visualizer.plot_residual_vs_s(
        csv_path, f"{output_dir}/Fig4_1b_residual_vs_s.png"
    )
    visualizer.plot_arc_length_error_vs_s(
        csv_path, f"{output_dir}/Fig4_1c_arc_vs_s.png"
    )
    visualizer.plot_training_curves(
        training_history, f"{output_dir}/Fig4_1d_training_curves.png",
        title="Fig.4.1(d) Training losses"
    )
    
    # 指标报告
    visualizer.generate_case1_metrics_report(
        csv_path, theory_fn, f"{output_dir}/Fig4_1_metrics.txt"
    )
    
    print(f"Case1 (with direction) completed. Results in {output_dir}/")
    return csv_path, training_history

def run_case1_baseline():
    """
    Case1: 鞍节点分叉 - 基线版本（无方向约束）
    用于对比实验
    """
    print("Running Case1 baseline (no direction constraints)...")
    
    # 获取物理系统
    physics_fn, nx, np_, theory_fn = get_system("case1_1d")
    
    # 配置参数
    config = Config()
    config.setup_case1(use_direction=False)
    config.S_MAX = 7.0  # 基线版本使用较短的弧长
    
    # 训练
    trainer = PINNTrainer(config, physics_fn)
    csv_path, training_history = trainer.train()
    
    # 生成图表
    visualizer = Visualizer(config, output_dir="assets")
    output_dir = "assets/figs_case1_baseline"
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer.fig_case1_px_publication(
        csv_path, f"{output_dir}/Fig4_1a_px_vs_theory_pub_baseline.png", theory_fn
    )
    visualizer.plot_residual_vs_s(
        csv_path, f"{output_dir}/Fig4_1b_residual_vs_s.png"
    )
    visualizer.plot_arc_length_error_vs_s(
        csv_path, f"{output_dir}/Fig4_1c_arc_vs_s.png"
    )
    visualizer.plot_training_curves(
        training_history, f"{output_dir}/Fig4_1d_training_curves.png",
        title="Fig.4.1(d) Training losses (baseline)"
    )
    visualizer.generate_case1_metrics_report(
        csv_path, theory_fn, f"{output_dir}/Fig4_1_metrics.txt"
    )
    
    print(f"Case1 baseline completed. Results in {output_dir}/")
    return csv_path, training_history

def run_case2():
    """
    Case2: 跨临界分叉
    论文Fig.4.2系列图表
    """
    print("Running Case2 (transcritical bifurcation)...")
    
    # 获取物理系统
    physics_fn, nx, np_, theory_fn = get_system("case2_2d")  # 2D嵌入更利于稳定性标记
    
    # 配置参数
    config = Config()
    config.setup_case2()
    
    # 训练
    trainer = PINNTrainer(config, physics_fn)
    csv_path, training_history = trainer.train()
    
    # 生成图表
    visualizer = Visualizer(config, output_dir="assets")
    output_dir = "assets/figs_case2"
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer.fig_case2_px_with_two_branches(
        csv_path, f"{output_dir}/Fig4_2a_two_branches.png", theory_fn,
        inset_range=(-0.5, 0.5, -0.5, 0.5)
    )
    visualizer.fig_case2_tangent_cos(
        csv_path, f"{output_dir}/Fig4_2b_tangent_cos.png"
    )
    visualizer.plot_training_curves(
        training_history, f"{output_dir}/Fig4_2d_training_curves.png",
        title="Fig.4.2(d) Training losses (proxy to natural param.)"
    )
    
    print(f"Case2 completed. Results in {output_dir}/")
    return csv_path, training_history

def run_case2_with_autospawn():
    """
    Case2: 跨临界分叉 - 带自动派生版本
    展示框架的自动分支检测和派生能力
    """
    print("\n" + "="*60)
    print("Running Case2 with AUTO-SPAWN capability...")
    print("="*60)
    
    # 获取物理系统
    physics_fn, nx, np_, theory_fn = get_system("case2_2d")
    
    # 配置参数
    config = Config()
    config.setup_case2()
    
    # 创建训练器
    trainer = PINNTrainer(config, physics_fn)
    
    # 使用自动派生训练
    primary_csv, branch_results = trainer.train_with_autospawn()
    
    # 生成可视化
    visualizer = Visualizer(config, output_dir="assets")
    output_dir = "assets/figs_case2_autospawn"
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制自动派生结果
    visualizer.plot_case2_autospawn_results(
        branch_results,
        f"{output_dir}/Fig_case2_autospawn_complete.png",
        theory_fn
    )
    
    # 生成性能报告
    with open(f"{output_dir}/autospawn_report.txt", "w") as f:
        f.write("Case 2: Auto-spawn Performance Report\n")
        f.write("="*50 + "\n\n")
        
        for branch in branch_results:
            f.write(f"Branch: {branch['name'].upper()}\n")
            f.write(f"  CSV: {branch['csv_path']}\n")
            f.write(f"  Initial condition: {branch['initial_condition']}\n")
            
            if 'spawn_point' in branch:
                sp = branch['spawn_point']
                f.write(f"  Spawn point: s={sp.s:.3f}, p={sp.p:.3f}\n")
                f.write(f"  Deviation: {sp.deviation:.3f}\n")
            
            f.write("\n")
    
    print(f"\n✅ Case2 with auto-spawn completed!")
    print(f"Results saved to {output_dir}/")
    
    return branch_results

def run_case3():
    """
    Case3: Hopf分叉幅值流形
    论文Fig.4.3系列图表
    """
    print("Running Case3 (Hopf amplitude manifold)...")
    
    # 获取物理系统
    physics_fn, nx, np_, theory_fn = get_system("case3_amp")
    
    # 配置参数
    config = Config()
    config.setup_case3()
    
    # 训练
    trainer = PINNTrainer(config, physics_fn)
    csv_path, training_history = trainer.train()
    
    # 生成图表
    visualizer = Visualizer(config, output_dir="assets")
    output_dir = "assets/figs_case3"
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer.fig_case3_r_mu(
        csv_path, f"{output_dir}/Fig4_3a_r_of_mu.png", theory_fn
    )
    visualizer.fig_case3_curvature(
        csv_path, f"{output_dir}/Fig4_3b_curvature.png"
    )
    visualizer.plot_training_curves(
        training_history, f"{output_dir}/Fig4_3d_training_curves.png",
        title="Fig.4.3(d) Training losses"
    )
    
    print(f"Case3 completed. Results in {output_dir}/")
    return csv_path, training_history

def run_case2_comprehensive():
    """运行Case2的完整实验，包括对比和指标计算"""
    
    # 1. 基线实验（无方向约束）
    config_baseline = Config()
    config_baseline.setup_case2()
    config_baseline.DIRECTION_WEIGHTS = {"cosine": 0.0, "forward": 0.0, "global": 0.0}
    config_baseline.LOSS_WEIGHTS["direction"] = 0.0
    
    # 2. 增强实验（有方向约束）
    config_enhanced = Config()
    config_enhanced.setup_case2()
    
    physics_fn, nx, np_, theory_fn = get_system("case2_2d")
    
    # 运行对比实验
    results = {}
    for name, config in [("baseline", config_baseline), ("enhanced", config_enhanced)]:
        print(f"Running Case2 {name}...")
        trainer = PINNTrainer(config, physics_fn)
        csv_path, history = trainer.train()
        
        # 计算指标
        from bifurcation import compute_case2_metrics
        metrics = compute_case2_metrics(csv_path, theory_fn)
        
        results[name] = {
            "csv_path": csv_path,
            "metrics": metrics,
            "history": history
        }
        
        print(f"\n=== Case2 {name.title()} Results ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.3e}")
    
    # 生成对比表格和图表
    generate_comparison_table(results, "assets/case2_comparison.txt")
    
    # 生成可视化
    visualizer = Visualizer(Config())
    visualizer.plot_convergence_analysis(
        [results["baseline"]["history"], results["enhanced"]["history"]],
        ["Baseline", "Enhanced"],
        "assets/case2_convergence_comparison.png"
    )
    
    return results

def generate_comparison_table(results: Dict, output_path: str):
    """生成对比表格"""
    with open(output_path, "w") as f:
        f.write("Case2 Method Comparison\n")
        f.write("=" * 50 + "\n")
        
        for method_name, data in results.items():
            f.write(f"\n{method_name.upper()}:\n")
            for metric, value in data["metrics"].items():
                f.write(f"  {metric}: {value:.3e}\n")

def run_all_cases_with_metrics():
    """运行所有case并生成完整的指标报告"""
    
    all_results = {}
    
    # Case1
    print("Running Case1 comprehensive...")
    try:
        case1_enhanced = run_case1_with_direction()
        case1_baseline = run_case1_baseline()
        all_results["case1"] = {
            "enhanced": case1_enhanced,
            "baseline": case1_baseline
        }
    except Exception as e:
        print(f"Case1 failed: {e}")
    
    # Case2  
    print("Running Case2 comprehensive...")
    try:
        case2_results = run_case2_comprehensive()
        all_results["case2"] = case2_results
    except Exception as e:
        print(f"Case2 failed: {e}")
    
    # Case3
    print("Running Case3...")
    try:
        case3_results = run_case3()
        all_results["case3"] = case3_results
    except Exception as e:
        print(f"Case3 failed: {e}")
    
    # 生成综合报告
    generate_final_report(all_results, "assets/comprehensive_results.txt")
    
    return all_results

def generate_final_report(all_results: Dict, output_path: str):
    """生成最终综合报告"""
    with open(output_path, "w") as f:
        f.write("PINN Bifurcation Analysis - Comprehensive Results\n")
        f.write("=" * 60 + "\n")
        
        for case_name, case_data in all_results.items():
            f.write(f"\n{case_name.upper()}:\n")
            f.write("-" * 20 + "\n")
            
            if isinstance(case_data, dict) and "enhanced" in case_data:
                # Case1格式
                f.write("Enhanced (with direction):\n")
                # 写入增强版指标
                f.write("Baseline (no direction):\n")
                # 写入基线指标
            else:
                # 其他case格式
                f.write("Results recorded\n")
            
        f.write(f"\nReport generated: {output_path}\n")
    """
    Case4: 预留接口 - 添加新case的模板
    
    要添加新case时：
    1. 在physics.py中定义F_case4和theory_case4函数
    2. 注册到SYSTEMS字典
    3. 在config.py中添加setup_case4方法
    4. 在viz.py中添加对应的可视化函数
    """
    print("Case4 is not implemented yet. Please define it in physics.py first.")
    
    # 示例模板:
    # physics_fn, nx, np_, theory_fn = get_system("case4")
    # config = Config()
    # config.setup_case4()
    # trainer = PINNTrainer(config, physics_fn)
    # csv_path, history = trainer.train()
    # 
    # visualizer = Visualizer(config)
    # visualizer.plot_case4_results(csv_path, theory_fn)

def run_case5():
    """
    Case5: 预留接口 - 另一个扩展槽位
    """
    print("Case5 is not implemented yet. Please define it in physics.py first.")

def main():
    """主函数 - 运行所有已实现的case"""
    
    # 确保输出目录存在
    os.makedirs("assets/tables", exist_ok=True)
    
    print("=== PINN Bifurcation Analysis ===")
    print("Running all implemented cases...")
    
    # 运行已实现的cases
    '''
    try:
        run_case1_with_direction()
        print()
    except Exception as e:
        print(f"Case1 (with direction) failed: {e}")
    
    try:
        run_case1_baseline() 
        print()
    except Exception as e:
        print(f"Case1 baseline failed: {e}")
    '''
    try:
        run_case2_comprehensive()
        print()
    except Exception as e:
        print(f"Case2 comprehensive failed: {e}")
    try:
        run_case2_with_autospawn()
        print()
    except Exception as e:
        print(f"Case2 with auto-spawn failed: {e}") 
    try:
        run_case2()
        print()
    except Exception as e:
        print(f"Case2 failed: {e}")
    
    try:
        run_case3()
        print()
    except Exception as e:
        print(f"Case3 failed: {e}")
    
    print("=== All cases completed ===")
    print("Check assets/figs_case*/ directories for results.")

if __name__ == "__main__":
    main()