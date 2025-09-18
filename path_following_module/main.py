# main.py
import os
from config import Config
from train import PINNTrainer
from physics import get_system
from viz import Visualizer

def run_case1_with_direction():
    """
    Case1: 鞍节点分叉 - 开启方向约束版本
    论文Fig.4.1系列图表
    """
    import time
    start_time = time.time()
    
    print("Running Case1 with direction constraints...")
    
    # 获取物理系统
    physics_fn, nx, np_, theory_fn = get_system("case1_1d")
    
    # 配置参数
    config = Config()
    config.setup_case1(use_direction=True)
    
    # 训练
    trainer = PINNTrainer(config, physics_fn)
    csv_path, training_history = trainer.train()
    
    training_time = time.time() - start_time
    
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
        training_history, f"{output_dir}/training_curves.png",
        title="Training losses"
    )
    
    # 计算指标
    from bifurcation import compute_case1_metrics
    
    try:
        metrics = compute_case1_metrics(csv_path, theory_fn)
        
        print("\n=== Case 1 Performance Metrics ===")
        print(f"Max |F(x,p)|: {metrics['max_residual']:.2e}")
        print(f"MAE vs Theory: {metrics['mae_theory']:.2e}")  
        print(f"Mean Arc Error: {metrics['mean_arc_error']:.2e}")
        print(f"Bifurcation Error: {metrics['bifurcation_error']:.2e}")
        print(f"Training Time: {training_time:.1f}s")
        
        # 保存到文件
        with open(f"{output_dir}/metrics_summary.txt", "w") as f:
            f.write(f"Max |F(x,p)|: {metrics['max_residual']:.2e}\n")
            f.write(f"MAE vs Theory: {metrics['mae_theory']:.2e}\n")
            f.write(f"Mean Arc Error: {metrics['mean_arc_error']:.2e}\n")
            f.write(f"Training Time: {training_time:.1f}s\n")
            
        print(f"Metrics saved to {output_dir}/metrics_summary.txt")
        
    except Exception as e:
        print(f"Error computing metrics: {e}")
        print("CSV file structure:")
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
    
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

def run_case4():
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