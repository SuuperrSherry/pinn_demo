# train.py
import os
from typing import Tuple, Dict, List
import torch
import time
from torch.amp import autocast, GradScaler
from config import Config, set_random_seeds
from model import PINN
from losses import compute_loss
from sampler import SamplerFactory, AdaptiveSampler
from bifurcation import BifurcationExporter, SaddleNodeDetector

class PINNTrainer:
    """PINN训练器 - 支持自动混合精度、自适应采样、梯度裁剪等"""

    def train(self):
        start_time = time.time()
    
        # ... 训练过程 ...
    
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f}s")
    
        # 保存训练时间到历史记录
        self.training_history.append({"training_time": training_time})
    
        return csv_path, self.training_history
    
    def __init__(self, config: Config, physics_fn):
        self.config = config
        self.physics_fn = physics_fn
        
        # 设置随机种子
        set_random_seeds()
        
        # 设备和AMP设置
        self.device = config.DEVICE
        self.use_amp = config.USE_AMP and (self.device.type == "cuda")
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        
        # 初始化模型
        self.model = PINN(
            nx=config.NX,
            np=config.NP,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            activation=config.ACTIVATION
        ).to(self.device)
        
        # 初始化优化器和调度器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.LEARNING_RATE
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.LR_STEP_SIZE,
            gamma=config.LR_GAMMA
        )
        
        # 初始化采样器
        self._setup_sampler()
        
        # 初始化分叉检测器
        self.bifurcation_detector = SaddleNodeDetector(
            residual_threshold=config.BIFURCATION_EPS_R,
            sigma_threshold=config.BIFURCATION_EPS_SIGMA,
            tau_threshold=config.BIFURCATION_EPS_TAU,
            debounce_window=config.BIFURCATION_DEBOUNCE
        )
        
        # 准备初始条件
        self.initial_condition = torch.tensor(
            [config.Y0], 
            dtype=torch.float32, 
            device=self.device
        )
        
        # 训练状态
        self.current_epoch = 0
        self.training_history = []
    
    def _setup_sampler(self):
        """设置采样器"""
        config = self.config
        
        # 创建采样器
        sampler_kwargs = {
            "grid_size": config.ADAPTIVE_GRID_SIZE,
            "score_type": config.ADAPTIVE_SCORE_TYPE,
            "mix_ratio": config.ADAPTIVE_MIX_RATIO,
            "temperature": config.ADAPTIVE_TEMPERATURE
        }
        
        self.sampler = SamplerFactory.create_sampler(
            strategy=config.SAMPLING_STRATEGY,
            s_max=config.S_MAX,
            device=self.device,
            **sampler_kwargs
        )
        
        # 初始训练网格
        self.training_grid = self.sampler.sample_grid(config.BATCH_SIZE)
        
        # 自适应采样参数
        self.use_adaptive = (config.SAMPLING_STRATEGY == "adaptive")
        if self.use_adaptive:
            self.adaptive_warmup = config.ADAPTIVE_WARMUP_ITERS
            self.adaptive_update_freq = config.ADAPTIVE_UPDATE_EVERY
    
    def _update_sampling_grid(self):
        """更新采样网格（自适应采样）"""
        if not self.use_adaptive:
            return
            
        if self.current_epoch < self.adaptive_warmup:
            return
            
        # 周期性更新重要性分布
        if ((self.current_epoch - self.adaptive_warmup) % 
            self.adaptive_update_freq == 0):
            
            if hasattr(self.sampler, 'update_distribution'):
                self.sampler.update_distribution(self.model, self.physics_fn)
            
            # 重新采样训练网格
            self.training_grid = self.sampler.sample_grid(self.config.BATCH_SIZE)
    
    def _compute_training_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算训练损失"""
        with autocast("cuda", enabled=self.use_amp):
            total_loss, loss_components = compute_loss(
                model=self.model,
                s=self.training_grid,
                y0=self.initial_condition,
                physics_fn=self.physics_fn,
                config=self.config
            )
        
        return total_loss, loss_components
    
    def _optimization_step(self, loss: torch.Tensor) -> float:
        """执行一步优化"""
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.GRADIENT_CLIP_NORM
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.GRADIENT_CLIP_NORM
            )
            
            self.optimizer.step()
        
        self.scheduler.step()
        return float(grad_norm)
    
    def _log_training_progress(self, loss_components: Dict[str, float], 
                              grad_norm: float):
        """记录训练进度"""
        if self.current_epoch % self.config.LOG_EVERY == 0 or \
           self.current_epoch == self.config.EPOCHS - 1:
            
            # 添加训练信息
            loss_components["grad_norm"] = grad_norm
            loss_components["learning_rate"] = float(self.scheduler.get_last_lr()[0])
            loss_components["s_min"] = float(self.training_grid.min().item())
            loss_components["s_max"] = float(self.training_grid.max().item())
            
            # 打印日志
            print(
                f"Epoch {self.current_epoch:05d} | "
                f"Total={loss_components.get('total', 0):.3e} | "
                f"Phys={loss_components.get('physics', 0):.3e} | "
                f"Arc={loss_components.get('arc_length', 0):.3e} | "
                f"IC={loss_components.get('initial_condition', 0):.3e} | "
                f"Smooth={loss_components.get('smoothness', 0):.3e} | "
                f"Dir={loss_components.get('direction', 0):.3e} | "
                f"Grad={grad_norm:.2e} | "
                f"LR={loss_components['learning_rate']:.1e} | "
                f"s∈[{loss_components['s_min']:.2f},{loss_components['s_max']:.2f}]"
            )
        
        # 记录历史
        self.training_history.append(loss_components)
    
    def train(self) -> Tuple[str, List[Dict[str, float]]]:
        """
        执行完整的训练过程
        
        Returns:
            (csv_path, training_history): 导出的CSV路径和训练历史
        """
        print(f"Starting training for {self.config.EPOCHS} epochs...")
        print(f"Device: {self.device}, AMP: {self.use_amp}, Sampling: {self.config.SAMPLING_STRATEGY}")
        
        self.model.train()
        
        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            
            # 更新采样网格（如果使用自适应采样）
            self._update_sampling_grid()
            
            # 计算损失
            total_loss, loss_components = self._compute_training_loss()
            
            # 优化步骤
            grad_norm = self._optimization_step(total_loss)
            
            # 记录和日志
            self._log_training_progress(loss_components, grad_norm)
        
        # 导出结果
        csv_path = self._export_results()
        
        print(f"Training completed. Results saved to: {csv_path}")
        return csv_path, self.training_history
    
    def _export_results(self) -> str:
        """导出训练结果到CSV"""
        # 准备评估网格
        eval_grid = torch.linspace(
            0.0, self.config.S_MAX,
            steps=self.config.EXPORT_POINTS,
            device=self.device
        ).view(-1, 1)
        
        # 输出路径
        output_dir = os.path.join(self.config.OUTPUT_DIR, "tables")
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "branch.csv")
        
        # 导出
        self.model.eval()
        with torch.no_grad():
            BifurcationExporter.export_to_csv(
                physics_fn=self.physics_fn,
                model=self.model,
                s_eval=eval_grid,
                output_path=csv_path,
                detector=self.bifurcation_detector
            )
        
        return csv_path
    
    def evaluate_at_points(self, s_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在指定点评估模型
        
        Args:
            s_points: 评估点 [N, 1]
            
        Returns:
            (x, p): 状态变量和参数
        """
        self.model.eval()
        with torch.no_grad():
            s_points = s_points.to(self.device)
            x, p = self.model(s_points)
        return x.squeeze(), p.squeeze()
    
    def get_model_state(self) -> dict:
        """获取模型状态字典"""
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.current_epoch,
            "config": self.config
        }
    
    def load_model_state(self, state_dict: dict):
        """加载模型状态"""
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        self.current_epoch = state_dict["epoch"]