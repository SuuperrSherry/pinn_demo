# train.py  —  minimal, keep Case1/Case3 & common path; remove Case2 autospawn bits

import os
from typing import Tuple, Dict, List
import torch

# ---- AMP 兼容导入（不同 PyTorch 版本 API 差异） ----
try:
    from torch.amp import autocast, GradScaler  # 2.0+
except Exception:  # pragma: no cover
    from torch.cuda.amp import autocast, GradScaler  # 1.x fallback

from config import Config, set_random_seeds
from model import PINN
from losses import compute_loss
from sampler import SamplerFactory
from bifurcation import BifurcationExporter, SaddleNodeDetector


class PINNTrainer:
    """PINN训练器 - 支持自动混合精度、自适应采样、梯度裁剪等（精简：去除Case2自动派生相关）"""

    def __init__(self, config: Config, physics_fn):
        self.config = config
        self.physics_fn = physics_fn

        set_random_seeds()

        # 设备 & AMP
        self.device = config.DEVICE if isinstance(config.DEVICE, torch.device) else torch.device(config.DEVICE)
        self.use_amp = bool(config.USE_AMP and (self.device.type == "cuda"))
        self.scaler = GradScaler("cuda" if self.device.type == "cuda" else "cpu", enabled=self.use_amp)

        # 模型
        self.model = PINN(
            nx=config.NX,
            np=config.NP,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            activation=config.ACTIVATION,
        ).to(self.device)

        # 优化器 & 调度器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA
        )

        # 采样器
        self._setup_sampler()

        # 分叉检测器（导出 CSV 时使用）
        self.bifurcation_detector = SaddleNodeDetector(
            residual_threshold=config.BIFURCATION_EPS_R,
            sigma_threshold=config.BIFURCATION_EPS_SIGMA,
            tau_threshold=config.BIFURCATION_EPS_TAU,
            debounce_window=config.BIFURCATION_DEBOUNCE,
        )

        # 初始条件（统一成 float32 张量并放到设备上）
        y0 = torch.as_tensor(config.Y0, dtype=torch.float32, device=self.device)
        if y0.ndim == 1:
            y0 = y0.unsqueeze(0)
        self.initial_condition = y0  # shape: [1, ?]

        # 训练状态
        self.current_epoch = 0
        self.training_history: List[Dict[str, float]] = []
        self.training_stats = {
            "convergence_epoch": None,
            "final_loss": None,
            "gradient_norms": [],
            "loss_plateau_detection": [],
        }

    # -------------------- 内部工具 --------------------

    def _setup_sampler(self):
        cfg = self.config
        sampler_kwargs = {
            "grid_size": cfg.ADAPTIVE_GRID_SIZE,
            "score_type": cfg.ADAPTIVE_SCORE_TYPE,
            "mix_ratio": cfg.ADAPTIVE_MIX_RATIO,
            "temperature": cfg.ADAPTIVE_TEMPERATURE,
        }
        self.sampler = SamplerFactory.create_sampler(
            strategy=cfg.SAMPLING_STRATEGY, s_max=cfg.S_MAX, device=self.device, **sampler_kwargs
        )
        self.training_grid = self.sampler.sample_grid(cfg.BATCH_SIZE)
        self.use_adaptive = (cfg.SAMPLING_STRATEGY == "adaptive")
        if self.use_adaptive:
            self.adaptive_warmup = cfg.ADAPTIVE_WARMUP_ITERS
            self.adaptive_update_freq = cfg.ADAPTIVE_UPDATE_EVERY

    def _detect_convergence(self, current_loss, tolerance=1e-6, window=100) -> bool:
        if len(self.training_history) < window:
            return False
        recent = [h["total"] for h in self.training_history[-window:]]
        if abs(max(recent) - min(recent)) < tolerance and self.training_stats["convergence_epoch"] is None:
            self.training_stats["convergence_epoch"] = self.current_epoch
            return True
        return False

    def _update_sampling_grid(self):
        if not self.use_adaptive:
            return
        if self.current_epoch < self.adaptive_warmup:
            return
        if ((self.current_epoch - self.adaptive_warmup) % self.adaptive_update_freq) == 0:
            if hasattr(self.sampler, "update_distribution"):
                self.sampler.update_distribution(self.model, self.physics_fn)
            self.training_grid = self.sampler.sample_grid(self.config.BATCH_SIZE)

    def _compute_training_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        with autocast(device_type="cuda", enabled=self.use_amp):
            total_loss, loss_components = compute_loss(
                model=self.model,
                s=self.training_grid,
                y0=self.initial_condition,
                physics_fn=self.physics_fn,
                config=self.config,
            )
        return total_loss, loss_components

    def _optimization_step(self, loss: torch.Tensor) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP_NORM)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP_NORM)
            self.optimizer.step()
        return float(grad_norm)

    def _log_training_progress(self, loss_components: Dict[str, float], grad_norm: float):
        loss_components = dict(loss_components)
        loss_components["grad_norm"] = grad_norm
        loss_components["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        loss_components["s_min"] = float(self.training_grid.min().item())
        loss_components["s_max"] = float(self.training_grid.max().item())
        if self.current_epoch % self.config.LOG_EVERY == 0 or self.current_epoch == self.config.EPOCHS - 1:
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
        self.training_history.append(loss_components)

    # -------------------- 训练主流程 --------------------

    def train(self) -> Tuple[str, List[Dict[str, float]]]:
        print(f"Starting training for {self.config.EPOCHS} epochs...")
        print(f"Device: {self.device}, AMP: {self.use_amp}, Sampling: {self.config.SAMPLING_STRATEGY}")

        self.model.train()
        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            self._update_sampling_grid()

            total_loss, loss_components = self._compute_training_loss()
            grad_norm = self._optimization_step(total_loss)
            self._log_training_progress(loss_components, grad_norm)

            # 调度器按 epoch 步进
            self.scheduler.step()

        csv_path = self._export_results()
        print(f"Training completed. Results saved to: {csv_path}")
        return csv_path, self.training_history

    def _export_results(self) -> str:
        eval_grid = torch.linspace(0.0, self.config.S_MAX, steps=self.config.EXPORT_POINTS, device=self.device).view(-1, 1)
        output_dir = os.path.join(self.config.OUTPUT_DIR, "tables")
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "branch.csv")

        self.model.eval()
        with torch.no_grad():
            BifurcationExporter.export_to_csv(
                physics_fn=self.physics_fn,
                model=self.model,
                s_eval=eval_grid,
                output_path=csv_path,
                detector=self.bifurcation_detector,
            )
        return csv_path

    # -------------------- 评估/保存 --------------------

    def evaluate_at_points(self, s_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            s_points = s_points.to(self.device)
            x, p = self.model(s_points)
        return x.squeeze(), p.squeeze()

    def get_model_state(self) -> dict:
        return {
            "model_state_dict": self.model.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "config": self.config,
        }

    def load_model_state(self, state_dict: dict):
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        self.current_epoch = state_dict["epoch"]
