# sampler.py
import torch
from typing import Callable, Protocol
from abc import ABC, abstractmethod

class Sampler(ABC):
    """采样器基类"""
    
    @abstractmethod
    def sample_grid(self, num_points: int) -> torch.Tensor:
        """生成训练网格点"""
        pass
    
    @abstractmethod
    def sample_batch(self, batch_size: int) -> torch.Tensor:
        """生成随机批次点"""
        pass

class UniformSampler(Sampler):
    """均匀采样器 - 默认稳定的采样策略"""
    
    def __init__(self, s_max: float, device: torch.device):
        self.s_max = float(s_max)
        self.device = device
    
    def sample_grid(self, num_points: int) -> torch.Tensor:
        """生成均匀网格（包含端点）"""
        return torch.linspace(
            0.0, self.s_max, 
            steps=num_points, 
            device=self.device
        ).view(-1, 1)
    
    def sample_batch(self, batch_size: int) -> torch.Tensor:
        """生成均匀随机采样点"""
        return torch.rand(batch_size, 1, device=self.device) * self.s_max

class AdaptiveSampler(Sampler):
    """
    自适应重要性采样器
    根据物理残差或最小奇异值自适应分配采样点
    """
    
    def __init__(
        self,
        s_max: float,
        device: torch.device,
        grid_size: int = 256,
        score_type: str = "res",
        mix_ratio: float = 0.5,
        temperature: float = 0.5,
        eps: float = 1e-8
    ):
        self.s_max = float(s_max)
        self.device = device
        self.grid_size = grid_size
        self.score_type = score_type
        self.mix_ratio = mix_ratio
        self.temperature = temperature
        self.eps = eps
        
        # 评估网格和概率分布
        self.evaluation_grid = torch.linspace(
            0.0, self.s_max, 
            steps=self.grid_size, 
            device=self.device
        ).view(-1, 1)
        
        # 初始为均匀分布
        self.probability = torch.full(
            (self.grid_size,), 
            1.0 / self.grid_size, 
            device=self.device
        )
    
    def _compute_importance_scores(self, model, physics_fn: Callable) -> torch.Tensor:
        """计算重要性分数"""
        with torch.no_grad():
            x, p = model(self.evaluation_grid)
            
            if self.score_type == "res":
                # 基于残差的重要性
                residual = physics_fn(x, p)  # [G, nx]
                scores = torch.linalg.norm(residual, dim=1)  # [G]
                return scores
                
            elif self.score_type == "sigma":
                # 基于最小奇异值的重要性（计算成本更高）
                from physics import compute_jacobian_x
                
                x_grad = x.detach().requires_grad_(True)
                p_grad = p.detach().requires_grad_(True)
                jacobian = compute_jacobian_x(physics_fn, x_grad, p_grad)  # [G, nx, nx]
                
                singular_values = torch.linalg.svdvals(jacobian)  # [G, min(nx,nx)]
                min_singular_values = singular_values[..., -1]  # [G]
                
                # 转换为重要性分数：越小越重要
                scores = 1.0 / (min_singular_values + self.eps)
                return scores
            
            else:
                raise ValueError(f"Unknown score_type: {self.score_type}")
    
    def update_distribution(self, model, physics_fn: Callable):
        """更新重要性分布"""
        scores = self._compute_importance_scores(model, physics_fn)
        
        # 转换为概率分布（使用温度参数）
        logits = torch.log(scores + self.eps) / max(self.temperature, self.eps)
        importance_prob = torch.softmax(logits, dim=0)
        
        # 与均匀分布混合
        uniform_prob = torch.full_like(importance_prob, 1.0 / self.grid_size)
        mixed_prob = self.mix_ratio * uniform_prob + (1.0 - self.mix_ratio) * importance_prob
        
        # 数值稳定化
        self.probability = (mixed_prob + self.eps)
        self.probability = self.probability / self.probability.sum()
    
    def sample_grid(self, num_points: int) -> torch.Tensor:
        """根据当前重要性分布采样训练网格"""
        indices = torch.multinomial(
            self.probability, 
            num_samples=num_points, 
            replacement=True
        )
        return self.evaluation_grid.index_select(0, indices)
    
    def sample_batch(self, batch_size: int) -> torch.Tensor:
        """随机批次采样"""
        return self.sample_grid(batch_size)
    
    def get_distribution_info(self) -> dict:
        """获取当前分布信息用于调试"""
        return {
            "entropy": -torch.sum(self.probability * torch.log(self.probability + self.eps)).item(),
            "max_prob": self.probability.max().item(),
            "min_prob": self.probability.min().item(),
            "effective_samples": (1.0 / torch.sum(self.probability ** 2)).item()
        }

class SamplerFactory:
    """采样器工厂类"""
    
    @staticmethod
    def create_sampler(strategy: str, s_max: float, device: torch.device, **kwargs) -> Sampler:
        """创建采样器实例"""
        if strategy == "uniform":
            return UniformSampler(s_max, device)
        
        elif strategy == "adaptive":
            return AdaptiveSampler(
                s_max=s_max,
                device=device,
                grid_size=kwargs.get("grid_size", 256),
                score_type=kwargs.get("score_type", "res"),
                mix_ratio=kwargs.get("mix_ratio", 0.5),
                temperature=kwargs.get("temperature", 0.5)
            )
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}. "
                           f"Available: ['uniform', 'adaptive']")

# 兼容性别名，保持原代码接口
class UniformSampler_Legacy:
    """保持与原代码的兼容性"""
    def __init__(self, S_max: float, device):
        self.sampler = UniformSampler(S_max, device)
    
    def sample_grid(self, N: int) -> torch.Tensor:
        return self.sampler.sample_grid(N)
    
    def sample(self, B: int) -> torch.Tensor:
        return self.sampler.sample_batch(B)

class AdaptiveSampler_Legacy:
    """保持与原代码的兼容性"""
    def __init__(self, S_max: float, device, grid_size: int = 256, 
                 score_type: str = "res", mix: float = 0.5, 
                 temperature: float = 0.5, eps: float = 1e-8):
        self.sampler = AdaptiveSampler(
            s_max=S_max, device=device, grid_size=grid_size,
            score_type=score_type, mix_ratio=mix, temperature=temperature, eps=eps
        )
    
    def update(self, model, F: Callable):
        self.sampler.update_distribution(model, F)
    
    def sample_grid(self, N: int) -> torch.Tensor:
        return self.sampler.sample_grid(N)
    
    def sample(self, B: int) -> torch.Tensor:
        return self.sampler.sample_batch(B)