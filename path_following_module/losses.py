# losses.py
import torch
from typing import Dict, Optional, Tuple
from model import PINN

class LossComponents:
    """损失组件计算器 - 保持你的原始损失函数设计"""
    
    @staticmethod
    def physics_residual(physics_fn, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """物理残差损失 - 核心约束"""
        residual = physics_fn(x, p)  # [B, nx]
        return (residual**2).sum(dim=1).mean()
    
    @staticmethod
    def arc_length_constraint(dy_ds: torch.Tensor, target_speed: float = 1.0) -> torch.Tensor:
        """弧长约束 - 保持单位速度参数化"""
        speed = torch.linalg.norm(dy_ds, dim=1)  # [B]
        return ((speed - target_speed) ** 2).mean()
    
    @staticmethod
    def initial_condition(y_at_s0: torch.Tensor, y0_target: torch.Tensor,
                         t_at_s0: Optional[torch.Tensor] = None,
                         t0_target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """初始条件约束"""
        position_loss = ((y_at_s0 - y0_target) ** 2).sum(dim=1).mean()
        
        tangent_loss = 0.0
        if t_at_s0 is not None and t0_target is not None:
            tangent_loss = ((t_at_s0 - t0_target) ** 2).sum(dim=1).mean()
        
        return position_loss + tangent_loss
    
    @staticmethod
    def smoothness_regularization(d2y_ds2: torch.Tensor) -> torch.Tensor:
        """平滑性正则化 - 控制曲率"""
        return (d2y_ds2**2).sum(dim=1).mean()

class DirectionConstraints:
    """方向约束 - 你设计的方向一致性约束"""
    
    @staticmethod
    def _get_sorted_indices(s: torch.Tensor) -> torch.Tensor:
        """获取s的排序索引"""
        return s.squeeze(-1).argsort()
    
    @staticmethod
    def cosine_consistency(dy_ds: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """余弦方向一致性 - 相邻切向量的余弦相似度"""
        sorted_idx = DirectionConstraints._get_sorted_indices(s)
        tangents = dy_ds.index_select(0, sorted_idx)
        
        # 归一化切向量
        t_current = tangents[1:]  # [N-1, D]
        t_previous = tangents[:-1]  # [N-1, D]
        
        t_current_norm = t_current / (t_current.norm(dim=1, keepdim=True) + 1e-8)
        t_previous_norm = t_previous / (t_previous.norm(dim=1, keepdim=True) + 1e-8)
        
        # 余弦相似度损失
        cosine_similarity = (t_current_norm * t_previous_norm).sum(dim=1)
        return (1 - cosine_similarity).mean()
    
    @staticmethod
    def forward_consistency(y: torch.Tensor, dy_ds: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """前向一致性 - 确保切向量与实际位移方向一致"""
        sorted_idx = DirectionConstraints._get_sorted_indices(s)
        positions = y.index_select(0, sorted_idx)
        tangents = dy_ds.index_select(0, sorted_idx)
        
        # 实际位移方向
        actual_displacement = positions[1:] - positions[:-1]  # [N-1, D]
        predicted_tangent = tangents[:-1]  # [N-1, D]
        
        # 归一化
        actual_displacement_norm = actual_displacement / (actual_displacement.norm(dim=1, keepdim=True) + 1e-8)
        predicted_tangent_norm = predicted_tangent / (predicted_tangent.norm(dim=1, keepdim=True) + 1e-8)
        
        # 一致性损失
        consistency = (actual_displacement_norm * predicted_tangent_norm).sum(dim=1)
        return (1 - consistency).mean()
    
    @staticmethod
    def global_direction(dy_ds: torch.Tensor, nx: int, param_idx: int = 0, margin: float = 0.0) -> torch.Tensor:
        """全局方向约束 - 对特定参数的导数施加方向约束"""
        if nx >= dy_ds.shape[1]:
            return torch.tensor(0.0, device=dy_ds.device)
            
        # 提取参数方向的导数
        dp_ds = dy_ds[:, nx + param_idx]
        
        if margin > 0:
            # 鼓励dp/ds >= margin
            return torch.relu(margin - dp_ds).mean()
        else:
            # 鼓励dp/ds >= 0
            return torch.relu(-dp_ds).mean()

class AdaptiveWeighting:
    """自适应权重 - Kendall方法和手动权重"""
    
    @staticmethod
    def kendall_combination(losses: Dict[str, torch.Tensor], 
                          alphas: Dict[str, float],
                          log_sigma_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Kendall自适应权重组合"""
        total_loss = torch.tensor(0.0, device=list(losses.values())[0].device)
        
        for loss_name, loss_value in losses.items():
            if loss_value is None:
                continue
                
            log_sigma = log_sigma_dict.get(loss_name, torch.tensor(0.0))
            alpha = alphas.get(loss_name, 1.0)
            
            # Kendall公式: α * (0.5 * exp(-σ) * L + 0.5 * σ)
            weighted_loss = alpha * (
                0.5 * torch.exp(-log_sigma) * loss_value + 0.5 * log_sigma
            )
            total_loss = total_loss + weighted_loss
        
        return total_loss
    
    @staticmethod
    def manual_combination(losses: Dict[str, torch.Tensor], 
                         weights: Dict[str, float]) -> torch.Tensor:
        """手动权重组合"""
        total_loss = torch.tensor(0.0, device=list(losses.values())[0].device)
        
        for loss_name, loss_value in losses.items():
            if loss_value is None:
                continue
                
            weight = weights.get(loss_name, 1.0)
            total_loss = total_loss + weight * loss_value
        
        return total_loss

def compute_loss(model: PINN, s: torch.Tensor, y0: torch.Tensor, 
                physics_fn, config) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算总损失 - 保持你的原始损失函数设计
    
    Args:
        model: PINN模型
        s: 弧长参数点 [B, 1]
        y0: 初始条件 [1, nx+np]
        physics_fn: 物理方程函数
        config: 配置对象
        
    Returns:
        (total_loss, loss_components): 总损失和各组件损失
    """
    device = s.device
    s = s.requires_grad_(True)
    
    # 前向传播
    x, p = model(s)  # x: [B, nx], p: [B, np]
    y = torch.cat([x, p], dim=1)  # y: [B, nx+np]
    
    # === 基础损失组件 ===
    
    # 物理残差
    physics_loss = LossComponents.physics_residual(physics_fn, x, p)
    
    # 弧长约束
    dy_ds = PINN.compute_first_derivative(y, s)
    arc_length_loss = LossComponents.arc_length_constraint(dy_ds, target_speed=1.0)
    
    # 初始条件
    y_at_s0 = model(torch.zeros(1, 1, device=device))
    y_s0_combined = torch.cat([y_at_s0[0], y_at_s0[1]], dim=1) if isinstance(y_at_s0, tuple) else y_at_s0
    initial_condition_loss = LossComponents.initial_condition(y_s0_combined, y0)
    
    # 平滑性
    d2y_ds2 = PINN.compute_second_derivative(y, s)
    smoothness_loss = LossComponents.smoothness_regularization(d2y_ds2)
    
    # === 方向约束组合 ===
    direction_weights = config.DIRECTION_WEIGHTS
    direction_loss = None
    
    if any(w > 0 for w in direction_weights.values()):
        direction_components = []
        
        if direction_weights.get("cosine", 0.0) > 0:
            cosine_loss = DirectionConstraints.cosine_consistency(dy_ds, s)
            direction_components.append(direction_weights["cosine"] * cosine_loss)
        
        if direction_weights.get("forward", 0.0) > 0:
            forward_loss = DirectionConstraints.forward_consistency(y, dy_ds, s)
            direction_components.append(direction_weights["forward"] * forward_loss)
        
        if direction_weights.get("global", 0.0) > 0:
            global_loss = DirectionConstraints.global_direction(
                dy_ds, 
                nx=config.NX,
                param_idx=config.DIRECTION_GLOBAL_PARAM_IDX,
                margin=config.DIRECTION_GLOBAL_MARGIN
            )
            direction_components.append(direction_weights["global"] * global_loss)
        
        if direction_components:
            direction_loss = sum(direction_components)
    
    # === 组装损失字典 ===
    def _ensure_tensor(value, device):
        """确保值是tensor标量"""
        if value is None:
            return torch.zeros((), device=device)
        if isinstance(value, torch.Tensor):
            return value
        return torch.tensor(value, dtype=torch.float32, device=device)
    
    individual_losses = {
        "physics": physics_loss,
        "arc_length": arc_length_loss,
        "initial_condition": initial_condition_loss,
        "smoothness": smoothness_loss,
        "direction": direction_loss
    }
    
    # 确保所有损失都是tensor
    for name, loss in list(individual_losses.items()):
        individual_losses[name] = _ensure_tensor(loss, device)
    
    # === 总损失计算 ===
    if config.USE_KENDALL:
        # 使用Kendall自适应权重
        total_loss = AdaptiveWeighting.kendall_combination(
            losses=individual_losses,
            alphas=config.LOSS_WEIGHTS,
            log_sigma_dict=model.log_sigma
        )
    else:
        # 使用手动权重
        total_loss = AdaptiveWeighting.manual_combination(
            losses=individual_losses,
            weights=config.LOSS_WEIGHTS
        )
    
    # === 返回结果 ===
    loss_components = {}
    for name, loss in individual_losses.items():
        loss_components[name] = float(loss.detach().cpu().item())
    
    loss_components["total"] = float(total_loss.detach().cpu().item())
    
    return total_loss, loss_components