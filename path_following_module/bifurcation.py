# bifurcation.py
import torch
import csv
import os
from typing import Callable, Dict, NamedTuple
from collections import deque
from physics import compute_jacobian_x, compute_jacobian_p
from model import PINN

class BifurcationIndicators(NamedTuple):
    """分叉指标数据结构"""
    residual: torch.Tensor          # 物理残差范数
    min_singular_value: torch.Tensor # 最小奇异值
    tau: torch.Tensor               # tau = ||w^T F_p||
    
class GeometricMetrics(NamedTuple):
    """几何指标数据结构"""
    arc_length_error: torch.Tensor   # 弧长一致性误差
    curvature: torch.Tensor          # 曲率
    tangent_cosine: torch.Tensor     # 相邻切向量余弦

class BifurcationAnalyzer:
    """分叉分析器 - 计算各种分叉指标和几何量"""
    
    @staticmethod
    def compute_indicators(physics_fn: Callable, x: torch.Tensor, 
                          p: torch.Tensor) -> BifurcationIndicators:
        """
        计算分叉检测指标
        
        Args:
            physics_fn: 物理方程 F(x,p)
            x: 状态变量 [B, nx]
            p: 参数 [B, np]
            
        Returns:
            BifurcationIndicators包含残差、最小奇异值、tau
        """
        with torch.enable_grad():
            # 重新构建计算图
            x_grad = x.detach().requires_grad_(True)
            p_grad = p.detach().requires_grad_(True)
            
            # 计算残差范数
            residual = torch.linalg.norm(physics_fn(x_grad, p_grad), dim=1)  # [B]
            
            # 计算雅可比矩阵
            jac_x = compute_jacobian_x(physics_fn, x_grad, p_grad)  # [B, nx, nx]  
            jac_p = compute_jacobian_p(physics_fn, x_grad, p_grad)  # [B, nx, np]
            
            # SVD分解获取最小奇异值和对应的左奇异向量
            U, S, Vh = torch.linalg.svd(jac_x, full_matrices=False)
            min_singular_value = S[..., -1]  # [B] 最小奇异值
            w = U[..., :, -1]  # [B, nx] 最小左奇异向量
            
            # 计算tau = ||w^T F_p||
            projection = torch.matmul(w.unsqueeze(1), jac_p).squeeze(1)  # [B, np]
            tau = torch.linalg.norm(projection, dim=1)  # [B]
        
        return BifurcationIndicators(
            residual=residual.detach(),
            min_singular_value=min_singular_value.detach(), 
            tau=tau.detach()
        )
    
    @staticmethod
    def compute_stability(physics_fn: Callable, x: torch.Tensor, 
                         p: torch.Tensor) -> torch.Tensor:
        """
        计算稳定性标记
        
        Args:
            physics_fn: 物理方程
            x: 状态变量 [B, nx]
            p: 参数 [B, np]
            
        Returns:
            稳定性布尔向量 [B]，True表示稳定
        """
        with torch.enable_grad():
            x_grad = x.detach().requires_grad_(True)
            p_grad = p.detach().requires_grad_(True)
            
            jac_x = compute_jacobian_x(physics_fn, x_grad, p_grad)  # [B, nx, nx]
            eigenvalues = torch.linalg.eigvals(jac_x)  # [B, nx] (复数)
            
            # 稳定性：所有特征值实部 < 0
            max_real_part = eigenvalues.real.max(dim=1).values  # [B]
            stable = (max_real_part < 0)
        
        return stable.detach()
    
    @staticmethod
    def compute_geometric_metrics(model: PINN, s_eval: torch.Tensor) -> GeometricMetrics:
        """
        计算几何指标
        
        Args:
            model: PINN模型
            s_eval: 弧长参数评估点 [B, 1]
            
        Returns:
            GeometricMetrics包含弧长误差、曲率、切向量余弦
        """
        with torch.enable_grad():
            s = s_eval.detach().clone().requires_grad_(True)
            x, p = model(s)
            y = torch.cat([x, p], dim=1)  # [B, nx+np]
            
            # 一阶和二阶导数
            dy_ds = PINN.compute_first_derivative(y, s)    # [B, nx+np]
            d2y_ds2 = PINN.compute_second_derivative(y, s) # [B, nx+np]
            
            # 弧长一致性误差
            speed = torch.linalg.norm(dy_ds, dim=1)  # [B]
            arc_length_error = (speed - 1.0) ** 2
            
            # 曲率
            curvature = torch.linalg.norm(d2y_ds2, dim=1)  # [B]
            
            # 相邻切向量余弦相似度
            tangent_unit = dy_ds / (speed.unsqueeze(-1) + 1e-8)  # [B, nx+np]
            tangent_cosine = torch.ones_like(speed)
            if len(tangent_unit) > 1:
                tangent_cosine[1:] = (tangent_unit[1:] * tangent_unit[:-1]).sum(dim=1)
        
        return GeometricMetrics(
            arc_length_error=arc_length_error.detach(),
            curvature=curvature.detach(),
            tangent_cosine=tangent_cosine.detach()
        )

class SaddleNodeDetector:
    """鞍节点分叉检测器（带去抖功能）"""
    
    def __init__(self, residual_threshold: float = 1e-4, 
                 sigma_threshold: float = 1e-3,
                 tau_threshold: float = 1e-3, 
                 debounce_window: int = 5):
        self.residual_threshold = residual_threshold
        self.sigma_threshold = sigma_threshold  
        self.tau_threshold = tau_threshold
        self.detection_buffer = deque(maxlen=debounce_window)
    
    def update(self, indicators: BifurcationIndicators) -> bool:
        """
        更新检测器并返回是否检测到分叉点
        
        Args:
            indicators: 分叉指标
            
        Returns:
            bool: 是否检测到稳定的分叉信号
        """
        # 检查是否满足所有阈值条件
        condition = (
            (indicators.residual < self.residual_threshold) &
            (indicators.min_singular_value < self.sigma_threshold) & 
            (indicators.tau > self.tau_threshold)
        ).any().item()
        
        self.detection_buffer.append(bool(condition))
        
        # 只有连续满足条件才触发检测
        return (len(self.detection_buffer) == self.detection_buffer.maxlen and 
                all(self.detection_buffer))
    
    def reset(self):
        """重置检测器状态"""
        self.detection_buffer.clear()

class BifurcationExporter:
    """分叉分析结果导出器"""
    
    @staticmethod
    def export_to_csv(physics_fn: Callable, model: PINN, s_eval: torch.Tensor,
                     output_path: str, detector: SaddleNodeDetector):
        """
        导出分叉分析结果到CSV文件
        
        Args:
            physics_fn: 物理方程
            model: PINN模型
            s_eval: 评估点
            output_path: 输出路径
            detector: 分叉检测器
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 获取模型预测
        with torch.no_grad():
            x, p = model(s_eval)
        
        # 计算各种指标
        indicators = BifurcationAnalyzer.compute_indicators(physics_fn, x, p)
        metrics = BifurcationAnalyzer.compute_geometric_metrics(model, s_eval)
        stability = BifurcationAnalyzer.compute_stability(physics_fn, x, p)
        
        # 分叉标记
        bifurcation_flags = (
            (indicators.residual < detector.residual_threshold) &
            (indicators.min_singular_value < detector.sigma_threshold) &  
            (indicators.tau > detector.tau_threshold)
        ).to(torch.int32)
        
        # 转移到CPU用于导出
        data_to_export = {
            's': s_eval.detach().cpu(),
            'x': x.detach().cpu(),
            'p': p.detach().cpu(), 
            'residual': indicators.residual.cpu(),
            'min_singular_value': indicators.min_singular_value.cpu(),
            'tau': indicators.tau.cpu(),
            'stability': stability.cpu().to(torch.int32),
            'bifurcation_flag': bifurcation_flags.cpu(),
            'arc_length_error': metrics.arc_length_error.cpu(),
            'curvature': metrics.curvature.cpu(),
            'tangent_cosine': metrics.tangent_cosine.cpu()
        }
        
        # 构建列名
        nx = x.shape[1] 
        column_names = ['s']
        column_names.extend([f'x_{i}' for i in range(nx)])
        column_names.extend(['p_0', 'res', 'sigma_min', 'tau', 'stable', 
                           'bif_flag', 'arc_err', 'curv', 'tangent_cos'])
        
        # 写入CSV
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_names)
            
            batch_size = s_eval.shape[0]
            for i in range(batch_size):
                row = [float(data_to_export['s'][i].item())]
                
                # x分量
                for j in range(nx):
                    row.append(float(data_to_export['x'][i, j].item()))
                
                # 其余数据
                row.extend([
                    float(data_to_export['p'][i, 0].item()),
                    float(data_to_export['residual'][i].item()),
                    float(data_to_export['min_singular_value'][i].item()),
                    float(data_to_export['tau'][i].item()),
                    int(data_to_export['stability'][i].item()),
                    int(data_to_export['bifurcation_flag'][i].item()),
                    float(data_to_export['arc_length_error'][i].item()),
                    float(data_to_export['curvature'][i].item()),
                    float(data_to_export['tangent_cosine'][i].item())
                ])
                
                writer.writerow(row)

# ========= 兼容性函数 =========
# 保持与原代码的接口兼容性

def compute_indicators(F: Callable, x: torch.Tensor, p: torch.Tensor) -> Dict[str, torch.Tensor]:
    """兼容性函数"""
    indicators = BifurcationAnalyzer.compute_indicators(F, x, p)
    return {
        "res": indicators.residual,
        "sigma_min": indicators.min_singular_value,
        "tau": indicators.tau
    }

def compute_stability(F: Callable, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """兼容性函数"""
    return BifurcationAnalyzer.compute_stability(F, x, p)

def eval_geom_terms(net, s_eval: torch.Tensor) -> Dict[str, torch.Tensor]:
    """兼容性函数"""
    metrics = BifurcationAnalyzer.compute_geometric_metrics(net, s_eval)
    return {
        "arc_err": metrics.arc_length_error,
        "curv": metrics.curvature, 
        "tangent_cos": metrics.tangent_cosine
    }

def export_branch_csv(F: Callable, net, s_eval: torch.Tensor, out_csv: str,
                      nx: int, np_: int, detector: SaddleNodeDetector):
    """兼容性函数"""
    BifurcationExporter.export_to_csv(F, net, s_eval, out_csv, detector)