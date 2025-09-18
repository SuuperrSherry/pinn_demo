# physics.py
import torch
import numpy as np
from typing import Callable, Dict, Tuple
from abc import ABC, abstractmethod

class PhysicsSystem(ABC):
    """物理系统基类，为扩展新case提供标准接口"""
    
    @abstractmethod
    def equation(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """物理方程 F(x,p) = 0"""
        pass
    
    @abstractmethod  
    def theory_solution(self, p: np.ndarray) -> np.ndarray:
        """理论解，用于对比和误差分析"""
        pass
    
    @property
    @abstractmethod
    def nx(self) -> int:
        """状态变量x的维度"""
        pass
    
    @property  
    @abstractmethod
    def np(self) -> int:
        """参数p的维度"""
        pass

# ========= Case 1: 鞍节点分叉 =========
class Case1SaddleNode(PhysicsSystem):
    """Case1: 鞍节点分叉 F(x,p) = x² + p = 0"""
    
    def __init__(self, embed_2d: bool = False):
        self.embed_2d = embed_2d
    
    def equation(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if self.embed_2d:
            # 2D嵌入版本：第二维衰减到0
            x1, x2 = x[:, 0:1], x[:, 1:2]  
            f1 = x1**2 + p
            f2 = -x2  # 稳定化第二维
            return torch.cat([f1, f2], dim=1)
        else:
            # 1D版本
            return x**2 + p
    
    def theory_solution(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (稳定分支, 不稳定分支)"""
        p = np.asarray(p)
        x_abs = np.sqrt(np.clip(-p, 0.0, None))
        return -x_abs, x_abs  # 稳定(x<0), 不稳定(x>0)
    
    @property
    def nx(self) -> int:
        return 2 if self.embed_2d else 1
    
    @property
    def np(self) -> int:
        return 1

# ========= Case 2: 跨临界分叉 =========  
class Case2Transcritical(PhysicsSystem):
    """Case2: 跨临界分叉 F(x,p) = px - x² = 0"""
    
    def __init__(self, embed_2d: bool = True):
        self.embed_2d = embed_2d
    
    def equation(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if self.embed_2d:
            x1, x2 = x[:, 0:1], x[:, 1:2]
            f1 = p * x1 - x1**2
            f2 = -x2
            return torch.cat([f1, f2], dim=1)
        else:
            return p * x - x**2
    
    def theory_solution(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (x=0分支, x=p分支)"""
        p = np.asarray(p)
        return np.zeros_like(p), p
    
    @property
    def nx(self) -> int:
        return 2 if self.embed_2d else 1
    
    @property
    def np(self) -> int:
        return 1

# ========= Case 3: Hopf分叉幅值流形 =========
class Case3HopfAmplitude(PhysicsSystem):
    """Case3: Hopf分叉幅值流形 F(r,μ) = μr - r³ = 0"""
    
    def equation(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # x为幅值r≥0, p为μ
        return p * x - x**3
    
    def theory_solution(self, p: np.ndarray) -> np.ndarray:
        """幅值解 r = sqrt(max(μ,0))"""
        mu = np.asarray(p)
        return np.sqrt(np.clip(mu, 0.0, None))
    
    @property
    def nx(self) -> int:
        return 1
    
    @property
    def np(self) -> int:
        return 1

# ========= 雅可比矩阵计算 =========
def compute_jacobian_x(F: Callable, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """计算 ∂F/∂x，返回 [B, nx, nx]"""
    F_val = F(x, p)  # [B, nx]
    nx = F_val.shape[1]
    jacobian_rows = []
    
    for i in range(nx):
        grad_i = torch.autograd.grad(
            F_val[:, i].sum(), x, 
            create_graph=True, retain_graph=True
        )[0]  # [B, nx]
        jacobian_rows.append(grad_i.unsqueeze(1))  # [B, 1, nx]
    
    return torch.cat(jacobian_rows, dim=1)  # [B, nx, nx]

def compute_jacobian_p(F: Callable, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """计算 ∂F/∂p，返回 [B, nx, np]"""
    F_val = F(x, p)  # [B, nx]
    nx = F_val.shape[1]
    jacobian_cols = []
    
    for i in range(nx):
        grad_i = torch.autograd.grad(
            F_val[:, i].sum(), p,
            create_graph=True, retain_graph=True
        )[0]  # [B, np]  
        jacobian_cols.append(grad_i.unsqueeze(-1))  # [B, np, 1]
    
    jacobian = torch.cat(jacobian_cols, dim=-1)  # [B, np, nx]
    return jacobian.transpose(-1, -2)  # [B, nx, np]

# ========= 系统注册表 ========= 
_REGISTERED_SYSTEMS = {
    "case1_1d": Case1SaddleNode(embed_2d=False),
    "case1_2d": Case1SaddleNode(embed_2d=True), 
    "case2_1d": Case2Transcritical(embed_2d=False),
    "case2_2d": Case2Transcritical(embed_2d=True),
    "case3_amp": Case3HopfAmplitude(),
}

def get_system(name: str) -> Tuple[Callable, int, int, Callable]:
    """
    获取物理系统
    
    Returns:
        F: 物理方程函数
        nx: x维度
        np_: p维度  
        theory: 理论解函数
    """
    if name not in _REGISTERED_SYSTEMS:
        raise ValueError(f"Unknown system: {name}. Available: {list(_REGISTERED_SYSTEMS.keys())}")
    
    system = _REGISTERED_SYSTEMS[name]
    return system.equation, system.nx, system.np, system.theory_solution

def register_system(name: str, system: PhysicsSystem):
    """注册新的物理系统"""
    _REGISTERED_SYSTEMS[name] = system

def list_systems() -> list:
    """列出所有可用的物理系统"""
    return list(_REGISTERED_SYSTEMS.keys())

# ========= 兼容性别名 (保持原代码接口) =========
# 这些函数保持与原代码的兼容性
def jac_x(F: Callable, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """兼容性别名"""
    return compute_jacobian_x(F, x, p)

def jac_p(F: Callable, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """兼容性别名"""
    return compute_jacobian_p(F, x, p)