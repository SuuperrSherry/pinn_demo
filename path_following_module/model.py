# model.py
import torch
import torch.nn as nn
from typing import Tuple, Dict

class MLP(nn.Module):
    """多层感知机基础网络"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                 num_layers: int, activation: str = "tanh"):
        super().__init__()
        
        # 激活函数映射
        activation_map = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(), 
            "gelu": nn.GELU(),
            "silu": nn.SiLU()
        }
        
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}. "
                           f"Available: {list(activation_map.keys())}")
        
        act_fn = activation_map[activation]
        
        # 构建网络层
        layers = []
        current_dim = input_dim
        
        # 隐藏层
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_size),
                act_fn
            ])
            current_dim = hidden_size
        
        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class PINN(nn.Module):
    """物理信息神经网络 (Physics-Informed Neural Network)"""
    
    def __init__(self, nx: int, np: int, hidden_size: int, 
                 num_layers: int, activation: str = "tanh"):
        super().__init__()
        
        self.nx = nx  # 状态变量维度
        self.np = np  # 参数维度
        
        # 主网络: s -> [x, p]
        self.mlp = MLP(
            input_dim=1,
            output_dim=nx + np, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation
        )
        
        # Kendall自适应权重的log_sigma参数
        self.log_sigma = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) 
            for name in ["physics", "arc_length", "initial_condition", 
                        "smoothness", "direction"]
        })
    
    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            s: 弧长参数 [B, 1]
            
        Returns:
            x: 状态变量 [B, nx]  
            p: 参数 [B, np]
        """
        y = self.mlp(s)  # [B, nx+np]
        
        x = y[:, :self.nx]                    # [B, nx]
        p = y[:, self.nx:self.nx + self.np]   # [B, np]
        
        return x, p
    
    @staticmethod
    def compute_first_derivative(y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        计算一阶导数 dy/ds
        
        Args:
            y: 函数值 [B, D] 
            s: 自变量 [B, 1]
            
        Returns:
            dy/ds: 一阶导数 [B, D]
        """
        batch_size, dim = y.shape
        gradients = []
        
        for i in range(dim):
            grad_i = torch.autograd.grad(
                outputs=y[:, i].sum(),
                inputs=s,
                create_graph=True,
                retain_graph=True
            )[0]  # [B, 1]
            gradients.append(grad_i)
        
        return torch.cat(gradients, dim=1)  # [B, D]
    
    @staticmethod
    def compute_second_derivative(y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        计算二阶导数 d²y/ds²
        
        Args:
            y: 函数值 [B, D]
            s: 自变量 [B, 1] 
            
        Returns:
            d²y/ds²: 二阶导数 [B, D]
        """
        batch_size, dim = y.shape
        second_derivatives = []
        
        for i in range(dim):
            # 先计算一阶导数
            first_deriv = torch.autograd.grad(
                outputs=y[:, i].sum(),
                inputs=s,
                create_graph=True,
                retain_graph=True
            )[0]  # [B, 1]
            
            # 再计算二阶导数
            second_deriv = torch.autograd.grad(
                outputs=first_deriv.sum(),
                inputs=s,
                create_graph=True,
                retain_graph=True
            )[0]  # [B, 1]
            
            second_derivatives.append(second_deriv)
        
        return torch.cat(second_derivatives, dim=1)  # [B, D]
    
    def get_kendall_weights(self) -> Dict[str, torch.Tensor]:
        """获取当前的Kendall自适应权重"""
        return {name: torch.exp(-log_sigma) for name, log_sigma in self.log_sigma.items()}

    # 保持与原代码的兼容性
    @staticmethod
    def first_derivative(y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """兼容性别名"""
        return PINN.compute_first_derivative(y, s)
    
    @staticmethod 
    def second_derivative(y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """兼容性别名"""
        return PINN.compute_second_derivative(y, s)