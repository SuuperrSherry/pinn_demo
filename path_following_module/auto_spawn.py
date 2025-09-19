# auto_spawn.py - 新文件，完全独立模块
import numpy as np
import torch
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class BranchPoint:
    """分支点信息"""
    s: float
    x: np.ndarray
    p: float
    tangent: np.ndarray
    deviation: float

class AutoSpawnDetector:
    """自动分支检测器"""
    
    def __init__(self, 
                 cos_threshold: float = 0.7,
                 window_size: int = 21,
                 min_s_before_spawn: float = 1.0):
        self.cos_threshold = cos_threshold
        self.window_size = window_size
        self.min_s_before_spawn = min_s_before_spawn
        
    def detect_branch_points(self, df: pd.DataFrame) -> List[BranchPoint]:
        """从CSV数据检测潜在的分支点"""
        branch_points = []
        
        # 提取数据
        s_data = df['s'].values
        x_data = df[['x_0', 'x_1']].values if 'x_1' in df.columns else df[['x_0']].values
        p_data = df['p_0'].values
        
        # 计算切向量余弦
        if 'tangent_cos' in df.columns:
            cos_values = df['tangent_cos'].values
            
            # 使用滑动窗口检测异常
            for i in range(self.window_size, len(cos_values) - self.window_size):
                if s_data[i] < self.min_s_before_spawn:
                    continue
                    
                # 计算局部平均
                local_avg = np.mean(cos_values[i-self.window_size//2:i+self.window_size//2])
                deviation = abs(cos_values[i] - local_avg)
                
                # 检测显著偏离
                if cos_values[i] < self.cos_threshold and deviation > 0.2:
                    # 估计切向量
                    if i > 0:
                        dx_ds = (x_data[i] - x_data[i-1]) / (s_data[i] - s_data[i-1] + 1e-8)
                        dp_ds = (p_data[i] - p_data[i-1]) / (s_data[i] - s_data[i-1] + 1e-8)
                        tangent = np.concatenate([dx_ds, [dp_ds]])
                        tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
                    else:
                        tangent = np.array([1.0, 0.0, 0.0])
                    
                    branch_point = BranchPoint(
                        s=s_data[i],
                        x=x_data[i],
                        p=p_data[i],
                        tangent=tangent,
                        deviation=deviation
                    )
                    branch_points.append(branch_point)
        
        return branch_points