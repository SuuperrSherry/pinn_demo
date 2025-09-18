# config.py
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class Config:
    """统一配置管理 - 所有训练参数都在这里修改"""
    
    # ===== 系统维度 =====
    NX: int = 2                    # x的维度 (状态变量)
    NP: int = 1                    # p的维度 (参数)
    
    # ===== 可视化索引 =====
    X_PLOT_IDX: int = 0            # p-x图中选择哪个x分量
    P_PLOT_IDX: int = 0            # p-x图中选择哪个p分量
    
    # ===== 网络结构 =====
    HIDDEN_SIZE: int = 32          # 隐藏层神经元数量
    NUM_LAYERS: int = 3            # 隐藏层数量
    ACTIVATION: str = "tanh"       # 激活函数: tanh, relu, gelu, silu
    
    # ===== 训练参数 =====
    LEARNING_RATE: float = 1e-3    # 学习率
    EPOCHS: int = 8000             # 训练轮数
    BATCH_SIZE: int = 128          # 批大小
    
    # ===== 弧长参数化范围 =====
    S_MAX: float = 10.0            # 弧长参数最大值
    
    # ===== 初始条件 (长度应为NX+NP) =====
    Y0: List[float] = field(default_factory=lambda: [2.0, -4.0])  # [x0, p0]
    
    # ===== 损失函数权重 =====
    USE_KENDALL: bool = True       # 是否使用Kendall自适应权重
    LOSS_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "physics": 2.0,            # 物理残差权重
        "arc_length": 1.0,         # 弧长约束权重  
        "initial_condition": 1.0,   # 初始条件权重
        "smoothness": 0.2,         # 平滑性权重
        "direction": 0.5           # 方向约束权重
    })
    
    # ===== 方向约束配置 =====
    DIRECTION_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "cosine": 0.0,             # 余弦方向约束
        "forward": 0.0,            # 前向方向约束
        "global": 0.5              # 全局方向约束
    })
    DIRECTION_GLOBAL_PARAM_IDX: int = 0  # 全局方向约束的参数索引
    DIRECTION_GLOBAL_MARGIN: float = 1e-5  # 全局方向约束的边际
    
    # ===== 分叉检测阈值 =====
    BIFURCATION_EPS_R: float = 1e-4      # 残差阈值
    BIFURCATION_EPS_SIGMA: float = 1e-3   # 最小奇异值阈值
    BIFURCATION_EPS_TAU: float = 1e-3     # tau阈值
    BIFURCATION_DEBOUNCE: int = 5         # 去抖窗口大小
    
    # ===== 采样策略 =====
    SAMPLING_STRATEGY: str = "adaptive"    # "uniform" 或 "adaptive"
    ADAPTIVE_GRID_SIZE: int = 256         # 自适应采样网格大小
    ADAPTIVE_SCORE_TYPE: str = "res"      # 评分类型: "res" 或 "sigma"
    ADAPTIVE_MIX_RATIO: float = 0.5       # 与均匀分布混合比例
    ADAPTIVE_TEMPERATURE: float = 0.5     # softmax温度
    ADAPTIVE_WARMUP_ITERS: int = 800      # 自适应采样预热轮数
    ADAPTIVE_UPDATE_EVERY: int = 200      # 自适应采样更新频率
    
    # ===== 训练控制 =====
    USE_AMP: bool = True                  # 自动混合精度
    GRADIENT_CLIP_NORM: float = 1.0       # 梯度裁剪范数
    LR_STEP_SIZE: int = 2000             # 学习率调度步长
    LR_GAMMA: float = 0.3                # 学习率衰减因子
    
    # ===== 日志和输出 =====
    LOG_EVERY: int = 2000                 # 日志输出频率
    EXPORT_POINTS: int = 600             # 导出CSV的点数
    OUTPUT_DIR: str = "assets"           # 输出目录
    
    # ===== 图表设置 =====
    FIG_DPI: int = 300                   # 图片DPI
    FONT_SIZE: int = 12                  # 字体大小
    LINE_WIDTH: float = 2.0              # 线宽
    MARKER_SIZE: int = 60                # 标记大小
    
    # ===== 设备设置 =====
    DEVICE: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    def setup_case1(self, use_direction: bool = True):
        """Case1: 鞍节点分叉配置"""
        self.NX, self.NP = 1, 1
        self.S_MAX = 10.0
        self.Y0 = [2.0, -4.0]  # x0=2, p0=-4 (在流形上)
        
        if use_direction:
            self.DIRECTION_WEIGHTS = {"cosine": 0.2, "forward": 0.15, "global": 0.30}
            self.DIRECTION_GLOBAL_MARGIN = 1e-5
            self.LOSS_WEIGHTS["direction"] = 0.2
        else:
            self.DIRECTION_WEIGHTS = {"cosine": 0.0, "forward": 0.0, "global": 0.0}
            self.LOSS_WEIGHTS["direction"] = 0.0
            
        self.LOSS_WEIGHTS.update({
            "physics": 2.0,
            "arc_length": 1.0,
            "initial_condition": 1.0,
            "smoothness": 0.2
        })
    
    def setup_case2(self):
        """Case2: 跨临界分叉配置 - 优化版本"""
        self.NX, self.NP = 2, 1  # 2D嵌入更利于稳定性标记
        self.S_MAX = 7.0
        self.Y0 = [0.05, 0.0, 0.05]  # x=[0.05, 0], p=0.05 接近原点但避免奇点
        
        # 大幅降低方向约束权重 - 跨临界分叉在原点处方向变化剧烈
        self.DIRECTION_WEIGHTS = {"cosine": 0.0, "forward": 0.1, "global": 0.0}
        
        # 重新平衡损失权重
        self.LOSS_WEIGHTS.update({
            "physics": 2.0,            # 加强物理约束
            "arc_length": 1.0,
            "initial_condition": 0.3,   # 大幅降低IC权重，避免过度约束原点
            "smoothness": 0.4,          # 增加平滑性帮助处理分支交汇
            "direction": 0.05           # 极低的方向约束权重
        })
        
        # 调整采样和训练参数
        self.BATCH_SIZE = 150           # 增加采样密度
        self.LEARNING_RATE = 5e-4       # 降低学习率增加稳定性
        self.EPOCHS = 10000             # 增加训练轮数
    
    def setup_case3(self):
        """Case3: Hopf分叉幅值流形配置"""
        self.NX, self.NP = 1, 1
        self.S_MAX = 6.0
        self.Y0 = [0.0, 0.0]  # r=0, μ=0
        
        self.DIRECTION_WEIGHTS = {"cosine": 0.3, "forward": 0.4, "global": 0.1}
        self.LOSS_WEIGHTS.update({
            "physics": 1.0,
            "arc_length": 1.0, 
            "initial_condition": 1.0,
            "smoothness": 0.5,
            "direction": 0.3
        })

def set_random_seeds(seed: int = 42):
    """设置随机种子保证可重现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)