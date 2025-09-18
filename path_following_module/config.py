# config.py
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Config:
    # ===== 维度（默认 2D 系统；Case1/3 可设 NX=1）=====
    NX: int = 2
    NP: int = 1

    # ===== 可视化索引（p-x 图选哪个 x_i / p_j）=====
    X_PLOT_IDX: int = 0
    P_PLOT_IDX: int = 0

    # ===== 网络与训练 =====
    HIDDEN: int = 32
    LAYERS: int = 3
    ACT: str = "tanh"
    LR: float = 1e-3
    STEPS: int = 8000
    BATCH_S: int = 128
    S_MAX: float = 10.0       # Case1 需要 s∈[0,7]
    LOG_EVERY: int = 200
    EXPORT_N: int = 600      # 导出密度更高些，画图平滑

    # AMP（自动混合精度，仅 CUDA 有效）
    AMP: bool = True

    # ===== 初值 y0（长度 NX+NP）=====
    Y0: List[float] = field(default_factory=lambda: [2.0, 0.0])  # Case1: [x0=2, p0=0]

    # ===== 损失权重（Kendall + α 缩放；或静态）=====
    USE_KENDALL: bool = True
    ALPHA: Dict[str, float] = field(default_factory=lambda: {"phys":1.0, "arc":1.0, "ic":1.0, "smooth":0.2, "dir":0.0})
    W_STATIC: Dict[str, float] = field(default_factory=lambda: {"phys":1.0, "arc":0.1, "ic":1.0, "smooth":0.0, "dir":0.0})

    # ===== 方向约束（Case2/3 用；权重=0 等价关闭）=====
    DIR_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "cos": 0.0, "forward": 0.0, "global": 0.0
    })
    DIR_GLOBAL_PARAM_IDX: int = 0
    DIR_GLOBAL_MARGIN: float = 0.0

    # ===== 折叠检测阈值 + 去抖 =====
    EPS_R: float = 1e-4
    EPS_SIGMA: float = 1e-3
    EPS_TAU: float = 1e-3
    DEBOUNCE: int = 5

    # ===== 采样器 =====
    SAMPLER: str = "uniform"  # 预留 "adaptive"

    # ===== 论文绘图风格 =====
    FIG_DPI: int = 300
    FONT_SIZE: int = 12
    LINE_WIDTH: float = 2.0
    MARKER_SIZE: int = 60

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_random_seeds(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)