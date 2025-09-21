# config.py
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Config:
    """统一配置管理 - 所有训练参数都在这里修改"""

    # ===== 系统维度 =====
    NX: int = 2                    # x 的维度
    NP: int = 1                    # p 的维度

    # ===== 可视化索引 =====
    X_PLOT_IDX: int = 0
    P_PLOT_IDX: int = 0

    # ===== 网络结构 =====
    HIDDEN_SIZE: int = 32
    NUM_LAYERS: int = 3
    ACTIVATION: str = "tanh"

    # ===== 训练参数 =====
    LEARNING_RATE: float = 1e-3
    EPOCHS: int = 8000
    BATCH_SIZE: int = 128

    # ===== 弧长参数化范围 =====
    S_MAX: float = 10.0

    # ===== 初始条件 (长度应为 NX+NP) =====
    Y0: List[float] = field(default_factory=lambda: [2.0, -4.0])  # [x0, p0] 或 [x0,x1,p0]

    # ===== 损失函数权重 =====
    USE_KENDALL: bool = True
    LOSS_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "physics": 2.0,
        "arc_length": 1.0,
        "initial_condition": 1.0,
        "smoothness": 0.2,
        "direction": 0.5,
        # 仅 2D case2 会启用该项；1D/其它保持 0 不生效
        "embed_collapse": 0.0,
    })

    # ===== 方向约束配置 =====
    DIRECTION_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "cosine": 0.0,
        "forward": 0.0,
        "global": 0.5,
    })
    DIRECTION_GLOBAL_PARAM_IDX: int = 0
    DIRECTION_GLOBAL_MARGIN: float = 1e-5

    # ===== 分叉检测阈值 =====
    BIFURCATION_EPS_R: float = 1e-4
    BIFURCATION_EPS_SIGMA: float = 1e-3
    BIFURCATION_EPS_TAU: float = 1e-3
    BIFURCATION_DEBOUNCE: int = 5

    # ===== 采样策略 =====
    SAMPLING_STRATEGY: str = "adaptive"
    ADAPTIVE_GRID_SIZE: int = 256
    ADAPTIVE_SCORE_TYPE: str = "res"
    ADAPTIVE_MIX_RATIO: float = 0.5
    ADAPTIVE_TEMPERATURE: float = 0.5
    ADAPTIVE_WARMUP_ITERS: int = 800
    ADAPTIVE_UPDATE_EVERY: int = 200

    # ===== 训练控制 =====
    USE_AMP: bool = True
    GRADIENT_CLIP_NORM: float = 1.0
    LR_STEP_SIZE: int = 2000
    LR_GAMMA: float = 0.3

    # ===== 日志和输出 =====
    LOG_EVERY: int = 2000
    EXPORT_POINTS: int = 600
    OUTPUT_DIR: str = "assets"

    # ===== 图表设置 =====
    FIG_DPI: int = 300
    FONT_SIZE: int = 12
    LINE_WIDTH: float = 2.0
    MARKER_SIZE: int = 60

    # ===== 设备设置 =====
    DEVICE: torch.device = field(default_factory=lambda:
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ===================== 预置场景 =====================

    def setup_case1(self, use_direction: bool = True):
        """Case1: 鞍节点分叉"""
        self.NX, self.NP = 1, 1
        self.S_MAX = 10.0
        self.Y0 = [2.0, -4.0]

        if use_direction:
            self.DIRECTION_WEIGHTS = {"cosine": 0.2, "forward": 0.15, "global": 0.2}
            self.LOSS_WEIGHTS["direction"] = 0.2
        else:
            self.DIRECTION_WEIGHTS = {"cosine": 0.0, "forward": 0.0, "global": 0.0}
            self.LOSS_WEIGHTS["direction"] = 0.0

        self.LOSS_WEIGHTS.update({
            "physics": 2.0,
            "arc_length": 1.0,
            "initial_condition": 1.0,
            "smoothness": 0.2,
            "embed_collapse": 0.0,  # 1D 不需要
        })

    def setup_case2(self):
        """Case2: 跨临界分叉（2D 嵌入，推荐）"""
        self.NX, self.NP = 2, 1          # x=[x0,x1]，p=[p]
        self.S_MAX = 9.0
        self.EXPORT_POINTS = 2500        # 导出更密，0 附近更连续
        self.Y0 = [0.05, 0.0, 0.05]      # x=[0.05,0], p=0.05

        # 方向约束：略推前进，帮助跨过 p=0
        self.DIRECTION_WEIGHTS = {"cosine": 0.0, "forward": 0.10, "global": 0.0}

        # 轻微压 2D 的第 2 个嵌入维（防止蓝线“漂起”）
        self.LOSS_WEIGHTS.update({
            "physics": 5.0,
            "arc_length": 1.0,
            "initial_condition": 0.3,
            "smoothness": 0.1,
            "direction": 0.05,
            "embed_collapse": 0.10,     # 仅 2D 生效
        })

        self.BATCH_SIZE = 150
        self.LEARNING_RATE = 5e-4
        self.EPOCHS = 8000

    def setup_case2_1d(self, branch: str = "x0"):
        """
        Case2：1D 版本（可把两条直线分别学出来）
        branch="x0" -> 追 x=0 分支；"xp" -> 追 x=p 分支
        """
        self.NX, self.NP = 1, 1
        self.S_MAX = 9.0
        self.EXPORT_POINTS = 2500

        if branch == "x0":
            self.Y0 = [0.0, -3.0]   # x=0, p=-3
        else:
            self.Y0 = [0.3, 0.3]    # x=0.3, p=0.3

        self.DIRECTION_WEIGHTS = {"cosine": 0.0, "forward": 0.08, "global": 0.0}
        self.LOSS_WEIGHTS.update({
            "physics": 6.0,
            "arc_length": 1.0,
            "initial_condition": 0.5,
            "smoothness": 0.1,
            "direction": 0.08,
            "embed_collapse": 0.0,  # 1D 无此项
        })

        self.BATCH_SIZE = 150
        self.LEARNING_RATE = 5e-4
        self.EPOCHS = 8000

    def setup_case3(self):
        """Case3: Hopf 分叉幅值流形"""
        self.NX, self.NP = 1, 1
        self.S_MAX = 6.0
        self.Y0 = [0.0, 0.0]
        self.DIRECTION_WEIGHTS = {"cosine": 0.3, "forward": 0.4, "global": 0.1}
        self.LOSS_WEIGHTS.update({
            "physics": 1.0,
            "arc_length": 1.0,
            "initial_condition": 1.0,
            "smoothness": 0.5,
            "direction": 0.3,
            "embed_collapse": 0.0,  # 1D 不需要
        })

def set_random_seeds(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
