import torch
import numpy as np

class Config:
    # sturcture parameters
    HIDDEN_LAYERS = 3
    HIDDEN_NEURONS = 32
    
    # training parameters
    LEARNING_RATE = 1e-3
    EPOCHS = 3000
    POINTS_PER_SEGMENT = 80
    PRINT_INTERVAL = 100
    
    # initial conditions
    START_X = 1.0
    START_P = 1.0
    ARC_LENGTH = 7.0

    # initial direction constraint
    INIT_DIRECTION = torch.tensor([0.0, 0.0])  # 可自定义方向向量
    INIT_DIRECTION_WEIGHT = 0.1  # 惩罚强度

    
    # strategy parameters weights
    DIRECTION_PENALTY_WEIGHT = 0.1
    SMOOTH_PENALTY_WEIGHT = 0.05
    GLOBAL_PENALTY_WEIGHT = 0.1
    ADAPTIVE_PENALTY_WEIGHT = 0.1
    FORWARD_CONSISTENCY_WEIGHT = 0.1
    
    # random seed
    TORCH_SEED = 42
    NUMPY_SEED = 42

def set_random_seeds(torch_seed=Config.TORCH_SEED, numpy_seed=Config.NUMPY_SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(torch_seed)
    np.random.seed(numpy_seed)


