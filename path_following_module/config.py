class Config:
    HIDDEN_LAYERS = 3
    HIDDEN_NEURONS = 32
    LEARNING_RATE = 1e-3
    MAX_EPOCHS = 1000
    PRINT_INTERVAL = 100
    TOTAL_LENGTH = 3.0
    BASE_STEP_SIZE = 0.25
    MAX_SEGMENTS = 15
    POINTS_PER_SEGMENT = 40
    MONOTONIC_PENALTY_WEIGHT = 1.0
    MIN_STEP_SIZE = 0.05
    MAX_STEP_SIZE = 0.3
    TORCH_SEED = 123
    NUMPY_SEED = 123

def set_random_seeds(torch_seed=Config.TORCH_SEED, numpy_seed=Config.NUMPY_SEED):
    import torch, numpy as np
    torch.manual_seed(torch_seed)
    np.random.seed(numpy_seed)
