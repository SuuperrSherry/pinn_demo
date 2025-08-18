import torch
import numpy as np


class Config:
    # network
    HIDDEN_LAYERS = 3
    HIDDEN_NEURONS = 32

    # training
    LEARNING_RATE = 1e-3
    EPOCHS = 5000
    POINTS_PER_SEGMENT = 80
    PRINT_INTERVAL = 1000

    # initial conditions (arc-length continuation domain for s ∈ [0, ARC_LENGTH])
    START_X = 1.0   # start on branch x=+1
    START_P = -1.0  # r = -1 so that x^2 + p = 0 holds
    ARC_LENGTH = 5.0

    # initial direction (dx, dp). Choose dp>0 to move toward fold at p→0.
    # Use a gentle upslope so the path grows p while moderating x.
    
    INIT_DIRECTION = torch.tensor([1.0, 2.0])
    INIT_DIRECTION_WEIGHT = 0.1

    # strategy penalty weights (used when AUTO_DIRECTION_WEIGHT=False)
    DIRECTION_PENALTY_WEIGHT = 0.1
    SMOOTH_PENALTY_WEIGHT = 0.05
    GLOBAL_PENALTY_WEIGHT = 0.1
    ADAPTIVE_PENALTY_WEIGHT = 0.1
    FORWARD_CONSISTENCY_WEIGHT = 0.1

    # automatic weighting for direction penalty
    AUTO_DIRECTION_WEIGHT = False  # if True, learn log_sigma_dir; else use fixed weights above

    # physics weighting (prefer strong physics to stay on manifold)
    USE_FIXED_PHYS_WEIGHT = True
    PHYS_FIXED_WEIGHT = 50.0

    # extra constraints for saddle-node manifold tracking
    P_NONPOS_WEIGHT = 1.0   # penalize p>0
    P_MONOTONIC_WEIGHT = 0.5  # encourage dp/ds ≥ margin
    P_MONO_MARGIN = 1e-3

    # seeds
    TORCH_SEED = 42
    NUMPY_SEED = 42

    # device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # bifurcation detection tolerances
    BIF_TOL_F = 1e-2
    BIF_TOL_FX = 1e-2
    BIF_TOL_DX = 1e-2


def set_random_seeds():
    torch.manual_seed(Config.TORCH_SEED)
    torch.cuda.manual_seed_all(Config.TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(Config.NUMPY_SEED)



