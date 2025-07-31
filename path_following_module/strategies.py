import torch
import torch.nn.functional as F
from torch.autograd import grad

class DirectionStrategies:
    @staticmethod
    def l2_penalty(dx_ds, dp_ds, dx_last, dp_last):
        # L2 penalty for direction consistency
        return torch.mean((dx_ds - dx_last)**2 + (dp_ds - dp_last)**2)
    
    @staticmethod
    def cosine_penalty(dx_ds, dp_ds, dx_last, dp_last):
        # Cosine similarity penalty for direction consistency
        u = torch.cat([dx_ds, dp_ds], dim=1)
        v = torch.cat([dx_last.expand_as(dx_ds), dp_last.expand_as(dp_ds)], dim=1)
        u_norm = F.normalize(u, dim=1)
        v_norm = F.normalize(v, dim=1)
        cosine_sim = torch.sum(u_norm * v_norm, dim=1)
        return torch.mean(1 - cosine_sim)
    
    @staticmethod
    def smooth_penalty(dx_ds, dp_ds, s_points):
        # Smoothness penalty for direction derivatives
        try:
            d2x_ds2 = grad(dx_ds, s_points, torch.ones_like(dx_ds), create_graph=True)[0]
            d2p_ds2 = grad(dp_ds, s_points, torch.ones_like(dp_ds), create_graph=True)[0]
            return torch.mean(d2x_ds2**2 + d2p_ds2**2)
        except:
            return torch.tensor(0.0)
    
    @staticmethod
    def global_forward_penalty(dx_ds, dp_ds, global_direction=None):
        # Global forward direction penalty
        if global_direction is None:
            global_direction = torch.tensor([[1.0, 0.0]])
        
        v = torch.cat([dx_ds, dp_ds], dim=1)
        g = global_direction / torch.norm(global_direction)
        projection = torch.sum(v * g, dim=1)
        return torch.mean(torch.relu(-projection))
    
    @staticmethod
    def adaptive_direction_penalty(dx_ds, dp_ds):
        """Adaptive direction penalty based on unit direction vectors."""
        direction_vecs = torch.cat([dx_ds, dp_ds], dim=1)  # [N, 2]
        direction_unit = direction_vecs / (torch.norm(direction_vecs, dim=1, keepdim=True) + 1e-8)

        # compare with previous direction
        v1 = direction_unit[:-1]
        v2 = direction_unit[1:]
        dot_product = torch.sum(v1 * v2, dim=1)  # cos(θ)
        penalty = torch.mean(1 - dot_product)  # the higher the penalty, the less consistent the direction change

        return penalty
    
    @staticmethod
    def forward_consistency_penalty(x_pred, p_pred, dx_ds, dp_ds):
        """ Forward consistency penalty based on the tangent direction."""
        delta = torch.cat([x_pred[1:] - x_pred[:-1], p_pred[1:] - p_pred[:-1]], dim=1)
        tangent = torch.cat([dx_ds[:-1], dp_ds[:-1]], dim=1)

        delta_unit = F.normalize(delta, dim=1)
        tangent_unit = F.normalize(tangent, dim=1)

        cosine_sim = torch.sum(delta_unit * tangent_unit, dim=1)
        penalty = torch.mean(1 - cosine_sim)
        return penalty



# Mapping of strategy names to their corresponding penalty functions
STRATEGY_MAP = {
    'l2': DirectionStrategies.l2_penalty,
    'cosine': DirectionStrategies.cosine_penalty,
    'smooth': DirectionStrategies.smooth_penalty,
    'global_forward': DirectionStrategies.global_forward_penalty,
    'adaptive': DirectionStrategies.adaptive_direction_penalty,
    'forward_consistency': DirectionStrategies.forward_consistency_penalty
}

def get_available_strategies():
    """Return a list of available strategies."""
    return list(STRATEGY_MAP.keys())