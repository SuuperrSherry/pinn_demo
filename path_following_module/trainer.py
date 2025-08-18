# pinn_pathfollowing/trainer.py
import torch
from typing import List, Dict, Tuple
from config import Config, set_random_seeds
from model import StreamlinedPINN
from loss import compute_loss


class PINNTrainer:
    """Manage training for single or multiple direction strategies."""
    def __init__(self):
        set_random_seeds()
        self.device = Config.DEVICE

    def _make_s_grid(self, arc_length: float) -> torch.Tensor:
        return torch.linspace(0, arc_length, Config.POINTS_PER_SEGMENT, device=self.device).view(-1, 1)

    def train(self, physics_fn, strategies: List[str], start_x: float, start_p: float, arc_length: float
              ) -> Dict[str, Tuple[torch.nn.Module, list]]:
        s = self._make_s_grid(arc_length)
        x0 = torch.tensor([[start_x]], device=self.device)
        p0 = torch.tensor([[start_p]], device=self.device)

        results: Dict[str, Tuple[torch.nn.Module, list]] = {}
        for strat in strategies:
            model = StreamlinedPINN(hidden_dim=Config.HIDDEN_NEURONS, hidden_layers=Config.HIDDEN_LAYERS).to(self.device)
            opt = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
            history: list = []
            for epoch in range(Config.EPOCHS):
                opt.zero_grad()
                total, comps = compute_loss(model, s, x0, p0, physics_fn, strategy=strat)
                total.backward()
                opt.step()
                history.append(comps)
                if epoch % Config.PRINT_INTERVAL == 0:
                    print(f"[{strat}] epoch={epoch:04d} | total={comps['total']:.3e} | phys={comps['physics']:.3e} | arc={comps['arc']:.3e} | ic={comps['ic']:.3e} | dir={comps['direction']:.3e}")
            results[strat] = (model, history)
        return results

    def train_single_strategy(self, physics_fn, start_x: float, start_p: float, arc_length: float, epochs: int,
                               direction_strategy: str = None, **strategy_kwargs) -> Tuple[torch.nn.Module, list]:
        # temporarily override loop length (kept for API compatibility)
        old_epochs = Config.EPOCHS
        Config.EPOCHS = epochs
        try:
            results = self.train(physics_fn, [direction_strategy], start_x, start_p, arc_length)
            model, history = results[direction_strategy]
        finally:
            Config.EPOCHS = old_epochs
        return model, history

    def evaluate_path(self, model: torch.nn.Module, s_eval: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        with torch.no_grad():
            x, p = model(s_eval.to(self.device))
        return x.squeeze(), p.squeeze()
