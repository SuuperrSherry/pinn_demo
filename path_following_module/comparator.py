from pathlib import Path
from typing import Dict, List
import torch
import json
from trainer import PINNTrainer
from visualizer import Visualizer
from config import Config
from bifurcation import analyze_bifurcations


class StrategyComparator:
    def __init__(self, trainer: PINNTrainer, out_dir: str = "assets"):
        self.trainer = trainer
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "models").mkdir(parents=True, exist_ok=True)

    def compare_strategies(self, strategies: List[str], physics_fn, start_x: float, start_p: float, arc_length: float, epochs: int):
        old_epochs = Config.EPOCHS
        Config.EPOCHS = epochs
        results = self.trainer.train(physics_fn, strategies, start_x, start_p, arc_length)
        Config.EPOCHS = old_epochs

        s_eval = torch.linspace(0, arc_length, Config.POINTS_PER_SEGMENT).view(-1, 1)
        packaged = {}
        for strat, (model, history) in results.items():
            x, p = self.trainer.evaluate_path(model, s_eval)
            # bifurcation analysis
            bif = analyze_bifurcations(model, s_eval, self.trainer.device,
                                     tol_F=Config.BIF_TOL_F,
                                     tol_Fx=Config.BIF_TOL_FX,
                                     tol_dx=Config.BIF_TOL_DX)
            print(f"[{strat}] fold candidates detected: {len(bif['points'])}")
            for j, pt in enumerate(bif['points']):
                print(f"    - idx={pt['i']}, s={pt['s']:.4f}, p={pt['p']:.4f}, x={pt['x']:.4f}, Fx={pt['Fx']:.2e}, F={pt['F']:.2e}, stability={pt['stability']}")

            strat_pack = {
                's': s_eval.squeeze().cpu().numpy().tolist(),
                'x': x.cpu().numpy().tolist(),
                'p': p.cpu().numpy().tolist(),
                'loss_history': history,
                'bifurcation': bif,
            }
            packaged[strat] = strat_pack
            torch.save(model.state_dict(), self.out_dir / "models" / f"{strat}.pt")

        with (self.out_dir / "comparison.json").open("w") as f:
            json.dump(packaged, f, indent=2)

        viz = Visualizer(output_dir=str(self.out_dir))
        viz.plot_comparison(packaged)
        viz.plot_loss_comparison(packaged)
        viz.plot_bifurcation_markers(packaged)
        return packaged
