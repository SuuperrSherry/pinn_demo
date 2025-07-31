import os
import torch
from typing import Dict, List
from trainer import PINNTrainer
from visualizer import Visualizer
from config import Config
from physics import physics_equation

class StrategyComparator:
    def __init__(self, trainer=None, visualizer=None):
        self.trainer = trainer or PINNTrainer()
        self.visualizer = visualizer or Visualizer()
        os.makedirs("assets", exist_ok=True)
        os.makedirs("assets/models", exist_ok=True)
    
    def compare_strategies(self, strategies: List[str], physics_fn=physics_equation, 
                          start_x=Config.START_X, start_p=Config.START_P, 
                          arc_length=Config.ARC_LENGTH, epochs=Config.EPOCHS):
        """compare different training strategies for PINN."""
        results = {}
        s_eval = torch.linspace(0, arc_length, 300).view(-1, 1)
        
        for strategy in strategies:
            print(f"\n{'='*50}")
            print(f"strategy: {strategy}")
            print(f"{'='*50}")
            
            model, loss_history = self.trainer.train_single_strategy(
                physics_fn, start_x, start_p, arc_length, epochs, 
                direction_strategy=strategy
            )
            
            # save model
            torch.save(model.state_dict(), f'assets/models/model_{strategy}.pt')
            
            x_pred, p_pred = self.trainer.evaluate_path(model, s_eval)
            
            results[strategy] = {
                'model': model,
                'x': x_pred,
                'p': p_pred,
                's': s_eval.squeeze().numpy(),
                'loss_history': loss_history
            }
        
        # visualize results
        self.visualizer.plot_comparison(results)
        self.visualizer.plot_loss_comparison(results)
        
        return results