import torch
from config import Config, set_random_seeds
from model import StreamlinedPINN
from loss import compute_loss

class PINNTrainer:
    def __init__(self, config=None):
        self.config = config or Config()
        set_random_seeds()
    
    def train_single_strategy(self, physics_fn, start_x, start_p, arc_length, 
                            epochs, direction_strategy=None, **strategy_kwargs):
        """train a single strategy"""
        model = StreamlinedPINN(
            hidden_dim=self.config.HIDDEN_NEURONS,
            hidden_layers=self.config.HIDDEN_LAYERS
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        
        s_train = torch.linspace(0.0, arc_length, self.config.POINTS_PER_SEGMENT).view(-1, 1)
        x0 = torch.tensor([[start_x]], dtype=torch.float32)
        p0 = torch.tensor([[start_p]], dtype=torch.float32)
        s_ic = 0.0
        
        loss_history = []

        # ensure strategy_kwargs is a dictionary
        strategy_kwargs = strategy_kwargs or {}
        if 'init_direction' in strategy_kwargs and strategy_kwargs['init_direction'] is None:
            strategy_kwargs['init_direction'] = self.config.INIT_DIRECTION
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            total_loss, loss_phys, loss_arc, loss_ic = compute_loss(
                model, s_train, x0, p0, s_ic, physics_fn,
                direction_strategy=direction_strategy, **strategy_kwargs
            )
            
            total_loss.backward()
            optimizer.step()
            
            loss_history.append({
                'total': total_loss.item(),
                'physics': loss_phys.item(),
                'arc': loss_arc.item(),
                'ic': loss_ic.item()
            })
            
            if epoch % self.config.PRINT_INTERVAL == 0:
                print(f"[{direction_strategy or 'baseline'}] Epoch {epoch:4d} | "
                      f"Total: {total_loss.item():.6f} | "
                      f"Physics: {loss_phys.item():.6f} | "
                      f"Arc: {loss_arc.item():.6f} | "
                      f"IC: {loss_ic.item():.6f}")
        
        return model, loss_history
    
    def evaluate_path(self, model, s_eval):
        """Evaluate the path given a trained model and evaluation points."""
        model.eval()
        with torch.no_grad():
            x, p = model(s_eval)
        return x.squeeze().numpy(), p.squeeze().numpy()