import torch
import torch.nn as nn
from config import Config

class StreamlinedPINN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=2, hidden_layers=3):
        super().__init__()
        
        # build the neural network
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.network = nn.Sequential(*layers)
        
        # logarithmic scales for loss weights
        self.log_sigma_physics = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_arc = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_ic = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, s):
        output = self.network(s)
        return output[:, [0]], output[:, [1]]  # x(s), p(s)

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    return model