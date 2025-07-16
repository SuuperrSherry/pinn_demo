import torch
import torch.nn as nn
from config import Config

class PINN_ArcLength(nn.Module):
    def __init__(self, hidden_layers=Config.HIDDEN_LAYERS, hidden_neurons=Config.HIDDEN_NEURONS):
        super().__init__()
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_neurons, 2)) 
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

        self.log_sigma_physics = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_arc = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_ic = nn.Parameter(torch.tensor(0.0))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, s):
        output = self.network(s)
        return output[:, 0:1], output[:, 1:2]