"""

 █████  ██         ███    ███  ██████  ██████  ███████ ██      ███████    ██████  ██    ██ 
██   ██ ██         ████  ████ ██    ██ ██   ██ ██      ██      ██         ██   ██  ██  ██  
███████ ██         ██ ████ ██ ██    ██ ██   ██ █████   ██      ███████    ██████    ████   
██   ██ ██         ██  ██  ██ ██    ██ ██   ██ ██      ██           ██    ██         ██    
██   ██ ██ ███████ ██      ██  ██████  ██████  ███████ ███████ ███████ ██ ██         ██    
                                                                                           
                                                                                           

AI models for the driving simulation.
Contains neural network architectures and AI-related functionality.
"""

import torch
import torch.nn as nn


class SimpleCarAI(nn.Module):
    """Simple neural network for controlling a car's movement."""
    
    def __init__(self, input_size=11, hidden_size=32, output_size=2):  # output: accel, steer
        super(SimpleCarAI, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.network(x)