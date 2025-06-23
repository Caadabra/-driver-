"""
AI Models for Car Simulation
Contains neural network architectures for car AI
"""
import torch
import torch.nn as nn


class AdvancedCarAI(nn.Module):
    def __init__(self, input_size=24, hidden_size=256, output_size=6):
        super(AdvancedCarAI, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size),
            nn.Tanh()
        )
        
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        return self.network(x)


class CarAI(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, output_size=4):
        super(CarAI, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)
