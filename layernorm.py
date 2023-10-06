import torch 
import torch.nn as nn 

class LayerNormalization(nn.Module):
    # epsilon is for numerical stability, prvent div by 0, and GPU repr
    def __init__(self, epsilon = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))
        self.bais = nn.Parameter(torch.ones(1))
        
    def forward(self, X):
        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True)
        return self.alpha * (X - mean) / (std + self.epsilon) + self.bais
    