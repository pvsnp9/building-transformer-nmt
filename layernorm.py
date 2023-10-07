import torch 
import torch.nn as nn 

class LayerNormalization(nn.Module):
    # epsilon is for numerical stability, prvent div by 0, and GPU repr
    def __init__(self, features, epsilon = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        
    def forward(self, X):
        mean = X.mean(dim=-1, keepdim=True) # (batch, seq_len, 1)
        std = X.std(dim=-1, keepdim=True) # (batch, seq_len, 1)
        return self.alpha * (X - mean) / (std + self.epsilon) + self.bias
    