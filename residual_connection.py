import torch
import torch.nn as nn
from layernorm import LayerNormalization

class ResidualConnection(nn.Module):
    def __init__(self,features, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(features)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))