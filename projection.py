import torch 
import torch.nn as nn

class Projection(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        
        self.projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        #(batch, seq_len, d_model) -> (batch, seq_len, vocan_size)
        return self.projection(x)