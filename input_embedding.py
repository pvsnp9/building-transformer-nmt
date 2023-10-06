import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    # d_model: embedding length, vocab_size = number of tokens
    def __init__(self, d_model, vocab_size):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        # Based on papaer, we multiply by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)