import torch 
import torch.nn as nn
import math

# it only carries the position of token in sentence. The length will be d_mode
# we add the positional encoding values to embedding from input embeddings

class PositionalEncoding(nn.Module):
    # seq_length is max length of sentence 
    def __init__(self, d_model, seq_length, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
        
        # create a mtrix of (seq_length X d_model)
        pos_encoding = torch.zeros(seq_length, d_model)
        #create a vector (seq_length, 1)
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model,2, dtype=float) * (-math.log(10000.0) / d_model))
        #even position sin function
        pos_encoding[:, 0::2] = torch.sin(pos * denominator)
        #odd postion cosine function
        pos_encoding[:,1::2] = torch.cos(pos * denominator)
        
        # during training we will receive batch of sentences. i,e, (btach_size, seq_length, d_model)
        pos_encoding = pos_encoding.unsqueeze(0)
        
        self.register_buffer('pos_encoding', pos_encoding)
    
    def forward(self, X):
        X = X + (self.pos_encoding[:, :X.shape[1], :]).requires_grad_(False)
        return self.dropout(X)