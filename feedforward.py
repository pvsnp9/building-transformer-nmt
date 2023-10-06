import torch 
import torch.nn as nn

class FeedForward(nn.Module):
    #based on paper d_model= 512, d_ff = 2048
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff) #W1, default bias true
        self.dropout = nn.Dropout(dropout) 
        self.l2 = nn.Linear(d_ff, d_model) #W2, default bias true
        
    def forward(self, X):
        #convert (batch, seq_len, d_models) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.l2(self.dropout(torch.relu(self.l1(X))))