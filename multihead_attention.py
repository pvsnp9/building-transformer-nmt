import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(q, k, v, mask, dropout:nn.Dropout):
        d_k = q.shape[-1]
        
        #(bathc, h, seq_len, d_k) -> (bathc, h, seq_len, seq_len)
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        #(batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)   
        return (attention_scores @ v), attention_scores
        
    def forward(self, q, k, v, mask):
        #(batch, seq_len, d_model) -> (batch, seq_len, d_model)
        queries = self.w_q(q)
        keyes = self.w_k(k)
        values = self.w_v(v)
        
        # splitting into multi-heaad. 
        # Here we are not splitting the sentence, but the embeddings. The transposition will help each head to interact/see the full sentence 
        #(batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, heads, seq_len, d_k) d_K= splitted_embedding_size
        queries = queries.view(queries.shape[0],queries.shape[1], self.h, self.d_k).transpose(1,2)
        keyes = keyes.view(keyes.shape[0],keyes.shape[1], self.h, self.d_k).transpose(1,2)
        values = values.view(values.shape[0],values.shape[1], self.h, self.d_k).transpose(1,2)
        
        x, self.attention_scores = MultiHeadAttention.attention(queries, keyes, values, mask, self.dropout)
        
        # concatinating the heads 
        #(batch, heads, seq_len, d_k) -> (batch, seq_len, heads, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        #(batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)
        
        
        
        