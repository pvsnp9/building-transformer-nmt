import torch 
import torch.nn as nn 
from multihead_attention import MultiHeadAttention
from residual_connection import ResidualConnection
from feedforward import FeedForward
from layernorm import LayerNormalization


class EncoderBlock(nn.Module):
    def __init__(self, features:int, attention_block:MultiHeadAttention, ff_block:FeedForward, dropout:float) -> None:
        super().__init__()
        
        self.attention_block = attention_block
        self.ff_block = ff_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(2)])
    
    # The purpose of the source mask is to prevent tokens to interact with paddings
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.attention_block(x,x,x, src_mask))
        x = self.residual_connection[1](x, self.ff_block)
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, features, layers):
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization(features)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.layer_norm(x)
        
        
        
        
        
        
        
        
        
        
        
        