import torch
import torch.nn as nn
from residual_connection import ResidualConnection
from layernorm import LayerNormalization

class DecoderBlock(nn.Module):
    # attention_block: decoder attention block
    # cross_attenion: k, v from encoder block
    # ff_block
    def __init__(self, features, attention, cross_attention, ff_block, dropout) -> None:
        super().__init__()
        self.attention = attention
        self.cross_attention = cross_attention
        self.ff_block = ff_block
        
        # three residual connection
        self.residual_connection = nn.ModuleList([ ResidualConnection(features, dropout) for _ in range(3)])
        
    # src_mask: mask applied to encoder
    # target_mask: maks applied to decoder. Since it is a translation task, srcmask is for english, and target_mask is for any lang of choice 
    def forward(self, x, encoder_output, src_mask, target_mask):
        # first block in decoder targeted tokens and target masks
        x = self.residual_connection[0](x, lambda x: self.attention(x,x,x, target_mask))
        #second block in decoder, where we get k, v from encoder along with src_mask
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.ff_block)
        
        return x

    
class Decoder(nn.Module):
    def __init__(self, features, layers) -> None:
        super().__init__()
        self.layers = layers
        self.normalization = LayerNormalization(features)
        
    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        
        return self.normalization(x)