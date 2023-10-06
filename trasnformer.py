import torch 
import torch.nn as nn
from encoder_block import Encoder
from decoder_block import Decoder
from input_embedding import InputEmbeddings
from positional_encoding import PositionalEncoding
from projection import Projection

class Transformer(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embeddings:InputEmbeddings, target_embeddings:InputEmbeddings, src_pos_emb:PositionalEncoding, 
                 target_pos_emb:PositionalEncoding, proj_layer:Projection) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embeddings = src_embeddings
        self.target_embeddings = target_embeddings
        self.src_pos_embeddings = src_pos_emb
        self.target_post_embeddings = target_pos_emb
        self.projection_layer = proj_layer
        
    def encode(self, src, src_mask):
        src = self.src_embeddings(src)
        src = self.src_pos_embeddings(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, target, target_mask):
        target = self.target_embeddings(target)
        target = self.target_post_embeddings(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
    