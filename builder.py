from trasnformer import Transformer
from multihead_attention import MultiHeadAttention
from input_embedding import InputEmbeddings
from positional_encoding import PositionalEncoding
from feedforward import FeedForward
from encoder_block import EncoderBlock, Encoder
from decoder_block import DecoderBlock, Decoder
from projection import Projection

import torch 
import torch.nn as nn

def build_transformer(src_vocab_size, target_vocab_size, src_seq_len, target_seq_len, d_model =512, N = 6, h=8, dropout=0.1, d_ff=2048) -> Transformer:
    # embeddings 
    src_embeddings = InputEmbeddings(d_model, src_vocab_size)
    target_embeddings = InputEmbeddings(d_model, target_vocab_size)
    
    # positional encoding layers
    src_pos_emb = PositionalEncoding(d_model, src_seq_len, dropout)
    target_pos_emb = PositionalEncoding(d_model, target_seq_len, dropout)
    
    # encider blocks
    encoder_blocks = []
    for _ in range(N):
        enc_attention_block = MultiHeadAttention(d_model, h, dropout)
        ff_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(enc_attention_block, ff_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_attnetion_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        ff_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_attnetion_block, decoder_cross_attention_block, ff_block, dropout)
        decoder_blocks.append(decoder_block)
        
    # create a complete encoder decoder 
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # projection layer
    projection = Projection(d_model, target_vocab_size)
    
    # Transformer 
    transformer = Transformer(encoder, decoder, src_embeddings, target_embeddings, src_pos_emb, target_pos_emb, projection)
    
    # parameter initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    
    return transformer