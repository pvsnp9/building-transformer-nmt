import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, src_tokenizer, tgt_toeknizer, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.src_tokenizer = src_tokenizer
        self.tgt_toeknizer = tgt_toeknizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor([src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
    
    @staticmethod
    def causal_mask(size):
        maks = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
        # mask is all 1 above main daigonal, size (1, size, size)
        return maks == 0
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_tgt_pair = self.ds[idx]
        src_txt = src_tgt_pair["translation"][self.src_lang]
        tgt_txt = src_tgt_pair["translation"][self.tgt_lang]
        
        # convert sentence into tokens -> input IDs
        enc_input_tokens = self.src_tokenizer.encode(src_txt).ids
        dec_input_toekns = self.tgt_toeknizer.encode(tgt_txt).ids
        
        enc_pad_tokens = self.seq_len - len(enc_input_tokens) - 2 #2 [SOS] [EOS]
        dec_pad_tokens = self.seq_len - len(dec_input_toekns) - 1 # [SOS]
        
        assert enc_pad_tokens > 0 or dec_pad_tokens > 0, 'sequence length is smaller than token lengths'
        
        # model inputs: encoder, decoder, and label
        # encoder input format: sos, token_ids, eos, padding to fill the seq_len
        #size(seq_len)
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_pad_tokens, dtype=torch.int64)
        ], dim=0)
        
        # decoder input format: sos, token_ids, padding to fill the seq_len
        #size(seq_len)
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_toekns, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_pad_tokens, dtype=torch.int64)
        ], dim=0)

        #size(seq_len)
        label = torch.cat([
            torch.tensor(dec_input_toekns, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_pad_tokens, dtype=torch.int64)
        ], dim=0)
        
        assert all(ins.size(0) == self.seq_len for ins in (encoder_input, decoder_input, label))
        
        # ecncoder mask: we only mask for padding tokens 
        #decoder mask (causal mask): only looks at previous words, and non padding tokens 
        
        return {
            "enc_input":encoder_input,
            "dec_input": decoder_input,
            "enc_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1, 1, seq_len)
            "dec_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & BilingualDataset.causal_mask(decoder_input.size(0)), #(1,1, seq_len) & #(1, seq_len, seq_len) broadcast
            "label": label,
            "src_txt": src_txt,
            "tgt_txt": tgt_txt
        }
    