from pathlib import Path
import warnings
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import BilingualDataset
from builder import build_transformer
from config import get_cfg, get_model_file_path


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_build_tokenizer(cfg, ds, lang):
    tokenizer_path = Path(cfg['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer =  Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trianer = WordLevelTrainer(min_frequency =2, special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trianer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# finding max seq length
def get_max_seq_len(cfg, ds, tokenizer):
    src_max_seq_len, tgt_max_seq_len = 0, 0
    for item in ds:
        src_max_seq_len = max(src_max_seq_len, len(tokenizer.encode(item["translation"][cfg["lang_src"]]).ids))
        tgt_max_seq_len = max(tgt_max_seq_len, len(tokenizer.encode(item["translation"][cfg["lang_tgt"]]).ids))

    print(f"Source max seq len: {src_max_seq_len}\nTarget max seq len: {tgt_max_seq_len}")
    
    return src_max_seq_len, tgt_max_seq_len

def get_dataset(cfg):
    ds_raw = load_dataset('opus_books', f"{cfg['lang_src']}-{cfg['lang_tgt']}", split="train")
    
    #build tokenizers
    tokenizer_src = get_build_tokenizer(cfg, ds_raw, cfg['lang_src'])
    tokenizer_tgt = get_build_tokenizer(cfg, ds_raw, cfg['lang_tgt'])
    
    #data split train:val -> 90:10
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, cfg["lang_src"], cfg["lang_tgt"], cfg["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, cfg["lang_src"], cfg["lang_tgt"], cfg["seq_len"])
    
    
    src_seq_len, tgt_seq_len = get_max_seq_len(cfg , ds_raw, tokenizer_src)
    
    train_data_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_data_loader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    return train_data_loader, val_data_loader, tokenizer_src, tokenizer_tgt


def get_model(cfg, vocab_src_len, vocab_target_len):
    model = build_transformer(vocab_src_len, vocab_target_len, cfg["seq_len"], cfg["seq_len"], cfg['d_model'])
    return model

def train_model(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    Path(cfg['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_data_loader, val_data_loader, tokenizer_src, tokenizer_tgt = get_dataset(cfg)
    model = get_model(cfg, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # init tensorboard 
    chalk = SummaryWriter(cfg["experiment_name"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    if cfg["preload"]:
        model_filename = get_model_file_path(cfg, cfg["preload"])
        print(f"Preloading the model from {model_filename} \n")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        del state
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    
    # define our custom x axis metric
    wandb.define_metric("global_step")
    # define which metrics will be plotted against it
    wandb.define_metric("validation/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    
    for epoch in range(initial_epoch, cfg["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_data_loader, desc=f"Processing epoch {epoch:02d}")
        
        for batch in batch_iterator:
            encoder_input = batch["enc_input"].to(device) # (batch_size, seq_len)
            encoder_mask = batch["enc_mask"].to(device) # (batch_size, 1, 1, seq_len)
            decoder_input = batch["dec_input"].to(device) # (batch_size, seq_len)
            decoder_mask = batch["dec_mask"].to(device) #(batch_size, 1, seq_len, seq_len)
            
            # train 
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) #(batch_size, seq_len, d_model))
            projection = model.project(decoder_output) # (batch_size, seq_len, target_vocab)
            
            label = batch["label"].to(device) # (batch_size, seq_len)
            
            #projection: (batch_size, seq_len, target_vocab) -> (batch_size * seq_len, target_vocab)
            #label: (batch_size, seq_len) -> (batch_size * seq_len)
            loss = loss_fn(projection.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():.3f}"})
            
            # Log the loss
            wandb.log({'train/loss': loss.item(), 'global_step': global_step})
            
            chalk.add_scalar("train_loss", loss.item(), global_step)
            chalk.flush()
            
            # backprop
            loss.backward()
            
            #update weights 
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            global_step += 1
            
        # validation 
        eval_model(model, val_data_loader, tokenizer_src, tokenizer_tgt, cfg['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step)
        
        #save the model 
        model_filename = get_model_file_path(cfg, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)
        

def eval_model(model, val_ds, src_tokenizer, target_tokenizer, max_len, device, print_msg, global_step, num_examples =3):
    model.eval()    
    count  = 0
    console_width = 120
    
    src_txts = []
    expected = []
    predicted = []
    
    
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 100
    
    with torch.no_grad():
        for batch in val_ds:
            count += 1 
            encoder_input = batch["enc_input"].to(device) 
            encoder_mask = batch["enc_mask"].to(device) 

            assert encoder_input.size(0) == 1, "Validation must have single batch"
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, src_tokenizer, target_tokenizer, max_len, device)
            
            src_txt = batch["src_txt"][0]
            tgt_txt = batch["tgt_txt"][0]
            
            out_txt = target_tokenizer.decode(model_out.detach().cpu().numpy())
            
            src_txts.append(src_txt)
            expected.append(tgt_txt)
            predicted.append(out_txt)
            
            print_msg("=" * console_width)
            print_msg(f"{f'SOURCE: ':>12}{src_txt}")
            print_msg(f"{f'TARGET: ':>12}{tgt_txt}")
            print_msg(f"{f'PREDICTED: ':>12}{out_txt}")
            
            
            if count == num_examples:
                print_msg('-'*console_width)
                break           
           


def greedy_decode(model, src, src_mask, src_tokenizer, tgt_tokenizer, max_len, device):
    sos_idx = tgt_tokenizer.token_to_id('[SOS]')
    eos_idx = tgt_tokenizer.token_to_id('[EOS]')
    
    # precompute the encoder output and re-use it for every token we get from decoder 
    enc_output = model.encode(src, src_mask)
    # init decoder input with the [SOS] token 
    dec_input = torch.empty(1,1).fill_(sos_idx).type_as(src).to(device) # (1,1) (batch, token_id)
    
    # generate next token until the [EOS] or max_len
    while True:
        if dec_input.size(1) == max_len:
            break
        # mask for the target words 
        dec_maask = BilingualDataset.causal_mask(dec_input.size(1)).type_as(src_mask).to(device)
        # compute output of the decoder 
        output = model.decode(enc_output, src_mask, dec_input, dec_maask) #(batch_size, seq_len, d_model)) (1, 1, d_model)
        # get next token 
        logits = model.project(output[:,-1]) #  (batch_size, d_model) => (batch_size, target_vocab) (1,  target_vocab)
        # select the token with max prob from last token in seq
        _, next_token = torch.max(logits, dim=1) # idx, (tensor) token_id
        dec_input = torch.cat([dec_input, torch.empty(1,1).type_as(src).fill_(next_token.item()).to(device)], dim=1)
        
        if next_token == eos_idx: break
        
    return dec_input.squeeze(0)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    cfg = get_cfg()
    cfg['num_epochs'] = 30
    cfg['preload'] = 1

    wandb.init(
        # set the wandb project where this run will be logged
        project="Transformer",
        # track hyperparameters and run metadata
        config=cfg
    )
    train_model(cfg)