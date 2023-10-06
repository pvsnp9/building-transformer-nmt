from pathlib import Path

def get_cfg():
    return {
        "batch_size": 8,
        "num_epochs": 30,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "models",
        "model_basename": "transformer_model_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodels"
    }
    
    
def get_model_file_path(cfg, epoch):
    model_folder = cfg["model_folder"]
    model_basename = cfg["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    
    return str(Path('.')/ model_folder / model_filename)
    