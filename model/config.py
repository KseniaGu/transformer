import torch

model_cfg = {
    'input_vocab_size': 28782,
    'output_vocab_size': 28782,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'dropout': 0.2,
    'emb_size': 512,
    'hidden_size': 64,
    'nrof_heads': 8,
    'f_hidden_size': 2048,
    'nrof_layers': 6
}
