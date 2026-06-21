"""
Training loop para GPT ternario.
Basado en nanoGPT pero con soporte para BitLinear.
Entrena en CPU o GPU, prepara el modelo para exportar a GGUF.
"""

import os
import sys
import time
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import GPTConfig
from model import GPT, BitLinear

# -----------------------------------------------------------------------------
# Dataset: Shakespeare (character-level, como nanoGPT)
# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor([self.stoi[c] for c in chunk[:-1]], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in chunk[1:]], dtype=torch.long)
        return x, y

def get_shakespeare():
    """Descarga o carga Shakespeare."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    path = os.path.join(os.path.dirname(__file__), "data", "shakespeare.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("Descargando Shakespeare...")
        import urllib.request
        urllib.request.urlretrieve(url, path)
    with open(path, 'r') as f:
        return f.read()

# -----------------------------------------------------------------------------
# Entrenamiento
# -----------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, dataset, config, device):
    """Estima perplexity en eval."""
    model.eval()
    losses = []
    for _ in range(config.eval_iters):
        x, y = dataset[np.random.randint(len(dataset))]
        x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
        out = model(x, y)
        loss = out[1] if isinstance(out, (list, tuple)) else out
        losses.append(loss.item())
    model.train()
    return float(torch.tensor(losses).mean())

def train():
    config = GPTConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    print(f"Modo ternario: {config.ternary}")
    
    # Datos
    text = get_shakespeare()
    dataset = CharDataset(text, config.block_size)
    config.vocab_size = dataset.vocab_size
    print(f"Vocabulario: {dataset.vocab_size} caracteres")
    print(f"Tamaño del dataset: {len(dataset)} muestras")
    
    # Modelo
    model = GPT(config)
    m = model.to(device)
    params = sum(p.numel() for p in m.parameters())
    print(f"Parámetros totales: {params:,}")
    
    # Optimizer con weight decay excluido para capas ternarias
    ternary_params = set(m.get_ternary_params())
    
    decay_params = []
    no_decay_params = []
    for name, p in m.named_parameters():
        if p in ternary_params and config.no_weight_decay_ternary:
            no_decay_params.append(p)
            print(f"  No decay (ternario): {name}")
        elif p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(0.9, 0.95))
    
    # Learning rate scheduler: warmup + cosine decay
    def get_lr(it):
        if it < config.warmup_iters:
            return config.learning_rate * it / config.warmup_iters
        progress = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
        return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    # Training loop
    print("\n=== ENTRENANDO ===")
    best_loss = float('inf')
    t0 = time.time()
    
    for iter_num in range(config.max_iters):
        # LR schedule
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward + backward
        idx = torch.randint(0, len(dataset), (config.batch_size,))
        x_batch = []
        y_batch = []
        for i in idx:
            x, y = dataset[i]
            x_batch.append(x)
            y_batch.append(y)
        x = torch.stack(x_batch).to(device)
        y = torch.stack(y_batch).to(device)
        
        out = m(x, y)
        loss = out[1] if isinstance(out, (list, tuple)) else out
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1:
            t1 = time.time()
            dt = t1 - t0
            loss_val = estimate_loss(m, dataset, config, device)
            tokens_per_sec = config.batch_size * config.block_size / dt
            print(f"iter {iter_num:5d} | loss {loss_val:.4f} | ppl {math.exp(loss_val):.2f} | "
                  f"lr {lr:.2e} | {tokens_per_sec:.0f} tok/s | {dt:.1f}s")
            
            if loss_val < best_loss:
                best_loss = loss_val
                torch.save(m.state_dict(), os.path.join(os.path.dirname(__file__), "checkpoint.pt"))
                print(f"  -> checkpoint guardado (loss {best_loss:.4f})")
            
            t0 = time.time()
    
    print(f"\n=== ENTRENAMIENTO COMPLETO ===")
    print(f"Mejor loss: {best_loss:.4f}")
    
    # Muestra generación de ejemplo
    print("\n=== GENERACIÓN DE EJEMPLO ===")
    m.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    if config.use_hgrn:
        state = m.init_hgrn_state(1, device)
        y, _ = m.generate(context, max_new_tokens=200, temperature=0.8, state=state)
        generated = ''.join([dataset.itos[i] for i in y[0].tolist()])
    else:
        y = m.generate(context, max_new_tokens=200, temperature=0.8)
        generated = ''.join([dataset.itos[i] for i in y[0].tolist()])
    print(generated)
    
    print("\n=== MODELO LISTO ===")
    print(f"Checkpoint: checkpoint.pt")
    print(f"Usá sample.py para generar más texto.")

if __name__ == "__main__":
    train()
