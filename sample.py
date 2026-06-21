"""
Generar texto con el modelo ternario entrenado.
Carga el checkpoint y genera texto interactivo.
"""

import os
import sys
import torch

if len(sys.argv) > 1:
    config_module = __import__(sys.argv[1])
else:
    config_module = __import__('config')
GPTConfig = None
for attr in dir(config_module):
    if attr.endswith('Config'):
        GPTConfig = getattr(config_module, attr)
        break

from model import GPT
from train import CharDataset, get_shakespeare

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando: {device}")
    
    # Cargar datos (para el tokenizer)
    text = get_shakespeare()
    dataset = CharDataset(text, 1)  # block_size no importa, solo usamos stoi
    
    # Config y modelo
    config = GPTConfig()
    config.vocab_size = dataset.vocab_size
    model = GPT(config)
    
    # Cargar checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Checkpoint cargado: {ckpt_path}")
    else:
        print(f"No hay checkpoint en {ckpt_path}")
        print("Ejecutá train.py primero")
        return
    
    m = model.to(device)
    m.eval()
    
    # Modo interactivo
    print("\n=== GENERADOR TERNARIO ===")
    print("Escribí un prompt o 'quit' para salir")
    print()
    
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if prompt.lower() in ('quit', 'exit', 'q'):
            break
        
        if not prompt:
            prompt = " "
        
        # Tokenizar prompt
        context = torch.tensor([[dataset.stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
        
        # Generar
        with torch.no_grad():
            if config.use_hgrn:
                state = m.init_hgrn_state(1, device)
                y, _ = m.generate(context, max_new_tokens=200, temperature=0.8, state=state)
            else:
                y = m.generate(context, max_new_tokens=200, temperature=0.8)
        
        # Decodificar
        generated = ''.join([dataset.itos[i] for i in y[0].tolist()])
        print(generated[len(prompt):])
        print()

if __name__ == "__main__":
    main()
