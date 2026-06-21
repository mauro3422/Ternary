"""
Generar texto con el modelo ternario entrenado.
Uso: python sample.py [config_name] [prompt]
Ej:  python sample.py config_350m_hgrn "ROMEO:"
"""

import os, sys, torch

if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
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
    text = get_shakespeare()
    dataset = CharDataset(text, 1)
    config = GPTConfig()
    config.vocab_size = dataset.vocab_size
    model = GPT(config)
    
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        print(f"No hay checkpoint en {ckpt_path}")
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    m = model.to(device)
    m.eval()
    
    # Prompt desde argumento o interactivo
    if len(sys.argv) > 2:
        prompt = sys.argv[2]
        context = torch.tensor([[dataset.stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
        with torch.no_grad():
            if config.use_hgrn:
                state = m.init_hgrn_state(1, device)
                y, _ = m.generate(context, max_new_tokens=200, temperature=0.8, state=state)
            else:
                y = m.generate(context, max_new_tokens=200, temperature=0.8)
        generated = ''.join([dataset.itos[i] for i in y[0].tolist()])
        print(generated)
        return

    # Modo interactivo
    print("Prompt: ", end='', flush=True)
    for line in sys.stdin:
        prompt = line.strip()
        if prompt.lower() in ('quit', 'exit', 'q'):
            break
        if not prompt:
            prompt = " "
        context = torch.tensor([[dataset.stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
        with torch.no_grad():
            if config.use_hgrn:
                state = m.init_hgrn_state(1, device)
                y, _ = m.generate(context, max_new_tokens=200, temperature=0.8, state=state)
            else:
                y = m.generate(context, max_new_tokens=200, temperature=0.8)
        generated = ''.join([dataset.itos[i] for i in y[0].tolist()])
        print(generated[len(prompt):])
        print("Prompt: ", end='', flush=True)

if __name__ == "__main__":
    main()
