"""
Exporta el modelo ternario entrenado a un binario para el motor C.
Lee checkpoint.pt, ternariza pesos, empaqueta a 2 bits, escribe archivo.
"""

import os
import sys
import struct
import math
import torch
from config import GPTConfig
from model import GPT, ternarize

def pack_ternary(weights):
    """Empaqueta 4 pesos ternarios {-1,0,+1} en 1 byte (2 bits c/u).
       Mapeo: -1→0, 0→1, +1→2, reservado→3
    """
    packed = bytearray()
    for i in range(0, len(weights), 4):
        byte_val = 0
        for j in range(4):
            if i + j < len(weights):
                val = int(weights[i + j])
                if val == -1:
                    bits = 0
                elif val == 0:
                    bits = 1
                elif val == 1:
                    bits = 2
                else:
                    bits = 0
                byte_val |= (bits << (j * 2))
        packed.append(byte_val)
    return bytes(packed)

def export():
    device = 'cpu'
    
    config = GPTConfig()
    
    from train import CharDataset, get_shakespeare
    text = get_shakespeare()
    dataset = CharDataset(text, 1)
    config.vocab_size = dataset.vocab_size
    
    model = GPT(config)
    ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoint.pt')
    if not os.path.exists(ckpt_path):
        print(f"No hay checkpoint en {ckpt_path}")
        return
    
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    is_hgrn = config.use_hgrn
    print(f"Modelo: {'HGRN' if is_hgrn else 'Transformer'}")
    
    # Recolectar todos los pesos ternarizados
    ternary_weights = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and 'BitLinear' in type(module).__name__:
            w = module.weight.detach().cpu().numpy()
            w_ternary = ternarize(torch.from_numpy(w)).numpy()
            ternary_weights[name] = w_ternary
            print(f"  {name}: shape={w_ternary.shape} [{w_ternary.size} params]")
    
    # Embedding (no ternario)
    emb_weight = model.transformer.wte.weight.detach().cpu().numpy().astype('float32')
    pos_emb_weight = model.transformer.wpe.weight.detach().cpu().numpy().astype('float32')
    ln_f_weight = model.transformer.ln_f.weight.detach().cpu().numpy().astype('float32')
    
    out_path = os.path.join(os.path.dirname(__file__), 'engine', 'model.bin')
    
    with open(out_path, 'wb') as f:
        # MAGIC + HEADER
        f.write(b'TERN')
        # header: vocab_size, block_size, n_layer, n_head, n_embd, model_type
        header = struct.pack('IIIIII',
            config.vocab_size, config.block_size,
            config.n_layer, config.n_head, config.n_embd,
            1 if is_hgrn else 0
        )
        f.write(header)
        
        # stoi (256 bytes)
        stoi_bytes = bytearray(256)
        for c, i in dataset.stoi.items():
            stoi_bytes[ord(c)] = i
        f.write(bytes(stoi_bytes))
        
        # itos (vocab_size bytes)
        itos_bytes = bytearray(config.vocab_size)
        for i, c in dataset.itos.items():
            itos_bytes[i] = ord(c)
        f.write(bytes(itos_bytes))
        
        # --- TENSOR DATA ---
        # embedding weights (float32)
        emb_bytes = emb_weight.tobytes()
        f.write(struct.pack('I', len(emb_bytes)))
        f.write(emb_bytes)
        
        # pos embedding (float32)
        pos_emb_bytes = pos_emb_weight.tobytes()
        f.write(struct.pack('I', len(pos_emb_bytes)))
        f.write(pos_emb_bytes)
        
        # ln_f (float32)
        ln_f_bytes = ln_f_weight.tobytes()
        f.write(struct.pack('I', len(ln_f_bytes)))
        f.write(ln_f_bytes)
        
        if is_hgrn:
            # --- HGRN blocks ---
            for i in range(config.n_layer):
                block = model.transformer.h[i]
                
                # norm.weight (RMSNorm gamma, float32)
                norm_bytes = block.norm.weight.detach().cpu().numpy().astype('float32').tobytes()
                f.write(struct.pack('I', len(norm_bytes)))
                f.write(norm_bytes)
                
                # forget.weight (ternary packed)
                for sub in ['forget', 'input_gate', 'out_proj']:
                    w = ternary_weights.get(f'transformer.h.{i}.{sub}')
                    if w is not None:
                        packed = pack_ternary(w.ravel())
                        f.write(struct.pack('II', w.shape[0], w.shape[1]))
                        f.write(struct.pack('I', len(packed)))
                        f.write(packed)
                    else:
                        raise ValueError(f"Weight not found: transformer.h.{i}.{sub}")
                
                # forget.bias, input_gate.bias (float32)
                for sub in ['forget', 'input_gate']:
                    bias = getattr(block, sub).bias.detach().cpu().numpy().astype('float32')
                    bias_bytes = bias.tobytes()
                    f.write(struct.pack('I', len(bias_bytes)))
                    f.write(bias_bytes)
        else:
            # --- Transformer blocks (existing logic) ---
            for i in range(config.n_layer):
                ln1 = model.transformer.h[i].ln_1.weight.detach().cpu().numpy().astype('float32')
                ln2 = model.transformer.h[i].ln_2.weight.detach().cpu().numpy().astype('float32')
                
                ln1_bytes = ln1.tobytes()
                f.write(struct.pack('I', len(ln1_bytes)))
                f.write(ln1_bytes)
                
                ln2_bytes = ln2.tobytes()
                f.write(struct.pack('I', len(ln2_bytes)))
                f.write(ln2_bytes)
                
                for proj_name in ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']:
                    w_name = f'transformer.h.{i}.{proj_name}'
                    w = ternary_weights.get(w_name)
                    if w is not None:
                        packed = pack_ternary(w.ravel())
                        f.write(struct.pack('II', w.shape[0], w.shape[1]))
                        f.write(struct.pack('I', len(packed)))
                        f.write(packed)
                    else:
                        raise ValueError(f"Weight not found: {w_name}")
        
        # lm_head (ternary)
        for w_name in ['lm_head']:
            w = ternary_weights.get(w_name)
            if w is not None:
                packed = pack_ternary(w.ravel())
                f.write(struct.pack('II', w.shape[0], w.shape[1]))
                f.write(struct.pack('I', len(packed)))
                f.write(packed)
    
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nExportado a: {out_path}")
    print(f"Tamaño: {size_mb:.2f} MB")

if __name__ == '__main__':
    export()
