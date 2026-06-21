"""
Configuración del experimento ternario.
Basado en nanoGPT pero simplificado.
"""

from dataclasses import dataclass

@dataclass
class GPTConfig:
    # Arquitectura
    block_size: int = 256       # contexto máximo
    vocab_size: int = 65        # Shakespeare chars
    n_layer: int = 6            # cantidad de transformers layers
    n_head: int = 6             # cabezas de atención
    n_embd: int = 384           # dimensión embedding
    
    # Ternario
    ternary: bool = True        # True = usa BitLinear, False = nn.Linear normal
    
    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    warmup_iters: int = 1000
    
    # BitNet específico
    ternary_threshold: float = 0.7
    no_weight_decay_ternary: bool = True
    
    # Arquitectura
    use_hgrn: bool = False          # True = HGRN (MatMul-Free), False = Transformer
