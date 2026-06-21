"""
Config para entrenar 85M HGRN (MatMul-Free) en Shakespeare.
12 layers, 768 embedding — ~8× más parámetros que el 10.7M actual.
"""
from dataclasses import dataclass

@dataclass
class HGRN85MConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 12
    n_head: int = 1        # no usado en HGRN, pero necesario por compatibilidad
    n_embd: int = 768

    ternary: bool = True
    use_hgrn: bool = True

    batch_size: int = 16   # reducir para que quepa en 16GB de RAM de Colab
    learning_rate: float = 3e-4
    max_iters: int = 10000
    eval_interval: int = 500
    eval_iters: int = 100
    warmup_iters: int = 1000

    ternary_threshold: float = 0.7
    no_weight_decay_ternary: bool = True
