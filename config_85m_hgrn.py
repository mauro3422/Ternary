"""
Config para entrenar ~80M HGRN (MatMul-Free) en Shakespeare.
24 layers, 1024 embedding.
"""
from dataclasses import dataclass

@dataclass
class HGRN80MConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 24
    n_head: int = 1
    n_embd: int = 1024

    ternary: bool = True
    use_hgrn: bool = True

    batch_size: int = 8
    learning_rate: float = 3e-4
    max_iters: int = 10000
    eval_interval: int = 500
    eval_iters: int = 100
    warmup_iters: int = 1000

    ternary_threshold: float = 0.7
    no_weight_decay_ternary: bool = True
