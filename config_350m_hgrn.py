"""
Config para entrenar ~350M HGRN (MatMul-Free) en Shakespeare.
48 layers, 1536 embedding.
"""
from dataclasses import dataclass

@dataclass
class HGRN350MConfig:
    block_size: int = 64
    vocab_size: int = 65
    n_layer: int = 48
    n_head: int = 1
    n_embd: int = 1536

    ternary: bool = True
    use_hgrn: bool = True

    batch_size: int = 1
    learning_rate: float = 2e-4
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
    warmup_iters: int = 500

    ternary_threshold: float = 0.7
    no_weight_decay_ternary: bool = True
