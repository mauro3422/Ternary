"""
GPT con BitLinear ternario.
Basado en nanoGPT de Andrej Karpathy.
STE (Straight-Through Estimator) para entrenar con pesos {-1, 0, +1}.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GPTConfig

# -----------------------------------------------------------------------------
# BitLinear: capa linear ternaria con STE
# -----------------------------------------------------------------------------

def ternarize(w, threshold=0.7):
    """
    Convierte pesos float32 a ternarios {-1, 0, +1}.
    threshold: fracción de mean(|w|) para el límite (BitNet usa 0.7)
    """
    alpha = threshold * w.abs().mean()  # threshold adaptativo
    return torch.where(
        w > alpha, 1.0,
        torch.where(w < -alpha, -1.0, 0.0)
    )

class BitLinear(nn.Module):
    """
    Capa linear con pesos ternarios {-1, 0, +1}.
    - Forward: usa pesos ternarizados
    - Backward: STE — el gradiente ignora la ternarización
    """
    def __init__(self, in_features, out_features, bias=True, threshold=0.7):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        w_ternary = ternarize(self.weight, self.threshold)
        
        # STE: el gradiente fluye a través de self.weight directamente
        # (PyTorch lo maneja automático porque w_ternary depende de self.weight
        #  pero self.weight no depende de w_ternary en el grafo computacional)
        #
        #  OPCIÓN A: usar .detach() para separar, luego sumar:
        #    w_ste = w_ternary + (self.weight - self.weight.detach())
        #
        #  OPCIÓN B: usar la custom Function (más explícito)
        #
        # Usamos la función custom StraightThrough para más claridad:
        return F.linear(x, StraightThrough.apply(w_ternary, self.weight), self.bias)

class StraightThrough(torch.autograd.Function):
    """
    Función autograd custom que:
    - Forward: devuelve w_ternary (el valor ternarizado)
    - Backward: pasa el gradiente a través de w_real (ignora la ternarización)
    """
    @staticmethod
    def forward(ctx, w_ternary, w_real):
        return w_ternary

    @staticmethod
    def backward(ctx, grad_output):
        # STE: el gradiente fluye a w_real como si w_ternary = w_real
        return None, grad_output  # w_ternary no recibe gradiente, w_real sí

# -----------------------------------------------------------------------------
# Capa de atención multi-cabeza (igual que nanoGPT)
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = BitLinear(config.n_embd, 3 * config.n_embd, bias=False, threshold=config.ternary_threshold)
        self.c_proj = BitLinear(config.n_embd, config.n_embd, bias=False, threshold=config.ternary_threshold)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

# -----------------------------------------------------------------------------
# MLP con BitLinear
# -----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = BitLinear(config.n_embd, 4 * config.n_embd, bias=False, threshold=config.ternary_threshold)
        self.c_proj = BitLinear(4 * config.n_embd, config.n_embd, bias=False, threshold=config.ternary_threshold)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

# -----------------------------------------------------------------------------
# Transformer Block
# -----------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd)  # RMSNorm como en BitNet
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# -----------------------------------------------------------------------------
# GPT con weights ternarios
# -----------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(0.1),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.RMSNorm(config.n_embd),
        ))
        self.lm_head = BitLinear(config.n_embd, config.vocab_size, bias=False, threshold=config.ternary_threshold)

        # Weight tying: compartir pesos entre embedding y lm_head
        self.transformer.wte.weight = self.lm_head.weight

        # Inicialización
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, BitLinear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def get_ternary_params(self):
        """Devuelve los parámetros ternarios (BitLinear) para excluir de weight decay."""
        ternary_params = []
        for name, module in self.named_modules():
            if isinstance(module, BitLinear):
                ternary_params.append(module.weight)
        return ternary_params
