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
    - eval(): usa pesos ternarizados con STE
    - train(): usa pesos float32 (ocupa la mitad de VRAM)
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
        if not self.training:
            # Modo ternarizado con STE (solo en eval/export)
            w_ternary = ternarize(self.weight, self.threshold)
            return F.linear(x, StraightThrough.apply(w_ternary, self.weight), self.bias)
        else:
            # Modo training: usa float32 directo (sin ternarizar)
            # Ahorra ~50% de VRAM porque no crea copias ternarias en el grafo
            return F.linear(x, self.weight, self.bias)

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
# HGRN Block (MatMul-Free)
# Reemplaza atención + MLP por una RNN con compuertas (gated RNN).
# - O(n) en inferencia (vs O(n²) de atención)
# - Forget gate con límite inferior jerárquico por capa
# - Sin multiplicaciones de matrices en el core recurrente
# -----------------------------------------------------------------------------

class HGRNBlock(nn.Module):
    """Hierarchically Gated Recurrent Network block.
    
    En cada paso de tiempo:
      f = sigmoid(W_f @ x + b_f)  # forget gate
      i = sigmoid(W_i @ x + b_i)  # input gate
      h = (1 - f) * h + f * i      # state update
      o = RMSNorm(h)               # output
    
    El forget gate tiene un mínimo (lower_bound) que aumenta por capa,
    haciendo que capas altas recuerden más (contexto largo) y
    capas bajas olviden más (contexto corto).
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_embd = config.n_embd
        
        # Forget gate: decide cuánto mantener del estado anterior
        self.forget = BitLinear(config.n_embd, config.n_embd, bias=True, threshold=config.ternary_threshold)
        # Input gate: decide cuánto del nuevo input agregar
        self.input_gate = BitLinear(config.n_embd, config.n_embd, bias=True, threshold=config.ternary_threshold)
        # Output projection (opcional, mejora calidad)
        self.out_proj = BitLinear(config.n_embd, config.n_embd, bias=False, threshold=config.ternary_threshold)
        
        # Lower bound del forget gate: aumenta con la capa
        # Capa 0: min=0.0 (puede olvidar todo)
        # Capa N-1: min=0.8 (debe recordar casi todo)
        self.lower_bound = 0.8 * layer_idx / max(config.n_layer - 1, 1)
        
        self.norm = nn.RMSNorm(config.n_embd)
    
    def forward(self, x, state=None):
        B, T, C = x.shape
        device = x.device
        
        if state is None:
            state = torch.zeros(B, C, device=device)
        
        outputs = []
        h = state
        for t in range(T):
            xt = x[:, t, :]
            
            if self.training and T > 1:
                # Gradient checkpointing: no guarda activaciones intermedias
                # para ahorrar VRAM (recalcula en el backward)
                f, i = torch.utils.checkpoint.checkpoint(
                    self._compute_gates, xt)
            else:
                f, i = self._compute_gates(xt)
            
            f = torch.clamp(f, min=self.lower_bound)
            
            h = (1 - f) * h + f * i
            o = self.out_proj(self.norm(h))
            outputs.append(o)
        
        return torch.stack(outputs, dim=1), h
    
    def _compute_gates(self, xt):
        """Separa el cómputo de gates para checkpointing."""
        f = torch.sigmoid(self.forget(xt))
        i = torch.sigmoid(self.input_gate(xt))
        return f, i

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

        if config.use_hgrn:
            # HGRN: sin position embedding (es RNN, no necesita posición explícita)
            # Pero añadimos una pequeña para ayudar al modelo
            self.transformer = nn.ModuleDict(dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(0.1),
                h=nn.ModuleList([HGRNBlock(config, i) for i in range(config.n_layer)]),
                ln_f=nn.RMSNorm(config.n_embd),
            ))
        else:
            self.transformer = nn.ModuleDict(dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(0.1),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.RMSNorm(config.n_embd),
            ))
        self.lm_head = BitLinear(config.n_embd, config.vocab_size, bias=False, threshold=config.ternary_threshold)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, BitLinear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, state=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        if self.config.use_hgrn:
            new_state = []
            for i, block in enumerate(self.transformer.h):
                s = state[i] if state is not None and i < len(state) else None
                x, h = block(x, s)
                new_state.append(h)
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss, new_state
        else:
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, state=None):
        if self.config.use_hgrn:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -1:]  # HGRN: solo último token + state
                logits, _, state = self(idx_cond, state=state)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx, state
        else:
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
    
    def init_hgrn_state(self, batch_size, device):
        """Inicializa estado recurrente para HGRN (todos ceros)."""
        return [torch.zeros(batch_size, self.config.n_embd, device=device)
                for _ in range(self.config.n_layer)]
