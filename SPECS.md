# SPECS — PC Chica (Target de Inferencia)

## Hardware

| Componente | Especificación |
|---|---|
| **Modelo** | SUMA C10 (nettop / mini PC) |
| **CPU** | Intel Celeron N4020 @ 1.10GHz (burst 2.8GHz) |
| **Arquitectura** | Gemini Lake Refresh (2019) |
| **Cores/Threads** | 2 cores / 2 threads |
| **Cache** | 4MB L2 |
| **SIMD** | SSE3, SSSE3, SSE4.1, **SSE4.2** |
| **NO SOPORTA** | AVX, AVX2, AVX-512, FMA, F16C, NEON |
| **RAM** | 8 GB total (~3 GB disponible para modelos) |
| **GPU** | Intel UHD Graphics 600 (integradísima, inútil para ML) |
| **Almacenamiento** | SSD 447 GB (SATA) |
| **OS** | Arch Linux |
| **Kernel** | Linux 6.x |

## Límites para Modelos

| Concepto | Valor |
|---|---|
| **RAM usable para modelo** | ~4-5 GB (8GB total - sistema) |
| **Modelo máximo (GGUF Q4)** | ~3B params (~2.5 GB) |
| **Modelo máximo (ternario 1.58-bit)** | ~2B params (~0.4 GB) |
| **Velocidad esperada (transformer Q4)** | 1-4 tok/s |
| **Velocidad esperada (ternario custom)** | 5-15 tok/s (estimado) |
| **Threads óptimos** | 2 (mismos que cores físicos) |
| **Cuello de botella** | Ancho de banda de RAM (~15 GB/s) + sin AVX2 |

## Software Instalado en PC Chica

### Global
- **Ollama** v0.30.10 — en `/usr/local/bin/ollama`
- **llama-server** — compilado con soporte SSE4.2 (libs en `/usr/local/lib/ollama/`)
- **llama.cpp** b9743 — CPU-only, SSE4.2 fallback
- **clang** 22.1.5 — compilador

### Modelos descargados (Ollama)
- `tinyllama:latest` — 637 MB ✅ FUNCIONA

### Proyecto experimental
- **Repo:** `~/dev/ternary-exp/`
- **Frameworks:** PyTorch 2.12.1+cpu, numpy
- **Código:** 577 líneas total (modelo + training + sample + config + roadmap)
- **Tamaño modelo:** 10.7M parámetros (6 layers, 6 heads, 384 embedding)

## Arquitectura del Experimento Ternario

### Modelo (`model.py`)
- GPT basado en nanoGPT de Karpathy
- **BitLinear:** capa linear con pesos ternarios {-1, 0, +1}
- **STE** (Straight-Through Estimator): forward ternario, backward como si fuera float
- **RMSNorm** en vez de LayerNorm (como BitNet paper)
- **Weight tying** entre embedding y lm_head
- **Sin weight decay** en capas ternarias

### Dataset
- Shakespeare (tinyshakespeare, ~1MB, 65 caracteres)
- Descarga automática desde raw.githubusercontent.com

### Forward de BitLinear
```python
def ternarize(w, threshold=0.7):
    alpha = threshold * w.abs().mean()  # threshold adaptativo
    return {-1 si w > alpha, 0 si |w| <= alpha, +1 si w < -alpha}

class StraightThrough(torch.autograd.Function):
    forward:  devuelve w_ternary
    backward: pasa gradiente a w_real (ignora ternarización)
```

## Roadmap Completo

### Fase 1: Training (en PC grande con GPU)
1. Copiar `~/dev/ternary-exp/` a la PC grande
2. Instalar PyTorch con CUDA
3. Entrenar: `python train.py` (~10 min en GPU)
4. Verificar que loss baje (< 2.0) y genere texto coherente
5. Exportar pesos a GGUF (formato i2_s)

### Fase 2: Motor de Inferencia C (en PC chica)
Escribir un motor custom en C puro:

```
engine/
├── tensor.h       → Struct Tensor + alloc/free
├── gguf.h         → Parser de GGUF
├── ternary.h      → SSE4.2 intrinsics para matmul ternario
├── transformer.h  → Forward pass (embed → attn → ffn → head)
├── sampler.h      → Top-k, temperatura, softmax
├── main.c         → CLI prompt → generación
└── Makefile       → gcc -msse4.2 -O2
```

Target: binario < 100KB, 0 dependencias, puras sumas/restas.

### Fase 3: Validación
- Cargar GGUF entrenado → same outputs que PyTorch
- Medir tok/s en N4020
- Comparar vs TinyLlama en llama.cpp

### Fase 4: Experimentos (opcional)
- **MatMul-Free LM:** reemplazar atención por HGRN
- **MoE ternario:** expertos chicos {-1,0,+1}
- **JSON training:** dataset estructurado
- **Distillación:** TinyLlama enseña a nuestro modelo

## Stack Completo

```
Training (PC grande)              Inferencia (PC chica N4020)
─────────────────────             ────────────────────────────
Python + PyTorch + CUDA           C99 + SSE4.2 intrinsics
model.py (BitLinear + STE)        motor custom < 100KB
train.py (Shakespeare)            binario: ./tern
export → GGUF (i2_s)              input: prompt → output: tokens
```

## Referencias

| Concepto | Link |
|---|---|
| nanoGPT (base del modelo) | https://github.com/karpathy/nanoGPT |
| BitNet paper (ternario) | https://arxiv.org/abs/2402.17764 |
| bitnet.cpp (Microsoft) | https://github.com/microsoft/BitNet |
| llama.cpp | https://github.com/ggml-org/llama.cpp |
| SSE4.2 intrinsics | https://software.intel.com/sites/landingpage/IntrinsicsGuide/ |
| tinygrad (alternativa minimal) | https://github.com/tinygrad/tinygrad |
| micrograd (autograd ~200 lines) | https://github.com/karpathy/micrograd |
| MatMul-Free LM | https://arxiv.org/abs/2406.02528 |
| RWKV-7 (RNN rápido en CPU) | https://github.com/BlinkDL/RWKV-LM |
| ROADMAP completo | `ROADMAP.md` |

## Notas para la IA del otro lado

**Objetivo:** Este modelo TIENE QUE CORRER en el Celeron N4020 de arriba. NO hay GPU. NO hay AVX2. Solo SSE4.2. La inferencia tiene que ser en C con intrinsics manuales o en su defecto llama.cpp con backend SSE4.2.

**Decisiones de diseño:**
- Ternario {-1, 0, +1} para evitar multiplicaciones de floats
- SSE4.2 tiene `_mm_sign_epi8` que hace ×{-1,0,+1} para 16 valores a la vez
- RMSNorm (no LayerNorm, como BitNet)
- Sin weight decay en capas ternarias
- Contexto 256 tokens (no más, para mantener velocidad)
- Vocabulario 65 caracteres (Shakespeare) como prueba de concepto

**Si el training en Shakespeare funciona bien, después escalamos a:**
- TinyStories (vocabulario ~1000 tokens)
- Dataset JSON (generación estructurada)
- Modelo más grande (~85M params, 12 layers)

---

*Creado: 20 Junio 2026*
*Próximo paso: Entrenar en PC grande y traer el checkpoint*
