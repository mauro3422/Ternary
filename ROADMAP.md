# ROADMAP — Motor de Inferencia Ternario para CPU Débil

## Filosofía

Crear un transformer ternario {-1, 0, +1} completo: desde el entrenamiento hasta la inferencia, 100% código propio. Sin depender de frameworks gigantes para correr. Liviano, rápido en CPUs sin AVX2, y entendible línea por línea.

**Target:** Intel Celeron N4020 (SSE4.2, 2 cores, 8GB RAM, sin GPU, sin AVX2)

---

## Fase 1: Training (PC con GPU)

| Paso | Archivo | Qué |
|---|---|---|
| 1.1 | `config.py` | ✅ Hiperparámetros (6L, 6H, 384d) |
| 1.2 | `model.py` | ✅ GPT + BitLinear con STE (Straight-Through Estimator) |
| 1.3 | `train.py` | ✅ Training loop sobre Shakespeare |
| 1.4 | `sample.py` | ✅ Generación de texto con el modelo entrenado |
| **1.5** | **Entrenar** | ✅ Ejecutado en Colab GPU T4 (loss 2.47, PPL 11.80) |
| **1.6** | **Evaluar** | ✅ Loss baja progresivamente, genera texto con estructura |
| **1.7** | **Exportar** | ✅ `export.py` convierte checkpoint a binario ternario (3 MB) |

**Dependencias:** PyTorch, numpy (ya instalado)

### Lecciones aprendidas (Junio 2026)

#### Capacidad de parámetros ternarios
Cada peso ternario almacena solo log₂(3) ≈ 1.58 bits. Para igualar la capacidad de un modelo float32, necesitamos ~20× más parámetros. Por eso escalar de 10.7M → 85M → 350M es clave para que el modelo aprenda bien.

| Modelo | Bits totales | Equiv. float32 | Qué puede aprender |
|---|---|---|---|
| 10.7M ternarios (6L/6H/384d) | 17M bits | ~0.5M params | Patrones básicos, frases cortas |
| 85M ternarios (12L/12H/768d) | 135M bits | ~4M params | Oraciones coherentes |
| 350M ternarios (24L/16H/1024d) | 555M bits | ~17M params | Diálogo simple, texto estructurado |

**Conclusión:** 10.7M es prueba de concepto. Para algo útil hay que escalar. Ternario es una compensación: menos bits por peso, pero los pesos se pueden hacer muy pequeños (1.58 bits) y la inferencia es rapidísima en SSE4.2.

#### Atención cuadrática vs lineal
Para contexto corto (256 tokens), la atención cuadrática O(n²) es rápida y precisa. Para contexto largo conviene HGRN/RWKV (lineal O(n)). Decidimos mantener atención cuadrática con KV cache para la Fase 2, y explorar MatMul-Free (HGRN) en Fase 4.

#### GELU en el motor C
La activación GELU en el MLP usa multiplicaciones de floats. En el motor C se implementó con **lookup table de 1024 entradas** (4KB, cabe en L1 cache). Alternativa futura: ReLU (más barato, requiere reentrenar el modelo).

---

## Fase 2: Motor de Inferencia Custom en C ✅

| Paso | Archivo | Qué |
|---|---|---|
| 2.1 | `engine/tensor.h` | ✅ Struct Tensor + alloc/free + lectura de binario |
| 2.2 | `engine/ternary.h` | ✅ I2_S: pre-desempaqueta ternarios a int8, matmul con `_mm_sign_epi8` |
| 2.3 | `engine/kv_cache.h` | ✅ KV Cache: guarda K,V entre pasos, ~5-10× speedup |
| 2.4 | `engine/sampler.h` | ✅ Softmax con LUT (fast_exp), top-k, temperatura |
| 2.5 | `engine/transformer.h` | ✅ Forward pass + prefill + decode con multi-thread |
| 2.6 | `engine/main.c` | ✅ CLI con -m, -p, -n, -t, -k |
| 2.7 | `engine/Makefile` | ✅ `gcc -msse4.2 -O2 main.c -o tern -lm -lpthread` |
| 2.8 | `export.py` | ✅ Exporta checkpoint de PyTorch a `model.bin` |

**Target:** Binario < 300KB (con pthread estático), dependencias: libc + libm + pthread

### Optimizaciones implementadas

#### I2_S (Int2 with Scale)
En vez de desempaquetar pesos ternarios 2-bit en cada multiplicación (lento), los pre-desempaquetamos a int8 al cargar el modelo. Ocupa 4× más RAM (3MB → 12MB) pero elimina el cuello de botella de desempaquetado, dando ~10× de speedup.

#### KV Cache
Sin KV cache, cada token reprocesa el prompt completo desde cero. Con KV cache:
- **Prefill:** procesa el prompt una sola vez, guarda K y V de cada capa
- **Decode:** solo procesa 1 token nuevo, atención contra cache
- Speedup: ~5-10× para generación larga

#### Multi-threading
El Celeron N4020 tiene 2 cores físicos. Usamos pthreads para partir los matmuls (QKV, MLP) en 2 threads. Speedup: ~1.5-2×.

#### Lookup Tables
- **Softmax:** LUT de 256 entradas para exp() (rango [-16, 16], error < 0.1%)
- **GELU:** LUT de 1024 entradas (4KB, error < 0.1%)

### Bugs encontrados y corregidos

1. **`_mm_sad_epu8` usado como suma:** Esta instrucción suma valores ABSOLUTOS, no valores reales. Corregido usando `_mm_cvtepi8_epi16` + `_mm_madd_epi16`.
2. **Buffer overflow en x_int8:** Se alocaba con `n_embd` (384) pero el MLP fc necesita `4*n_embd` (1536). Causaba HEAP_CORRUPTION.
3. **Residual connection faltante:** Después del attention output projection, faltaba `x = residual + attn_out` y RMSNorm 2. Causaba ACCESS_VIOLATION.
4. **Formato binario con vistas corruptas:** El buffer mmap se corrompía al mezclar vistas. Corregido copiando los datos a tensores propios.
5. **`_mm_sign_epi8` es SSSE3, no SSE4.2:** El include correcto es `<tmmintrin.h>`, no `<smmintrin.h>`. Ambos están disponibles en el N4020.

### Rendimiento (Ryzen 5600G)

| Versión | 50 tokens | Tok/s |
|---|---|---|
| Sin optimizar | 1.92s | 26 |
| Con KV cache + MT | **0.37s** | **136** |

### Rendimiento real en N4020 (modelo 10.7M)

Benchmarks ejecutados el 21 Junio 2026 en el hardware target:

| Métrica | Valor |
|---|---|
| **Carga del modelo** | 0.04s |
| **RAM (peak RSS)** | 16 MB (I2_S pre-desempaquetado) |
| **CPU freq durante test** | 2.69 GHz burst |
| **Prefill 7 chars** | 0.09s |
| **Prefill 50 chars** | 0.47s |
| **Prefill 200 chars** | 1.78s |
| **Decode sostenido** | **~105 tok/s** |
| **Scalabilidad (roofline)** | Ver SPECS.md |

**Roofline analysis:**
- Cuello de botella: **bandwidth de RAM** (~15 GB/s), no compute
- Uso actual de BW: solo ~8% (modelo 10.7M cabe casi en L2)
- Para 85M: ~112 tok/s estimados (más matmul, menos overhead relativo)
- Para 350M: ~27 tok/s estimados (BW-bound puro)
- Margen de optimización: tiling, buffers pre-alocados, solapar memoria ↔ cómputo (~2× potencial)

### SSE4.2 intrinsics clave

```c
#include <tmmintrin.h>  // SSSE3 para _mm_sign_epi8
#include <smmintrin.h>  // SSE4.2 para hadd, blend

// Multiplicar 16 int8 por signo ternario (×1, ×0, ×-1)
__m128i sign = _mm_sign_epi8(valores, mascara_ternaria);

// Sumar 8 enteros horizontalmente
__m128i sum = _mm_hadd_epi32(a, b);
```

### Formato binario (model.bin)

```
MAGIC: "TERN" (4 bytes)
HEADER: vocab_size, block_size, n_layer, n_head, n_embd (20 bytes)
STOI: tabla ASCII 256 bytes (char→token_id)
ITOS: vocab_size bytes (token_id→char)
TENSORS: [size(4)] [data(size)] para cada tensor
  - wte (vocab_size × n_embd, float32)
  - wpe (block_size × n_embd, float32)
  - ln_f (n_embd, float32)
  - por cada layer:
    - ln_1, ln_2 (n_embd, float32)
    - c_attn (3*n_embd × n_embd, packed 2-bit)
    - c_proj (n_embd × n_embd, packed 2-bit)
    - mlp_fc (4*n_embd × n_embd, packed 2-bit)
    - mlp_proj (n_embd × 4*n_embd, packed 2-bit)
  - lm_head (vocab_size × n_embd, packed 2-bit)
```

Formato packed 2-bit: cada byte almacena 4 pesos ternarios. Mapeo: 0→-1, 1→0, 2→+1, 3→reservado.

---

## Fase 3: Validación

| Paso | Qué | Estado |
|---|---|---|
| 3.1 | Cargar modelo en motor C | ✅ `model_load` funciona |
| 3.2 | Generar texto | ✅ Genera texto (aunque sin sentido por modelo chico) |
| 3.3 | Medir tok/s en el N4020 | ✅ **105 tok/s decode** (21 Jun 2026) |
| 3.4 | Comparar vs TinyLlama en llama.cpp | ⏳ Pendiente |

---

## Fase 4: Experimentos (opcional)

| Experimento | Qué |
|---|---|
| **MatMul-Free (HGRN)** | Reemplazar atención + MLP por HGRN (gated RNN). Sin multiplicaciones de matrices, solo sumas/restas. Compite con Transformers hasta ~1B params. O(n) en inferencia. **Recomendado como próximo paso**. |
| **MoE ternario** | Varios expertos ternarios chicos, router aprende a elegir |
| **Dataset TinyStories** | Vocabulario ~1000 tokens, modelo aprende a contar historias |
| **JSON estructurado** | Entrenar en datos con formato para que genere JSON válido |
| **Distillación** | TinyLlama enseña a nuestro modelo (teacher → student) |
| **Trainer custom** | Reemplazar PyTorch con entrenador minimalista en C/puro Python sin autograd pesado |

### Investigación de arquitecturas eficientes (docs/ARQUITECTURAS_CPU.md)

Se investigaron las siguientes arquitecturas para CPU débil:

| Arquitectura | Complejidad | KV Cache | Madurez CPU | Ideal para N4020 |
|---|---|---|---|---|
| **Transformer ternario (actual)** | O(n²) atención | Sí | ✅ Implementado | Con KV cache sí |
| **HGRN (MatMul-Free)** | O(n) lineal | No | Académica | ✅ Excelente |
| **RWKV-7** | O(n) lineal | No | ✅ Producción (rwkv.cpp) | ✅ Bueno |
| **LFM2 (Liquid AI)** | O(n) convolución | No | Producción (propietario) | ❌ Cerrado |
| **Mamba** | O(n) SSM | No | Experimental | ❌ Sin ecosistema CPU |

**Conclusión:** HGRN (MatMul-Free) es la mejor opción para Fase 4 porque:
- Ya tenemos los ternarios implementados
- HGRN es simple de implementar en C (~200 líneas)
- Sin multiplicaciones float
- O(n) en inferencia vs O(n²) de atención
- Benchmarks del paper muestran calidad competitiva con Transformers

---

## Stack

```
Training:      Python + PyTorch + numpy + Google Colab
Formato:       Binario ternario custom (.bin, 1.58-bit)
Inferencia:    C99 + SSSE3/SSE4.2 intrinsics (~1000 líneas)
Build:         Makefile / gcc (MinGW o Linux)
Dependencias:  libm + libpthread
Tamaño:        Binario ~285KB, modelo 10.7M ~3MB empaquetado / ~12MB pre-desempaquetado
```

## Por qué este enfoque y no llama.cpp

| | llama.cpp | Motor custom |
|---|---|---|
| Tamaño binario | ~15MB | **~0.3MB** |
| Dependencias | OpenMP, BLAS, etc | **libm + pthread** |
| Control | Capa abstracta | **Cada byte importa** |
| SSE4.2 | Fallback genérico | **Kernels hechos a mano** |
| Ternario | Soporte parcial | **Nativo: {-1,0,+1}** |
| Curva de aprendizaje | Gigante | **~1000 líneas, se entiende todo** |
| Formato | GGUF | **Binario custom minimalista** |

---

*Inicio: Junio 2026*
*Última actualización: 21 Junio 2026 — Fase 2 completa + benchmarks N4020 (~105 tok/s)*
