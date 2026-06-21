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
| **1.5** | **Entrenar** | ⏳ Ejecutar en PC con GPU (10-30 min) |
| **1.6** | **Evaluar** | ⏳ Verificar que aprende (loss baja, texto coherente) |
| **1.7** | **Exportar a GGUF** | ⏳ Script que convierte los pesos a `i2_s` (formato ternario GGUF) |

**Dependencias:** PyTorch, numpy (ya instalado)

---

## Fase 2: Motor de Inferencia Custom en C

| Paso | Archivo | Qué |
|---|---|---|
| 2.1 | `engine/tensor.h` | Struct Tensor + operaciones básicas (crear, copiar, free) |
| 2.2 | `engine/gguf.h` | Parser de GGUF (leer pesos del archivo) |
| 2.3 | `engine/ternary.h` | SSE4.2 intrinsics para matmul ternario (`_mm_sign_epi8`) |
| 2.4 | `engine/transformer.h` | Forward pass completo (embed → attn → ffn → head) |
| 2.5 | `engine/sampler.h` | Sampling (top-k, temperatura, softmax) |
| 2.6 | `engine/main.c` | CLI: prompt → generate tokens |
| 2.7 | `engine/Makefile` | Build: `gcc -msse4.2 -O2 main.c -o tern` |

**Target:** Binario < 100KB, 0 dependencias externas

### SSE4.2 intrinsics clave

```c
#include <smmintrin.h>  // SSE4.2

// Multiplicar 16 floats por signo ternario (×1, ×0, ×-1)
__m128 sign = _mm_sign_epi8(valores, mascara_ternaria);

// Sumar 16 floats horizontalmente
__m128 sum = _mm_hadd_ps(a, b);
```

---

## Fase 3: Validación

| Paso | Qué |
|---|---|
| 3.1 | Cargar modelo entrenado (GGUF) en el motor C |
| 3.2 | Verificar que da los mismos outputs que PyTorch |
| 3.3 | Medir tok/s en el N4020 |
| 3.4 | Comparar vs TinyLlama en llama.cpp |

---

## Fase 4: Experimentos (opcional)

| Experimento | Qué |
|---|---|
| **MatMul-Free** | Reemplazar atención por HGRN (gated RNN) — 0 multiplicaciones |
| **MoE ternario** | Varios expertos ternarios chicos, router aprende a elegir |
| **JSON estructurado** | Entrenar en datos con formato para que genere JSON válido |
| **Distillación** | TinyLlama enseña a nuestro modelo (teacher → student) |

---

## Stack

```
Training:    Python + PyTorch + numpy  (~800KB de código nuestro)
Formato:     GGUF (i2_s - 1.58-bit ternary)
Inferencia:  C99 + SSE4.2 intrinsics   (~300 líneas, binario < 100KB)
Build:       Makefile / gcc
```

## Por qué este enfoque y no llama.cpp

| | llama.cpp | Motor custom |
|---|---|---|
| Tamaño binario | ~15MB | **< 100KB** |
| Dependencias | OpenMP, BLAS, etc | **0** |
| Control | Capa abstracta | **Cada byte importa** |
| SSE4.2 | Fallback genérico | **Kernels hechos a mano** |
| Ternario | Soporte parcial | **Nativo: {-1,0,+1}** |
| Curva de aprendizaje | Gigante | **300 líneas, se entiende todo** |

---

*Inicio: Junio 2026*
