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

### Lecciones aprendidas (Junio 2026)

#### Capacidad de parámetros ternarios
Cada peso ternario almacena solo log₂(3) ≈ 1.58 bits. Para igualar la capacidad de un modelo float32, necesitamos ~20× más parámetros. Por eso escalar de 10.7M → 85M → 350M es clave para que el modelo aprenda bien.

| Modelo | Bits totales | Equiv. float32 | Qué puede aprender |
|---|---|---|---|
| 10.7M ternarios (6L/6H/384d) | 17M bits | ~0.5M params | Patrones básicos, frases cortas |
| 85M ternarios (12L/12H/768d) | 135M bits | ~4M params | Oraciones coherentes |
| 350M ternarios (24L/16H/1024d) | 555M bits | ~17M params | Diálogo simple, texto estructurado |

**Conclusión:** 10.7M es prueba de concepto. Para algo útil hay que escalar. Ternario es una compensación: menos bits por peso, pero los pesos se pueden hacer muy pequeños (1.58 bits) y la inferencia es rapidísima en SSE4.2.

#### GELU en el motor C
La activación GELU en el MLP (`model.py`) usa multiplicaciones de floats. Para la Fase 2 (motor C), hay que decidir:
- **Opción A:** Lookup table de GELU (pre-calcular valores, ocupa ~1KB)
- **Opción B:** Reemplazar por ReLU (solo `max(0, x)`, pura comparación)
- **Opción C:** MatMul-Free (Fase 4 experimental, elimina atención y MLP por RNN)

Por ahora dejamos GELU. En el motor C usaremos una lookup table.

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
#include <tmmintrin.h>  // SSSE3 para _mm_sign_epi8
#include <smmintrin.h>  // SSE4.2 para hadd, blend

// Multiplicar 16 int8 por signo ternario (×1, ×0, ×-1)
__m128i sign = _mm_sign_epi8(valores, mascara_ternaria);

// Sumar 8 enteros horizontalmente
__m128i sum = _mm_hadd_epi32(a, b);
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
| **MatMul-Free** | Reemplazar atención + MLP por HGRN (gated RNN). Sin multiplicaciones de matrices, solo sumas/restas. No es "más tonto", compite con Transformers hasta ~1B params. Ideal para CPU débil porque inferencia es O(n) en vez de O(n²) |
| **MoE ternario** | Varios expertos ternarios chicos, router aprende a elegir |
| **JSON estructurado** | Entrenar en datos con formato para que genere JSON válido |
| **Distillación** | TinyLlama enseña a nuestro modelo (teacher → student) |
| **Trainer custom** | Reemplazar PyTorch con un entrenador minimalista en C++/Python. Sin autograd pesado, solo el forward/backward necesario para ternarios. Ideal para modelos chicos (<350M). Menos dependencias, más control, posiblemente más rápido por evitar overhead de Python en los loops |

### Fase 5: Entrenador Custom (visión a futuro)

Idea: crear un `tiny_train` en C++ que haga todo el training loop sin PyTorch.

```
tiny_train/
├── tensor.h        → Tensor con autograd manual (solo para ternarios)
├── layers.h        → Solo BitLinear, RMSNorm, atención (lo justo)
├── optim.h         → AdamW simplificado (solo lo necesario para ternarios)
├── data.h          → Loader de datasets chicos (Shakespeare, TinyStories)
├── train.c         → Training loop compacto
└── Makefile        → build en segundos
```

**Ventajas:**
- Binario < 1MB (vs PyTorch ~3GB)
- Sin Python overhead en el forward/backward
- Customizas cada operación (como en inferencia)
- Ideal para experimentos rápidos y modelos chicos
- Misma base de código que el motor de inferencia (reutilizás layers)

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
