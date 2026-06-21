# Guía Práctica: Optimización SSE4.2 para Inferencia de Redes Neuronales

> Target: Intel Celeron N4020 (SSE4.2, 2 cores, sin AVX2, sin FMA)
> Proyecto: Motor ternario {-1,0,+1} en C puro

---

## 1. `_mm_sign_epi8` — El Corazón del Matmul Ternario

### Qué hace

`_mm_sign_epi8(a, b)` toma 2 registros de 16 bytes cada uno y devuelve:

| b[i] | resultado |
|------|-----------|
| > 0  | +a[i]     |
| = 0  | 0         |
| < 0  | -a[i]     |

**Es una multiplicación vectorizada ×{-1,0,+1} en 1 instrucción.**

### Cómo se usa para matmul ternario

```c
#include <tmmintrin.h>  // SSSE3 para _mm_sign_epi8
// Nota: _mm_sign_epi8 es SSSE3, no SSE4.2.
// El N4020 tiene SSSE3, así que funciona.

// Ejemplo: producto punto ternario de 16 elementos
// w = pesos ternarios empaquetados como int8_t {-1,0,+1}
// x = activaciones como int8_t (cuantizadas)
// Queremos: sum(x[i] * w[i]) para i=0..15

int dot_product_ternary_int8(const int8_t* x, const int8_t* w, int n) {
    __m128i acc = _mm_setzero_si128();
    for (int i = 0; i < n; i += 16) {
        __m128i x_vec = _mm_loadu_si128((__m128i*)(x + i));
        __m128i w_vec = _mm_loadu_si128((__m128i*)(w + i));
        __m128i prod = _mm_sign_epi8(x_vec, w_vec);  // x * {-1,0,+1}
        // Suma horizontal
        acc = _mm_add_epi32(acc, _mm_sad_epu8(prod, _mm_setzero_si128()));
    }
    // Extraer suma total
    return _mm_extract_epi32(acc, 0) + _mm_extract_epi32(acc, 2);
}
```

### Alternativa: Int8 con escala

Si las activaciones son int8 con rango [-127, 127], la misma función sirve. El resultado es un producto punto de 16 elementos en ~4 ciclos.

### Patrón para capas completas (output_dim × input_dim)

```c
void ternary_matmul(int8_t* output, const int8_t* input,
                    const int8_t* weights, int M, int N) {
    // M = output_dim, N = input_dim
    for (int i = 0; i < M; i++) {
        int sum = 0;
        for (int j = 0; j < N; j += 16) {
            __m128i x = _mm_loadu_si128((__m128i*)(input + j));
            __m128i w = _mm_loadu_si128((__m128i*)(weights + i*N + j));
            __m128i p = _mm_sign_epi8(x, w);
            sum += _mm_extract_epi32(_mm_sad_epu8(p, _mm_setzero_si128()), 0);
        }
        output[i] = (int8_t)(sum >> 7);  // escala + saturación
    }
}
```

### Limitación importante

`_mm_sign_epi8` trata el 0 como 0, positivo como +1, negativo como -1. Esto es **exactamente** lo que necesitamos para ternarios. Pero el resultado de la multiplicación es int8, se desborda si x[i] es ±128 y w[i] es -1. Usá int8 con rango [-127, 127].

---

## 2. Otras Intrinsics SSE4.2/SSSE3 Útiles

### Reducción horizontal: `_mm_hadd_epi32` (SSSE3)

Suma pares de int32 dentro de un registro:

```c
// Entrada:  [a, b, c, d]
// Salida:   [a+b, c+d, e+f, g+h]  (con 2 registros)
__m128i v = _mm_set_epi32(4, 3, 2, 1);
__m128i sum = _mm_hadd_epi32(v, v);      // [1+2, 3+4, 1+2, 3+4]
sum = _mm_hadd_epi32(sum, sum);           // [1+2+3+4, ...]
int total = _mm_extract_epi32(sum, 0);
```

Mejor que `_mm_sad_epu8` para int32 porque no hay falsas sumas.

### Suma sin signo: `_mm_add_epi32` (SSE2, disponible siempre)

```c
__m128i a = _mm_set_epi32(1, 2, 3, 4);
__m128i b = _mm_set_epi32(5, 6, 7, 8);
__m128i c = _mm_add_epi32(a, b);  // [6, 8, 10, 12]
```

### Multiplicación int32: `_mm_mullo_epi32` (SSE4.1)

```c
// Multiplica 4 int32 a la vez, resultado en 32 bits bajos
__m128i a = _mm_set_epi32(10, 20, 30, 40);
__m128i b = _mm_set_epi32(2, 3, 4, 5);
__m128i c = _mm_mullo_epi32(a, b);  // [20, 60, 120, 200]
```

Esto es **clave para RMSNorm** (multiplicar por la inversa de la varianza).

### Cargar/guardar alineado vs desalineado

```c
// Preferir _mm_load_si128 (alineado a 16 bytes) cuando sea posible
// Si no sabés la alineación, usá _mm_loadu_si128 (penalidad mínima en N4020)
float datos[16] __attribute__((aligned(16)));
__m128 v = _mm_load_ps(datos);      // ✅ rápido
__m128 u = _mm_loadu_ps(datos+1);   // ⚠️ más lento pero funcional
```

### Setzero / Set1 (constantes comunes)

```c
__m128i zero = _mm_setzero_si128();
__m128i ones = _mm_set1_epi32(1);
__m128  fzero = _mm_setzero_ps();
__m128  fone = _mm_set1_ps(1.0f);
```

### Comparación y blend: `_mm_blendv_ps` (SSE4.1)

```c
// Blend condicional: si mask bit = 1, toma de b, si no de a
__m128 mask = _mm_cmplt_ps(a, b);    // a < b ?
__m128 result = _mm_blendv_ps(a, b, mask);  // min(a, b)
```

---

## 3. Softmax Aproximado (Sin Exponenciales)

### Problema

`expf()` en CPU sin AVX2 es carísimo (~40 ciclos por valor). Para un vocabulario de 65 caracteres o ~1000 tokens, necesitamos una alternativa.

### Opción 1: Lookup Table (LUT) con interpolación lineal

```c
static float exp_lut[256];  // pre-calculado

void init_exp_lut() {
    for (int i = 0; i < 256; i++) {
        float x = -16.0f + (float)i * 32.0f / 255.0f;  // rango [-16, 16]
        exp_lut[i] = expf(x);
    }
}

float fast_exp(float x) {
    // Clamp al rango de la LUT
    if (x < -16.0f) return 0.0f;
    if (x > 16.0f) return exp_lut[255];  // ~8.8e6, no va a pasar

    float idx_f = (x + 16.0f) * 255.0f / 32.0f;
    int idx = (int)idx_f;
    float frac = idx_f - (float)idx;
    if (idx >= 255) return exp_lut[255];

    // Interpolación lineal entre LUT[idx] y LUT[idx+1]
    return exp_lut[idx] + frac * (exp_lut[idx+1] - exp_lut[idx]);
}
```

**Precisión:** ~0.1% error, suficiente para sampling.

### Opción 2: Aproximación polinomial (minimax, grado 3-4)

```c
// Aproximación de exp(x) para x <= 0 (softmax opera sobre logits normalizados)
// exp(x) ≈ 1 + x + x²/2 + x³/6  (Taylor, barato)
// Mejor: polinomio minimax para x en [-16, 0]

float fast_exp_poly(float x) {
    if (x < -16.0f) return 0.0f;
    // float_i = floor(x / ln2)  para exp(x) = 2^(x/ln2) = 2^k * exp(f)
    // Pero más simple: polinomio directo para rango acotado

    // Para softmax, solo necesitamos valores > -16
    // exp(x) para x <= 0 se aproxima bien con grado 4
    union { float f; uint32_t i; } u;
    u.i = (uint32_t)(12102203.0f * x + 127 * (1 << 23));  // truco del bit hack
    return u.f;
}
```

### Opción 3: Softmax SSE4.2 completo (sin exp)

Usando la técnica "max-sub-exp-divide" con SSE:

```c
void softmax_sse(float* logits, int n) {
    // 1. Encontrar máximo (para estabilidad numérica)
    __m128 maxv = _mm_loadu_ps(logits);
    for (int i = 4; i < n; i += 4) {
        maxv = _mm_max_ps(maxv, _mm_loadu_ps(logits + i));
    }
    // Reducir: maxv = max(maxv[0], maxv[1], maxv[2], maxv[3])
    maxv = _mm_max_ps(maxv, _mm_shuffle_ps(maxv, maxv, _MM_SHUFFLE(2,3,0,1)));
    maxv = _mm_max_ps(maxv, _mm_shuffle_ps(maxv, maxv, _MM_SHUFFLE(1,0,3,2)));
    float max_val = _mm_cvtss_f32(maxv);

    // 2. Calcular exp(x_i - max) y sumar (con fast_exp)
    __m128 sumv = _mm_setzero_ps();
    float sum = 0.0f;
    __m128 max_v = _mm_set1_ps(max_val);
    for (int i = 0; i < n; i += 4) {
        __m128 x = _mm_loadu_ps(logits + i);
        x = _mm_sub_ps(x, max_v);   // x_i - max
        // fast_exp SSE (simplificado: aproximación lineal + LUT vectorizada)
        sumv = _mm_add_ps(sumv, fast_exp_sse(x));
        _mm_storeu_ps(logits + i, fast_exp_sse(x));
    }
    // Reducir sumv
    float hsum = _mm_cvtss_f32(_mm_hadd_ps(sumv, sumv));
    hsum += _mm_cvtss_f32(_mm_shuffle_ps(sumv, sumv, _MM_SHUFFLE(2,3,0,1)));

    // 3. Dividir cada exp por la suma
    __m128 div = _mm_set1_ps(hsum);
    for (int i = 0; i < n; i += 4) {
        __m128 x = _mm_loadu_ps(logits + i);
        _mm_storeu_ps(logits + i, _mm_div_ps(x, div));
    }
}
```

### Recomendación para vocabulario chico (65 chars)

Usar un loop scalar con LUT de 256 entradas. Cabe en L1, es más simple, y para 65 elementos el overhead SSE no se amortiza.

---

## 4. GELU Aproximado

### GELU exacto

GELU(x) = x · Φ(x) ≈ x · 0.5 · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))

Requiere tanh (carísimo sin FMA) y multiplicaciones float. En CPU sin AVX2, cada GELU cuesta ~100 ciclos.

### Opción 1: Lookup Table (recomendada)

```c
static float gelu_lut[1024];
static float gelu_min = -8.0f, gelu_max = 8.0f;

void init_gelu_lut() {
    for (int i = 0; i < 1024; i++) {
        float x = gelu_min + (float)i * (gelu_max - gelu_min) / 1023.0f;
        gelu_lut[i] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}

float fast_gelu(float x) {
    if (x <= gelu_min) return 0.0f;
    if (x >= gelu_max) return x;
    float idx_f = (x - gelu_min) * 1023.0f / (gelu_max - gelu_min);
    int idx = (int)idx_f;
    float frac = idx_f - (float)idx;
    return gelu_lut[idx] + frac * (gelu_lut[idx+1] - gelu_lut[idx]);
}
```

**Tamaño:** 1024 × 4 bytes = 4KB. Cabe en L1 cache del N4020 (32KB L1 data).

### Opción 2: ReLU (la más barata)

Si el modelo se reentrena con ReLU en vez de GELU (se puede hacer en PyTorch antes de exportar):

```c
float relu(float x) { return x > 0.0f ? x : 0.0f; }
// o con SSE:
__m128 relu_sse(__m128 x) {
    return _mm_max_ps(x, _mm_setzero_ps());
}
```

**Costo:** 1 ciclo (vs ~100 de GELU). El modelo puede necesitar reentrenamiento.

### Opción 3: Aproximación polinomial GELU (grado 3)

```c
float fast_gelu_poly(float x) {
    // GELU(x) ≈ x * sigmoid(1.702f * x)  — aproximación de la literatura
    // sigmoid(y) ≈ 1 / (1 + exp(-y))
    // Más barato: GELU ≈ max(0, x) + 0.044 * min(0, x)  (GELU-approximate)
    float x3 = x * x * x;
    float tanh_arg = 0.79788456f * (x + 0.044715f * x3);
    // tanh(y) ≈ y - y³/3 + 2y⁵/15 - 17y⁷/315  (Taylor, sirve para |y| < 1)
    // Para valores grandes, clamp a ±1
    return 0.5f * x * (1.0f + tanh_arg);
}
```

Esta versión no usa `tanhf()` pero requiere control de rango para que Taylor converja.

### Recomendación

- **Si podés reentrenar:** ReLU (la diferencia en modelos ternarios chicos es mínima)
- **Si no:** LUT de 1024 entradas (4KB, error < 0.1%)

---

## 5. RMSNorm con SSE4.2

### Fórmula

RMSNorm(x) = x / sqrt(mean(x²) + ε) · γ

Donde γ es un vector de escala aprendido (float32).

### Implementación SSE4.2

```c
void rmsnorm_sse(float* output, const float* input, const float* gamma,
                 int n, float eps) {
    // 1. Calcular sum(x_i²) con SSE
    __m128 sum_sq = _mm_setzero_ps();
    for (int i = 0; i < n; i += 4) {
        __m128 x = _mm_loadu_ps(input + i);
        sum_sq = _mm_add_ps(sum_sq, _mm_mul_ps(x, x));
    }
    // Reducir: sum_sq[0] + sum_sq[1] + sum_sq[2] + sum_sq[3]
    __m128 shuf = _mm_shuffle_ps(sum_sq, sum_sq, _MM_SHUFFLE(2,3,0,1));
    sum_sq = _mm_add_ps(sum_sq, shuf);
    shuf = _mm_shuffle_ps(sum_sq, sum_sq, _MM_SHUFFLE(1,0,3,2));
    sum_sq = _mm_add_ps(sum_sq, shuf);  // ahora sum_sq[0] contiene la suma total

    float mean_sq = _mm_cvtss_f32(sum_sq) / (float)n;
    float rms = 1.0f / sqrtf(mean_sq + eps);  // raíz inversa

    // 2. Normalizar: output_i = input_i * rms * gamma_i
    __m128 rms_v = _mm_set1_ps(rms);
    for (int i = 0; i < n; i += 4) {
        __m128 x = _mm_loadu_ps(input + i);
        __m128 g = _mm_loadu_ps(gamma + i);
        __m128 y = _mm_mul_ps(_mm_mul_ps(x, rms_v), g);
        _mm_storeu_ps(output + i, y);
    }
}
```

Para que sea rápido, `n` debe ser múltiplo de 4 (el embedding size del modelo es 384 → múltiplo de 4 ✅).

### Nota sobre el Celeron N4020

Sin FMA (Fused Multiply-Add), cada multiplicación es una instrucción separada. `_mm_mul_ps` tiene latencia 5 ciclos, throughput 1/ciclo. El loop de RMSNorm cuesta ~12n/4 ciclos. Para n=384, eso es ~1152 ciclos ≈ 0.5μs a 2.8GHz. Despreciable contra el matmul.

---

## 6. Optimizaciones de Layout de Memoria

### SoA vs AoS para SSE4.2

| Layout | Descripción | SSE |
|--------|------------|-----|
| **AoS** | `struct {float x,y,z,w;} data[N]` | Malo: hay que desempaquetar |
| **SoA** | `float xs[N], ys[N], zs[N], ws[N]` | Bueno: carga contigua |

Para transformers ternarios usamos **SoA implícito**: pesos son una matriz plana `w[i*N + j]` en orden row-major. Eso está bien porque el acceso es secuencial.

### Alineación a 16 bytes

```c
// Alineación manual (C11)
float* pesos = aligned_alloc(16, M * N * sizeof(float));

// Sin aligned_alloc (C99 portable)
float* pesos;
posix_memalign((void**)&pesos, 16, M * N * sizeof(float));

// Stack (variable local)
float buffer[384] __attribute__((aligned(16)));
```

Beneficio: `_mm_load_ps` (alineado) es ~1-2 ciclos más rápido que `_mm_loadu_ps` (desalineado). En el N4020 la diferencia es menor que en CPUs más viejas, pero suma para loops calientes.

### Prefetching

```c
// Prefetch 64 bytes adelante (para el siguiente batch de 16 int8)
_mm_prefetch((const char*)(input + i + 64), _MM_HINT_T0);
```

En el N4020 con 4MB L2, los pesos ternarios de un modelo 85M (~17MB) no entran en caché. El prefetch ayuda a esconder latencia de RAM.

### Tiling para matmul

El matmul ternario O(M·N) puede dividirse en tiles que quepan en L1/L2:

```c
#define TILE_M 64
#define TILE_N 64

void matmul_tiled(int8_t* C, const int8_t* A, const int8_t* B,
                  int M, int N, int K) {
    for (int i = 0; i < M; i += TILE_M) {
        for (int j = 0; j < N; j += TILE_N) {
            for (int k = 0; k < K; k += 16) {
                // Procesar tile (i..i+TILE_M, j..j+TILE_N)
                // con _mm_sign_epi8 en chunks de 16
            }
        }
    }
}
```

Tile de 64×64 int8 = 4KB → cabe en L1 (32KB). Evita recargar datos de RAM.

### Packing de pesos ternarios

Cada peso ternario necesita solo 2 bits (para representar -1, 0, +1). Se pueden empaquetar 4 pesos por byte:

```c
// Desempaquetado on-the-fly
int8_t unpack_ternary(uint8_t packed, int idx) {
    int shift = idx * 2;
    int val = (packed >> shift) & 0x03;
    // Mapeo: 0→ -1, 1→ 0, 2→+1 (o el que uses)
    static const int8_t map[4] = {-1, 0, 1, 0};
    return map[val];
}
```

**Costo:** Desempaquetar suma ~2-3 ciclos por byte. Si el cuello de botella es RAM (como en el N4020), comprimir 4× los pesos vale la pena aunque cueste un poco de CPU extra.

---

## 7. Proyectos Open Source de Referencia

### ggml (llama.cpp) — El más maduro

- **URL:** https://github.com/ggml-org/ggml
- **Qué tiene:** Backend SSE4.2 completo (no solo ternario, también Q4_0, Q5_0, etc.)
- **Archivos clave:**
  - `src/ggml.c` — kernels SIMD con `GGML_F32_VEC` macros
  - `src/ggml-quants.c` — cuantización + dot product con SSE
- **Cómo lo usamos:** Inspiración para kernels, no para dependencia directa
- `_mm_sign_epi8` lo usan en kernels de dot product int8

### bitnet.cpp (Microsoft) — El más cercano a ternario

- **URL:** https://github.com/microsoft/BitNet
- **Qué tiene:** Inferencia 1.58-bit (ternario) con kernels optimizados
- **Soporte:** ARM NEON, AVX2 (no SSE4.2 puro)
- **Lección:** La técnica de empaquetado 1.58-bit es directamente aplicable

### micrograd-sse (inspiración SSE)

- **URL:** Buscar "micrograd sse" en GitHub (proyectos académicos)
- **Concepto:** micrograd de Karpathy pero con aceleración SSE
- **No hay un repo estándar**, pero la técnica de vectorizar autograd es simple

### tinygrad / Tensor32 (referencia de kernels minimalistas)

- **URL:** https://github.com/tinygrad/tinygrad
- **Tiene:** Backend CPU con SSE en sus kernels JIT
- **Lección:** Cómo estructurar kernels genéricos que funcionan con SSE

### RWKV.cpp

- **URL:** https://github.com/BlinkDL/RWKV-LM
- **Tiene:** Implementación de RWKV (RNN) con ggml = SSE4.2
- **Útil para:** Comparar velocidad RNN vs transformer en el N4020

---

## 8. Optimización para CPUs de 2 Cores (Sin OpenMP)

### Por qué NO OpenMP

- OpenMP añade dependencia de biblioteca (~500KB)
- Para 2 cores, el overhead de threading puede superar la ganancia
- El N4020 tiene 2 cores SIN hyperthreading. No hay SMT.

### Estrategia: Pipeline manual con pthreads (o Win32 threads)

```c
#include <pthread.h>

typedef struct {
    float* output;
    const float* input;
    const float* weights;
    int start, end;  // filas asignadas a este thread
    int N;           // input_dim
} thread_arg_t;

void* worker(void* arg) {
    thread_arg_t* a = (thread_arg_t*)arg;
    for (int i = a->start; i < a->end; i++) {
        // Calcular output[i] = dot(input, weights[i])
        int sum = 0;
        for (int j = 0; j < a->N; j += 16) {
            __m128i x = _mm_loadu_si128((__m128i*)(a->input + j));
            __m128i w = _mm_loadu_si128((__m128i*)(a->weights + i*a->N + j));
            __m128i p = _mm_sign_epi8(x, w);
            sum += _mm_cvtsi128_si32(_mm_sad_epu8(p, _mm_setzero_si128()));
        }
        a->output[i] = (int8_t)sum;
    }
    return NULL;
}

void matmul_2threads(int8_t* output, const int8_t* input,
                     const int8_t* weights, int M, int N) {
    pthread_t threads[2];
    thread_arg_t args[2];
    int mid = M / 2;

    args[0] = (thread_arg_t){output, input, weights, 0, mid, N};
    args[1] = (thread_arg_t){output, input, weights, mid, M, N};

    pthread_create(&threads[0], NULL, worker, &args[0]);
    pthread_create(&threads[1], NULL, worker, &args[1]);
    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);
}
```

### Interleaving manual (sin threads)

Para modelos muy chicos donde el overhead de threads no se amortiza:

```c
// Alternar chunks entre "thread lógico"
for (int i = 0; i < M; i += 2) {
    int sum0 = 0, sum1 = 0;
    for (int j = 0; j < N; j += 16) {
        __m128i x = _mm_loadu_si128((__m128i*)(input + j));
        __m128i w0 = _mm_loadu_si128((__m128i*)(weights + i*N + j));
        __m128i w1 = _mm_loadu_si128((__m128i*)(weights + (i+1)*N + j));
        __m128i p0 = _mm_sign_epi8(x, w0);
        __m128i p1 = _mm_sign_epi8(x, w1);
        sum0 += _mm_cvtsi128_si32(_mm_sad_epu8(p0, _mm_setzero_si128()));
        sum1 += _mm_cvtsi128_si32(_mm_sad_epu8(p1, _mm_setzero_si128()));
    }
    output[i] = sum0;
    output[i+1] = sum1;
}
```

Esto duplica el uso de registros pero aprovecha que `input` ya está cargado en caché. Para M grande, la ganancia es ~1.5× sobre la versión single-thread.

### Pipeline manual con barreras (modelo productor-consumidor)

```c
// Para la atención: 1 thread calcula Q·K^T, el otro hace softmax
// Para el FFN: 1 thread calcula gate, el otro calcula up_proj
```

### Scheduling manual

Para 2 cores en un Celeron:
- **Core 0:** Atención (matmul + softmax)
- **Core 1:** FFN (up + gate + down)
- Se sincronizan al final del bloque
- Usar `volatile` flags o `__sync_synchronize()` (barreras de memoria)

### Recomendación final

- Si el modelo es < 85M param: **no usar threads** (el overhead de pthread_create es ~10μs, comparable a procesar 2-3 tokens)
- Si el modelo es ≥ 85M: **2 threads manuales** (dividir M en 2 mitades)
- Nunca más de 2 threads (el N4020 no se beneficia, solo empeora el thrashing de caché)

---

## Tabla Resumen de Intrinsics por Operación

| Operación | Intrinsic | Extensión | Ciclos | Notas |
|-----------|-----------|-----------|--------|-------|
| x × {-1,0,+1} | `_mm_sign_epi8` | SSSE3 | 1 | ⭐ Estrella del proyecto |
| x + y (int32) | `_mm_add_epi32` | SSE2 | 1 | Loop de acumulación |
| x × y (float) | `_mm_mul_ps` | SSE | 5 | RMSNorm, GELU |
| x × y (int32) | `_mm_mullo_epi32` | SSE4.1 | 5 | Para escalas int32 |
| x < y ? a : b | `_mm_blendv_ps` | SSE4.1 | 2 | Softmax, clamping |
| max(x, y) | `_mm_max_ps` | SSE | 1 | Reducción de logits |
| Suma horizontal | `_mm_hadd_epi32` | SSSE3 | 3 | Reducción de suma |
| Cargar alineado | `_mm_load_ps` | SSE | 1 | Datos aligned(16) |
| Cargar cualquiera | `_mm_loadu_ps` | SSE | 1-2 | Penalidad mínima |

---

## Checklist de Implementación

- [ ] Confirmar que el N4020 soporta SSSE3 (sí, Gemini Lake tiene SSSE3+SSE4.1+SSE4.2)
- [ ] Usar `-msse4.2` en gcc (habilita todo SSE4.x, SSSE3, SSE3, SSE2, SSE)
- [ ] Empaquetar pesos ternarios a 2 bits por peso (4 por byte) para reducir uso de RAM
- [ ] Desempaquetar on-the-fly en el kernel de matmul
- [ ] RMSNorm con `_mm_mul_ps` + `_mm_add_ps` (sin FMA, funciona igual)
- [ ] GELU → ReLU si se reentrena, o LUT de 4KB
- [ ] Softmax → LUT de 256 entradas (rango [-16, 16])
- [ ] 0 dependencias externas (ni siquiera pthreads para modelos chicos)
- [ ] Alinear todos los buffers grandes a 16 bytes
- [ ] Test de velocidad: esperar 30-100 tok/s para modelo 350M ternario
