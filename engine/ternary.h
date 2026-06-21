#ifndef TERNARY_H
#define TERNARY_H

#include <stdint.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include "tensor.h"

// --- Versión optimizada I2_S (Int2 con Scale) ---
// Los pesos ternarios se pre-desempaquetan a int8 al cargar el modelo.
// Esto ocupa 4x más RAM (3MB → 12MB) pero elimina el desempaquetado
// en cada multiplicación, haciendo el matmul ~10× más rápido.

// Desempaquetar tensor ternario a int8
static tensor_t unpack_ternary_to_i8(const tensor_t* t) {
    tensor_t result = tensor_alloc(t->rows, t->cols, TERN_I8);
    if (!result.data) return result;
    
    int8_t* dst = (int8_t*)result.data;
    for (int i = 0; i < t->rows * t->cols; i++) {
        dst[i] = unpack_ternary_val((uint8_t*)t->data, i);
    }
    return result;
}

// Producto punto int8 × int8 sin desempaquetado
// x: 16 valores int8, w: 16 valores int8 {-1,0,+1}
static inline int32_t dot_i8_16(const int8_t* x, const int8_t* w) {
    __m128i x_vec = _mm_loadu_si128((const __m128i*)(x));
    __m128i w_vec = _mm_loadu_si128((const __m128i*)(w));
    __m128i prod = _mm_sign_epi8(x_vec, w_vec);

    __m128i prod_lo = _mm_cvtepi8_epi16(prod);
    __m128i prod_hi = _mm_unpackhi_epi64(prod, prod);
    prod_hi = _mm_cvtepi8_epi16(prod_hi);

    __m128i ones = _mm_set1_epi16(1);
    __m128i sum32 = _mm_madd_epi16(prod_lo, ones);
    sum32 = _mm_add_epi32(sum32, _mm_madd_epi16(prod_hi, ones));

    sum32 = _mm_hadd_epi32(sum32, sum32);
    sum32 = _mm_hadd_epi32(sum32, sum32);
    return _mm_cvtsi128_si32(sum32);
}

// Producto punto int8 × int8 para N elementos
static inline int32_t dot_i8(const int8_t* x, const int8_t* w, int n) {
    int32_t sum = 0;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        sum += dot_i8_16(x + i, w + i);
    }
    for (; i < n; i++) {
        sum += (int32_t)x[i] * (int32_t)w[i];
    }
    return sum;
}

// Matmul con pesos pre-desempaquetados (I2_S)
static void matmul_i8(int32_t* C, const int8_t* A, const int8_t* B,
                      int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        const int8_t* row_A = A + i * K;
        int32_t* row_C = C + i * N;
        for (int j = 0; j < N; j++) {
            row_C[j] = dot_i8(row_A, B + j * K, K);
        }
    }
}

// --- Funciones legacy (pesos empaquetados) ---
// Se mantienen para referencia pero ya no se usan en el forward

static inline int32_t dot_ternary_16(const int8_t* x, const uint8_t* packed_w, int w_start) {
    int8_t w_unpacked[16];
    for (int i = 0; i < 16; i++) {
        w_unpacked[i] = unpack_ternary_val(packed_w, w_start + i);
    }
    return dot_i8_16(x, w_unpacked);
}

static inline int32_t dot_ternary(const int8_t* x, const uint8_t* packed_w, int w_start, int n) {
    int32_t sum = 0;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        sum += dot_ternary_16(x + i, packed_w, w_start + i);
    }
    for (; i < n; i++) {
        sum += (int32_t)x[i] * (int32_t)unpack_ternary_val(packed_w, w_start + i);
    }
    return sum;
}

// Cuantizar float32 a int8 con escala por fila
static void quantize_row(const float* src, int8_t* dst, int n) {
    float max_val = 0.000001f;
    for (int i = 0; i < n; i++) {
        float abs_v = src[i] > 0 ? src[i] : -src[i];
        if (abs_v > max_val) max_val = abs_v;
    }
    float scale = 127.0f / max_val;
    for (int i = 0; i < n; i++) {
        dst[i] = (int8_t)(src[i] * scale);
    }
}

#endif // TERNARY_H
