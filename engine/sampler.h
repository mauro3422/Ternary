#ifndef SAMPLER_H
#define SAMPLER_H

#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <smmintrin.h>   // SSE4.2

// --- Lookup Table para exp() ---
// Softmax necesita exp(x) para cada logit.
// En CPU sin AVX2, expf() es lenta (~40 ciclos).
// Precalculamos una LUT de 256 entradas para x en [-16, 16].

static float exp_lut[256];
static int exp_lut_initialized = 0;

static void init_exp_lut(void) {
    for (int i = 0; i < 256; i++) {
        float x = -16.0f + (float)i * 32.0f / 255.0f;
        exp_lut[i] = expf(x);
    }
    exp_lut_initialized = 1;
}

static inline float fast_exp(float x) {
    if (!exp_lut_initialized) init_exp_lut();
    if (x < -16.0f) return 0.0f;
    if (x >= 16.0f) return exp_lut[255];
    float idx_f = (x + 16.0f) * 255.0f / 32.0f;
    int idx = (int)idx_f;
    if (idx >= 255) return exp_lut[255];
    float frac = idx_f - (float)idx;
    return exp_lut[idx] + frac * (exp_lut[idx + 1] - exp_lut[idx]);
}

// --- Softmax ---
// Calcula softmax sobre un array de logits, modificándolo in-place
// para que cada elemento quede en [0, 1] sumando 1.
static void softmax(float* logits, int n) {
    float max_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fast_exp(logits[i] - max_val);
        logits[i] = e;
        sum += e;
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        logits[i] *= inv_sum;
    }
}

// --- Sampling ---
// Samplea un token del array de probabilidades
// Si temperature = 0, elige el token más probable (greedy)
// Si temperature > 0, aplica temperatura y samplea

static int sample_token(const float* probs, int n, float temperature) {
    if (temperature < 0.001f || temperature != temperature) {
        // Greedy: elegir el más probable
        int max_idx = 0;
        float max_val = probs[0];
        for (int i = 1; i < n; i++) {
            if (probs[i] > max_val) {
                max_val = probs[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    // Sampling con temperatura
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < n; i++) {
        cumsum += probs[i];
        if (r < cumsum) return i;
    }
    return n - 1;
}

// --- Top-k sampling ---
// Filtra a los k tokens más probables, redistribuye, samplea
static int sample_topk(const float* logits, int n, int k, float temperature) {
    if (k >= n || k <= 0) {
        softmax((float*)logits, n);
        return sample_token(logits, n, temperature);
    }

    // Buscar el threshold del top-k
    // Copiar logits a un array temporal
    float* probs = (float*)malloc(n * sizeof(float));
    if (!probs) return 0;
    memcpy(probs, logits, n * sizeof(float));

    // Ordenar descendente para encontrar el k-ésimo valor
    // (solo necesitamos el threshold, no orden completo)
    float* sorted = (float*)malloc(n * sizeof(float));
    memcpy(sorted, probs, n * sizeof(float));

    // Ordenamiento simple (burbuja parcial)k
    for (int i = 0; i < k; i++) {
        for (int j = i + 1; j < n; j++) {
            if (sorted[j] > sorted[i]) {
                float tmp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = tmp;
            }
        }
    }
    float threshold = sorted[k - 1];
    free(sorted);

    // Poner a -inf los que no están en top-k
    // (softmax los hará ~0)
    for (int i = 0; i < n; i++) {
        if (probs[i] < threshold) {
            probs[i] = -INFINITY;
        }
    }

    // Aplicar temperatura antes de softmax
    if (temperature > 0.001f) {
        for (int i = 0; i < n; i++) {
            if (probs[i] > -INFINITY / 2) {
                probs[i] /= temperature;
            }
        }
    }

    softmax(probs, n);
    int token = sample_token(probs, n, temperature);
    free(probs);
    return token;
}

#endif // SAMPLER_H
