#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "tensor.h"

// KV Cache: almacena K y V de cada capa entre pasos de generación
// Así en cada paso solo computamos 1 token nuevo en vez de reprocesar todo
typedef struct {
    int max_T;          // contexto máximo (block_size)
    int n_layer;
    int n_embd;
    int n_kv_heads;     // = n_head (por ahora)
    int head_size;      // n_embd / n_head

    float** k_cache;    // [n_layer][max_T * n_embd]
    float** v_cache;    // [n_layer][max_T * n_embd]
    int current_len;    // cuántos tokens hay cacheados
} kv_cache_t;

// Crear KV cache
static kv_cache_t kv_cache_alloc(int max_T, int n_layer, int n_embd, int n_head) {
    kv_cache_t c;
    c.max_T = max_T;
    c.n_layer = n_layer;
    c.n_embd = n_embd;
    c.n_kv_heads = n_head;
    c.head_size = n_embd / n_head;
    c.current_len = 0;

    c.k_cache = (float**)malloc(n_layer * sizeof(float*));
    c.v_cache = (float**)malloc(n_layer * sizeof(float*));
    if (!c.k_cache || !c.v_cache) {
        fprintf(stderr, "Error: no se pudo crear KV cache\n");
        return c;
    }

    for (int i = 0; i < n_layer; i++) {
        c.k_cache[i] = (float*)calloc(max_T * n_embd, sizeof(float));
        c.v_cache[i] = (float*)calloc(max_T * n_embd, sizeof(float));
        if (!c.k_cache[i] || !c.v_cache[i]) {
            fprintf(stderr, "Error: no se pudo crear KV cache layer %d\n", i);
        }
    }
    return c;
}

// Liberar KV cache
static void kv_cache_free(kv_cache_t* c) {
    if (c->k_cache) {
        for (int i = 0; i < c->n_layer; i++) {
            if (c->k_cache[i]) free(c->k_cache[i]);
            if (c->v_cache[i]) free(c->v_cache[i]);
        }
        free(c->k_cache);
        free(c->v_cache);
        c->k_cache = NULL;
        c->v_cache = NULL;
    }
    c->current_len = 0;
}

// Append K,V de un nuevo token a la cache (para una capa)
static inline void kv_cache_append(kv_cache_t* c, int layer,
                                    const float* k, const float* v) {
    if (c->current_len >= c->max_T) return;
    int off = c->current_len * c->n_embd;
    memcpy(c->k_cache[layer] + off, k, c->n_embd * sizeof(float));
    memcpy(c->v_cache[layer] + off, v, c->n_embd * sizeof(float));
}

// Obtener puntero K cache para una capa (útil para atención)
static inline float* kv_cache_get_k(const kv_cache_t* c, int layer) {
    return c->k_cache[layer];
}

// Obtener puntero V cache para una capa
static inline float* kv_cache_get_v(const kv_cache_t* c, int layer) {
    return c->v_cache[layer];
}

// Resetear cache (para empezar una nueva generación)
static inline void kv_cache_reset(kv_cache_t* c) {
    c->current_len = 0;
}

#endif // KV_CACHE_H
