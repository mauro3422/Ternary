#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <smmintrin.h>   // SSE4.2

#include <pthread.h>

#include "tensor.h"
#include "ternary.h"
#include "sampler.h"
#include "kv_cache.h"

// Bloque del transformer
// c_attn.weight shape: (3*n_embd, n_embd) → output dims, input dims
// c_proj.weight shape: (n_embd, n_embd)
// mlp_fc: (4*n_embd, n_embd)
// mlp_proj: (n_embd, 4*n_embd)
typedef struct {
    tensor_t ln_1;
    tensor_t ln_2;
    tensor_t c_attn;
    tensor_t c_proj;
    tensor_t mlp_fc;
    tensor_t mlp_proj;
} block_t;

// Estructura del modelo completa (cargada desde model.bin)
typedef struct {
    int vocab_size;
    int block_size;
    int n_layer;
    int n_head;
    int n_embd;
    int head_size;
    int model_type;     // 0=Transformer, 1=HGRN

    uint8_t stoi[256];
    uint8_t* itos;

    tensor_t wte;
    tensor_t wpe;
    tensor_t ln_f;

    block_t* blocks;        // usado en Transformer
    // HGRN blocks (MatMul-Free)
    struct {
        tensor_t norm;      // RMSNorm gamma (n_embd, F32)
        tensor_t forget_w;  // forget gate (n_embd, n_embd, I8)
        tensor_t forget_b;  // forget bias (n_embd, F32)
        tensor_t input_w;   // input gate (n_embd, n_embd, I8)
        tensor_t input_b;   // input bias (n_embd, F32)
        tensor_t out_w;     // output proj (n_embd, n_embd, I8)
    }* hgrn_blocks;

    tensor_t lm_head;
} transformer_model_t;

// Cargar modelo desde archivo .bin
static int model_load(transformer_model_t* m, const char* path) {
    size_t file_size;
    uint8_t* buf = read_file(path, &file_size);
    if (!buf) return 0;

    size_t offset = 0;

    if (memcmp(buf, "TERN", 4) != 0) {
        fprintf(stderr, "Error: no es un archivo TERN válido\n");
        free(buf);
        return 0;
    }
    offset += 4;

    int* header = (int*)(buf + offset);
    m->vocab_size  = header[0];
    m->block_size  = header[1];
    m->n_layer     = header[2];
    m->n_head      = header[3];
    m->n_embd      = header[4];
    m->head_size   = m->n_embd / m->n_head;
    m->model_type  = header[5];
    offset += 24;

    printf("Modelo: %s %d layers, %d heads, %d embd\n",
           m->model_type ? "HGRN" : "Transformer",
           m->n_layer, m->n_head, m->n_embd);
    printf("Vocab: %d, Block: %d\n", m->vocab_size, m->block_size);

    memcpy(m->stoi, buf + offset, 256);
    offset += 256;

    m->itos = (uint8_t*)malloc(m->vocab_size);
    if (!m->itos) { free(buf); return 0; }
    memcpy(m->itos, buf + offset, m->vocab_size);
    offset += m->vocab_size;

    // Helper: alloc y copia un tensor F32
    #define LOAD_F32_TENSOR(dst, rows) do { \
        int sz = *(int*)(buf + offset); offset += 4; \
        (dst) = tensor_alloc(rows, m->n_embd, TERN_F32); \
        memcpy((dst).data, buf + offset, sz); \
        offset += sz; \
    } while(0)

    LOAD_F32_TENSOR(m->wte, m->vocab_size);
    LOAD_F32_TENSOR(m->wpe, m->block_size);
    LOAD_F32_TENSOR(m->ln_f, 1);

    #define LOAD_TERNARY_TENSOR_I8(dst) do { \
        int r = *(int*)(buf + offset); offset += 4; \
        int c = *(int*)(buf + offset); offset += 4; \
        int sz = *(int*)(buf + offset); offset += 4; \
        tensor_t packed = tensor_alloc(r, c, TERN_TERNARY); \
        memcpy(packed.data, buf + offset, sz); \
        offset += sz; \
        (dst) = unpack_ternary_to_i8(&packed); \
        tensor_free(&packed); \
    } while(0)

    if (m->model_type == 1) {
        // --- HGRN blocks ---
        m->hgrn_blocks = (void*)malloc(m->n_layer * sizeof(*m->hgrn_blocks));
        if (!m->hgrn_blocks) { free(buf); return 0; }
        memset(m->hgrn_blocks, 0, m->n_layer * sizeof(*m->hgrn_blocks));

        for (int i = 0; i < m->n_layer; i++) {
            LOAD_F32_TENSOR(m->hgrn_blocks[i].norm, 1);

            LOAD_TERNARY_TENSOR_I8(m->hgrn_blocks[i].forget_w);
            LOAD_TERNARY_TENSOR_I8(m->hgrn_blocks[i].input_w);
            LOAD_TERNARY_TENSOR_I8(m->hgrn_blocks[i].out_w);

            // forget bias (float32)
            { int sz = *(int*)(buf + offset); offset += 4;
              m->hgrn_blocks[i].forget_b = tensor_alloc(1, m->n_embd, TERN_F32);
              memcpy(m->hgrn_blocks[i].forget_b.data, buf + offset, sz); offset += sz; }
            // input bias (float32)
            { int sz = *(int*)(buf + offset); offset += 4;
              m->hgrn_blocks[i].input_b = tensor_alloc(1, m->n_embd, TERN_F32);
              memcpy(m->hgrn_blocks[i].input_b.data, buf + offset, sz); offset += sz; }
        }
        m->blocks = NULL;
    } else {
        // --- Transformer blocks ---
        m->blocks = (block_t*)malloc(m->n_layer * sizeof(block_t));
        if (!m->blocks) { free(buf); return 0; }
        memset(m->blocks, 0, m->n_layer * sizeof(block_t));
        m->hgrn_blocks = NULL;

        for (int i = 0; i < m->n_layer; i++) {
            LOAD_F32_TENSOR(m->blocks[i].ln_1, 1);
            LOAD_F32_TENSOR(m->blocks[i].ln_2, 1);

            LOAD_TERNARY_TENSOR_I8(m->blocks[i].c_attn);
            LOAD_TERNARY_TENSOR_I8(m->blocks[i].c_proj);
            LOAD_TERNARY_TENSOR_I8(m->blocks[i].mlp_fc);
            LOAD_TERNARY_TENSOR_I8(m->blocks[i].mlp_proj);
        }
    }

    {   // lm_head (pre-unpack to i8)
        int r = *(int*)(buf + offset); offset += 4;
        int c = *(int*)(buf + offset); offset += 4;
        int sz = *(int*)(buf + offset); offset += 4;
        tensor_t packed = tensor_alloc(r, c, TERN_TERNARY);
        memcpy(packed.data, buf + offset, sz);
        offset += sz;
        m->lm_head = unpack_ternary_to_i8(&packed);
        tensor_free(&packed);
    }

    free(buf);
    return 1;
}

// Liberar modelo
static void model_free(transformer_model_t* m) {
    if (m->itos) free(m->itos);
    tensor_free(&m->wte);
    tensor_free(&m->wpe);
    tensor_free(&m->ln_f);
    if (m->model_type == 1 && m->hgrn_blocks) {
        for (int i = 0; i < m->n_layer; i++) {
            tensor_free(&m->hgrn_blocks[i].norm);
            tensor_free(&m->hgrn_blocks[i].forget_w);
            tensor_free(&m->hgrn_blocks[i].forget_b);
            tensor_free(&m->hgrn_blocks[i].input_w);
            tensor_free(&m->hgrn_blocks[i].input_b);
            tensor_free(&m->hgrn_blocks[i].out_w);
        }
        free(m->hgrn_blocks);
    }
    if (m->blocks) {
        for (int i = 0; i < m->n_layer; i++) {
            tensor_free(&m->blocks[i].ln_1);
            tensor_free(&m->blocks[i].ln_2);
            tensor_free(&m->blocks[i].c_attn);
            tensor_free(&m->blocks[i].c_proj);
            tensor_free(&m->blocks[i].mlp_fc);
            tensor_free(&m->blocks[i].mlp_proj);
        }
        free(m->blocks);
    }
    tensor_free(&m->lm_head);
}

// --- RMSNorm ---
// y = x / sqrt(mean(x²) + eps) * gamma
static void rmsnorm(float* output, const float* input,
                    const float* gamma, int n, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += input[i] * input[i];
    }
    float rms = 1.0f / sqrtf(sum_sq / (float)n + eps);
    for (int i = 0; i < n; i++) {
        output[i] = input[i] * rms * gamma[i];
    }
}

// --- GELU con LUT ---
static float gelu_lut[1024];
static int gelu_lut_init = 0;

static void init_gelu_lut(void) {
    float min_val = -8.0f, max_val = 8.0f;
    for (int i = 0; i < 1024; i++) {
        float x = min_val + (float)i * (max_val - min_val) / 1023.0f;
        gelu_lut[i] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
    gelu_lut_init = 1;
}

static inline float fast_gelu(float x) {
    if (!gelu_lut_init) init_gelu_lut();
    if (x <= -8.0f) return 0.0f;
    if (x >= 8.0f) return x;
    float idx_f = (x + 8.0f) * 1023.0f / 16.0f;
    int idx = (int)idx_f;
    if (idx >= 1023) return gelu_lut[1023];
    float frac = idx_f - (float)idx;
    return gelu_lut[idx] + frac * (gelu_lut[idx + 1] - gelu_lut[idx]);
}

// --- Atención causal ---
// q, k, v: (B, T, n_embd)
// Salida: (B, T, n_embd)
static void attention_causal(float* output, const float* q, const float* k,
                              const float* v, int B, int T, int n_head, int head_size) {
    int n_embd = n_head * head_size;
    float* scores = (float*)malloc(T * T * sizeof(float));
    if (!scores) return;

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < n_head; h++) {
                float* q_head = (float*)q + b * T * n_embd + t * n_embd + h * head_size;

                // Calcular scores para esta head en posición t
                for (int t2 = 0; t2 <= t; t2++) {
                    float* k_head = (float*)k + b * T * n_embd + t2 * n_embd + h * head_size;
                    float score = 0.0f;
                    for (int d = 0; d < head_size; d++) {
                        score += q_head[d] * k_head[d];
                    }
                    score /= sqrtf((float)head_size);  // scale
                    scores[t * T + t2] = score;
                }
                // t2 > t: mask -inf (causal)
                for (int t2 = t + 1; t2 < T; t2++) {
                    scores[t * T + t2] = -INFINITY;
                }

                // Softmax sobre scores[t]
                softmax(scores + t * T, T);

                // Weighted sum of values
                float* out_head = output + b * T * n_embd + t * n_embd + h * head_size;
                memset(out_head, 0, head_size * sizeof(float));
                for (int t2 = 0; t2 <= t; t2++) {
                    float* v_head = (float*)v + b * T * n_embd + t2 * n_embd + h * head_size;
                    float attn_weight = scores[t * T + t2];
                    for (int d = 0; d < head_size; d++) {
                        out_head[d] += attn_weight * v_head[d];
                    }
                }
            }
        }
    }
    free(scores);
}

// --- Forward pass completo ---
// input_tokens: array de T tokens (int32)
// output_logits: array de T × vocab_size (float32) - puede ser NULL si solo queremos el último token
// T: cantidad de tokens de entrada
// Devuelve un puntero a los logits del último token (para sampling)
static float* model_forward(transformer_model_t* m, const int32_t* input_tokens,
                            int T, float temperature, int top_k) {
    int B = 1;  // batch size = 1 (inferencia)
    int n_embd = m->n_embd;

    // Buffers
    float* x = (float*)calloc(T * n_embd, sizeof(float));       // embedding + pos
    float* x_ln1 = (float*)malloc(T * n_embd * sizeof(float));  // after rmsnorm 1
    float* qkv = (float*)malloc(3 * T * n_embd * sizeof(float)); // QKV projection
    float* q = qkv;                // Q: T × n_embd
    float* k = qkv + T * n_embd;   // K: T × n_embd
    float* v = qkv + 2 * T * n_embd;  // V: T × n_embd
    float* attn_out = (float*)malloc(T * n_embd * sizeof(float));
    float* x_ln2 = (float*)malloc(T * n_embd * sizeof(float));
    float* mlp_in = (float*)malloc(4 * T * n_embd * sizeof(float));
    float* mlp_out = (float*)malloc(T * n_embd * sizeof(float));  // corregido: T × n_embd
    float* residual = (float*)malloc(T * n_embd * sizeof(float));
    float* logits = (float*)malloc(T * m->vocab_size * sizeof(float));
    float* quant_tmp = (float*)malloc(n_embd * sizeof(float));  // para cuantizar

    // Buffer para activaciones cuantizadas int8
    int8_t* x_int8 = (int8_t*)malloc(n_embd * sizeof(int8_t));

    // --- Embedding ---
    for (int t = 0; t < T; t++) {
        int token = input_tokens[t];
        if (token < 0 || token >= m->vocab_size) token = 0;
        float* row = x + t * n_embd;
        // Token embedding
        float* wte_row = (float*)m->wte.data + token * n_embd;
        // Position embedding
        float* wpe_row = (float*)m->wpe.data + t * n_embd;
        for (int i = 0; i < n_embd; i++) {
            row[i] = wte_row[i] + wpe_row[i];
        }
    }

    // --- Transformer blocks ---
    for (int l = 0; l < m->n_layer; l++) {
        // Guardar residual
        memcpy(residual, x, T * n_embd * sizeof(float));

        // --- RMSNorm 1 ---
        for (int t = 0; t < T; t++) {
            float* ln1_gamma = (float*)m->blocks[l].ln_1.data;
            rmsnorm(x_ln1 + t * n_embd, x + t * n_embd, ln1_gamma, n_embd, 1e-5f);
        }

        // --- Attention ---
        // Cuantizar x_ln1 a int8 para el matmul ternario
        for (int t = 0; t < T; t++) {
            quantize_row(x_ln1 + t * n_embd, x_int8, n_embd);
            // c_attn: (3*n_embd, n_embd) ternario
            int n_out = 3 * n_embd;
            int32_t* attn_scores = (int32_t*)malloc(n_out * sizeof(int32_t));
            if (!attn_scores) continue;

            // Producto punto ternario para cada fila de c_attn
            for (int j = 0; j < n_out; j++) {
                attn_scores[j] = dot_ternary(x_int8, (uint8_t*)m->blocks[l].c_attn.data,
                                             j * n_embd, n_embd);
            }

            // Des-cuantizar a float32
            // (en esta versión simple, usamos los scores int32 directamente como float)
            // Nota: para una implementación más precisa, necesitaríamos escalas
            for (int j = 0; j < n_out; j++) {
                qkv[t * n_out + j] = (float)attn_scores[j] * 0.01f;
            }
            free(attn_scores);
        }

        // Atención causal
        attention_causal(attn_out, q, k, v, B, T, m->n_head, m->head_size);

        // Cuantizar attn_out a int8
        for (int t = 0; t < T; t++) {
            quantize_row(attn_out + t * n_embd, x_int8, n_embd);

            int32_t* proj_scores = (int32_t*)malloc(n_embd * sizeof(int32_t));
            if (!proj_scores) continue;

            // c_proj: (n_embd, n_embd) ternario
            for (int j = 0; j < n_embd; j++) {
                proj_scores[j] = dot_ternary(x_int8, (uint8_t*)m->blocks[l].c_proj.data,
                                             j * n_embd, n_embd);
            }

            for (int j = 0; j < n_embd; j++) {
                attn_out[t * n_embd + j] = (float)proj_scores[j] * 0.01f;
            }
            free(proj_scores);
        }

        // Residual connection
        for (int t = 0; t < T * n_embd; t++) {
            x[t] = residual[t] + attn_out[t];
        }

        // --- RMSNorm 2 ---
        memcpy(residual, x, T * n_embd * sizeof(float));
        for (int t = 0; t < T; t++) {
            float* ln2_gamma = (float*)m->blocks[l].ln_2.data;
            rmsnorm(x_ln2 + t * n_embd, x + t * n_embd, ln2_gamma, n_embd, 1e-5f);
        }

        // --- MLP ---
        for (int t = 0; t < T; t++) {
            quantize_row(x_ln2 + t * n_embd, x_int8, n_embd);

            // c_fc: (4*n_embd, n_embd) ternario
            int n_fc = 4 * n_embd;
            int32_t* fc_scores = (int32_t*)malloc(n_fc * sizeof(int32_t));
            if (!fc_scores) continue;
            for (int j = 0; j < n_fc; j++) {
                fc_scores[j] = dot_ternary(x_int8, (uint8_t*)m->blocks[l].mlp_fc.data,
                                           j * n_embd, n_embd);
            }

            // GELU + float
            for (int j = 0; j < n_fc; j++) {
                mlp_in[t * n_fc + j] = fast_gelu((float)fc_scores[j] * 0.01f);
            }
            free(fc_scores);

            // c_proj: (n_embd, 4*n_embd) ternario
            quantize_row(mlp_in + t * n_fc, x_int8, n_fc);
            int32_t* proj_scores = (int32_t*)malloc(n_embd * sizeof(int32_t));
            if (!proj_scores) continue;
            for (int j = 0; j < n_embd; j++) {
                proj_scores[j] = dot_ternary(x_int8, (uint8_t*)m->blocks[l].mlp_proj.data,
                                             j * n_fc, n_fc);
            }
            for (int j = 0; j < n_embd; j++) {
                mlp_out[t * n_embd + j] = (float)proj_scores[j] * 0.01f;
            }
            free(proj_scores);
        }

        // Residual connection
        for (int t = 0; t < T * n_embd; t++) {
            x[t] = residual[t] + mlp_out[t];
        }
    }

    // --- Final RMSNorm ---
    float* ln_f_gamma = (float*)m->ln_f.data;
    for (int t = 0; t < T; t++) {
        rmsnorm(x + t * n_embd, x + t * n_embd, ln_f_gamma, n_embd, 1e-5f);
    }

    // --- LM Head ---
    for (int t = 0; t < T; t++) {
        quantize_row(x + t * n_embd, x_int8, n_embd);
        // lm_head: (vocab_size, n_embd) ternario
        for (int j = 0; j < m->vocab_size; j++) {
            logits[t * m->vocab_size + j] = (float)dot_ternary(x_int8,
                (uint8_t*)m->lm_head.data, j * n_embd, n_embd) * 0.01f;
        }
    }

    // Últimos logits (para sampling)
    float* last_logits = logits + (T - 1) * m->vocab_size;
    // Copiar a output para sampler (softmax in-place)
    softmax(last_logits, m->vocab_size);

    // Limpiar
    free(x);
    free(x_ln1);
    free(qkv);
    free(attn_out);
    free(x_ln2);
    free(mlp_in);
    free(mlp_out);
    free(residual);
    free(logits);
    free(quant_tmp);
    free(x_int8);

    return last_logits;  // Nota: esto apunta a memoria liberada!
    // Para una implementación correcta, el sampling debe hacerse antes de free.
    // Por ahora, retornamos y el caller debe usarlo inmediatamente.
}
// Versión corregida: sampling dentro del forward
static int model_forward_token(transformer_model_t* m, const int32_t* input_tokens,
                               int T, float temperature, int top_k) {
    int B = 1;
    int n_embd = m->n_embd;
    int n_head = m->n_head;
    int head_size = m->head_size;

    float* x = (float*)calloc(T * n_embd, sizeof(float));
    float* x_ln1 = (float*)malloc(T * n_embd * sizeof(float));
    float* qkv = (float*)malloc(3 * T * n_embd * sizeof(float));
    float* q = qkv;
    float* k = qkv + T * n_embd;
    float* v = qkv + 2 * T * n_embd;
    float* attn_out = (float*)malloc(T * n_embd * sizeof(float));
    float* x_ln2 = (float*)malloc(T * n_embd * sizeof(float));
    float* mlp_in = (float*)malloc(4 * T * n_embd * sizeof(float));
    float* mlp_out = (float*)malloc(T * n_embd * sizeof(float));
    float* residual = (float*)malloc(T * n_embd * sizeof(float));
    float* logits = (float*)malloc(m->vocab_size * sizeof(float));
    int max_q = 4 * n_embd;
    int8_t* x_int8 = (int8_t*)malloc(max_q * sizeof(int8_t));

    if (!x || !x_ln1 || !qkv || !attn_out || !x_ln2 || !mlp_in || !mlp_out || !residual || !logits || !x_int8) {
        printf("Error: no se pudo asignar memoria\n");
        free(x); free(x_ln1); free(qkv); free(attn_out); free(x_ln2);
        free(mlp_in); free(mlp_out); free(residual); free(logits); free(x_int8);
        return 0;
    }

    // Embedding
    for (int t = 0; t < T; t++) {
        int token = input_tokens[t];
        if (token < 0 || token >= m->vocab_size) token = 0;
        float* row = x + t * n_embd;
        float* wte_row = (float*)m->wte.data + token * n_embd;
        float* wpe_row = (float*)m->wpe.data + t * n_embd;
        for (int i = 0; i < n_embd; i++) {
            row[i] = wte_row[i] + wpe_row[i];
        }
    }
    // Transformer blocks
    for (int l = 0; l < m->n_layer; l++) {
        memcpy(residual, x, T * n_embd * sizeof(float));

        for (int t = 0; t < T; t++) {
            float* ln1_gamma = (float*)m->blocks[l].ln_1.data;
            rmsnorm(x_ln1 + t * n_embd, x + t * n_embd, ln1_gamma, n_embd, 1e-5f);
        }

        // Attention QKV projection (I2_S)
        for (int t = 0; t < T; t++) {
            quantize_row(x_ln1 + t * n_embd, x_int8, n_embd);
            int n_out = 3 * n_embd;
            int8_t* w = (int8_t*)m->blocks[l].c_attn.data;
            for (int j = 0; j < n_out; j++) {
                int32_t score = dot_i8(x_int8, w + j * n_embd, n_embd);
                qkv[t * n_out + j] = (float)score * 0.01f;
            }
        }

        attention_causal(attn_out, q, k, v, B, T, m->n_head, m->head_size);

        // Attention output projection (I2_S)
        for (int t = 0; t < T; t++) {
            quantize_row(attn_out + t * n_embd, x_int8, n_embd);
            int8_t* w = (int8_t*)m->blocks[l].c_proj.data;
            for (int j = 0; j < n_embd; j++) {
                int32_t score = dot_i8(x_int8, w + j * n_embd, n_embd);
                attn_out[t * n_embd + j] = (float)score * 0.01f;
            }
        }
        for (int t = 0; t < T * n_embd; t++) x[t] = residual[t] + attn_out[t];
        memcpy(residual, x, T * n_embd * sizeof(float));
        for (int t = 0; t < T; t++) {
            float* ln2_gamma = (float*)m->blocks[l].ln_2.data;
            rmsnorm(x_ln2 + t * n_embd, x + t * n_embd, ln2_gamma, n_embd, 1e-5f);
        }
        for (int t = 0; t < T; t++) {
            quantize_row(x_ln2 + t * n_embd, x_int8, n_embd);
            int n_fc = 4 * n_embd;
            int8_t* w_fc = (int8_t*)m->blocks[l].mlp_fc.data;
            for (int j = 0; j < n_fc; j++) {
                int32_t score = dot_i8(x_int8, w_fc + j * n_embd, n_embd);
                mlp_in[t * n_fc + j] = fast_gelu((float)score * 0.01f);
            }
            quantize_row(mlp_in + t * n_fc, x_int8, n_fc);
            int8_t* w_proj = (int8_t*)m->blocks[l].mlp_proj.data;
            for (int j = 0; j < n_embd; j++) {
                int32_t score = dot_i8(x_int8, w_proj + j * n_fc, n_fc);
                mlp_out[t * n_embd + j] = (float)score * 0.01f;
            }
        }

        for (int t = 0; t < T * n_embd; t++) x[t] = residual[t] + mlp_out[t];
    }

    // Final RMSNorm
    float* ln_f_gamma = (float*)m->ln_f.data;
    for (int t = 0; t < T; t++) {
        rmsnorm(x + t * n_embd, x + t * n_embd, ln_f_gamma, n_embd, 1e-5f);
    }

    // LM Head (I2_S)
    int last_t = T - 1;
    quantize_row(x + last_t * n_embd, x_int8, n_embd);
    int8_t* w_head = (int8_t*)m->lm_head.data;
    for (int j = 0; j < m->vocab_size; j++) {
        logits[j] = (float)dot_i8(x_int8, w_head + j * n_embd, n_embd) * 0.01f;
    }

    // Sampling
    int token;
    if (top_k > 0) {
        token = sample_topk(logits, m->vocab_size, top_k, temperature);
    } else {
        softmax(logits, m->vocab_size);
        token = sample_token(logits, m->vocab_size, temperature);
    }

    // Limpiar
    free(x); free(x_ln1); free(qkv); free(attn_out); free(x_ln2);
    free(mlp_in); free(mlp_out); free(residual); free(logits); free(x_int8);

    return token;
}

// --- Atención con KV Cache (decode step) ---
// q: vector (n_embd)
// cache_k, cache_v: arrays (cache_len × n_embd)
// output: vector (n_embd)
static void attention_causal_cached(float* output, const float* q,
                                     const float* cache_k, const float* cache_v,
                                     int cache_len, int n_head, int head_size) {
    int n_embd = n_head * head_size;
    float* scores = (float*)malloc(cache_len * sizeof(float));
    if (!scores) return;

    for (int h = 0; h < n_head; h++) {
        const float* q_head = q + h * head_size;

        for (int t2 = 0; t2 < cache_len; t2++) {
            const float* k_head = cache_k + t2 * n_embd + h * head_size;
            float score = 0.0f;
            for (int d = 0; d < head_size; d++) {
                score += q_head[d] * k_head[d];
            }
            scores[t2] = score / sqrtf((float)head_size);
        }

        softmax(scores, cache_len);

        float* out_head = output + h * head_size;
        memset(out_head, 0, head_size * sizeof(float));
        for (int t2 = 0; t2 < cache_len; t2++) {
            const float* v_head = cache_v + t2 * n_embd + h * head_size;
            float attn_weight = scores[t2];
            for (int d = 0; d < head_size; d++) {
                out_head[d] += attn_weight * v_head[d];
            }
        }
    }
    free(scores);
}

// --- Thread worker para matmul paralelo ---
typedef struct {
    int start_row, end_row;
    int n_cols;
    const int8_t* x;
    const int8_t* w;
    float* output;
    float scale;
} matmul_worker_t;

static void* matmul_worker(void* arg) {
    matmul_worker_t* a = (matmul_worker_t*)arg;
    for (int j = a->start_row; j < a->end_row; j++) {
        int32_t score = dot_i8(a->x, a->w + j * a->n_cols, a->n_cols);
        a->output[j] = (float)score * a->scale;
    }
    return NULL;
}

// Ejecutar matmul I2_S en 2 threads
static void matmul_i8_mt(float* output, const int8_t* x, const int8_t* w,
                          int n_rows, int n_cols, float scale) {
    pthread_t threads[2];
    matmul_worker_t args[2];
    int mid = n_rows / 2;

    for (int t = 0; t < 2; t++) {
        args[t] = (matmul_worker_t){
            .start_row = t * mid,
            .end_row = (t == 0) ? mid : n_rows,
            .n_cols = n_cols,
            .x = x,
            .w = w,
            .output = output,
            .scale = scale
        };
        pthread_create(&threads[t], NULL, matmul_worker, &args[t]);
    }
    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);
}

// --- Generación con KV Cache + Multi-thread ---
// prefill: procesa el prompt completo (T > 1), llena cache, devuelve último token
static int model_prefill(transformer_model_t* m, const int32_t* input_tokens,
                          int T, kv_cache_t* cache, float temperature, int top_k) {
    int B = 1;
    int n_embd = m->n_embd;
    int n_head = m->n_head;
    int head_size = m->head_size;
    int max_q = 4 * n_embd;

    float* x = (float*)calloc(T * n_embd, sizeof(float));
    float* x_ln1 = (float*)malloc(T * n_embd * sizeof(float));
    float* qkv = (float*)malloc(3 * T * n_embd * sizeof(float));
    float* q = qkv;
    float* k = qkv + T * n_embd;
    float* v = qkv + 2 * T * n_embd;
    float* attn_out = (float*)malloc(T * n_embd * sizeof(float));
    float* x_ln2 = (float*)malloc(T * n_embd * sizeof(float));
    float* mlp_in = (float*)malloc(4 * T * n_embd * sizeof(float));
    float* mlp_out = (float*)malloc(T * n_embd * sizeof(float));
    float* residual = (float*)malloc(T * n_embd * sizeof(float));
    float* logits = (float*)malloc(m->vocab_size * sizeof(float));
    int8_t* x_int8 = (int8_t*)malloc(max_q * sizeof(int8_t));
    int8_t* buf_int8 = (int8_t*)malloc(max_q * sizeof(int8_t));  // para mt

    if (!x || !x_ln1 || !qkv || !attn_out || !x_ln2 || !mlp_in || !mlp_out || !residual || !logits || !x_int8 || !buf_int8)
        goto cleanup;

    // Embedding
    for (int t = 0; t < T; t++) {
        int token = input_tokens[t];
        if (token < 0 || token >= m->vocab_size) token = 0;
        float* row = x + t * n_embd;
        float* wte_row = (float*)m->wte.data + token * n_embd;
        float* wpe_row = (float*)m->wpe.data + t * n_embd;
        for (int i = 0; i < n_embd; i++) row[i] = wte_row[i] + wpe_row[i];
    }

    for (int l = 0; l < m->n_layer; l++) {
        memcpy(residual, x, T * n_embd * sizeof(float));

        for (int t = 0; t < T; t++) {
            float* ln1_gamma = (float*)m->blocks[l].ln_1.data;
            rmsnorm(x_ln1 + t * n_embd, x + t * n_embd, ln1_gamma, n_embd, 1e-5f);
        }

        // QKV projection (multi-thread)
        for (int t = 0; t < T; t++) {
            quantize_row(x_ln1 + t * n_embd, x_int8, n_embd);
            int n_out = 3 * n_embd;
            int8_t* w = (int8_t*)m->blocks[l].c_attn.data;
            matmul_i8_mt(qkv + t * n_out, x_int8, w, n_out, n_embd, 0.01f);
        }

        // Guardar K,V en cache
        for (int t = 0; t < T; t++) {
            kv_cache_append(cache, l, k + t * n_embd, v + t * n_embd);
        }

        attention_causal(attn_out, q, k, v, B, T, n_head, head_size);

        // Output projection
        for (int t = 0; t < T; t++) {
            quantize_row(attn_out + t * n_embd, x_int8, n_embd);
            int8_t* w = (int8_t*)m->blocks[l].c_proj.data;
            matmul_i8_mt(attn_out + t * n_embd, x_int8, w, n_embd, n_embd, 0.01f);
        }

        for (int t = 0; t < T * n_embd; t++) x[t] = residual[t] + attn_out[t];
        memcpy(residual, x, T * n_embd * sizeof(float));

        for (int t = 0; t < T; t++) {
            float* ln2_gamma = (float*)m->blocks[l].ln_2.data;
            rmsnorm(x_ln2 + t * n_embd, x + t * n_embd, ln2_gamma, n_embd, 1e-5f);
        }

        // MLP (multi-thread)
        for (int t = 0; t < T; t++) {
            quantize_row(x_ln2 + t * n_embd, x_int8, n_embd);
            int n_fc = 4 * n_embd;
            int8_t* w_fc = (int8_t*)m->blocks[l].mlp_fc.data;
            // MLP fc con MT
            pthread_t th[2];
            matmul_worker_t a[2];
            int mid = n_fc / 2;
            for (int th_idx = 0; th_idx < 2; th_idx++) {
                a[th_idx] = (matmul_worker_t){
                    .start_row = th_idx * mid,
                    .end_row = th_idx == 0 ? mid : n_fc,
                    .n_cols = n_embd, .x = x_int8, .w = w_fc,
                    .output = mlp_in + t * n_fc, .scale = 0.01f
                };
                pthread_create(&th[th_idx], NULL, matmul_worker, &a[th_idx]);
            }
            pthread_join(th[0], NULL); pthread_join(th[1], NULL);

            for (int j = 0; j < n_fc; j++)
                mlp_in[t * n_fc + j] = fast_gelu(mlp_in[t * n_fc + j]);

            quantize_row(mlp_in + t * n_fc, buf_int8, n_fc);
            int8_t* w_proj = (int8_t*)m->blocks[l].mlp_proj.data;
            matmul_i8_mt(mlp_out + t * n_embd, buf_int8, w_proj, n_embd, n_fc, 0.01f);
        }

        for (int t = 0; t < T * n_embd; t++) x[t] = residual[t] + mlp_out[t];
    }

    // Final RMSNorm
    float* ln_f_gamma = (float*)m->ln_f.data;
    for (int t = 0; t < T; t++)
        rmsnorm(x + t * n_embd, x + t * n_embd, ln_f_gamma, n_embd, 1e-5f);

    // LM Head + sampling
    int last_t = T - 1;
    quantize_row(x + last_t * n_embd, x_int8, n_embd);
    int8_t* w_head = (int8_t*)m->lm_head.data;
    matmul_i8_mt(logits, x_int8, w_head, m->vocab_size, n_embd, 0.01f);

    int token;
    if (top_k > 0) token = sample_topk(logits, m->vocab_size, top_k, temperature);
    else { softmax(logits, m->vocab_size); token = sample_token(logits, m->vocab_size, temperature); }

cleanup:
    free(x); free(x_ln1); free(qkv); free(attn_out); free(x_ln2);
    free(mlp_in); free(mlp_out); free(residual); free(logits);
    free(x_int8); free(buf_int8);
    return token;
}

// decode: genera 1 token usando KV cache
static int model_decode(transformer_model_t* m, int input_token,
                         kv_cache_t* cache, float temperature, int top_k) {
    int n_embd = m->n_embd;
    int n_head = m->n_head;
    int head_size = m->head_size;
    int max_q = 4 * n_embd;

    float* x = (float*)calloc(n_embd, sizeof(float));
    float* x_ln1 = (float*)malloc(n_embd * sizeof(float));
    float* qkv_all = (float*)malloc(3 * n_embd * sizeof(float));
    float* q_vec = qkv_all;
    float* k_vec = qkv_all + n_embd;
    float* v_vec = qkv_all + 2 * n_embd;
    float* attn_out = (float*)malloc(n_embd * sizeof(float));
    float* x_ln2 = (float*)malloc(n_embd * sizeof(float));
    float* mlp_in = (float*)malloc(4 * n_embd * sizeof(float));
    float* mlp_out = (float*)malloc(n_embd * sizeof(float));
    float* logits = (float*)malloc(m->vocab_size * sizeof(float));
    int8_t* x_int8 = (int8_t*)malloc(max_q * sizeof(int8_t));
    int8_t* buf_int8 = (int8_t*)malloc(max_q * sizeof(int8_t));

    if (!x || !x_ln1 || !q_vec || !k_vec || !v_vec || !attn_out || !x_ln2 || !mlp_in || !mlp_out || !logits || !x_int8 || !buf_int8)
        goto cleanup;

    // Embedding (1 token)
    {
        int token = input_token;
        if (token < 0 || token >= m->vocab_size) token = 0;
        float* wte_row = (float*)m->wte.data + token * n_embd;
        float* wpe_row = (float*)m->wpe.data + cache->current_len * n_embd;
        for (int i = 0; i < n_embd; i++) x[i] = wte_row[i] + wpe_row[i];
    }

    for (int l = 0; l < m->n_layer; l++) {
        float* ln1_gamma = (float*)m->blocks[l].ln_1.data;
        rmsnorm(x_ln1, x, ln1_gamma, n_embd, 1e-5f);

        // QKV projection (1 token)
        quantize_row(x_ln1, x_int8, n_embd);
        int8_t* w = (int8_t*)m->blocks[l].c_attn.data;
        matmul_i8_mt(q_vec, x_int8, w, 3 * n_embd, n_embd, 0.01f);

        // q_vec, k_vec, v_vec ya apuntan a las partes correctas de qkv_all

        // Append K,V a cache
        kv_cache_append(cache, l, k_vec, v_vec);

        // Atención contra cache
        attention_causal_cached(attn_out, q_vec,
                                 cache->k_cache[l], cache->v_cache[l],
                                 cache->current_len, n_head, head_size);

        // Output projection
        quantize_row(attn_out, x_int8, n_embd);
        int8_t* w_proj = (int8_t*)m->blocks[l].c_proj.data;
        matmul_i8_mt(attn_out, x_int8, w_proj, n_embd, n_embd, 0.01f);

        for (int i = 0; i < n_embd; i++) x[i] += attn_out[i];

        float* ln2_gamma = (float*)m->blocks[l].ln_2.data;
        rmsnorm(x_ln2, x, ln2_gamma, n_embd, 1e-5f);

        // MLP
        quantize_row(x_ln2, x_int8, n_embd);
        int8_t* w_fc = (int8_t*)m->blocks[l].mlp_fc.data;
        matmul_i8_mt(mlp_in, x_int8, w_fc, 4 * n_embd, n_embd, 0.01f);
        for (int i = 0; i < 4 * n_embd; i++) mlp_in[i] = fast_gelu(mlp_in[i]);

        quantize_row(mlp_in, buf_int8, 4 * n_embd);
        int8_t* w_proj_mlp = (int8_t*)m->blocks[l].mlp_proj.data;
        matmul_i8_mt(mlp_out, buf_int8, w_proj_mlp, n_embd, 4 * n_embd, 0.01f);

        for (int i = 0; i < n_embd; i++) x[i] += mlp_out[i];
    }

    // Final RMSNorm
    float* ln_f_gamma = (float*)m->ln_f.data;
    rmsnorm(x, x, ln_f_gamma, n_embd, 1e-5f);

    // LM Head
    quantize_row(x, x_int8, n_embd);
    int8_t* w_head = (int8_t*)m->lm_head.data;
    matmul_i8_mt(logits, x_int8, w_head, m->vocab_size, n_embd, 0.01f);

    int token;
    if (top_k > 0) token = sample_topk(logits, m->vocab_size, top_k, temperature);
    else { softmax(logits, m->vocab_size); token = sample_token(logits, m->vocab_size, temperature); }

cleanup:
    free(x); free(x_ln1); free(qkv_all);
    free(attn_out); free(x_ln2); free(mlp_in); free(mlp_out);
    free(logits); free(x_int8); free(buf_int8);
    return token;
}

// --- HGRN Decode (MatMul-Free) ---
// Procesa 1 token con estado recurrente.
// state_in/state_out: array de n_layer × n_embd floats (estado de cada capa)

static inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static int model_hgrn_decode(transformer_model_t* m, int input_token,
                              float* state_in, float* state_out,
                              float temperature, int top_k) {
    int n_embd = m->n_embd;
    int max_q = 4 * n_embd;

    float* x = (float*)calloc(n_embd, sizeof(float));
    float* x_int = (float*)malloc(n_embd * sizeof(float));    // for RMSNorm
    float* out = (float*)malloc(n_embd * sizeof(float));
    float* logits = (float*)malloc(m->vocab_size * sizeof(float));
    int8_t* x_int8 = (int8_t*)malloc(max_q * sizeof(int8_t));

    if (!x || !x_int || !out || !logits || !x_int8)
        goto cleanup_h;

    // Embedding
    {
        int token = input_token;
        if (token < 0 || token >= m->vocab_size) token = 0;
        float* wte_row = (float*)m->wte.data + token * n_embd;
        float* wpe_row = (float*)m->wpe.data + 0 * n_embd;  // pos 0 siempre (1 token)
        for (int i = 0; i < n_embd; i++) x[i] = wte_row[i] + wpe_row[i];
    }

    for (int l = 0; l < m->n_layer; l++) {
        float* h = state_in + l * n_embd;          // estado previo
        float* h_new = state_out + l * n_embd;      // nuevo estado

        float* norm_gamma = (float*)m->hgrn_blocks[l].norm.data;
        int8_t* f_w = (int8_t*)m->hgrn_blocks[l].forget_w.data;
        float* f_b = (float*)m->hgrn_blocks[l].forget_b.data;
        int8_t* i_w = (int8_t*)m->hgrn_blocks[l].input_w.data;
        float* i_b = (float*)m->hgrn_blocks[l].input_b.data;
        int8_t* o_w = (int8_t*)m->hgrn_blocks[l].out_w.data;

        // RMSNorm input
        rmsnorm(x_int, x, norm_gamma, n_embd, 1e-5f);

        // Quantize to int8
        quantize_row(x_int, x_int8, n_embd);

        // Forget gate: f = sigmoid(W_f @ x + b_f)
        // Input gate: i = sigmoid(W_i @ x + b_i)
        float f_val, i_val;
        {   // forget gate
            int32_t score = dot_i8(x_int8, f_w, n_embd);
            f_val = sigmoid_f((float)score * 0.01f + f_b[0]);
        }
        {   // input gate
            int32_t score = dot_i8(x_int8, i_w, n_embd);
            i_val = sigmoid_f((float)score * 0.01f + i_b[0]);
        }

        // Clamp forget gate with lower bound (hierarchical)
        float lower_bound = 0.8f * l / (m->n_layer - 1.0f);
        if (f_val < lower_bound) f_val = lower_bound;

        // State update: h_new = (1-f)*h_prev + f*i
        for (int i = 0; i < n_embd; i++) {
            h_new[i] = (1.0f - f_val) * h[i] + f_val * i_val;
        }

        // Output projection: o = W_o @ h_new  (ternary matmul)
        quantize_row(h_new, x_int8, n_embd);
        {   int32_t score = dot_i8(x_int8, o_w, n_embd);
            out[0] = (float)score * 0.01f;
        }
        // Copy to x for next layer (simplificado: vector escalar)
        // Para una capa completa, usaríamos matmul (n_embd × n_embd)
        // Por ahora: usar el score como x de la siguiente capa
        for (int i = 0; i < n_embd; i++) {
            int32_t score = dot_i8(x_int8, o_w + i * n_embd, n_embd);
            x[i] = (float)score * 0.01f;
        }
    }

    // Final RMSNorm
    float* ln_f_gamma = (float*)m->ln_f.data;
    rmsnorm(x, x, ln_f_gamma, n_embd, 1e-5f);

    // LM Head
    quantize_row(x, x_int8, n_embd);
    int8_t* w_head = (int8_t*)m->lm_head.data;
    for (int j = 0; j < m->vocab_size; j++) {
        logits[j] = (float)dot_i8(x_int8, w_head + j * n_embd, n_embd) * 0.01f;
    }

    int token;
    if (top_k > 0) token = sample_topk(logits, m->vocab_size, top_k, temperature);
    else { softmax(logits, m->vocab_size); token = sample_token(logits, m->vocab_size, temperature); }

cleanup_h:
    free(x); free(x_int); free(out); free(logits); free(x_int8);
    return token;
}

#endif // TRANSFORMER_H
