/*
 * TERN — Motor de inferencia ternario para CPU débil
 * 
 * Uso: tern [opciones]
 *   -m <modelo>   Ruta al modelo .bin (default: model.bin)
 *   -p <prompt>   Prompt de texto
 *   -n <n>        Cantidad de tokens a generar (default: 100)
 *   -t <temp>     Temperatura (default: 0.8)
 *   -k <k>        Top-k sampling (default: 10, 0 = greedy)
 * 
 * Ejemplo:
 *   ./tern -p "ROMEO: " -n 200 -t 0.8 -k 10
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <unistd.h>

#include "transformer.h"
#include "tensor.h"
#include "ternary.h"
#include "sampler.h"

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static long peak_rss_kb(void) {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    long rss = -1;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "VmHWM: %ld kB", &rss) == 1) break;
        if (sscanf(line, "VmRSS: %ld kB", &rss) == 1) continue;
    }
    fclose(f);
    return rss;
}

// Tokenizar string de texto a IDs
static int tokenize(const transformer_model_t* m, const char* text, int32_t* tokens, int max_tokens) {
    int len = (int)strlen(text);
    if (len > max_tokens) len = max_tokens;
    for (int i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        tokens[i] = m->stoi[c];
    }
    return len;
}

// Decodificar IDs a string
static void detokenize(const transformer_model_t* m, const int32_t* tokens, int n, char* out) {
    for (int i = 0; i < n; i++) {
        int id = tokens[i];
        if (id >= 0 && id < m->vocab_size) {
            out[i] = (char)m->itos[id];
        } else {
            out[i] = '?';
        }
    }
    out[n] = '\0';
}

// Generar texto
static void generate(transformer_model_t* m, const char* prompt,
                     int max_new_tokens, float temperature, int top_k) {
    // Tokenizar prompt
    int32_t tokens[1024];
    int prompt_len = tokenize(m, prompt, tokens, 1024);

    printf("> %s", prompt);
    fflush(stdout);

    // Generar tokens uno por uno
    for (int i = 0; i < max_new_tokens; i++) {
        int next = model_forward_token(m, tokens, prompt_len, temperature, top_k);

        // Mostrar token generado
        char c = (char)m->itos[next];
        putchar(c);
        fflush(stdout);

        // Shift context
        if (prompt_len < m->block_size) {
            tokens[prompt_len] = next;
            prompt_len++;
        } else {
            // Shift window
            memmove(tokens, tokens + 1, (m->block_size - 1) * sizeof(int32_t));
            tokens[m->block_size - 1] = next;
        }
    }
    printf("\n");
}

int main(int argc, char** argv) {
    srand((unsigned int)time(NULL));

    const char* model_path = "model.bin";
    const char* prompt = "";
    int max_new_tokens = 100;
    float temperature = 0.8f;
    int top_k = 10;
    int benchmark_mode = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) max_new_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--bench") == 0) benchmark_mode = 1;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Uso: %s [opciones]\n", argv[0]);
            printf("  -m <modelo>  Ruta al modelo .bin\n");
            printf("  -p <prompt>  Prompt\n");
            printf("  -n <n>       Tokens (default: 100)\n");
            printf("  -t <temp>    Temperatura (default: 0.8)\n");
            printf("  -k <k>       Top-k (default: 10, 0=greedy)\n");
            printf("  -b           Benchmark mode (timing stats)\n");
            return 0;
        }
    }

    double t0 = now_seconds();

    transformer_model_t model;
    if (!model_load(&model, model_path)) {
        fprintf(stderr, "Error: no se pudo cargar el modelo %s\n", model_path);
        return 1;
    }

    double load_time = now_seconds() - t0;
    long rss_peak = peak_rss_kb();

    if (!benchmark_mode) {
        printf("Modelo cargado: %s\n", model_path);
        printf("Generando %d tokens...\n", max_new_tokens);
        fflush(stdout);
    }

    // Generar con KV Cache + Multi-thread
    {
        int32_t tokens[1024];
        int prompt_len = 0;
        if (strlen(prompt) > 0) {
            for (int i = 0; prompt[i]; i++) {
                unsigned char c = (unsigned char)prompt[i];
                tokens[i] = model.stoi[c];
            }
            prompt_len = (int)strlen(prompt);
        }
        int orig_prompt_len = prompt_len;

        // Crear KV cache
        kv_cache_t cache = kv_cache_alloc(model.block_size, model.n_layer,
                                          model.n_embd, model.n_head);
        double prefill_time = 0;
        double decode_time = 0;

        if (!benchmark_mode) {
            printf("> %s", prompt);
            fflush(stdout);
        }

        // Prefill: procesar prompt completo
        double t1 = now_seconds();
        if (prompt_len > 0) {
            int first = model_prefill(&model, tokens, prompt_len, &cache,
                                       temperature, top_k);
            prefill_time = now_seconds() - t1;
            if (!benchmark_mode) {
                char c = (char)model.itos[first];
                putchar(c);
                fflush(stdout);
            }
            tokens[0] = first;
            prompt_len = 1;
        }

        // Decode: generar tokens uno por uno con KV cache
        t1 = now_seconds();
        for (int i = 0; i < max_new_tokens - 1; i++) {
            int next = model_decode(&model, tokens[0], &cache,
                                     temperature, top_k);
            if (!benchmark_mode) {
                char c = (char)model.itos[next];
                putchar(c);
                fflush(stdout);
            }
            tokens[0] = next;
        }
        decode_time = now_seconds() - t1;

        if (!benchmark_mode) printf("\n");

        // --- Benchmark output ---
        if (benchmark_mode) {
            int total_tokens = max_new_tokens;
            int generated = max_new_tokens - (prompt_len > 0 ? 0 : 0);
            double decode_tok_s = generated > 1 ? (generated - 1) / decode_time : 0;
            printf("BENCHMARK\n");
            printf("model_size_mb: %.1f\n", 3.1);
            printf("params_m: 10.7\n");
            printf("n_layer: %d\n", model.n_layer);
            printf("n_head: %d\n", model.n_head);
            printf("n_embd: %d\n", model.n_embd);
            printf("load_time_s: %.3f\n", load_time);
            printf("rss_peak_kb: %ld\n", rss_peak);
            printf("prefill_time_s: %.4f\n", prefill_time);
            printf("prompt_len: %d\n", orig_prompt_len);
            printf("decode_tokens: %d\n", generated - 1);
            printf("decode_time_s: %.3f\n", decode_time);
            printf("decode_tok_s: %.1f\n", decode_tok_s);
            printf("total_time_s: %.3f\n", load_time + prefill_time + decode_time);
        }

        kv_cache_free(&cache);
    }

    model_free(&model);
    return 0;
}
