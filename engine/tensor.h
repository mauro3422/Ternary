#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

// Tipos de datos soportados
typedef enum {
    TERN_F32,       // float32
    TERN_I8,        // int8 (pre-desempaquetado)
    TERN_TERNARY,   // packed 2-bit {-1,0,+1}
} tensor_type_t;

typedef struct {
    tensor_type_t type;
    int rows;
    int cols;
    size_t stride;  // bytes entre filas
    void* data;
} tensor_t;

// Crear tensor (alloc)
static tensor_t tensor_alloc(int rows, int cols, tensor_type_t type) {
    tensor_t t;
    t.type = type;
    t.rows = rows;
    t.cols = cols;
    size_t elem_size = (type == TERN_F32) ? 4 : (type == TERN_I8) ? 1 : 0;
    // Para ternario: cada fila tiene cols * 2 bits, empaquetados
    size_t row_bytes;
    if (type == TERN_TERNARY) {
        row_bytes = ((cols * 2) + 7) / 8;  // 2 bits por elem
    } else {
        row_bytes = cols * elem_size;
    }
    t.stride = row_bytes;
    t.data = malloc(rows * row_bytes);
    if (!t.data) {
        fprintf(stderr, "Error: no se pudo asignar %zu bytes\n", rows * row_bytes);
        t.rows = 0;
        t.cols = 0;
    }
    memset(t.data, 0, rows * row_bytes);
    return t;
}

// Crear tensor view (sin alloc, apunta a buffer externo)
static tensor_t tensor_view(void* data, int rows, int cols, tensor_type_t type) {
    tensor_t t;
    t.type = type;
    t.rows = rows;
    t.cols = cols;
    t.data = data;
    size_t elem_size = (type == TERN_F32) ? 4 : (type == TERN_I8) ? 1 : 0;
    t.stride = (type == TERN_TERNARY) ? ((cols * 2 + 7) / 8) : (cols * elem_size);
    return t;
}

// Liberar tensor
static void tensor_free(tensor_t* t) {
    if (t->data) {
        free(t->data);
        t->data = NULL;
    }
    t->rows = 0;
    t->cols = 0;
}

// Copiar tensor
static tensor_t tensor_clone(const tensor_t* src) {
    tensor_t t = tensor_alloc(src->rows, src->cols, src->type);
    if (t.data && src->data) {
        memcpy(t.data, src->data, src->rows * src->stride);
    }
    return t;
}

// Desempaquetar valor ternario de un byte empaquetado
static inline int8_t unpack_ternary_val(const uint8_t* packed, int idx) {
    int byte_idx = idx / 4;     // 4 valores por byte
    int bit_off = (idx % 4) * 2;
    uint8_t val = (packed[byte_idx] >> bit_off) & 0x03;
    // Mapeo: 0→-1, 1→0, 2→+1, 3→0 (reservado)
    static const int8_t map[4] = {-1, 0, 1, 0};
    return map[val];
}

// Obtener elemento float32
static inline float tensor_get_f32(const tensor_t* t, int r, int c) {
    assert(t->type == TERN_F32);
    assert(r < t->rows && c < t->cols);
    float* row = (float*)((uint8_t*)t->data + r * t->stride);
    return row[c];
}

// Setear elemento float32
static inline void tensor_set_f32(tensor_t* t, int r, int c, float val) {
    assert(t->type == TERN_F32);
    assert(r < t->rows && c < t->cols);
    float* row = (float*)((uint8_t*)t->data + r * t->stride);
    row[c] = val;
}

// Obtener elemento ternario desempaquetado como int8
static inline int8_t tensor_get_ternary(const tensor_t* t, int r, int c) {
    assert(t->type == TERN_TERNARY);
    assert(r < t->rows && c < t->cols);
    uint8_t* row = (uint8_t*)t->data + r * t->stride;
    return unpack_ternary_val(row, c);
}

// Leer archivo completo a buffer
static uint8_t* read_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: no se pudo abrir %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    rewind(f);
    uint8_t* buf = (uint8_t*)malloc(size);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    fread(buf, 1, size, f);
    fclose(f);
    *out_size = size;
    return buf;
}

#endif // TENSOR_H
