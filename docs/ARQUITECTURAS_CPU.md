# Arquitecturas Eficientes para Inferencia en CPU

> Resumen de investigación sobre arquitecturas alternativas a Transformer
> que corren eficientemente en CPUs débiles (SSE4.2, sin AVX2).

---

## 1. MatMul-Free LM (Paper: arXiv:2406.02528)

### Concepto
Elimina TODAS las multiplicaciones de matrices de un LLM:
- **Pesos ternarios {-1,0,+1}** en capas lineales (BitLinear)
- **HGRN** reemplaza atención (RNN con compuertas, O(n) en vez de O(n²))
- **0 multiplicaciones float** en inferencia

### Cómo funciona HGRN
```
h_t = forget_gate * h_{t-1} + input_gate * x_t
```
- Forget gate con límite inferior aprendible
- Capas bajas → olvidan rápido (contexto corto)
- Capas altas → recuerdan más (contexto largo)
- Sin softmax, sin Q@K^T, sin atención

### Ventajas para CPU
| Operación | Transformer | MatMul-Free |
|---|---|---|
| Atención | Q@K^T → O(n²) float | HGRN → O(n) sumas |
| MLP | GELU + MatMul float | Gates ternarios |
| Pesos | Float32 | {-1,0,+1} → `_mm_sign_epi8` |

### Resultados
- 61% menos memoria en training
- >10× menos memoria en inferencia
- Scaling law *más pronunciada* que Transformer++
- Ideal para CPUs sin AVX2 (solo SSE4.2)

---

## 2. LFM2 — Liquid Foundation Models v2 (Liquid AI)

### Arquitectura
Híbrido de **convolución corta con gating** + pocas capas de atención:

```
Bloque principal (80% de capas):
  y = Conv1D_k=3(Gate(x))  → O(n·d) lineal

Bloque de atención (20% de capas):
  y = GQA_Attention(x)      → O(n²) pero solo 6-8 capas de 16-30
```

### MoE (Mixture of Experts)
- LFM2-8B-A1B: 8.3B params totales, solo **1.5B activos** por token
- 32 expertos, selecciona Top-4 por token
- Router con sigmoide normalizado

### Modelos disponibles
| Modelo | Params | Capas | Contexto |
|---|---|---|---|
| LFM2-350M | 350M | 16 | 32K |
| LFM2-1.2B | 1.2B | 16 | 32K |
| LFM2-8B-A1B | 8.3B (1.5B activos) | 24 | 32K |

### Velocidad en CPU (Snapdragon 8 Elite, Q4_0)
- 350M: **194 tok/s** decode
- 1.2B: **70 tok/s** decode
- 8B-A1B: **49 tok/s** decode (calidad de 8B, velocidad de 1.5B)

### Por qué es relevante para nosotros
- Convoluciones 1D son amigables para CPU (buen caché)
- Sin SSMs complejas
- KV-cache mínimo
- Hardware-in-the-loop architecture search

---

## 3. RWKV (Receptance Weighted Key Value)

### Concepto
Fusión RNN-Transformer:
- Training: paralelizable como Transformer
- Inferencia: O(n) como RNN
- Estado recurrente con decaimiento aprendible por canal

### Ecosistema CPU (el más maduro)
- `rwkv.cpp` basado en ggml (mismo motor que llama.cpp)
- Cuantización: FP32, FP16, INT8, INT5, INT4
- Soporte: GGUF, Ollama, WebGPU, WASM, mobile

### Velocidad estimada en Celeron N4020 (SSE4.2)
| Modelo | Cuantización | tok/s estimado |
|---|---|---|
| RWKV-169M | Q4_0 | 3-5 |
| RWKV-430M | Q4_0 | 1-2 |
| RWKV-1.5B | Q4_0 | 0.3-0.8 |

---

## 4. Mamba (State Space Models)

### Concepto
SSM (State Space Model) selectivo:
- Parámetros A, B, C son función del input
- Training: algoritmo paralelo hardware-aware
- Inferencia: RNN pura `h_t = A·h_{t-1} + B·x_t`

### Problemas para nuestro caso
- **No hay ecosistema CPU maduro** (requiere CUDA)
- `bitmamba.cpp` requiere AVX2 (N4020 no tiene)
- Sin soporte GGUF
- Sin cuantización CPU estándar

### Veredicto: ❌ No viable para N4020 hoy

---

## 5. Comparativa para nuestro proyecto

| Arquitectura | Madurez CPU | SSE4.2 | tok/s (350M, est.) | Implementable en C |
|---|---|---|---|---|
| **HGRN (MatMul-Free)** | Académica | ✅ | 30-100 | ✅ (puras sumas) |
| **LFM2** | Producción | ✅ | 50-200 | ⚠️ Convoluciones complejas |
| **RWKV** | Producción | ✅ | 3-5 | ⚠️ Estado recurrente grande |
| **Mamba** | Experimental | ❌ | - | ❌ |

### Conclusión
**MatMul-Free LM (HGRN + ternarios)** es la mejor opción para nuestro motor C custom:
- Ya tenemos los ternarios implementados
- HGRN es simple de implementar en C
- Sin multiplicaciones float
- O(n) en inferencia
- Benchmarks prometedores

---

## Referencias

- MatMul-Free LM: https://arxiv.org/abs/2406.02528
- HGRN: https://arxiv.org/abs/2311.04823
- LFM2: https://www.liquid.ai/blog/lfm2-advancing-open-science-in-ai
- RWKV: https://github.com/BlinkDL/RWKV-LM
- Mamba: https://arxiv.org/abs/2312.00752
- bitnet.cpp (Microsoft): https://github.com/microsoft/BitNet
