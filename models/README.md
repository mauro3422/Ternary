# Model Cards — Registro de Versiones

Cada modelo entrenado se documenta con su configuración, métricas y benchmarks.

---

## v1 — Transformer 10.7M Shakespeare

| Campo | Valor |
|---|---|
| **Fecha** | 20 Jun 2026 |
| **Arquitectura** | Transformer (atención cuadrática) |
| **Config** | `GPTConfig` (6L, 6H, 384d) |
| **Dataset** | Shakespeare (65 chars) |
| **Checkpoint** | `checkpoint.pt` (raíz del repo) |
| **Model binary** | `engine/model.bin` (3.03 MB) |

### Training

| Métrica | Valor |
|---|---|
| GPU | Tesla T4 (Colab) |
| Tiempo total | ~16 min |
| Iters | 5000 |
| Batch size | 32 |
| Loss inicial | 33.12 |
| Loss final | **2.507** |
| Perplejidad | 12.27 |
| LR schedule | Warmup 1000 + cosine decay |

### Inference (motor C)

| Hardware | Tok/s |
|---|---|
| Ryzen 5600G (esta PC) | **257** |
| Celeron N4020 (target) | **105** |
| RAM (I2_S) | ~12 MB |
| Carga modelo | 0.013s |
| Prefill 6 chars | 0.02s |

### Calidad de texto

Genera texto con estructura Shakespeare (mayúsculas, puntuación, guiones) pero **sin sentido**. Equivale a ~0.5M params float32 — muy pequeño para lenguaje.

### Comentarios

- Prueba de concepto funcional
- Pipeline completo verificado (Python → export → C engine)
- Base para comparar contra HGRN

---

## Plantilla para próximos modelos

```
## v2 — [Arquitectura] [Tamaño] [Dataset]

| Campo | Valor |
|---|---|
| **Fecha** | dd Mmm aaaa |
| **Arquitectura** | Transformer / HGRN |
| **Config** | `ConfigName` (nL, nH, nEmbd) |
| **Dataset** | Shakespeare / TinyStories |
| **Checkpoint** | `models/v2/checkpoint.pt` |
| **Model binary** | `models/v2/model.bin` |

### Training

| Métrica | Valor |
|---|---|
| GPU | Tesla T4 (Colab) |
| Tiempo total | X min |
| Iters | X |
| Batch size | X |
| Loss inicial | X |
| Loss final | **X** |
| Perplejidad | X |

### Inference (motor C)

| Hardware | Tok/s |
|---|---|
| Ryzen 5600G | X |
| Celeron N4020 | X |
| RAM (I2_S) | X MB |

### Calidad de texto

[Descripción de lo que genera]

### Comentarios

- [Aprendizajes, diferencias vs versiones anteriores]
```
