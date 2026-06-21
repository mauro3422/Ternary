# SPECS — PC de Entrenamiento

## Hardware

| Componente | Especificación |
|---|---|
| **Modelo** | MSI MS-7C96 (MSI Pro) |
| **CPU** | AMD Ryzen 5 5600G @ 3.9GHz boost |
| **Arquitectura** | Zen 3 (Cezanne, 2021) |
| **Cores/Threads** | 6 cores / 12 threads |
| **Cache** | L2 3MB, L3 16MB |
| **SIMD** | SSE4.2, **AVX2**, FMA |
| **RAM** | 16 GB DDR4 @ 2667 MHz (2×8GB) |
| **GPU** | AMD Radeon RX 570 **4GB** (Polaris) |
| **Almacenamiento** | HP SSD S600 240GB (SO) + WD 1TB HDD (datos) |
| **OS** | Windows 11 Pro 23H2 |

## Capacidad de GPU

La RX 570 es una GPU dedicada con 4GB VRAM, pero **no soporta CUDA** (es AMD, no NVIDIA). Opciones para entrenar:

| Opción | Ventaja | Desventaja |
|---|---|---|
| **CPU (Ryzen 5600G)** | ✅ Ya funciona, AVX2 acelera | Más lento que GPU (~30-60 min vs ~10-30 min) |
| **DirectML (PyTorch)** | ✅ Usa GPU AMD | Experimental, setup complejo |
| **ROCm + WSL2** | ✅ Usa GPU AMD | Polaris no tiene buen soporte ROCm |
| **Dual boot Linux + ROCm** | ✅ CUDA alternativo | Mucha instalación |

**Recomendación para ahora:** Entrenar en CPU. El 5600G con AVX2 es suficientemente rápido para 10.7M params (~30 min los 5000 iters).

## Comparativa con PC Chica (objetivo de inferencia)

| Característica | PC Entrenamiento (esta) | PC Chica (N4020) |
|---|---|---|
| **CPU** | Ryzen 5 5600G ⚡ | Celeron N4020 🐢 |
| **Cores** | 6C / 12T | 2C / 2T |
| **SIMD** | **AVX2 + FMA** | Solo SSE4.2 |
| **RAM** | 16 GB | 8 GB |
| **GPU** | RX 570 4GB | Intel UHD 600 (inútil) |
| **Rol** | **Entrenar** modelos | **Correr** inferencia |
| **Velocidad training** | 30-60 min (CPU) | No aplica |

## Software

- **Python** 3.x
- **PyTorch** (CPU, sin CUDA)
- **Numpy**
- **Git**

### Para acelerar training en CPU
```python
# PyTorch usa AVX2 automáticamente en este CPU
# Para mejor rendimiento:
torch.set_num_threads(12)  # usar todos los hilos
```

## Límites de Training

| Concepto | Valor |
|---|---|
| **Modelo máximo entrenable (CPU)** | ~350M params (limitado por RAM + tiempo) |
| **Modelo máximo entrenable (si GPU funcionara)** | ~1B params (limitado por 4GB VRAM) |
| **Tiempo estimado 10.7M (CPU, 5000 iters)** | ~30 min |
| **Tiempo estimado 85M (CPU, 5000 iters)** | ~4-6 h |
| **Tiempo estimado 350M (CPU, 5000 iters)** | ~24-48 h |

---
*Creado: 20 Junio 2026*
