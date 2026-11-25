# Benchmark Results

Hardware: NVIDIA GeForce RTX 3060 12GB (Driver 617.14, sm_86)
OS: Windows 11 Pro 64-bit

## Single Precision (FP32) & Tensor Core (FP16) Results

| Matrix (MxNxK) | Execution Engine | Precision | Memory | Latency | Sustained TFLOPS | Max Error (vs CPU Reference) |
|---|---|---|---|---|---|---|
| 64x64x64 | CUDA Cores | FP32 | 0.05 MB | 5.0 us | 0.10 | 9.5e-7 |
| 128x128x128 | CUDA Cores | FP32 | 0.19 MB | 12.1 us | 0.35 | 1.9e-6 |
| 256x256x256 | CUDA Cores | FP32 | 0.75 MB | 73.5 us | 0.46 | 3.8e-6 |
| 512x512x512 | CUDA Cores | FP32 | 3.00 MB | 397.3 us | 0.68 | Verified |
| 1024x1024x1024 | CUDA Cores | FP32 | 12.00 MB | 3.35 ms | 0.64 | Verified |
| 2048x2048x2048 | CUDA Cores | FP32 | 48.00 MB | 26.93 ms | 0.64 | Verified |
| 512x512x512 | CUDA Cores + Fused ReLU | FP32 | 3.00 MB | 396.5 us | 0.68 | Verified |
| 512x512x512 | CUDA Cores + Fused GELU | FP32 | 3.00 MB | 389.3 us | 0.69 | Verified |
| 1024x1024x1024 | CUDA Cores + Fused GELU | FP32 | 12.00 MB | 3.16 ms | 0.68 | Verified |
| 1024x1024x1024 | Tensor Cores (WMMA) | FP16/FP32 Acc | 8.00 MB | 0.38 ms | 5.58 | 4.9e-4 |

## Transformer Primitives Results

| Operation | Input Shape | Latency | Max Absolute Error vs FP32 Reference |
|---|---|---|---|
| RMSNorm | 4 x 4096 (Batch x Hidden) | 18.2 us | 9.5e-7 |
| SwiGLU | 4096 elements | 4.1 us | 4.8e-7 |

## Notes

- Tensor Core WMMA execution achieves a **8.7x speedup** over CUDA Cores FP32 execution (5.58 TFLOPS vs 0.64 TFLOPS).
- Zero CUDA Toolkit dependency: all PTX microcode is dynamically assembled and loaded by `nvcuda.dll` directly in memory within ~0.27 ms.
- Shared memory access is swizzled with XOR stride hashing to eliminate bank conflicts.
- Hardware Auto-Tuner dynamically queries the device and selects the peak execution profile for any tensor dimensions.

## Running Benchmarks

```bash
cargo run --bin eval_suite --release
cargo run --bin wmma_bench --release
python examples/test_all_features.py
```
