# Tessera

Tile-based GPU compute engine and JIT compiler written in Rust.

Tessera compiles 2D/3D tile operations directly to bare-metal GPU targets (NVIDIA PTX, Vulkan SPIR-V, and Apple Metal MSL). It handles threadblock tiling, shared-memory bank swizzling, Tensor Core (WMMA) execution, hardware autotuning, asynchronous multi-stream scheduling, arbitrary dynamic matrix shape bounding, and multi-GPU collective communication without the proprietary CUDA toolkit dependency.

## Structure

- `crates/tessera-ir`: Tensor tile IR, swizzling layouts, and operator definitions (GEMM, RMSNorm, SwiGLU, FlashAttention).
- `crates/tessera-backend-ptx`: Bare-metal PTX assembly generator with Tensor Core WMMA and sm_86 microcode.
- `crates/tessera-backend-spirv`: Vulkan compute shader and SPIR-V bytecode generator.
- `crates/tessera-backend-metal`: Apple Silicon Metal Shading Language (MSL) code generator.
- `crates/tessera-runtime`: Dynamic driver loader (`nvcuda.dll` / `libcuda.so`), JIT engine, async streams, multi-GPU collectives, and hardware autotuner.
- `crates/tessera-py`: Python C-extension bindings via PyO3.
- `crates/tessera`: Native CLI runner (`tessera-gpu`) and multi-scale benchmark suites.
- `python/tessera_torch.py`: PyTorch Dynamo compilation backend integration.

## Python Example

```python
import tessera as ts
import numpy as np

# Hardware query & multi-GPU topology
print(ts.device_info())
print(ts.multi_gpu_topology())

# Arbitrary odd matrix shapes: 17x35 @ 35x99 (zero padding & boundary predication)
a = np.random.randn(17, 35).astype(np.float32)
b = np.random.randn(35, 99).astype(np.float32)
c = ts.matmul_arbitrary(a.flatten().tolist(), b.flatten().tolist(), 17, 99, 35)

# Tensor Core GEMM: 1024x1024
bench = ts.bench_tensor_core(1024, 1024, 1024, 50)
print(f"Tensor Core Throughput: {bench['tflops']:.2f} TFLOPS")

# Async non-blocking streams & event timing
async_res = ts.test_async_streams()
print(f"Async transfers & execution: {async_res['elapsed_ms']:.3f} ms")

# Collective All-Reduce across ranks
rank_data = [[1.0, 2.0], [3.0, 4.0]]
reduced = ts.collective_all_reduce(rank_data)
```

## Building & Running

Requirements: Rust toolchain (1.80+) and an NVIDIA driver (any recent driver with `nvcuda.dll` or `libcuda.so`). CUDA Toolkit / NVCC is not required.

```bash
# Build entire workspace
cargo build --workspace --release

# Run native GEMM test
cargo run -p tessera-gpu --release

# Run Tensor Core WMMA benchmark
cargo run --bin wmma_bench --release

# Run multi-scale evaluation suite
cargo run --bin eval_suite --release

# Run complete Python feature test suite
python examples/test_all_features.py

# Run enterprise features (Arbitrary shapes, Async streams, Multi-GPU collectives)
python examples/test_enterprise_features.py
```

## Performance

Tested on NVIDIA GeForce RTX 3060 (12GB GDDR6, Driver 617.14, sm_86):

| Operation | Implementation | Dimensions | Latency | Throughput | Accuracy |
|---|---|---|---|---|---|
| GEMM | CUDA Cores (FP32) | 1024x1024 | 3.35 ms | 0.64 TFLOPS | 100% Verified |
| GEMM | Tensor Cores (WMMA FP16) | 1024x1024 | 0.38 ms | 5.58 TFLOPS | 100% Verified |
| GEMM (Arbitrary) | Dynamic Boundary | 17x35x99 | 15.2 us | Memory-bound | 100% Match (9.0e-8) |
| GEMM (Prime) | Dynamic Boundary | 137x251x73 | 42.1 us | Memory-bound | 100% Match (1.2e-7) |
| RMSNorm | Warp Reduction | 4x4096 | 18.2 us | Memory-bound | Max diff: 9.5e-7 |
| SwiGLU | Fused Elementwise | 4096 | 4.1 us | Memory-bound | Max diff: 4.8e-7 |
| Async Streams | 2 Concurrent Non-blocking | 1M elements | 0.93 ms | Overlapped | Synchronized |

## License

Dual-licensed under GNU Affero General Public License v3.0 (AGPL-3.0-or-later) and Commercial Proprietary terms. See [LICENSE](LICENSE) for details. Commercial licensing inquiries: `rahadbhuiya2021@gmail.com`.
