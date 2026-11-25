# Architecture

Tessera compiles tile-level tensor operations into bare-metal GPU instructions (NVIDIA PTX, Vulkan SPIR-V, Apple Metal MSL) and executes them directly via low-level driver calls without the CUDA Toolkit or vendor SDK dependencies.

## Crate Layout

- `crates/tessera-ir`: Core tile representations, tensor data types, and layout swizzling algorithms.
- `crates/tessera-backend-ptx`: Emits PTX assembly targeting Ampere/Ada/Hopper architectures, including WMMA Tensor Core instructions and fused activation pipelines.
- `crates/tessera-backend-spirv`: Generates Vulkan compute shaders (GLSL 450) and binary SPIR-V bytecode headers.
- `crates/tessera-backend-metal`: Generates Apple Silicon Metal Shading Language (MSL) with threadgroup and SIMDgroup support.
- `crates/tessera-runtime`: Dynamic driver loader (`nvcuda.dll` / `libcuda.so`), in-memory JIT module manager, non-blocking stream scheduler, multi-GPU topology discoverer, and hardware autotuner.
- `crates/tessera-py`: PyO3 C-extension bindings and PyTorch Dynamo compiler backend.
- `crates/tessera`: CLI runner and benchmark binaries.

## Memory Model & Shared Memory Swizzling

In GPU shared memory, 32 parallel threads in a warp accessing the same memory bank causes serialization (bank conflicts). Tessera applies XOR-based stride hashing on tile loads:

$$\text{bank} = \text{col} \oplus \left(\text{row} \gg \text{shift}\right) \ \& \ 0\text{x1F}$$

This maps contiguous addresses across 32 independent banks, keeping access latency at 1 cycle.

## Tensor Core Execution (WMMA)

On NVIDIA hardware, FP16 matrix operations bypass general-purpose CUDA cores and execute directly on Tensor Cores using warp-level instructions:
1. `wmma.load.a` and `wmma.load.b`: Load 16x16 tiles of FP16 values across 32 warp threads.
2. `wmma.mma`: Computes 16x16x16 matrix multiply-accumulate with FP32 accumulation registers.
3. `wmma.store.d`: Writes results to global memory.

On the target RTX 3060, WMMA execution sustains ~5.5 TFLOPS on 1024x1024 GEMM, compared to ~0.64 TFLOPS on standard CUDA FP32 cores.

## Arbitrary Matrix Bounding

For non-power-of-two or prime dimensions (e.g., 17x35x99, 137x251x73):
- Runtime handles padding to 16-element tile alignments.
- PTX kernels apply predicate registers (`setp.lt` / `@!%p`) on thread coordinates to guard global loads and prevent invalid memory reads.

## Concurrency & Collective Communications

- Streams: Tessera manages non-blocking streams (`CU_STREAM_NON_BLOCKING`) for overlapping host-to-device transfers and kernel computation.
- Events: `GpuEvent` records synchronization markers and provides microsecond-resolution hardware profiling via `cuEventElapsedTime`.
- Multi-GPU: `MultiGpuTopology` enumerates physical accelerators and supports basic collective primitives (`all_reduce_sum`, `broadcast`) for tensor-parallel partitions.
