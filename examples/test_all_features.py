import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "target", "release")))

try:
    import tessera as ts
except ImportError as e:
    print(f"error importing tessera: {e}")
    sys.exit(1)

def f32_to_half_u16(arr: np.ndarray) -> list:
    return np.frombuffer(arr.astype(np.float16).tobytes(), dtype=np.uint16).tolist()

def main():
    print(f"engine version: {ts.version()}")
    dev = ts.device_info()
    print(f"device: {dev.get('device_name')} (driver: {dev.get('driver_version')})")

    # 1. Multi-Vendor Codegen: PTX, Vulkan SPIR-V, Apple Metal
    print("\n--- 1. Multi-Vendor Codegen (PTX, Vulkan, Apple Metal) ---")
    ptx = ts.compile_ptx("tessera_gemm", 64, 64, 16)
    vulkan = ts.compile_vulkan("tessera_vulkan_gemm")
    metal = ts.compile_metal("tessera_metal_gemm")
    print(f"ptx emitted: {len(ptx)} bytes")
    print(f"vulkan compute shader emitted: {len(vulkan)} bytes")
    print(f"apple metal msl shader emitted: {len(metal)} bytes")

    # 2. Tensor Core FP16 Hardware Execution & Pure Benchmark (RTX 3060 WMMA)
    print("\n--- 2. Tensor Core (WMMA FP16) Hardware Execution ---")
    m, n, k = 1024, 1024, 1024
    np.random.seed(42)
    a_f32 = np.random.randn(m, k).astype(np.float32) * 0.1
    b_f32 = np.random.randn(k, n).astype(np.float32) * 0.1

    a_u16 = f32_to_half_u16(a_f32)
    b_u16 = f32_to_half_u16(b_f32)

    out = ts.matmul_tensor_core(a_u16, b_u16, m, n, k)
    c_actual = np.array(out, dtype=np.float32).reshape((m, n))
    c_expected = np.matmul(a_f32.astype(np.float16), b_f32.astype(np.float16)).astype(np.float32)
    diff = float(np.max(np.abs(c_expected - c_actual)))

    bench = ts.bench_tensor_core(1024, 1024, 1024, 50)
    print(f"tensor core 1024x1024: latency {bench['latency_us'] / 1000:.3f} ms, throughput {bench['tflops']:.2f} tflops")
    print(f"tensor core numerical diff vs cpu reference: {diff:.6f}")

    # 3. Transformer Primitive: RMSNorm
    print("\n--- 3. Transformer Primitive: RMSNorm ---")
    num_rows = 4
    hidden_dim = 4096
    eps = 1e-5
    x = np.random.randn(num_rows, hidden_dim).astype(np.float32)
    weight = np.random.randn(hidden_dim).astype(np.float32)

    expected_rms = x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps) * weight
    actual_rms_list = ts.rmsnorm(x.flatten().tolist(), weight.tolist(), num_rows, hidden_dim, eps)
    actual_rms = np.array(actual_rms_list, dtype=np.float32).reshape((num_rows, hidden_dim))

    rms_diff = float(np.max(np.abs(expected_rms - actual_rms)))
    print(f"rmsnorm ({num_rows}x{hidden_dim}): max diff: {rms_diff:.8f}")

    # 4. Transformer Primitive: SwiGLU
    print("\n--- 4. Transformer Primitive: SwiGLU ---")
    dim = 4096
    sw_x = np.random.randn(dim).astype(np.float32)
    sw_g = np.random.randn(dim).astype(np.float32)

    def swish(v):
        return v / (1.0 + np.exp(-v))

    expected_swiglu = swish(sw_x) * sw_g
    actual_swiglu_list = ts.swiglu(sw_x.tolist(), sw_g.tolist())
    actual_swiglu = np.array(actual_swiglu_list, dtype=np.float32)

    sw_diff = float(np.max(np.abs(expected_swiglu - actual_swiglu)))
    print(f"swiglu ({dim}): max diff: {sw_diff:.8f}")

    # 5. Hardware Auto-Tuner
    print("\n--- 5. Hardware Auto-Tuner ---")
    profile = ts.autotune_gemm(1024, 1024, 1024)
    print(f"optimal profile for 1024x1024:")
    print(f"  use_tensor_cores: {profile['use_tensor_cores']}")
    print(f"  latency: {profile['latency_us'] / 1000:.3f} ms")
    print(f"  throughput: {profile['tflops']:.2f} tflops")

if __name__ == "__main__":
    main()
