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

def main():
    print(f"tessera version: {ts.version()}")
    dev = ts.device_info()
    print(f"device: {dev.get('device_name')} (driver: {dev.get('driver_version')})")

    ptx = ts.compile_ptx("py_gemm_kernel", 64, 64, 16)
    print(f"ptx size: {len(ptx)} bytes")

    vulkan_glsl = ts.compile_vulkan("py_vulkan_gemm")
    print(f"vulkan shader size: {len(vulkan_glsl)} bytes")

    m, n, k = 64, 64, 64
    np.random.seed(42)
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)

    start_cpu = time.perf_counter()
    expected = np.matmul(a, b)
    cpu_us = (time.perf_counter() - start_cpu) * 1e6

    start_gpu = time.perf_counter()
    out = ts.matmul(a.flatten().tolist(), b.flatten().tolist(), m, n, k)
    gpu_us = (time.perf_counter() - start_gpu) * 1e6

    actual = np.array(out, dtype=np.float32).reshape((m, n))
    max_err = float(np.max(np.abs(expected - actual)))

    print(f"gemm {m}x{n}x{k}: cpu {cpu_us:.1f}us, gpu {gpu_us:.1f}us, max diff: {max_err:.8f}")

    bench = ts.bench_gemm(1024, 1024, 1024, 20)
    print(
        f"bench 1024x1024: {bench['latency_us'] / 1000:.3f} ms, "
        f"{bench['gflops'] / 1000:.2f} tflops"
    )

if __name__ == "__main__":
    main()
