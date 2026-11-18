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
    print(f"engine version: {ts.version()}")
    dev = ts.device_info()
    print(f"device: {dev.get('device_name')} (driver: {dev.get('driver_version')})")

    # 1. Arbitrary Dynamic Matrix Shape Bounding
    print("\n--- 1. Arbitrary Shape Dynamic Bounding Test ---")
    test_shapes = [
        (17, 35, 99),     # Odd, non-multiple of 16
        (137, 251, 73),   # Prime dimensions
        (1000, 1000, 1000), # Round non-power-of-2
    ]

    for m, n, k in test_shapes:
        np.random.seed(42)
        a = np.random.randn(m, k).astype(np.float32) * 0.1
        b = np.random.randn(k, n).astype(np.float32) * 0.1

        expected = np.matmul(a, b)
        actual_list = ts.matmul_arbitrary(a.flatten().tolist(), b.flatten().tolist(), m, n, k)
        actual = np.array(actual_list, dtype=np.float32).reshape((m, n))

        diff = float(np.max(np.abs(expected - actual)))
        status = "PASSED" if diff < 1e-3 else "FAILED"
        print(f"  matrix {m}x{n}x{k}: max diff {diff:.8f} -> [{status}]")

    # 2. Asynchronous Streams & Events Concurrency
    print("\n--- 2. Async Non-Blocking Streams & Event Concurrency ---")
    async_res = ts.test_async_streams()
    print(f"  streams created: {async_res['streams_created']}")
    print(f"  concurrency mode: {async_res['concurrency']}")
    print(f"  event timing elapsed: {async_res['elapsed_ms']:.3f} ms")

    # 3. Multi-GPU Cluster Topology & Collective Communication
    print("\n--- 3. Multi-GPU Topology & Collective Communication ---")
    topo = ts.multi_gpu_topology()
    print(f"  detected GPUs: {topo['device_count']}")
    for d in topo['devices']:
        print(f"    GPU {d['ordinal']}: {d['name']}")

    # Test Collective All-Reduce across 4 simulated ranks
    num_ranks = 4
    elements = 8
    rank_buffers = [
        [float(rank + 1) for _ in range(elements)]
        for rank in range(num_ranks)
    ]
    # Ranks: [1..1], [2..2], [3..3], [4..4]. Sum should be [10..10]
    reduced = ts.collective_all_reduce(rank_buffers)
    print(f"  collective all-reduce (4 ranks, {elements} elems):")
    print(f"    input ranks: {[b[0] for b in rank_buffers]}")
    print(f"    reduced output: {reduced[0][:4]} (expected all 10.0)")
    assert all(abs(v - 10.0) < 1e-5 for v in reduced[0]), "All-reduce mismatch!"
    print("    [ALL-REDUCE VERIFIED]")

if __name__ == "__main__":
    main()
