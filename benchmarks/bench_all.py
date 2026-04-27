"""TurStack Performance Benchmarks"""

import time
import numpy as np

def bench_sparse_matmul(seq, hidden, output):
    from vibeblade.sparse import sparse_matmul, compute_sparsity, predict_activations
    
    activations = np.random.randn(seq, hidden).astype(np.float32)
    weights = np.random.randn(hidden, output).astype(np.float32)
    mask = predict_activations(activations.reshape(-1), threshold=0.0)
    
    # Dense baseline
    start = time.perf_counter()
    for _ in range(100):
        np.dot(activations, weights)
    dense_time = (time.perf_counter() - start) / 100
    
    # Sparse
    start = time.perf_counter()
    for _ in range(100):
        sparse_matmul(activations, weights, mask)
    sparse_time = (time.perf_counter() - start) / 100
    
    sparsity = compute_sparsity(activations.reshape(-1))
    speedup = dense_time / sparse_time if sparse_time > 0 else 0
    print(f"Sparse MatMul: seq={seq} hidden={hidden} output={output}")
    print(f"  Dense: {dense_time*1000:.3f}ms | Sparse: {sparse_time*1000:.3f}ms | Sparsity: {sparsity:.1%} | Speedup: {speedup:.2f}x")
    return speedup

def bench_quant(hidden, group_size):
    from vibeblade.quant import quantize_4bit, dequantize_4bit, quantization_error
    
    weights = np.random.randn(hidden).astype(np.float32)
    
    start = time.perf_counter()
    for _ in range(100):
        packed, scales, rotors = quantize_4bit(weights, group_size)
    quant_time = (time.perf_counter() - start) / 100
    
    start = time.perf_counter()
    for _ in range(100):
        reconstructed = dequantize_4bit(packed, scales, rotors, hidden)
    dequant_time = (time.perf_counter() - start) / 100
    
    error = quantization_error(weights, reconstructed)
    compression = hidden * 4 / len(packed)  # float32 -> 4-bit
    
    print(f"RotorQuant: hidden={hidden} group_size={group_size}")
    print(f"  Quantize: {quant_time*1000:.3f}ms | Dequant: {dequant_time*1000:.3f}ms | RMS error: {error:.4f} | Compression: {compression:.1f}x")
    return error

def bench_kv_cache(layers, heads, head_dim, seq_len):
    from vibeblade.cache import KVCache
    
    cache = KVCache(layers, heads, head_dim, seq_len)
    k = np.random.randn(heads, 1, head_dim).astype(np.float16)
    v = np.random.randn(heads, 1, head_dim).astype(np.float16)
    
    # Write benchmark
    start = time.perf_counter()
    for pos in range(seq_len):
        for layer in range(layers):
            cache.update(layer, k, v, pos)
    write_time = time.perf_counter() - start
    
    # Read benchmark
    start = time.perf_counter()
    for layer in range(layers):
        cache.get(layer)
    read_time = time.perf_counter() - start
    
    mem_mb = cache.memory_usage_bytes() / (1024 * 1024)
    print(f"KV Cache: layers={layers} heads={heads} head_dim={head_dim} seq={seq_len}")
    print(f"  Write: {write_time*1000:.3f}ms | Read: {read_time*1000:.3f}ms | Memory: {mem_mb:.1f}MB")
    return mem_mb

def bench_inference(hidden, layers, heads, seq_len):
    from vibeblade.transformer import forward
    
    token_ids = np.random.randint(0, 1000, seq_len)
    weights = {
        'embed': np.random.randn(1000, hidden).astype(np.float32) * 0.02,
    }
    
    # Warmup
    forward(token_ids, weights, num_layers=layers, hidden_dim=hidden, num_heads=heads)
    
    start = time.perf_counter()
    for _ in range(10):
        forward(token_ids, weights, num_layers=layers, hidden_dim=hidden, num_heads=heads)
    total = (time.perf_counter() - start) / 10
    
    tps = seq_len / total
    print(f"Inference: hidden={hidden} layers={layers} heads={heads} seq={seq_len}")
    print(f"  Time: {total*1000:.3f}ms | Throughput: {tps:.1f} tokens/sec")
    return tps

if __name__ == "__main__":
    print("=" * 60)
    print("VibeBlade Benchmarks")
    print("=" * 60)
    print(f"Backend: {'C++ AVX-512' if __import__('vibeblade')._CPP_BACKEND else 'NumPy'}")
    print()
    
    bench_sparse_matmul(128, 512, 512)
    print()
    bench_quant(4096, 32)
    print()
    bench_kv_cache(32, 32, 128, 2048)
    print()
    bench_inference(256, 4, 8, 32)
    print()
    print("=" * 60)
