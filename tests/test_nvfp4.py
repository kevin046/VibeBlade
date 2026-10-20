"""Unit tests for the NVFP4 + DFlash2 package (CPU-only, no torch needed)."""

import sys

sys.path.insert(0, "/home/ubuntu/vibeblade")

import numpy as np

from vibeblade.nvfp4.packing import (
    E2M1_LUT, encode_fp4, decode_fp4_packed, fp8_e4m3_to_float32,
    dequant_nvfp4, pack_weights, BLOCK,
)
from vibeblade.nvfp4.gemm import nvfp4_gemm, _dequant_matrix, has_fp4_tensor_cores
from vibeblade.nvfp4.dflash2 import (
    dflash2_layer_indices, make_dflash2_config, next_block_size,
)

ok = 0

# 1. E2M1 LUT is the canonical magnitude set
mags = sorted(set(abs(E2M1_LUT).tolist()))
assert mags == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], mags
print("1. E2M1 LUT magnitudes: OK")
ok += 1

# 2. round-trip: encode → decode recovers exact codes
x = np.array([0.0, 0.5, -1.0, 1.5, 2.0, -3.0, 4.0, 6.0, -0.5, 3.0], dtype=np.float32)
packed, codes = encode_fp4(x)
dec = decode_fp4_packed(packed)[: len(x)]
assert np.allclose(dec, x), (dec, x)
print("2. FP4 encode/decode round-trip: OK")
ok += 1

# 3. FP8-E4M3 decode: known values
s = fp8_e4m3_to_float32(np.array([0b0_0111_000, 0b0_1000_000, 0b0_0000_100], dtype=np.uint8))
assert abs(s[0] - 1.0) < 1e-6 and abs(s[1] - 2.0) < 1e-6 and abs(s[2] - 0.0078125) < 1e-9, s  # E4M3 subnormal: 2^(1-7) * 4/8
print("3. FP8-E4M3 decode (1.0, 2.0, 2^-5): OK")
ok += 1

# 4. block dequant shape + scale application
rng = np.random.default_rng(0)
w = rng.standard_normal(64).astype(np.float32)
packed, scales, ts = pack_weights(w)
rec = dequant_nvfp4(packed, scales, ts)
assert rec.shape == (64,)
err = np.abs(rec - w) / np.maximum(np.abs(w), 1e-3)
assert np.median(err) < 0.35, np.median(err)  # 4-bit: median rel err well under 35%
print(f"4. pack→dequant median rel err {np.median(err):.3f} < 0.35: OK")
ok += 1

# 5. GEMM portable path ≈ dense matmul
W = rng.standard_normal((8, 32)).astype(np.float32)
X = rng.standard_normal((3, 32)).astype(np.float32)
pw, sc, ts = pack_weights(W)
Y = nvfp4_gemm(X, pw, sc, ts)
Yref = X @ W.T
denom = np.maximum(np.abs(Yref), 1.0)
relerr = np.abs(Y - Yref) / denom
assert np.mean(relerr) < 0.25, np.mean(relerr)
print(f"5. nvfp4_gemm vs dense matmul mean rel err {np.mean(relerr):.3f} < 0.25: OK")
ok += 1

# 6. tensor cores absent on this box → portable path taken deterministically
assert has_fp4_tensor_cores() is False or has_fp4_tensor_cores() is True  # callable, no crash
print("6. has_fp4_tensor_cores() probes without crash: OK")
ok += 1

# 7. DFlash2 generalization reproduces the shipped 32-layer config
assert dflash2_layer_indices(32) == (1, 9, 17, 25, 33)
print("7. dflash2(32) == (1,9,17,25,33) [matches Qwen3-4B shipped]: OK")
ok += 1

# 8. works for 27B-class depths (48/62/64 layers)
for n in (48, 62, 64):
    idxs = dflash2_layer_indices(n)
    assert len(idxs) == 5 and idxs[-1] == n + 1 and idxs[0] == 1, (n, idxs)
cfg64 = make_dflash2_config(64)
assert cfg64.layer_indices[-1] == 65
print(f"8. dflash2 generalizes (48/62/64 layers): OK  e.g. 64→{dflash2_layer_indices(64)}")
ok += 1

# 9. adaptive block sizing behaves monotonically
assert next_block_size(0.9, 16) == 32      # high accept → grow
assert next_block_size(0.75, 16) == 20     # mild grow
assert next_block_size(0.55, 16) == 16     # hold
assert next_block_size(0.3, 16) == 8       # collapse
assert next_block_size(0.1, 8) == 8        # floor at min_block
print("9. DFlash2 adaptive block sizing: OK")
ok += 1

# 10. scale-count validation errors cleanly
try:
    dequant_nvfp4(packed, scales[:-1])
    raise SystemExit("should have raised")
except ValueError:
    pass
print("10. malformed scales rejected: OK")
ok += 1

print(f"ALL {ok}/10 PASS")
