# Benchmark — May 9, 2026 (1B–4B models)

**Hardware:** Oracle A1 ARM64, 4 cores, NEON SIMD
**Config:** 256 ctx, 4 threads, temp=0.0

## Models tested

| Model | Arch | Size | Quant | File |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct | Dense | 1B | Q4_K_S | `llama-3.2-1b-q4ks.gguf` |
| Llama-3.2-3B-Instruct | Dense | 3B | Q4_K_M | `llama-3.2-3b-q4km.gguf` |
| Phi-3.5-mini-instruct | Dense | 3.8B | Q4_K_M | `phi-3.5-mini-3.8b-q4km.gguf` |
| Qwen2.5-3B-Instruct | Dense | 3B | Q4_K_M | `qwen2.5-3b-q4km.gguf` |
| Qwen2.5-MoE 2×1.5B | MoE | 2×1.5B | Q4_K_M | `qwen25-moe-2x1.5b-q4km.gguf` |

---

## 6-Config Sweep (16 tokens generated)

| Model | Base t/s | TS | Spec | Spec+TS | PI | PI+TS |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B | 2.69 | 3.12× | 1.13× | 7.77× | 1.48× | **8.71×** |
| Llama-3.2-3B | 3.18 | 0.91× | **1.51×** | 1.21× | 0.88× | 0.93× |
| Phi-3.5-mini | 3.08 | 0.97× | 0.95× | 0.87× | 1.08× | **1.91×** |
| Qwen2.5-3B | 3.84 | 0.40× | 0.49× | 0.99× | 0.42× | **1.18×** |
| Qwen2.5-MoE | 2.64 | 1.09× | 0.90× | 0.67× | **2.05×** | 1.57× |

### Notes

- **Llama-3.2-1B**: Spec+TS and PI+TS both exceptional (7.77× and 8.71×). The batch decode path with TurboSparse is extremely efficient on this small model's FFN structure.
- **Llama-3.2-3B**: Speculative alone wins (1.51×). Adding TS hurts. PI also regresses. Larger dense models respond differently than smaller ones.
- **Phi-3.5-mini**: PowerInfer is the only individual win (1.08×), but PI+TS is the clear best at 1.91× — TS amplifies PI significantly.
- **Qwen2.5-3B**: Most challenging model. PI+TS barely positive (1.18×). All other configs regress. High baseline variance (3.5–8.9 t/s cold vs warm) makes measurements noisy.
- **Qwen2.5-MoE**: PowerInfer alone (2.05×) beats PI+TS (1.57×). Adding TS overhead hurts MoE on this platform.

---

## Auto-Tune Results (32 tokens generated)

| Model | Baseline | Auto-Tune | Speedup |
|---|---:|---:|---:|
| Llama-3.2-1B | 5.09 t/s | 6.05 t/s | 1.19× |
| Llama-3.2-3B | 4.27 t/s | 4.16 t/s | 0.97× |
| Phi-3.5-mini | 3.11 t/s | 2.39 t/s | 0.77× |
| Qwen2.5-3B | 4.43 t/s | 3.58 t/s | 0.81× |
| Qwen2.5-MoE | 3.64 t/s | 3.82 t/s | 1.05× |

**Note:** Auto-tune regresses on larger dense models (3B+). The heuristic is biased toward small models where PI+TS dominates. For larger models, manual tuning (Speculative for Llama-3.2-3B, PI+TS for Phi-3.5-mini) significantly outperforms auto-select.

---

## Key Findings

1. **PI+TS is most reliable across model sizes** — works on 1B, 3B, and 3.8B dense models, though magnitude varies (1.18×–8.71×). Small models benefit more.

2. **Speculative decoding shifts optimal config as model grows** — wins on mid-size 3B (1.51×) but loses on smaller 1B (1.13×) and larger 3.8B (0.95×). The sweet spot is model-dependent.

3. **MoE + PowerInfer = natural fit** — 2.05× with PI alone. MoE's sparse expert routing mirrors PowerInfer's hot/cold neuron classification. No TS needed.

4. **Baseline variance is extreme** — cold page cache gives 2–3× lower t/s than warm. Always compare speedup ratios, not absolute values, across sessions.

5. **Qwen2.5-3B is the hardest model** — every optimization regresses except PI+TS barely (1.18×). Qwen's FFN structure may have poor activation sparsity alignment.

6. **Auto-tune is conservative** — safe 1.05–1.19× gains for small/MoE models, but regressions on larger models. Manual config selection recommended for 3B+.

---

## Recommended Config Per Model

| Model | Best Config | Expected Speedup | Auto-Tune? |
|---|---|---|---|
| Llama-3.2-1B | PI+TS (PI=0.20, TS=0.05) | ~8.7× | ✅ OK |
| Llama-3.2-3B | Speculative | ~1.5× | ❌ manual |
| Phi-3.5-mini | PI+TS (PI=0.15, TS=0.05) | ~1.9× | ❌ manual |
| Qwen2.5-3B | PI+TS (PI=0.10, TS=0.05) | ~1.2× | ❌ manual |
| Qwen2.5-MoE | PowerInfer (PI=0.15) | ~2.0× | ✅ OK |

---

## Disk usage after cleanup

| File | Size |
|---|---:|
| Llama-3.2-3B-Instruct-Q4_K_M.gguf | 2.02 GB |
| Phi-3.5-mini-instruct-Q4_K_M.gguf | 2.39 GB |
| Qwen2.5-3B-Instruct-Q4_K_M.gguf | 1.93 GB |
| Qwen2.5-MOE-2×1.5B-Q4_K_M.gguf | 2.40 GB |
| **Total** | **~8.7 GB** |

Deleted: TinyLlama (638+609 MB), Phi-2 (1.8 GB), Llama-3.2-1B (740 MB), Qwen MoE Q2_K (1.5 GB). Reclaimed ~6 GB.