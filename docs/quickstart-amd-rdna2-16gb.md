# Quickstart: AMD RDNA2 16GB (RX 6800/6800 XT/6900 XT)

> **Tested on:** RX 6800 16GB + Ryzen 7 5800X3D + 64GB DDR4, BeeLlama v0.3.2 (HIP build)

## Best Config: Qwen3.6-35B-A3B MoE + DFlash d4 + KVarN4

**99 tok/s counting, 78 tok/s code generation** on a single 16GB AMD GPU.

### Models

| File | Size | Source |
|------|------|--------|
| `Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf` | 18 GB | Target model (3B active params, 20 experts) |
| `qwen36-35b-a3b-dflash-IQ4_XS.gguf` | 254 MB | DFlash draft model |

Download from HuggingFace:
- Target: [byteshape/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/byteshape/Qwen3.6-35B-A3B-MTP-GGUF) (IQ4_XS recommended)
- Draft: [Anbeeld/Qwen3.6-35B-A3B-DFlash-GGUF](https://huggingface.co/Anbeeld/Qwen3.6-35B-A3B-DFlash-GGUF) → `qwen36-35b-a3b-dflash-IQ4_XS.gguf`

### Build (HIP/ROCm)

```bash
cmake -B build -DGGML_HIP=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Server Command (32K context — fastest)

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 build/bin/llama-server \
  -m models/Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf \
  --alias qwen \
  -n 32768 -fa on --no-mmap --fit on \
  --host 0.0.0.0 --port 23420 --threads 8 \
  --reasoning on --reasoning-budget 8192 \
  --reasoning-loop-guard force-close \
  -c 32768 --parallel 1 --jinja \
  --batch-size 2048 --ubatch-size 512 \
  --cache-type-k kvarn4 --cache-type-v kvarn4 --cache-ram 0 \
  --spec-type dflash \
  --spec-draft-model models/qwen36-35b-a3b-dflash-IQ4_XS.gguf \
  --spec-draft-n-max 4 --spec-branch-budget 0 \
  --spec-dflash-cross-ctx 512 --spec-draft-ngl all \
  -lv 3
```

### Server Command (256K context — full context window)

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 build/bin/llama-server \
  -m models/Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf \
  --alias qwen \
  -n 262144 -fa on --no-mmap --fit on \
  --host 0.0.0.0 --port 23420 --threads 8 \
  --reasoning on --reasoning-budget 8192 \
  --reasoning-loop-guard force-close \
  -c 262144 --parallel 1 --jinja \
  --batch-size 2048 --ubatch-size 512 \
  --cache-type-k kvarn4 --cache-type-v kvarn4 --cache-ram 0 \
  --spec-type dflash \
  --spec-draft-model models/qwen36-35b-a3b-dflash-IQ4_XS.gguf \
  --spec-draft-n-max 4 --spec-branch-budget 0 \
  --spec-dflash-cross-ctx 512 --spec-draft-ngl all \
  -lv 3
```

> **256K context fits entirely in 16GB VRAM** thanks to KVarN4 (4-bit KV cache). Speed drops ~20% vs 32K due to larger KV overhead per decode step.

## Alternative: MTP (no extra model needed)

If you don't want to download a separate draft model, MTP heads are built into the GGUF:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 build/bin/llama-server \
  -m models/Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf \
  --alias qwen \
  -n 32768 -fa on --no-mmap --fit on \
  --host 0.0.0.0 --port 23420 --threads 8 \
  --reasoning on --reasoning-budget 8192 \
  --reasoning-loop-guard force-close \
  -c 32768 --parallel 1 --jinja \
  --batch-size 2048 --ubatch-size 512 \
  --cache-type-k kvarn4 --cache-type-v kvarn4 --cache-ram 0 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
  -lv 3
```

## Benchmark Results (fresh sweep, consistent methodology)

### 35B MoE — All Configs (32K context)

| Config | Counting t/s | Code t/s | Acceptance | Quality |
|--------|-------------|----------|------------|---------|
| Baseline (q4_0 KV, no spec) | 61.3 | 61.4 | — | — |
| KVarN4 only | 63.3 | 62.7 | — | 11/11 |
| KVarN4 + MTP (n3, p0.75) | 86.0 | 73.0 | 94% | 11/11 |
| KVarN4 + DFlash d8 | 82.0 | 73.5 | 52% | 11/11 |
| **KVarN4 + DFlash d4** | **99.2** | **78.4** | **63%** | **11/11** |
| q4_0 KV + DFlash d8 | 76.0 | 75.0 | 56% | 11/11 |

### 27B Dense — All Configs (32K context)

| Config | Counting t/s | Code t/s | Notes |
|--------|-------------|----------|-------|
| Baseline (q4_0 KV) | 18.8 | 18.8 | Dense, all 27B active |
| KVarN4 | 18.4 | 19.0 | Same speed |
| TCQ (turbo3_tcq, dequant→F16) | 22.9 | 23.1 | +22% vs baseline |

### Context Window Scaling (DFlash d4 + KVarN4)

| Context | Counting t/s | Code t/s | Fits in 16GB |
|---------|-------------|----------|-------------|
| 32K | 99.2 | 78.4 | Yes |
| 65K | 62.8 | 54.3 | Yes |
| 130K | 66.4 | 55.1 | Yes |
| **256K** | **58.5** | **55.8** | **Yes** |

> All context sizes up to the model's maximum 256K window fit in 16GB VRAM. For best interactive speed use 32K; for long-context workloads 256K is viable at ~55-60 t/s.

### Why 35B MoE over 27B Dense

| Model | Active params | Best t/s | Speed ratio |
|-------|--------------|----------|-------------|
| 35B MoE (20 experts, 3 active) | 3B | 78–99 | **3.4–4.2x** |
| 27B Dense | 27B | 19–23 | 1.0x |

The 35B MoE only activates 3 of 20 experts per token, making it 3-4x faster than the 27B dense despite being a "larger" model.

### Key Finding: DFlash d4 > d8

Shorter draft horizon (4 vs 8) wins because:
- Higher acceptance rate (63% vs 52%) — fewer wasted drafts
- Less compute per draft cycle — faster turnaround
- Works better on entropy-varied workloads (both counting and code)

## ROCm-Specific Notes

- `HSA_OVERRIDE_GFX_VERSION=10.3.0` required for RDNA2 (gfx1030)
- `--spec-draft-ngl all` mandatory — CPU draft placement silently fails on AMD (host memory not GPU-accessible)
- `--no-mmap` recommended on ROCm for stability
- `--fit on` auto-adjusts GPU layers to fit VRAM
- Do NOT use `--n-cpu-moe` with `--fit on` (conflicts with tensor_buft_overrides)
- KVarN4 is NOT classified as "turbo" KV — allows CPU KV fallback (critical for 16GB)
- TCQ/turbo KV requires 100% GPU KV — won't fit 35B + DFlash in 16GB
- DFlash GPU ring works on ROCm (~40 MB overhead for 5 capture layers)

## Quality Verification

11/11 factual recall with KVarN4 + DFlash (tested with multi-fact context). KVarN4 has 99.74% mean precision vs F16 KV.

## What Doesn't Work

| Config | Problem |
|--------|---------|
| 27B Dense + DFlash | OOM — 15.7GB + 2.4GB > 16GB |
| 27B DFlash draft on 35B target | n_embd mismatch (5120 vs 2048) |
| DFlash + MTP combo | DFlash takes exclusive control, MTP ignored |
| TCQ/turbo KV on 35B | Requires all KV on GPU, won't fit |
| CPU draft (`--spec-draft-ngl 0`) | Silent failure — host memory not GPU-accessible on AMD |
