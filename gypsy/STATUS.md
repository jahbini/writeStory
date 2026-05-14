# Gypsy Status

Current date context: May 13, 2026.

## Current Outcome

The native gypsy generation path is working end-to-end with adapters.

It is the active speed/memory path used by the `prompt_ite` recipe via the new
`generate_prompt_gypsy_ite` step and by the internal lifecycle tests under
`test/helpers/native_64_mlx_lazy_generation_probe.coffee`.

Manual direct-Metal staging (the previously-recorded failed path) remains
rejected — see "Rejected Path" below — but it is no longer the live story.

## Measured Result (May 13, 2026)

Qwen3-4B Instruct (quantized 4-bit, group 64) on Apple M2:

- generation speed: ~9–10.5 tok/s at 64 tokens (was 1.37 tok/s baseline)
- memory peak: ~1.9–2.0 GB resident (was ~6 GB baseline)
- wall clock for 64 tokens (incl. model load + prefill): ~16 s
- correctness: matches MLX output in fp32 mode; produces a coherent bf16
  variant in compute mode
- adapter-trained generation verified: "What is the town that Southwick and
  Tommy live in?" → "Southwick and Tommy live in Portland, Oregon."

## What Got Us Here

Four layered changes inside `metal/metal_llm_node.cpp`:

1. **Embedding lookup kernel.** Replaced `dequantize(ew) + take(tok_idx, 0)`
   (which materialized the full 151936×2560 float32 matrix ≈ 1.56 GB per
   token) with a custom Metal kernel that dequantizes only one row of the
   quantized embedding into 2560 floats. This is the single biggest memory
   win.
2. **Preallocated expanded KV cache (`mlx_prealloc_kv` backend).** Each layer
   keeps `[1, q_heads, capacity, head_dim]` resident and uses
   `slice_update` per token instead of growing a `concatenate` chain. This
   was the single biggest speed win (≈13× over the original).
3. **bfloat16 hidden states and KV cache.** All in-layer activations
   (attn_norm, q/k/v projections, q_normed/k_normed, q_rope/k_rope, mlp_norm,
   gate/up/down, residuals, final norm) use `compute_dtype = bfloat16`, and
   the preallocated KV cache is stored as bfloat16. ~15 % additional speed,
   smaller activation bandwidth, halved KV bandwidth in SDPA.
4. **MLX cache-pool cap.** `mlx::core::set_cache_limit(512 MB)` is called
   right after backend selection so that any large transient (e.g. the
   intermediate held during a quantized matmul) is returned to the OS
   instead of being parked in the MLX free pool.

The fused `quantized_matmul + rms_norm + RoPE` kernel exists in the source
(`fused_qk_proj_norm_rope` lambda) and is correctness-preserving, but it is
not currently invoked: a hand-rolled matmul on one threadgroup per head with
128 threads cannot beat MLX's tiled SIMD-group quantized matmul. See
`GPT/gypsy_strategy.md` for the standard derived from that result.

## CoffeeScript Surface

`gypsy/session_api.coffee` exposes a thin wrapper around the native addon at
`metal/metal_llm.node`. The high-level entry is `api.lifecycle({...})` which
performs load → warm → generate → free in one call. The granular methods
(`loadModelResident`, `loadTokenizer`, `loadAdapter`, `createSession`,
`warmSession`, `generate`, `freeSession`, `unloadAdapter`,
`unloadTokenizer`, `unloadModel`) remain available for the probes.

## Recipe Wiring

`generate_prompt_gypsy_ite` (config in `config/prompt_ite.yaml`, script in
`scripts/prompt_ite/generate_prompt_gypsy_ite.coffee`) is the production
entry point. It consumes the standard `prompt_ite` knobs (`prompt_text`,
`quantized_model_dir`, `mlx:` controls, optional `adapter_dir`) and emits
`prompt_gypsy_raw / prompt_gypsy_meta / prompt_gypsy_text` artifacts.

## Rejected Path (historical)

Still rejected, recorded for negative evidence only:

- staged terminal direct-Metal kernels
- fused MLP tail
- full-layer resident helper
- terminal-stack helper based on CPU-visible shared buffers
- any path that copies weights during generation
- any path that returns hidden vectors to CPU between layers
- any path that validates each GPU segment in the hot loop
- hand-rolled tiled matmul to replace `quantized_matmul` (the rust project's
  experience and the current `fused_qk_proj_norm_rope` benchmark both confirm
  MLX's matmul is already very good)

These remain as correctness artifacts or teaching examples only. Do not rerun
them as speed benchmarks.

## Relationship To Rusty

Rusty and Gypsy live on sibling branches:

- on `main` (this branch): `gypsy/` is populated, `rusty/` is empty.
  Gypsy is the production speed path here.
- on the `rusty` branch: `rusty/` is populated, `gypsy/` is empty.
  Rusty is the earlier teaching/correctness path for the token journey
  through the model.

The two paths are independent — nothing in the gypsy codepath depends on
Rusty's code, and the production build on `main` does not need Rusty to
be present. References to Rusty in this directory's docs (and in
`GPT/gypsy_strategy.md`) are cross-branch architectural lessons kept on
`main` on purpose. To read Rusty's actual source, `git checkout rusty`.
