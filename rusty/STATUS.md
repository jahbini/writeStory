# Rusty Status

Date: 2026-05-08

Branch: `main`

## Current State

- The Rust resident bridge scaffold exists under `rusty/bridge/`.
- The bridge uses a JSONL protocol, handle tables, shutdown-state validation,
  and backend discovery commands.
- The native C++ shim exists, Rust can call into it, and the shim can link to
  the selected Homebrew MLX install.
- `verify_bridge.coffee` is the active verifier. `verify_bridge.mjs` is no
  longer the verifier entrypoint.
- The default verifier profile is `smoke`.
- The standard human validation command is:
  - `RUSTY_RUN_MLX_RUNTIME=1 make -C rusty verify >rusty.log 2>rusty.err`

## Current Generation Source Of Truth

- Normal Rusty generation uses the native session API with resident model
  groups, resident projection arrays, corrected Qwen q_norm/k_norm before
  RoPE, and MLX/Metal projection execution.
- The default attention backend is `chunked_compact_mlx`.
- `chunked_compact_mlx` stores K/V in compact kv-head chunks and expands to
  q-head layout at attention time. For this Qwen GQA model, that reduces K/V
  chunk memory by `4x` versus expanded q-head chunk storage.
- `chunked_expanded_kv` remains available explicitly with
  `RUSTY_ATTENTION_BACKEND=chunked_expanded_kv` for diagnostics/benchmarking.
- The active cache is now named behind a cache abstraction:
  - `CachePolicy::Full { step }` is the implemented policy used by the
    current full-context chunked cache; `step` is the chunk growth size,
    currently `256` by default.
  - `Rotating { max_size, keep, step }`, `Quantized { bits, group_size,
    start_at, base }`, and `Recompute { window }` are defined for future work
    only.
  - `Recompute` is diagnostic-only and must not become the default generation
    policy.
- Generation metadata includes `cache_stats` with active backend, KV length,
  capacity, active/reserved bytes, chunk count/size, model geometry, and
  placeholder transient scratch arena accounting.
- `cache_stats` separates compact MLX K/V bytes, expanded MLX K/V bytes, and
  the optional CPU compact K/V mirror. The CPU mirror is disabled by default
  for MLX K/V backends and can be restored for diagnostics/fallback with
  `RUSTY_KEEP_CPU_KV_MIRROR=1`.
- Default chunk size is `256`; override with `RUSTY_KV_CHUNK_SIZE`.
- `RUSTY_ATTENTION_BACKEND=expanded_kv` is retained for explicit diagnostics
  only; do not use it as a normal benchmark path.
- `RUSTY_ATTENTION_BACKEND=cpu` / `compact_cpu` is retained as a correctness
  fallback and diagnostic only; do not run it in normal development loops.
- The native generation helper must not run old second-generation proof paths
  unless explicitly requested with `RUSTY_VERIFY_SECOND_GENERATION=1`.
- Important correction, 2026-05-10: after disabling the CPU compact K/V mirror,
  the active decode path still derived RoPE position from the CPU mirror
  (`layer.keys.size()`). With the mirror disabled this stayed at zero and
  corrupted multi-token prompt/decode attention. The active path now derives
  RoPE position through the cache abstraction (`ChunkedExpandedKvCache::len()`).
  The exact Southwick prompt parity check now matches Python/MLX top token and
  top-10:
  - prompt: `Who are Southwick and Tommy?`
  - prompt ids: `[151644,872,198,15191,220,546,220,25018,21188,220,437,220,24732,2408,30,151645,198,151644,77091,198]`
  - Rusty top token: `2121`
  - Python/MLX top token: `2121`
  - top-10 overlap: `10/10`

## KV/Cache Benchmark Validity After RoPE-Position Fix

- Any KV/cache quality, parity, or attention-behavior conclusion recorded
  after the CPU K/V mirror was disabled and before the RoPE-position fix is
  suspect.
- Memory allocation numbers from those runs may still be useful as rough
  allocation evidence, but output quality, long-context collapse, speed, and
  attention timing should not be used as final conclusions unless rerun after
  the fix.
- Do revalidate:
  - the corrected current default `chunked_compact_mlx` memory/speed/quality
    baseline,
  - any future KV strategy that changes cache length accounting,
  - any test that disables the CPU K/V mirror or changes cache append/fetch
    behavior.
- Do not revalidate:
  - tokenizer special-token/newline handling,
  - Qwen q_norm/k_norm-before-RoPE ordering,
  - adapter metadata/loading wiring, except for a short adapter-active parity
    check when adapter quality is being evaluated,
  - thread-env tests,
  - full resident block and narrow resident gates unless their execution
    structure becomes relevant again.
- Restart memory work from the corrected active path only:
  1. establish one fresh corrected `chunked_expanded_kv` baseline,
  2. keep Southwick top-token/top-10 parity as a guardrail,
  3. investigate compact/chunked MLX K/V storage without changing full-context
     semantics,
  4. avoid rotating/eviction policies until full-context compact storage has
     been evaluated.

## Recorded Runtime Baselines

These are records. Do not rerun these old paths in normal tests just to compare
timings.

Warning: several recorded KV/cache runtime baselines below may predate the
RoPE-position fix described above. Treat them as historical records, not as
current active-path proof. Re-establish the corrected current default baseline
before making new memory/speed decisions.

- Problem scenario:
  - prompt: `what are the important emotions?`
  - controls: `max_tokens=2000`, `temperature=0.7`, `top_k=40`, `seed=1234`
  - prompt tokens: `22`
  - generated tokens before EOS: `683`
- Old full `expanded_kv` preallocation:
  - full expanded K/V estimate: `2,386,427,904` bytes
  - total estimated runtime memory: `6,693,475,328` bytes
  - observed issue: rusty-bridge can rise past 7 GB and pressure swap
  - status: disproved as default; keep only for explicit diagnostics
- `compact_cpu` attention diagnostic:
  - compact K/V estimate: `596,606,976` bytes
  - generation timing: `614,180 ms`
  - throughput: `1.11 tok/s`
  - status: too slow; correctness/fallback microscope only
- Former `chunked_expanded_kv` default:
  - active K/V bytes for 683-token run: `905,969,664`
  - chunks: `108` total, max `3` per layer, `256` tokens per chunk
  - total estimated runtime memory: `5,213,017,088` bytes
  - memory avoided vs full expanded allocation: `1,480,458,240` bytes
  - generation timing: `114,439 ms`
  - throughput: `5.97 tok/s`
  - cleanup: session/model/tokenizer freed, active sessions after free `0`
- Thread env diagnostic, same prompt/settings:
  - `MLX_NUM_THREADS=8`, `OMP_NUM_THREADS=8`, `VECLIB_MAXIMUM_THREADS=8`,
    `RAYON_NUM_THREADS=8`: `110,058 ms`, `6.21 tok/s`
  - `MLX_NUM_THREADS=12`, `OMP_NUM_THREADS=12`,
    `VECLIB_MAXIMUM_THREADS=12`, `RAYON_NUM_THREADS=12`: `115,436 ms`,
    `5.92 tok/s`
  - observed process thread count stayed mostly `5-7` in both cases
  - `threads_12` increased RSS and slowed generation
  - status: do not rerun thread-count tests unless the MLX execution
    structure changes
- Full resident block diagnostic, same prompt/settings:
  - gate: `RUSTY_EXPERIMENTAL_MLX_RESIDENT_BLOCK=1`
  - old default path: `121,122 ms`, `5.64 tok/s`
  - full resident block: `222,298 ms`, `3.07 tok/s`
  - token IDs/text matched through EOS, `683` generated tokens
  - fallback count: `0`
  - memory estimate unchanged at roughly `5.21 GB`
  - observed max memory was roughly `5.7 GB`
  - status: correct but slower; do not promote, keep gate for diagnostics only
  - do not rerun the long full-resident-block comparison unless layer
    execution structure changes
- Narrow resident O-residual diagnostic, same prompt/settings:
  - gate: `RUSTY_RESIDENT_O_RESIDUAL=1`
  - default path: `129,883 ms`, `5.26 tok/s`
  - O-residual resident path: `250,638 ms`, `2.73 tok/s`
  - generated token IDs/text matched through EOS, `683` generated tokens
  - fallback count: `0`
  - sync count unchanged at `99,574`
  - bucket regression was mainly `o_proj` (`+98,156 ms`) and `MLP_gate_up`
    (`+23,359 ms`)
  - status: correct but slower on long generation; do not promote and do not
    rerun as a speed candidate unless layer execution structure changes
- Narrow resident MLP-only diagnostic, same prompt/settings:
  - gate: `RUSTY_RESIDENT_MLP_ONLY=1`
  - comparison run default path: `155,502 ms`, `4.39 tok/s`
  - MLP-only gate path: `124,093 ms`, `5.50 tok/s`
  - generated token IDs/text matched through EOS, `683` generated tokens
  - fallback count: `0`
  - sync count unchanged at `99,574`
  - improvement appeared across major buckets, including `MLP_gate_up`
    (`-7,118 ms`), `attention_eval_sync` (`-5,322 ms`), `qkv` (`-4,111 ms`),
    and `MLP_down` (`-3,907 ms`)
  - follow-up inspection showed `RUSTY_RESIDENT_MLP_ONLY=1` was not a distinct
    execution path in the MLX-native decode path; the resident MLP chain was
    already the default
  - status: the env flag is deprecated/no-op; the actual default path is the
    resident MLP chain, reported as `mlx_resident_mlp_chain_default_path`

## Current Arithmetic State

- Qwen math parity was fixed by applying self-attention `q_norm` and `k_norm`
  before RoPE.
- Chat tokenizer/prompt formatting preserves Qwen special tokens and newlines.
- Active promoted decode flags:
  - corrected q_norm/k_norm before RoPE
  - resident MLX projection arrays
  - MLX-native decode state across layers where available
  - MLX RMSNorm/residual/MLP chain
  - tied-embedding logits/top-k selection
  - chunked expanded K/V attention
- Scalar/CPU paths remain available as fallback and diagnostics only.

## Verifier Profile Rule

- Smoke/default verifier must not repeatedly execute proven expensive baseline
  paths just to compare timings.
- Record known baseline timing/result metadata.
- Re-run old-vs-new expensive comparisons only when:
  - an explicit full/comparison profile is selected, or
  - the math contract changes.
- Once a path is disproved for speed or memory, do not put it in root
  `test.sh` or default verifier runs. Refer to the recorded baseline above.

## Boundary Rules

- CoffeeScript/JavaScript owns orchestration only.
- Rust/C++ owns tensors, model residency, KV cache, MLX/Metal memory, and native
  lifetimes.
- CoffeeScript may only see opaque handles and structured scalar metadata.
- No tensor payloads may escape the C++ shim.
- `node-mlx` is a reference map, not a runtime ownership model.

## Current Limitations

- Chunked compact K/V still uses concat of active compact chunks during
  attention and expands to q-head layout at attention time; future work can add
  chunk-aware attention or spill/swap old chunks.
- `chunked_compact_mlx` is the promoted default backend. It stores K/V chunks
  as `[kv_heads, chunk_size, head_dim]`. `chunked_expanded_kv` remains
  available only as an explicit diagnostic/benchmark backend.
- Compact K/V long validation on the corrected active path passed:
  - prompt/settings: `what are the important emotions?`, `max_tokens=2000`,
    `temp=0.7`, `top_k=40`, `seed=1234`
  - generated tokens: `757`, stop reason: `eos`
  - timing: `138,771 ms`, `5.46 tok/s`
  - recorded expanded baseline: `142,435 ms`, `5.31 tok/s`
  - compact reserved K/V bytes: `301,989,888`
  - recorded expanded reserved K/V bytes: `1,207,959,552`
  - byte ratio: `0.25` (expected 4x GQA reduction)
  - CPU K/V mirror: disabled, fallbacks: none, cleanup: clean
- Swapped K/V is still a future backend, not implemented yet.
- Rotating K/V and quantized K/V are future policies only. The current default
  remains full-context retention, now through the compact MLX chunk cache.
- Transient layer/token scratch accounting is exposed as placeholder fields
  (`arena_active_bytes`, `arena_cached_bytes`, `arena_peak_bytes`) but is not
  implemented yet.
- Runtime thread count is currently about 6 on this machine, while Python MLX
  has been observed around 21 threads; this is diagnostic context, not the
  next optimization target until memory policy is stable.
- Explicit MLX/OMP/VECLIB/RAYON thread env settings at 8 and 12 did not
  materially increase Rusty process thread count; thread knobs are not a
  useful lever for the current execution structure.

## Recommended Next Step

- Keep normal tests on the active `chunked_compact_mlx` path only.
- If memory remains too high, improve chunk policy or add swapped MLX K/V; do
  not fall back to full preallocated or expanded q-head K/V.
- If speed is the next target, compare against Python MLX behavior using one
  focused current-path test, not old CPU or full-expanded baselines.
