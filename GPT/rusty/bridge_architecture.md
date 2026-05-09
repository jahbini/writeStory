Area: `rusty/`

Purpose:
- bridge the writeStory pipeline world to a resident native ML process while
  keeping JavaScript/CoffeeScript out of tensor, model, KV-cache, and GPU
  ownership

Ownership contract:
- JavaScript/CoffeeScript owns:
  - orchestration
  - DAG scheduling
  - Memo visibility
  - YAML
  - SQLite
  - UI
  - opaque handles only
- Rust owns:
  - tensors
  - model lifetime
  - tokenizer lifetime
  - KV cache lifetime
  - GPU / Metal memory
  - resident ML process lifetime

Non-negotiable boundary:
- JS must never directly own tensors, model weights, GPU buffers, or KV cache memory
- JS may keep opaque strings like:
  - `model:1`
  - `sess:1`
  - `kv:1`
  - `job:1`

Current scaffold:
- `rusty/bridge/`
  - Rust crate with JSONL stdin/stdout protocol
  - async dispatch skeleton
  - handle tables for descriptors, models, tokenizers, sessions, tensor
    groups, layer groups, KV caches, arrays, and jobs
  - C++ shim boundary for native MLX/Metal-owned objects
- `rusty/meta/mlx_bridge.coffee`
  - resident-process wrapper skeleton
  - lazy-start on first use
  - request/response correlation by id
  - no new Memo API
- `rusty/examples/rusty_generate.yaml`
  - illustrative artifact/command shape only

Current probe and test shape:
- `rusty/verify_bridge.coffee` is the active verifier; `verify_bridge.mjs` is
  not the verifier entrypoint
- verifier profiles exist:
  - `smoke`
  - `layer`
  - `full`
  - `generate`
- default profile is `smoke`
- smoke source of truth is `session_layer_residency_probe`
- latest passing smoke result:
  - generated token id `24`
  - decoded token `9`
  - final norm checksum `130.289`
  - clean native handle counts before/after
- `rusty/bridge/` includes read-only inspection commands for model/tokenizer
  layout, persistent model descriptors, embedding/tensor/layer-group handles,
  structural KV-cache storage, resident layer groups, and verifier-only
  CPU/provisional full-stack arithmetic
- smoke/default verifier runs must not re-run proven expensive baseline paths
  just to compare timings. Keep the promoted source-of-truth path in smoke,
  log recorded baseline timings/results as metadata, and reserve old-vs-new
  execution for explicit full/comparison profiles or math-contract changes.
- runtime tests follow the same rule: once a path is disproved, record it and
  stop running it in normal development. For current Rusty generation:
  - `chunked_expanded_kv` is the active default attention backend
  - full preallocated `expanded_kv` is diagnostic/benchmark-only
  - `compact_cpu` attention is fallback/diagnostic-only
  - `RUSTY_EXPERIMENTAL_MLX_RESIDENT_BLOCK=1` is diagnostic-only: it is
    correct but slower on the long default prompt (`222,298 ms`, `3.07 tok/s`)
    than the default path (`121,122 ms`, `5.64 tok/s`), with matching token
    IDs/text through EOS and no fallbacks
  - `RUSTY_RESIDENT_O_RESIDUAL=1` is also diagnostic-only: it looked promising
    on a 16-token run, but the long default prompt regressed from `129,883 ms`
    (`5.26 tok/s`) to `250,638 ms` (`2.73 tok/s`) with matching output and no
    fallbacks
  - `RUSTY_RESIDENT_MLP_ONLY=1` is the current best narrow resident gate on
    the long default prompt: it matched output through EOS and improved one
    comparison run from `155,502 ms` (`4.39 tok/s`) to `124,093 ms`
    (`5.50 tok/s`) with no fallbacks. Follow-up inspection showed this was not
    a distinct execution path in the MLX-native decode path; the resident MLP
    chain is already the default. Treat `RUSTY_RESIDENT_MLP_ONLY` as deprecated
    reporting/no-op, and use `mlx_resident_mlp_chain_default_path` for the
    actual state.
  - recorded long-generation baselines are in `rusty/STATUS.md`
- `rusty/test_tensor_group.sh`
  - Q/A-style CLI compliance probe for a single tensor group
  - prints each question before sending the bridge request
  - prints each bridge answer before asserting the result
  - verifies the handle is freed and then rejected cleanly
- the tensor-group probe is intentionally narrow:
  - one quantized group only
  - metadata only at the Rust boundary
  - payload ownership remains inside the C++ shim
  - no model build or inference path is implied by the test

Current status:
- Rusty native generation is wired through the prompt recipe path and JS/Coffee
  wrappers via opaque handles
- Qwen math parity is corrected with q_norm/k_norm before RoPE
- tokenizer/chat formatting handles Qwen special tokens and newlines
- default attention backend is chunked expanded K/V on MLX/Metal
- CPU and full-expanded K/V paths remain available only as diagnostics
- the CLI probe style should be explicit Q/A, not silent answers-only output
- active promoted smoke decode flags:
  - optimized row-block quantized linear
  - cached quantized layout metadata
  - paired MLP gate/up projection
  - tied-embedding top-1 logits projection
- scalar quantized linear remains fallback
- correct but not promoted:
  - `down_full_block_optimized_path`
  - `gate_up_full_block_optimized_path`
- largest current smoke timing bucket:
  - `gate_up_paired_projection`, roughly `6900 ms`

`node-mlx` usage rule:
- `node-mlx` is a reference map, not a runtime model

Borrow:
- command names
- parameter shapes
- module boundaries
- native operation grouping
- build/link clues

Do not borrow:
- JS-visible tensor ownership
- JS object wrappers around MLX arrays
- host-runtime lifetime management
- GC-adjacent native allocation pressure

Long-term target:
- resident Rust process backed by MLX bindings / MLX C API
- Node/CoffeeScript remains the orchestrator
