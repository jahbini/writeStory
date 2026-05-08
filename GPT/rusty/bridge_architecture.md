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
- verifier-only resident bridge scaffold with real model4 metadata, selected
  tensor-group residency, structural KV cache, and CPU/provisional full-stack
  Qwen3 arithmetic
- not integrated into live `meta/`
- not wired into production recipes
- not production generation
- no optimized KV-cache attention kernel yet
- no Metal-native quantized matmul yet
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
