Area: `rusty/`

Purpose:
- scaffold a bridge between the writeStory pipeline world and a future resident native ML process

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
  - stub commands only
  - async dispatch skeleton
  - handle tables for models, tokenizers, sessions, KV caches, and jobs
- `rusty/meta/mlx_bridge.coffee`
  - resident-process wrapper skeleton
  - lazy-start on first use
  - request/response correlation by id
  - no new Memo API
- `rusty/examples/rusty_generate.yaml`
  - illustrative artifact/command shape only

Current status:
- scaffold only
- not integrated into live `meta/`
- not wired into production recipes
- no actual MLX inference yet

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
