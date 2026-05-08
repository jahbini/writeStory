# rusty

`rusty/` is the scaffold for a resident Rust ML bridge that sits between the
existing writeStory pipeline world and the future native ML world.

## Why this bridge exists

The current pipeline architecture is good at orchestration:

- CoffeeScript/Node owns DAG scheduling
- Memo owns visibility and notifier behavior
- YAML owns step wiring
- SQLite owns persistent story/KAG state
- the UI owns operator control

But JavaScript is the wrong place to own long-lived ML allocations:

- model weights
- tensors
- KV cache memory
- Metal / unified-memory lifetime
- long-lived session state

The bridge exists to move ML lifetime and memory into one resident Rust process
while preserving the current writeStory orchestration model.

This is specifically meant to avoid JS GC interference with unified memory and
long-lived ML allocations while preserving the existing DAG/Memo architecture.

## Ownership boundary

JavaScript:

- orchestration only
- DAG scheduling only
- Memo visibility only
- YAML / SQLite / UI only
- opaque handles only

JavaScript must never directly own:

- tensors
- model weights
- GPU buffers
- KV cache memory
- Metal lifetime

Rust:

- owns tensors
- owns GPU memory
- owns KV cache
- owns model lifetime
- owns resident ML process lifetime

Rusty must not mix MLX headers and libraries from different installs.

## Design rule

`node-mlx` is a reference map, not a runtime model.

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

## What is scaffolded here

- `bridge/`
  - minimal Rust resident process
  - JSONL stdin/stdout protocol
  - handle tables
  - lightweight backend discovery probe
  - tiny C++ shim skeleton with a C ABI
  - optional MLX compile/link probe through that shim
  - clean startup/shutdown
  - async dispatch skeleton
  - verifier-only native model, tensor-group, KV-cache, and arithmetic probes
- `meta/mlx_bridge.coffee`
  - CoffeeScript wrapper skeleton for a resident bridge process
  - designed to fit existing Memo semantics
  - no new Memo API
- `examples/rusty_generate.yaml`
  - illustrative future recipe shape only
- `examples/step_generate_via_memo.coffee`
  - example of an ordinary pipeline step writing an `mlx/...` request key and
    awaiting the normal Memo result path
- `examples/rusty_pipeline.yaml`
  - isolated example recipe showing a single pipeline step materializing
    `out/rusty_story.txt`
- `examples/session_transcript.jsonl`
  - example request/response flow for a single stub session
- `verify_bridge.coffee`
  - CoffeeScript verifier with `smoke`, `layer`, `full`, and `generate`
    profiles
- `Makefile`
  - `make verify` runs the local verifier

## What is intentionally not implemented yet

- production generation wiring
- optimized KV-cache attention
- Metal-native quantized matmul
- Unix socket transport
- production integration into the main pipeline
- JavaScript-visible tensor/model ownership

## Current bridge status

- `verify_bridge.coffee` is the active verifier. `verify_bridge.mjs` is not the
  verifier entrypoint.
- The default verifier profile is `smoke`.
- JavaScript/CoffeeScript owns orchestration and sees only opaque handles plus
  scalar metadata.
- Rust/C++ owns tensors, model residency, KV cache structures, MLX/Metal
  lifetimes, and native handle cleanup.
- The native bridge can inspect Qwen3 model4 config and safetensors
  descriptors, create model descriptors, load selected tensor groups, and
  perform verifier-only CPU/provisional quantized arithmetic.
- Smoke keeps all 36 Qwen3 model4 layer groups resident in the native session.
- `session_layer_residency_probe` is the promoted smoke source-of-truth
  generation path.
- The latest passing smoke source-of-truth result:
  - generated token id: `24`
  - decoded token: `9`
  - final norm checksum: `130.289`
  - clean native handle counts before and after probes
- Active promoted decode flags:
  - optimized row-block quantized linear
  - cached quantized layout metadata
  - paired MLP gate/up projection
  - tied-embedding top-1 logits projection
- Scalar quantized linear remains available as fallback.
- Correct but not promoted:
  - `down_full_block_optimized_path`
  - `gate_up_full_block_optimized_path`
- Both full-block variants matched token/text/checksum but were slower.

## Verification

Standard human validation command:

- `RUSTY_RUN_MLX_RUNTIME=1 make -C rusty verify >rusty.log 2>rusty.err`

The default profile is `smoke`. Smoke is dependency-aware and must not rerun
proven expensive full-stack or generation baselines just to compare timings.
Recorded baseline result/timing metadata should be reused unless an explicit
full/comparison profile is selected or a math contract changes.

Smoke runs:

- backend and MLX discovery
- model4 load plan and descriptor checks
- tokenizer fixture load/encode/decode
- one quantized slice/layout path
- one RMSNorm primitive
- one minimal MLX runtime array test when `RUSTY_RUN_MLX_RUNTIME=1`
- one promoted resident full-stack/incremental generation source via
  `session_layer_residency_probe`
- one focused comparison probe for the current development target
- native handle-count checks before and after probes

MLX runtime tests must be run from a normal user terminal with Metal/IOKit
access, not from Codex.

Other profiles may enable older expensive probes, including layer-stack,
multi-token greedy, prompt-session, and incremental comparison sections.

## Completed verifier milestones

- descriptor creation from config and safetensors metadata
- embedding and tensor-group loading with clean freed-handle behavior
- quantization layout and tiny dequantization probes
- q_proj quantized linear slice/row/fullrow/vector probes
- RMSNorm primitive
- layer-0 attention, MLP, and consolidated block probes
- N-layer and full-stack single-token probes
- greedy next-token and prompt-session verifier probes
- structural native KV-cache storage
- incremental-attention verifier scaffolding
- resident all-layer group loading for native sessions
- optimized row-block quantized linear
- cached group layout metadata
- paired MLP gate/up projection
- tied-embedding top-1 logits projection

## Current optimization frontier

- The full model math remains verifier-only and CPU/provisional.
- The largest current smoke timing bucket is `gate_up_paired_projection`,
  roughly `6900 ms`.
- Recent full-block variants for `down_proj` and gate/up were correct but
  slower, so they remain disabled.
- The next useful speed lever is real gate/up paired projection work:
  traversal, scale/bias access, native parallelism, or a Metal/native
  quantized matmul path.

`test-meta`:

- imports the existing Memo implementation from `pipeline_runner.coffee`
- attaches `rusty/meta/mlx_bridge.coffee` only inside the test
- submits an `mlx/...` Memo key
- waits for a normal Memo result/notifier path
- verifies the stubbed bridge response
- shuts the bridge down cleanly

## Pipeline-facing example

The files:

- `rusty/examples/step_generate_via_memo.coffee`
- `rusty/examples/rusty_pipeline.yaml`

show the first ordinary-step shape for bridge usage:

- the step writes an ML request to a Memo key such as `mlx/generate/example.json`
- the step waits on the matching normal Memo entry/notifier
- the step turns the bridge reply into a normal pipeline output artifact

This remains intentionally isolated:

- it is not wired into production `meta/index.coffee`
- it does not modify live recipes
- it is a pattern example for future integration work

## Long-term target

The intended long-term native side is MLX bindings / MLX C API accessed from
Rust, with Node/CoffeeScript staying as the orchestrator.

## TODO: next real milestone

- keep smoke dependency-aware and avoid rerunning proven expensive baselines
- optimize the current largest arithmetic bucket:
  `gate_up_paired_projection`
- preserve token `24`, decoded text `9`, checksum `130.289`, and clean handle
  counts while optimizing
- keep scalar/current optimized quantized linear paths available as fallback
- move production generation wiring only after verifier-only arithmetic and
  residency contracts are stable
