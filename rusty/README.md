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
  - stubbed commands only
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
- `verify_bridge.mjs`
  - minimal local verifier for the stub bridge
- `Makefile`
  - `make verify` runs the local verifier

## What is intentionally not implemented yet

- actual MLX inference
- actual tokenizer/model loading
- actual Metal / MLX ownership
- Unix socket transport
- production integration into the main pipeline
- real ML backend calls beyond lightweight environment probing

## Current bridge status

- the current bridge is stubbed
- JavaScript owns only opaque handles
- Rust currently mocks ML-side objects and lifetime tables
- Rust can probe local MLX/backend candidate paths without loading models
- Rust can now build and call a tiny native C++ shim without invoking MLX
- Rust can now attempt a minimal MLX include/link probe without loading models
- Rust can now exercise a tiny native MLX object handle lifecycle entirely inside the shim
- Rust can now diagnose MLX runtime failure stage and exception text without exposing tensors
- live integration into production meta is intentionally deferred
- handle lifecycle validation is active in the stub:
  - unknown handles fail
  - freed handles fail
  - handles remain opaque strings
- shutdown-state validation is active in the stub:
  - shutdown is accepted explicitly
  - health may still report `shutting_down`
  - non-health commands are rejected after shutdown starts

## Verification

If `cargo` exists locally:

- `make -C rusty verify`
- `make -C rusty test-meta`

What it does:

- builds the Rust bridge
- runs `bridge_health`
- runs `backend_probe` and prints the returned structure
- runs `shim_probe` and prints the shim version
- runs `mlx_link_probe` and prints whether MLX linkage is available
- creates, sums, and frees a tiny native MLX test object by opaque handle
- runs `load_model` with a fake path
- runs `create_session` using the returned opaque model handle
- runs `generate` with prompt `hello`
- verifies `ok: true` responses
- runs `bridge_shutdown`

If `cargo` is not installed, the verifier exits cleanly and reports that it
skipped the build/run.

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

- locate the MLX C/C++ callable boundary
- decide whether Rust links directly to a C API or wraps a C++ shim
- implement tokenizer loading
- implement real `load_model`
- implement real `generate` without exposing tensors to JS
