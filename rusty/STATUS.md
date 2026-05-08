# Rusty Status

Date: 2026-05-08

Branch: `rusty`

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

## Current Smoke Source Of Truth

- `session_layer_residency_probe` is the promoted source-of-truth generation
  path for smoke.
- Smoke keeps all 36 Qwen3 model4 layer groups resident in the native session.
- Smoke performs one prompt/prefill pass and one resident incremental decode
  pass.
- Smoke reuses the resident result for downstream verifier sections instead of
  rerunning full-stack generation.
- The latest passing smoke verifier produced:
  - generated token id: `24`
  - decoded token: `9`
  - final norm checksum: `130.289`
  - clean native handle counts before and after probes

## Current Arithmetic State

- The quantized layout is still CPU/provisional, not production inference.
- Active promoted decode flags:
  - optimized row-block quantized linear
  - cached quantized layout metadata
  - paired MLP gate/up projection
  - tied-embedding top-1 logits projection
- Scalar quantized linear remains available as a fallback.
- The largest remaining smoke timing bucket is:
  - `gate_up_paired_projection`, roughly `6900 ms`
- Correct but not promoted:
  - `down_full_block_optimized_path`
  - `gate_up_full_block_optimized_path`
- Both full-block variants matched token/text/checksum, but did not improve
  timing.

## Verifier Profile Rule

- Smoke/default verifier must not repeatedly execute proven expensive baseline
  paths just to compare timings.
- Record known baseline timing/result metadata.
- Re-run old-vs-new expensive comparisons only when:
  - an explicit full/comparison profile is selected, or
  - the math contract changes.

## Boundary Rules

- CoffeeScript/JavaScript owns orchestration only.
- Rust/C++ owns tensors, model residency, KV cache, MLX/Metal memory, and native
  lifetimes.
- CoffeeScript may only see opaque handles and structured scalar metadata.
- No tensor payloads may escape the C++ shim.
- `node-mlx` is a reference map, not a runtime ownership model.

## Current Limitations

- No production generation wiring.
- No KV-cache optimized attention kernel beyond verifier scaffolding.
- No Metal-native quantized matmul path yet.
- The full model math remains verifier-only and CPU/provisional.
- Smoke is intentionally narrow and dependency-aware.

## Recommended Next Step

- Continue optimizing quantized linear arithmetic, but avoid full-block tail
  removal as a speed lever; recent probes showed it is correct but slower.
- The next useful target is a real gate/up paired projection optimization, such
  as memory traversal, scale/bias handling, or native/parallel execution.
- Keep smoke to one active generation source plus one focused comparison probe.
