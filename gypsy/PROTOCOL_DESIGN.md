# Gypsy Protocol Design

This document defines the first target protocol for the new CoffeeScript to
native MLX/Metal generation path.

The goal is not just to make generation work. Rusty already proved the model
chain can be implemented correctly. The goal here is to shape the boundary so
Metal stays hot: native code owns the graph, tensors, memory residency,
evaluation cadence, and synchronization policy.

## Design Principle

CoffeeScript controls intent.

Native code controls execution.

That means CoffeeScript may say:

```text
load this model
attach this adapter
generate N tokens from this prompt with these controls
return text and scalar metadata
```

CoffeeScript must not say:

```text
run q_proj
apply RoPE
append KV
run attention
return full logits
now run the next token
```

The second shape recreates the manual coordination tax that made Rusty slow.

## Direct Metal Migration Strategy

The low-level Metal path should be migrated from the end of the decode chain
backward, not from the first layer forward.

The immediate reason is diagnostic isolation. The existing CPU/MLX pipeline is
the correctness oracle. If only the final logits/top-token step is moved to
Metal, then any changed token is caused by that final step. After that is
correct, move final RMSNorm. After that is correct, move the previous terminal
projection/residual boundary. Repeat backward through the layer.

Preferred migration order:

1. Final tied-embedding logits and top-token selection.
2. Final RMSNorm.
3. Last layer MLP down projection and residual.
4. Last layer MLP activation plus gate/up projections.
5. Last layer attention output projection and residual.
6. Last layer attention value mix / score computation / KV access.
7. Last layer q/k/v projection, q_norm/k_norm, and RoPE.
8. Earlier layers once the terminal layer chain is Metal-owned.
9. Prompt prefill after decode correctness is stable.

This is deliberately different from a broad "rewrite the forward pass" plan.
Broad rewrites hide multiple boundary bugs at once and make token-zero or
all-zero failures hard to localize.

## Metal ABI And Datatype Discipline

Metal kernels must receive exactly the data representation they are written to
consume.

Current representation contracts:

- Quantized INT4 model weights are packed in raw `uint32_t` words. Metal
  kernels should consume them as `device uint*` and unpack nibbles internally.
- Quantized scales and biases are BF16 bit patterns stored as raw 16-bit words.
  Metal kernels should consume them as `device ushort*` and convert BF16 bits
  to `float` inside the kernel.
- RMSNorm weights in the Qwen path are also BF16 raw 16-bit words and must not
  be passed as `float*`.
- Hidden states, normalized vectors, activation vectors, attention outputs, and
  logits are currently `float` buffers unless a specific kernel changes that
  contract.
- Dimension constants should be passed or compiled as Metal-friendly unsigned
  integer values. Avoid mixed signed/unsigned host assumptions in kernel
  parameters.

Known trap:

Passing BF16 payloads as `float*` may compile and run but produces wrong
values. Treat datatype mismatches as first-class correctness bugs, not
performance anomalies.

## Failure Record: Manual Direct-Metal Layer Stack

The direct-Metal layer-stack experiments established useful facts but failed as
an execution strategy.

### What Was Tried

The migration followed the "last first" strategy:

1. final logits/top-token selection
2. final RMSNorm
3. terminal MLP down projection and residual
4. terminal gate/up projections and activation
5. terminal attention-to-o path
6. terminal q/k/v plus q_norm/k_norm/RoPE
7. generalized terminal layer ranges

These steps proved that the local kernels can match the known-good generation
sequence when shapes and raw datatypes are correct.

### What The Timings Proved

The correct staged direct-Metal path still had CPU/GPU time ratios around 10:1.
The host was still doing the important work:

- coordinating many small kernel dispatches
- copying or checking CPU-visible buffers
- managing per-layer state
- waiting at command-buffer boundaries
- converting Metal results back into host/MLX arrays for the next segment

This is not a "Metal is slow" result. It is a boundary-shape failure.

### Fused Tail Result

The fused MLP tail combined:

- post-attention residual
- post-attention RMSNorm
- gate/up projections
- SiLU multiply
- down projection
- final residual

It was correct, but it did not fix the core issue. The layer loop was still
host-orchestrated and the path still returned to CPU-visible state too often.

### Full-Layer Resident Result

A full-layer helper was added to keep a layer's intermediate values in Metal
buffers. It initially produced bad tokens because shared projection buffers
were reused with stale layer-specific weights. The test harness was updated to
assert the exact 64-token known-good sequence so this kind of failure cannot
pass as "generated text".

After the stale-buffer fix, correctness returned, but the path was still not
the desired architecture because it remained one full-layer host call per
layer/token.

### Terminal-Stack Command-Buffer Result

A terminal-stack command-buffer path attempted to run layers 1-35 through one
native call. This was the wrong implementation of the right idea. It copied
many layer weights into CPU-visible Metal buffers before execution, causing
high CPU time, higher memory pressure, and low GPU utilization.

This path should not be rerun as a benchmark. It is diagnostic history only.

### Root Cause

The failed implementations confused "larger helper call" with "Metal-owned
program." They still made the CPU responsible for resource residency,
scheduling, synchronization, and validation.

The requirement is stricter:

- CPU may request generation and receive scalar results.
- CPU must not drive layer math.
- CPU must not copy weights per token.
- CPU must not read hidden vectors between layers.
- CPU must not validate every segment in the hot loop.
- Metal resources must be resident and reused.
- Command encoding must represent a token/chunk execution plan, not ad-hoc
  layer fragments.

### Do-Not-Rerun List

Do not rerun these paths except as explicitly labeled diagnostics after their
architecture changes:

- staged direct-Metal terminal layer ranges as speed benchmarks
- fused MLP tail as a speed benchmark
- full-layer resident helper as a speed benchmark
- terminal-stack helper that copies weights into shared buffers per token

Correctness checks on a single local kernel are still acceptable. Speed claims
must come only from the active architecture.

### Next Architecture Requirement

The next credible implementation must introduce a real execution object:

- persistent Metal buffers for every model tensor needed by the active path
- persistent KV buffers
- persistent scalar/config buffers
- one reusable command encoding plan for a decode token or decode chunk
- no hidden-state host readback between layers
- no weight copies during generation
- readback only for selected token id, stop flag, and explicitly requested
  debug telemetry
performance details.

Boundary review checklist for every new Metal step:

- source tensor shape matches the kernel's compile-time or runtime constants
- source element type matches the Metal pointer type
- packed-column count and logical input length are both explicit
- BF16 conversion is done exactly once
- output buffer element type matches the next consumer
- no full tensor readback is added except for a gated diagnostic or terminal
  scalar/top-token result

## Public CoffeeScript Protocol

The production-facing protocol should be high-level and handle-based.

Candidate API:

```text
loadModel(model_dir) -> model_handle + scalar metadata
loadTokenizer(tokenizer_dir) -> tokenizer_handle + scalar metadata
loadAdapter(adapter_dir, model_handle?) -> adapter_handle + scalar metadata
createSession(model_handle, tokenizer_handle, options?) -> session_handle
warmSession(session_handle) -> scalar timing/cache metadata
generate(session_handle, prompt_or_tokens, controls) -> text + token ids + scalar metadata
freeSession(session_handle) -> cleanup metadata
unloadAdapter(adapter_handle) -> cleanup metadata
unloadTokenizer(tokenizer_handle) -> cleanup metadata
unloadModel(model_handle) -> cleanup metadata
```

Acceptable first implementation shortcut:

```text
createSession(model_handle, tokenizer_handle, { adapterPath })
```

Long-term preference:

```text
loadAdapter(...) -> adapter_handle
attachAdapter(session_handle, adapter_handle)
```

## Values Returned To CoffeeScript

CoffeeScript may receive:

- opaque handles
- token ids
- decoded text
- stop reason
- timing
- backend names
- memory estimates
- cache stats
- adapter metadata
- fallback flags
- cleanup counts

CoffeeScript must not receive:

- model weight tensors
- intermediate hidden states
- K/V tensors
- projection outputs
- full logits arrays in the production path
- MLX array pointers

Diagnostic profiles may expose extra scalar summaries, but they must be gated.

## Native Internal Execution Units

The native implementation should organize around execution units that are large
enough for MLX/Metal to schedule effectively.

Internal units:

- model config inspection
- tokenizer setup
- adapter loading
- resident model/session creation
- prompt prefill
- decode loop
- per-layer forward block
- KV cache append/fetch
- logits/top-k/sampling
- stop handling
- metadata collection
- cleanup

These are native functions or classes, not CoffeeScript commands.

## Rusty Lessons To Carry Forward

Rusty proved several correctness contracts. These must remain true.

- Full prompt prefill must process every prompt token in order.
- Decode position must follow the full prompt length.
- Qwen self-attention `q_norm` and `k_norm` happen after q/k projection and
  before RoPE.
- LoRA deltas for q/k projections are added before q_norm/k_norm and before
  RoPE.
- V projection LoRA is applied before attention.
- O projection LoRA is applied before residual addition.
- MLP gate/up/down LoRA is applied at the matching projection input.
- Qwen chat tokenizer must preserve special tokens and newlines.
- Cleaned decoded text should hide stop/special tokens while preserving normal
  Unicode.
- KV cache is persistent session state, not transient scratch.
- CPU mirrors and recompute modes are diagnostics, not the default path.

## Rusty Patterns Not To Carry Forward

Rusty was valuable as a teaching engine, but some patterns are explicitly not
the new target.

- Per-layer or per-projection CoffeeScript orchestration.
- Full-logit `Float32Array` returned to JavaScript each token.
- Always-on debug checkpoints in the hot path.
- Repeated host readbacks for checksums or first values.
- Eager evaluation after each small operation when it can be deferred.
- Eager full-model fp32 upcast.
- Manual execution that prevents MLX from seeing larger fused/grouped work.

## Current Copied `mlxCoffee` Baseline

The copied `metal/` code currently exposes an old API:

```text
loadModel(model_dir)
forwardStep(model, token_id, position, out_logits)
resetKV(model)
freeModel(model)
```

This is useful only as a native addon baseline. It is not the production
protocol.

Known problem:

- old `loadModel` loads all safetensors,
- evaluates each tensor,
- upcasts fp16 weights to float32,
- allocates CPU KV storage,
- and can blow up memory on Qwen-sized models.

Therefore the legacy `loadModel` export is disabled by default. It may only be
used for explicit diagnostics with:

```text
GYPSY_ALLOW_LEGACY_LOAD_MODEL=1
```

Normal code must use `loadModelResident`, which is metadata-only until the safe
resident loader milestone is implemented.

## First Implementation Milestones

### Milestone 1: Safe Addon Baseline

Status: established.

- Build addon.
- Load `metal/metal_llm.node`.
- Verify exported functions.
- Do not load model weights.

### Milestone 2: Metadata-Only Model Inspection

Add a native or CoffeeScript-visible command that reads only:

- `config.json`
- model architecture fields
- safetensors index/header metadata
- tokenizer metadata if needed

It must not load tensor payloads.

Expected output:

- model type
- layer count
- hidden size
- q heads
- kv heads
- head dim
- dtype inventory
- safetensor shard count
- rough byte totals

### Milestone 3: New Handle-Based API Skeleton

Add exported functions for the intended protocol even if some are stubs:

```text
inspectModel
loadModelResident
describeModelGroups
loadTokenizer
loadAdapter
createSession
warmSession
generate
freeSession
unloadModel
```

The first test should verify export shape and safe metadata flow only.

Status: established.

Current behavior:

- `loadModelResident(model_dir)` creates an opaque model handle and carries
  inspection metadata.
- The model handle owns a cached native descriptor index parsed from
  safetensors headers.
- `describeModelGroups(model_handle)` uses that cached header-derived index to
  build the Qwen group plan that a future safe loader should consume.
- No tensor payloads are loaded.
- No MLX arrays are constructed.
- The descriptor source is explicitly reported as `safetensors_headers_only`.
- The group planner reports `descriptor_index_owner: model_handle`, proving it
  uses the cached model-handle descriptor index rather than asking
  CoffeeScript to own tensor layout.

For the current Qwen model4 target, `describeModelGroups` verifies:

- 36 layers.
- 904 tensor descriptors.
- 396 expected layer groups.
- 396 complete layer groups.
- 0 missing layer-group tensors.
- Per-layer groups: 7 quantized projections plus 4 norm weights.
- Global tensors include tied embeddings and final norm; `lm_head.weight` is
  absent as expected.

Each tensor descriptor also carries file-backed location metadata:

- safetensors-relative `byte_offsets`
- `payload_data_offset` (`8 + header_bytes`)
- absolute file byte offsets
- source file name
- source path

These fields are the required handoff to a future mmap-backed loader. They let
native code locate tensor payloads without copying them through CoffeeScript and
without reparsing headers in the hot path.

This is the source of truth for the next resident-loader plan. It is still
metadata-only; it must not be confused with loading resident weights.

Adapter handles follow the same rule:

- `loadAdapter(adapter_dir)` creates an opaque adapter handle.
- The adapter handle owns a cached native descriptor index parsed from
  `adapters.safetensors` headers.
- `describeAdapterGroups(adapter_handle)` uses that cached index to build the
  LoRA layer/target plan.
- No adapter payload tensors are loaded.
- The descriptor source is reported as `safetensors_headers_only`.
- The group planner reports `descriptor_index_owner: adapter_handle`.

For the current Qwen LoRA adapter target, `describeAdapterGroups` verifies:

- 224 tensor descriptors.
- Rank 8.
- Scale 20.
- Layers 20 through 35.
- 7 projection targets per covered layer.
- 112 expected LoRA groups.
- 112 complete LoRA groups.
- 0 missing LoRA group tensors.

Adapter tensor descriptors expose the same file-backed location metadata, so
LoRA tensors can later be loaded from safetensors by native code without JSONL
payload transfer or CoffeeScript ownership.

This is the source of truth for future adapter-resident loading. CoffeeScript
still sees scalar metadata only; adapter tensor payload ownership remains native.

Session creation now performs metadata-only compatibility planning:

- `createSession(model_handle, tokenizer_handle, { adapter })` builds a
  `session_plan`.
- The session plan checks model dimensions against adapter LoRA tensor shapes.
- The attached adapter must fit the Qwen projection dimensions before a session
  is accepted.
- For q/k/v/o/gate/up/down, the plan validates:
  `lora_a` shape `[projection_input, rank]` and `lora_b` shape
  `[rank, projection_output]`.
- For the current Qwen adapter, the accepted session plan reports:
  - adapter attached.
  - adapter compatible.
  - layer range `20..35`.
  - 112 checked projection groups.
  - 112 compatible projection groups.
  - 0 adapter compatibility errors.

This compatibility gate is still metadata-only. It does not load adapter
payloads and does not run generation.

`warmSession(session_handle)` now returns a metadata-only warm plan:

- `warm_plan_version: gypsy-metadata-warm-plan/1`.
- Payload loading remains false.
- Resident MLX array construction remains false.
- Execution owner is native.
- Generation-loop owner is native.
- CoffeeScript tensor payloads are explicitly disallowed.
- The plan reports model layer/global byte totals, adapter byte totals, and
  total resident-plan bytes.
- The plan reports file residency plans for model and adapter safetensors:
  source path, descriptor count, file bytes, payload bytes, absolute payload
  span, `mmap_planned: true`, `mapped_now`, and
  `payload_loaded: false`.
- Each file residency plan validates source existence, non-empty byte ranges,
  absolute payload span inside the file, and reports
  `residency_plan_valid: true` before future mapping is allowed.
- The plan names the future native execution units:
  - tokenizer prompt plan
  - resident model projection arrays
  - resident norm weights
  - resident adapter LoRA arrays
  - native prompt prefill
  - native decode loop
  - native KV cache
  - native sampling/stop handling

This milestone defines what warming means before implementing actual resident
MLX array construction.

Warm now owns native read-only file mappings at the session layer:

- model and adapter safetensors files are opened and `mmap`ed read-only during
  first warm
- file mappings are owned by the session, not CoffeeScript
- tensor payload bytes are not copied through CoffeeScript
- MLX arrays are still not constructed
- safetensors payload pages are not intentionally touched during warm
- later warm calls reuse the existing mappings
- `freeSession` releases the mappings before model/adapter handles can be
  unloaded

The warm plan reports:

- `mapped_file_count`
- `mapped_file_bytes`
- model/adapter mapped file counts and bytes
- per-file `mapped_now: true`
- per-file `mapping_owner: native_session`

This is the first safe resident-loading substep. It establishes native file
ownership without yet creating tensor views or MLX arrays.

`describeTensorViews(session_handle)` is the next safe resident-loading
substep:

- requires a warmed session
- resolves every model and adapter tensor descriptor to a byte range inside
  the session-owned mapped files
- reports mapped descriptor counts, missing mappings, out-of-range views, and
  resolved payload bytes
- returns a few scalar sample views for inspection
- does not expose raw pointers
- does not copy tensor payloads to CoffeeScript
- does not intentionally touch payload pages
- does not construct MLX arrays

For the current Qwen model and adapter, the expected resolved view counts are:

- model descriptors: 904 resolved out of 904
- adapter descriptors: 224 resolved out of 224
- missing mappings: 0
- out-of-range views: 0

This establishes the handoff from safetensors-header metadata to native
session-owned byte views. The next resident-loading step can build typed MLX
arrays from these views without reparsing headers or involving CoffeeScript.

`describeTypedTensorPlan(session_handle)` classifies those resolved byte views
into the typed resident-array specs that the loader will construct next:

- quantized packed weights
- quantized scales
- quantized biases
- norm weights
- dense LoRA matrices
- role counts for attention, MLP, embeddings, and norms
- dtype counts
- selected sample array specs

This still does not construct MLX arrays or touch payload pages. It is the
last metadata-only handoff before selected resident MLX array construction.

For the current Qwen target, the expected model typed plan is:

- 904 planned model arrays
- 253 `U32` packed quantized weights
- 253 quantized scale arrays
- 253 quantized bias arrays
- 145 norm-weight arrays
- 651 `BF16` arrays

For the current adapter, the expected typed plan is:

- 224 planned adapter arrays
- 224 `F32` LoRA dense matrices
- 32 matrices per covered target role across the 16 LoRA layers

`constructSelectedResidentArrays(session_handle)` is the first deliberately
small MLX-array construction milestone:

- requires a warmed session
- uses session-owned mapped byte views
- constructs only a small selected set of model/adapter arrays
- keeps array lifetime owned by the session
- copies selected payloads natively into MLX-allocator memory because MLX does
  not support arbitrary external mmap pointers as live array buffers
- returns only scalar metadata about the arrays
- does not expose tensor payloads to CoffeeScript
- is idempotent

The initial selected set is intentionally tiny:

- layer 0 input RMSNorm
- layer 0 q_norm
- layer 0 q_proj weight/scales/biases
- final norm
- layer 20 q_proj LoRA A/B

This is not generation and not full resident loading. Its purpose is to prove
that mapped safetensors views can safely become native MLX arrays without
returning payload data across the JavaScript boundary. The selected payload copy
happens entirely inside native code.

`describeSelectedResidentGroups(session_handle)` assembles the selected arrays
into native execution groups:

- a complete layer 0 `q_proj` quantized projection group
- selected norm weights: input RMSNorm, q_norm, final norm
- a layer 20 q_proj LoRA A/B delta group

This still does not execute math. It proves that the resident-array layer can
form the group-level objects that future projection/norm probes and generation
will consume.

`warmSession` is idempotent:

- The first call reports `already_warmed: false` and
  `reused_existing_plan: false`.
- Later calls report `already_warmed: true` and
  `reused_existing_plan: true`.
- Payload loading remains false in both cases.

`generate` requires a warmed session:

- Calling `generate` before `warmSession` fails with
  `Session must be warmed before generate`.
- This enforces lifecycle order before the real native execution path exists.

Handle unloading respects live session dependencies:

- `unloadModel` fails while any session references that model.
- `unloadTokenizer` fails while any session references that tokenizer.
- `unloadAdapter` fails while any session references that adapter.
- After `freeSession`, the referenced handles can be unloaded normally.

This prevents dangling native session references before resident payloads,
mmaps, or MLX arrays are introduced.

`describeSession(session_handle)` exposes current session state without
changing lifecycle state:

- referenced model/tokenizer/adapter handles
- whether referenced handles are still live
- whether the session is warmed
- session compatibility plan
- warm plan only when already warmed
- native ownership fields
- `payload_loaded: false`

This is intended as the UI/pipeline trust probe before calling generation.

`protocolStatus()` exposes global protocol-handle state without changing
lifecycle state:

- live model/tokenizer/adapter/session handle counts
- scalar summaries for each live handle
- warmed and ready-for-generate session state
- whether session references still point to live handles
- ready session handles
- dangling session-reference count and handles
- legacy loader enablement
- native execution ownership fields
- `payload_loaded_anywhere: false`
- `resident_arrays_constructed_anywhere: false`

This is a metadata-only trust snapshot. It must remain safe to call from the
UI before and after generation. It is also the cleanup assertion: after
freeing sessions and unloading handles, counts and ready-session state should
return to zero.

`generate(session_handle, request)` now returns a first logits-backed generation
response for valid requests:

- `generation_plan_version: gypsy-generation-plan/1`.
- Payload loading remains false.
- The native side owns prompt formatting/tokenization, full prompt prefill,
  decode loop, KV updates, adapter delta application, logits projection,
  sampling/greedy selection, EOS handling, and decoded-text cleanup.
- CoffeeScript does not drive tokens.
- CoffeeScript does not receive tensor payloads or full logits.
- The plan reports prompt source (`text` or `token_ids`), chat mode, max token
  request, sampling mode, adapter attachment, adapter prefill/decode use, and
  descriptor counts.
- The response returns a short greedy sequence selected from native final
  logits. The current milestone caps the native loop at sixty-four tokens so runtime
  tests stay bounded while the protocol wiring is still being built.
- The response explicitly reports `model_math_executed: true` and
  `logits_computed: true` when `max_tokens > 0`.
- For `prompt_tokens`, each decode iteration uses the current token to perform
  a real quantized embedding lookup, runs real layer blocks 0 through 35, then
  applies final RMSNorm and tied logits projection. The selected greedy token
  is fed into the next iteration.
- For text-only prompts, a narrow native tokenizer fallback is wired for the
  current smoke prompt: raw `"hello"` maps to token `14990`, and chat `"hello"`
  maps to the Qwen fallback chat sequence
  `[151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198]`.
  Other text still uses the temporary synthetic prompt-conditioned hidden path
  until full tokenizer JSON/BPE support is wired.
- The response reports `full_layer_stack_executed: true` for the
  `prompt_tokens` path.
- The response now runs every available prompt token through the same full
  36-layer path before decode and stores native per-layer K/V history. Attention
  uses cached QK softmax/value mixing instead of the previous single-token
  value shortcut. This is the first correctness-oriented KV milestone:
  `kv_cache_materialized` is true, with the temporary default backend reported
  as `native_cpu_float_per_layer`.
- A first MLX-resident expanded-K/V attention experiment was correct but slower
  on the 64-token smoke (`~6.4 tok/s`) than the CPU K/V default (`~7.8-9.4
  tok/s`). It is retained only as an explicit experiment via
  `GYPSY_ATTENTION_BACKEND=mlx` or `GYPSY_ATTENTION_BACKEND=mlx_expanded_kv`.
  Do not promote or rerun that path as a baseline unless the layer execution
  structure changes.
- A preallocated expanded-K/V MLX attention experiment was correct and kept
  memory under 3GB, but still did not solve the scheduler/eval problem. The
  64-token smoke reached about `8.8 tok/s`, with final lazy evaluation still
  dominating readback/eval time. It is retained only as an explicit experiment
  via `GYPSY_ATTENTION_BACKEND=mlx_prealloc_kv`.
- A custom MLX `fast::metal_kernel` single-token attention experiment was also
  correct and kept memory near 3GB, but was slower than the default path:
  about `5.6 tok/s` after removing the obvious score recomputation bug. The
  bottleneck again appeared as delayed final graph evaluation, not CPU
  arithmetic. It is retained only as an explicit experiment via
  `GYPSY_ATTENTION_BACKEND=metal_kernel_attention` and must not be run as a
  baseline unless the execution structure changes substantially.
- Prompt-prefill logits are skipped by default. Decode still computes logits for
  generated tokens. Greedy selection uses MLX `argmax` with scalar readback by
  default; the old host-side full-logit checksum/top-10 scan is diagnostic-only
  behind `GYPSY_FULL_LOGIT_SUMMARY=1`.
- Current speed limit: the active default path still performs thousands of
  q/k/v readbacks because attention is CPU-side. Reaching the 20-25 tok/s target
  requires a new attention/KV execution structure, likely below MLX op
  composition. One custom kernel per layer/token is not enough; it still leaves
  native C++ orchestrating a large lazy graph and paying the final synchronization
  cost. The next serious design target is direct Metal buffer ownership or a
  larger fused native execution graph, not another small MLX-op variant.
- `loadTokenizer` caches the tokenizer vocabulary natively. Text prompts now
  use a greedy Qwen vocab fallback rather than a single hard-coded `"hello"`
  path, including Qwen chat wrapper tokens. Generated token IDs are decoded
  through the same native vocab map when possible. This is not yet full BPE
  merge-compatible tokenization, but it is a real native vocab-index milestone.

The function returns `status: selected_logits_generated` for `max_tokens > 0`.
This is model-backed repeated token selection at the final-logits tail, but not
full prompt-conditioned generation yet. KV cache, full prompt prefill, full
native tokenizer JSON/BPE support, and uncapped decode are still future
milestones.

### Current Safe Execution Probe

`runSelectedNormProbe(session_handle)` is the first deliberately tiny native
math probe in the Gypsy protocol.

It requires:

- a warmed session
- constructed selected resident arrays
- selected resident norm weights:
  - `model.layers.0.input_layernorm.weight`
  - `model.layers.0.self_attn.q_norm.weight`
  - `model.norm.weight`

It runs deterministic MLX RMSNorm subprobes for layer-0 input norm, layer-0
q-norm, and final norm. The probe returns only scalar summaries, first-value
samples, timings, and explicit readback reasons. It does not execute projection
math, LoRA application, attention, logits, sampling, or generation.

The next native execution step is:

- `run_selected_quantized_projection_probe`

`runSelectedQuantizedProjectionProbe(session_handle)` is the first deliberately
tiny quantized projection math probe.

It requires:

- a warmed session
- constructed selected resident arrays
- selected layer0 `q_proj` packed U32 weight
- selected layer0 `q_proj` BF16 scales and biases

It runs deterministic MLX `quantized_matmul` for layer0 `q_proj` using:

- input width `2560`
- output width `4096`
- group size `64`
- `4` bits
- affine mode

The probe returns only scalar summaries, first-value samples, timing, and one
explicit readback reason. It does not execute LoRA, q_norm, RoPE, attention,
logits, sampling, or generation.

The next native execution step is:

- `run_selected_lora_projection_delta_probe`

`runSelectedLoraProjectionDeltaProbe(session_handle)` is the first deliberately
tiny adapter math probe.

It requires:

- a warmed session
- constructed selected resident arrays
- selected layer20 `q_proj` LoRA A/B tensors

It runs deterministic MLX dense matmuls for the adapter delta:

```text
scale * ((input @ lora_a) @ lora_b)
```

For the current adapter this means:

- input width `2560`
- rank `8`
- output width `4096`
- scale `20.0`

The probe returns only scalar summaries, first-value samples, timing, and one
explicit readback reason. It does not execute the base quantized projection,
base-plus-LoRA addition, q_norm, RoPE, attention, logits, sampling, or
generation.

The next native execution step is:

- `run_selected_base_plus_lora_projection_probe`

`runSelectedBasePlusLoraProjectionProbe(session_handle)` is the first selected
probe that combines a quantized base projection with its LoRA adapter delta.

It requires:

- a warmed session
- constructed selected resident arrays
- selected layer20 `q_proj` packed U32 weight
- selected layer20 `q_proj` BF16 scales and biases
- selected layer20 `q_proj` LoRA A/B tensors

It runs deterministic MLX math:

```text
quantized_matmul(input, base_q_proj) + scale * ((input @ lora_a) @ lora_b)
```

This is the ordering point needed before Qwen q_norm/RoPE for `q_proj` and
`k_proj`: the adapter delta is part of the projection output before norm/RoPE.
The probe returns only scalar summaries, first-value samples, timing, and
explicit readback reasons. It does not execute q_norm, RoPE, attention, logits,
sampling, or generation.

The next native execution step is:

- `run_selected_q_norm_after_lora_probe`

`runSelectedQNormAfterLoraProbe(session_handle)` proves the next Qwen ordering
boundary for adapter-active q projection:

```text
q = quantized_matmul(input, base_q_proj)
q = q + scale * ((input @ lora_a) @ lora_b)
q = q_norm(q reshaped by head)
```

It requires selected layer20 base q-proj tensors, layer20 q-proj LoRA A/B, and
layer20 q_norm. It reshapes the 4096-wide q projection to `[32, 128]` and
applies MLX RMSNorm over the head dimension. This probe still does not execute
RoPE, attention, logits, sampling, or generation.

The next native execution step is:

- `run_selected_rope_after_q_norm_probe`

`runSelectedRopeAfterQNormProbe(session_handle)` proves the selected
adapter-active q path reaches positional rotation:

```text
q = base_q_proj + lora_q_delta
q = q_norm(q reshaped as [32, 128])
q = rope(q, position=7, theta=5000000, traditional=false)
```

The fixed position is a probe value only. The generation path must use the
actual token position from full prompt prefill/decode state.

The probe returns only scalar summaries, first-value samples, timing, and
explicit readback reasons. It does not execute k/v projection, attention,
logits, sampling, or generation.

The next native execution step is:

- `add_selected_kv_projection_path`

`runSelectedKvProjectionPathProbe(session_handle)` expands the selected
attention-input proof from q only to q/k/v:

- q projection: base + LoRA, q_norm, RoPE
- k projection: base + LoRA, k_norm, RoPE
- v projection: base + LoRA, no norm/RoPE

The selected layer is still layer20 because that is inside the current adapter
range. This probe proves the three vectors needed by attention can be produced
with native resident tensors and the correct Qwen ordering.

The next native execution step is:

- `run_selected_single_token_attention_probe`

`runSelectedSingleTokenAttentionProbe(session_handle)` proves the selected
q/k/v vectors can feed native attention:

- q: base + LoRA, q_norm, RoPE
- k: base + LoRA, k_norm, RoPE
- v: base + LoRA
- GQA expansion: 8 KV heads repeated to 32 Q heads
- attention: MLX scaled dot-product attention over a single selected token

The sequence length is intentionally `1` for this milestone. It proves the
native attention boundary without introducing KV cache state yet.

The next native execution step is:

- `run_selected_o_projection_path_probe`

`runSelectedOProjectionPathProbe(session_handle)` proves the selected
attention output can feed layer20 `self_attn.o_proj`:

- input: MLX single-token attention output
- base: resident quantized `o_proj`
- adapter: resident `o_proj` LoRA A/B
- output: 2560-wide post-attention projection vector

This is still a selected-path probe. It does not add residual state yet.

The next native execution step is:

- `run_selected_post_attention_residual_probe`

`runSelectedPostAttentionResidualProbe(session_handle)` proves the selected
post-attention residual boundary:

- q/k/v: same adapter-active attention path
- o: adapter-active `o_proj`
- residual: original layer input + `o_proj` output
- output: 2560-wide post-attention residual vector

The next native execution step is:

- `run_selected_post_attention_rmsnorm_probe`

`runSelectedPostAttentionRmsNormProbe(session_handle)` proves the selected
post-attention normalization boundary:

- input: 2560-wide post-attention residual
- norm: layer20 `post_attention_layernorm`
- output: 2560-wide MLP input vector

The next native execution step is:

- `run_selected_mlp_gate_up_probe`

`runSelectedMlpGateUpProbe(session_handle)` proves the selected MLP expansion
boundary:

- input: 2560-wide post-attention normalized vector
- gate: adapter-active layer20 `mlp.gate_proj`
- up: adapter-active layer20 `mlp.up_proj`
- output: two 9728-wide MLP intermediate vectors

The next native execution step is:

- `run_selected_mlp_activation_probe`

`runSelectedMlpActivationProbe(session_handle)` proves the selected MLP
activation boundary:

- gate: adapter-active gate projection
- up: adapter-active up projection
- activation: `SiLU(gate) * up`
- output: 9728-wide activated MLP intermediate vector

The next native execution step is:

- `run_selected_mlp_down_projection_probe`

`runSelectedMlpDownProjectionProbe(session_handle)` proves the selected MLP
down projection boundary:

- input: 9728-wide activated MLP intermediate
- down: adapter-active layer20 `mlp.down_proj`
- output: 2560-wide MLP output vector

The next native execution step is:

- `add_selected_mlp_residual_path`

`runSelectedMlpResidualProbe(session_handle)` proves the selected MLP residual
boundary:

- input: 2560-wide post-attention residual stream
- normalization: layer20 `post_attention_layernorm`
- MLP: adapter-active gate/up, `SiLU(gate) * up`, adapter-active down
- residual: down projection added back to the post-attention residual stream
- output: 2560-wide layer output vector

The next native execution step is:

- `add_selected_layer_output_contract`

`runSelectedLayerOutputContractProbe(session_handle)` proves the selected layer
output handoff contract:

- producer: `runSelectedMlpResidualProbe`
- output: native `float32` value shaped `[1, 2560]`
- valid consumers: next-layer input, final-norm input, or explicit debug summary
- CoffeeScript payload transfer: forbidden
- readbacks: zero
- ordering invariants: LoRA before q/k norm, q/k norm before RoPE, attention
  and MLP residuals applied in Qwen order

The next native execution step is:

- `add_selected_next_layer_handoff_probe`

`runSelectedNextLayerHandoffProbe(session_handle)` proves the selected
layer-to-layer handoff contract:

- source: native layer20 output shaped `[1, 2560]`
- target: layer21 input shaped `[1, 2560]`
- layer21 model descriptors are present for norms, q/k/v/o, and MLP projections
- layer21 adapter descriptors are present for all seven LoRA targets
- layer21 payload residency is not constructed yet
- CoffeeScript payload transfer: forbidden
- readbacks: zero

The next native execution step is:

- `add_selected_layer21_residency_probe`

`runSelectedLayer21ResidencyProbe(session_handle)` proves the target layer
payload residency boundary:

- target: layer21
- arrays: layer21 norms, q/k/v/o projections, MLP projections, and all seven
  adapter LoRA targets
- constructed: native MLX arrays only
- session mutation: none; the probe is local and leaves the existing layer20
  selected-array state intact
- CoffeeScript payload transfer: forbidden
- readbacks: zero

The next native execution step is:

- `add_selected_layer21_input_norm_probe`

`runSelectedLayer21InputQkvProbe(session_handle)` proves the target layer can
start executing from the handoff value:

- input contract: layer20 output shaped `[1, 2560]`
- layer21 input RMSNorm runs in MLX
- adapter-active q/k/v projections run in MLX
- LoRA deltas are added before q/k norm
- q_norm and k_norm run before RoPE
- q and k receive RoPE at the next sequence position
- v remains projection-only
- session mutation: none; layer21 arrays are local to the probe

The next native execution step is:

- `add_selected_layer21_attention_probe`

`runSelectedFinalLogitsProbe(session_handle)` proves the generation tail:

- input contract: layer output shaped `[1, 2560]`
- final RMSNorm runs in MLX
- tied `model.embed_tokens` quantized projection produces logits
- native code scans logits and returns top-token metadata
- full logits are not returned to CoffeeScript
- session mutation: none; final norm and embedding arrays are local to the probe

The next native execution step is:

- `replace_selected_hidden_contract_with_full_prompt_layer_stack`

### Milestone 4: Safe Resident Loading

Replace the old eager fp32 loader with a resident MLX-friendly loader.

Rules:

- no full fp32 upcast
- no forced eval of every tensor unless intentionally warmed
- no full CPU mirror of weights
- explicit memory metadata
- adapter tensors loaded once and kept native

### Milestone 5: Native Generation Loop

Generation should run inside native code:

- prompt prefill
- decode loop
- KV updates
- sampling/top-k
- EOS/stop checks

CoffeeScript calls one `generate(...)` operation per request, not one
`forwardStep(...)` per token.

## Required Metadata For Trust

Normal generation should return enough scalar metadata to prove the intended
path ran:

- active backend
- model path
- adapter active/requested
- tokenizer/chat mode
- prompt token count
- generated token count
- stop reason
- generation timing
- tokens per second
- cache backend
- cache bytes active/reserved
- readback count and reasons
- fallback count and reasons
- session/model cleanup counts

## Test Discipline

Runtime/Metal tests are human-run only through repo-root `test.sh`.

`test.sh` must:

- write logs under `test/logs/...`
- capture stdout and stderr separately
- write a manifest
- avoid old/disproved baselines
- avoid unsafe model loading unless the test is explicitly about the new safe
  loader

The current safe smoke is addon-export only.
