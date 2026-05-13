# Gypsy

`gypsy/` holds strategy and protocol documents for the next native generation
path.

The teaching frame is:

- Rusty taught the token journey through the model.
- Gypsy teaches why Metal is fast when the native boundary is shaped correctly.

This directory is documentation-first. It should describe the intended
CoffeeScript-to-native protocol, ownership boundaries, implementation
milestones, and known traps before the C++/Metal path is rewritten.

## Core Goal

Create the best intermediate protocol between CoffeeScript orchestration and
native MLX/Metal execution.

CoffeeScript should express generation intent. Native code should own tensor
residency, model weights, KV cache, adapters, scheduling, synchronization, and
sampling.

## Metal Strategy Rules

### Move Last First

Move computation into Metal from the output end of the generation chain
backward.

The useful order is:

- final logits / top token selection
- final RMSNorm
- final layer MLP down projection and residual
- final layer MLP activation and gate/up projections
- final layer attention output projection and residual
- attention value mix / scores / KV access
- q/k/v projection, q_norm/k_norm, RoPE
- earlier layers and prompt prefill once the terminal chain is correct

This keeps each new Metal step local and testable. If a token sequence changes,
the defect is in the newly moved terminal step, not somewhere hidden inside a
large rewritten forward pass.

Do not start by pushing the first model step into Metal and then dragging every
intermediate value back to CPU. That reproduces the same handoff problem the
new path is meant to remove.

### Use Metal Datatypes At Boundaries

Every value handed to a Metal kernel must use the datatype Metal expects, not a
convenient host reinterpretation.

Current concrete rules:

- packed INT4 model weights are raw `uint32_t` words and enter Metal as
  `device uint*`.
- BF16 scales, biases, and norm weights are raw 16-bit words and enter Metal as
  `device ushort*`; kernels explicitly convert BF16 bits to `float`.
- hidden states, activations, logits, and accumulation outputs are `float`
  unless a kernel explicitly states otherwise.
- scalar dimensions and indexes should use Metal-friendly `uint` constants or
  buffers, not mixed signed host integer types.

Do not pass BF16 payloads as `float*`. That bug produces plausible-looking
wrong math and wastes debugging time.

When a Metal step fails, debug the single boundary and its datatypes first:
shape, source pointer, element type, stride/layout, and conversion rule.

## What Not To Repeat

- Do not recreate Rusty as another manual per-layer teaching engine.
- Do not expose tensors or full logits to CoffeeScript in the production path.
- Do not make CoffeeScript drive one token or one layer at a time.
- Do not eagerly load/evaluate all Qwen weights and upcast them to float32.
- Do not add diagnostic readbacks to the hot loop by default.
- Do not treat "one larger Objective-C++ helper call" as the same thing as a
  Metal-owned execution program. If that helper still copies weights, waits at
  layer boundaries, or returns hidden vectors to host, it is still CPU
  orchestration.
- Do not rerun the failed manual direct-Metal stack variants as benchmarks.
  They are diagnostic history only.

## Failed Direct-Metal Stack Attempts

The recent direct-Metal experiments are now recorded as negative evidence.

Validated successes:

- The individual terminal kernels can be made numerically correct when fed the
  right shapes and raw Metal datatypes.
- Final logits/top-token, final RMSNorm, terminal MLP pieces, attention/o, and
  q/k/v plus q_norm/k_norm/RoPE were each validated in isolation.
- A terminal layer range could preserve known-good 64-token output when using
  carefully staged direct-Metal calls.

What failed:

- The staged path still made many host-orchestrated calls. CPU time stayed
  around 1.5 minutes while GPU time was far lower. That means CPU scheduling,
  buffer management, and synchronization dominated.
- A fused MLP tail was correct but did not materially improve speed. It removed
  some sub-step calls but still kept the layer loop and host boundary.
- A full-layer resident path initially produced garbage because shared
  projection buffers reused stale layer weights. Token-level strict validation
  caught this.
- A single terminal-stack command-buffer attempt was worse. It copied many
  layer weights through CPU-visible shared buffers before execution and caused
  high CPU time and memory pressure. This moved the orchestration cost; it did
  not eliminate it.

Conclusion:

The failed path was still CPU-led. The CPU continued to act as scheduler,
buffer manager, and validator. A real fast path must instead prebuild resident
Metal-side resources and execute a token or chunk of decode with minimal host
participation.

Do not continue this path by adding more small staged kernels. The next attempt
must start from a proper execution object:

- model weights resident once, not copied per token
- KV cache resident once, not reassembled through host buffers
- one command encoding plan for the decode step or decode chunk
- no hidden-vector readback between layers
- final readback limited to selected token id, stop status, and optional
  telemetry

## Starting Point

The initial native source was copied from `~/development/mlxCoffee/metal`.
That code is useful as a Node-addon baseline, but its old `loadModel` and
`forwardStep` protocol are not the target protocol.

The safe current baseline is:

- build the addon
- load `metal/metal_llm.node`
- verify native exports
- inspect Qwen model/tokenizer/adapter metadata from config and safetensors
  headers
- create opaque metadata handles for model, tokenizer, adapter, and session
- describe the Qwen layer group plan from model-handle metadata
- inspect global protocol status, live handles, warmed sessions, and cleanup
  state
- warm a session by opening read-only native file mappings for model and
  adapter safetensors, without constructing arrays or touching tensor payload
  pages
- resolve model and adapter tensor descriptors to session-owned mapped byte
  views, still without exposing payloads or constructing MLX arrays
- classify those mapped views into typed resident-array construction specs
  for quantized model components, norm weights, and LoRA matrices
- construct a deliberately small selected set of session-owned MLX arrays from
  mapped views to prove native array lifetime and idempotence. MLX requires
  arrays to own MLX-allocator buffers, so this step copies selected payloads
  natively into MLX-managed memory while still exposing no tensor payloads to
  CoffeeScript.
- assemble selected resident arrays into native execution groups for one
  quantized projection, norm weights, and one LoRA projection delta
- run a deliberately small selected norm probe over resident norm arrays,
  executing MLX RMSNorm math while keeping tensor payloads native and returning
  only scalar/sample summaries
- run a deliberately small selected quantized projection probe over layer0
  `q_proj`, executing MLX quantized matmul from resident packed weight/scales/
  biases and returning only scalar/sample summaries
- run a deliberately small selected LoRA delta probe over layer20 `q_proj`,
  executing `scale * ((input @ lora_a) @ lora_b)` from resident adapter arrays
  and returning only scalar/sample summaries
- run a deliberately small selected base-plus-LoRA projection probe over
  layer20 `q_proj`, executing base quantized projection plus adapter delta in
  the required Qwen ordering point
- run a deliberately small selected q_norm-after-LoRA probe over layer20
  `q_proj`, proving the combined base-plus-adapter projection is normalized
  after the adapter delta has been added
- run a deliberately small selected RoPE-after-q_norm probe over layer20
  `q_proj`, proving the adapter-active q path reaches positional rotation
  with Qwen RoPE settings
- run a deliberately small selected q/k/v projection path probe for layer20,
  proving adapter-active q and k reach norm/RoPE while v remains a projection
  output ready for attention value mixing
- run a deliberately small selected single-token attention probe for layer20,
  proving q/k/v can feed MLX scaled dot-product attention with Qwen GQA
  expansion
- run a deliberately small selected attention-to-`o_proj` probe for layer20,
  proving the MLX attention output can feed adapter-active `o_proj` without
  exposing tensor payloads to CoffeeScript
- run a deliberately small selected post-attention residual probe for layer20,
  proving the adapter-active `o_proj` output can be added back to the
  residual stream natively
- run a deliberately small selected post-attention RMSNorm probe for layer20,
  proving the residual stream can feed the MLP-side normalization boundary
- run a deliberately small selected MLP gate/up probe for layer20, proving
  post-attention RMSNorm can feed adapter-active MLP expansion projections
- run a deliberately small selected MLP activation probe for layer20, proving
  `SiLU(gate) * up` stays in native MLX execution
- run a deliberately small selected MLP down projection probe for layer20,
  proving the activated MLP intermediate can project back to hidden size
- run a deliberately small selected MLP residual probe for layer20, proving
  the adapter-active MLP output can be added back to the residual stream
- run a selected layer output contract probe for layer20, proving the completed
  layer output boundary is native, 2560-wide, and not a CoffeeScript payload
- run a selected next-layer handoff probe, proving layer20 output is a valid
  native input for layer21 and layer21 descriptors are available before payload
  residency is extended
- run a selected layer21 residency probe, proving the target layer's model and
  adapter arrays can be constructed natively without exposing payloads or
  mutating the current layer20 selected-array state
- run a selected layer21 input/qkv execution probe, proving layer20-shaped
  output can enter the next layer's input RMSNorm and adapter-active q/k/v path
  with q/k norm and RoPE ordering intact
- run a selected final logits probe, proving a native layer-output-shaped value
  can pass through final RMSNorm and tied embedding projection to produce top
  token metadata without returning full logits
- do not load model weight payloads

See `STATUS.md` for the current outcome and rejected-path summary.

See `PROTOCOL_DESIGN.md` for the first target protocol.

See `DIRECTIVE_FAILURES.md` for the record of repeated failures to follow the
human's CPU/GPU-boundary directives during the direct-Metal work.
