# Rusty Backend Boundary Survey

Date:
- local survey performed on branch `rusty`

Scope:
- survey only
- no inference implementation
- no production wiring

Additional local reference surveyed:

- `../development/node-mlx`

That repo is useful because it demonstrates an existing direct binding approach
from a host runtime into MLX C++, even though its MLX version is obsolete for
our purposes.

Design rule:

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

## Executive summary

The cleanest path is:

- `Rust + small C++ shim`

Reason:

- local MLX provides a real native library and public C++ headers
- local MLX does **not** expose an obvious stable first-class C API for model/session/tokenizer work
- `mlx-lm` model loading, tokenizer loading, and adapter logic currently live in Python
- direct Rust binding to the MLX C++ surface is possible, but it is a larger ABI/lifetime risk than a narrow C-compatible shim

Short recommendation:

- keep the resident bridge process in Rust
- add a very small native shim layer that exposes a narrow C ABI
- let that shim own MLX C++ objects, tokenizer objects, and session/KV state
- let Rust own protocol, handle tables, process lifecycle, and request dispatch

This recommendation is now also reflected in the bridge-side `backend_probe`
classification:

- `c_api_detected: false`
- `cpp_api_detected: true`
- `python_package_detected: true`
- `recommended_path: rust_cpp_shim`

First concrete step now scaffolded:

- a tiny C++ shim with a C ABI lives under `rusty/bridge/native/`
- Rust builds and calls that shim through `shim_probe`
- this keeps the intended ownership boundary on the native side before any MLX
  inference code is introduced

Next concrete step now scaffolded:

- `mlx_link_probe` verifies that the shim can include and link the local MLX
  C++ boundary
- include/lib discovery is overrideable with:
  - `MLX_INCLUDE_DIR`
  - `MLX_LIB_DIR`
- the current probe uses the smallest safe surface found locally:
  - `mlx/device.h`
  - `default_device()`
  - `is_available(...)`

Runtime diagnosis now also records:

- the exact failure stage of the tiny-array lifecycle
- the exact exception string returned by the native MLX runtime
- whether CPU fallback changes the result

Standalone native smoke-test result:

- added `rusty/native_smoke/mlx_smoke.cpp`
- built directly with:
  - `/usr/bin/clang++ -std=c++17 -I/opt/homebrew/Cellar/mlx/0.30.0/include mlx_smoke.cpp -L/opt/homebrew/Cellar/mlx/0.30.0/lib -Wl,-rpath,/opt/homebrew/Cellar/mlx/0.30.0/lib -lmlx -o mlx_smoke`
- the standalone binary and the Rust bridge binary both link:
  - `/opt/homebrew/opt/mlx/lib/libmlx.dylib`
- standalone constructor results:
  - scalar int: constructor unknown exception
  - scalar float: constructor unknown exception
  - vector float: constructor unknown exception
  - `zeros({1})`: constructor unknown exception
  - `ones({1})`: constructor unknown exception

Conclusion from the smoke test:

- the tiny-array failure is not caused by Rust/C++ shim linkage
- the same failure reproduces in plain standalone C++ against the coherent
  Homebrew MLX install
- this strongly points to MLX runtime behavior on this machine rather than to
  the Rust bridge architecture

Expanded smoke matrix results:

- `-std=c++17` against `/opt/homebrew/Cellar/mlx/0.30.0`:
  - builds
  - runtime failure unchanged: constructor unknown exception
- `-std=c++20` against `/opt/homebrew/Cellar/mlx/0.30.0`:
  - builds
  - runtime failure unchanged: constructor unknown exception
- `-std=c++17 -stdlib=libc++` against `/opt/homebrew/Cellar/mlx/0.30.0`:
  - builds
  - runtime failure unchanged: constructor unknown exception
- `/opt/homebrew/opt/mlx` symlink root:
  - builds
  - runtime failure unchanged: constructor unknown exception
- local `../development/mlx` source tree:
  - headers and `MLXConfig.cmake` exist
  - no local `libmlx.dylib` was present under `build/`
  - smoke test intentionally skipped

Python comparison:

- tiny `mlx.core` array creation crashed natively during import/use
- rerun explicitly with the repo venv interpreter:
  - `/Users/jahbini/writeStory/.venv/bin/python`
- the exception was an `NSRangeException` from MLX Metal device construction
- this means Python MLX is not a clean working baseline on this machine either

Runtime policy:

- the native MLX object-handle runtime test succeeded from a normal terminal
- observed result:
  - `mlx_create_test_array handle: 2`
  - `mlx_test_array_sum sum: 7`
- this proves the boundary layering we want:
  - CoffeeScript sees only command/result data
  - Rust routes JSONL commands
  - the C++ shim owns MLX objects
  - MLX/Metal owns device memory
  - no MLX tensor escapes into Rust or CoffeeScript
- Codex/sandbox may fail because Metal device enumeration can return no devices
- therefore the runtime probe must remain gated behind:
  - `RUSTY_RUN_MLX_RUNTIME=1`
- the default verifier should not run MLX object allocation tests unless that
  env var is set
- build/link probes remain safe by default

This is the transition from probe phase to real resource lifecycle phase.
The next native milestone is actual tokenizer parsing and model weight loading,
still without exposing tensors or model objects to Rust or CoffeeScript.

Resource lifecycle shell:

- `load_tokenizer` / `unload_tokenizer` create and free tokenizer handles only
- `load_tokenizer` now grounds the handle in real tokenizer files on disk and
  records:
  - `path`
  - `detected_files`
  - `tokenizer_kind`
- for `tokenizer.json` files, the bridge now also records lightweight
  HuggingFace metadata only:
  - `model_type`
  - `normalizer_type`
  - `pre_tokenizer_type`
  - `decoder_type`
  - `vocab_size`
  - `merges_count`
  - `added_tokens_count`
- fixture-only `tokenizer_encode` and `tokenizer_decode` now prove the command
  shape and handle ownership path using exact vocab lookup from
  `tokenizer.json`; this is not yet real tokenizer compatibility
- when `RUSTY_RUN_MLX_RUNTIME=1`, encoded token IDs can be promoted to a native
  MLX token-array handle inside the C++ shim, with metadata-only inspection and
  clean free behavior
- when `RUSTY_RUN_MLX_RUNTIME=1`, `native_mock_forward` proves a live session
  can consume a token-array handle and return a logits-like MLX array handle,
  again with opaque ownership only
- when `RUSTY_RUN_MLX_RUNTIME=1`, `native_mock_sample` reads the logits-like
  array inside the shim and returns a plain token id, completing the fake
  generation-loop shape without model weights
- when `RUSTY_RUN_MLX_RUNTIME=1`, `native_mock_generate` wraps the full fake
  loop into one command and uses shim handle counts to verify the temporary
  token/logit arrays are released
- `tokenizer_info` reports tokenizer metadata without exposing internals
- `inspect_model_dir` inspects model directory structure and light config
  metadata without loading weights
- `inspect_tensor_descriptors` classifies tensor groups from safetensors
  headers only, including shape, dtype, and byte offsets, without loading
  payloads
- `load_tensor_group` can read one logical quantized block from a safetensors
  shard into shim-owned native records while keeping payload bytes off the
  Rust boundary
- `load_embedding_group` now promotes `model.embed_tokens` into a distinct
  native embedding handle with metadata-only inspection and freed-handle
  cleanup
- `dequantize_group_slice` now proves a tiny placeholder dequantization path
  from a loaded quantized group into a shim-owned `float32` array. The
  current math is explicitly provisional (`value = nibble * scale + bias`)
  and must be matched against MLX's real quantized kernel layout before any
  meaningful matmul or generate work
- `compare_dequant_slice` compares that provisional unpack against MLX's
  native `dequantize` on a tiny packed block when the comparison path is
  available, and otherwise records the searched public entry points and
  headers
- `quantized_linear_slice_probe` demonstrates that a tiny input vector can
  participate in arithmetic against the verified quantized slice, but it is
  still not full matmul
- `quantized_linear_rows_probe` extends that to a tiny multi-row projection and
  proves row traversal for the quantized slice path
- `quantized_linear_fullrow_probe` traverses the full packed input width for
  one q_proj group, with an explicit `input_len: 2560` check on model4, but it
  still uses the provisional nibble/scale/bias layout
- `quantized_linear_vector_probe` extends that to the full 4096-row output
  vector for q_proj and records summary slices plus a simple checksum, but it
  is still provisional and not a real matmul kernel
- sparse and dense deterministic full-width q_proj sanity checks pass
- `rmsnorm_probe` is the next isolated arithmetic primitive after dense q_proj:
  it uses the layer 0 input_layernorm weight and `rms_norm_eps` from config to
  verify RMSNorm math before any attention path is started
- `layer0_single_token_probe` now composes the first layer-0 forward-pass
  skeleton for one token: embedding row lookup, input RMSNorm, q/k/v
  projections, single-token grouped-query attention, o_proj, and residual add.
  It reports lengths, checksums, first values, timing, and handle counts while
  keeping all tensor payloads inside the shim.
- `layer0_mlp_probe` extends the verifier-only layer 0 path through the MLP:
  post-attention RMSNorm, gate/up projections, SiLU(gate) * up, down
  projection, and final residual add. It reports lengths, checksums, first
  values, timing, and handle counts while creating no persistent handles.
- `layer0_block_probe` consolidates the full verifier-only layer 0 block into
  one log section: token id, input embedding checksum, attention residual
  checksum, MLP residual checksum, output length, first output values, timing,
  and handle counts before/after.
- `layer_stack_probe` generalizes that verifier-only scaffold across N blocks:
  the current verifier runs token id 1 through the first 4 layers, reporting
  per-layer output checksums, final output length, first values, timing, and
  handle counts without adding production generation.
- `full_stack_single_token_probe` extends the same verifier-only path through
  all configured model layers, final RMSNorm, and tied embedding projection to
  report top logits. It still has no KV cache, no generation loop, and no
  production wiring.
- `greedy_next_token_probe` narrows that full-stack path to the argmax next
  token and decodes it through a loaded tokenizer handle for verifier coverage
  only. It still has no KV cache, no generation loop, and no production wiring.
- `greedy_two_token_probe` repeats that verifier-only full-stack pass twice:
  token id 1 produces argmax token A, token A produces argmax token B, and the
  two generated tokens are decoded together. It still has no KV cache and no
  production generation loop.
- `greedy_session_generate_probe` keeps a native model/session handle live
  across three verifier-only greedy steps. Each step still runs a full-stack
  pass without KV cache, but the model/session/tokenizer lifecycle is now
  represented explicitly.
- `greedy_prompt_session_probe` adds a fixed-prompt verifier path: the loaded
  tokenizer maps `hello` to token ids, those prompt tokens are run through the
  same full-stack greedy session path, and one next token is decoded. It still
  has no KV cache and no production generation loop.
- `incremental_session_probe` introduces verifier-only native-session KV cache
  metadata beside the full recompute path. It compares three greedy tokens from
  prompt `hello`, decoded text, checksum tolerance, timing totals, and explicit
  CPU/provisional fallback stage while preserving zero shim handles after the
  probe.
- real matmul/generate remains blocked until the dequant path is verified or
  the native MLX quantized linear path is used directly
- `quantization_layout_probe` now documents the raw quantized packing layout
  for one loaded group, including tiny U32/BF16 debug samples and the inferred
  block size
- the slice probe is runtime-gated because it materializes a native MLX array
  and should be exercised from a normal terminal with Metal/IOKit access
- `load_model_native` / `unload_model_native` create and free native model
  handles only
- `create_native_session` / `free_native_session` create and free a session
  handle that depends on a live native model and tokenizer
- the shell validates handle relationships and freed-handle failures without
  loading real weights yet
- `load_layer_groups` / `layer_groups_info` / `free_layer_groups` now compose
  a full Qwen3 layer wrapper over native tensor-group handles, keeping the
  quantized attention/MLP groups separate from the norm groups while still
  exposing only opaque handles and metadata at the Rust boundary
- `model_load_plan` now synthesizes a read-only model loading plan from config
  and safetensors descriptors without loading payloads, which is the next
  useful step before any real weight loading
- `create_model_descriptor` now promotes that plan into a persistent opaque
  descriptor handle with loaded_weights=false, loaded_layers=0, and the same
  validated metadata without loading tensor payloads

Milestone proven:

- text -> encode -> token array -> native session -> mock forward -> logits ->
  sample -> decode
- `native_mock_generate` wraps that loop into one command
- handle counts before and after the runtime path prove no temporary native
  token or logits arrays leak
- Rust and CoffeeScript only see handles, text, token ids, and metadata
- the C++ shim owns the MLX arrays
- no model weights or real inference yet

Next real milestone:

1. inspect the real model directory layout
2. detect `config.json`, tokenizer files, and `safetensors` files
3. parse model config metadata
4. create a native model descriptor
5. only then attempt real weight loading

Discovery clues:

- `brew info mlx` reports stable `0.31.2` with installed `0.30.0` on request
- `pkg-config` is not installed in the current shell environment
- `MLXConfig.cmake` exists in:
  - `/opt/homebrew/Cellar/mlx/0.30.0/share/cmake/MLX/MLXConfig.cmake`
  - `/Users/jahbini/development/mlx/build/MLXConfig.cmake`
  - `.venv/lib/python3.14/site-packages/mlx/share/cmake/MLX/MLXConfig.cmake`

Direct C++ usability:

- Homebrew C++ MLX is linkable and callable enough to reach runtime
- but the coherent Homebrew install still fails at tiny array construction
- so Homebrew MLX appears linkable but not operationally usable for the smallest array case on this machine

## 1. Native MLX callable surfaces available locally

### A. C API

What I found:

- no obvious standalone MLX C API for the full runtime/model/tokenizer surface
- no dedicated `c_api.h` or similar public C interface was found in the local MLX tree
- search for `extern "C"` only turned up limited/internal cases, not a public bridge-ready API

Evidence:

- local MLX source tree:
  - `../development/mlx`
- `extern "C"` hits were sparse and not a public general API surface

Conclusion:

- there is not an obvious first-class public C API here for end-to-end LLM serving

### B. C++ API

What I found:

- MLX exposes a broad public C++ API through installed headers and source headers
- the main umbrella header is:
  - `../development/mlx/mlx/mlx.h`
- installed headers are also present in the active venv:
  - `.venv/lib/python3.14/site-packages/mlx/include/mlx/*.h`

Relevant examples:

- `.venv/lib/python3.14/site-packages/mlx/include/mlx/mlx.h`
- `.venv/lib/python3.14/site-packages/mlx/include/mlx/io.h`
- `.venv/lib/python3.14/site-packages/mlx/include/mlx/export.h`
- `../development/mlx/mlx/io.h`
- `../development/mlx/mlx/io/load.h`

Important detail:

- `mlx/api.h` exposes `MLX_API` symbol visibility for public native APIs
- that indicates a real exported native library boundary, but still in C++ form

### C. Headers

Installed headers found in active venv:

- `.venv/lib/python3.14/site-packages/mlx/include/mlx/allocator.h`
- `.venv/lib/python3.14/site-packages/mlx/include/mlx/api.h`
- `.venv/lib/python3.14/site-packages/mlx/include/mlx/array.h`
- `.venv/lib/python3.14/site-packages/mlx/include/mlx/device.h`
- `.venv/lib/python3.14/site-packages/mlx/include/mlx/io.h`
- `.venv/lib/python3.14/site-packages/mlx/include/mlx/mlx.h`
- plus many others under the same include tree

### D. Libraries

Installed native artifacts found in the active venv:

- `.venv/lib/python3.14/site-packages/mlx/lib/libmlx.dylib`
- `.venv/lib/python3.14/site-packages/mlx/core.cpython-314-darwin.so`

Link dependencies observed:

- `Metal.framework`
- `Foundation.framework`
- `QuartzCore.framework`
- `Accelerate.framework`
- `libc++`

This confirms:

- MLX native execution is already packaged locally as a dynamic library
- the Python extension links against `libmlx.dylib`

### E. Build artifacts

Local source/build trees:

- `../development/mlx`
- `../development/mlx-lm`

Notable build artifact found:

- `../development/mlx/build/mlx/io/libgguflib.a`

Also notable:

- `../development/mlx/CMakeLists.txt` clearly builds MLX as a native C/C++ project
- `../development/mlx/python/src/CMakeLists.txt` builds Python bindings via `nanobind`

### F. MLX-LM layer

What I found:

- `mlx-lm` appears to be Python-first
- no native serving library surfaced in the local `mlx-lm` tree
- local `mlx-lm` uses:
  - Python model class loading
  - Hugging Face tokenizer machinery
  - Python adapter / LoRA utilities

Evidence:

- `../development/mlx-lm/mlx_lm/utils.py`
- `../development/mlx-lm/mlx_lm/tokenizer_utils.py`
- `../development/mlx-lm/setup.py`

Important consequence:

- the native MLX tensor library is available
- but the LLM-serving convenience layer is not already packaged as a clean native API

### G. node-mlx reference project

What I found:

- `../development/node-mlx` is a Node-API binding layer over MLX C++
- it vendors MLX source under:
  - `../development/node-mlx/deps/mlx`
- it builds a native addon:
  - `node_mlx.node`
- it exposes a large TypeScript surface over the native binding

Useful files:

- `../development/node-mlx/CMakeLists.txt`
- `../development/node-mlx/src/bindings.cc`
- `../development/node-mlx/src/array.h`
- `../development/node-mlx/lib/core.ts`
- `../development/node-mlx/node_mlx.node.d.ts`

Why it matters:

- it proves that direct host-language binding to MLX C++ is feasible
- it also demonstrates the downside for our architecture:
  - the host runtime directly owns MLX objects
  - JS/GC interaction becomes part of the object lifetime story

Architectural implication for `rusty`:

- `node-mlx` is a useful reference for:
  - CMake wiring
  - C++ object wrapping
  - API surface mapping
- but it is **not** the ownership model we want for writeStory
- our goal is specifically to avoid JS owning arrays/tensors/native objects

## 2. Can Rust call directly through bindgen/cbindgen?

### Direct Rust via bindgen

Possible only in a limited sense.

Why:

- `bindgen` is best when there is a C ABI
- the surfaced MLX boundary here is primarily C++
- C++ direct binding from Rust is possible with other tooling patterns, but it is much more fragile

Problems with direct binding to current C++ surface:

- template-heavy and STL-heavy signatures
- C++ ownership semantics
- exceptions
- ABI stability concerns
- Apple framework / Metal-linked runtime details

Additional evidence from `node-mlx`:

- another project already chose a direct C++ binding route rather than a clean C
  API route
- that strengthens the conclusion that the practical public surface today is the
  C++ library, not a bridge-ready C interface

### cbindgen

Not applicable to MLX as-is.

Why:

- `cbindgen` is for generating C headers from Rust
- it does not solve Rust calling existing C++ MLX APIs

Conclusion:

- direct Rust binding is technically possible, but not the clean first move
- plain `bindgen` alone is not the right tool for the current exposed surface

## 3. Is a small C++ shim safer than direct Rust binding?

Yes.

This is the safest path for the first real native bridge milestone.

Why:

- C++ shim can speak MLX’s native language directly
- shim can hide:
  - `mlx::core::array`
  - model classes
  - tokenizer instances
  - KV cache structures
  - exception translation
  - Metal/context lifetime
- Rust can talk to the shim through a narrow C ABI

What the shim should do:

- create/destroy model handles
- create/destroy tokenizer handles
- create/destroy session handles
- encode prompt text
- step generation one token at a time
- decode output incrementally
- apply adapters if supported

What the shim should not do:

- own protocol
- own request queueing
- own JSONL transport
- own high-level orchestration

Those stay in Rust.

Why `node-mlx` strengthens this conclusion:

- `node-mlx` shows the amount of wrapper code needed just to project MLX into a
  host runtime
- reproducing that kind of direct object exposure in Rust would still leave us
  with a large native binding surface
- a narrower shim is more aligned with the resident-bridge goal

## 4. Minimum native calls needed

These are the minimum native operations the future bridge will need, regardless
of whether the implementation is direct Rust binding or Rust + shim.

### A. load_model

Needed native operation:

- create model object from local model directory
- load weight files from `model*.safetensors`
- prepare runtime state for inference

Likely native inputs:

- model path
- quantization/runtime options
- optional adapter path later

Likely native output:

- opaque model handle

### B. tokenize / encode

Needed native operation:

- convert input text to token ids

Complication:

- local `mlx-lm` tokenizer flow currently depends on Hugging Face tokenizer logic in Python
- tokenizer loading may be the first major boundary problem if staying entirely native

Likely native output:

- opaque tokenizer handle
- encoded token buffer owned natively

### C. generate one token

Needed native operation:

- advance one session step using:
  - model
  - current token stream
  - KV cache/session state
- return next token id

Likely native output:

- token id
- updated session/KV state, owned natively

### D. decode

Needed native operation:

- convert token id stream or incremental token to printable text segment

This may need:

- tokenizer-side streaming detokenizer state

### E. free resources

Needed native operations:

- free model handle
- free tokenizer handle
- free session handle
- free KV cache handle
- free job state if retained

## 5. Risks

### Ownership / lifetime

Highest risk.

Why:

- MLX arrays and model internals must never leak into JS ownership
- Rust must not accidentally expose raw tensor pointers back across the protocol boundary
- if Rust binds directly to C++, drop semantics become delicate

### Thread safety

Unclear from this survey.

Conservative assumption:

- treat MLX runtime/model/session operations as single-owner and serialize bridge-side inference until proven otherwise

Best early rule:

- one bridge thread/event loop owns ML handles
- no concurrent mutation of the same model/session/KV objects

### Metal device context

Important risk.

Evidence:

- installed MLX native library links directly to Metal and QuartzCore

Practical implication:

- bridge should probably centralize MLX runtime use in one resident process/thread model
- device/context assumptions should not be spread across many short-lived subprocesses

### Tokenizer dependency

Major architectural risk.

Evidence:

- `mlx-lm` tokenizer path currently depends on Python `transformers`
- streaming detokenizers are implemented in Python in local `mlx_lm/tokenizer_utils.py`

Implication:

- tokenizer support may be harder than raw tensor inference
- fully native tokenizer loading may require:
  - using Hugging Face tokenizer artifacts natively through another library
  - or introducing a tokenizer-specific shim/library boundary
  - or keeping some model-family-specific tokenizer handling in a C++ sidecar

### Adapter / LoRA support

Important medium-term risk.

Evidence:

- local `mlx-lm` adapter flow is Python-side in `mlx_lm/tuner/utils.py` and related logic

Implication:

- base model loading can likely be solved sooner than adapter support
- adapter/LoRA application may require extra native reimplementation or another shim boundary

### Host-runtime ownership pressure

Important architectural risk highlighted by `node-mlx`.

Why:

- direct host bindings make it easy for the host runtime to own or retain native
  ML objects longer than intended
- `node-mlx` is useful for API ideas, but its direct Node ownership model is
  exactly what `rusty` is trying to avoid for long-lived generation sessions

### Safetensors loading

Better position here.

Evidence:

- MLX native C++ surface already exposes `load_safetensors` in `mlx/io.h`

Implication:

- raw tensor weight loading is probably feasible natively
- higher-level model construction is still the harder part

## 6. Recommendation

### Best recommendation: Rust + C++ shim

This is the cleanest first real implementation path.

Why:

- preserves Rust as the resident bridge owner
- keeps JS far away from tensors and GPU memory
- uses MLX through its natural C++ boundary
- avoids forcing Rust to directly own a large unstable C++ ABI surface
- allows a narrow, explicit C ABI between Rust and native ML code
- keeps `node-mlx` in the role of reference material instead of architectural template

### Not recommended as first move: direct Rust binding

Why not first:

- no obvious public C API for the needed LLM lifecycle
- MLX surface is C++
- tokenizer/model-loading pieces are not already exposed as a small stable native API
- ownership and exception bridging risk is higher

### Less preferred fallback: native bridge in C++ with Rust protocol wrapper

Possible, but less clean.

Why less preferred:

- splits “bridge ownership” across two native layers
- weakens the point of using Rust as the long-lived process/lifecycle manager

Still acceptable if:

- the C++ side ends up needing to own nearly everything initially
- and Rust is only supervising protocol and restart behavior

## Final recommendation

Recommended path:

1. Keep resident bridge process in Rust
2. Add a minimal C ABI shim in C++
3. Let the shim own MLX objects and tokenizer/session internals
4. Let Rust own:
   - process lifetime
   - handle tables
   - protocol
   - request dispatch
   - restart behavior
5. Keep JS/CoffeeScript limited to:
   - orchestration
   - Memo visibility
   - opaque handles

That best matches the architectural goal:

- JavaScript owns orchestration
- Rust owns ML lifetime and memory
- ML objects never become JS-owned
