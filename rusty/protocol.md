# Rusty Bridge Protocol

Initial transport:

- resident child process
- stdin/stdout
- one JSON object per line
- request/response correlation by `id`

Later upgrade path:

- Unix socket
- same command model
- same opaque-handle discipline

## Requests

Every request is JSON:

```json
{"id":"1","cmd":"bridge_health","args":{}}
{"id":"2","cmd":"backend_probe","args":{}}
{"id":"3","cmd":"shim_probe","args":{}}
{"id":"4","cmd":"mlx_link_probe","args":{}}
{"id":"5","cmd":"load_model","args":{"path":"build/model4"}}
{"id":"6","cmd":"load_tokenizer","args":{"path":"build/tokenizer"}}
{"id":"7","cmd":"load_model_native","args":{"path":"build/model4"}}
{"id":"8","cmd":"create_native_session","args":{"model":"nmodel:1","tokenizer":"tok:1"}}
{"id":"9","cmd":"free_native_session","args":{"session":"nsess:1"}}
{"id":"10","cmd":"tokenizer_info","args":{"tokenizer":"tok:1"}}
{"id":"11","cmd":"create_model_descriptor","args":{"model_dir":"build/model4"}}
{"id":"12","cmd":"model_descriptor_info","args":{"descriptor":"mdesc:1"}}
{"id":"13","cmd":"free_model_descriptor","args":{"descriptor":"mdesc:1"}}
{"id":"14","cmd":"generate","args":{"session":"sess:1","prompt":"hello"}}
```

Fields:

- `id`
  - caller-owned correlation id
- `cmd`
  - command name
- `args`
  - object

## Responses

Success:

```json
{"id":"1","ok":true,"value":{"status":"alive"}}
```

Failure:

```json
{"id":"3","ok":false,"error":{"code":"unknown_session","message":"session not found"}}
```

Current validation-oriented error codes:

- `bad_request`
- `bad_args`
- `unknown_cmd`
- `unknown_handle`
- `already_freed`
- `bridge_shutting_down`

## Opaque handles

Handles are strings only.

Examples:

- `model:jim-v1`
- `sess:story-001`
- `kv:abc`
- `job:001`

JavaScript may store and pass these handles, but may not inspect them for ML
state or infer ownership from them.

## Initial commands

- `bridge_health`
- `backend_probe`
- `inspect_model_dir`
- `model_load_plan`
- `inspect_tensor_descriptors`
- `load_tensor_group`
- `tensor_group_info`
- `quantization_layout_probe`
- `compare_dequant_slice`
- `quantized_linear_slice_probe`
- `quantized_linear_rows_probe`
- `quantized_linear_fullrow_probe`
- `quantized_linear_vector_probe`
- `rmsnorm_probe`
- `layer0_single_token_probe`
- `layer0_mlp_probe`
- `layer0_block_probe`
- `layer_stack_probe`
- `full_stack_single_token_probe`
- `greedy_next_token_probe`
- `greedy_two_token_probe`
- `greedy_session_generate_probe`
- `greedy_prompt_session_probe`
- `incremental_session_probe`
- `free_tensor_group`
- `load_embedding_group`
- `embedding_group_info`
- `free_embedding_group`
- `dequantize_group_slice`
- `load_layer_groups`
- `layer_groups_info`
- `free_layer_groups`
- `shim_probe`
- `mlx_link_probe`
- `mlx_runtime_diagnose`
- `mlx_create_test_array`
- `mlx_test_array_sum`
- `mlx_free_test_array`
- `load_tokenizer`
- `unload_tokenizer`
- `tokenizer_info`
- `tokenizer_encode`
- `tokenizer_decode`
- `mlx_create_token_array`
- `mlx_token_array_info`
- `mlx_free_token_array`
- `native_mock_forward`
- `native_mock_sample`
- `native_mock_generate`
- `mlx_array_info`
- `mlx_free_array`
- `mlx_handle_counts`
- `load_model_native`
- `unload_model_native`
- `create_native_session`
- `free_native_session`
- `create_model_descriptor`
- `model_descriptor_info`
- `free_model_descriptor`
- `bridge_shutdown`
- `load_model`
- `unload_model`
- `create_session`
- `free_session`
- `generate`

## shim_probe

`shim_probe` verifies that Rust can build and call the first tiny C++ shim
boundary without linking to MLX yet.

Return shape:

```json
{
  "version": "rusty-mlx-shim/0.1",
  "reachable": true
}
```

## mlx_link_probe

`mlx_link_probe` checks whether the shim can compile and link against the local
MLX C++ surface.

It is intentionally tiny:

- no model loading
- no inference
- no meaningful tensor allocation
- no tensor exposure across the C ABI

Return shape:

```json
{
  "linked": true,
  "notes": []
}
```

## mlx_runtime_diagnose

`mlx_runtime_diagnose` inspects the local MLX runtime environment and reports
where the tiny-array lifecycle fails, if it fails.

It reports:

- `DYLD_LIBRARY_PATH`
- `MLX_METAL_PATH`
- resolved `libmlx` path
- `mlx.metallib` presence near `libmlx`
- `mlx.metallib` presence in known source/build locations
- current working directory
- exact exception string
- failure stage among:
  - `array construction`
  - `eval`
  - `scalar extraction`
  - `device sync`

## Native MLX resource lifecycle

These commands mark the transition from probe-only scaffolding to real
resource lifecycle management.

### `load_tokenizer`

Creates an opaque tokenizer handle and stores tokenizer discovery metadata for
now.

The returned metadata includes:

- `path`
- `detected_files`
- `tokenizer_kind`
- lightweight HuggingFace JSON metadata when available:
  - `model_type`
  - `normalizer_type`
  - `pre_tokenizer_type`
  - `decoder_type`
  - `vocab_size`
  - `merges_count`
  - `added_tokens_count`

```json
{"id":"6","cmd":"load_tokenizer","args":{"path":"build/tokenizer"}}
```

### `unload_tokenizer`

Releases a tokenizer handle.

```json
{"id":"7","cmd":"unload_tokenizer","args":{"tokenizer":"tok:1"}}
```

### `tokenizer_info`

Returns tokenizer metadata only. For now the bridge reports loaded `false` and
does not expose native tokenizer internals.

For HuggingFace JSON tokenizer files, the bridge also returns lightweight
metadata fields:

- `model_type`
- `normalizer_type`
- `pre_tokenizer_type`
- `decoder_type`
- `vocab_size`
- `merges_count`
- `added_tokens_count`

```json
{"id":"8","cmd":"tokenizer_info","args":{"tokenizer":"tok:1"}}
```

### `tokenizer_encode`

Fixture-only tokenizer path. The bridge does exact vocab lookup from
`tokenizer.json`, splitting text on spaces. It does not normalize, apply real
BPE merges, or perform inference.

Unknown tokens return `tokenizer_unknown_token`.

```json
{"id":"9","cmd":"tokenizer_encode","args":{"tokenizer":"tok:1","text":"hello world"}}
```

### `tokenizer_decode`

Fixture-only tokenizer path. The bridge uses inverse vocab lookup from
`tokenizer.json` and joins decoded pieces with spaces.

Unknown token ids return `tokenizer_unknown_token`.

```json
{"id":"10","cmd":"tokenizer_decode","args":{"tokenizer":"tok:1","tokens":[1,2]}}
```

### `mlx_create_token_array`

Creates a tiny MLX array inside the C++ shim from integer token IDs. This is a
fixture-only bridge step: Rust and CoffeeScript see only an opaque array
handle, not the native array object.

```json
{"id":"11","cmd":"mlx_create_token_array","args":{"tokens":[1,2]}}
```

### `mlx_token_array_info`

Returns metadata only for a live token array handle:

- `dtype`
- `ndim`
- `size`
- `shape`

```json
{"id":"12","cmd":"mlx_token_array_info","args":{"array":1}}
```

### `mlx_free_token_array`

Releases a token array handle owned by the C++ shim.

```json
{"id":"13","cmd":"mlx_free_token_array","args":{"array":1}}
```

### `native_mock_forward`

Consumes a live native session handle plus a live token array handle and
creates a logits-like MLX array inside the C++ shim. This is only proving the
ownership chain; it does not load model weights or run real inference.

The `tokens` argument is the token array handle as a string.

```json
{"id":"14","cmd":"native_mock_forward","args":{"session":"nsess:1","tokens":"1001"}}
```

### `native_mock_sample`

Reads the logits-like MLX array inside the C++ shim and returns a plain token
id integer only. This is the last step of the fake generation loop shape and
does not run real inference.

```json
{"id":"15","cmd":"native_mock_sample","args":{"logits":"1001"}}
```

### `native_mock_generate`

Wraps the fake generation loop shape into one command: prompt encoding,
token-array creation, mock forward, mock sample, and decode of the sampled
token. It still uses only fixture tokenizer metadata plus shim-owned MLX
arrays.

```json
{"id":"15a","cmd":"native_mock_generate","args":{"session":"nsess:1","tokenizer":"tok:1","prompt":"hello","max_tokens":2}}
```

### `mlx_array_info`

Returns metadata only for a live logits-like MLX array handle:

- `dtype`
- `ndim`
- `shape`
- `size`

```json
{"id":"16","cmd":"mlx_array_info","args":{"array":1001}}
```

### `mlx_free_array`

Releases a logits-like MLX array handle owned by the C++ shim.

```json
{"id":"17","cmd":"mlx_free_array","args":{"array":1001}}
```

### `mlx_handle_counts`

Reports shim-owned native handle counts so the verifier can check that the fake
generation loop frees its temporary token and logits arrays.

```json
{"id":"17a","cmd":"mlx_handle_counts","args":{}}
```

### `load_model_native`

Creates an opaque native-model handle and stores only lifecycle metadata for
now.

```json
{"id":"8","cmd":"load_model_native","args":{"path":"build/model4"}}
```

### `unload_model_native`

Releases a native-model handle.

```json
{"id":"9","cmd":"unload_model_native","args":{"model":"nmodel:1"}}
```

Notes:

- `load_model` remains the existing mock path for compatibility
- `load_model_native` is the new lifecycle scaffold for the eventual real
  native model loader
- `load_tokenizer` is the matching scaffold for tokenizer residency
- `tokenizer_info` reports tokenizer metadata only
- `create_native_session` is the matching scaffold for the eventual real
  tokenizer+model session object
- `free_native_session` releases that opaque session handle

## Native MLX test-array handle commands

These commands prove that the shim can create, store, reference, and free a
tiny MLX object without exposing it to Rust or JavaScript.

Create:

```json
{"id":"7","cmd":"mlx_create_test_array","args":{}}
```

Response:

```json
{"id":"7","ok":true,"value":{"handle":1,"created":true}}
```

Use:

```json
{"id":"8","cmd":"mlx_test_array_sum","args":{"handle":1}}
```

Free:

```json
{"id":"9","cmd":"mlx_free_test_array","args":{"handle":1}}
```

## backend_probe

`backend_probe` is discovery infrastructure for the future ML device daemon
layer.

It returns lightweight structured environment information only. It must:

- inspect local candidate MLX/native paths
- avoid model loading
- avoid large tensor allocation
- remain safe when MLX is absent

Return shape:

```json
{
  "platform": "macos",
  "arch": "aarch64",
  "mlx_detected": true,
  "metal_detected": true,
  "candidate_include_paths": [],
  "candidate_library_paths": [],
  "installation_groups": [],
  "native_boundary": {
    "c_api_detected": false,
    "cpp_api_detected": true,
    "python_package_detected": true,
    "recommended_path": "rust_cpp_shim",
    "preferred_installation_root": null,
    "installation_notes": [],
    "evidence": []
  },
  "notes": []
}
```

`native_boundary` is intended to classify what kind of MLX/native boundary is
actually present, without linking to MLX yet:

- `c_api_detected`
- `cpp_api_detected`
- `python_package_detected`
- `recommended_path`
- `preferred_installation_root`
- `installation_notes`
- `evidence`

`installation_groups` groups discovered headers, libraries, and metallib files
by inferred installation root so cross-install ABI mixing is visible.

## inspect_model_dir

`inspect_model_dir` performs a read-only directory inspection for model
loading preparation. It reports whether the path exists, which tokenizer and
safetensors files are present, whether `config.json` can be parsed, and a
lightweight selection of metadata fields from that config. It does not load
weights or initialize any runtime objects.

## model_load_plan

`model_load_plan` combines lightweight config metadata with safetensors
descriptors to produce a read-only loading plan for a model directory. It does
not load weight payloads. The returned plan includes:

- model type and architecture
- layer count, hidden size, vocab size
- quantized flag
- tensor count and estimated total bytes
- embedding group summary
- per-layer group summary with found and missing groups
- final norm and lm head presence
- likely tied-embedding inference when `lm_head` is absent

## create_model_descriptor / model_descriptor_info / free_model_descriptor

`create_model_descriptor` promotes the read-only `model_load_plan` result into
a persistent opaque descriptor handle. The descriptor stores metadata only:

- model type and architecture
- layer count, hidden size, vocab size
- quantized flag
- tied-embedding flag
- total estimated bytes
- loaded layer count
- loaded weights flag

`model_descriptor_info` returns the same metadata for a live descriptor.

`free_model_descriptor` releases the descriptor handle and marks it freed.

## inspect_tensor_descriptors

`inspect_tensor_descriptors` reads only the safetensors index and shard
headers. It returns per-group metadata such as dtype, shape, byte offsets,
source file, and whether the group looks quantized, without loading tensor
payloads.

## load_tensor_group / tensor_group_info / free_tensor_group

These commands load a single logical tensor group from a safetensors shard
into the C++ shim, expose metadata only, and free the shim-owned group handle
without building the full model or exposing tensor payloads to Rust.

## quantization_layout_probe

`quantization_layout_probe` inspects the already-loaded quantized group layout
and reports dtype/shape metadata, a likely block size, and tiny raw samples
from the stored weight/scales/biases payloads. It does not dequantize the full
tensor and it does not expose tensor payloads outside the shim. The note field
emphasizes that any real dequantization must match the MLX quantized layout
before matmul or generation work can be meaningful.

## load_embedding_group / embedding_group_info / free_embedding_group

These commands load the `model.embed_tokens` group from a safetensors shard
into a native embedding handle, exposing only weight/scales/biases metadata
and freeing the shim-owned record without building the full model.

## dequantize_group_slice

`dequantize_group_slice` is a tiny placeholder dequantization probe. It reads
only a small slice from a loaded quantized tensor group and materializes a
shim-owned `float32` array for the requested `row` and `cols`. The current
implementation is provisional: it unpacks the first 4-bit nibble slice from
the stored U32 word, converts the paired BF16 scale/bias to float32, and uses
`value = unpacked_nibble * scale + bias` as a placeholder until the result is
matched against MLX's real quantized matmul/dequant behavior. It is
intentionally scoped to a tiny slice only and does not attempt full tensor
dequantization.

## compare_dequant_slice

`compare_dequant_slice` runs the same provisional nibble unpack as
`dequantize_group_slice` and, when available, compares it against MLX's own
`mlx::core::dequantize` on a tiny packed block from the same loaded quantized
group. It returns the provisional values either way, plus `mlx_values`,
`max_abs_diff`, and `comparison_available` when the native comparison path
succeeds. If the comparison path is unavailable, the response records which
public MLX dequant entry points and headers were searched. This command is a
probe only; it does not load full tensors or attempt any inference.

## quantized_linear_slice_probe

`quantized_linear_slice_probe` is a tiny arithmetic probe, not full matrix
multiplication. It takes a small numeric input vector, dequantizes the loaded
quantized row slice with the provisional layout probe, and computes a dot
product over that tiny slice. The response returns the input, the dequantized
slice, the scalar dot, and `provisional: true`. This proves that one verified
quantized slice can participate in arithmetic, but full linear work still
requires vector/block traversal or MLX's native quantized linear path.

## load_layer_groups / layer_groups_info / free_layer_groups

These commands compose a single transformer layer from multiple shim-owned
tensor-group handles. The layer wrapper keeps Rust and CoffeeScript at the
opaque-handle level while the native side owns the loaded group records. For
Qwen3 layer 0 the bridge loads the expected attention, MLP, and norm groups,
reports group handles plus byte-size summary data, and frees the child groups
when the layer wrapper is released.

## Intended visibility model

ML operations should become visible artifacts in the DAG, but JavaScript should
only see command envelopes, responses, and opaque handles.

The bridge owns:

- model residency
- tokenizer residency
- session lifetime
- KV cache lifetime
- job lifetime

Handle lifecycle:

- `load_model` creates a model handle and stores it
- `unload_model` removes that handle
- `load_tokenizer` creates a tokenizer handle and stores it
- `unload_tokenizer` removes that handle
- `load_model_native` creates a native-model handle and stores it
- `unload_model_native` removes that handle
- `create_session` requires a currently live model handle
- `free_session` removes the session handle
- `generate` requires a currently live session handle

Shutdown behavior:

- `bridge_shutdown` returns `ok:true`
- once shutdown is accepted:
  - `bridge_health` may still return `ok:true` with `status: "shutting_down"`
  - all other commands fail with `bridge_shutting_down`

## Determinism constraints for scaffolding

- all current responses are stubbed and deterministic
- no actual ML work is performed yet
- shutdown must be explicit and restart-friendly
