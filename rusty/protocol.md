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
{"id":"6","cmd":"generate","args":{"session":"sess:1","prompt":"hello"}}
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
- `shim_probe`
- `mlx_link_probe`
- `mlx_runtime_diagnose`
- `mlx_create_test_array`
- `mlx_test_array_sum`
- `mlx_free_test_array`
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
