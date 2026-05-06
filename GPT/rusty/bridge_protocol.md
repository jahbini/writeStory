Area: `rusty/protocol.md` and `rusty/bridge/src/*.rs`

Purpose:
- define the initial resident-bridge command envelope

Transport:
- resident process
- stdin/stdout
- one JSON object per line
- later upgrade path: Unix socket

Request shape:
- `id`
- `cmd`
- `args`

Response shape:
- success:
  - `id`
  - `ok: true`
  - `value`
- failure:
  - `id`
  - `ok: false`
  - `error.code`
  - `error.message`

Initial commands:
- `bridge_health`
- `bridge_shutdown`
- `load_model`
- `unload_model`
- `create_session`
- `free_session`
- `generate`

Current Rust scaffold behavior:
- deterministic stub responses only
- handle tables exist for:
  - models
  - tokenizers
  - sessions
  - kv caches
  - jobs
- `generate` returns a stubbed text payload and records a completed job

Known pitfalls:
- do not let JS infer ML ownership from handle naming
- do not collapse bridge residency back into spawn-per-request subprocess calls
- do not bypass DAG visibility for ML requests
