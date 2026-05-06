Area: `rusty/meta/mlx_bridge.coffee`

Purpose:
- provide the future Memo-facing bridge wrapper for resident ML commands

Intended Memo contract:
- match keys under `mlx/`
- use normal Memo save/read semantics only
- do not add a new Memo API
- preserve notifier behavior by resolving existing Memo entries when bridge replies arrive

Current behavior:
- scaffold only
- lazy-starts a resident child process on first request
- uses JSONL request/response protocol over stdin/stdout
- logs bridge traffic to `state/rusty-bridge.jsonl`
- caches last response by Memo key

Request model:
- key shape is expected to look like:
  - `mlx/bridge_health.json`
  - `mlx/load_model.json`
  - `mlx/create_session.json`
  - `mlx/generate.json`
- command name is derived from the key basename
- args come from the Memo write value

Known limitations:
- currently points at `cargo run --manifest-path rusty/bridge/Cargo.toml`
- not installed in top-level `meta/index.coffee`
- no actual ML ownership yet; Rust side is stubbed
- current completion path uses direct entry resolver access, which is acceptable for scaffold notes but should be revisited carefully during real integration

Important design rule:
- ML operations must become visible DAG artifacts
- JS still only sees opaque handles and structured responses
