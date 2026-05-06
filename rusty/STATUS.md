# Rusty Status

Date: 2026-05-06

Branch: `rusty`

## Current State

- The Rust resident bridge scaffold exists under `rusty/bridge/`.
- The bridge has a JSONL protocol, handle tables, shutdown-state validation, and backend discovery commands.
- The native C++ shim exists and Rust can call into it.
- The shim can link to MLX successfully.
- A small native smoke harness exists under `rusty/native_smoke/`.
- The smoke harness compares several compiler/runtime variants against the selected Homebrew MLX root.

## What We Learned

- Plain C++ MLX array construction fails on this machine even when built directly against the coherent Homebrew MLX install.
- The failure is reproducible in the Rust bridge shim, in standalone C++ smoke tests, and in the official repo `.venv` Python interpreter.
- The failure happens at array construction and throws an `NSRangeException` / `unknown exception` path inside `libmlx.dylib`.
- Changing `-std=c++17`, `-std=c++20`, or adding `-stdlib=libc++` does not change the failure.
- Switching between `/opt/homebrew/Cellar/mlx/0.30.0` and `/opt/homebrew/opt/mlx` does not change the failure.
- The local `../development/mlx` tree has headers and `MLXConfig.cmake`, but no built `libmlx.dylib` in the obvious `build/` path, so it is not a usable runtime target yet.

## Conclusion

- The current blocker is MLX runtime behavior on this machine, not Rust bridge linkage.
- The bridge architecture remains sound: JS orchestration, Rust process ownership, C++ shim for MLX boundary.

## Recommended Next Step

- Decide whether to keep diagnosing the local MLX runtime path, or move to a newer MLX build/runtime combination and repeat the smoke matrix.

## Prompt For ChatGPT

Use this prompt if you want a clean continuation:

```text
I have a Rust resident bridge scaffold under rusty/ and the native MLX diagnosis is complete for now.

Current facts:
- Rust can build and call a tiny C++ shim.
- The shim can link MLX successfully.
- backend_probe, shim_probe, mlx_link_probe, and mlx_runtime_diagnose are all in place.
- The native smoke test under rusty/native_smoke compares C++17, C++20, libc++, Homebrew cellar root, Homebrew opt symlink, Python venv, and local-repo variants.
- Every tiny MLX array construction path fails on this machine with the same NSRangeException / unknown exception inside libmlx.dylib.
- The failure reproduces in:
  - Rust bridge shim
  - standalone C++ smoke test
  - the repo .venv Python interpreter
- So the blocker is MLX runtime behavior on this machine, not Rust linkage.

What I want next:
- either keep diagnosing the local MLX runtime path
- or pivot to a newer MLX version/build and rerun the smoke matrix
- do not change the bridge architecture unless the evidence supports it
```
