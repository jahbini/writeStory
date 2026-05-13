# Build Guide

This file documents how to build the native gypsy generation path (the
`metal_llm.node` Node addon that powers `generate_prompt_gypsy_ite` and the
internal probes). The CoffeeScript pipeline itself needs no build — only the
C++/Metal native addon does.

If you only want to **use** writeStory and a prebuilt `metal/metal_llm.node`
is already in your tree, you can skip straight to "Verify" below.

---

## TL;DR

```bash
# Once per machine: install MLX via Homebrew
brew install mlx

# Every time native sources change (or on a fresh clone):
npx node-gyp rebuild

# Verify everything works end-to-end:
bash test.sh
```

`npx node-gyp rebuild` takes ~20–25 s. `test.sh` re-runs the rebuild itself,
so on a fresh clone with Homebrew MLX installed you can go directly to
`bash test.sh`.

---

## What gets built

| Artifact | Built by | Location | Tracked? |
|---|---|---|---|
| node-gyp Makefiles + intermediate `.o` | `npx node-gyp rebuild` | `build/` | no |
| the native addon | `npx node-gyp rebuild` | `metal/metal_llm.node` | no |
| C++/Metal sources | (hand-written) | `metal/*.cpp`, `metal/*.mm`, `metal/*.h` | yes |
| build recipe | (hand-written) | `binding.gyp` | yes |

`build/` (node-gyp output) and `metal/metal_llm.node` (the linked addon) are
both gitignored. They are platform-specific and trivially regenerable from
the tracked sources.

---

## Prerequisites

### macOS toolchain

- **Xcode** installed at `/Applications/Xcode.app` (not just the CLT). The
  Metal compiler toolchain is part of the full Xcode install, not the CLT.
- **Apple Silicon** (M1/M2/M3/M4). The build is hard-wired to arm64 + Metal
  + Accelerate. Intel macOS is untested.
- The `.local_bin/metal` and `.local_bin/xcrun` shims in this repo work
  around a current macOS bug where the Xcode `xcrun metal` stub fails to
  delegate to the cryptex toolchain. They are only needed if you want to
  compile Metal kernel source yourself (e.g. for studying MLX internals via
  the optional `mlx/` checkout — see below). The production addon build does
  not invoke them.

### MLX

- Install via Homebrew: `brew install mlx`
- The addon uses several APIs that are only present in recent MLX (notably
  `mlx::core::slice_update`, `mlx::core::set_cache_limit`, the
  `fast::metal_kernel` host API, and the vector-2-pass SDPA path). If
  Homebrew's bottle is behind, `brew upgrade mlx` to the current stable.
- `binding.gyp` links against `-L/opt/homebrew/lib -lmlx` with headers from
  `/opt/homebrew/include`. If you install MLX somewhere else, edit
  `binding.gyp`'s `include_dirs` and `libraries` accordingly.

### Node + package manager

- **Node.js** (tested on v24.4.0)
- **pnpm** for `node_modules` (the repo ships a `pnpm-lock.yaml`-style
  layout). `npm install` also works for just `node-addon-api`.
- **node-gyp** (invoked via `npx`, no global install required)

### Disk

- ~33 MB for `build/` (node-gyp output)
- ~5 MB for `metal/metal_llm.node`

---

## Build steps

### 1. Install MLX

```bash
brew install mlx
# or, if already installed but stale:
brew upgrade mlx
```

### 2. Build the native addon

```bash
npx node-gyp rebuild
```

What it does:

- reads `binding.gyp`
- compiles `metal/metal_llm.cpp`, `metal/metal_llm_node.cpp`, and
  `metal/direct_metal_probe.mm` (C++17, exceptions enabled,
  `MACOSX_DEPLOYMENT_TARGET=15.0`)
- links against Homebrew MLX (`-L/opt/homebrew/lib -lmlx`) plus
  `-framework Metal -framework Foundation -framework Accelerate`
- produces `build/Release/metal_llm.node`
- copies it to `metal/metal_llm.node` (per `"product_dir": "../metal"` in
  binding.gyp)

Takes ~20–25 s on a warm cache.

### 3. Verify

```bash
bash test.sh
```

This rebuilds the addon (idempotent) and runs the 64-token lazy generation
probe at `test/helpers/native_64_mlx_lazy_generation_probe.coffee`. A
successful run produces:

- stdout: a JSON record including `tokens_per_second` (expect ~9–10 on M2)
  and a coherent decode of "July is the seventh month..."
- stderr: `[mlx-diag] metal_available=YES default_device=GPU
  backend=mlx_prealloc_kv` plus real/user/sys wall-clock times
- logs: timestamped under `test/logs/mlx_lazy_<YYYYMMDD_HHMMSS>/`

If the speed is dramatically lower (< 1 tok/s) or memory is dramatically
higher (> 5 GB), see `gypsy/STATUS.md` and `GPT/gypsy_strategy.md` —
something has likely regressed.

---

## Optional: local MLX checkout for reading source

The repo's `.gitignore` excludes `mlx/`. You may keep a local checkout of
the MLX source tree there if you want to read its internals (kernel
selection logic, RoPE math, allocator behavior, etc.). The strategy
document `GPT/gypsy_strategy.md` was written by reading
`mlx/backend/metal/scaled_dot_product_attention.cpp`,
`mlx/backend/metal/sdpa_vector.h`, `mlx/fast.cpp`, and
`mlx/backend/metal/allocator.cpp`.

If `mlx/build/libmlx.a` exists, `binding.gyp` will prefer it for linking
(static link against the local source). The companion script
`build_mlx.sh` produces that static library — but it is **a dev aid, not
part of the production build**. Use it only if you need to apply local
patches to MLX while developing gypsy.

The production build path is exclusively Homebrew MLX. The `mlx/`
directory is never committed and can be deleted at any time without
affecting the production addon build.

---

## Common failures

**`Error: Cannot find module '...metal_llm.node'`**
The addon didn't get built or wasn't copied to `metal/`. Re-run
`npx node-gyp rebuild`.

**`Error: libmlx.dylib (no such file)` at runtime**
Homebrew MLX isn't installed, or isn't on the dyld path. Run
`brew install mlx` (or fix `DYLD_LIBRARY_PATH`).

**`ld: symbol not found for architecture arm64 ... mlx::core::slice_update`
(or `set_cache_limit`, or other recent MLX names)**
Homebrew MLX is too old. `brew upgrade mlx`.

**`node-gyp` python errors**
node-gyp uses Python internally for its build scripts. If your `python3`
isn't on PATH or is too new/old, `npx node-gyp rebuild` will fail before
even invoking the C++ compiler. Install a modern python3 and ensure
`python3 --version` works in your shell.

---

## Files referenced

- [BUILD.md](BUILD.md) — this file
- [binding.gyp](binding.gyp) — node-gyp recipe (links against Homebrew MLX
  by default; prefers a local static `mlx/build/libmlx.a` if present)
- [test.sh](test.sh) — rebuild + run the 64-token probe
- [build_mlx.sh](build_mlx.sh) — **dev aid only**: build a local MLX
  static library when developing against an unreleased MLX or applying
  local patches. Not part of the production build.
- [.local_bin/metal](.local_bin/metal), [.local_bin/xcrun](.local_bin/xcrun)
  — shims around the cryptex Metal compiler, used by `build_mlx.sh`. Not
  needed for the production addon build.
- [gypsy/STATUS.md](gypsy/STATUS.md) — what the working path looks like and
  the measured envelope
- [GPT/gypsy_strategy.md](GPT/gypsy_strategy.md) — standards / anti-standards
  derived from the speed-and-memory work
