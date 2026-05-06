# node-mlx Interface Notes

Scope:

- inspection only
- interface-map use only
- no protocol change
- no production wiring

`node-mlx` is useful as a map of how MLX concepts were projected into a host
runtime. It is not the runtime model to copy into `rusty`.

## Useful TypeScript-facing operation groups

From `../development/node-mlx/lib/`:

- core tensor/runtime surface
  - `core.ts`
  - `index.ts`
- neural-network surface
  - `nn/index.ts`
  - `nn/init.ts`
  - `nn/layers/*`
  - `nn/losses.ts`
  - `nn/utils.ts`
- optimizer surface
  - `optimizers/index.ts`
  - `optimizers/optimizers.ts`
  - `optimizers/schedulers.ts`
- host-side utility layer
  - `utils.ts`

Why this is useful:

- it suggests natural operation families for future Rust command grouping
- it shows how a host-facing surface can be kept modular instead of becoming one
  giant binding blob
- it suggests naming for future Rust internal modules even if the external
  bridge protocol stays narrower

What to borrow from the TypeScript side:

- high-level operation grouping
- naming consistency
- argument-shape discipline
- separation between core math/runtime and higher-level NN helpers

## Useful native wrapper boundaries

From `../development/node-mlx/src/`:

- array ownership/wrapping
  - `array.cc`
  - `array.h`
- bindings entrypoint
  - `bindings.cc`
  - `bindings.h`
- device / stream boundaries
  - `device.cc`
  - `device.h`
  - `stream.cc`
  - `stream.h`
- math/runtime families
  - `ops.cc`
  - `random.cc`
  - `fft.cc`
  - `linalg.cc`
  - `transforms.cc`
  - `fast.cc`
  - `indexing.cc`
- I/O and memory
  - `io.cc`
  - `memory.cc`
- platform-specific surface
  - `metal.cc`
- structural helpers
  - `trees.cc`
  - `utils.cc`
  - `constants.cc`
  - `complex.cc`

Why this is useful:

- it is a practical decomposition of the MLX native surface
- it suggests where a future C++ shim can stay narrow
- it shows which concerns are likely to deserve separate native modules instead
  of one giant `bridge.cc`

What to borrow from the native wrapper side:

- file/module partitioning
- separation of device/stream concerns from math ops
- keeping array/storage concerns separate from higher-level command dispatch
- explicit I/O and memory wrapper boundaries

## Useful build/link clues

From `../development/node-mlx/CMakeLists.txt` and `package.json`:

- C++17 is enough for the binding layer
- the project links directly against MLX as a native library target
- `add_subdirectory(deps/mlx)` is a working local-development pattern
- the host binding is built as a shared native addon
- MLX can be vendored and compiled inside a larger build

Useful clues for `rusty`:

- a small native shim can likely be built by CMake and linked against MLX
- vendored-local MLX builds are practical for development, even if production
  later links against installed MLX artifacts
- a split build is reasonable:
  - Rust owns process/protocol
  - C++ owns the narrow MLX shim

## What must NOT be copied

These parts of `node-mlx` conflict directly with `rusty`'s ownership rules:

- JS-visible tensor ownership
- JS object wrappers around MLX arrays
- host-runtime lifetime management of native ML objects
- GC-adjacent native allocation pressure

Why not:

- `rusty` is explicitly trying to keep JS/CoffeeScript out of tensor ownership
- the resident bridge should own ML lifetime, not the host runtime
- exposing native arrays into JS would recreate the exact GC/unified-memory
  interaction we are trying to avoid

So the correct use of `node-mlx` is:

- study its interface grouping
- study its wrapper decomposition
- study its build/link assumptions

And the incorrect use is:

- copying its JS object model
- copying its host-owned native lifetime model
- projecting MLX arrays/tensors into JS-visible objects
