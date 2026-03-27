# Runner Invariants

Read this before editing `pipeline_runner.coffee`.

## Critical Contract

Converted pipelines in this repo depend on `StepLedger`, explicit `artifacts`, and `needs` / `makes`.

Do not edit `pipeline_runner.coffee` unless these invariants still hold afterward.

## Required Invariants

1. New-style CoffeeScript `@step` files must be invoked as:

```coffee
step.action(L, n, M)
```

Where:

- `L` is `StepLedger`
- `n` is the step name
- `M` is raw `Memo`

2. `runStep(...)` must construct:

```coffee
L = createStepLedger(M, n, resolveArtifact, artifactSpecFor)
```

3. The scheduler must pass `resolveArtifact` and `artifactSpecFor` into `runStep(...)`.

4. After meta initialization, artifact source/target persistence and recovery must go through `Memo` only.

Do not add direct runner-side artifact file reads/writes.

5. Opaque MLX filesystem locations stay as params, not artifacts:

- model directories
- adapter directories
- safetensor files

6. Converted recipes must not drift back to old-style path-handle params for pipeline-owned data.

## Quick Regression Check

After editing `pipeline_runner.coffee`, verify all of these:

- `createStepLedger(...)` exists
- `runStep(...)` creates `L`
- new-style CoffeeScript path calls `step.action(L, n, M)`
- artifact resolver uses `M.theLowdown(...)`, not direct file reads
- artifact materialization uses `M.saveThis(target, value)`, not direct file writes
- scheduler passes `resolveArtifact` and `artifactSpecFor` into `runStep(...)`

## Ground Truth

If behavior and documentation disagree, trust the actual file and re-audit it end to end before making more edits.
