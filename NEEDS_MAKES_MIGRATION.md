# Needs/Makes Migration Guide

This file explains how to convert an older pipeline in this architecture from ad hoc param-path handles to the current explicit `artifacts` + `needs` / `makes` + `StepLedger` model.

The target audience is another Codex or engineer working in a different directory that still uses the older style.

## Goal

Move from this older pattern:

- step params used as both behavior knobs and file handles
- scripts reading and writing arbitrary memo keys or disk paths directly
- no explicit artifact contract in config

to this newer pattern:

- pipeline-owned data declared in `artifacts:`
- every step explicitly declares `needs: []` and `makes: []`
- scripts use `StepLedger`
- data artifacts persist and recover through `Memo` and meta devices
- model / adapter / safetensor directories remain opaque params

## Core Rule

After meta devices are initialized, artifact persistence and recovery must go through `Memo`.

Do not add new direct filesystem read/write code in the runner for artifact data.

The only normal exception is opaque MLX filesystem inputs such as:

- model directories
- adapter directories
- safetensor files

Those are passed through as params to MLX calls and are not inspected as pipeline artifacts.

## Old Style vs New Style

Old style config:

```yaml
md2segments:
  run: train_markdown/md2segments.coffee
  depends_on: [download_model]
  needs: []
  makes: []
  stories_md: data/jim.md
  marshalled_stories: data/marshalled_stories.jsonl
  split_mode: paragraph
```

New style config:

```yaml
artifacts:
  stories_md:
    source: data/jim.md
  marshalled_stories:
    target: data/marshalled_stories.jsonl

md2segments:
  run: train_markdown/md2segments.coffee
  depends_on: [download_model]
  needs: [stories_md]
  makes: [marshalled_stories]
  split_mode: paragraph
```

Old style script:

```coffee
action: (M, stepName) ->
  storiesKey = M.getStepParam stepName, 'stories_md'
  outputKey = M.getStepParam stepName, 'marshalled_stories'
  input = M.theLowdown(storiesKey)?.value
  ...
  M.saveThis outputKey, rows
  M.saveThis "done:#{stepName}", true
```

New style script:

```coffee
action: (S) ->
  splitMode = S.param 'split_mode'
  raw = await S.need 'stories_md'
  ...
  S.make 'marshalled_stories', rows
  S.done()
```

## What Belongs In `artifacts`

Declare pipeline-owned data products in `artifacts:`.

Examples:

- source markdown
- marshalled story rows
- selected story ids
- oracle raw replies
- normalized oracle outputs
- reject piles
- prompt cache records
- train / valid / test rows
- generated story text

Typical declarations:

```yaml
artifacts:
  stories_md:
    source: data/jim.md
  marshalled_stories:
    target: data/marshalled_stories.jsonl
  new_story_ids:
    target: out/new_story_ids.json
  kag_rejects:
    target: out/kag_rejects.jsonl
```

## What Should Stay As Params

Keep opaque MLX paths and behavior knobs as params.

Examples:

- `model_dir`
- `loraLand`
- `adapter_path`
- `resume_adapter_file`
- `quantized_model_dir`
- `batch_size`
- `max_tokens`
- `temperature`
- `top_p`
- `top_k`
- `prompt_text`
- `debug_s`
- `debug_mlx`

Rule of thumb:

- if the pipeline semantically owns the data, use an artifact
- if the step is just passing a location or option through to MLX, keep it as a param

## StepLedger Contract

The runner provides a per-step `StepLedger`.

Preferred step form:

```coffee
@step =
  action: (S) ->
    input = await S.need 'some_input'
    mode = S.param 'mode'
    S.make 'some_output', transform(input, mode)
    S.done()
```

Key methods:

- `S.param(key)`
  - required param
  - throws if missing
- `S.param(key, defaultValue)`
  - optional param with explicit default
- `await S.need(artifactKey)`
  - required declared artifact
- `await S.peek(artifactKey, defaultValue)`
  - optional declared artifact probe
- `S.make(artifactKey, value)`
  - write declared output artifact
- `S.done()`
  - mark step done
- `S.fail(err)`
  - mark step failed and throw
- `S.theLowdown(key)`
  - direct memo access for special cases
- `S.callMLX(type, args)`
  - MLX invocation, using `debug_mlx` param by default

## Important Usage Rules

1. Every step must declare `needs: []` and `makes: []`, even if empty.

2. `S.need(...)` only reads artifacts declared in that step’s `needs` or `makes`.

3. `S.make(...)` only writes artifacts declared in that step’s `makes`.

4. If a step only wants to know whether a previously produced output exists, a direct memo check is acceptable:

```coffee
existing = S.theLowdown('marshalled_stories')?.value
if existing isnt undefined
  console.log "[md2segments] output already exists, skipping"
  S.done()
  return
```

5. Do not create new runner-side file parsers for artifacts. If a new file type is needed, add support in a meta device.

## Required Runner Behavior

The runner side needs these properties:

- merged config saved as `experiment.yaml`
- explicit `artifacts`
- explicit `needs` / `makes` validation
- single producer per artifact
- `StepLedger`
- startup restore from step state
- artifact resolution through `Memo`
- artifact materialization through `Memo.saveThis(target, value)`

The critical flow is:

1. resolve an input artifact from its `source`, `target`, literal `value`, or live produced artifact
2. save the resolved value into the artifact key in memo
3. run the step
4. materialize declared outputs to their artifact key and `target`

## Recommended Migration Order

Convert one recipe at a time.

For each recipe:

1. Add or normalize `artifacts:`
2. Make every step declare `needs: []` and `makes: []`
3. Move path-like data handles out of arbitrary step params and into artifacts
4. Keep MLX directories and knobs as params
5. Convert scripts to `StepLedger`
6. Compile each changed CoffeeScript file
7. Run the pipeline and inspect `experiment.yaml` and logs

Best first conversions:

- source markdown
- marshalled rows
- story id selections
- oracle raw output
- normalized oracle output
- training row datasets
- final generated story text

## Common Conversion Patterns

### Pattern 1: Source file to artifact

Before:

```yaml
stories_md: data/jim.md
```

After:

```yaml
artifacts:
  stories_md:
    source: data/jim.md
```

### Pattern 2: Output path to artifact

Before:

```yaml
new_story_ids: out/new_story_ids.json
```

After:

```yaml
artifacts:
  new_story_ids:
    target: out/new_story_ids.json
```

### Pattern 3: Raw output then normalization

Do not force a weak model output directly into the final pipeline shape.

Instead:

1. store raw output
2. decode / normalize downstream
3. produce a reject pile

Example:

```yaml
oracle_ask:
  needs: [marshalled_stories]
  makes: [new_story_ids, kag_oracle_raw, kag_viewed]

decode_kag_oracle:
  depends_on: [oracle_ask]
  needs: [kag_oracle_raw]
  makes: [kag_emotions, kag_rejects]
```

This makes debugging much easier and preserves determinism.

## Debugging

`StepLedger` debug should be config-driven, not hardcoded.

Recommended global or step params:

```yaml
run:
  debug_s: true
  debug_mlx: false
```

`debug_s` should log:

- param request / resolved / default / missing
- need request / waiting / resolved / missing
- make request / wrote
- done / fail
- source / target binding info where available
- timestamps

This is also the right boundary for future UI state updates.

## A Frequent YAML Failure Mode

Be careful with YAML spacing in overrides.

This is wrong:

```yaml
record_trained_batch:
  debug_s:true
```

That parses as a scalar string and overwrites the whole step object.

This is correct:

```yaml
record_trained_batch:
  debug_s: true
```

When a run looks wrong, check `experiment.yaml` first. It is the authoritative merged config.

## Validation Checklist

For each migrated recipe:

1. `experiment.yaml` contains the expected step objects
2. every step has `needs` and `makes`
3. artifact declarations are present
4. no artifact is produced by multiple steps unless renamed
5. changed CoffeeScript files compile cleanly
6. outputs materialize through memo/meta
7. rerunning the pipeline reuses persisted artifacts where appropriate

## Minimal Example

Config:

```yaml
run:
  debug_s: true
  debug_mlx: false

artifacts:
  input_text:
    source: data/input.txt
  transformed_rows:
    target: out/transformed.jsonl

transform_step:
  run: my_pipeline/transform_step.coffee
  depends_on: []
  needs: [input_text]
  makes: [transformed_rows]
  mode: simple
```

Script:

```coffee
@step =
  action: (S) ->
    raw = await S.need 'input_text'
    mode = S.param 'mode'

    rows = transform raw, mode

    S.make 'transformed_rows', rows
    S.done()
```

## Final Guidance

Do not try to convert everything at once.

Make one recipe fully compliant, run it, inspect `experiment.yaml`, inspect logs, then move to the next recipe.

That approach worked well for:

- `kag_oracle`
- `train_lora`

and should be repeated elsewhere.
