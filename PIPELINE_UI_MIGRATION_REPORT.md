# Pipeline/UI Migration Report

## Scope

This report summarizes the current runner, StepLedger, Memo/meta, SQLite, and run-loop structure before any UI implementation.

It is based on the current code in:

- `pipeline_runner.coffee`
- `meta/index.coffee`
- `meta/json.coffee`
- `meta/jsonl.coffee`
- `meta/csv.coffee`
- `meta/txt.coffee`
- `meta/yaml.coffee`
- `meta/slash.coffee`
- `meta/sqlite.coffee`
- `run_between_hours.sh`
- current SQLite-oriented steps under `scripts/kag_oracle_ite/`

## Authoritative Files

### Pipeline execution

- `pipeline_runner.coffee`
  - `main()`
  - `discoverSteps(...)`
  - `toposort(...)`
  - `runStep(...)`
  - `resolveArtifact(...)`
  - `materializeArtifact(...)`
  - `wireInputsForStep(...)`
  - `collectOutputsForStep(...)`

### Step API and scheduling contract

- `pipeline_runner.coffee`
  - `createStepLedger(...)`
  - `StepLedger.param(...)`
  - `StepLedger.need(...)`
  - `StepLedger.peek(...)`
  - `StepLedger.make(...)`
  - `StepLedger.done(...)`
  - `StepLedger.fail(...)`
  - `StepLedger.callMLX(...)`

- `RUNNER_INVARIANTS.md`
  - current runner/ledger contract

### Step state / run state

- `pipeline_runner.coffee`
  - `class StepStateStore`
  - `markRunning(...)`
  - `markDone(...)`
  - `markFailed(...)`
  - `clearRestartHere(...)`
  - `writePipelineShutdown(...)`
  - `readPipeline(...)`

- Runtime files
  - `state/step-<name>.json`
  - `pipeline.json`

### Artifact persistence

- `pipeline_runner.coffee`
  - `class Memo`
  - `Memo.saveThis(...)`
  - `Memo.theLowdown(...)`
  - `Memo.waitFor(...)`
  - `Memo.addMetaRule(...)`
  - `Memo.selectMetaHandler(...)`

- `meta/index.coffee`
  - meta device load order
  - `sqlite.coffee` is intentionally loaded first

- File-backed meta devices
  - `meta/json.coffee`
  - `meta/jsonl.coffee`
  - `meta/csv.coffee`
  - `meta/txt.coffee`
  - `meta/yaml.coffee`
  - `meta/slash.coffee`

### SQLite schema and access

- `meta/sqlite.coffee`
  - opens DB with `DatabaseSync` from `node:sqlite`
  - default DB file: `runtime.sqlite`
  - request registry: `REQUESTS`
  - format registry: `FORMATTERS`
  - single meta rule: `M.addMetaRule "sqlite", /\.(json|jsonl|txt|csv)$/i, ...`

- Current tables created in `meta/sqlite.coffee`
  - `stories`
  - `story_parts`
  - `kag_entries`

- Current request keys implemented in `meta/sqlite.coffee`
  - `storyByID{storyID}.json|txt|csv`
  - `partsFor{storyID}.json|txt|csv`
  - `kagFor{storyID}.json|txt|csv`
  - `storiesWithKag{keyword}.jsonl|txt|csv`
  - `storiesMissingKag.jsonl|txt|csv`

### Existing UI / daemon scaffolding

There is no current frontend app, HTTP server, websocket server, SSE server, or daemon process in the repo.

What exists:

- `pipeline_runner.coffee`
  - an in-file UI integration design comment block
  - no implementation yet

- `run_between_hours.sh`
  - cron-style launcher / gatekeeper
  - time-window loop
  - one-run-at-a-time shell orchestration
  - log file creation under `logs/`

## Current Data Flow

1. `main()` loads `override.yaml`, resolves the selected recipe in `config/`, and writes:
   - `experiment.yaml`
   - `params/_global.yaml`
   - `params/<step>.yaml`

2. Runner discovers DAG steps from recipe entries with `run`.

3. Runner restores prior step completion/failure only from `state/step-<name>.json`.

4. For each scheduled step:
   - `wireInputsForStep(...)` resolves declared `needs`
   - resolved values are written into Memo with `M.saveThis(artifactKey, value)`
   - `runStep(...)` creates `L = createStepLedger(...)`
   - new-style CoffeeScript step runs as `step.action(L, n, M)`

5. Inside steps:
   - `L.need(...)` reads declared inputs from Memo
   - `L.peek(...)` checks optional values
   - `L.make(...)` writes declared outputs into Memo
   - direct `L.saveThis(...)` / `M.saveThis(...)` can also write arbitrary keys

6. After a step finishes:
   - `collectOutputsForStep(...)` reads each declared `makes`
   - `materializeArtifact(...)` writes:
     - artifact logical key
     - artifact target path if present
   - persistence is delegated to the meta device selected by key

7. State persistence is separate from artifact persistence:
   - `StepStateStore` writes `running` / `done` / `failed` files in `state/`
   - shutdown writes `pipeline.json`

8. SQLite writes happen only when a key matches the SQLite request-key patterns in `meta/sqlite.coffee`.
   - ordinary file-path keys like `out/foo.json` do not go to SQLite

## Current Waiting Representation

### What exists now

Waiting exists only as runtime behavior, not as persisted UI state.

Current waiting paths:

- Dependency waiting
  - `Memo.waitFor(keys, andDo)`
  - used by scheduler for `done:<dep>` keys before starting a step

- Artifact waiting
  - `StepLedger.need(...)`
  - if artifact value is undefined, it awaits `entry.notifier`

- Artifact resolution waiting
  - `resolveArtifact(...)`
  - can await source or produced artifact notifier

### What is recorded now

- step state files record:
  - `status`
  - `done`
  - timestamps
  - error
- no persisted field records:
  - which key a step is waiting on
  - whether the step is blocked on dependency vs artifact
  - how long it has been waiting

### Conclusion

Step waiting is not currently represented in a durable, queryable form.

## How A Front-End Could Observe In Real Time Today

### Active run

Current options:

- Poll `state/step-*.json` for steps with `status: "running"`
- Poll process existence externally
- Send `SIGUSR1` to the runner and parse stdout listing of `active.names`

Current authoritative runtime source:

- in-memory `active.names` inside `pipeline_runner.coffee`

Limitation:

- `active.names` is not persisted anywhere

### Step status

Current source:

- `state/step-<name>.json`

Status values currently written:

- `running`
- `done`
- `failed`

Supplemental run-wide source:

- `pipeline.json` for shutdown reason/by

### Waiting-on key / artifact

Not observable in durable form today.

The only current hook points are code paths:

- `StepLedger.need(...)`
- `StepLedger.peek(...)`
- `Memo.waitFor(...)`

### Recent events

Current practical sources:

- runner stdout/stderr
- `logs/pipe_*.log`
- `logs/pipe_*.err`

There is no append-only structured event store yet.

### Cooldown state

There is no runner-native cooldown model.

The only cooldown-like behavior currently lives in `run_between_hours.sh`:

- time-window gating via `is_within_window()`
- 60-second sleeps outside the window
- 60-second sleep between loop iterations
- minimum elapsed runtime guard

This state is not persisted in JSON, Memo, SQLite, or step state.

## Gaps For A Live Status UI

1. No structured event stream exists yet.
2. No persisted representation of "waiting on X".
3. No persisted representation of active in-memory step set.
4. No unified "current run snapshot" document exists.
5. No machine-readable cooldown state exists.
6. No dedicated UI transport exists.
7. `Memo.saveThis(...)` currently swallows meta exceptions with `catch then null`, which can hide persistence failures from a monitor unless explicit event logging is added.

## Recommended Minimal Integration Points

These fit the current structure and the UI notes already present in `pipeline_runner.coffee`.

### 1. Instrument StepLedger only

Add observational writes at these existing hook points:

- `StepLedger.param(...)`
- `StepLedger.need(...)`
- `StepLedger.peek(...)`
- `StepLedger.make(...)`
- `StepLedger.done(...)`
- `StepLedger.fail(...)`

Reason:

- these are already the explicit contract boundary between steps and artifacts
- step scripts do not need UI code

### 2. Persist UI state through Memo/meta, not ad hoc files in runner

Use Memo keys like the existing comment block already proposes:

- `ui/steps/<step>/needs/<artifact>`
- `ui/steps/<step>/peeks/<artifact>`
- `ui/steps/<step>/makes/<artifact>`
- `ui/events/<id>`

Reason:

- conforms to current Memo-first architecture
- keeps runner as source of truth
- allows file-backed or SQLite-backed storage through meta

### 3. Add one live run snapshot key

Recommended minimal key:

- `ui/run/current.json`

Fields should be derived from existing runner state:

- selected pipeline
- run start time
- active steps
- final steps
- shutdown status

Reason:

- minimizes frontend joins for the initial monitor

### 4. Add one persisted waiting field at the actual wait sites

Minimal hook points:

- just before `await entry.notifier` in `StepLedger.need(...)`
- just before `Memo.waitFor(...)` scheduler dependency wait

Record:

- step
- wait type: `artifact` or `dependency`
- key(s)
- observed timestamp

### 5. Keep SQLite optional and observational

Do not redesign around DB-first UI state.

Use the existing SQLite request-key structure where useful, but keep the first monitor aligned with current StepLedger + Memo structure. The runner comment already points to Memo keys as the stable integration surface.

## Recommended First Minimal Real-Time Monitor UI

### Goal

A read-only live status page with:

- current pipeline/run
- step list with status
- currently active steps
- waiting-on artifact or dependency if known
- recent event list
- shutdown reason if present

### Minimal backend changes

1. Emit `ui/run/current.json` from runner.
2. Emit `ui/events/<id>.json` append-only from StepLedger hooks.
3. Emit `ui/steps/<step>.json` or the already proposed per-need/per-make keys from StepLedger hooks.
4. Emit wait-state updates at:
   - `StepLedger.need(...)`
   - scheduler dependency wait site

### Minimal frontend/observer strategy

No new daemon required for first cut.

Start with a simple poller over persisted JSON state:

- poll `ui/run/current.json`
- poll recent `ui/events/*`
- poll `state/step-*.json`
- optionally read `pipeline.json`

This is the smallest conforming path because it:

- reuses current runner and Memo structure
- does not require a new transport layer
- does not redesign scheduling
- stays compatible with the current StepLedger + SQLite direction

