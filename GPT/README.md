This directory is assistant-owned working memory for recipe and step contracts.

Purpose:
- keep step-local memory outside the transient conversation
- record proven contracts and costly failure modes
- help trace downstream failures back to upstream causes
- preserve current pipe/workspace assumptions so future Codex work does not drift back to older top-level-only behavior

Rules:
- keep files short and factual
- update a step memory when its contract changes
- prefer one file per important step
- record inputs, outputs, invariants, and pitfalls
- if a stale machine bug is diagnosed from logs, update the affected step memory so the failure mode is explicit
- do not use this directory for general notes or speculation
- universal verifier commandment: do not repeatedly execute a proven expensive
  baseline in smoke/default verification. Record the baseline timing/result as
  metadata and re-run the old path only under an explicit full/comparison
  profile or when its math contract changed.
- repository-level runtime tests are coordinated through root `test.sh`.
  Codex owns that file's contents and should replace it as needed for the
  current task, record every test's `.log` and `.err` output explicitly, and
  tell the human to run `./test.sh`. Put temporary helper scripts under
  `test/`. If Python is needed, `test.sh` must activate `.venv` first with
  `source .venv/bin/activate`.

Current repository assumptions worth preserving:
- the repo is pipe-centric; active workspaces live under `pipes/<organization>_<model>/`
- `override.yaml` is foundational and must carry the base `run.model`
- `control_override.yaml` is UI-owned run control, not a replacement for `override.yaml`
- new empty pipes infer their model identity from the pipe directory name
- `base_ite` now owns base preparation through quantization, so downstream inference recipes consume prepared artifacts instead of rebuilding them
- the UI drives recipe fields through recipe-declared directives, currently `UI_dropdown`, `UI_checkbox`, and `UI_textarea`
- the UI layout is intentionally split into:
  - left column for death/output/step/log visibility
  - right column for one merged controls pane
- the UI should not poll every 2 seconds all the time; it now polls continuously only while a pipeline is active or cooling down
- the `rusty/` branch introduces a future resident Rust ML bridge:
  - JS/CoffeeScript continues to own orchestration, DAGs, Memo, YAML, SQLite, and UI
  - Rust/C++ owns ML lifetime, tensors, KV cache structures, Metal/unified-memory pressure, and model residency
  - JS may only hold opaque handles for ML-side objects
  - the current smoke verifier source of truth is `session_layer_residency_probe`
  - current corrected fastsmoke generation uses q_norm/k_norm before RoPE and
    validates deterministic same-session generation rather than the old
    pre-q_norm token sequence
  - latest passing fastsmoke generated token ids are `[15, 13, 16]`; major
    projection backends are Metal and handle counts are clean

Suggested use:
- when a step fails, inspect its memory file first
- if the real cause is upstream, follow the listed dependency chain
- when code changes invalidate a memory file, update it in the same work
- for the `rusty/` bridge work, prefer explicit CLI Q/A probes for tensor and
  model inspection, not answers-only logging
