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

Suggested use:
- when a step fails, inspect its memory file first
- if the real cause is upstream, follow the listed dependency chain
- when code changes invalidate a memory file, update it in the same work
