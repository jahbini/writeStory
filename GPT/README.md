This directory is assistant-owned working memory for recipe and step contracts.

Purpose:
- keep step-local memory outside the transient conversation
- record proven contracts and costly failure modes
- help trace downstream failures back to upstream causes

Rules:
- keep files short and factual
- update a step memory when its contract changes
- prefer one file per important step
- record inputs, outputs, invariants, and pitfalls
- do not use this directory for general notes or speculation

Suggested use:
- when a step fails, inspect its memory file first
- if the real cause is upstream, follow the listed dependency chain
- when code changes invalidate a memory file, update it in the same work
