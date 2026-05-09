# Rusty Bluesky Ideas

This file records speculative ideas only. Entries here are not active defaults,
not validated implementation plans, and should not be benchmarked unless a
future task explicitly promotes one into an experiment.

## Generational KV Cache Compaction

Idea: maintain two KV stores for decode:

- **Active/read store**: used by attention for the current generation window.
- **Survivor/build store**: gradually receives KV entries that appear useful.

During generation, attention reads from the active store while Rusty records
which prior positions receive meaningful attention. Useful slots are copied or
represented in the survivor store. When the active store reaches a threshold,
swap the stores, clear the old active store, and begin building the next
survivor set.

Potential policy:

- Pin system/prompt tokens.
- Always retain a recent window, for example the last 128 or 256 tokens.
- Track a global per-token usage score from attention probability mass.
- Keep the top-scoring older positions in the survivor store.
- Preserve metadata for each slot: original logical position, token id, pinned
  state, occupied state, and usage score.

Why this might be useful:

- It could bound KV memory without recomputing old tokens.
- It batches compaction work instead of making random per-token eviction choices.
- It may keep the attention input dense, which is friendlier to MLX/Metal than
  scattered single-slot gathers.
- It creates a future path toward spill/swap policies without changing the
  higher-level generation API.

Risks and open questions:

- RoPE/logical position semantics must be preserved. Physical cache slot reuse
  must not imply that a new token has the old token's logical position.
- If retained positions are non-contiguous, output quality may drift.
- Copying MLX KV arrays between stores must avoid host readback/sync.
- Usage scoring should be global and cheap; per-layer/per-head policies may be
  too complex for a first implementation.
- This is not a substitute for current `chunked_expanded_kv` until correctness
  and quality are proven.

Possible future backend name:

- `generational_chunked_kv`

## Least-Used KV Slot Eviction

Idea: after a bounded KV cache fills, replace the least-used unpinned KV slot
with the newest token's K/V entry.

Important correction: the KV cache stores keys and values, not logits. Logits
are final vocabulary scores and should not be cached in this structure.

A practical version would require:

- Slot metadata: logical position, occupied, pinned, recent/protected, and
  usage score.
- Attention-derived usage scoring, such as cumulative attention mass or a
  thresholded "ever attended" marker.
- Causal/logical masking based on original positions, not physical slot index.

This is likely less attractive than generational compaction because it risks
fragmented/scattered attention inputs, but it remains a useful conceptual
baseline.
