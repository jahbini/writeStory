Step: `build_lora_story_dataset_ite`
Recipe: `lora_story_ite`

Purpose:
- build chat-formatted `train.jsonl`, `valid.jsonl`, and `test.jsonl` from
  SQLite-backed stories
- companion to `build_lora_dataset_ite`; differs only in output row format

Inputs:
- artifact `selected_story_ids`
- meta reads `storyByID{story_id}.json`

Outputs:
- artifacts `train_rows`, `valid_rows`, `test_rows` — each row is shaped
  `{"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`
  rather than `{"text": "..."}`

Why this exists:
- Qwen3-4B-Instruct expects chat-structured input at inference time. The
  `generate_prompt_gypsy_ite` step wraps every prompt as
  `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` via
  `chat: true`.
- Training on `{"text": ...}` continuation rows trains the adapter on a
  *different* structural shape than inference uses. The adapter then learns
  "after the chat prefix, the right thing to do is end early," which
  presents as immediate-stop-token generation. See `gypsy/STATUS.md`
  history and the May 2026 EOS-failure diagnosis.
- This step emits rows that match the inference shape, so the adapter
  trains and generates in the same distribution.

Segmentation (same as `build_lora_dataset_ite`):
- split each story into 5 paragraph groups when paragraph count >= 5
- if paragraph count < 5, use all paragraphs as one group
- token budget: 1024 total, ~96 reserved for chat-wrapper overhead

User-prompt construction:
- the first 1–2 paragraphs of a group become the "seed text" the user
  pretends to show the model
- the user content is wrapped in one of several rotating instruction
  templates (e.g. "Continue this in your distinctive voice:") so the
  adapter learns the voice is the constant, not the trigger phrase
- the rotation is deterministic on (row_index, story_id) so re-runs are
  reproducible

Invariants:
- assistant content is plain prose from the corpus, with no trailing
  marker tokens — the tokenizer's chat template adds `<|im_end|>`
- single-paragraph stories fall back to sentence-level split, same as
  the `text`-format step
- use SQLite-seeded cleaned text, never raw `jim.md`

Known pitfalls:
- if you copy this step's row format to a non-Instruct base model, the
  base model has no chat template and will throw at training time
- if `chat: false` is set in `generate_prompt_gypsy_ite`, training on
  this step's output is the wrong distribution — switch back to
  `build_lora_dataset_ite` or change inference to `chat: true`
- the variety in `USER_TEMPLATES` is small; resist the urge to make it
  large — the adapter should learn voice from many examples × a few
  template families, not from many template families × few examples
- story IDs in this repo's SQLite are NON-numeric strings. Any code that
  hashes or indexes them must not assume `Number(storyID)` works — it
  returns `NaN`. `pickTemplate` uses a char-code-sum (`storyOffset`)
  instead. (Fixed May 2026 after a `template is not a function` crash.)

Training-cycle design (with `select_lora_stories_ite` +
`run_lora_train_ite`) — CONFIRMED with the human May 2026:

This recipe trains in MANY TINY GENTLE BATCHES, not one big run. The
mental model "iters = total training budget" is WRONG here.

- `select_lora_stories_ite` selects only `batch_size: 4` stories per
  cycle → this step emits roughly 5-15 chat rows per cycle, NOT the
  full corpus.
- `run_lora_train_ite` RESUMES on `build/adapter/adapters.safetensors`.
  The resume IS the accumulation mechanism — each 4-story batch builds
  on the prior batch's adapter. The voice accumulates across dozens of
  batches, not within any one batch. NEVER "fix" a problem by deleting
  the adapter mid-cycle; that throws away the whole accumulated run.
- the UI continuous-loop repeats batch after batch until every story is
  consumed, then `select_lora_stories_ite` shuts the pipeline down. One
  full loop = one pass over all stories.
- small batches are also fault tolerance: a power loss or thermal
  shutdown on the training mini only costs the in-flight batch — every
  completed batch is already checkpointed into the adapter.
- `iters` is PER BATCH and must stay tiny (~5). Each batch should only
  NUDGE the adapter. Driving a single batch to low loss memorizes that
  batch; across the loop that produces an overfit / no-output adapter.
- `out/lora_train.txt` is overwritten every batch — it shows ONLY the
  last batch, never the whole run. A per-batch curve from val loss ~6
  down to ~0.008 means that batch memorized — iters is too high.
- the cycle auto-resets: when `select_lora_stories_ite` exhausts the
  story pool it sets `cycleState.ready_for_reset = true` and shuts the
  pipeline down once; the NEXT run sees that flag and starts a fresh
  full pass. "Shutdown: no remaining stories" is expected end-of-pass
  behavior, not an error.

RESOLVED (May 2026): a chat-format run at `iters: 80` per batch trained
each batch to train loss < 0.02 (memorization), and the accumulated
adapter produced NO output in gypsy. Root cause: `iters: 80` is far too
high for this gentle-nudge design — the intended per-batch value is ~5.
The assistant (me) had earlier argued the human UP from `iters: 5` to
500/80 based on the wrong "iters = total budget" model; that was the
mistake. Recipe default is now `iters: 5`. Secondary issue still worth
tightening (see "Known pitfalls" — short assistant completions): some
rows put more text in the user turn than the assistant turn, which
nudges toward terseness, but it is NOT the primary no-output cause.
