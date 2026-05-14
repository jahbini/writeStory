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
