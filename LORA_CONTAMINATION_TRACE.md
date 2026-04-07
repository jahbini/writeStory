# LoRA Contamination Trace

## Active submission paths

Current `_ite` LoRA path:

1. [`data/jim.md`](/Users/jahbini/writeStory/data/jim.md)
2. [`scripts/kag_oracle_ite/seed_story_sqlite.coffee`](/Users/jahbini/writeStory/scripts/kag_oracle_ite/seed_story_sqlite.coffee)
3. `runtime.sqlite` `stories.text`
4. [`scripts/lora_ite/select_lora_stories_ite.coffee`](/Users/jahbini/writeStory/scripts/lora_ite/select_lora_stories_ite.coffee)
5. [`scripts/lora_ite/build_lora_dataset_ite.coffee`](/Users/jahbini/writeStory/scripts/lora_ite/build_lora_dataset_ite.coffee)
6. [`build/train/train.jsonl`](/Users/jahbini/writeStory/build/train/train.jsonl), [`build/train/valid.jsonl`](/Users/jahbini/writeStory/build/train/valid.jsonl), [`build/train/test.jsonl`](/Users/jahbini/writeStory/build/train/test.jsonl)
7. [`scripts/lora_ite/run_lora_train_ite.coffee`](/Users/jahbini/writeStory/scripts/lora_ite/run_lora_train_ite.coffee)

Older markdown path:

1. [`data/jim.md`](/Users/jahbini/writeStory/data/jim.md)
2. [`scripts/train_markdown/md2segments.coffee`](/Users/jahbini/writeStory/scripts/train_markdown/md2segments.coffee)
3. [`data/marshalled_stories.jsonl`](/Users/jahbini/writeStory/data/marshalled_stories.jsonl)
4. [`scripts/train_markdown/select_training_batch.coffee`](/Users/jahbini/writeStory/scripts/train_markdown/select_training_batch.coffee)
5. [`scripts/train_markdown/prepare_training_data.coffee`](/Users/jahbini/writeStory/scripts/train_markdown/prepare_training_data.coffee)
6. [`build/train/train.jsonl`](/Users/jahbini/writeStory/build/train/train.jsonl), [`build/train/valid.jsonl`](/Users/jahbini/writeStory/build/train/valid.jsonl), [`build/train/test.jsonl`](/Users/jahbini/writeStory/build/train/test.jsonl)
7. [`scripts/train_markdown/lora_train.coffee`](/Users/jahbini/writeStory/scripts/train_markdown/lora_train.coffee)

## Confirmed contamination sources

### 1. Raw story source already contains template and markdown residue

File and function:
- raw source [`data/jim.md`](/Users/jahbini/writeStory/data/jim.md)
- preserved by [`clean`](/Users/jahbini/writeStory/scripts/train_markdown/md2segments.coffee) in [`scripts/train_markdown/md2segments.coffee`](/Users/jahbini/writeStory/scripts/train_markdown/md2segments.coffee)
- preserved by [`clean`](/Users/jahbini/writeStory/scripts/kag_oracle_ite/seed_story_sqlite.coffee) in [`scripts/kag_oracle_ite/seed_story_sqlite.coffee`](/Users/jahbini/writeStory/scripts/kag_oracle_ite/seed_story_sqlite.coffee)

Upstream or introduced:
- already present upstream in raw story text

Contaminated outputs:
- [`data/marshalled_stories.jsonl`](/Users/jahbini/writeStory/data/marshalled_stories.jsonl)
- `runtime.sqlite` `stories.text`
- downstream `build/train/*.jsonl`

Examples observed:
- blockquote residue `>`
- template fragments like `{{{comment:...}}}`
- placeholder residue like `{{{first Name}}}`

Snippet:
```coffee
s = s.replace(/{{{First Name}}}/g, 'friend')
...
s = s.replace(/[_*]{1,3}([^*_]+)[_*]{1,3}/g, '$1')
```

Why it matters:
- the raw source itself is contaminated before any pipeline transform runs
- ingress cleaning must remain the authoritative boundary

### 2. `<stop>` is injected into marshalled stories on the old markdown path

File and function:
- [`action`](/Users/jahbini/writeStory/scripts/train_markdown/md2segments.coffee) in [`scripts/train_markdown/md2segments.coffee`](/Users/jahbini/writeStory/scripts/train_markdown/md2segments.coffee)

Upstream or introduced:
- introduced by transform step

Contaminated outputs:
- [`data/marshalled_stories.jsonl`](/Users/jahbini/writeStory/data/marshalled_stories.jsonl)
- downstream old-path LoRA rows

Snippet:
```coffee
text: "#{story.text}\n\n<stop>"
```

And in paragraph mode:
```coffee
paragraphs[paragraphs.length - 1] = "#{paragraphs[paragraphs.length - 1]}\n\n<stop>"
```

### 3. Prompt scaffolding is intentionally injected into final LoRA rows

Files and functions:
- [`scripts/lora_ite/build_lora_dataset_ite.coffee`](/Users/jahbini/writeStory/scripts/lora_ite/build_lora_dataset_ite.coffee)
- [`scripts/train_markdown/prepare_training_data.coffee`](/Users/jahbini/writeStory/scripts/train_markdown/prepare_training_data.coffee)

Upstream or introduced:
- introduced by transform step

Contaminated outputs:
- [`build/train/train.jsonl`](/Users/jahbini/writeStory/build/train/train.jsonl)
- [`build/train/valid.jsonl`](/Users/jahbini/writeStory/build/train/valid.jsonl)
- [`build/train/test.jsonl`](/Users/jahbini/writeStory/build/train/test.jsonl)

Snippet:
```coffee
prompt = pre_prompt + fragmentText + "\n\nBegin:\n\n"
textOut = prompt + completionText
```

Observed consequence:
- every final training row starts with instruction text like:
  - `You are writing in the narrative voice...`
  - `Story fragment:`
  - `Begin:`

### 4. `<stop>` is deliberately added again to final LoRA rows

Files and functions:
- [`scripts/lora_ite/build_lora_dataset_ite.coffee`](/Users/jahbini/writeStory/scripts/lora_ite/build_lora_dataset_ite.coffee)
- [`scripts/train_markdown/prepare_training_data.coffee`](/Users/jahbini/writeStory/scripts/train_markdown/prepare_training_data.coffee)

Upstream or introduced:
- introduced by transform step

Contaminated outputs:
- [`build/train/train.jsonl`](/Users/jahbini/writeStory/build/train/train.jsonl)
- [`build/train/valid.jsonl`](/Users/jahbini/writeStory/build/train/valid.jsonl)
- [`build/train/test.jsonl`](/Users/jahbini/writeStory/build/train/test.jsonl)

Snippet:
```coffee
STOP_TEXT = "\n\n<stop>"
...
textOut = sanitizeStop(textOut) + STOP_TEXT
```

### 5. Old markdown residue survives into final LoRA JSONL

File and function:
- preserved from [`data/jim.md`](/Users/jahbini/writeStory/data/jim.md)
- passed through [`scripts/train_markdown/md2segments.coffee`](/Users/jahbini/writeStory/scripts/train_markdown/md2segments.coffee)
- packed by [`scripts/train_markdown/prepare_training_data.coffee`](/Users/jahbini/writeStory/scripts/train_markdown/prepare_training_data.coffee)
- packed by [`scripts/lora_ite/build_lora_dataset_ite.coffee`](/Users/jahbini/writeStory/scripts/lora_ite/build_lora_dataset_ite.coffee) when the same stories were already seeded into SQLite

Upstream or introduced:
- present upstream and preserved by transform steps

Contaminated outputs:
- [`data/marshalled_stories.jsonl`](/Users/jahbini/writeStory/data/marshalled_stories.jsonl)
- [`build/train/train.jsonl`](/Users/jahbini/writeStory/build/train/train.jsonl)

Observed examples in current files:
- `> I got a huge backlog...`
- `{{{comment:Soto: ...}}}`
- `{{{roar:Roar of the Lion}}}`
- `{{{first Name}}}`

### 6. Current SQLite ingress is now the correct cleaning boundary

File and function:
- [`clean`](/Users/jahbini/writeStory/scripts/kag_oracle_ite/seed_story_sqlite.coffee) in [`scripts/kag_oracle_ite/seed_story_sqlite.coffee`](/Users/jahbini/writeStory/scripts/kag_oracle_ite/seed_story_sqlite.coffee)

Upstream or introduced:
- cleanup is applied here before stories enter SQLite

Effect:
- strips template residue, markup residue, machine markers, and symbolic junk
- no longer writes `<stop>` into `runtime.sqlite` story text

Why it matters:
- oracle, diary, and `_ite` LoRA now share the same cleaned DB corpus
- the remaining confirmed contamination in LoRA submission is primarily introduced later by training-row assembly

## Best insertion point for preflight validation

Best immediate gate:
- after dataset assembly
- before LoRA training launch

Concretely:
- for [`config/lora_ite.yaml`](/Users/jahbini/writeStory/config/lora_ite.yaml):
  - after `build_lora_dataset_ite`
  - before `run_lora_train_ite`
- for old markdown recipes:
  - after `prepare_training_data`
  - before `lora_train`

Why this point is best:
- it sees the actual submitted `build/train/*.jsonl`
- it catches both upstream residue and transform-introduced scaffolding
- it fails before the expensive LoRA run starts

Recommended secondary check:
- also scan raw source before seeding/segmenting
- but the hard gate should be on final training rows
