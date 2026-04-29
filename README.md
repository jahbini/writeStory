# writeStory

`writeStory` is a CoffeeScript pipeline system for story processing, oracle tagging, LoRA training, and diary generation. The repository is organized around a single runner, a memo-based artifact model, and per-model workspaces under `pipes/`.

## Intent

The core idea is:

- the pipeline is declarative
- each step says what it `needs` and what it `makes`
- the runner wires artifacts through a shared memo instead of hard-coding step-to-step calls

This gives the project a few useful properties:

- steps stay small and focused
- artifacts can be inspected on disk in `out/`, `state/`, `params/`, and `logs/`
- a failed step can usually be understood by checking its inputs, outputs, and recorded state
- the same repo logic can be reused across multiple model workspaces

The main runtime pieces are:

- [pipeline_runner.coffee](./pipeline_runner.coffee): executes a recipe from `config/*.yaml`
- [config/](./config): pipeline recipes such as `base_ite`, `oracle_ite`, `lora_ite`, and `diary_ite`
- [scripts/](./scripts): individual step implementations
- [meta/](./meta): memo-backed file and SQLite loaders/savers

## The Memo And Pipeline Model

The memo is the runner’s shared artifact registry.

In practice that means:

- source artifacts come from files, YAML, JSON, text, or SQLite-backed records
- steps read inputs through the memo
- steps publish outputs back into the memo
- the runner materializes declared outputs to disk when a step finishes

The recipe controls orchestration:

- `run:` holds global settings such as `model`, directories, and debug flags
- `artifacts:` maps logical artifact names to files or values
- each step declares `depends_on`, `needs`, and `makes`

This is why the repository can support several workflows without changing the runner:

- `base_ite` prepares a pipe workspace, downloads the model, quantizes it, and seeds SQLite
- `oracle_ite` uses the prepared quantized model for keyword/emotion tagging
- `lora_ite` trains adapters against the full base model
- `diary_ite` and `diary_translate_ite` use the quantized model plus optional adapter

## Pipe Workspaces

Each model gets its own workspace under `pipes/`.

Examples:

- `pipes/Qwen_Qwen3-4B-Instruct-2507/`
- `pipes/mlx-community_Phi-3.5-mini-instruct-4bit/`
- `pipes/google_gemma-2b-it/`

Each pipe is intended to hold its own runtime state:

- `override.yaml`
- `control_override.yaml`
- `runtime.sqlite`
- `build/`
- `logs/`
- `state/`
- `out/`
- `diary/`

Important contract:

- `override.yaml` is foundational
- it must contain the base model identity for that pipe
- `control_override.yaml` is the UI-owned run control layer

For a new empty pipe, the directory name is treated as the model identity. For example:

- `pipes/google_gemma-2b-it/` implies `run.model: google/gemma-2b-it`

## Setting Up A New Pipe

Install repo dependencies first:

```bash
pnpm install
./.venv/bin/python -m pip install -r requirements.txt
```

Then start the UI for a new model:

```bash
./scripts/admin/start_pipe_ui.sh --model google/gemma-2b-it
```

That launcher will:

- create `pipes/google_gemma-2b-it/`
- write a foundational `override.yaml`
- start the UI server with `EXEC` pointing at the repo and `CWD` pointing at the pipe

You can also let it auto-select an existing pipe:

```bash
./scripts/admin/start_pipe_ui.sh
```

Or expose the UI on the network:

```bash
./scripts/admin/start_pipe_ui.sh net
./scripts/admin/start_pipe_ui.sh net --model google/gemma-2b-it
```

Recommended first run inside a new pipe:

1. Start the UI for the pipe.
2. Select `base_ite`.
3. Run it once.

`base_ite` now prepares the pipe in the correct order:

1. reset base environment
2. download the full model into `build/model`
3. quantize it into `build/model4`
4. seed story SQLite after the model preparation is complete

After that:

- `oracle_ite` uses `build/model4`
- `diary_ite` uses `build/model4`
- `diary_translate_ite` uses `build/model4`
- `lora_ite` still trains against `build/model`

If you are moving trained work from another machine, the intended merge targets are the current pipe’s:

- `build/adapter`
- `runtime.sqlite`

The merge utility and UI merge button are pipe-aware.

## UI And Overrides

The UI is meant to operate inside a specific pipe workspace.

Practical rules:

- the human override editor writes `override.yaml`
- the controls pane writes `control_override.yaml`
- `experiment.yaml` is the effective merged run configuration for the next launch

If a new pipe has little or no local `data/` content yet, UI dropdown sources fall back to the repo-level files where appropriate.

## GPT Working Memory

The [GPT/](./GPT) directory is assistant-oriented working memory for this repository.

Its purpose is to give Codex enough local context to help with future pipeline work without rediscovering everything from logs and conversation history.

Typical contents:

- recipe notes
- step contracts
- invariants
- known failure modes
- dependency-chain explanations

Recommended use:

- when a step fails, inspect the matching file in `GPT/<pipeline>/`
- when a step contract changes, update the matching memory file
- keep it factual and local to the step or recipe

Start with:

- [GPT/README.md](./GPT/README.md)

## Repository Notes

- `README_SETUP.md` is an older minimal setup note and does not describe the current pipe-based workflow.
- The repository expects the project virtualenv to match [requirements.txt](./requirements.txt).
- Model downloads are materialized into the pipe and then stripped of `.git/` after validation, so the pipeline keeps the usable model files without the extra Git/LFS storage overhead.
