# writeStory test pipeline scaffold

This directory contains only the test pipeline runtime pieces.

- Config: `config/test.yaml`
- Override: `override.yaml`
- Runner: `pipeline_runner.coffee`
- Steps: `scripts/test/*.coffee`

Run:

```bash
pnpm install
coffee pipeline_runner.coffee
```
