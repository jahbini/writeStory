Area: `ui_server.coffee` + `ui/index.html`

Purpose:
- provide the pipe-local UI for selecting recipes, editing overrides, launching runs, and inspecting outputs/logs

Current UI layout contract:
- left column is observability:
  - `Pipeline Death`
  - `Outputs`
  - `Steps`
  - `Latest Err`
  - `Latest Log`
- right column is one merged `Controls` pane containing:
  - `Pipe`
  - `Run`
  - `Recipe And Overrides`
  - recipe selector
  - launch / kill / restart controls
  - recipe UI fields

Current control model:
- recipe UI fields are discovered from the active recipe YAML
- supported directives are:
  - `UI_checkbox`
  - `UI_dropdown`
  - `UI_textarea`
- UI-backed values are stored in `state/ui-control.json`
- effective run control is materialized to `control_override.yaml`
- foundational model identity still belongs in `override.yaml`

Pipe/workspace rules:
- the active UI is pipe-local under `CWD`
- if a pipe-local file exists, prefer it over the repo-top fallback
- a new empty pipe infers `run.model` from the pipe directory name and materializes `override.yaml` if needed

Polling contract:
- do one `refresh()` on page load
- do not run the 2-second polling loop all the time
- start the polling loop when `Write Override And Run` is pressed
- keep polling only while run status is one of:
  - `launching`
  - `running`
  - `killing`
  - `skipped`
  - `cooldown`

Known pitfalls:
- do not silently reintroduce always-on polling
- do not split `Latest Err` and `Latest Log` into one pane unless explicitly requested
- do not treat `control_override.yaml` as a substitute for foundational `override.yaml`
- if a new recipe needs freeform text from the UI, prefer `UI_textarea` over inventing a separate ad hoc endpoint
