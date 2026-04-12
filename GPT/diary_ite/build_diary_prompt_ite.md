Step: `build_diary_prompt_ite`
Recipe: `diary_ite`

Purpose:
- build the final diary generation prompt from story event scaffold and matched KAG

Inputs:
- artifact `story_parts`
- artifact `diary_kag`

Outputs:
- artifact `diary_prompt_text`

Current event source:
- `story_parts` is the current valid event object
- it comes from `load_library -> select_story_recipe -> resolve_story_parts`

Invariants:
- `diary_prompt_text` must be produced before either diary generator runs
- `generate_diary_with_adapter_ite` and `generate_diary_without_adapter_ite` are sibling consumers

Known pitfalls:
- if downstream sees bad `diary_prompt_text`, inspect merged graph in `experiment.yaml`
- invalid override semantics can break this step indirectly
