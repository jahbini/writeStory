Step: `collect_diary_kag_ite`
Recipe: `diary_ite`

Purpose:
- select KAG entries that best fit the chosen diary events

Inputs:
- artifact `story_parts`
- meta read `kagFor{story_id}.json`
- optional param `pretend_story_ids`

Outputs:
- artifact `diary_kag`

Current matching mode:
- can use `pretend_story_ids` as a stable KAG source set
- otherwise scores KAG entries across all stories

Important KAG fields:
- `keyword`
- `headline`
- `chunk_index`
- `paragraph_index`

Why `chunk_index` matters:
- diary event matching should eventually align events to story chunks
