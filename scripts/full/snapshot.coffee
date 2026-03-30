yaml = require 'js-yaml'

pickArtifacts = (runEntry) ->
  out = []
  if runEntry.quantized_dir? then out.push { model: runEntry.quantized_dir, adapter: null, label: 'quantized' }
  if runEntry.fused_dir? then out.push { model: runEntry.fused_dir, adapter: null, label: 'fused' }
  out.push { model: runEntry.model_id, adapter: runEntry.adapter_dir, label: 'base+adapter' }
  uniq = []
  seen = new Set()
  for item in out
    key = "#{item.model}|#{item.adapter or ''}"
    continue if seen.has key
    seen.add key
    uniq.push item
  uniq

runOneModel = (S, modelPath, adapterPath, prompts, maxTokens) ->
  outs = []
  for prompt in prompts
    args =
      model: modelPath
      prompt: prompt
      "max-tokens": maxTokens
    args["adapter-path"] = adapterPath if adapterPath?
    raw = S.callMLX 'generate', args
    text = String(raw ? '').trim()
    text = text.slice(prompt.length).trim() if text.indexOf(prompt) is 0
    outs.push text
  outs

@step =
  desc: "Generate prompt snapshots using MLX"

  action: (S) ->
    registry = await S.need 'artifacts_registry'
    prompts = S.param 'prompts'
    maxNew = S.param 'max_new_tokens'
    onlyModelId = S.param 'only_model_id'

    runs = registry.runs ? []
    if onlyModelId? and String(onlyModelId).length
      runs = runs.filter (r) -> r.model_id is onlyModelId

    allRows = []
    stamp = new Date().toISOString().replace(/\.\d+Z$/, 'Z')

    for runEntry in runs
      for variant in pickArtifacts(runEntry)
        outs = runOneModel S, variant.model, variant.adapter, prompts, maxNew
        for prompt, idx in prompts
          generation = outs[idx] ? ''
          allRows.push
            timestamp_utc: stamp
            model_id: runEntry.model_id
            artifact: variant.label
            prompt: prompt
            generation: generation
            len_chars: generation.length
            len_words: generation.split(/\s+/).filter((x) -> x.length).length
            is_empty: if generation.trim().length is 0 then 1 else 0

    grouped = {}
    for row in allRows
      key = row.prompt.trim()
      grouped[key] ?= []
      grouped[key].push row

    S.make 'generation_rows', allRows
    S.make 'generation_groups', yaml.dump(grouped, { sortKeys: false })
    S.done()
    return
