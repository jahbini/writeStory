yaml = require 'js-yaml'

pickArtifacts = (runEntry) ->
  out = []
  if runEntry.quantized_dir? then out.push [runEntry.quantized_dir, null, 'quantized']
  if runEntry.fused_dir? then out.push [runEntry.fused_dir, null, 'fused']
  out.push [runEntry.model_id, runEntry.adapter_dir, 'base+adapter']
  uniq = []
  seen = new Set()
  for [modelPath, adapterPath, label] in out
    key = "#{modelPath}|#{adapterPath or ''}"
    continue if seen.has key
    seen.add key
    uniq.push [modelPath, adapterPath, label]
  uniq

runOne = (S, modelPath, adapterPath, prompts, maxTokens) ->
  outs = []
  for prompt in prompts
    args =
      model: modelPath
      prompt: prompt
      "max-tokens": maxTokens
    args["adapter-path"] = adapterPath if adapterPath?
    raw = S.callMLX 'generate', args
    txt = String(raw ? '').trim()
    txt = txt.slice(prompt.length).trim() if txt.indexOf(prompt) is 0
    outs.push txt
  outs

@step =
  desc: "Run regeneration ablations using MLX"

  action: (S) ->
    registry = await S.need 'artifacts_registry'
    prompts = S.param 'prompts'
    maxShort = S.param 'max_new_short'
    maxLong = S.param 'max_new_long'
    onlyModelId = S.param 'only_model_id'

    runs = registry.runs ? []
    if onlyModelId? and String(onlyModelId).length
      runs = runs.filter (r) -> r.model_id is onlyModelId

    pvPlain = (p) -> p
    pvDirective = (p) -> "#{p}\n\nAnswer with a single important thought:"
    pvFewshot = (p) ->
      shots = [
        'The moon does not race the tide.'
        'A river carves stone by lingering.'
      ]
      "Proverbs:\n- #{shots.join('\n- ')}\n\n#{p}\n- "

    variants = [
      ['plain', pvPlain]
      ['directive', pvDirective]
      ['fewshot', pvFewshot]
    ]

    allRows = []
    stamp = new Date().toISOString().replace(/\.\d+Z$/, 'Z')

    for runEntry in runs
      for [modelPath, adapterPath, artifactLabel] in pickArtifacts(runEntry)
        for [variantLabel, variantFn] in variants
          prompted = prompts.map variantFn
          shortOuts = runOne S, modelPath, adapterPath, prompted, maxShort
          longOuts = runOne S, modelPath, adapterPath, prompted, maxLong

          for prompt, idx in prompts
            for budget, generation in [['short', shortOuts[idx] ? ''], ['long', longOuts[idx] ? '']]
              allRows.push
                timestamp_utc: stamp
                model_id: runEntry.model_id
                artifact: artifactLabel
                prompt_variant: variantLabel
                budget: budget
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

    S.make 'ablation_rows', allRows
    S.make 'ablation_groups', yaml.dump(grouped, { sortKeys: false })
    S.done()
    return
