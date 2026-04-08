cleanGeneratedText = (prompt, rawOutput) ->
  text = String(rawOutput ? '').trim()
  return '' unless text.length

  if text.indexOf(prompt) is 0
    text = text.slice(prompt.length).trim()

  lines = text.split /\r?\n/
  lines = lines.filter (line) ->
    trimmed = line.trim()
    return false if /^Prompt:\s+\d+\s+tokens/.test trimmed
    return false if /^Generation:\s+\d+\s+tokens/.test trimmed
    return false if /^Peak memory:\s+/.test trimmed
    true

  lines.join("\n").trim()

coerceJSON = (value) ->
  return value unless typeof value is 'string'
  try
    JSON.parse value
  catch
    value

readArtifactTarget = (L, artifactKey) ->
  experiment = L.theLowdown('experiment.yaml')?.value ? {}
  targetKey = experiment?.artifacts?[artifactKey]?.target
  return undefined unless typeof targetKey is 'string'

  targetEntry = L.theLowdown targetKey
  targetValue = targetEntry?.value
  if targetValue is undefined
    if typeof targetEntry?.waitFor is 'function'
      targetValue = await targetEntry.waitFor()
    else if targetEntry?.notifier?
      targetValue = await targetEntry.notifier
  targetValue

resolveRunTag = (L) ->
  raw = process.env.HH_MM ? L.theLowdown('env/HH_MM')?.value ? null
  return null unless raw?
  text = String(raw).trim()
  text = text.replace(/^"+|"+$/g, '')
  text = text.replace(/^'+|'+$/g, '')
  return null unless text.length
  text

@step =
  desc: "Generate a diary entry without the adapter"

  action: (L) ->
    storyParts = await L.need 'story_parts'
    prompt = await L.need 'diary_prompt_text'
    storyParts = coerceJSON storyParts

    unless storyParts? and typeof storyParts is 'object' and not Array.isArray(storyParts)
      storyParts = await readArtifactTarget L, 'story_parts'
      storyParts = coerceJSON storyParts

    storyID = String(storyParts?.story_id ? '').trim()
    throw new Error "[#{L.stepName}] story_parts must be an object" unless storyParts? and typeof storyParts is 'object' and not Array.isArray(storyParts)
    throw new Error "[#{L.stepName}] diary_prompt_text must be a string" unless typeof prompt is 'string'

    modelDir = L.param 'quantized_model_dir', null

    throw new Error "[#{L.stepName}] Missing quantized_model_dir param" unless modelDir?

    rawOutput = L.callMLX 'generate',
      model: modelDir
      prompt: prompt

    text = cleanGeneratedText prompt, rawOutput
    meta =
      story_id: storyID
      mode: 'base'
      model_dir: modelDir
      adapter_path: null
      raw_chars: String(rawOutput ? '').length
      text_chars: text.length

    console.log "[generate_diary_without_adapter_ite] story:", storyID
    console.log "[generate_diary_without_adapter_ite] text chars:", text.length

    L.make 'diary_base_raw', String(rawOutput ? '')
    L.make 'diary_base_meta', meta
    L.make 'diary_base_text', text
    runTag = resolveRunTag L
    if typeof runTag is 'string' and runTag.length
      L.saveThis "diary/diary_#{runTag}.txt", text
    L.done()
    return
