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

@step =
  desc: "Generate a diary entry using the trained adapter"

  action: (M, stepName) ->
    storyIDEntry = M.theLowdown 'selected_story_id'
    storyID = storyIDEntry?.value
    if storyID is undefined
      if typeof storyIDEntry?.waitFor is 'function'
        storyID = await storyIDEntry.waitFor()
      else if storyIDEntry?.notifier?
        storyID = await storyIDEntry.notifier

    promptEntry = M.theLowdown 'diary_prompt_text'
    prompt = promptEntry?.value
    if prompt is undefined
      if typeof promptEntry?.waitFor is 'function'
        prompt = await promptEntry.waitFor()
      else if promptEntry?.notifier?
        prompt = await promptEntry.notifier

    throw new Error "[#{stepName}] selected_story_id must be a string" unless typeof storyID is 'string'
    throw new Error "[#{stepName}] diary_prompt_text must be a string" unless typeof prompt is 'string'

    modelMemoKey = M.getStepParam(stepName, 'model_memo_key')
    modelMemoKey = 'modelDir' if modelMemoKey is undefined
    explicitModelDir = M.getStepParam(stepName, 'model_dir')
    modelDir = explicitModelDir ? M.theLowdown(modelMemoKey)?.value ? M.theLowdown('modelDir')?.value
    adapterPath = M.getStepParam(stepName, 'adapter_path')

    throw new Error "[#{stepName}] Missing modelDir in memo" unless modelDir?
    throw new Error "[#{stepName}] Missing adapter_path" unless adapterPath?

    rawOutput = M.callMLX 'generate',
      model: modelDir
      prompt: prompt
      "adapter-path": adapterPath
    , M.getStepParam(stepName, 'debug_mlx') is true

    text = cleanGeneratedText prompt, rawOutput
    meta =
      story_id: storyID
      mode: 'adapter'
      model_dir: modelDir
      adapter_path: adapterPath
      raw_chars: String(rawOutput ? '').length
      text_chars: text.length

    console.log "[generate_diary_with_adapter_ite] story:", storyID
    console.log "[generate_diary_with_adapter_ite] text chars:", text.length

    M.saveThis 'diary_adapted_raw', String(rawOutput ? '')
    M.saveThis 'diary_adapted_meta', meta
    M.saveThis 'diary_adapted_text', text
    M.saveThis "done:#{stepName}", true
    return
