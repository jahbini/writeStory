###
Use LLM to expand and polish assembled story into Jim narrative voice.
###

renderPromptText = (template, storyText) ->
  text = String(template ? '')
  text.replace /\{\{story_text\}\}/g, String(storyText ? '')

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
  desc: "Expand and polish story using LLM"

  action: (S) ->
    storyKey = S.param 'story_key', null
    storyFragment = S.param 'story_fragment', null
    promptKey = S.param 'prompt_key', null
    promptText = S.param 'prompt_text', null
    rawOutputKey = S.param 'raw_output_key', null
    modelMemoKey = S.param 'model_memo_key', 'modelDir'
    explicitModelDir = S.param 'model_dir', null

    story = null
    baseText = ''

    if storyKey?
      storyEntry = S.theLowdown storyKey
      story = storyEntry?.value
      if story is undefined
        if typeof storyEntry?.waitFor is 'function'
          story = await storyEntry.waitFor()
        else if storyEntry?.notifier?
          story = await storyEntry.notifier
      throw new Error "[#{S.stepName}] Missing input key '#{storyKey}'" if story is undefined
      baseText = story.text ? ''
    else
      baseText = storyFragment ? ''
      story = text: baseText
    prompt = null
    if promptText?
      prompt = renderPromptText promptText, baseText
    else if promptKey?
      promptEntry = S.theLowdown promptKey
      prompt = promptEntry?.value
      if prompt is undefined
        if typeof promptEntry?.waitFor is 'function'
          prompt = await promptEntry.waitFor()
        else if promptEntry?.notifier?
          prompt = await promptEntry.notifier
      throw new Error "[#{S.stepName}] Missing input key '#{promptKey}'" if prompt is undefined
    else
      throw new Error "[#{S.stepName}] Missing prompt_text or prompt_key"
    modelDir = explicitModelDir ? S.theLowdown(modelMemoKey)?.value
    throw new Error "[#{S.stepName}] Missing modelDir in memo" unless modelDir?

    adapterPath = S.param 'adapter_path', null

    mlxArgs =
      model: modelDir
      prompt: prompt

    mlxArgs["adapter-path"] = adapterPath if adapterPath?

    rawOutput = S.callMLX 'generate', mlxArgs

    polishedText = cleanGeneratedText prompt, rawOutput

    out =
      story_id: story.story_id ? null
      text: polishedText
      source_story: story

    S.saveThis "story_polished", out
    S.make rawOutputKey, String(rawOutput ? '') if rawOutputKey?
    S.make 'story_text', polishedText
    S.done()
    return
