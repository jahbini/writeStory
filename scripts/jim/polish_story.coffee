###
Use LLM to expand and polish assembled story into Jim narrative voice.
###

buildPrompt = (storyText) ->
  prompt = """
You are writing in the narrative voice of Jim from St. John's.

Expand the following story fragment into a short reflective narrative of at least 500 words.
Maintain the same events and ideas, but improve flow, imagery, and voice.

Rules:
- Keep the same order of events.
- Do not introduce new plot elements.
- Add natural narration and sensory detail.
- The tone should be observational, slightly humorous, and reflective.
- The final length should be about 800–2000 words.

Return only the finished story.
Story fragment:

#{storyText}
"""

  return prompt

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

  action: (M, stepName) ->
    storyKey = "story"
    storyEntry = M.theLowdown storyKey
    story = storyEntry?.value
    if story is undefined
      if typeof storyEntry?.waitFor is 'function'
        story = await storyEntry.waitFor()
      else if storyEntry?.notifier?
        story = await storyEntry.notifier
    throw new Error "[#{stepName}] Missing input key '#{storyKey}'" if story is undefined

    baseText = story.text ? ''

    prompt = buildPrompt baseText
    modelDir = M.theLowdown('modelDir')?.value
    throw new Error "[#{stepName}] Missing modelDir in memo" unless modelDir?

    adapterPath = M.getStepParam(stepName, 'adapter_path')

    temperature = M.getStepParam(stepName, 'temperature')
    maxTokens   = M.getStepParam(stepName, 'max_tokens')
    minTokens   = M.getStepParam(stepName, 'min_tokens')

    rawOutput = M.callMLX 'generate',
      model: modelDir
      "adapter-path": adapterPath
      prompt: prompt
      temp: String(temperature)
      "max-tokens": String(maxTokens)
      "min-tokens": String(minTokens)

    polishedText = cleanGeneratedText prompt, rawOutput

    out =
      story_id: story.story_id ? null
      text: polishedText
      source_story: story

    M.saveThis "story_polished", out
    M.saveThis "out/story.txt", polishedText
    M.saveThis "done:#{stepName}", true
    return
