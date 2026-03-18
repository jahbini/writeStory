buildPrompt = (storyText) ->
  """
You are writing in the narrative voice of Jim from St. John's.

Expand the following story fragment into a short reflective narrative of at least 500 words.
Maintain the same events and ideas, but improve flow, imagery, and voice.

Rules:
- You are Jim and you write in the first personn
- Keep the same order of events.
- Do not introduce new plot elements.
- Add natural narration and sensory detail.
- The tone should be observational, slightly humorous, and reflective.
- The final length should be about 800–2000 words.

Return only the finished story.
Story fragment:

#{storyText}
"""

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
  desc: "Generate a story sample using the markdown-trained LoRA adapter"

  action: (M, stepName) ->
    adapterPath  = M.getStepParam stepName, 'adapter_path'
    storyFragment = M.getStepParam stepName, 'story_fragment'
    maxTokens    = M.getStepParam stepName, 'max_tokens'
    temperature  = M.getStepParam stepName, 'temperature'
    minTokens    = M.getStepParam stepName, 'min_tokens'
    outputText   = M.getStepParam stepName, 'output_text'

    modelDir = M.theLowdown('modelDir')?.value
    prompt = buildPrompt storyFragment

    rawOutput = M.callMLX 'generate',
      model: modelDir
      "adapter-path": adapterPath
      prompt: prompt
      "max-tokens": String(maxTokens)
      temp: String(temperature)
      "min-tokens": String(minTokens)
      , true

    output = cleanGeneratedText prompt, rawOutput

    M.saveThis outputText, output
    M.saveThis "done:#{stepName}", true
    return
