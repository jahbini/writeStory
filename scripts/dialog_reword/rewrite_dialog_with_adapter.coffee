cleanGeneratedText = (prompt, rawOutput) ->
  text = String(rawOutput ? '').trim()
  return '' unless text.length

  if text.indexOf(prompt) is 0
    text = text.slice(prompt.length).trim()

  lines = text.split /\r?\n/
  lines = lines.filter (line) ->
    trimmed = line.trim()
    return false if /^=+$/.test trimmed
    return false if /^Prompt:\s+\d+\s+tokens/.test trimmed
    return false if /^Generation:\s+\d+\s+tokens/.test trimmed
    return false if /^Peak memory:\s+/.test trimmed
    true

  lines.join("\n").trim()

resolveRunTag = (L) ->
  raw = process.env.HH_MM ? L.theLowdown('env/HH_MM')?.value ? null
  return null unless raw?
  text = String(raw).trim()
  text = text.replace(/^"+|"+$/g, '')
  text = text.replace(/^'+|'+$/g, '')
  return null unless text.length
  text

renderPrompt = (template, dialogText, keywordHint) ->
  throw new Error "rewrite_prompt_text must be a string" unless typeof template is 'string'
  text = template
  text = text.replace /\{dialog_text\}/g, String(dialogText ? '')
  text = text.replace /\{keyword_hint\}/g, String(keywordHint ? '')
  text

@step =
  desc: "Rewrite a dialog text file with adapter and keyword hint"

  action: (L) ->
    sourceText = await L.need 'dialog_source_text'
    throw new Error "[#{L.stepName}] dialog_source_text must be a non-empty string" unless typeof sourceText is 'string' and sourceText.trim().length

    modelDir = L.param 'quantized_model_dir', null
    adapterPath = L.param 'adapter_path'
    mlxConfig = L.param 'mlx', null
    promptTemplate = L.param 'rewrite_prompt_text'
    keywordHint = String(L.param('keyword_hint', '') ? '').trim()
    outputPrefix = String(L.param('output_file_prefix', 'dialog_reword') ? 'dialog_reword').trim() or 'dialog_reword'

    throw new Error "[#{L.stepName}] Missing quantized_model_dir param" unless modelDir?
    throw new Error "[#{L.stepName}] Missing adapter_path" unless adapterPath?
    throw new Error "[#{L.stepName}] mlx must be an object when provided" if mlxConfig? and (typeof mlxConfig isnt 'object' or Array.isArray(mlxConfig))

    prompt = renderPrompt promptTemplate, sourceText, keywordHint

    mlxArgs =
      model: modelDir
      prompt: prompt
      "adapter-path": adapterPath

    if mlxConfig? and typeof mlxConfig is 'object'
      for own key, value of mlxConfig
        continue unless value?
        mlxArgs[key] = value

    rawOutput = L.callMLX 'generate', mlxArgs
    text = cleanGeneratedText prompt, rawOutput

    meta =
      mode: 'dialog_reword_adapter'
      model_dir: modelDir
      adapter_path: adapterPath
      keyword_hint: keywordHint
      source_chars: sourceText.length
      raw_chars: String(rawOutput ? '').length
      text_chars: text.length

    console.log "[rewrite_dialog_with_adapter] source chars:", sourceText.length
    console.log "[rewrite_dialog_with_adapter] text chars:", text.length
    console.log "[rewrite_dialog_with_adapter] keyword hint:", keywordHint

    L.make 'dialog_reword_raw', String(rawOutput ? '')
    L.make 'dialog_reword_meta', meta
    L.make 'dialog_reword_text', text

    runTag = resolveRunTag L
    if typeof runTag is 'string' and runTag.length
      L.saveThis "out/#{outputPrefix}_#{runTag}.txt", text

    L.done()
    return
