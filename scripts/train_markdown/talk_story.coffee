@step =
  desc: "Generate a story sample using the markdown-trained LoRA adapter"

  action: (M, stepName) ->
    adapterPath  = M.getStepParam stepName, 'adapter_path'
    promptText   = M.getStepParam stepName, 'prompt_text'
    maxTokens    = M.getStepParam stepName, 'max_tokens'
    topP         = M.getStepParam stepName, 'top_p'
    temperature  = M.getStepParam stepName, 'temperature'
    outputText   = M.getStepParam stepName, 'output_text'

    modelDir = M.theLowdown('modelDir')?.value

    output = M.callMLX 'generate',
      model: modelDir
      "adapter-path": adapterPath
      prompt: promptText
      "max-tokens": String(maxTokens)
      "top-p": String(topP)
      temp: String(temperature)

    M.saveThis outputText, output
    M.saveThis "done:#{stepName}", true
    return
