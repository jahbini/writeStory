fs = require 'fs'
path = require 'path'

loadArray = (M, key) ->
  entry = M.theLowdown key
  value = entry?.value
  value = await entry.notifier if value is undefined
  value

resolveResumeFile = (adapterPath, configuredResumeFile) ->
  return configuredResumeFile if configuredResumeFile? and fs.existsSync(configuredResumeFile)

  return null unless adapterPath? and fs.existsSync(adapterPath)

  finalAdapter = path.join(adapterPath, 'adapters.safetensors')
  return finalAdapter if fs.existsSync(finalAdapter)

  checkpoints = fs.readdirSync(adapterPath)
    .filter (name) -> /^\d+_adapters\.safetensors$/.test(name)
    .sort()

  return null unless checkpoints.length
  path.join adapterPath, checkpoints[checkpoints.length - 1]

@step =
  desc: "Run MLX LoRA training for markdown-derived train and valid sets"

  action: (M, stepName) ->
    batchSize    = M.getStepParam stepName, 'batch_size'
    iters        = M.getStepParam stepName, 'iters'
    maxSeqLength = M.getStepParam stepName, 'max_seq_length'
    learningRate = M.getStepParam stepName, 'learning_rate'
    trainKey     = M.getStepParam stepName, 'train_file'
    validKey     = M.getStepParam stepName, 'valid_file'
    adapterPath  = M.getStepParam stepName, 'adapter_path'
    resumeFile   = M.getStepParam stepName, 'resume_adapter_file'
    stdoutKey    = M.getStepParam stepName, 'stdout_text'
    modelDir     = M.getStepParam stepName, "loraLand"

    console.error "JIM lora"

    console.log "[lora_train]"

    actualResumeFile = resolveResumeFile adapterPath, resumeFile

    args =
      train: null
      model: modelDir
      data: trainKey.replace(/\/train.jsonl$/, '')
      "adapter-path": adapterPath
      "batch-size": String(batchSize)
      iters: String(iters)
      "max-seq-length": String(maxSeqLength)
      "learning-rate": String(learningRate)

    args["resume-adapter-file"] = actualResumeFile if actualResumeFile?

    if actualResumeFile?
      console.log "[lora_train] resuming from:", actualResumeFile
    else
      console.log "[lora_train] no prior adapter found, starting fresh"

    stdout = M.callMLX 'lora', args, true

    M.saveThis stdoutKey, stdout
    M.saveThis "done:#{stepName}", true
    return
