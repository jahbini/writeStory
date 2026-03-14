loadArray = (M, key) ->
  entry = M.theLowdown key
  value = entry?.value
  value = await entry.notifier if value is undefined
  value

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
    stdoutKey    = M.getStepParam stepName, 'stdout_text'

    trainData = await loadArray M, trainKey
    validData = await loadArray M, validKey
    modelDir = M.theLowdown('modelDir')?.value

    throw new Error "#{trainKey} must be an array" unless Array.isArray(trainData)
    throw new Error "#{validKey} must be an array" unless Array.isArray(validData)

    console.log "[lora_train]"
    console.log "  train rows:", trainData.length
    console.log "  valid rows:", validData.length

    if trainData.length is 0
      console.log "[lora_train] no new training data, skipping"
      M.saveThis "done:#{stepName}", true
      return

    stdout = M.callMLX 'lora',
      train: null
      model: modelDir
      data: adapterPath.replace(/\/adapter$/, '')
      "adapter-path": adapterPath
      "batch-size": String(batchSize)
      iters: String(iters)
      "max-seq-length": String(maxSeqLength)
      "learning-rate": String(learningRate)

    M.saveThis stdoutKey, stdout
    M.saveThis "done:#{stepName}", true
    return
