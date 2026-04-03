fs = require 'fs'
path = require 'path'

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

detectCheckpointPath = (adapterPath) ->
  return null unless adapterPath? and fs.existsSync(adapterPath)

  finalAdapter = path.join(adapterPath, 'adapters.safetensors')
  return finalAdapter if fs.existsSync(finalAdapter)

  checkpoints = fs.readdirSync(adapterPath)
    .filter (name) -> /^\d+_adapters\.safetensors$/.test(name)
    .sort()

  return null unless checkpoints.length
  path.join adapterPath, checkpoints[checkpoints.length - 1]

hasAdapterConfig = (adapterPath) ->
  return false unless adapterPath?
  fs.existsSync path.join(adapterPath, 'adapter_config.json')

@step =
  desc: "Run MLX LoRA training using direct Memo access"

  action: (M, stepName) ->
    trainEntry = M.theLowdown 'train_rows'
    trainRows = trainEntry?.value
    if trainRows is undefined
      if typeof trainEntry?.waitFor is 'function'
        trainRows = await trainEntry.waitFor()
      else if trainEntry?.notifier?
        trainRows = await trainEntry.notifier

    validEntry = M.theLowdown 'valid_rows'
    validRows = validEntry?.value
    if validRows is undefined
      if typeof validEntry?.waitFor is 'function'
        validRows = await validEntry.waitFor()
      else if validEntry?.notifier?
        validRows = await validEntry.notifier

    testEntry = M.theLowdown 'test_rows'
    testRows = testEntry?.value
    if testRows is undefined
      if typeof testEntry?.waitFor is 'function'
        testRows = await testEntry.waitFor()
      else if testEntry?.notifier?
        testRows = await testEntry.notifier

    throw new Error "[#{stepName}] train_rows must be an array" unless Array.isArray trainRows
    throw new Error "[#{stepName}] valid_rows must be an array" unless Array.isArray validRows
    throw new Error "[#{stepName}] test_rows must be an array" unless Array.isArray testRows

    testOnly = M.getStepParam(stepName, 'test_only')
    testOnly = false if testOnly is undefined
    adapterPath = M.getStepParam(stepName, 'adapter_path')
    resumeFile = M.getStepParam(stepName, 'resume_adapter_file')
    modelMemoKey = M.getStepParam(stepName, 'model_memo_key')
    modelMemoKey = 'modelDir' if modelMemoKey is undefined
    explicitModelDir = M.getStepParam(stepName, 'model_dir')
    trainingDir = M.getStepParam(stepName, 'training_dir')
    modelDir = explicitModelDir ? M.theLowdown(modelMemoKey)?.value ? M.theLowdown('modelDir')?.value

    throw new Error "[#{stepName}] Missing model directory" unless modelDir?
    throw new Error "[#{stepName}] Missing training_dir" unless trainingDir?

    actualResumeFile = resolveResumeFile adapterPath, resumeFile
    adapterConfigExists = hasAdapterConfig adapterPath

    args =
      model: modelDir
      data: trainingDir

    if testOnly
      args.test = null
      console.log "[run_lora_train_ite] mode: test"
    else
      args.train = null
      console.log "[run_lora_train_ite] mode: train"

    if testOnly
      if adapterConfigExists
        args["adapter-path"] = adapterPath
      else
        console.log "[run_lora_train_ite] no adapter_config.json at:", adapterPath
    else
      args["adapter-path"] = adapterPath

    args["resume-adapter-file"] = actualResumeFile if actualResumeFile? and not testOnly

    startedAt = new Date().toISOString()
    runID = "lora-#{startedAt.replace(/[:.]/g, '-')}"

    mlxDebug = M.getStepParam(stepName, 'debug_mlx')
    mlxDebug = false if mlxDebug is undefined

    stdoutText = M.callMLX 'lora', args, mlxDebug

    finishedAt = new Date().toISOString()
    checkpointPath = detectCheckpointPath adapterPath

    runRecord =
      run_id: runID
      started_at: startedAt
      finished_at: finishedAt
      status: 'done'
      model_dir: modelDir
      adapter_path: adapterPath
      resume_adapter_file: actualResumeFile
      training_dir: trainingDir
      stdout_text: stdoutText
      train_rows_count: trainRows.length
      valid_rows_count: validRows.length
      test_rows_count: testRows.length
      checkpoint_path: checkpointPath

    console.log "[run_lora_train_ite] train rows:", trainRows.length
    console.log "[run_lora_train_ite] valid rows:", validRows.length
    console.log "[run_lora_train_ite] test rows:", testRows.length
    console.log "[run_lora_train_ite] run id:", runID

    M.saveThis 'lora_stdout', stdoutText
    M.saveThis 'lora_run_record', runRecord
    M.saveThis "done:#{stepName}", true
    return
