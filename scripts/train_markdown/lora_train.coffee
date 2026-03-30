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

hasAdapterConfig = (adapterPath) ->
  return false unless adapterPath?
  fs.existsSync path.join(adapterPath, 'adapter_config.json')

@step =
  desc: "Run MLX LoRA training for markdown-derived train and valid sets"

  action: (S) ->
    testOnly     = S.param 'test_only', false
    adapterPath  = S.param 'adapter_path'
    resumeFile   = S.param 'resume_adapter_file'
    modelDir     = S.param 'loraLand'
    trainingDir  = S.param 'training_dir'

    await S.need 'train_rows'
    await S.need 'valid_rows'
    await S.need 'test_rows'

    console.log "[lora_train]"

    actualResumeFile = resolveResumeFile adapterPath, resumeFile
    adapterConfigExists = hasAdapterConfig adapterPath

    args =
      model: modelDir
      data: trainingDir

    if testOnly
      args.test = null
      console.log "[lora_train] mode: test"
    else
      args.train = null
      console.log "[lora_train] mode: train"

    if testOnly
      if adapterConfigExists
        args["adapter-path"] = adapterPath
        console.log "[lora_train] test mode using existing adapter path:", adapterPath
      else
        console.log "[lora_train] test mode without adapter path; no adapter_config.json at:", adapterPath
    else
      args["adapter-path"] = adapterPath

    args["resume-adapter-file"] = actualResumeFile if actualResumeFile? and not testOnly

    if actualResumeFile? and not testOnly
      console.log "[lora_train] resuming from:", actualResumeFile
    else if not testOnly
      console.log "[lora_train] no prior adapter found, starting fresh"

    stdout = S.callMLX 'lora', args

    S.make 'lora_stdout', stdout
    S.done()
    return
