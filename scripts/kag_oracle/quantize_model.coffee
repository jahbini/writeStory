fs = require 'fs'
path = require 'path'
cp = require 'child_process'

getParam = (M, stepName, key, defaultValue = undefined) ->
  stepParams = M.theLowdown("params/#{stepName}.yaml")?.value ? {}
  globalParams = M.theLowdown("params/_global.yaml")?.value ? {}

  if Object.prototype.hasOwnProperty.call(stepParams, key)
    return stepParams[key]

  if Object.prototype.hasOwnProperty.call(globalParams, key)
    return globalParams[key]

  defaultValue

listFiles = (rootDir) ->
  out = []

  walk = (currentDir) ->
    for name in fs.readdirSync(currentDir)
      fullPath = path.join(currentDir, name)
      stat = fs.statSync(fullPath)
      if stat.isDirectory()
        walk fullPath
      else
        out.push path.relative(rootDir, fullPath)

  walk rootDir
  out

inspectMLXModelDir = (modelDir) ->
  return { valid: false, reason: 'missing directory' } unless fs.existsSync(modelDir)
  return { valid: false, reason: 'not a directory' } unless fs.statSync(modelDir).isDirectory()

  files = listFiles modelDir
  hasConfig = files.includes 'config.json'
  hasTokenizer = files.includes('tokenizer.json') or files.includes('tokenizer.model')
  hasWeights = files.some (fileName) -> /\.safetensors$/.test(fileName)

  missing = []
  missing.push 'config.json' unless hasConfig
  missing.push 'tokenizer.json|tokenizer.model' unless hasTokenizer
  missing.push '*.safetensors' unless hasWeights

  return {
    valid: missing.length is 0
    reason: if missing.length then "missing #{missing.join(', ')}" else 'ok'
  }

runQuantize = (sourceDir, targetDir) ->
  res = cp.spawnSync 'mlx_lm.convert', [
    '--hf-path', sourceDir
    '--mlx-path', targetDir
    '-q'
    '--q-bits', '4'
  ],
    stdio: 'inherit'

  throw res.error if res.error?
  return if res.status is 0
  throw new Error "mlx_lm.convert failed with exit #{res.status}"

@step =
  desc: "Quantize the laptop oracle MLX model into build/model4"

  action: (M, stepName) ->
    sourceParam = getParam M, stepName, 'source_model_dir', 'build/model'
    targetParam = getParam M, stepName, 'quantized_model_dir', 'build/model4'
    memoKey = getParam M, stepName, 'quantized_model_memo_key', 'quantizedModelDir'

    sourceDir = path.resolve sourceParam
    targetDir = path.resolve targetParam

    sourceState = inspectMLXModelDir sourceDir
    throw new Error "[quantize_model] source model invalid at #{sourceDir}: #{sourceState.reason}" unless sourceState.valid

    targetState = inspectMLXModelDir targetDir
    if targetState.valid
      console.log "[quantize_model] quantized model already exists, skipping"
      M.saveThis memoKey, targetDir
      M.saveThis "done:#{stepName}", true
      return

    if fs.existsSync(targetDir)
      console.log "[quantize_model] removing invalid existing #{targetParam}"
      fs.rmSync targetDir, recursive: true, force: true

    fs.mkdirSync path.dirname(targetDir), recursive: true

    console.log "[quantize_model] creating #{targetParam} from #{sourceParam}"
    runQuantize sourceDir, targetDir

    finalState = inspectMLXModelDir targetDir
    throw new Error "[quantize_model] quantized model invalid at #{targetDir}: #{finalState.reason}" unless finalState.valid

    console.log "[quantize_model] quantization complete"
    M.saveThis memoKey, targetDir
    M.saveThis "done:#{stepName}", true
    return
