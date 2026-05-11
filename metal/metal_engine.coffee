###
metal_engine.coffee — CoffeeScript wrapper for metal_llm.node
-------------------------------------------------------------------------------
Synchronous, debug-friendly, pipeline-ready.
No async. No hidden states. No magic.
###

path = require 'path'

# Load the native addon once.
metal = require path.resolve(__dirname, './metal_llm.node')


# ---- internal helper --------------------------------------------------------
makeEngine = (handle) ->
  eng =
    handle: handle
    vocabSize: metal.getVocabSize(handle)
    hiddenSize: metal.getHiddenSize(handle)
    numLayers: metal.getNumLayers(handle)
  return eng


# ---- public API -------------------------------------------------------------


###
loadEngine(modelDir)
  modelDir  – string, path to model directory
Returns:
  { handle, vocabSize, hiddenSize, numLayers }
###
exports.loadEngine = (modelDir) ->
  if not modelDir?
    throw new Error "loadEngine: modelDir is required"

  modelPath = path.resolve(modelDir)
  tokPath   = path.resolve(modelDir)   # same path for now

  handle = metal.loadModel(modelPath, tokPath)
  if not handle?
    throw new Error "loadEngine: native loadModel returned null"

  eng = makeEngine(handle)
  return eng

###
forwardStep(engine, tokenId, pos)
  engine   – result of loadEngine()
  tokenId  – integer
  pos      - integer
Returns Float32Array of logits
###
exports.forwardStep = (engine, tokenId, pos) ->
  if not engine? or not engine.handle?
    throw new Error "forwardStep: invalid engine"
  console.log "DEBUG: vocab =", engine.vocabSize
  console.log "DEBUG: tokenId =", tokenId

  vocab = engine.vocabSize
  if vocab <= 0
    throw new Error "forwardStep: vocab size invalid"

  # Allocate logits
  logits = new Float32Array(vocab)

  rc = metal.forwardStep(engine.handle, tokenId, pos, logits)
  if rc isnt 0
    throw new Error "forwardStep: native returned status #{rc}"

  return logits



###
resetKV(engine)
###
exports.resetKV = (engine) ->
  if not engine? or not engine.handle?
    throw new Error "resetKV: invalid engine"

  rc = metal.resetKV(engine.handle)
  if rc isnt 0
    throw new Error "resetKV: status #{rc}"

  return true



###
freeEngine(engine)
###
exports.freeEngine = (engine) ->
  if not engine? or not engine.handle?
    return false

  metal.freeModel(engine.handle)
  engine.handle = null
  return true
