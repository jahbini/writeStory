# generator/generator.coffee

CoffeeScript = require 'coffeescript'
CoffeeScript.register()

tokenizer = require '../tokenizer/tokenizer.coffee'
metal     = require '../metal/metal_llm.node'

{ encode, decode } = tokenizer


class MetalSession
  constructor: ({modelPath}) ->
    @model = metal.loadModel modelPath

    @vocabSize  = metal.getVocabSize @model
    console.log "model constructed"
    @hiddenSize = metal.getHiddenSize @model
    @numLayers  = metal.getNumLayers @model

    metal.resetKV @model
    console.log "JIM constructing"

  isEos: (tok) ->
    tok == 2   # Phi-style EOS

  free: ->
    metal.freeModel @model
    @model = null

  forward: (tokenId, pos) ->
    logits = new Float32Array @vocabSize
    console.log "JIM forward", tokenId, tokenizer.decode [ tokenId ]
    metal.forwardStep @model, tokenId, pos, logits
    logits

  greedy: (logits) ->
    best    = 0
    bestVal = -Infinity
    i = 0
    while i < logits.length
      v = logits[i]
      if v > bestVal
        bestVal = v
        best    = i
      i++
    best

  nextTokenSample: (logits, temperature = 1.0, topP = 0.9) ->
    vocabSize = logits.length
    if vocabSize <= 0
      throw new Error "nextTokenSample: empty logits"

    if temperature <= 1e-6
      return @greedy logits

    scaled = new Array vocabSize
    i = 0
    while i < vocabSize
      scaled[i] = logits[i] / temperature
      i++

    maxLogit = scaled[0]
    i = 1
    while i < vocabSize
      if scaled[i] > maxLogit then maxLogit = scaled[i]
      i++

    probs  = new Array vocabSize
    sumExp = 0.0

    i = 0
    while i < vocabSize
      ex = Math.exp(scaled[i] - maxLogit)
      probs[i] = ex
      sumExp  += ex
      i++

    invSum = 1.0 / sumExp
    i = 0
    while i < vocabSize
      probs[i] *= invSum
      i++

    if topP > 0 and topP < 1.0
      pairs = []
      i = 0
      while i < vocabSize
        pairs.push
          id: i
          p:  probs[i]
        i++

      pairs.sort (a, b) -> b.p - a.p

      cutoff = []
      cum = 0.0
      for pair in pairs
        cutoff.push pair
        cum += pair.p
        break if cum >= topP

      sumCut = 0.0
      for pair in cutoff
        sumCut += pair.p

      invCut = 1.0 / sumCut
      for pair in cutoff
        pair.p *= invCut

      r = Math.random()
      cum = 0.0
      for pair in cutoff
        cum += pair.p
        if r <= cum
          return pair.id

      return cutoff[cutoff.length - 1].id

    r = Math.random()
    cum = 0.0
    i = 0
    while i < vocabSize
      cum += probs[i]
      if r <= cum
        return i
      i++

    return vocabSize - 1

  generate: ({prompt, promptTokens, maxTokens = 128, temperature = 1.0, topP = 0.9}) ->
    # fresh sequence → fresh KV cache
    if @model?
      metal.resetKV @model

    tokens = []
    if promptTokens? and promptTokens.length > 0
      tokens = [...promptTokens]
    else if prompt?
      tokens = tokenizer.encode prompt
    else
      throw new Error "generate() requires prompt or promptTokens"

    output = [...tokens]

    effectiveTemp = if temperature? and temperature > 1e-3 then temperature else 0

    # --- Warm up KV cache with all prompt tokens except the last ---
    if tokens.length > 1
      pos = 0
      while pos < tokens.length - 1
        tok = tokens[pos]
        @forward tok, pos
        pos++
    # --- Autoregressive generation ---
    for step in [0...maxTokens]
      lastIdx = output.length - 1
      last    = output[lastIdx]

      # position is current index in sequence
      logits = @forward last, lastIdx

      nextId =
        if effectiveTemp <= 0
          @greedy logits
        else
          @nextTokenSample logits, effectiveTemp, topP

      output.push nextId
      break if @isEos nextId

    output

  generateText: ({prompt, maxTokens = 128, temperature = 1.0, topP = 0.9}) ->
    console.log "JIM prompt", prompt
    console.log "Jim Encoded as", tokenizer.encode [ prompt ]
    ids = @generate {prompt, maxTokens, temperature, topP}
    tokenizer.decode ids


module.exports =
  createSession: (cfg) -> new MetalSession cfg
