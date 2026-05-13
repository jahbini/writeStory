path = require 'path'

defaultAddonPath = path.resolve __dirname, '../metal/metal_llm.node'

class GypsySessionApi
  constructor: (opts = {}) ->
    addonPath = opts.addonPath ? defaultAddonPath
    @metal = require addonPath
    @addonPath = addonPath

  requireFunction: (name) ->
    fn = @metal[name]
    unless typeof fn is 'function'
      throw new Error "Gypsy native addon missing function: #{name}"
    fn

  inspectModel: (modelDir) ->
    @requireFunction('inspectModel') modelDir

  inspectTokenizer: (tokenizerDir) ->
    @requireFunction('inspectTokenizer') tokenizerDir

  inspectAdapter: (adapterDir) ->
    @requireFunction('inspectAdapter') adapterDir

  loadModelResident: (modelDir) ->
    @requireFunction('loadModelResident') modelDir

  describeModelGroups: (model) ->
    handle = if typeof model is 'string' then model else model?.handle
    throw new Error 'describeModelGroups requires model handle' unless handle?
    @requireFunction('describeModelGroups') handle

  loadTokenizer: (tokenizerDir) ->
    @requireFunction('loadTokenizer') tokenizerDir

  loadAdapter: (adapterDir) ->
    @requireFunction('loadAdapter') adapterDir

  describeAdapterGroups: (adapter) ->
    handle = if typeof adapter is 'string' then adapter else adapter?.handle
    throw new Error 'describeAdapterGroups requires adapter handle' unless handle?
    @requireFunction('describeAdapterGroups') handle

  createSession: (modelHandle, tokenizerHandle, opts = {}) ->
    @requireFunction('createSession') modelHandle, tokenizerHandle, opts

  describeSession: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'describeSession requires session handle' unless handle?
    @requireFunction('describeSession') handle

  describeTensorViews: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'describeTensorViews requires session handle' unless handle?
    @requireFunction('describeTensorViews') handle

  describeTypedTensorPlan: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'describeTypedTensorPlan requires session handle' unless handle?
    @requireFunction('describeTypedTensorPlan') handle

  constructSelectedResidentArrays: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'constructSelectedResidentArrays requires session handle' unless handle?
    @requireFunction('constructSelectedResidentArrays') handle

  describeSelectedResidentGroups: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'describeSelectedResidentGroups requires session handle' unless handle?
    @requireFunction('describeSelectedResidentGroups') handle

  runSelectedNormProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedNormProbe requires session handle' unless handle?
    @requireFunction('runSelectedNormProbe') handle

  runSelectedQuantizedProjectionProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedQuantizedProjectionProbe requires session handle' unless handle?
    @requireFunction('runSelectedQuantizedProjectionProbe') handle

  runSelectedLoraProjectionDeltaProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedLoraProjectionDeltaProbe requires session handle' unless handle?
    @requireFunction('runSelectedLoraProjectionDeltaProbe') handle

  runSelectedBasePlusLoraProjectionProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedBasePlusLoraProjectionProbe requires session handle' unless handle?
    @requireFunction('runSelectedBasePlusLoraProjectionProbe') handle

  runSelectedQNormAfterLoraProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedQNormAfterLoraProbe requires session handle' unless handle?
    @requireFunction('runSelectedQNormAfterLoraProbe') handle

  runSelectedRopeAfterQNormProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedRopeAfterQNormProbe requires session handle' unless handle?
    @requireFunction('runSelectedRopeAfterQNormProbe') handle

  runSelectedKvProjectionPathProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedKvProjectionPathProbe requires session handle' unless handle?
    @requireFunction('runSelectedKvProjectionPathProbe') handle

  runSelectedSingleTokenAttentionProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedSingleTokenAttentionProbe requires session handle' unless handle?
    @requireFunction('runSelectedSingleTokenAttentionProbe') handle

  runSelectedOProjectionPathProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedOProjectionPathProbe requires session handle' unless handle?
    @requireFunction('runSelectedOProjectionPathProbe') handle

  runSelectedPostAttentionResidualProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedPostAttentionResidualProbe requires session handle' unless handle?
    @requireFunction('runSelectedPostAttentionResidualProbe') handle

  runSelectedPostAttentionRmsNormProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedPostAttentionRmsNormProbe requires session handle' unless handle?
    @requireFunction('runSelectedPostAttentionRmsNormProbe') handle

  runSelectedMlpGateUpProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedMlpGateUpProbe requires session handle' unless handle?
    @requireFunction('runSelectedMlpGateUpProbe') handle

  runSelectedMlpActivationProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedMlpActivationProbe requires session handle' unless handle?
    @requireFunction('runSelectedMlpActivationProbe') handle

  runSelectedMlpDownProjectionProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedMlpDownProjectionProbe requires session handle' unless handle?
    @requireFunction('runSelectedMlpDownProjectionProbe') handle

  runSelectedMlpResidualProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedMlpResidualProbe requires session handle' unless handle?
    @requireFunction('runSelectedMlpResidualProbe') handle

  runSelectedLayerOutputContractProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedLayerOutputContractProbe requires session handle' unless handle?
    @requireFunction('runSelectedLayerOutputContractProbe') handle

  runSelectedNextLayerHandoffProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedNextLayerHandoffProbe requires session handle' unless handle?
    @requireFunction('runSelectedNextLayerHandoffProbe') handle

  runSelectedLayer21ResidencyProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedLayer21ResidencyProbe requires session handle' unless handle?
    @requireFunction('runSelectedLayer21ResidencyProbe') handle

  runSelectedLayer21InputQkvProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedLayer21InputQkvProbe requires session handle' unless handle?
    @requireFunction('runSelectedLayer21InputQkvProbe') handle

  runSelectedFinalLogitsProbe: (session) ->
    handle = if typeof session is 'string' then session else session?.handle
    throw new Error 'runSelectedFinalLogitsProbe requires session handle' unless handle?
    @requireFunction('runSelectedFinalLogitsProbe') handle

  protocolStatus: ->
    @requireFunction('protocolStatus')()

  warmSession: (sessionHandle) ->
    @requireFunction('warmSession') sessionHandle

  generate: (sessionHandle, request = {}) ->
    @requireFunction('generate') sessionHandle, request

  freeSession: (sessionHandle) ->
    @requireFunction('freeSession') sessionHandle

  unloadAdapter: (adapterHandle) ->
    @requireFunction('unloadAdapter') adapterHandle

  unloadTokenizer: (tokenizerHandle) ->
    @requireFunction('unloadTokenizer') tokenizerHandle

  unloadModel: (modelHandle) ->
    @requireFunction('unloadModel') modelHandle

  lifecycle: ({modelDir, tokenizerDir, adapterDir, prompt = 'hello', maxTokens = 1, temperature = 0, topK = 0, topP = 1.0, seed = 1234, chat = false}) ->
    model = @loadModelResident modelDir
    tokenizer = @loadTokenizer tokenizerDir
    adapter = if adapterDir? then @loadAdapter(adapterDir) else null
    sessionOpts = {}
    sessionOpts.adapter = adapter.handle if adapter?.handle?
    session = @createSession model.handle, tokenizer.handle, sessionOpts
    warm = @warmSession session.handle
    generation = @generate session.handle,
      prompt: prompt
      max_tokens: maxTokens
      temperature: temperature
      top_k: topK
      top_p: topP
      seed: seed
      chat: chat

    cleanup = {}
    cleanup.free_session = @freeSession session.handle
    cleanup.unload_adapter = if adapter? then @unloadAdapter adapter.handle else null
    cleanup.unload_tokenizer = @unloadTokenizer tokenizer.handle
    cleanup.unload_model = @unloadModel model.handle

    {
      model
      tokenizer
      adapter
      session
      warm
      generation
      cleanup
    }

module.exports =
  GypsySessionApi: GypsySessionApi
  createApi: (opts = {}) -> new GypsySessionApi opts
