{ spawn, spawnSync } = require 'node:child_process'
{ createInterface } = require 'node:readline'
path = require 'node:path'

manifestPath = path.join __dirname, 'bridge', 'Cargo.toml'

class RustyNativeSessionApi
  constructor: (opts = {}) ->
    @manifestPath = opts.manifestPath ? manifestPath
    @cwd = opts.cwd ? __dirname
    @build = opts.build ? true
    @bridge = null
    @rl = null
    @pending = new Map()
    @sessionTokenizers = new Map()
    @nextId = 0

  start: ->
    return if @bridge?
    if @build
      built = spawnSync 'cargo', ['build', '--manifest-path', @manifestPath],
        stdio: 'inherit'
      throw new Error "cargo build failed with status #{built.status}" unless built.status is 0

    @bridge = spawn 'cargo', ['run', '--quiet', '--manifest-path', @manifestPath],
      cwd: @cwd
      stdio: ['pipe', 'pipe', 'pipe']

    @rl = createInterface input: @bridge.stdout
    @bridge.stderr.on 'data', (buf) ->
      text = String(buf ? '').trim()
      process.stderr.write "#{text}\n" if text.length

    @rl.on 'line', (line) =>
      return unless line.trim().length
      payload = JSON.parse line
      waiter = @pending.get payload.id
      return unless waiter?
      @pending.delete payload.id
      waiter.resolve payload

    @bridge.on 'exit', (code, signal) =>
      for [_id, waiter] from @pending
        waiter.reject new Error "bridge exited before reply code=#{code} signal=#{signal}"
      @pending.clear()

  send: (cmd, args = {}) ->
    @start()
    @nextId += 1
    request =
      id: String @nextId
      cmd: cmd
      args: args

    new Promise (resolve, reject) =>
      @pending.set request.id, { resolve, reject }
      @bridge.stdin.write "#{JSON.stringify request}\n", 'utf8'

  value: (cmd, args = {}) ->
    response = await @send cmd, args
    unless response.ok is true
      code = response.error?.code ? 'bridge_error'
      message = response.error?.message ? JSON.stringify response
      throw new Error "#{cmd} failed: #{code}: #{message}"
    response.value

  loadModel: (modelPath) ->
    value = await @value 'load_model_native', path: modelPath
    value.model

  unloadModel: (model) ->
    await @value 'unload_model_native', model: model

  loadTokenizer: (tokenizerPath) ->
    value = await @value 'load_tokenizer', path: tokenizerPath
    value.tokenizer

  unloadTokenizer: (tokenizer) ->
    await @value 'unload_tokenizer', tokenizer: tokenizer

  tokenizerInfo: (tokenizer) ->
    await @value 'tokenizer_info', tokenizer: tokenizer

  createSession: (model, tokenizer = null, opts = {}) ->
    args = model: model
    args.tokenizer = tokenizer if tokenizer?
    adapterPath = opts.adapterPath ? opts.adapter_path
    args.adapter_path = adapterPath if adapterPath?
    value = await @value 'create_native_session', args
    @sessionTokenizers.set value.session, tokenizer if tokenizer?
    value

  tokenizerFor: (target) ->
    target = target.session if target? and typeof target is 'object' and typeof target.session is 'string'
    return target if typeof target is 'string' and target.indexOf('tok:') is 0
    tokenizer = @sessionTokenizers.get target
    throw new Error "no tokenizer associated with #{target}" unless tokenizer?
    tokenizer

  encodePrompt: (target, text) ->
    tokenizer = @tokenizerFor target
    value = await @value 'tokenizer_encode',
      tokenizer: tokenizer
      text: text
    value.tokens

  decodeTokens: (target, tokenIds, opts = {}) ->
    tokenizer = @tokenizerFor target
    value = await @value 'tokenizer_decode',
      tokenizer: tokenizer
      tokens: tokenIds
      skip_special_tokens: opts.skipSpecialTokens ? opts.skip_special_tokens ? false
      diagnostics: opts.diagnostics ? false
    if opts.returnFull ? false then value else value.text

  warmSession: (session) ->
    sessionHandle = if typeof session is 'string' then session else session.session
    await @value 'warm_resident_session', session: sessionHandle

  generateTokens: (session, promptTokenIds, n, opts = {}) ->
    sessionHandle = if typeof session is 'string' then session else session.session
    temperature = opts.temperature ? 0
    topK = opts.topK ? opts.top_k ? 0
    topP = opts.topP ? opts.top_p ? 1.0
    seed = opts.seed ? 1234
    throw new Error 'native generateTokens supports top_p only when disabled at 1.0' unless Number(topP) is 1
    throw new Error 'native generateTokens requires top_k > 0 when temperature > 0' if Number(temperature) > 0 and Number(topK) <= 0
    await @value 'generate_tokens',
      session: sessionHandle
      prompt_token_ids: promptTokenIds
      first_decode_token_id: opts.firstDecodeTokenId ? promptTokenIds[promptTokenIds.length - 1] ? 15
      generated_tokens: n
      temperature: temperature
      top_k: topK
      top_p: topP
      seed: seed
      stop_on_eos: opts.stopOnEos ? opts.stop_on_eos ? false
      eos_token_id: opts.eosTokenId ? opts.eos_token_id ? 0
      stop_token_ids: opts.stopTokenIds ? opts.stop_token_ids ? []
      sampling_mode: if Number(temperature) > 0 and Number(topK) > 0 then 'sampled_top_k' else 'greedy'

  freeSession: (session) ->
    sessionHandle = if typeof session is 'string' then session else session.session
    @sessionTokenizers.delete sessionHandle
    await @value 'free_native_session', session: sessionHandle

  shutdown: ->
    return unless @bridge?
    try
      await @send 'bridge_shutdown', {}
    catch _err
      # The process may already be exiting during cleanup.
      null
    @bridge = null

createRustyNativeApi = (opts = {}) ->
  new RustyNativeSessionApi opts

module.exports =
  RustyNativeSessionApi: RustyNativeSessionApi
  createRustyNativeApi: createRustyNativeApi
