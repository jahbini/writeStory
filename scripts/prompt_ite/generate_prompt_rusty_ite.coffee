path = require 'node:path'
{ readFileSync } = require 'node:fs'
{ createRustyNativeApi } = require '../../rusty/native_session_api'

repoRoot = path.resolve __dirname, '..', '..'
rustyRoot = path.join repoRoot, 'rusty'
rustyBridgeManifest = path.join rustyRoot, 'bridge', 'Cargo.toml'

cleanGeneratedText = (_prompt, cleanedOutput) ->
  text = String(cleanedOutput ? '').trim()
  return '' unless text.length

  specialIndex = text.search /<\|(?:im_start|im_end|endoftext)\|>/
  text = text.slice(0, specialIndex).trim() if specialIndex >= 0

  lines = text.split /\r?\n/
  lines = lines.filter (line) ->
    trimmed = line.trim()
    return false if /^=+$/.test trimmed
    return false if /^Prompt:\s+\d+\s+tokens/.test trimmed
    return false if /^Generation:\s+\d+\s+tokens/.test trimmed
    return false if /^Peak memory:\s+/.test trimmed
    true

  lines.join("\n").trim()

resolveRunTag = (L) ->
  raw = process.env.HH_MM ? L.theLowdown('env/HH_MM')?.value ? null
  return null unless raw?
  text = String(raw).trim()
  text = text.replace(/^"+|"+$/g, '')
  text = text.replace(/^'+|'+$/g, '')
  return null unless text.length
  text

buildDiaryRecord = (prompt, text) ->
  [
    "Prompt:"
    prompt
    ""
    "Generation:"
    text
    ""
  ].join "\n"

readJson = (filePath) ->
  try
    JSON.parse readFileSync(filePath, 'utf8')
  catch _err
    null

readEosTokenId = (modelDir) ->
  for filename in ['generation_config.json', 'config.json']
    config = readJson path.join(modelDir, filename)
    continue unless config?
    eos = config.eos_token_id ? config.eos_token_ids
    return eos[0] if Array.isArray(eos) and eos.length
    return eos if Number.isInteger(eos)
  null

readTokenizerConfig = (modelDir) ->
  readJson(path.join(modelDir, 'tokenizer_config.json')) ? {}

resolveModelDir = (modelDir) ->
  text = String(modelDir ? '').trim()
  return text if path.isAbsolute text
  path.resolve process.cwd(), text

qwenChatFallback = (rawPrompt, systemText = '') ->
  parts = []
  if String(systemText ? '').length
    parts.push "<|im_start|>system\n#{systemText}<|im_end|>"
  parts.push "<|im_start|>user\n#{rawPrompt}<|im_end|>"
  parts.push "<|im_start|>assistant\n"
  parts.join '\n'

formatChatPrompt = (modelDir, rawPrompt, systemText = '') ->
  config = readTokenizerConfig modelDir
  template = config.chat_template
  formatted: qwenChatFallback rawPrompt, systemText
  chat_template_present: typeof template is 'string' and template.length > 0
  chat_template_applied: false
  chat_template_note: if template? then 'chat_template present but Jinja rendering is not implemented in this script' else 'chat_template not present; used Qwen fallback wrapper'

firstDefined = (values...) ->
  for value in values
    return value if value?
  null

boolParam = (value, defaultValue = false) ->
  return defaultValue unless value?
  return value if typeof value is 'boolean'
  String(value).toLowerCase() in ['1', 'true', 'yes', 'on']

intParam = (value, defaultValue) ->
  parsed = Number.parseInt String(value ? defaultValue), 10
  return defaultValue unless Number.isFinite(parsed)
  parsed

numberParam = (value, defaultValue) ->
  parsed = Number value ? defaultValue
  return defaultValue unless Number.isFinite(parsed)
  parsed

buildControls = (L) ->
  mlx = L.param('mlx', {}) ? {}
  temperature = numberParam firstDefined(mlx.temperature, mlx.temp), 0
  topKDefault = if temperature > 0 then 40 else 0
  controls =
    max_tokens: intParam firstDefined(mlx.max_tokens, mlx['max-tokens'], mlx.maxTokens), 256
    temperature: temperature
    top_k: intParam firstDefined(mlx.top_k, mlx['top-k'], mlx.topK), topKDefault
    top_p: numberParam firstDefined(mlx.top_p, mlx['top-p'], mlx.topP), 1.0
    seed: intParam firstDefined(mlx.seed, L.param('seed', null)), 1234
    chat: boolParam L.param('chat', true), true
    system_prompt: L.param('system_prompt', '')

  throw new Error "[#{L.stepName}] rusty generation currently supports top_p only when disabled at 1.0" unless Number(controls.top_p) is 1
  throw new Error "[#{L.stepName}] top_k must be greater than 0 when temperature > 0" if Number(controls.temperature) > 0 and Number(controls.top_k) <= 0
  controls

resolvePrompt = (L) ->
  ownPrompt = L.param 'prompt_text', null
  return String(ownPrompt ? '').trim() if ownPrompt?

  sharedPrompt = L.getStepParam 'generate_prompt_ite', 'prompt_text', null
  return String(sharedPrompt ? '').trim() if sharedPrompt?

  ''

@step =
  desc: "Generate text from a UI-supplied prompt using Rusty native generation"

  action: (L) ->
    prompt = resolvePrompt L
    throw new Error "[#{L.stepName}] generate_prompt_ite.prompt_text must be a non-empty string" unless prompt.length

    modelDir = L.param 'quantized_model_dir', null
    outputPrefix = String(L.param('output_file_prefix', 'prompt_rusty') ? 'prompt_rusty').trim() or 'prompt_rusty'
    throw new Error "[#{L.stepName}] Missing quantized_model_dir param" unless modelDir?
    modelDir = resolveModelDir modelDir

    controls = buildControls L
    formattedInfo = if controls.chat then formatChatPrompt(modelDir, prompt, controls.system_prompt) else null
    formattedPrompt = if controls.chat then formattedInfo.formatted else prompt
    eosTokenId = readEosTokenId modelDir

    api = createRustyNativeApi
      cwd: rustyRoot
      manifestPath: rustyBridgeManifest
    model = null
    tokenizer = null
    session = null
    cleanup =
      session_freed: false
      tokenizer_unloaded: false
      model_unloaded: false
      bridge_shutdown_requested: false

    meta = null
    rawRecord = null
    text = ''
    try
      model = await api.loadModel modelDir
      tokenizer = await api.loadTokenizer modelDir
      tokenizerInfo = await api.tokenizerInfo tokenizer
      session = await api.createSession model, tokenizer

      promptTokenIds = await api.encodePrompt session, formattedPrompt
      console.log "[generate_prompt_rusty_ite] prompt chars:", prompt.length
      console.log "[generate_prompt_rusty_ite] prompt tokens:", promptTokenIds.length
      warm = await api.warmSession session
      console.log "[generate_prompt_rusty_ite] warmup ms:", warm.warmup_ms
      console.log "[generate_prompt_rusty_ite] generating tokens:", controls.max_tokens
      generation = await api.generateTokens session, promptTokenIds, controls.max_tokens,
        temperature: controls.temperature
        topK: controls.top_k
        topP: controls.top_p
        seed: controls.seed
        stopOnEos: eosTokenId?
        eosTokenId: eosTokenId ? 0

      rawDecodedText = try
        await api.decodeTokens session, generation.generated_token_ids,
          skipSpecialTokens: false
          diagnostics: false
      catch _err
        generation.decoded_generated_text
      cleanedDecoded = try
        await api.decodeTokens session, generation.generated_token_ids,
          skipSpecialTokens: true
          diagnostics: true
          returnFull: true
      catch _err
        text: rawDecodedText
        diagnostics: []

      text = cleanGeneratedText formattedPrompt, cleanedDecoded.text
      meta =
        mode: 'prompt_rusty'
        model_dir: modelDir
        attention_backend_default: generation.attention_backend_default
        attention_backend_active: generation.attention_backend_active
        expanded_kv_cache: generation.expanded_kv_cache
        experimental_mlx_attention_enabled: generation.experimental_mlx_attention_enabled
        experimental_mlx_attention_mode: generation.experimental_mlx_attention_mode
        chat: controls.chat
        chat_template_present: formattedInfo?.chat_template_present ? false
        chat_template_applied: formattedInfo?.chat_template_applied ? false
        chat_template_note: formattedInfo?.chat_template_note ? null
        prompt_chars: prompt.length
        formatted_prompt_chars: String(formattedPrompt ? '').length
        prompt_token_count: promptTokenIds.length
        generated_token_count: generation.generated_token_ids?.length ? 0
        text_chars: text.length
        controls: controls
        eos_token_id: eosTokenId
        tokenizer_added_tokens_count: tokenizerInfo.added_tokens_count
        warmup_ms: warm.warmup_ms
        warmup_reused: warm.reused
        generation_timing_ms: generation.generation_1_total_ms ? generation.total_generation_ms
        tokens_per_second: if (generation.generation_1_total_ms ? generation.total_generation_ms) > 0 then ((generation.generated_token_ids?.length ? 0) * 1000.0) / (generation.generation_1_total_ms ? generation.total_generation_ms) else 0
        per_token_incremental_ms: generation.per_token_incremental_ms
        stopped: generation.stopped
        stop_reason: generation.stop_reason
        stop_token_id: generation.stop_token_id
        fallback_used: generation.fallback_used
        fallback_steps_per_token: generation.fallback_steps_per_token
        readback_count: generation.readback_count
        readback_reasons: generation.readback_reasons
        last_token_backend_report: generation.last_token_backend_report
        cleanup: cleanup
      rawRecord =
        prompt: prompt
        formatted_prompt: formattedPrompt
        prompt_token_ids: promptTokenIds
        generated_token_ids: generation.generated_token_ids
        raw_decoded_text: rawDecodedText
        cleaned_decoded_text: cleanedDecoded.text
        decoded_generated_text: text
        decode_diagnostics_tail: (cleanedDecoded.diagnostics ? []).slice -24
        meta: meta

      console.log "[generate_prompt_rusty_ite] text chars:", text.length
      console.log "[generate_prompt_rusty_ite] backend:", meta.attention_backend_active
      console.log "[generate_prompt_rusty_ite] generation ms:", meta.generation_timing_ms
      console.log "[generate_prompt_rusty_ite] tokens/sec:", meta.tokens_per_second
      console.log "[generate_prompt_rusty_ite] stop:", meta.stop_reason, meta.stop_token_id ? ''
    finally
      if session?
        try
          await api.freeSession session
          cleanup.session_freed = true
        catch _err
          null
      if tokenizer?
        try
          await api.unloadTokenizer tokenizer
          cleanup.tokenizer_unloaded = true
        catch _err
          null
      if model?
        try
          await api.unloadModel model
          cleanup.model_unloaded = true
        catch _err
          null
      await api.shutdown()
      cleanup.bridge_shutdown_requested = true
      meta.cleanup = cleanup if meta?

    if meta?
      L.make 'prompt_rusty_raw', JSON.stringify(rawRecord, null, 2)
      L.make 'prompt_rusty_meta', meta
      L.make 'prompt_rusty_text', text

      runTag = resolveRunTag L
      if typeof runTag is 'string' and runTag.length
        L.saveThis "out/#{outputPrefix}_#{runTag}.txt", text
        L.saveThis "diary/#{outputPrefix}_#{runTag}.txt", buildDiaryRecord(prompt, text)

    L.done()
    return
