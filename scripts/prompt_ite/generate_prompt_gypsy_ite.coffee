path = require 'node:path'
{ readFileSync } = require 'node:fs'
{ createApi } = require '../../gypsy/session_api'

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
  temperature = numberParam firstDefined(mlx.temperature, mlx.temp), 0.7
  topKDefault = if temperature > 0 then 40 else 0
  controls =
    max_tokens: intParam firstDefined(mlx.max_tokens, mlx['max-tokens'], mlx.maxTokens), 256
    temperature: temperature
    top_k: intParam firstDefined(mlx.top_k, mlx['top-k'], mlx.topK), topKDefault
    top_p: numberParam firstDefined(mlx.top_p, mlx['top-p'], mlx.topP), 1.0
    seed: intParam firstDefined(mlx.seed, L.param('seed', null)), 1234
    chat: boolParam L.param('chat', true), true
    system_prompt: L.param('system_prompt', '')
  controls

resolveAdapterDir = (L) ->
  mlx = L.param('mlx', {}) ? {}
  sharedMlx = L.getStepParam('generate_prompt_ite', 'mlx', {}) ? {}
  adapterPath = firstDefined(
    L.param('adapter_path', null),
    L.param('adapter_dir', null),
    mlx.adapter_path,
    mlx.adapterPath,
    mlx.adapter,
    mlx.adapter_dir,
    sharedMlx.adapter_path,
    sharedMlx.adapterPath,
    sharedMlx.adapter,
    sharedMlx.adapter_dir
  )
  return null unless adapterPath?
  adapterPath = String(adapterPath).trim()
  return null unless adapterPath.length
  if path.isAbsolute adapterPath
    adapterPath
  else
    path.resolve adapterPath

resolvePrompt = (L) ->
  ownPrompt = L.param 'prompt_text', null
  return String(ownPrompt ? '').trim() if ownPrompt?

  sharedPrompt = L.getStepParam 'generate_prompt_ite', 'prompt_text', null
  return String(sharedPrompt ? '').trim() if sharedPrompt?

  ''

@step =
  desc: "Generate text from a UI-supplied prompt using the gypsy MLX-Metal native addon"

  action: (L) ->
    prompt = resolvePrompt L
    throw new Error "[#{L.stepName}] prompt_text must be a non-empty string" unless prompt.length

    modelDir = L.param 'quantized_model_dir', null
    outputPrefix = String(L.param('output_file_prefix', 'prompt_gypsy') ? 'prompt_gypsy').trim() or 'prompt_gypsy'
    throw new Error "[#{L.stepName}] Missing quantized_model_dir param" unless modelDir?
    modelDir = resolveModelDir modelDir
    tokenizerDir = resolveModelDir(L.param('tokenizer_dir', modelDir) ? modelDir)

    controls = buildControls L
    adapterDir = resolveAdapterDir L
    # When chat=true, let the native gypsy addon apply chat formatting (matches the
    # behavior of test/helpers/native_64_mlx_lazy_generation_probe.coffee).
    formattedInfo = if controls.chat then formatChatPrompt(modelDir, prompt, controls.system_prompt) else null

    api = createApi()

    console.log "[generate_prompt_gypsy_ite] prompt chars:", prompt.length
    console.log "[generate_prompt_gypsy_ite] model dir:", modelDir
    console.log "[generate_prompt_gypsy_ite] adapter:", adapterDir ? '(none)'
    console.log "[generate_prompt_gypsy_ite] generating tokens:", controls.max_tokens

    started = Date.now()
    result = api.lifecycle
      modelDir: modelDir
      tokenizerDir: tokenizerDir
      adapterDir: adapterDir
      prompt: prompt
      maxTokens: controls.max_tokens
      temperature: controls.temperature
      topK: controls.top_k
      topP: controls.top_p
      seed: controls.seed
      chat: controls.chat
    elapsedMs = Date.now() - started

    generation = result.generation ? {}
    timing = generation.timing ? {}
    logitsSummary = generation.logits_summary ? {}
    rawDecoded = String(generation.raw_decoded_text ? '')
    cleanedDecoded = String(generation.cleaned_decoded_text ? rawDecoded)
    text = cleanGeneratedText prompt, cleanedDecoded

    meta =
      mode: 'prompt_gypsy'
      model_dir: modelDir
      tokenizer_dir: tokenizerDir
      adapter_requested: adapterDir?
      adapter_dir: adapterDir
      # Native-side adapter diagnostics — these tell us whether the adapter
      # is actually being applied during generation, not just requested.
      # adapter_active is true only when the LoRA delta was applied to at
      # least one projection; adapter_applied_projection_count is the count
      # of per-layer-per-token projection calls that actually found and used
      # lora_a/lora_b tensors. If requested-but-not-active, the adapter
      # tensor names did not match what the native loader looks up.
      adapter_active: logitsSummary.adapter_active
      adapter_applied_projection_count: logitsSummary.adapter_applied_projection_count
      adapter_scale: logitsSummary.adapter_scale
      adapter_layer_range: logitsSummary.adapter_layer_range
      adapter_applied_to_prefill: logitsSummary.adapter_applied_to_prefill
      adapter_applied_to_decode: logitsSummary.adapter_applied_to_decode
      chat: controls.chat
      chat_template_present: formattedInfo?.chat_template_present ? false
      chat_template_applied: formattedInfo?.chat_template_applied ? false
      chat_template_note: formattedInfo?.chat_template_note ? null
      prompt_chars: prompt.length
      formatted_prompt_chars: prompt.length
      generated_token_count: generation.generated_token_count
      generated_token_ids: generation.generated_token_ids
      text_chars: text.length
      controls: controls
      generation_status: generation.status
      stop: generation.stop
      generation_timing_ms: timing.generation_timing_ms
      tokens_per_second: timing.tokens_per_second
      attention_backend: timing.active_timing_buckets?.attention_backend
      kv_cache_backend: timing.active_timing_buckets?.kv_cache_backend
      timing_buckets: timing.active_timing_buckets
      total_elapsed_ms: elapsedMs
      cleanup:
        freed_session: result.cleanup?.free_session?.freed ? false
        unloaded_model: result.cleanup?.unload_model?.unloaded ? false
        unloaded_tokenizer: result.cleanup?.unload_tokenizer?.unloaded ? false
        unloaded_adapter: result.cleanup?.unload_adapter?.unloaded ? null

    rawRecord =
      prompt: prompt
      generated_token_ids: generation.generated_token_ids
      raw_decoded_text: rawDecoded
      cleaned_decoded_text: cleanedDecoded
      decoded_generated_text: text
      meta: meta

    console.log "[generate_prompt_gypsy_ite] text chars:", text.length
    console.log "[generate_prompt_gypsy_ite] generated tokens:", meta.generated_token_count
    console.log "[generate_prompt_gypsy_ite] generation ms:", meta.generation_timing_ms
    console.log "[generate_prompt_gypsy_ite] tokens/sec:", meta.tokens_per_second
    console.log "[generate_prompt_gypsy_ite] backend:", meta.attention_backend
    console.log "[generate_prompt_gypsy_ite] stop:", meta.stop?.stop_reason

    L.make 'prompt_gypsy_raw', JSON.stringify(rawRecord, null, 2)
    L.make 'prompt_gypsy_meta', meta
    L.make 'prompt_gypsy_text', text

    runTag = resolveRunTag L
    if typeof runTag is 'string' and runTag.length
      L.saveThis "out/#{outputPrefix}_#{runTag}.txt", text
      L.saveThis "diary/#{outputPrefix}_#{runTag}.txt", buildDiaryRecord(prompt, text)

    L.done()
    return
