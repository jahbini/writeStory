#!/usr/bin/env coffee

path = require 'node:path'
{ readFileSync } = require 'node:fs'
{ createRustyNativeApi } = require '../native_session_api'

modelDir = process.env.RUSTY_MODEL_DIR ? path.join __dirname, '..', '..', 'pipes', 'Qwen_Qwen3-4B-Instruct-2507', 'build', 'model4'
argv = process.argv.slice 2
readFlag = (name, defaultValue = null) ->
  index = argv.indexOf name
  return defaultValue if index < 0
  value = argv[index + 1]
  argv.splice index, if value? and value.indexOf('--') isnt 0 then 2 else 1
  if value? and value.indexOf('--') isnt 0 then value else 'true'

chatMode = String(readFlag('--chat', 'false')).toLowerCase() in ['1', 'true', 'yes', 'on']
systemPrompt = readFlag('--system', '')
mode = argv[0] ? 'greedy'
if mode not in ['greedy', 'sampled_top_k']
  # Sampling controls are surfaced for the API shape, but native generation is
  # still intentionally limited to top-k; top-p remains disabled.
  throw new Error 'mode must be greedy or sampled_top_k'
prompt = argv[1] ? 'hello'
generatedCount = Number.parseInt(argv[2] ? '8', 10)
temperature = Number argv[3] ? '0'
topK = Number.parseInt(argv[4] ? '0', 10)
seed = Number.parseInt(argv[5] ? '1234', 10)
topP = Number argv[6] ? '1.0'
stopTokenIds = String(argv[7] ? '').split(',').filter((part) -> part.length).map((part) -> Number.parseInt(part, 10))
streamMode = String(argv[8] ? '').toLowerCase() in ['stream', '--stream', '1', 'true', 'yes']
throw new Error 'generated token count must be a positive integer' unless Number.isInteger(generatedCount) and generatedCount > 0
throw new Error 'temperature must be 0 for greedy mode' if mode is 'greedy' and temperature isnt 0
throw new Error 'top_k must be 0 for greedy mode' if mode is 'greedy' and topK isnt 0
throw new Error 'temperature must be greater than 0 for sampled_top_k mode' if mode is 'sampled_top_k' and temperature <= 0
throw new Error 'top_k must be greater than 0 for sampled_top_k mode' if mode is 'sampled_top_k' and topK <= 0
throw new Error 'seed must be an integer' unless Number.isInteger seed
throw new Error 'top_p must be 1.0 for greedy mode' unless topP is 1
throw new Error 'stop token ids must be comma-separated integers' unless stopTokenIds.every((id) -> Number.isInteger(id) and id >= 0)

readEosTokenId = ->
  try
    config = JSON.parse readFileSync(path.join(modelDir, 'generation_config.json'), 'utf8')
    eos = config.eos_token_id ? config.eos_token_ids
    return eos[0] if Array.isArray(eos) and eos.length
    return eos if Number.isInteger(eos)
  catch _err
    null
  try
    config = JSON.parse readFileSync(path.join(modelDir, 'config.json'), 'utf8')
    eos = config.eos_token_id ? config.eos_token_ids
    return eos[0] if Array.isArray(eos) and eos.length
    return eos if Number.isInteger(eos)
  catch _err
    null
  null

readTokenizerConfig = ->
  try
    JSON.parse readFileSync(path.join(modelDir, 'tokenizer_config.json'), 'utf8')
  catch _err
    {}

qwenChatFallback = (rawPrompt, systemText = '') ->
  parts = []
  if String(systemText ? '').length
    parts.push "<|im_start|>system\n#{systemText}<|im_end|>"
  parts.push "<|im_start|>user\n#{rawPrompt}<|im_end|>"
  parts.push "<|im_start|>assistant\n"
  parts.join '\n'

formatChatPrompt = (rawPrompt, systemText = '') ->
  config = readTokenizerConfig()
  template = config.chat_template
  # Full Jinja chat-template execution is intentionally not implemented here.
  # If the model ships no template, use the standard Qwen message wrapper.
  return {
    formatted: qwenChatFallback rawPrompt, systemText
    chat_template_present: typeof template is 'string' and template.length > 0
    chat_template_applied: false
    chat_template_note: if template? then 'chat_template present but Jinja rendering is not implemented in this example' else 'chat_template not present; used Qwen fallback wrapper'
  }

main = ->
  api = createRustyNativeApi()
  model = null
  session = null
  tokenizer = null
  summary = null
  cleanup =
    session_freed: false
    tokenizer_unloaded: false
    model_unloaded: false
    bridge_shutdown_requested: false
  try
    model = await api.loadModel modelDir
    tokenizer = await api.loadTokenizer modelDir
    tokenizerInfo = await api.tokenizerInfo tokenizer
    session = await api.createSession model, tokenizer

    chatFormatted = if chatMode then formatChatPrompt(prompt, systemPrompt) else null
    formattedPrompt = if chatMode then chatFormatted.formatted else prompt
    promptTokenizationNote = 'formatted_prompt'
    fallbackUsed = false
    promptTokenIds = try
      await api.encodePrompt session, formattedPrompt
    catch err
      throw err unless chatMode
      fallbackUsed = true
      promptTokenizationNote = "formatted prompt was not encodable by bridge tokenizer; fell back to raw prompt: #{err?.message ? err}"
      await api.encodePrompt session, prompt
    imStartTokenIds = await api.encodePrompt session, '<|im_start|>'
    imEndTokenIds = await api.encodePrompt session, '<|im_end|>'
    eosTokenId = readEosTokenId()
    warm1 = await api.warmSession session

    generation = await api.generateTokens session, promptTokenIds, generatedCount,
      temperature: temperature
      topK: topK
      topP: topP
      seed: seed
      stopOnEos: eosTokenId?
      eosTokenId: eosTokenId ? 0
      stopTokenIds: stopTokenIds
    decodedGeneratedText = try
      await api.decodeTokens session, generation.generated_token_ids
    catch _err
      generation.decoded_generated_text
    decodedFormattedPrompt = try
      await api.decodeTokens session, promptTokenIds
    catch _err
      null
    tokenizerProbes = []
    for probeText in ['assistant', "assistant\n", 'hello', "hello\n", "<|im_start|>assistant\n"]
      probeIds = await api.encodePrompt session, probeText
      probeDecoded = try
        await api.decodeTokens session, probeIds
      catch _err
        null
      tokenizerProbes.push
        input_escaped: JSON.stringify(probeText)
        token_ids: probeIds
        decoded_text_escaped: if probeDecoded? then JSON.stringify(probeDecoded) else null
        roundtrip_preserves_newline: probeText.indexOf("\n") < 0 or probeDecoded?.indexOf("\n") >= 0

    if streamMode
      elapsedMs = 0
      for tokenId, index in generation.generated_token_ids
        elapsedMs += generation.per_token_incremental_ms?[index] ? 0
        decodedTokenText = try
          await api.decodeTokens session, [tokenId]
        catch _err
          String tokenId
        tokenEvent =
          event: 'token'
          index: index
          token_id: tokenId
          decoded_token_text: decodedTokenText
          elapsed_ms: elapsedMs
        if index is generation.generated_token_ids.length - 1 and generation.stopped
          tokenEvent.stopped = true
          tokenEvent.stop_reason = generation.stop_reason
          tokenEvent.stop_token_id = generation.stop_token_id
        console.log JSON.stringify tokenEvent

    summary =
      model: model
      session: session
      mode: mode
      stream: streamMode
      chat: chatMode
      raw_prompt: prompt
      formatted_prompt: formattedPrompt
      formatted_prompt_escaped: JSON.stringify(formattedPrompt)
      formatted_prompt_ends_with_assistant_newline: formattedPrompt.endsWith "<|im_start|>assistant\n"
      chat_template_present: chatFormatted?.chat_template_present ? false
      chat_template_applied: chatFormatted?.chat_template_applied ? false
      chat_template_note: chatFormatted?.chat_template_note ? null
      special_tokens_loaded: (tokenizerInfo.added_tokens_count ? 0) > 0
      im_start_token_id: imStartTokenIds[0] ? null
      im_end_token_id: imEndTokenIds[0] ? null
      fallback_used: fallbackUsed
      prompt_tokenization_note: promptTokenizationNote
      prompt_token_ids: promptTokenIds
      decoded_formatted_prompt: decodedFormattedPrompt
      decoded_formatted_prompt_escaped: if decodedFormattedPrompt? then JSON.stringify(decodedFormattedPrompt) else null
      tokenizer_probe: tokenizerProbes
      controls:
        max_tokens: generatedCount
        temperature: temperature
        top_k: topK
        top_p: topP
        top_p_status: 'stubbed_disabled'
        stop_token_ids: stopTokenIds
        eos_token_id: eosTokenId
      seed: seed
      warmup_ms: warm1.warmup_ms
      warmup_reused: warm1.reused
      generated_token_ids: generation.generated_token_ids
      decoded_generated_text: decodedGeneratedText
      generation_timing_ms: generation.generation_1_total_ms ? generation.total_generation_ms
      per_token_incremental_ms: generation.per_token_incremental_ms
      stopped: generation.stopped
      stop_reason: generation.stop_reason
      stop_token_id: generation.stop_token_id
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
    if summary?
      summary.cleanup = cleanup
      console.log JSON.stringify summary, null, 2

main().catch (err) ->
  console.error err?.stack ? String err
  process.exit 1
