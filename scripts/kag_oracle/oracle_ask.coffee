cleanFragment = (value) ->
  text = String(value ? '').trim()
  text = text.replace /^\*+|\*+$/g, ''
  text = text.replace /^["'“”]+|["'“”]+$/g, ''
  text.trim()

toEmotionKey = (value, fallbackIndex) ->
  text = cleanFragment(value).toLowerCase()
  text = text.replace /^#/, ''
  text = text.replace /#/g, '_'
  text = text.replace /[^a-z0-9]+/g, '_'
  text = text.replace /^_+|_+$/g, ''
  text = "emotion_#{fallbackIndex}" unless text.length
  text

extractJSON = (raw) ->
  return {} unless raw?
  block = raw.match(/\{[\s\S]*\}/)?[0]
  if block?
    try
      return JSON.parse block
    catch
      null

  emotions = {}
  lines = String(raw).split /\r?\n/

  for line, idx in lines
    cleanedLine = String(line ? '').trim()
    continue unless cleanedLine.length
    continue if /^=+$/.test(cleanedLine)

    numbered = cleanedLine.match /^\s*(\d+)(?!\d)[^A-Za-z\s]*\s*(.+?)\s*$/
    continue unless numbered?

    ordinal = Number numbered[1]
    body = cleanFragment numbered[2]
    continue unless body.length

    strictHash = body.match /^#([A-Za-z0-9_-]+)\s*(?:---|—|–)\s*(.+?)\s*$/
    if strictHash?
      emotionKey = toEmotionKey strictHash[1], ordinal
      emotionText = cleanFragment strictHash[2]
      continue unless emotionText.length
      emotions[emotionKey] = emotionText
      continue

    looseStructured = body.match /^(.+?)\s*(?:---|—|–)\s*(.+?)\s*$/
    if looseStructured?
      emotionKey = toEmotionKey looseStructured[1], ordinal
      emotionText = cleanFragment looseStructured[2]
      continue unless emotionText.length
      emotions[emotionKey] = emotionText
      continue

    emotionKey = toEmotionKey body, ordinal
    emotions[emotionKey] = cleanFragment body

  emotions

filterEmotions = (emotions) ->
  return {} unless emotions? and typeof emotions is 'object'

  rejectPatterns = [
    /\bshort headline\b/i
    /\bfinal answer\b/i
    /\bnote\b/i
    /\bprompt\b/i
    /\bgeneration\b/i
    /\bpeak memory\b/i
    /\btokens-per-sec\b/i
    /\bno response\b/i
    /\bi(?:'| a)?m sorry\b/i
    /\bcan(?:not|'t)\b/i
    /\bmisunderstanding\b/i
    /\bclarify\b/i
    /\brequested content formatted\b/i
    /\bplaceholder\b/i
  ]

  filtered = {}
  seenValues = new Set()

  for own key, value of emotions
    emotionKey = toEmotionKey key, Object.keys(filtered).length + 1
    emotionText = cleanFragment value
    continue unless emotionText.length
    continue if rejectPatterns.some (pattern) -> pattern.test(emotionKey) or pattern.test(emotionText)
    continue if /^emotion_\d+$/.test(emotionKey) and /^(disturbing|amused|nostalgic|speculative|serene|joyous|blissful|melancholic|absurd)$/i.test(emotionText)
    dedupeKey = "#{emotionKey}|#{emotionText.toLowerCase()}"
    continue if seenValues.has dedupeKey
    seenValues.add dedupeKey
    filtered[emotionKey] = emotionText

  filtered

isUsableEmotionList = (emotions) ->
  return false unless emotions? and typeof emotions is 'object'
  keys = Object.keys emotions
  return false if keys.length < 1
  true

runOracleOnce = (S, modelDir, prompt, adapterPath, debugMlx = false) ->
  args =
    model: modelDir
    prompt: prompt

  args["adapter-path"] = adapterPath if adapterPath?

  raw = S.callMLX 'generate', args, debugMlx

  parsed = extractJSON raw
  filtered = filterEmotions parsed
  {raw, parsed, filtered}

renderPrompt = (template, text) ->
  throw new Error "oracle prompt_text must be a string" unless typeof template is 'string'
  throw new Error "oracle prompt_text must contain a {...} insertion marker" unless /\{[^}]*\}/.test(template)
  template.replace /\{[^}]*\}/, String(text ? '')

@step =
  desc: "Classify a batch of untagged markdown segments with the emotion oracle"

  action: (S) ->
    promptText = S.param 'prompt_text'
    batchSz = S.param 'batch_size'
    quantizedModelMemoKey = S.param 'quantized_model_memo_key', 'quantizedModelDir'
    adapterPath = S.param 'adapter_path', null
    modelDir = S.theLowdown(quantizedModelMemoKey)?.value ? S.param('model_dir') ? S.theLowdown('modelDir')?.value
    throw new Error "[oracle_ask] Missing model_dir/quantized model path" unless modelDir?
    segments = await S.need 'marshalled_stories'
    taggedRows = await S.peek 'kag_emotions', []
    rejectRows = await S.peek 'kag_rejects', []

    throw new Error "[#{S.stepName}] marshalled_stories must be an array" unless Array.isArray(segments)
    taggedRows = [] unless Array.isArray(taggedRows)
    rejectRows = [] unless Array.isArray(rejectRows)

    tagged = new Set()
    for row in taggedRows when row?.meta?
      tagged.add "#{row.meta.doc_id}|#{row.meta.paragraph_index}"

    remaining = 0
    for segment in segments
      key = "#{segment.meta?.doc_id}|#{segment.meta?.paragraph_index}"
      continue if tagged.has key
      remaining += 1

    pending = []
    for segment in segments
      key = "#{segment.meta?.doc_id}|#{segment.meta?.paragraph_index}"
      continue if tagged.has key
      pending.push segment
      break if pending.length >= batchSz

    console.log "[oracle_ask] pending:", pending.length
    console.log "[oracle_ask] stories left after this batch:", Math.max(remaining - pending.length, 0)
    S.make 'kag_viewed', pending

    newStoryIdSet = new Set()
    for segment in pending
      title = segment?.meta?.title
      continue unless title?
      newStoryIdSet.add title
    newStoryIds = Array.from newStoryIdSet

    if pending.length is 0
      console.error "JIM BAD EXIT"
      S.saveThis 'pipeline:shutdown',
        by: S.stepName
        reason: 'all stories have already been passed to the oracle'
        timestamp: new Date().toISOString()
      S.make 'new_story_ids', newStoryIds
      S.make 'kag_emotions', taggedRows
      S.make 'kag_rejects', rejectRows
      S.done()
      return

    outRows = taggedRows.slice()
    outRejects = rejectRows.slice()

    for segment in pending
      text = segment.text ? ''
      meta = segment.meta ? {}
      prompt = renderPrompt promptText, text
      attempt1 = runOracleOnce S, modelDir, prompt, adapterPath
      finalAttempt = attempt1

      unless isUsableEmotionList(attempt1.filtered)
        console.log "[oracle_ask] retrying #{meta.doc_id} #{meta.paragraph_index} after filter rejection"
        attempt2 = runOracleOnce S, modelDir, prompt, adapterPath, true
        finalAttempt = attempt2
        unless isUsableEmotionList(attempt2.filtered)
          console.error "[oracle_ask] FAILED #{meta.doc_id} #{meta.paragraph_index} oracle did not produce a usable filtered emotion list after retry"
          outRejects.push
            meta:
              doc_id: meta.doc_id
              paragraph_index: meta.paragraph_index
              title: meta.title
            prompt: prompt
            raw_attempt_1: attempt1.raw
            parsed_attempt_1: attempt1.parsed
            filtered_attempt_1: attempt1.filtered
            raw_attempt_2: attempt2.raw
            parsed_attempt_2: attempt2.parsed
            filtered_attempt_2: attempt2.filtered

      outRows.push
        meta:
          doc_id: meta.doc_id
          paragraph_index: meta.paragraph_index
        emotions: finalAttempt.filtered

      console.log "JIM Emotions from", meta.doc_id, finalAttempt.raw
      console.log "[oracle_ask] tagged #{meta.doc_id} #{meta.paragraph_index}"

    S.make 'new_story_ids', newStoryIds
    S.make 'kag_emotions', outRows
    S.make 'kag_rejects', outRejects
    S.done()
    return
