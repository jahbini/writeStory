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
    numbered = line.match /^\s*([1-5])(?!\d)[^A-Za-z\s]*\s*(.+?)\s*$/
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
  Object.keys(emotions).length >= 3

runOracleOnce = (S, modelDir, prompt) ->
  raw = S.callMLX 'generate',
     model: modelDir
     prompt: prompt

  parsed = extractJSON raw
  filtered = filterEmotions parsed
  {raw, parsed, filtered}

@step =
  desc: "Classify sqlite-backed stories with the emotion oracle"

  action: (S) ->
    promptPrefix = S.param 'prompt_prefix'
    promptSuffix = S.param 'prompt_suffix'
    batchSz = S.param 'batch_size'
    quantizedModelMemoKey = S.param 'quantized_model_memo_key', 'quantizedModelDir'
    modelDir = S.theLowdown(quantizedModelMemoKey)?.value ? S.param('model_dir') ? S.theLowdown('modelDir')?.value
    throw new Error "[oracle_ask_sqlite] Missing model_dir/quantized model path" unless modelDir?

    await S.need 'story_seed_ids'

    pendingStories = S.theLowdown('storiesMissingKag.jsonl')?.value
    throw new Error "[#{S.stepName}] storiesMissingKag.jsonl must be an array" unless Array.isArray pendingStories

    pending = pendingStories.slice 0, batchSz
    rejectRows = await S.peek 'kag_rejects', []
    rejectRows = [] unless Array.isArray rejectRows

    console.log "[oracle_ask_sqlite] pending:", pending.length
    console.log "[oracle_ask_sqlite] stories left after this batch:", Math.max(pendingStories.length - pending.length, 0)
    S.make 'kag_viewed', pending

    newStoryIds = []

    if pending.length is 0
      S.saveThis 'pipeline:shutdown',
        by: S.stepName
        reason: 'all stories have already been passed to the sqlite oracle'
        timestamp: new Date().toISOString()
      S.make 'new_story_ids', newStoryIds
      S.make 'kag_rejects', rejectRows
      S.done()
      return

    outRejects = rejectRows.slice()

    for story in pending
      storyID = story?.story_id
      title = story?.title ? storyID
      text = story?.text ? ''
      continue unless storyID?

      newStoryIds.push storyID

      prompt = "#{promptPrefix}#{text}#{promptSuffix}"
      attempt1 = runOracleOnce S, modelDir, prompt
      finalAttempt = attempt1

      unless isUsableEmotionList(attempt1.filtered)
        console.log "[oracle_ask_sqlite] retrying #{storyID} after filter rejection"
        attempt2 = runOracleOnce S, modelDir, prompt
        finalAttempt = attempt2
        unless isUsableEmotionList(attempt2.filtered)
          console.error "[oracle_ask_sqlite] FAILED #{storyID} oracle did not produce a usable filtered emotion list after retry"
          outRejects.push
            story_id: storyID
            title: title
            prompt: prompt
            raw_attempt_1: attempt1.raw
            parsed_attempt_1: attempt1.parsed
            filtered_attempt_1: attempt1.filtered
            raw_attempt_2: attempt2.raw
            parsed_attempt_2: attempt2.parsed
            filtered_attempt_2: attempt2.filtered

      entries = []
      keywords = []

      for own keyword, headline of finalAttempt.filtered
        entries.push
          meta:
            doc_id: storyID
            paragraph_index: '001'
            title: title
          keyword: keyword
          headline: headline
        keywords.push keyword

      S.saveThis "kagFor{#{storyID}}.json",
        story_id: storyID
        entries: entries
        keywords: keywords

      console.log "[oracle_ask_sqlite] tagged #{storyID}"

    S.make 'new_story_ids', newStoryIds
    S.make 'kag_rejects', outRejects
    S.done()
    return
