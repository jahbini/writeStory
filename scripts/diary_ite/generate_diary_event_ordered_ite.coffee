cleanGeneratedText = (prompt, rawOutput) ->
  text = String(rawOutput ? '').trim()
  return '' unless text.length

  if text.indexOf(prompt) is 0
    text = text.slice(prompt.length).trim()

  lines = text.split /\r?\n/
  lines = lines.filter (line) ->
    trimmed = line.trim()
    return false if /^=+$/.test trimmed
    return false if /^Prompt:\s+\d+\s+tokens/.test trimmed
    return false if /^Generation:\s+\d+\s+tokens/.test trimmed
    return false if /^Peak memory:\s+/.test trimmed
    true

  lines.join("\n").trim()

coerceJSON = (value) ->
  return value unless typeof value is 'string'
  try
    JSON.parse value
  catch
    value

normalizeDiaryKag = (value) ->
  value = coerceJSON value
  return value if Array.isArray(value?.entries)

  if value? and typeof value is 'object' and not Array.isArray(value)
    if Array.isArray(value.value?.entries)
      return value.value
    if typeof value.entries is 'string'
      parsedEntries = coerceJSON value.entries
      if Array.isArray parsedEntries
        out = Object.assign {}, value
        out.entries = parsedEntries
        return out

  value

resolveEventMatches = (diaryKag, kind) ->
  row = diaryKag?.events?[kind]
  matches = row?.matches
  return [] unless Array.isArray matches
  matches

readArtifactTarget = (L, artifactKey) ->
  experiment = L.theLowdown('experiment.yaml')?.value ? {}
  targetKey = experiment?.artifacts?[artifactKey]?.target
  return undefined unless typeof targetKey is 'string'

  targetEntry = L.theLowdown targetKey
  targetValue = targetEntry?.value
  if targetValue is undefined
    if typeof targetEntry?.waitFor is 'function'
      targetValue = await targetEntry.waitFor()
    else if targetEntry?.notifier?
      targetValue = await targetEntry.notifier
  targetValue

resolveRunTag = (L) ->
  raw = process.env.HH_MM ? L.theLowdown('env/HH_MM')?.value ? null
  return null unless raw?
  text = String(raw).trim()
  text = text.replace(/^"+|"+$/g, '')
  text = text.replace(/^'+|'+$/g, '')
  return null unless text.length
  text

wordSet = (text) ->
  words = String(text ? '').toLowerCase().match(/[a-z0-9]+/g) ? []
  new Set(words)

scoreEntryForEvent = (entry, eventWords) ->
  return 0 unless entry? and typeof entry is 'object'
  score = 0
  keyword = String(entry.keyword ? '').trim().toLowerCase()
  headlineWords = wordSet String(entry.headline ? '')
  score += 4 if keyword.length and eventWords.has keyword
  score += 2 if entry.chunk_index?
  for word in headlineWords
    score += 1 if eventWords.has word
  score

describeEvent = (kind, event) ->
  return null unless event? and typeof event is 'object'
  text = String(event.text ? '').trim()
  location = String(event.location ? '').trim()
  character = String(event.character ? '').trim()
  theme = String(event.theme ? '').trim()
  lines = []
  lines.push "Event kind: #{kind}"
  lines.push "Event text: #{text}" if text.length
  lines.push "Location: #{location}" if location.length
  lines.push "Character: #{character}" if character.length
  lines.push "Theme: #{theme}" if theme.length
  return null unless lines.length
  lines.join "\n"

eventWordsFor = (kind, event) ->
  bits = [
    kind
    event?.text ? ''
    event?.location ? ''
    event?.character ? ''
    event?.theme ? ''
    event?.emotion ? ''
    (event?.emotions ? []).join(' ')
  ]
  wordSet bits.join ' '

pickEventKagEntries = (entries, kind, event, limit = 4) ->
  eventWords = eventWordsFor kind, event
  scored = []

  for entry, idx in (entries ? [])
    score = scoreEntryForEvent entry, eventWords
    continue unless score > 0
    scored.push
      idx: idx
      score: score
      entry: entry

  if scored.length is 0
    return (entries ? []).slice 0, Math.min(limit, (entries ? []).length)

  scored.sort (a, b) ->
    if b.score isnt a.score then b.score - a.score else a.idx - b.idx

  (row.entry for row in scored.slice(0, limit))

renderKagLines = (entries) ->
  lines = []
  for entry in (entries ? [])
    keyword = String(entry?.keyword ? '').trim()
    headline = String(entry?.headline ? '').trim()
    chunkIndex = entry?.chunk_index
    label = []
    label.push "chunk #{chunkIndex}" if chunkIndex?
    label.push keyword if keyword.length
    prefix = if label.length then "- #{label.join(' / ')}" else "- cue"
    if headline.length
      lines.push "#{prefix}: #{headline}"
    else
      lines.push prefix
  lines

renderSourcePassages = (matches) ->
  lines = []
  for match in (matches ? [])
    storyID = String(match?.story_id ? '').trim()
    chunkIndex = match?.chunk_index
    keyword = String(match?.keyword ? '').trim()
    headline = String(match?.headline ? '').trim()
    chunkText = String(match?.chunk_text ? '').trim()
    metaBits = []
    metaBits.push storyID if storyID.length
    metaBits.push "chunk #{chunkIndex}" if chunkIndex?
    metaBits.push keyword if keyword.length
    metaBits.push headline if headline.length
    lines.push "- #{metaBits.join(' / ')}"
    if chunkText.length
      for line in chunkText.split /\r?\n/
        lines.push "  #{line}"
  lines

buildEventPrompt = (storyID, kind, event, chosenEntries, chosenMatches, priorSections, mode) ->
  eventBlock = describeEvent kind, event
  kagLines = renderKagLines chosenEntries
  sourceLines = renderSourcePassages chosenMatches
  historyText = priorSections.map((section) -> "#{section.kind}:\n#{section.text}").join "\n\n"
  transitionRule = null
  if mode is 'adapter' and priorSections.length > 0
    transitionRule = "- Transition naturally from the previous diary section into this event"

  promptLines = [
    "You are writing in the narrative voice of Jim from St. John's."
    ""
    "Write only the next diary section."
    "Do not write the whole diary."
    "Stay in first person."
    "Use the current event as the focus."
    "Use the KAG cues as emotional guidance."
    "Preserve continuity with the prior diary sections."
    ""
    "Rules:"
    "- Keep the narration concrete and grounded"
    "- Keep the tone observational, slightly humorous, and reflective"
    "- Do not contradict earlier sections"
    "- Do not summarize future events"
    "- Return only the current section prose"
  ]

  promptLines.push transitionRule if transitionRule?
  promptLines.push ""
  promptLines.push "Diary story id:"
  promptLines.push "#{storyID}"
  promptLines.push ""
  promptLines.push "Prior diary history:"
  promptLines.push if historyText.length then historyText else "- none"
  promptLines.push ""
  promptLines.push "Current event:"
  promptLines.push if eventBlock? then eventBlock else "- none"
  promptLines.push ""
  promptLines.push "Current KAG cues:"
  promptLines.push if kagLines.length then kagLines.join("\n") else "- none"
  promptLines.push ""
  promptLines.push "Raw source passages:"
  promptLines.push if sourceLines.length then sourceLines.join("\n") else "- none"

  promptLines.join "\n"

resolveMode = (L) ->
  if /with_adapter/.test(L.stepName)
    return
      mode: 'adapter'
      rawKey: 'diary_adapted_raw'
      metaKey: 'diary_adapted_meta'
      textKey: 'diary_adapted_text'
      fileSuffix: '.adapter.txt'
  if /without_adapter/.test(L.stepName)
    return
      mode: 'base'
      rawKey: 'diary_base_raw'
      metaKey: 'diary_base_meta'
      textKey: 'diary_base_text'
      fileSuffix: '.txt'

  throw new Error "[#{L.stepName}] event ordered diary generator expects a with_adapter or without_adapter step name"

callDiaryGenerate = (L, modelDir, prompt, adapterPath, mlxConfig) ->
  args =
    model: modelDir
    prompt: prompt

  args["adapter-path"] = adapterPath if adapterPath?
  if mlxConfig? and typeof mlxConfig is 'object' and not Array.isArray(mlxConfig)
    for own key, value of mlxConfig
      continue unless value?
      args[key] = value

  L.callMLX 'generate', args

@step =
  desc: "Generate a diary entry one event at a time in story order"

  action: (L) ->
    modeInfo = resolveMode L

    storyParts = await L.need 'story_parts'
    storyParts = coerceJSON storyParts
    unless storyParts? and typeof storyParts is 'object' and not Array.isArray(storyParts)
      storyParts = await readArtifactTarget L, 'story_parts'
      storyParts = coerceJSON storyParts
    throw new Error "[#{L.stepName}] story_parts must be an object" unless storyParts? and typeof storyParts is 'object' and not Array.isArray(storyParts)

    diaryKag = L.theLowdown('diary_kag')?.value
    unless Array.isArray(diaryKag?.entries)
      diaryKag = await readArtifactTarget L, 'diary_kag'
    diaryKag = normalizeDiaryKag diaryKag
    throw new Error "[#{L.stepName}] diary_kag must be an object with entries" unless Array.isArray(diaryKag?.entries)

    storyID = String(storyParts.story_id ? '').trim()
    modelDir = L.param 'quantized_model_dir', null
    adapterPath = L.param 'adapter_path', null
    mlxConfig = L.param 'mlx', null
    throw new Error "[#{L.stepName}] Missing quantized_model_dir param" unless modelDir?
    throw new Error "[#{L.stepName}] Missing adapter_path" if modeInfo.mode is 'adapter' and not adapterPath?
    throw new Error "[#{L.stepName}] mlx must be an object when provided" if mlxConfig? and (typeof mlxConfig isnt 'object' or Array.isArray(mlxConfig))

    eventOrder = [
      ['scene', storyParts.scene]
      ['arrival', storyParts.arrival]
      ['disturbance', storyParts.disturbance]
      ['reflection', storyParts.reflection]
      ['realization', storyParts.realization]
    ]

    priorSections = []
    rawSections = []

    for [kind, event] in eventOrder
      continue unless event? and typeof event is 'object'

      chosenMatches = resolveEventMatches diaryKag, kind
      chosenEntries = if chosenMatches.length > 0 then chosenMatches else pickEventKagEntries diaryKag.entries, kind, event, 4
      prompt = buildEventPrompt storyID, kind, event, chosenEntries, chosenMatches, priorSections, modeInfo.mode
      rawOutput = callDiaryGenerate L, modelDir, prompt, adapterPath, mlxConfig
      sectionText = cleanGeneratedText prompt, rawOutput

      continue unless sectionText.length

      rawSections.push
        kind: kind
        prompt: prompt
        raw: String(rawOutput ? '')
        text: sectionText
        chunk_indexes: (entry.chunk_index for entry in chosenEntries when entry?.chunk_index?)
        keywords: (String(entry.keyword ? '').trim() for entry in chosenEntries when String(entry?.keyword ? '').trim().length)

      priorSections.push
        kind: kind
        text: sectionText

      console.log "[#{L.stepName}] generated section #{kind} chars:", sectionText.length

    finalText = priorSections.map((section) -> section.text).join "\n\n"
    rawJoined = rawSections.map((section) -> "[#{section.kind}]\n#{section.raw}").join "\n\n==========\n\n"
    meta =
      story_id: storyID
      mode: modeInfo.mode
      model_dir: modelDir
      adapter_path: if modeInfo.mode is 'adapter' then adapterPath else null
      section_count: priorSections.length
      sections: rawSections.map (section) ->
        kind: section.kind
        text_chars: section.text.length
        chunk_indexes: section.chunk_indexes
        keywords: section.keywords
      raw_chars: rawJoined.length
      text_chars: finalText.length

    console.log "[#{L.stepName}] story:", storyID
    console.log "[#{L.stepName}] section count:", priorSections.length
    console.log "[#{L.stepName}] text chars:", finalText.length

    L.make modeInfo.rawKey, rawJoined
    L.make modeInfo.metaKey, meta
    L.make modeInfo.textKey, finalText

    runTag = resolveRunTag L
    if typeof runTag is 'string' and runTag.length
      L.saveThis "diary/diary_#{runTag}#{modeInfo.fileSuffix}", finalText

    L.done()
    return
