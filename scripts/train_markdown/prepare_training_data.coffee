pre_prompt = """You are writing in the narrative voice of Jim from St. John's.

Expand the following story fragment into a complete reflective narrative.

Maintain the same events and ideas, but improve flow, imagery, and voice.

Rules:
- Speak in the first person as Jim
- Keep the same order of events
- Do not introduce new plot elements
- Add natural narration and sensory detail
- The tone should be observational, slightly humorous, and reflective
- Return only the finished story

Story fragment:
"""

estimateTokens = (text) ->
  return 0 unless text?
  cleaned = String(text)
  cleaned = cleaned.trim()
  return 0 unless cleaned.length > 0
  # crude but safe enough for packing
  rval = Math.ceil(cleaned.length / 4)
  return rval

buildFragmentParagraphs = (paragraphs) ->
  rval = []
  return rval unless Array.isArray(paragraphs)
  return rval if paragraphs.length is 0

  firstPara = paragraphs[0] ? ''
  if firstPara.trim().length > 0
    rval.push firstPara.trim()

  currentText = rval.join "\n\n"
  currentLen = currentText.length

  if currentLen < 300 and paragraphs.length > 1
    secondPara = paragraphs[1] ? ''
    if secondPara.trim().length > 0
      rval.push secondPara.trim()

  return rval

sanitizeStop = (text) ->
  rval = String(text ? '')
  rval = rval.replace /(?:\s*<stop>\s*)+$/g, ''
  rval = rval.trim()
  return rval

@step =
  desc: "Build MLX training data from newly identified full stories"

  action: (M, stepName) ->
    storiesKey = M.getStepParam stepName, 'marshalled_stories'
    newIdsKey = M.getStepParam stepName, 'new_story_ids'
    trainFile = M.getStepParam stepName, 'train_file'
    validFile = M.getStepParam stepName, 'valid_file'
    testFile = M.getStepParam stepName, 'test_file'

    storiesEntry = M.theLowdown storiesKey
    stories = storiesEntry?.value
    stories = await storiesEntry.notifier if stories is undefined

    newIdsEntry = M.theLowdown newIdsKey
    newStoryIds = newIdsEntry?.value
    throw new Error "[#{stepName}] Missing input key '#{newIdsKey}'. Run kag_oracle first or provide #{newIdsKey} on disk before train_lora." if newStoryIds is undefined

    throw new Error "[#{stepName}] #{storiesKey} must be an array" unless Array.isArray stories
    throw new Error "[#{stepName}] #{newIdsKey} must be an array" unless Array.isArray newStoryIds

    if newStoryIds.length is 0
      console.log "[prepare_training_data] stories processed: 0"
      console.log "[prepare_training_data] rows written: 0"
      M.saveThis trainFile, []
      M.saveThis validFile, []
      M.saveThis testFile, []
      M.saveThis "done:#{stepName}", true
      return

    wantedTitles = new Set()
    for title in newStoryIds
      continue unless title?
      wantedTitles.add title

    grouped = {}
    for segment in stories
      title = segment?.meta?.title
      text = segment?.text
      continue unless title?
      continue unless text?
      continue unless wantedTitles.has title
      grouped[title] ?= []
      grouped[title].push segment

    rows = []
    rowsWritten = 0
    storiesProcessed = 0

    MAX_TOTAL_TOKENS = 1024
    SAFETY_TOKENS = 64
    STOP_TEXT = "\n\n<stop>"
    STOP_TOKENS = estimateTokens STOP_TEXT

    for own title, segments of grouped
      continue unless Array.isArray segments
      continue unless segments.length > 0

      segments.sort (a, b) ->
        left = a?.meta?.paragraph_index ? ''
        right = b?.meta?.paragraph_index ? ''
        if left < right then -1 else if left > right then 1 else 0

      fullStoryText = ''
      first = true

      for segment in segments
        text = segment?.text
        continue unless text?
        cleanText = String(text).trim()
        continue unless cleanText.length > 0

        if first
          fullStoryText = cleanText
          first = false
        else
          fullStoryText += "\n\n" + cleanText

      continue unless fullStoryText.length > 0

      rawParagraphs = fullStoryText.split /\n\s*\n/
      paragraphs = []

      for para in rawParagraphs
        cleanPara = String(para ? '').trim()
        continue unless cleanPara.length > 0
        paragraphs.push cleanPara

      continue unless paragraphs.length > 0

      fragmentParagraphs = buildFragmentParagraphs paragraphs
      continue unless fragmentParagraphs.length > 0

      fragmentText = fragmentParagraphs.join "\n\n"
      prompt = pre_prompt + fragmentText + "\n\nBegin:\n\n"
      promptTokens = estimateTokens prompt

      completionStartIndex = fragmentParagraphs.length
      completionParagraphs = paragraphs.slice completionStartIndex

      # If the story is so short that all paragraphs were used for fragment,
      # emit one row with empty continuation and <stop>.
      if completionParagraphs.length is 0
        textOut = prompt.trim() + STOP_TEXT
        rows.push { text: textOut }
        rowsWritten += 1
        storiesProcessed += 1
        continue

      maxCompletionTokens = MAX_TOTAL_TOKENS - promptTokens - SAFETY_TOKENS - STOP_TOKENS
      if maxCompletionTokens < 80
        throw new Error "[#{stepName}] prompt too large for token budget on title #{title}"

      chunkParagraphs = []
      chunkTokens = 0

      flushChunk = (isLastChunk) ->
        return unless chunkParagraphs.length > 0

        completionText = chunkParagraphs.join "\n\n"
        completionText = sanitizeStop completionText

        textOut = prompt + completionText
        if isLastChunk
          textOut = sanitizeStop(textOut) + STOP_TEXT

        rows.push { text: textOut }
        rowsWritten += 1
        chunkParagraphs = []
        chunkTokens = 0
        return

      for para, idx in completionParagraphs
        paraTokens = estimateTokens para
        proposedTokens = chunkTokens + paraTokens

        if chunkParagraphs.length > 0 and proposedTokens > maxCompletionTokens
          flushChunk false

        # If a single paragraph is too large, hard-split it by sentences.
        if paraTokens > maxCompletionTokens
          sentences = para.split /(?<=[.!?])\s+/
          sentenceChunk = []
          sentenceTokens = 0

          for sentence in sentences
            cleanSentence = String(sentence ? '').trim()
            continue unless cleanSentence.length > 0

            sentTokens = estimateTokens cleanSentence
            proposedSentenceTokens = sentenceTokens + sentTokens

            if sentenceChunk.length > 0 and proposedSentenceTokens > maxCompletionTokens
              completionText = sentenceChunk.join " "
              textOut = prompt + completionText
              rows.push { text: textOut }
              rowsWritten += 1
              sentenceChunk = []
              sentenceTokens = 0

            sentenceChunk.push cleanSentence
            sentenceTokens += sentTokens

          if sentenceChunk.length > 0
            isLastSentenceChunk = idx is completionParagraphs.length - 1
            completionText = sentenceChunk.join " "
            completionText = sanitizeStop completionText
            textOut = prompt + completionText
            if isLastSentenceChunk
              textOut = sanitizeStop(textOut) + STOP_TEXT
            rows.push { text: textOut }
            rowsWritten += 1

          continue

        chunkParagraphs.push para
        chunkTokens += paraTokens

      if chunkParagraphs.length > 0
        flushChunk true

      storiesProcessed += 1

    console.log "[prepare_training_data] stories processed:", storiesProcessed
    console.log "[prepare_training_data] rows written:", rowsWritten

    M.saveThis trainFile, rows
    M.saveThis validFile, rows
    M.saveThis testFile, rows
    M.saveThis "done:#{stepName}", true
    return
