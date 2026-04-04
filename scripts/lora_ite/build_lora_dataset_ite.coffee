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
  cleaned = String(text).trim()
  return 0 unless cleaned.length > 0
  Math.ceil(cleaned.length / 4)

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

  rval

sanitizeStop = (text) ->
  rval = String(text ? '')
  rval = rval.replace /(?:\s*<stop>\s*)+$/g, ''
  rval.trim()

@step =
  desc: "Build LoRA train/valid/test rows from SQLite-backed stories"

  action: (L) ->
    selectedStoryIDs = await L.need 'selected_story_ids'

    throw new Error "[#{L.stepName}] selected_story_ids must be an array" unless Array.isArray selectedStoryIDs

    rows = []
    rowsWritten = 0
    storiesProcessed = 0

    MAX_TOTAL_TOKENS = 1024
    SAFETY_TOKENS = 64
    STOP_TEXT = "\n\n<stop>"
    STOP_TOKENS = estimateTokens STOP_TEXT

    for storyID in selectedStoryIDs
      continue unless storyID?

      storyEntry = L.theLowdown "storyByID{#{storyID}}.json"
      story = storyEntry?.value
      if story is undefined
        if typeof storyEntry?.waitFor is 'function'
          story = await storyEntry.waitFor()
        else if storyEntry?.notifier?
          story = await storyEntry.notifier

      throw new Error "[#{L.stepName}] Missing storyByID for #{storyID}" unless story?

      fullStoryText = String(story.text ? '').trim()
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

      if completionParagraphs.length is 0
        textOut = prompt.trim() + STOP_TEXT
        rows.push text: textOut
        rowsWritten += 1
        storiesProcessed += 1
        continue

      maxCompletionTokens = MAX_TOTAL_TOKENS - promptTokens - SAFETY_TOKENS - STOP_TOKENS
      throw new Error "[#{L.stepName}] prompt too large for token budget on story #{storyID}" if maxCompletionTokens < 80

      chunkParagraphs = []
      chunkTokens = 0

      flushChunk = (isLastChunk) ->
        return unless chunkParagraphs.length > 0

        completionText = chunkParagraphs.join "\n\n"
        completionText = sanitizeStop completionText

        textOut = prompt + completionText
        if isLastChunk
          textOut = sanitizeStop(textOut) + STOP_TEXT

        rows.push text: textOut
        rowsWritten += 1
        chunkParagraphs = []
        chunkTokens = 0
        return

      for para, idx in completionParagraphs
        paraTokens = estimateTokens para
        proposedTokens = chunkTokens + paraTokens

        if chunkParagraphs.length > 0 and proposedTokens > maxCompletionTokens
          flushChunk false

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
              rows.push text: textOut
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
            rows.push text: textOut
            rowsWritten += 1

          continue

        chunkParagraphs.push para
        chunkTokens += paraTokens

      if chunkParagraphs.length > 0
        flushChunk true

      storiesProcessed += 1

    console.log "[build_lora_dataset_ite] stories processed:", storiesProcessed
    console.log "[build_lora_dataset_ite] rows written:", rowsWritten

    L.make 'train_rows', rows
    L.make 'valid_rows', rows
    L.make 'test_rows', rows
    L.done()
    return
