estimateTokens = (text) ->
  return 0 unless text?
  cleaned = String(text).trim()
  return 0 unless cleaned.length > 0
  Math.ceil(cleaned.length / 4)

splitParagraphs = (text) ->
  rawParagraphs = String(text ? '').split /\n\s*\n/
  paragraphs = []
  for para in rawParagraphs
    cleanPara = String(para ? '').trim()
    continue unless cleanPara.length > 0
    paragraphs.push cleanPara
  paragraphs

buildStoryGroups = (paragraphs) ->
  return [] unless Array.isArray(paragraphs)
  return [] unless paragraphs.length

  if paragraphs.length < 5
    return [paragraphs.slice()]

  groups = []
  total = paragraphs.length
  baseSize = Math.floor(total / 5)
  remainder = total % 5
  startIndex = 0

  for groupIndex in [0...5]
    groupSize = baseSize
    groupSize += 1 if groupIndex < remainder
    selected = paragraphs.slice startIndex, startIndex + groupSize
    groups.push selected
    startIndex += groupSize

  groups

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

@step =
  desc: "Build LoRA train/valid/test rows from SQLite-backed stories"

  action: (L) ->
    selectedStoryIDs = await L.need 'selected_story_ids'

    throw new Error "[#{L.stepName}] selected_story_ids must be an array" unless Array.isArray selectedStoryIDs

    if selectedStoryIDs.length is 0
      console.log "[build_lora_dataset_ite] no selected stories; writing empty datasets and stopping"
      L.make 'train_rows', []
      L.make 'valid_rows', []
      L.make 'test_rows', []
      L.done()
      return

    rows = []
    rowsWritten = 0
    storiesProcessed = 0

    MAX_TOTAL_TOKENS = 1024
    SAFETY_TOKENS = 64

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

      paragraphs = splitParagraphs fullStoryText

      continue unless paragraphs.length > 0

      storyGroups = buildStoryGroups paragraphs

      for groupParagraphs in storyGroups
        continue unless Array.isArray(groupParagraphs)
        continue unless groupParagraphs.length > 0

        fragmentParagraphs = buildFragmentParagraphs groupParagraphs
        continue unless fragmentParagraphs.length > 0

        fragmentText = fragmentParagraphs.join "\n\n"
        prompt = fragmentText.trim() + "\n\n"
        promptTokens = estimateTokens prompt

        completionStartIndex = fragmentParagraphs.length
        completionParagraphs = groupParagraphs.slice completionStartIndex

        if completionParagraphs.length is 0
          continue

        maxCompletionTokens = MAX_TOTAL_TOKENS - promptTokens - SAFETY_TOKENS
        throw new Error "[#{L.stepName}] prompt too large for token budget on story #{storyID}" if maxCompletionTokens < 80

        chunkParagraphs = []
        chunkTokens = 0

        flushChunk = (isLastChunk) ->
          return unless chunkParagraphs.length > 0

          completionText = chunkParagraphs.join "\n\n"

          textOut = prompt + completionText

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
              textOut = prompt + completionText
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

    if rows.length is 0
      shutdownAt = new Date().toISOString()
      console.log "[build_lora_dataset_ite] selected stories produced no trainable rows; shutting down pipeline cleanly"
      L.saveThis 'pipeline:shutdown',
        by: L.stepName
        reason: 'selected stories produced no LoRA training rows'
        timestamp: shutdownAt
      L.make 'train_rows', []
      L.make 'valid_rows', []
      L.make 'test_rows', []
      L.done()
      return

    L.make 'train_rows', rows
    L.make 'valid_rows', rows
    L.make 'test_rows', rows
    L.done()
    return
