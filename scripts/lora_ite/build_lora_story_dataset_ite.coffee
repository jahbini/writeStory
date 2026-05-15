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

# Short generic instructions that match what a real user types at inference time.
# All voice prose belongs in the assistant turn; nothing here echoes the target style.
USER_INSTRUCTIONS = [
  "Write a passage in your distinctive voice."
  "Tell me a story in your voice."
  "Share another story in your usual style."
  "Write something in your usual way."
  "Continue in your distinctive voice."
  "Write a passage the way you normally do."
  "Give me a story in the way only you would tell it."
]

# Deterministic rotation keyed on (rowIndex, storyID). storyIDs are
# non-numeric strings; derive the numeric offset via char-code sum so
# Number() coercion is never attempted and NaN is impossible.
storyOffset = (storyID) ->
  key = String(storyID ? '')
  sum = 0
  sum += key.charCodeAt(i) for i in [0...key.length]
  sum

pickInstruction = (rowIndex, storyID) ->
  index = (rowIndex + storyOffset(storyID)) %% USER_INSTRUCTIONS.length
  USER_INSTRUCTIONS[index]

makeChatRow = (instructionText, assistantText) ->
  messages: [
    {role: "user",      content: String(instructionText ? '').trim()}
    {role: "assistant", content: String(assistantText   ? '').trim()}
  ]

@step =
  desc: "Build chat-formatted LoRA train/valid/test rows from SQLite-backed stories"

  action: (L) ->
    selectedStoryIDs = await L.need 'selected_story_ids'
    throw new Error "[#{L.stepName}] selected_story_ids must be an array" unless Array.isArray selectedStoryIDs

    if selectedStoryIDs.length is 0
      console.log "[build_lora_story_dataset_ite] no selected stories; writing empty datasets and stopping"
      L.make 'train_rows', []
      L.make 'valid_rows', []
      L.make 'test_rows', []
      L.done()
      return

    rows = []
    rowIndex = 0
    rowsWritten = 0
    skippedOverBudget = 0
    storiesProcessed = 0

    MAX_TOTAL_TOKENS = 1024
    SAFETY_TOKENS = 96   # headroom for chat-format wrapper tokens

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

        instruction   = pickInstruction rowIndex, storyID
        assistantText = groupParagraphs.join "\n\n"

        totalTokens = estimateTokens(instruction) + estimateTokens(assistantText) + SAFETY_TOKENS
        if totalTokens > MAX_TOTAL_TOKENS
          console.log "[build_lora_story_dataset_ite] skipping over-budget group for story #{storyID}: ~#{totalTokens} tokens"
          skippedOverBudget += 1
          continue

        rows.push makeChatRow(instruction, assistantText)
        rowIndex += 1
        rowsWritten += 1

      storiesProcessed += 1

    console.log "[build_lora_story_dataset_ite] stories processed:", storiesProcessed
    console.log "[build_lora_story_dataset_ite] chat-formatted rows written:", rowsWritten
    console.log "[build_lora_story_dataset_ite] over-budget groups skipped:", skippedOverBudget

    if rows.length is 0
      shutdownAt = new Date().toISOString()
      console.log "[build_lora_story_dataset_ite] selected stories produced no trainable rows; shutting down pipeline cleanly"
      L.saveThis 'pipeline:shutdown',
        by: L.stepName
        reason: 'selected stories produced no LoRA training rows'
        timestamp: shutdownAt
      L.make 'train_rows', []
      L.make 'valid_rows', []
      L.make 'test_rows', []
      L.done()
      return

    # Same-rows-everywhere matches the existing build_lora_dataset_ite contract.
    L.make 'train_rows', rows
    L.make 'valid_rows', rows
    L.make 'test_rows', rows
    L.done()
    return
