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
    newStoryIds = await newIdsEntry.notifier if newStoryIds is undefined

    throw new Error "[#{stepName}] #{storiesKey} must be an array" unless Array.isArray(stories)
    throw new Error "[#{stepName}] #{newIdsKey} must be an array" unless Array.isArray(newStoryIds)

    console.error "JIM newstory IDs", newStoryIds
    if newStoryIds.length is 0
      console.log "[prepare_training_data] stories processed: 0"
      console.log "[prepare_training_data] rows written: 0"
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

    storiesProcessed = 0
    rowsWritten = 0
    rows = []

    for own title, segments of grouped
      continue unless Array.isArray(segments)
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
        if first
          fullStoryText = text
          first = false
        else
          fullStoryText += "\n\n" + text

      continue unless fullStoryText.length > 0

      paragraphs = fullStoryText.split /\n\s*\n/
      fragment = ''
      if paragraphs.length > 0
        fragment = paragraphs[0]
      if fragment.length < 300 and paragraphs.length > 1
        fragment += "\n\n" + paragraphs[1]

      continue unless fragment.length > 0
      row =
        text: pre_prompt + "\n" + fragment + "\n\nFinished story:\n" + fullStoryText

      rows.push row
      storiesProcessed += 1
      rowsWritten += 1

    console.log "[prepare_training_data] stories processed:", storiesProcessed
    console.log "[prepare_training_data] rows written:", rowsWritten

    M.saveThis trainFile, rows
    M.saveThis validFile, rows
    M.saveThis testFile, rows
    M.saveThis "done:#{stepName}", true
    return
