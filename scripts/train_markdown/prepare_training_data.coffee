fs = require 'fs'
path = require 'path'

@step =
  desc: "Build MLX training data from newly identified full stories"

  action: (M, stepName) ->
    storiesKey = M.getStepParam stepName, 'marshalled_stories'
    newIdsKey = M.getStepParam stepName, 'new_story_ids'
    trainFile = M.getStepParam stepName, 'train_file'
    validFile = M.getStepParam stepName, 'valid_file'
    testFile = M.getStepParam stepName, 'test_file'

    console.error "JIM", storiesKey, newIdsKey, trainFile

    storiesEntry = M.theLowdown storiesKey
    stories = storiesEntry?.value
    stories = await storiesEntry.notifier if stories is undefined

    newIdsEntry = M.theLowdown newIdsKey
    newStoryIds = newIdsEntry?.value
    newStoryIds = await newIdsEntry.notifier if newStoryIds is undefined

    throw new Error "[#{stepName}] #{storiesKey} must be an array" unless Array.isArray(stories)
    throw new Error "[#{stepName}] #{newIdsKey} must be an array" unless Array.isArray(newStoryIds)

    console.error "JIM", newStoryIds
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

    fs.mkdirSync path.dirname(trainFile), { recursive: true }

    storiesProcessed = 0
    rowsWritten = 0

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

      row =
        text: fullStoryText

      fs.appendFileSync trainFile, JSON.stringify(row) + "\n", 'utf8'
      fs.appendFileSync validFile, JSON.stringify(row) + "\n", 'utf8'
      fs.appendFileSync testFile, JSON.stringify(row) + "\n", 'utf8'
      storiesProcessed += 1
      rowsWritten += 1

    console.log "[prepare_training_data] stories processed:", storiesProcessed
    console.log "[prepare_training_data] rows written:", rowsWritten

    M.saveThis "done:#{stepName}", true
    return
