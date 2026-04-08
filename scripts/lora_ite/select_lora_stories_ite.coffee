@step =
  desc: "Select a small SQLite-backed story batch for LoRA training"

  action: (L) ->
    batchSize = L.param 'batch_size'

    usageEntry = L.theLowdown 'loraStoryUsage.jsonl'
    usageRows = usageEntry?.value
    if usageRows is undefined
      if typeof usageEntry?.waitFor is 'function'
        usageRows = await usageEntry.waitFor()
      else if usageEntry?.notifier?
        usageRows = await usageEntry.notifier

    throw new Error "[#{stepName}] loraStoryUsage.jsonl must be an array" unless Array.isArray usageRows

    selectedStoryIDs = []

    for row in usageRows
      storyID = row?.story_id
      continue unless storyID?
      selectedStoryIDs.push storyID
      break if selectedStoryIDs.length >= batchSize

    console.log "[select_lora_stories_ite] usage rows:", usageRows.length
    console.log "[select_lora_stories_ite] selected batch size:", selectedStoryIDs.length

    for storyID, idx in selectedStoryIDs
      console.log "[select_lora_stories_ite] batch[#{idx}] #{storyID}"

    if selectedStoryIDs.length is 0
      L.saveThis 'pipeline:shutdown',
        by: L.stepName
        reason: 'no remaining SQLite-backed stories are available for LoRA training'
        timestamp: new Date().toISOString()

    L.make 'selected_story_ids', selectedStoryIDs
    L.done()
    return
