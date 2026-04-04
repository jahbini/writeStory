@step =
  desc: "Materialize LoRA-trained story ids from SQLite usage metadata"

  action: (L) ->
    usageEntry = L.theLowdown 'loraStoryUsage.jsonl'
    usageRows = usageEntry?.value
    if usageRows is undefined
      if typeof usageEntry?.waitFor is 'function'
        usageRows = await usageEntry.waitFor()
      else if usageEntry?.notifier?
          usageRows = await usageEntry.notifier

    throw new Error "[#{L.stepName}] loraStoryUsage.jsonl must be an array" unless Array.isArray usageRows

    trainedStoryIDs = []
    for row in usageRows
      storyID = row?.story_id
      useCount = row?.use_count ? 0
      continue unless storyID?
      continue unless useCount > 0
      trainedStoryIDs.push storyID

    console.log "[record_lora_training_ite] total stories with LoRA usage:", trainedStoryIDs.length

    L.make 'trained_story_ids', trainedStoryIDs
    L.done()
    return
