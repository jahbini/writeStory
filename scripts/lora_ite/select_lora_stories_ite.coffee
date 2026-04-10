@step =
  desc: "Select a small SQLite-backed story batch for LoRA training"

  action: (L) ->
    batchSize = L.param 'batch_size'
    cycleState = L.peek 'lora_cycle_state', {}
    cycleState = {} unless cycleState? and typeof cycleState is 'object' and not Array.isArray(cycleState)

    usageEntry = L.theLowdown 'loraStoryUsage.jsonl'
    usageRows = usageEntry?.value
    if usageRows is undefined
      if typeof usageEntry?.waitFor is 'function'
        usageRows = await usageEntry.waitFor()
      else if usageEntry?.notifier?
        usageRows = await usageEntry.notifier

    throw new Error "[#{stepName}] loraStoryUsage.jsonl must be an array" unless Array.isArray usageRows

    totalStories = usageRows.length
    remainingRows = []
    for row in usageRows
      useCount = row?.use_count ? 0
      continue unless row?.story_id?
      if useCount <= 0
        remainingRows.push row

    resetThisRun = false

    if remainingRows.length is 0 and cycleState.ready_for_reset is true and totalStories > 0
      resetAt = new Date().toISOString()
      console.log "[select_lora_stories_ite] previous cycle marked complete; resetting LoRA usage for a fresh full retrain"
      L.saveThis 'loraCycleReset.json',
        mode: 'full'
        reset_at: resetAt
      resetThisRun = true
      remainingRows = usageRows.slice().sort (a, b) ->
        String(a?.story_id ? '').localeCompare String(b?.story_id ? '')

    selectedStoryIDs = []

    for row in remainingRows
      storyID = row?.story_id
      continue unless storyID?
      selectedStoryIDs.push storyID
      break if selectedStoryIDs.length >= batchSize

    console.log "[select_lora_stories_ite] usage rows:", usageRows.length
    console.log "[select_lora_stories_ite] remaining stories:", remainingRows.length
    console.log "[select_lora_stories_ite] selected batch size:", selectedStoryIDs.length

    for storyID, idx in selectedStoryIDs
      console.log "[select_lora_stories_ite] batch[#{idx}] #{storyID}"

    remainingCount = remainingRows.length
    nextCycleState =
      total_stories: totalStories
      remaining_stories: remainingCount
      ready_for_reset: false
      reset_this_run: resetThisRun
      updated_at: new Date().toISOString()

    if selectedStoryIDs.length is 0
      nextCycleState.ready_for_reset = true
      nextCycleState.completed_at = new Date().toISOString()
      L.make 'lora_cycle_state', nextCycleState
      L.make 'lora_remaining_count', 0
      L.saveThis 'pipeline:shutdown',
        by: L.stepName
        reason: 'no remaining SQLite-backed stories are available for LoRA training'
        timestamp: new Date().toISOString()
      L.make 'selected_story_ids', []
      L.done()
      return

    L.make 'lora_cycle_state', nextCycleState
    L.make 'lora_remaining_count', remainingCount
    L.make 'selected_story_ids', selectedStoryIDs
    L.done()
    return
