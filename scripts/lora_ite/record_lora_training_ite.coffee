@step =
  desc: "Persist LoRA training usage and run metadata through SQLite meta keys"

  action: (M, stepName) ->
    selectedEntry = M.theLowdown 'selected_story_ids'
    selectedStoryIDs = selectedEntry?.value
    if selectedStoryIDs is undefined
      if typeof selectedEntry?.waitFor is 'function'
        selectedStoryIDs = await selectedEntry.waitFor()
      else if selectedEntry?.notifier?
        selectedStoryIDs = await selectedEntry.notifier

    runEntry = M.theLowdown 'lora_run_record'
    runRecord = runEntry?.value
    if runRecord is undefined
      if typeof runEntry?.waitFor is 'function'
        runRecord = await runEntry.waitFor()
      else if runEntry?.notifier?
        runRecord = await runEntry.notifier

    throw new Error "[#{stepName}] selected_story_ids must be an array" unless Array.isArray selectedStoryIDs
    throw new Error "[#{stepName}] lora_run_record must be an object" unless runRecord? and typeof runRecord is 'object' and not Array.isArray(runRecord)
    throw new Error "[#{stepName}] lora_run_record missing run_id" unless runRecord.run_id?

    payload =
      run_id: runRecord.run_id
      started_at: runRecord.started_at ? null
      finished_at: runRecord.finished_at ? null
      status: runRecord.status ? 'done'
      model_dir: runRecord.model_dir ? null
      adapter_path: runRecord.adapter_path ? null
      resume_adapter_file: runRecord.resume_adapter_file ? null
      training_dir: runRecord.training_dir ? null
      stdout_text: runRecord.stdout_text ? null
      train_rows_count: runRecord.train_rows_count ? null
      valid_rows_count: runRecord.valid_rows_count ? null
      test_rows_count: runRecord.test_rows_count ? null
      checkpoint_path: runRecord.checkpoint_path ? null
      story_ids: selectedStoryIDs

    M.saveThis "loraTrainingRun{#{payload.run_id}}.json", payload

    usageEntry = M.theLowdown 'loraStoryUsage.jsonl'
    usageRows = usageEntry?.value
    if usageRows is undefined
      if typeof usageEntry?.waitFor is 'function'
        usageRows = await usageEntry.waitFor()
      else if usageEntry?.notifier?
        usageRows = await usageEntry.notifier

    throw new Error "[#{stepName}] loraStoryUsage.jsonl must be an array" unless Array.isArray usageRows

    trainedStoryIDs = []
    for row in usageRows
      storyID = row?.story_id
      useCount = row?.use_count ? 0
      continue unless storyID?
      continue unless useCount > 0
      trainedStoryIDs.push storyID

    console.log "[record_lora_training_ite] run id:", payload.run_id
    console.log "[record_lora_training_ite] recorded stories:", selectedStoryIDs.length
    console.log "[record_lora_training_ite] total stories with LoRA usage:", trainedStoryIDs.length

    M.saveThis 'trained_story_ids', trainedStoryIDs
    M.saveThis "done:#{stepName}", true
    return
