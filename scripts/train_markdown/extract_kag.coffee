@step =
  desc: "Extract structured KAG fields for prompt caching"

  action: (M, stepName) ->
    sourceKey = M.getStepParam(stepName, 'kag_source_key')
    sampleIndex = M.getStepParam(stepName, 'sample_index')
    sampleIndex = parseInt(sampleIndex, 10)
    sampleIndex = 0 if Number.isNaN(sampleIndex)

    sourceEntry = M.theLowdown sourceKey
    sourceRows = sourceEntry?.value
    if sourceRows is undefined
      console.error "JIM awaits", sourceKey
      sourceRows = await sourceEntry.notifier
    throw new Error "[#{stepName}] Missing input key '#{sourceKey}'" if sourceRows is undefined
    throw new Error "[#{stepName}] #{sourceKey} must be an array" unless Array.isArray(sourceRows)
    throw new Error "[#{stepName}] #{sourceKey} is empty" unless sourceRows.length > 0

    row = sourceRows[sampleIndex]
    throw new Error "[#{stepName}] sample_index #{sampleIndex} is out of range" unless row?

    out =
      story_id: row?.meta?.doc_id ? null
      source_index: sampleIndex
      source_meta: row?.meta ? {}
      fields:
        source_prompt: row?.prompt ? row?.text ? null
        emotions: row?.emotions ? {}

    M.saveThis "kag_record", out
    M.saveThis "done:#{stepName}", true
    return
