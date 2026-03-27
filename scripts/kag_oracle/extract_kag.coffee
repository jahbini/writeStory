@step =
  desc: "Extract structured KAG fields for prompt caching"

  action: (S) ->
    sampleIndex = S.param 'sample_index'
    sampleIndex = parseInt(sampleIndex, 10)
    sampleIndex = 0 if Number.isNaN(sampleIndex)

    sourceRows = await S.need 'kag_segments'
    throw new Error "[#{S.stepName}] kag_segments must be an array" unless Array.isArray(sourceRows)
    throw new Error "[#{S.stepName}] kag_segments is empty" unless sourceRows.length > 0

    row = sourceRows[sampleIndex]
    throw new Error "[#{S.stepName}] sample_index #{sampleIndex} is out of range" unless row?

    out =
      story_id: row?.meta?.doc_id ? null
      source_index: sampleIndex
      source_meta: row?.meta ? {}
      fields:
        source_prompt: row?.prompt ? row?.text ? null
        emotions: row?.emotions ? {}

    S.make 'kag_record', out
    S.done()
    return
