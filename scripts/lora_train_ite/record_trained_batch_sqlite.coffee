@step =
  desc: "Record the completed SQLite-backed LoRA training batch"

  action: (S) ->
    newStoryIds = await S.need 'new_story_ids'
    trainedRows = S.theLowdown('trainedStories.jsonl')?.value
    trainedRows = [] if trainedRows is undefined

    throw new Error "[#{S.stepName}] new_story_ids must be an array" unless Array.isArray newStoryIds
    throw new Error "[#{S.stepName}] trainedStories.jsonl must be an array" unless Array.isArray trainedRows

    mergedRows = []
    seen = new Set()

    for row in trainedRows
      if typeof row is 'string'
        storyID = row
        trainedAt = null
      else
        storyID = row?.story_id
        trainedAt = row?.trained_at ? null

      continue unless storyID?
      continue if seen.has storyID
      seen.add storyID
      mergedRows.push
        story_id: storyID
        trained_at: trainedAt

    stamp = new Date().toISOString()

    for storyID in newStoryIds
      continue unless storyID?
      continue if seen.has storyID
      seen.add storyID
      mergedRows.push
        story_id: storyID
        trained_at: stamp

    console.log "[record_trained_batch_sqlite] previous trained stories:", trainedRows.length
    console.log "[record_trained_batch_sqlite] current batch stories:", newStoryIds.length
    console.log "[record_trained_batch_sqlite] total trained stories:", mergedRows.length

    for storyID, idx in newStoryIds
      console.log "[record_trained_batch_sqlite] trained[#{idx}] #{storyID}" if storyID?

    S.saveThis 'trainedStories.jsonl', mergedRows
    S.make 'trained_story_ids', (row.story_id for row in mergedRows)
    S.done()
    return
