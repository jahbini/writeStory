@step =
  desc: "Select the next batch of SQLite-backed stories for LoRA training"

  action: (S) ->
    batchSize = S.param 'batch_size'

    stories = S.theLowdown('allStories.jsonl')?.value
    trainedRows = S.theLowdown('trainedStories.jsonl')?.value

    throw new Error "[#{S.stepName}] allStories.jsonl must be an array" unless Array.isArray stories
    trainedRows = [] if trainedRows is undefined
    throw new Error "[#{S.stepName}] trainedStories.jsonl must be an array" unless Array.isArray trainedRows

    trainedSet = new Set()
    for row in trainedRows
      if typeof row is 'string'
        trainedSet.add row
      else
        storyID = row?.story_id
        trainedSet.add storyID if storyID?

    batchIDs = []
    unseenCount = 0

    for story in stories
      storyID = story?.story_id
      continue unless storyID?

      unless trainedSet.has storyID
        unseenCount += 1
        if batchIDs.length < batchSize
          batchIDs.push storyID

    storiesLeft = Math.max unseenCount - batchIDs.length, 0

    console.log "[select_training_batch_sqlite] total unseen stories:", unseenCount
    console.log "[select_training_batch_sqlite] already trained stories:", trainedSet.size
    console.log "[select_training_batch_sqlite] selected batch size:", batchIDs.length
    console.log "[select_training_batch_sqlite] stories left after this batch:", storiesLeft

    for storyID, idx in batchIDs
      console.log "[select_training_batch_sqlite] batch[#{idx}] #{storyID}"

    S.make 'new_story_ids', batchIDs
    S.done()
    return
