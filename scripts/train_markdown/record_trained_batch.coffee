@step =
  desc: "Record the completed LoRA training batch"

  action: (S) ->
    newStoryIds = await S.need 'new_story_ids'
    trainedStoryIds = await S.peek 'trained_story_ids', []

    throw new Error "[#{S.stepName}] new_story_ids must be an array" unless Array.isArray newStoryIds
    throw new Error "[#{S.stepName}] trained_story_ids must be an array" unless Array.isArray trainedStoryIds

    merged = []
    seen = new Set()

    for title in trainedStoryIds
      continue unless title?
      continue if seen.has title
      seen.add title
      merged.push title

    for title in newStoryIds
      continue unless title?
      continue if seen.has title
      seen.add title
      merged.push title

    console.log "[record_trained_batch] previous trained stories:", trainedStoryIds.length
    console.log "[record_trained_batch] current batch stories:", newStoryIds.length
    console.log "[record_trained_batch] total trained stories:", merged.length

    for title, idx in newStoryIds
      console.log "[record_trained_batch] trained[#{idx}] #{title}" if title?

    S.make 'trained_story_ids', merged
    S.done()
    return
