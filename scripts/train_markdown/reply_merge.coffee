loadArray = (M, key) ->
  entry = M.theLowdown key
  value = entry?.value
  value = await entry.notifier if value is undefined
  value

@step =
  desc: "Merge oracle replies back into marshalled markdown segments"

  action: (M, stepName) ->
    segKey = M.getStepParam stepName, 'marshalled_stories'
    emoKey = M.getStepParam stepName, 'kag_emotions'
    outKey = M.getStepParam stepName, 'merged_segments'

    segments = await loadArray M, segKey
    replies  = await loadArray M, emoKey
    existingEntry = M.theLowdown outKey
    existingMerged = existingEntry?.value
    existingMerged = [] if existingMerged is undefined

    throw new Error "#{segKey} must be an array" unless Array.isArray(segments)
    throw new Error "#{emoKey} must be an array" unless Array.isArray(replies)
    existingMerged = [] unless Array.isArray(existingMerged)

    if replies.length is 0
      console.log "[reply_merge] no oracle replies yet"
      M.saveThis "done:#{stepName}", true
      return

    existingKeys = new Set()
    for row in existingMerged when row?.meta?
      existingKeys.add "#{row.meta.doc_id}|#{row.meta.paragraph_index}"

    lookup = Object.create null
    for reply in replies when reply?.meta?
      lookup["#{reply.meta.doc_id}|#{reply.meta.paragraph_index}"] = reply.emotions

    merged = []
    for segment in segments
      key = "#{segment.meta?.doc_id}|#{segment.meta?.paragraph_index}"
      continue if existingKeys.has key
      emotions = lookup[key]
      continue unless emotions? 
      merged.push
        meta: segment.meta
        prompt: segment.text ? segment.prompt
        emotions: emotions

    console.log "[reply_merge] newly merged segments:", merged.length
    M.saveThis outKey, merged
    M.saveThis "done:#{stepName}", true
    return
