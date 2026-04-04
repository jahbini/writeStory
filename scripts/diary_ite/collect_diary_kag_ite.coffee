wordSet = (text) ->
  words = String(text ? '').toLowerCase().match(/[a-z0-9]+/g) ? []
  new Set(words)

coerceJSON = (value) ->
  return value unless typeof value is 'string'
  try
    JSON.parse value
  catch
    value

scoreEntry = (entry, eventWords) ->
  words = wordSet("#{entry?.keyword ? ''} #{entry?.headline ? ''}")
  score = 0
  score += 3 if String(entry?.keyword ? '').trim().length and eventWords.has(String(entry.keyword).toLowerCase())
  for word in words
    score += 1 if eventWords.has(word)
  score

@step =
  desc: "Collect the KAG entries that best fit the selected diary events"

  action: (L) ->
    storyParts = await L.need 'story_parts'
    storyParts = coerceJSON storyParts

    throw new Error "[#{L.stepName}] story_parts must be an object" unless storyParts? and typeof storyParts is 'object' and not Array.isArray(storyParts)

    storyID = String(storyParts.story_id ? '').trim()
    throw new Error "[#{L.stepName}] story_parts missing story_id" unless storyID.length

    pretendStoryIDs = L.param 'pretend_story_ids', null

    eventWords = new Set()
    fields = [
      ['scene', storyParts.scene]
      ['arrival', storyParts.arrival]
      ['disturbance', storyParts.disturbance]
      ['reflection', storyParts.reflection]
      ['realization', storyParts.realization]
    ]

    for [kind, event] in fields
      for word in wordSet("#{kind} #{event?.text ? ''} #{event?.character ? ''} #{event?.location ? ''} #{event?.theme ? ''}")
        eventWords.add word

    scored = []
    if Array.isArray(pretendStoryIDs) and pretendStoryIDs.length > 0
      for dbStoryID in pretendStoryIDs
        continue unless dbStoryID?

        kagEntry = L.theLowdown "kagFor{#{dbStoryID}}.json"
        kag = kagEntry?.value
        continue unless Array.isArray(kag?.entries)

        for entry, idx in kag.entries
          scored.push
            story_id: dbStoryID
            entry: entry
            idx: idx
            score: scoreEntry(entry, eventWords)
    else
      allStoriesEntry = L.theLowdown 'allStories.jsonl'
      allStories = allStoriesEntry?.value
      if allStories is undefined
        if typeof allStoriesEntry?.waitFor is 'function'
          allStories = await allStoriesEntry.waitFor()
        else if allStoriesEntry?.notifier?
          allStories = await allStoriesEntry.notifier

      throw new Error "[#{L.stepName}] allStories.jsonl must be an array" unless Array.isArray allStories

      for storyRow in allStories
        dbStoryID = storyRow?.story_id
        continue unless dbStoryID?

        kagEntry = L.theLowdown "kagFor{#{dbStoryID}}.json"
        kag = kagEntry?.value
        continue unless Array.isArray(kag?.entries)

        for entry, idx in kag.entries
          scored.push
            story_id: dbStoryID
            entry: entry
            idx: idx
            score: scoreEntry(entry, eventWords)

    scored.sort (a, b) ->
      if b.score isnt a.score then b.score - a.score else a.idx - b.idx

    chosen = (row.entry for row in scored.slice(0, Math.min(5, scored.length)))

    keywordSet = new Set()
    keywords = []
    for entry in chosen
      keyword = String(entry?.keyword ? '').trim()
      continue unless keyword.length
      continue if keywordSet.has keyword
      keywordSet.add keyword
      keywords.push keyword

    payload =
      story_id: storyID
      keywords: keywords
      entries: chosen

    console.log "[collect_diary_kag_ite] diary story:", storyID
    console.log "[collect_diary_kag_ite] pretend stories:", pretendStoryIDs?.join(', ') ? 'none'
    console.log "[collect_diary_kag_ite] chosen KAG entries:", chosen.length

    L.make 'diary_kag', payload
    L.done()
    return
