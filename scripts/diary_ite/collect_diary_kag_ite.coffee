wordSet = (text) ->
  words = String(text ? '').toLowerCase().match(/[a-z0-9]+/g) ? []
  new Set(words)

scoreEntry = (entry, eventWords) ->
  words = wordSet("#{entry?.keyword ? ''} #{entry?.headline ? ''}")
  score = 0
  score += 3 if String(entry?.keyword ? '').trim().length and eventWords.has(String(entry.keyword).toLowerCase())
  for word in words
    score += 1 if eventWords.has(word)
  score

@step =
  desc: "Collect the KAG entries that best fit the selected diary events"

  action: (M, stepName) ->
    storyEntry = M.theLowdown 'selected_story_id'
    storyID = storyEntry?.value
    if storyID is undefined
      if typeof storyEntry?.waitFor is 'function'
        storyID = await storyEntry.waitFor()
      else if storyEntry?.notifier?
        storyID = await storyEntry.notifier

    eventsEntry = M.theLowdown 'diary_events'
    diaryEvents = eventsEntry?.value
    if diaryEvents is undefined
      if typeof eventsEntry?.waitFor is 'function'
        diaryEvents = await eventsEntry.waitFor()
      else if eventsEntry?.notifier?
        diaryEvents = await eventsEntry.notifier

    throw new Error "[#{stepName}] selected_story_id must be a string" unless typeof storyID is 'string'
    throw new Error "[#{stepName}] diary_events must be an object" unless diaryEvents?.events?

    kagEntry = M.theLowdown "kagFor{#{storyID}}.json"
    kag = kagEntry?.value
    if kag is undefined
      if typeof kagEntry?.waitFor is 'function'
        kag = await kagEntry.waitFor()
      else if kagEntry?.notifier?
        kag = await kagEntry.notifier

    throw new Error "[#{stepName}] Missing sqlite KAG #{storyID}" unless Array.isArray(kag?.entries)

    eventWords = new Set()
    for event in diaryEvents.events
      for word in wordSet("#{event?.kind ? ''} #{event?.keyword ? ''} #{event?.headline ? ''} #{event?.text ? ''}")
        eventWords.add word

    scored = []
    for entry, idx in kag.entries
      scored.push
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

    console.log "[collect_diary_kag_ite] story:", storyID
    console.log "[collect_diary_kag_ite] chosen KAG entries:", chosen.length

    M.saveThis 'diary_kag', payload
    M.saveThis "done:#{stepName}", true
    return
