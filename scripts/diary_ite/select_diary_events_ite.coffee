eventTextFor = (storyTitle, entry, idx) ->
  headline = String(entry?.headline ? '').trim()
  keyword = String(entry?.keyword ? '').trim()
  fallback = switch idx
    when 0 then "the day around #{storyTitle} already felt slightly off"
    when 1 then "someone or something arrived and shifted the mood"
    when 2 then "the quiet shape of the day gave way to trouble"
    when 3 then "I kept turning the moment over in my head"
    else "something about the day finally became clear"

  return fallback unless headline.length or keyword.length
  return "#{headline} while #{keyword} stayed close to everything" if headline.length and keyword.length
  return headline if headline.length
  "#{storyTitle} felt marked by #{keyword}"

@step =
  desc: "Select a story and derive diary events from its SQLite-backed KAG"

  action: (M, stepName) ->
    storyID = M.getStepParam(stepName, 'story_id')
    throw new Error "[#{stepName}] Missing required param 'story_id'" unless storyID?

    storyEntry = M.theLowdown "storyByID{#{storyID}}.json"
    story = storyEntry?.value
    if story is undefined
      if typeof storyEntry?.waitFor is 'function'
        story = await storyEntry.waitFor()
      else if storyEntry?.notifier?
        story = await storyEntry.notifier

    kagEntry = M.theLowdown "kagFor{#{storyID}}.json"
    kag = kagEntry?.value
    if kag is undefined
      if typeof kagEntry?.waitFor is 'function'
        kag = await kagEntry.waitFor()
      else if kagEntry?.notifier?
        kag = await kagEntry.notifier

    throw new Error "[#{stepName}] Missing sqlite story #{storyID}" unless story?.text?
    throw new Error "[#{stepName}] Missing sqlite KAG #{storyID}" unless Array.isArray(kag?.entries) and kag.entries.length > 0

    storyTitle = String(story.title ? storyID).trim()
    kinds = ['scene', 'arrival', 'disturbance', 'reflection', 'realization']
    events = []

    for kind, idx in kinds
      entry = kag.entries[idx] ? kag.entries[kag.entries.length - 1]
      events.push
        kind: kind
        keyword: String(entry?.keyword ? '').trim()
        headline: String(entry?.headline ? '').trim()
        text: eventTextFor(storyTitle, entry, idx)

    payload =
      story_id: storyID
      title: storyTitle
      events: events
      source_keywords: (String(keyword ? '').trim() for keyword in (kag.keywords ? []) when String(keyword ? '').trim().length > 0)

    console.log "[select_diary_events_ite] story:", storyID
    console.log "[select_diary_events_ite] event count:", events.length

    M.saveThis 'selected_story_id', storyID
    M.saveThis 'diary_events', payload
    M.saveThis "done:#{stepName}", true
    return
