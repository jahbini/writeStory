renderEvent = (event) ->
  kind = String(event?.kind ? '').trim()
  text = String(event?.text ? '').trim()
  keyword = String(event?.keyword ? '').trim()
  headline = String(event?.headline ? '').trim()
  lines = []
  lines.push "- #{kind}: #{text}" if kind.length or text.length
  lines.push "  keyword: #{keyword}" if keyword.length
  lines.push "  headline: #{headline}" if headline.length
  lines.join "\n"

renderKagEntry = (entry) ->
  keyword = String(entry?.keyword ? '').trim()
  headline = String(entry?.headline ? '').trim()
  return "- #{keyword}: #{headline}" if keyword.length and headline.length
  return "- #{headline}" if headline.length
  return "- #{keyword}" if keyword.length
  "- unlabelled KAG cue"

@step =
  desc: "Build the final diary prompt from diary events and matched KAG"

  action: (M, stepName) ->
    storyIDEntry = M.theLowdown 'selected_story_id'
    storyID = storyIDEntry?.value
    if storyID is undefined
      if typeof storyIDEntry?.waitFor is 'function'
        storyID = await storyIDEntry.waitFor()
      else if storyIDEntry?.notifier?
        storyID = await storyIDEntry.notifier

    eventsEntry = M.theLowdown 'diary_events'
    diaryEvents = eventsEntry?.value
    if diaryEvents is undefined
      if typeof eventsEntry?.waitFor is 'function'
        diaryEvents = await eventsEntry.waitFor()
      else if eventsEntry?.notifier?
        diaryEvents = await eventsEntry.notifier

    kagEntry = M.theLowdown 'diary_kag'
    diaryKag = kagEntry?.value
    if diaryKag is undefined
      if typeof kagEntry?.waitFor is 'function'
        diaryKag = await kagEntry.waitFor()
      else if kagEntry?.notifier?
        diaryKag = await kagEntry.notifier

    storyEntry = M.theLowdown "storyByID{#{storyID}}.json"
    story = storyEntry?.value
    if story is undefined
      if typeof storyEntry?.waitFor is 'function'
        story = await storyEntry.waitFor()
      else if storyEntry?.notifier?
        story = await storyEntry.notifier

    throw new Error "[#{stepName}] selected_story_id must be a string" unless typeof storyID is 'string'
    throw new Error "[#{stepName}] diary_events must be an object" unless Array.isArray(diaryEvents?.events)
    throw new Error "[#{stepName}] diary_kag must be an object" unless Array.isArray(diaryKag?.entries)
    throw new Error "[#{stepName}] Missing sqlite story #{storyID}" unless story?.text?

    eventLines = (renderEvent(event) for event in diaryEvents.events when event?).filter(Boolean)
    kagLines = (renderKagEntry(entry) for entry in diaryKag.entries when entry?).filter(Boolean)

    prompt = [
      "You are writing in the narrative voice of Jim from St. John's."
      ""
      "Write a diary entry in first person."
      "Use the diary events as the backbone of the entry."
      "Use the KAG cues as emotional guidance, but keep the entry grounded and concrete."
      ""
      "Rules:"
      "- Keep the events in the listed order"
      "- Do not introduce plot contradictions"
      "- Add sensory detail and reflective narration"
      "- Keep the voice observational, slightly humorous, and reflective"
      "- Return only the finished diary entry"
      ""
      "Story source:"
      "#{story.title ? storyID}"
      ""
      "Diary events:"
      if eventLines.length then eventLines.join("\n") else "- none"
      ""
      "KAG cues:"
      if kagLines.length then kagLines.join("\n") else "- none"
      ""
      "Source text for grounding:"
      ""
      "#{String(story.text ? '').trim()}"
    ].join "\n"

    console.log "[build_diary_prompt_ite] story:", storyID
    console.log "[build_diary_prompt_ite] prompt chars:", prompt.length

    M.saveThis 'diary_prompt_text', prompt
    M.saveThis "done:#{stepName}", true
    return
