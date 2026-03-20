@step =
  desc: "Build structured prompt-cache spec from story and KAG fields"

  action: (M, stepName) ->
    kagEntry = M.theLowdown 'kag_record'
    kag = kagEntry?.value
    if kag is undefined
      if typeof kagEntry?.waitFor is 'function'
        kag = await kagEntry.waitFor()
      else if kagEntry?.notifier?
        kag = await kagEntry.notifier
    throw new Error "[#{stepName}] Missing input key 'kag_record'" if kag is undefined

    storyKey = M.getStepParam(stepName, 'story_key')
    storyFragment = M.getStepParam(stepName, 'story_fragment')
    storyText = ''
    storyId = kag?.story_id ? null

    if storyKey?
      storyEntry = M.theLowdown storyKey
      story = storyEntry?.value
      if story is undefined
        if typeof storyEntry?.waitFor is 'function'
          story = await storyEntry.waitFor()
        else if storyEntry?.notifier?
          story = await storyEntry.notifier
      throw new Error "[#{stepName}] Missing input key '#{storyKey}'" if story is undefined
      storyText = story?.text ? ''
      storyId = story?.story_id ? storyId
    else
      storyText = storyFragment ? ''

    out =
      story_id: storyId
      stable_instructions: [
        "You are writing in the narrative voice of Jim from St. John's."
        "Expand the following story fragment into a short reflective narrative of at least 500 words."
        "Maintain the same events and ideas, but improve flow, imagery, and voice."
      ]
      rules: [
        "Speak in the first person as Jim"
        "Keep the same order of events."
        "Do not introduce new plot elements."
        "Add natural narration and sensory detail."
        "The tone should be observational, slightly humorous, and reflective."
        "The final length should be about 800–2000 words."
        "Return only the finished story."
      ]
      story_template:
        label: "Story fragment"
        text: storyText
      kag_fields: kag?.fields ? {}

    M.saveThis "prompt_cache_spec", out
    M.saveThis "done:#{stepName}", true
    return
