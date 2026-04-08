@step =
  desc: "Select deterministic story recipe by story_id"

  action: (M, stepName) ->
    storyId = M.getStepParam(stepName, 'story_id')
    overrideScene = M.getStepParam(stepName, 'scene')
    overrideArrival = M.getStepParam(stepName, 'arrival')
    overrideDisturbance = M.getStepParam(stepName, 'disturbance')
    overrideReflection = M.getStepParam(stepName, 'reflection')
    overrideRealization = M.getStepParam(stepName, 'realization')
    libraryKey = "story_library"
    libraryEntry = M.theLowdown libraryKey
    bundle = libraryEntry?.value
    if bundle is undefined
      if typeof libraryEntry?.waitFor is 'function'
        bundle = await libraryEntry.waitFor()
      else if libraryEntry?.notifier?
        bundle = await libraryEntry.notifier
    throw new Error "[#{stepName}] Missing input key '#{libraryKey}'" if bundle is undefined

    stories = bundle?.stories ? {}
    selected = stories?[storyId]

    unless selected?
      known = Object.keys(stories)
      throw new Error "[#{stepName}] story_id '#{storyId}' not found. Known: #{known.join(', ')}"

    recipe = Object.assign {}, selected
    recipe.scene = overrideScene if overrideScene?
    recipe.arrival = overrideArrival if overrideArrival?
    recipe.disturbance = overrideDisturbance if overrideDisturbance?
    recipe.reflection = overrideReflection if overrideReflection?
    recipe.realization = overrideRealization if overrideRealization?

    out =
      story_id: storyId
      recipe: recipe

    M.saveThis "story_recipe", out
    M.saveThis "done:#{stepName}", true
    return
