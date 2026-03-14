#!/usr/bin/env coffee
###
Step 2 — transform: read input.json and write derived output
###
@step =
  name: 'step2_transform'
  desc: 'Transform input.json into doubled numeric output.'

  action: (M, stepName) ->
    inputKey = "input_data"
    inputEntry = M.theLowdown inputKey
    input = inputEntry?.value
    if input is undefined
      if typeof inputEntry?.waitFor is 'function'
        input = await inputEntry.waitFor()
      else if inputEntry?.notifier?
        input = await inputEntry.notifier
    unless input?
      throw new Error "[#{stepName}] Missing input key '#{inputKey}'"

    transformed =
      greeting: "#{input.greeting}, world!"
      doubled: input.value * 2

    M.saveThis "transformed_data", transformed
    M.saveThis "done:#{stepName}", true
    console.log "[#{stepName}] wrote output artifact transformed_data"
    return
