#!/usr/bin/env coffee
###
Step 3 — table: generate CSV summary
###
@step =
  name: 'step3_table'
  desc: 'Create tabular summary from transformed data.'

  action: (M, stepName) ->
    transformedKey = "transformed_data"
    transformedEntry = M.theLowdown transformedKey
    transformed = transformedEntry?.value
    if transformed is undefined
      if typeof transformedEntry?.waitFor is 'function'
        transformed = await transformedEntry.waitFor()
      else if transformedEntry?.notifier?
        transformed = await transformedEntry.notifier
    unless transformed?
      throw new Error "[#{stepName}] Missing input key '#{transformedKey}'"

    row =
      greeting: transformed.greeting
      doubled: transformed.doubled

    M.saveThis "summary_row", row
    M.saveThis "done:#{stepName}", true
    console.log "[#{stepName}] wrote output artifact summary_row"
    return
