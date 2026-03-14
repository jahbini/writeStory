#!/usr/bin/env coffee
###
Step 5 — finalize: aggregate results
###
@step =
  name: 'step5_finalize'
  desc: 'Aggregate upstream results into final summary.'

  action: (M, stepName) ->
    readInput = (key) ->
      memoKey = key
      entry = M.theLowdown memoKey
      value = entry?.value
      if value is undefined
        if typeof entry?.waitFor is 'function'
          value = await entry.waitFor()
        else if entry?.notifier?
          value = await entry.notifier
      throw new Error "[#{stepName}] Missing input key '#{memoKey}'" if value is undefined
      value

    inputVal = await readInput 'input_data'
    transformedVal = await readInput 'transformed_data'
    waited = await readInput 'wait_data'

    summary =
      original:  inputVal
      doubled:   transformedVal?.doubled
      transformed: transformedVal
      waited:    waited
      timestamp: new Date().toISOString()

    M.saveThis "final_summary_json", summary
    M.saveThis "final_summary_yaml", summary
    M.saveThis "final_summary_csv", summary
    M.saveThis "done:#{stepName}", true
    console.log "[#{stepName}] wrote output artifacts final_summary_*"
    return
