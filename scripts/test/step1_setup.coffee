#!/usr/bin/env coffee
###
Step 1 — setup: create dummy input data (no defaults)
###
@step =
  name: 'step1_setup'
  desc: 'Generate initial input data for downstream tests.'

  action: (M, stepName) ->
    greeting = M.getStepParam stepName, 'greeting'
    unless greeting?
      throw new Error "[#{stepName}] Missing required key 'greeting'"
    value = M.getStepParam stepName, 'value'
    unless value?
      throw new Error "[#{stepName}] Missing required key 'value'"

    data = { greeting, value }

    M.saveThis "input_data", data
    M.saveThis "done:#{stepName}", true
    console.log "[#{stepName}] wrote output artifact input_data"
    return
