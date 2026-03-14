#!/usr/bin/env coffee
###
Step 6 — curl: external network test
###
{ spawnSync } = require 'child_process'

@step =
  name: 'step6_curl'
  desc: 'Spawn a curl request and memoize its result.'

  action: (M, stepName) ->
    console.log "[#{stepName}] running curl..."
    inputKey = "final_summary_json"
    inputEntry = M.theLowdown inputKey
    inputVal = inputEntry?.value
    if inputVal is undefined
      if typeof inputEntry?.waitFor is 'function'
        inputVal = await inputEntry.waitFor()
      else if inputEntry?.notifier?
        inputVal = await inputEntry.notifier
    throw new Error "[#{stepName}] Missing input key '#{inputKey}'" if inputVal is undefined

    cmd  = 'curl'
    args = ['-sI', 'https://example.com']
    result = spawnSync(cmd, args, encoding: 'utf8')

    if result.error
      console.error "[#{stepName}] curl failed:", result.error
      M.saveThis "curl_result", { status: 'failed', error: String(result.error) }
      M.saveThis "done:#{stepName}", true
      return

    output = result.stdout.trim()
    console.log "[#{stepName}] curl completed; length:", output.length

    M.saveThis "curl_result", { status: 'ok', output }
    M.saveThis "done:#{stepName}", true
    return
