#!/usr/bin/env coffee
###
Step 7 — python: external interpreter test
###
{ spawnSync } = require 'child_process'
fs = require 'fs'
path = require 'path'

resolvePython = ->
  candidates = [
    process.env.PYTHON
    path.join(process.cwd(), '.venv', 'bin', 'python')
    path.join(process.cwd(), '.venv', 'bin', 'python3')
    'python3'
    'python'
  ].filter(Boolean)

  for candidate in candidates
    if candidate.includes(path.sep)
      return candidate if fs.existsSync(candidate)
    else
      return candidate

  'python'

@step =
  name: 'step7_python'
  desc: 'Run Python interpreter and capture version.'

  action: (M, stepName) ->
    console.log "[#{stepName}] querying Python version..."
    inputKey = "curl_result"
    inputEntry = M.theLowdown inputKey
    inputVal = inputEntry?.value
    if inputVal is undefined
      if typeof inputEntry?.waitFor is 'function'
        inputVal = await inputEntry.waitFor()
      else if inputEntry?.notifier?
        inputVal = await inputEntry.notifier
    throw new Error "[#{stepName}] Missing input key '#{inputKey}'" if inputVal is undefined

    cmd  = resolvePython()
    args = ['-V']
    result = spawnSync(cmd, args, encoding: 'utf8')

    if result.error
      console.error "[#{stepName}] Python failed:", result.error
      M.saveThis "python_result", { status: 'failed', error: String(result.error) }
      M.saveThis "done:#{stepName}", true
      return

    output = (result.stdout or result.stderr).trim()
    console.log "[#{stepName}] Python responded:", output

    M.saveThis "python_result", { status: 'ok', version: output }
    M.saveThis "done:#{stepName}", true
    return
