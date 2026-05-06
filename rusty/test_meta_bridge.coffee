#!/usr/bin/env coffee

fs = require 'fs'
path = require 'path'
vm = require 'vm'
CoffeeScript = null
try
  CoffeeScript = require 'coffeescript'
catch err
  console.error "[rusty test-meta] CoffeeScript runtime is required:", String(err?.message ? err)
  process.exit 2

{ spawnSync } = require 'child_process'

ROOT = path.resolve __dirname, '..'
PIPELINE_RUNNER_PATH = path.join ROOT, 'pipeline_runner.coffee'
MLX_BRIDGE_META_PATH = path.join __dirname, 'meta', 'mlx_bridge.coffee'

hasCargo = ->
  probe = spawnSync 'cargo', ['--version'], stdio: 'ignore'
  probe.status is 0

loadMemoClass = ->
  source = fs.readFileSync PIPELINE_RUNNER_PATH, 'utf8'
  start = source.indexOf 'class Memo'
  end = source.indexOf '# -------------------------------------------------------------------\n# Experiment + DAG'
  throw new Error 'could not locate Memo class in pipeline_runner.coffee' if start < 0 or end < 0 or end <= start

  memoChunk = source.slice start, end
  wrapper = """
#{memoChunk}
module.exports = Memo
"""

  compiled = CoffeeScript.compile wrapper, bare: true
  context =
    module: { exports: {} }
    exports: {}
    require: require
    console: console
    process: process
    setTimeout: setTimeout
    clearTimeout: clearTimeout
    Promise: Promise
  vm.createContext context
  vm.runInContext compiled, context, filename: 'memo_extract.js'
  context.module.exports

waitForDefined = (entry, timeoutMs = 10000) ->
  Promise.race [
    new Promise (resolve, reject) ->
      timer = setTimeout (-> reject new Error "timeout waiting for Memo entry"), timeoutMs
      Promise.resolve(entry.notifier)
        .then (value) ->
          clearTimeout timer
          resolve value
        .catch (err) ->
          clearTimeout timer
          reject err
    new Promise (resolve, reject) ->
      poll = ->
        if entry.value? and entry.value.submitted isnt true
          resolve entry.value
          return
        setTimeout poll, 25
      poll()
  ]

shutdownBridge = (M) ->
  key = 'mlx/bridge_shutdown/test.json'
  entry = M.saveThis key, {}
  await waitForDefined entry, 5000

main = ->
  unless hasCargo()
    console.log '[rusty test-meta] cargo not found; skipping Memo/meta bridge test'
    process.exit 0

  Memo = loadMemoClass()
  throw new Error 'Memo class load failed' unless typeof Memo is 'function'

  bridgeStateDir = path.join ROOT, 'state'
  fs.mkdirSync bridgeStateDir, recursive: true

  M = new Memo()
  bridgeMeta = require MLX_BRIDGE_META_PATH
  bridgeMeta M, baseDir: ROOT

  key = 'mlx/generate/test.json'
  req =
    model: 'model:fake'
    session: 'sess:test'
    prompt: 'hello'

  entry = M.saveThis key, req
  result = await waitForDefined entry

  try
    throw new Error 'expected ok:true' unless result?.ok is true
    throw new Error 'expected mocked generated text' unless /\[stub generate\] hello/.test String(result?.value?.text ? '')
    console.log '[rusty test-meta] Memo/meta bridge test passed'
  finally
    try
      await shutdownBridge M
    catch err
      console.error '[rusty test-meta] bridge shutdown warning:', String(err?.message ? err)

main().catch (err) ->
  console.error '[rusty test-meta] failed:', String(err?.message ? err)
  process.exit 1
