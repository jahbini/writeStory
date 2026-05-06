# rusty/meta/mlx_bridge.coffee
#
# This is a scaffold-only Memo meta device for the future resident Rust ML
# bridge. It is intentionally not installed into EXEC/meta yet.
#
# Ownership boundary:
# - JS/CoffeeScript owns orchestration only.
# - JS may hold opaque handles only.
# - JS must never directly own tensors, model weights, GPU buffers, or KV cache.
# - Rust owns ML lifetime and memory.

fs = require 'fs'
path = require 'path'
readline = require 'readline'
{ spawn } = require 'child_process'

module.exports = (M, opts = {}) ->
  baseDir = opts.baseDir ? process.cwd()
  bridgeRoot = opts.bridgeRoot ? path.join(baseDir, 'rusty', 'bridge')
  bridgeManifest = opts.bridgeManifest ? path.join(baseDir, 'rusty', 'bridge', 'Cargo.toml')
  stateDir = path.join(baseDir, 'state')
  bridgeLogPath = path.join(stateDir, 'rusty-bridge.jsonl')

  bridge =
    child: null
    rl: null
    nextId: 0
    pending: new Map()
    cache: new Map()
    started: false

  appendBridgeLog = (payload) ->
    fs.mkdirSync stateDir, { recursive: true }
    fs.appendFileSync bridgeLogPath, "#{JSON.stringify(payload)}\n", 'utf8'

  recordResult = (key, result) ->
    bridge.cache.set key, result
    entry = M.theLowdown key
    entry.value = result
    try entry.resolver?(result) catch then null
    result

  ensureBridgeStarted = ->
    return bridge.child if bridge.child?

    # Scaffold only:
    # Start a resident process shape now. The long-term target is a compiled
    # Rust binary or cargo-run dev path. For now this wrapper just knows how
    # the lifecycle will be owned.
    cmd = 'cargo'
    args = ['run', '--quiet', '--manifest-path', bridgeManifest]
    child = spawn cmd, args,
      cwd: bridgeRoot
      stdio: ['pipe', 'pipe', 'pipe']
      env: Object.assign {}, process.env

    bridge.child = child
    bridge.started = true

    bridge.rl = readline.createInterface input: child.stdout
    bridge.rl.on 'line', (line) ->
      payload = null
      try payload = JSON.parse(line) catch err
        appendBridgeLog kind:'decode_error', line: line, error: String(err?.message ? err)
        return

      id = String(payload?.id ? '')
      pending = bridge.pending.get(id)
      appendBridgeLog kind:'response', payload: payload
      return unless pending?

      bridge.pending.delete id
      if payload?.ok is true
        pending.resolve payload
      else
        pending.reject new Error(payload?.error?.message ? 'bridge request failed')

    child.stderr.on 'data', (buf) ->
      appendBridgeLog kind:'stderr', text: String(buf ? '')

    child.on 'exit', (code, signal) ->
      appendBridgeLog kind:'exit', code: code, signal: signal
      for [id, pending] from bridge.pending
        pending.reject new Error("rusty bridge exited before replying to #{id}")
      bridge.pending.clear()
      bridge.child = null
      bridge.rl = null
      bridge.started = false

    child

  submitRequest = (cmd, args = {}) ->
    ensureBridgeStarted()
    bridge.nextId += 1
    id = "mlx:#{bridge.nextId}"
    envelope =
      id: id
      cmd: cmd
      args: args

    appendBridgeLog kind:'request', payload: envelope

    p = new Promise (resolve, reject) ->
      bridge.pending.set id, { resolve, reject }
      bridge.child.stdin.write "#{JSON.stringify(envelope)}\n", 'utf8'

    p

  parseCommandFromKey = (key) ->
    # Suggested future key forms:
    # - mlx/bridge_health.json
    # - mlx/load_model/request.json
    # - mlx/generate/test.json
    #
    # The key remains the visible DAG artifact. The value is the command args.
    normalized = String(key ? '').replace /\\/g, '/'
    parts = normalized.split('/').filter (part) -> part.length > 0
    return '' unless parts.length >= 2 and parts[0] is 'mlx'
    parts[1]

  M.addMetaRule 'mlx_bridge',
    /^mlx\//,
    (key, value) ->
      return bridge.cache.get(key) if value is undefined

      cmd = parseCommandFromKey(key)
      args = if value? and typeof value is 'object' and not Array.isArray(value) then value else {}

      # Memo remains the visibility layer. We do not add a new Memo API here.
      # Instead we submit a request, then resolve the existing entry when the
      # resident bridge replies.
      submitRequest(cmd, args)
        .then (payload) ->
          recordResult key, payload
        .catch (err) ->
          recordResult key,
            ok: false
            error:
              message: String(err?.message ? err)

      {
        submitted: true
        via: 'rusty-bridge'
        cmd: cmd
      }

  process.once 'exit', ->
    return unless bridge.child?
    try
      bridge.child.stdin.write JSON.stringify(id:'mlx:shutdown', cmd:'bridge_shutdown', args:{}) + "\n"
    catch then null
