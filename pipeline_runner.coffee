#!/usr/bin/env coffee
###
pipeline_runner.coffee a micro OS for AI 
===========================================================================

Hard guarantees:
• callMLX EXISTS and is syncronous via the pipeline, and not in the memo
• Memo meta-dispatch preserved (read + write)
• experiment.yaml is saved into Memo BEFORE any step runs
• Step params are saved into Memo BEFORE any step runs
• State protocol:
    - One file per step: state/step-<name>.json
    - State is consulted ONLY at startup
    - Runner records running/done/failed for each step
    - restart_here is consumed at startup; downstream state is DELETED (so old "done" can’t inhibit reruns)
• CRITICAL FIX:
    - Memo.saveThis resolves notifier for boolean values EVERY TIME (not just first write)
###

fs        = require 'fs'
path      = require 'path'
yaml      = require 'js-yaml'
{ spawn, execSync } = require 'child_process'
CoffeeScript = require 'coffeescript'
CoffeeScript.register()

EXEC = process.env.EXEC ? path.dirname(__filename)
CWD  = process.cwd()

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
banner = (msg) -> console.log "\n=== #{msg} ==="
prefixLines = (pfx, s) -> (s ? '').split(/\r?\n/).map((l)-> pfx + l).join("\n")
isPlainObject = (o) -> Object.prototype.toString.call(o) is '[object Object]'

resolvePython = (baseDir = CWD) ->
  candidates = [
    process.env.PYTHON
    path.join(baseDir, '.venv', 'bin', 'python')
    path.join(baseDir, '.venv', 'bin', 'python3')
    'python3'
    'python'
  ].filter(Boolean)

  for candidate in candidates
    if candidate.includes(path.sep)
      return candidate if fs.existsSync(candidate)
    else
      return candidate

  'python'

deepMerge = (target, source) ->
  return target unless source?
  for own k, v of source
    if v is null
      delete target[k]; continue
    if isPlainObject(v) and isPlainObject(target[k])
      deepMerge target[k], v
    else
      target[k] = Array.isArray(v) and v.slice() or v
  target

loadYamlSafe = (p) ->
  return {} unless p? and fs.existsSync(p)
  yaml.load fs.readFileSync(p,'utf8') or {}

expandIncludes = (spec, baseDir) ->
  incs = spec.include
  return spec unless incs? and Array.isArray(incs)
  merged = JSON.parse(JSON.stringify(spec))
  for inc in incs
    incPath = path.isAbsolute(inc) and inc or path.join(baseDir, inc)
    merged = deepMerge merged, loadYamlSafe(incPath)
  merged

ensureSingleInstance = ->
  try
    scriptPath = path.resolve(__filename)
    out = execSync("ps -Ao pid,command | grep 'coffee' | grep '#{scriptPath}' | grep -v grep || true").toString()
    lines = out.trim().split("\n").filter (l)-> l.length>0
    others = lines.filter (l)-> not l.startsWith(process.pid.toString())
    if others.length>0 then process.exit(0)
  catch then null

# -------------------------------------------------------------------
# State Directory: One file per step
# -------------------------------------------------------------------
class StepStateStore
  constructor: (@dir) ->
    fs.mkdirSync(@dir, {recursive:true})

  _pathFor: (n) -> path.join(@dir, "step-#{n}.json")

  read: (n) ->
    p = @_pathFor(n)
    return null unless fs.existsSync(p)
    try JSON.parse(fs.readFileSync(p,'utf8')) catch then null

  write: (n, obj) ->
    payload = Object.assign {}, obj,
      step: n
      updated_at: new Date().toISOString()
    fs.writeFileSync @_pathFor(n), JSON.stringify(payload, null, 2), 'utf8'
    payload

  delete: (n) ->
    p = @_pathFor(n)
    return false unless fs.existsSync(p)
    fs.unlinkSync(p)
    true

  markRunning: (n) ->
    @write n,
      status: 'running'
      done: false
      started_at: new Date().toISOString()

  markDone: (n, extra={}) ->
    @write n, Object.assign {}, extra,
      status: 'done'
      done: true
      dirty: false
      finished_at: new Date().toISOString()

  markFailed: (n, errMsg, extra={}) ->
    @write n, Object.assign {}, extra,
      status: 'failed'
      done: false
      error: String(errMsg ? 'unknown error')
      finished_at: new Date().toISOString()

  clearRestartHere: (n) ->
    st = @read(n)
    return unless st?.restart_here is true
    st.restart_here = false
    st.restart_consumed_at = new Date().toISOString()
    @write n, st

  writePipelineShutdown: (info) ->
    payload =
      status: 'shutdown'
      by: info.by
      reason: info.reason
      timestamp: info.timestamp ? new Date().toISOString()
    fs.writeFileSync(
      path.join('.', 'pipeline.json'),
      JSON.stringify(payload, null, 2),
      'utf8'
    )

  readPipeline: ->
    p = path.join('.', 'pipeline.json')
    return null unless fs.existsSync(p)
    JSON.parse fs.readFileSync(p,'utf8')
# -------------------------------------------------------------------
# Memo with Meta-Dispatcher (CALLMLX PRESERVED)
# -------------------------------------------------------------------
class Memo
  constructor: ->
    @MM = {}
    @metaRules = []

  _newEntry: (key, value) ->
    breaker = null
    p = new Promise (resolve) -> breaker = resolve
    entry =
      value: value
      notifier: p
      resolver: breaker
      meta: @selectMetaHandler(key)
    entry

  _resolve: (entry, value) ->
    try entry.resolver?(value) catch then null

  saveThis: (key, value) ->
    entry = @MM[key]
    unless entry?
      entry = @_newEntry(key, value)
      @MM[key] = entry
      try rv = entry.meta(key, value) catch then null
      entry.value = rv if rv?
      @_resolve(entry, value) if value is true or value is false
      return entry

    old = entry.resolver
    entry = @MM[key] = @_newEntry(key, value)
    try old?(value) catch then null
    try rv = entry.meta(key, value) catch then null
    entry.value = rv if rv?
    @_resolve(entry, value) if value is true or value is false   # <<< CRITICAL FIX
    entry.notifier.then (nv) -> entry.value = nv if nv?
    entry

  theLowdown: (key) ->
    entry = @MM[key]
    unless entry?
      entry = @_newEntry(key, undefined)
      @MM[key] = entry
      try rv = entry.meta(key, undefined) catch then null
      if rv?
        entry.value = rv
        @_resolve(entry, rv)
      return entry

    if entry.value is undefined
      try rv = entry.meta(key, undefined) catch then null
      if rv?
        entry.value = rv
        @_resolve(entry, rv)
    entry

  waitFor: (keys, andDo) ->
    entries = ( @theLowdown(k) for k in keys )
    return if entries.some((e)-> e.value is false)
    if entries.every((e)-> e.value is true)
      try andDo() catch then null
      return
    Promise.all(entries.map((e)-> e.notifier)).then =>
      return if keys.some((k)=> @theLowdown(k).value is false)
      try andDo() catch then null

  addMetaRule: (name, regex, handler) ->
    @metaRules.push {name, regex, handler}

  selectMetaHandler: (key) ->
    for r in @metaRules when r.regex.test(key)
      return r.handler
    (k,v)-> return

  # ------------------------------------------------------------
  # Parameter resolution (authoritative)
  # ------------------------------------------------------------
  getStepParam: (stepName, key, defaultValue = undefined) ->
    stepParams =
      @theLowdown("params/#{stepName}.yaml").value ? {}

    globalParams =
      @theLowdown("params/_global.yaml").value ? {}

    if stepParams.hasOwnProperty key
      return stepParams[key]

    if globalParams.hasOwnProperty key
      return globalParams[key]

    return defaultValue

  callMLX: (cmdType, payload, dbug = false) ->
    buildArgs = (cmdType, params) ->
      args = ['-m','mlx_lm',cmdType]
      for k,v of params
        args.push "--#{k}" if k
        args.push v if v?
      args

    args = buildArgs(cmdType, payload)
    console.error "MLX args",args if dbug
    spawnSync = require('child_process').spawnSync
    res = spawnSync resolvePython(CWD), args, {encoding:'utf8'}
    console.error "MLX result" ,res if dbug
    
    if res.error?
      throw res.error
    if res.status isnt 0
      throw new Error "MLX failed: #{res.stderr ? res.stdout ? "exit #{res.status}"}"
    res.stdout

# -------------------------------------------------------------------
# Experiment + DAG
# -------------------------------------------------------------------
createExperimentObject = (configPath, overridePath) ->
  recipe = expandIncludes loadYamlSafe(configPath), path.dirname(configPath)
  merged = deepMerge {}, recipe
  merged = deepMerge merged, loadYamlSafe(overridePath)
  return merged

normalizeDeps = (d) ->
  return [] unless d?
  return d.slice() if Array.isArray(d)
  return [d] if typeof d is 'string'
  []

normalizeArtifactKeys = (d) ->
  return [] unless d?
  return d.slice() if Array.isArray(d)
  return [d] if typeof d is 'string'
  throw new Error "needs/makes must be string or array"

discoverSteps = (spec) ->
  steps = {}
  for own k, v of spec
    continue unless isPlainObject(v)
    continue unless v.run? or v.run_mlx
    def = Object.assign {}, v
    deps = normalizeDeps(v.depends_on)
    if deps.length is 1 and String(deps[0]).toLowerCase() is 'never'
      console.log "⏭️  skipping step #{k} (depends_on: never)"
      continue
    def.depends_on = deps
    steps[k] = def
  steps

toposort = (steps) ->
  indeg = {}; g = {}
  for own n of steps
    indeg[n]=0; g[n]=[]
  for own n, d of steps
    for dep in (d.depends_on or [])
      throw new Error "Undefined dependency '#{dep}' (by '#{n}')" unless steps[dep]?
      indeg[n] += 1
      g[dep].push n
  q = (n for own n,d of indeg when d is 0)
  o = []
  while q.length
    n = q.shift()
    o.push n
    for m in g[n]
      indeg[m] -= 1
      q.push(m) if indeg[m] is 0
  if o.length isnt Object.keys(steps).length
    throw new Error "Topo sort failed (cycle?)"
  o

downstreamMap = (steps) ->
  g = {}
  for own n of steps then g[n]=[]
  for own n, d of steps
    for dep in (d.depends_on or [])
      g[dep].push n
  g

collectDownstream = (g, start) ->
  seen = new Set()
  stack = [start]
  while stack.length
    n = stack.pop()
    continue if seen.has(n)
    seen.add(n)
    for c in (g[n] or []) then stack.push c
  Array.from(seen)

terminalSteps = (steps) ->
  hasDependent = new Set()
  for own n, d of steps
    for dep in (d.depends_on or []) then hasDependent.add(dep)
  (n for own n of steps when not hasDependent.has(n))

# -------------------------------------------------------------------
# Step Runner (SACRED new-style loader preserved)
# -------------------------------------------------------------------
isNewStyleStep = (p) ->
  try /\@step\s*=/.test fs.readFileSync(p,'utf8') catch then false

createStepLedger = (memo, stepName, resolveArtifact, artifactSpecFor) ->
  getDecls = ->
    stepP = memo.theLowdown("params/#{stepName}.yaml")?.value ? {}
    needs = normalizeArtifactKeys(stepP.needs ? [])
    makes = normalizeArtifactKeys(stepP.makes ? [])
    { needs, makes }

  debugEnabled = ->
    memo.getStepParam(stepName, 'debug_s') is true

  describeArtifact = (artifactKey) ->
    spec = artifactSpecFor artifactKey
    return artifactKey unless spec?
    if isPlainObject(spec)
      source = spec.source ? spec.key
      target = spec.target
      return "#{artifactKey} source=#{source}" if source?
      return "#{artifactKey} target=#{target}" if target?
      return artifactKey
    "#{artifactKey} source=#{spec}"

  debug = (parts...) ->
    return unless debugEnabled()
    console.log "[#{new Date().toISOString()}] [S #{stepName}]", parts...

  getStepParams = ->
    memo.theLowdown("params/#{stepName}.yaml")?.value ? {}

  legacyMlxAliases =
    generate:
      temperature: 'temp'
      max_tokens: 'max-tokens'
      min_tokens: 'min-tokens'
      top_p: 'top-p'
      top_k: 'top-k'
      max_kv_size: 'max-kv-size'
      repeat_penalty: 'repeat-penalty'
    lora:
      batch_size: 'batch-size'
      iters: 'iters'
      max_seq_length: 'max-seq-length'
      learning_rate: 'learning-rate'
    convert:
      q_bits: 'q-bits'
      q_group: 'q-group-size'
      dtype: 'dtype'
    fuse:
      dtype: 'dtype'

  getLegacyMlxConfig = (cmdType) ->
    aliases = legacyMlxAliases[cmdType] ? {}
    legacy = {}
    for own paramKey, cliKey of aliases
      value = memo.getStepParam stepName, paramKey
      legacy[cliKey] = value if value isnt undefined
    legacy

  getMlxConfig = (cmdType) ->
    stepParams = getStepParams()
    mlxCfg = stepParams.mlx
    merged = getLegacyMlxConfig(cmdType)
    return merged unless isPlainObject(mlxCfg)
    genericCfg = {}
    commandCfg = {}
    for own k, v of mlxCfg
      continue if isPlainObject(v)
      genericCfg[k] = v
    if isPlainObject(mlxCfg[cmdType])
      commandCfg = Object.assign {}, mlxCfg[cmdType]
    Object.assign merged, genericCfg, commandCfg

  mergeMlxPayload = (cmdType, payload) ->
    merged = getMlxConfig(cmdType)
    for own k, v of (payload ? {})
      merged[k] = v
    merged

  ledger =
    stepName: stepName

    param: (key, defaultValue) ->
      debug "param request", key
      value = memo.getStepParam stepName, key
      if value is undefined and arguments.length >= 2
        debug "param default", key, defaultValue
        return defaultValue
      if value is undefined
        debug "param missing", key
        console.error "[#{stepName}] Missing required param '#{key}'"
        throw new Error "[#{stepName}] Missing required param '#{key}'"
      debug "param resolved", key, "(#{typeof value})", value
      value

    getStepParam: (nameOrKey, key, defaultValue = undefined) ->
      if arguments.length >= 2
        return memo.getStepParam(nameOrKey, key, defaultValue)
      @param(nameOrKey, defaultValue)

    need: (artifactKey) ->
      { needs, makes } = getDecls()
      debug "need request", describeArtifact(artifactKey), "declared needs=", needs.join(','), "makes=", makes.join(',')
      declared = needs.includes(artifactKey) or makes.includes(artifactKey)
      throw new Error "[#{stepName}] Artifact '#{artifactKey}' must be declared in needs or makes" unless declared

      entry = memo.theLowdown artifactKey
      value = entry?.value
      if value is undefined and needs.includes(artifactKey)
        debug "need waiting", describeArtifact(artifactKey)
        value = await entry.notifier
        debug "need awakened", describeArtifact(artifactKey), "(#{typeof value})"

      if value is undefined
        debug "need missing", describeArtifact(artifactKey)
        console.error "[#{stepName}] Missing required artifact '#{artifactKey}'"
        throw new Error "[#{stepName}] Missing required artifact '#{artifactKey}'"
      debug "need resolved", describeArtifact(artifactKey), "(#{typeof value})"
      value

    peek: (artifactKey, defaultValue = undefined) ->
      { needs, makes } = getDecls()
      debug "peek request", describeArtifact(artifactKey), "declared needs=", needs.join(','), "makes=", makes.join(',')
      declared = needs.includes(artifactKey) or makes.includes(artifactKey)
      throw new Error "[#{stepName}] Artifact '#{artifactKey}' must be declared in needs or makes" unless declared

      entry = memo.theLowdown artifactKey
      value = entry?.value
      if value is undefined
        spec = artifactSpecFor artifactKey
        source = if isPlainObject(spec) then spec.source ? spec.key else spec
        target = if isPlainObject(spec) then spec.target else null

        if typeof source is 'string'
          debug "peek checking source key", source
          srcEntry = memo.theLowdown source
          value = srcEntry?.value
          if value isnt undefined
            memo.saveThis artifactKey, value
            debug "peek resolved from source key", describeArtifact(artifactKey), "(#{typeof value})"

        if value is undefined and typeof target is 'string'
          debug "peek checking target key", target
          targetEntry = memo.theLowdown target
          value = targetEntry?.value
          if value isnt undefined
            memo.saveThis artifactKey, value
            debug "peek resolved from target key", describeArtifact(artifactKey), "(#{typeof value})"

      if value is undefined
        debug "peek default", describeArtifact(artifactKey), defaultValue
        return defaultValue
      debug "peek resolved", describeArtifact(artifactKey), "(#{typeof value})"
      value

    make: (artifactKey, value) ->
      { makes } = getDecls()
      debug "make request", describeArtifact(artifactKey), "declared makes=", makes.join(',')
      throw new Error "[#{stepName}] Artifact '#{artifactKey}' must be declared in makes" unless makes.includes artifactKey
      memo.saveThis artifactKey, value
      debug "make wrote", describeArtifact(artifactKey), "(#{typeof value})"
      value

    done: ->
      debug "done"
      memo.saveThis "done:#{stepName}", true
      true

    fail: (err) ->
      debug "fail", String(err?.message ? err)
      memo.saveThis "done:#{stepName}", false
      throw err

    saveThis: (key, value) -> memo.saveThis key, value
    theLowdown: (key) -> memo.theLowdown key
    waitFor: (keys, andDo) -> memo.waitFor keys, andDo
    addMetaRule: (name, regex, handler) -> memo.addMetaRule name, regex, handler
    callMLX: (cmdType, payload, dbug) ->
      mlxDebug = if arguments.length >= 3 then dbug else @param('debug_mlx', false)
      finalPayload = mergeMlxPayload cmdType, payload
      debug "mlx request", cmdType, finalPayload
      memo.callMLX cmdType, finalPayload, mlxDebug

  ledger

runStep = (n, def, exp, M, S, active, resolveArtifact, artifactSpecFor) ->
  new Promise (res, rej) ->
    active.count += 1
    active.names ?= new Set()
    active.names.add n
    S.markRunning n

    finish = (ok, errMsg=null) ->
      active.count -= 1
      active.names?.delete n
      if ok
        wantsRestart = M.theLowdown("restart_here:#{n}")?.value is true
        if wantsRestart
          S.markDone n, restart_here:true
        else
          S.markDone n
        M.saveThis "done:#{n}", true
        res(true)
      else
        S.markFailed n, errMsg ? "failed"
        S.writePipelineShutdown
          status: 'shutdown'
          by: n
          reason: errMsg ? "failed"
        M.saveThis "done:#{n}", false
        rej new Error(String(errMsg ? "failed"))

    script = path.join(EXEC,'scripts',def.run)

    primeStepMakes = ->
      for artifactKey in (def.makes ? [])
        entry = M.theLowdown artifactKey
        continue if entry?.value isnt undefined

        spec = artifactSpecFor artifactKey
        continue unless isPlainObject(spec)
        target = spec.target
        continue unless typeof target is 'string'

        targetEntry = M.theLowdown target
        targetVal = targetEntry?.value
        continue if targetVal is undefined

        M.saveThis artifactKey, targetVal

    primeStepMakes()

    # ---- SACRED PATH ----
    if /\.coffee$/i.test(script) and isNewStyleStep(script)
      try delete require.cache[require.resolve(script)] catch then null
      step = require(script)?.step
      unless step?.action?
        finish(false, "Missing @step.action in #{script}")
        return
      try
        L = createStepLedger(M, n, resolveArtifact, artifactSpecFor)
        pp=Promise.resolve(step.action(L,n,M))
        pp.then -> finish(true)
        pp.catch (e)-> finish(false, e.message)
      catch e 
        finish(false,e)
        throw e        
      return

    # legacy spawn (only for non-newstyle)
    interp = if /\.py$/i.test(script) then resolvePython(CWD) else 'coffee'
    proc = spawn interp, [script],
      env: Object.assign process.env,
        PYTHON: resolvePython(CWD)
        CFG_OVERRIDE: exp
        STEP_NAME: n
        STEP_PARAMS_JSON: JSON.stringify(def)
      stdio: ['ignore','pipe','pipe']

    proc.stdout.on 'data', (buf) -> process.stdout.write prefixLines("┆ #{n} | ", buf.toString())
    proc.stderr.on 'data', (buf) -> process.stderr.write prefixLines("! #{n} | ", buf.toString())
    proc.on 'error', (err) -> finish(false, err.message)
    proc.on 'exit', (c) ->
      if c is 0 then finish(true) else finish(false, "exit #{c}")

# -------------------------------------------------------------------

installGetStepParam = (M) ->
  M.getStepParam = (stepName, key) ->
    stepP = M.theLowdown("params/#{stepName}.yaml")?.value
    return stepP[key] if stepP? and stepP[key]?

    globalP = M.theLowdown("params/_global.yaml")?.value
    return globalP[key] if globalP? and globalP[key]?

    undefined

###
UI integration design for artifact I/O
-------------------------------------------------------------------
`StepLedger.need(artifactKey)`, `StepLedger.peek(artifactKey, defaultValue)`,
and `StepLedger.make(artifactKey, value)` are the correct hook points for
UI-facing pipeline state because they sit exactly on the explicit
`needs` / `makes` contract between a step and its declared resources.

If a UI layer is added, emit structured state updates here rather than
inside individual step scripts. That keeps the UI deterministic and lets
the runner remain the single source of truth for resource flow.

Critical boundary:
- After meta devices are initialized, artifact persistence and artifact
  recovery must flow through Memo only.
- Do not add direct filesystem reads/writes for artifact sources or
  artifact targets inside the runner.
- If an artifact path needs support, add or extend the appropriate meta
  device instead of bypassing Memo.

Recommended UI memo keys:

  ui/steps/<stepName>/needs/<artifactKey>
    direction: "need"
    step: <stepName>
    artifact: <artifactKey>
    declared_in: "needs" | "makes"
    status: "waiting" | "resolved" | "defaulted" | "missing"
    used_default: true | false
    observed_at: <ISO timestamp>
    resolved_at: <ISO timestamp or null>
    value_summary:
      kind: "array" | "object" | "string" | "number" | "boolean" | "null" | "undefined"
      count: <array/object size if cheap>
      bytes: <string length if cheap>

  ui/steps/<stepName>/makes/<artifactKey>
    direction: "make"
    step: <stepName>
    artifact: <artifactKey>
    status: "written"
    observed_at: <ISO timestamp>
    value_summary: <same shape as above>

  ui/steps/<stepName>/peeks/<artifactKey>
    direction: "peek"
    step: <stepName>
    artifact: <artifactKey>
    status: "resolved" | "defaulted" | "missing"
    observed_at: <ISO timestamp>

Recommended append-only event stream:

  ui/events/<monotonic id>
    type: "artifact_need" | "artifact_make"
    step: <stepName>
    artifact: <artifactKey>
    phase: "start" | "resolved" | "defaulted" | "written" | "error"
    at: <ISO timestamp>

Implementation notes:
- Emit "start" before awaiting a missing `need`.
- Emit "resolved" once the artifact value is available.
- Emit a required-param error when `param(key)` has no configured value and no explicit default.
- Emit "written" immediately before or after `saveThis` in `make`.
- Keep summaries cheap and deterministic; do not deep-inspect large values.
- Never require step scripts to know about UI keys.
- Keep UI state observational only; it must not affect scheduling.
###

# -------------------------------------------------------------------
# MODIFY main()
# -------------------------------------------------------------------
main = ->
  ensureSingleInstance()

  M = new Memo()
  metaLoader = require path.join(EXEC, 'meta')
  metaLoader(M, { baseDir: CWD })
  S = new StepStateStore path.join(CWD,'state')

  M.saveThis "env/EXEC", EXEC
  M.saveThis "env/CWD",  CWD
  M.saveThis "env/PYTHON", resolvePython(CWD)

  overridePath = path.join(CWD,'override.yaml')
  override = loadYamlSafe overridePath
  unless override.pipeline?
    console.error "override.yaml missing pipeline"
    process.exit(1)

  configPath = path.join(EXEC,'config',"#{override.pipeline}.yaml")
  experiment = createExperimentObject configPath, overridePath
  if experiment.run.model && experiment.run.loraLand
    modelDirName = experiment.run.model.replace /\//g, '--'
    targetDir    = path.resolve experiment.run.loraLand, modelDirName
    M.saveThis 'modelDir', targetDir
  M.saveThis 'experiment.yaml',experiment

  steps  = discoverSteps experiment
  artifacts = experiment.artifacts ? {}
  throw new Error "experiment.artifacts must be an object" unless isPlainObject(artifacts)
  order  = toposort steps
  graph  = downstreamMap steps
  finals = terminalSteps steps


  # ---------------- GLOBAL PARAMS (AUTHORITATIVE) ----------------
  globalParams = experiment.run ? {}
  fs.mkdirSync(path.join(CWD,'params'), {recursive:true})
  M.saveThis "params/_global.yaml", globalParams

  installGetStepParam M

  pipeState = S.readPipeline()
  if pipeState?.status is 'shutdown'
    banner "🛑 PIPELINE PREVIOUSLY SHUT DOWN"
    console.log "  by:", pipeState.by
    console.log "  reason:", pipeState.reason
    process.exit(0)

  active = {count: 0, names: new Set()}

  # ---------------- STEP PARAMS ----------------
  producedBy = {}
  for n in order
    unless Object.prototype.hasOwnProperty.call(steps[n], 'needs')
      throw new Error "Step '#{n}' must declare needs: []"
    unless Object.prototype.hasOwnProperty.call(steps[n], 'makes')
      throw new Error "Step '#{n}' must declare makes: []"
    steps[n].needs = normalizeArtifactKeys(steps[n].needs).sort()
    steps[n].makes = normalizeArtifactKeys(steps[n].makes).sort()
    for k in steps[n].makes
      throw new Error "Artifact '#{k}' is produced by multiple steps: #{producedBy[k]} and #{n}" if producedBy[k]?
      producedBy[k] = n
    M.saveThis "params/#{n}.yaml", steps[n]

  # ---------------- ARTIFACT WIRING ----------------
  resolveArtifact = (artifactKey) ->
    spec = artifacts[artifactKey]
    throw new Error "Artifact '#{artifactKey}' not declared in experiment.artifacts" unless spec?
    if isPlainObject(spec) and spec.hasOwnProperty('value')
      return spec.value
    source = if isPlainObject(spec) then spec.source ? spec.key else spec
    target = if isPlainObject(spec) then spec.target else null
    unless source?
      if producedBy[artifactKey]?
        outEntry = M.theLowdown(artifactKey)
        outVal = outEntry.value
        return outVal if outVal isnt undefined
        if typeof target is 'string'
          targetEntry = M.theLowdown(target)
          targetVal = targetEntry.value
          return targetVal if targetVal isnt undefined
        outVal = await outEntry.notifier if outVal is undefined
        return outVal
      throw new Error "Artifact '#{artifactKey}' missing source/value declaration"
    srcEntry = M.theLowdown(source)
    val = srcEntry.value
    val = await srcEntry.notifier if val is undefined
    val

  materializeArtifact = (artifactKey, value) ->
    spec = artifacts[artifactKey]
    return unless spec?
    target = if isPlainObject(spec) then spec.target else null
    if target?
      M.saveThis(target, value)
    M.saveThis(artifactKey, value)

  wireInputsForStep = (stepName) ->
    for k in (steps[stepName].needs ? [])
      v = await resolveArtifact(k)
      M.saveThis k, v

  collectOutputsForStep = (stepName) ->
    for k in (steps[stepName].makes ? [])
      e = M.theLowdown(k)
      throw new Error "Step #{stepName} missing required output #{k}" if e.value is undefined
      await materializeArtifact(k, e.value)

  # ---- remainder of main() UNCHANGED ----
  # (startup restore, scheduling, tick loop, etc.)
  chosen = null
  for n in order
    st = S.read(n)
    if st?.restart_here is true
      chosen = n
      break

  skipRestore = new Set()
  if chosen?
    banner "🔁 restart_here detected at startup: #{chosen}"
    affected = collectDownstream(graph, chosen)   # includes chosen and all downstream
    for a in affected
      skipRestore.add(a)
      if S.delete(a)
        console.log "🧹 deleted obsolete state:", a
    S.clearRestartHere(chosen)  # harmless if file now gone; will just no-op

  # ---------------- STARTUP: restore done/failed from state (only if NOT in skipRestore) ----------------
  for n in order when not skipRestore.has(n)
    st = S.read(n)
    if st?.status is 'done' and st?.done is true and st?.dirty isnt true
      M.saveThis "done:#{n}", true
    else if st?.status is 'failed'
      M.saveThis "done:#{n}", false
    else
      M.theLowdown "done:#{n}"  # leave undefined

  # For affected steps: ensure done key exists but remains undefined (so step WILL run)
  for n in order when skipRestore.has(n)
    M.theLowdown "done:#{n}"    # do NOT set true/false at startup

  scheduled = new Set()
  # ---------------- EXECUTION (DAG scheduling) ----------------
  for n in order
    do (n) ->
      deps = steps[n].depends_on or []
      start = ->
        return if M.theLowdown("done:#{n}").value is true
        scheduled.add n
        Promise.resolve(wireInputsForStep(n))
          .then -> runStep(n, steps[n], experiment, M, S, active, resolveArtifact, (artifactKey) -> artifacts[artifactKey])
          .then -> collectOutputsForStep(n)
          .catch (e) ->
            console.error "! Step #{n} error:", e.message
      if deps.length is 0
        start()
      else
        M.waitFor (deps.map((d)->"done:#{d}")), start

  # ---------------- Completion tick (no hanging on unresolved Promises) ----------------
  tick = ->
    sd = M.theLowdown("pipeline:shutdown").value
    if sd?
      S.writePipelineShutdown sd
      banner "🛑 PIPELINE SHUTDOWN"
      console.log "  by:", sd.by
      console.log "  reason:", sd.reason
      process.exit(0)

    doneFinals = true
    anyFail = false
    for f in finals
      unless scheduled.has f
        doneFinals = false
        continue
      v = M.theLowdown("done:#{f}").value
      if v isnt true and v isnt false then doneFinals = false
      if v is false then anyFail = true

    if doneFinals and active.count is 0
      if anyFail
        banner "💥 Pipeline finished with failures (final: #{finals.join(', ')})"
        process.exit(1)
      else
        banner "🌟 Pipeline finished (final: #{finals.join(', ')})"
        process.exit(0)

    setTimeout(tick, 2000)

  tick()

  printActiveSteps = (signalName) ->
    names = Array.from(active.names ? [])
    banner "📶 Signal received: #{signalName}"
    if names.length
      console.log "  active (#{names.length}):", names.join(', ')
    else
      console.log "  active (0): none"

  process.on 'SIGUSR1', ->
    printActiveSteps('SIGUSR1')

  process.on 'SIGTERM', ->
    printActiveSteps('SIGTERM')
    process.exit(143)

  process.on 'SIGINT', ->
    printActiveSteps('SIGINT')
    console.log "\n(CTRL+C) Exiting..."
    process.exit(130)

main()
