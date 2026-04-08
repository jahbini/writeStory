#!/usr/bin/env coffee
fs = require 'fs'
path = require 'path'
http = require 'http'
yaml = require 'js-yaml'
{ spawn } = require 'child_process'

CWD = process.cwd()
PORT = Number(process.env.UI_PORT ? 4311)
RUNNER = path.join(CWD, 'pipeline_runner.coffee')
EXEC_ROOT = path.dirname(RUNNER)
repeatLoop =
  enabled: false
  payload: null
  timer: null
  next_launch_at: null

readJson = (p, fallback = null) ->
  return fallback unless fs.existsSync(p)
  try JSON.parse(fs.readFileSync(p, 'utf8')) catch then fallback

readText = (p, fallback = '') ->
  return fallback unless fs.existsSync(p)
  try fs.readFileSync(p, 'utf8') catch then fallback

writeText = (p, text) ->
  fs.mkdirSync path.dirname(p), { recursive: true }
  fs.writeFileSync p, text, 'utf8'

writeUiRunPatch = (patch) ->
  runPath = path.join(CWD, 'state', 'ui-run.json')
  current = readJson(runPath, {})
  current = {} unless current? and typeof current is 'object' and not Array.isArray(current)
  next = Object.assign {}, current, patch
  writeText runPath, JSON.stringify(next, null, 2)
  next

pad2 = (n) ->
  text = String(Number(n) ? 0)
  if text.length < 2 then "0#{text}" else text

buildRunTag = ->
  now = new Date()
  hhmm = "#{pad2(now.getHours())}_#{pad2(now.getMinutes())}"
  {
    hh_mm: hhmm
    logdir: "pipe_#{hhmm}"
  }

tailText = (p, maxLines = 120) ->
  text = readText(p, '')
  lines = text.split /\r?\n/
  lines.slice(Math.max(lines.length - maxLines, 0)).join "\n"

listFiles = (dir) ->
  return [] unless fs.existsSync(dir)
  names = fs.readdirSync(dir).sort()
  out = []
  for name in names
    full = path.join(dir, name)
    stat = fs.statSync(full)
    out.push
      name: name
      path: full
      is_dir: stat.isDirectory()
      size: stat.size
      mtime: stat.mtime.toISOString()
  out

readJsonlTail = (p, maxRows = 80) ->
  return [] unless fs.existsSync(p)
  text = fs.readFileSync(p, 'utf8')
  rows = []
  for line in text.split(/\r?\n/) when line.trim().length
    try rows.push JSON.parse(line) catch then null
  rows.slice Math.max(rows.length - maxRows, 0)

latestLogStem = ->
  logDir = path.join(CWD, 'logs')
  return null unless fs.existsSync(logDir)
  names = fs.readdirSync(logDir).filter (name) -> /^pipe_\d{2}_\d{2}\.(log|err)$/.test(name)
  return null unless names.length
  stems = {}
  for name in names
    stem = name.replace /\.(log|err)$/, ''
    stems[stem] = true
  ordered = Object.keys(stems).sort()
  ordered[ordered.length - 1]

collectStepStates = ->
  stateDir = path.join(CWD, 'state')
  return [] unless fs.existsSync(stateDir)
  names = fs.readdirSync(stateDir).filter (name) -> /^step-.*\.json$/.test(name)
  rows = []
  for name in names
    row = readJson path.join(stateDir, name), {}
    continue unless row?
    rows.push row
  rows.sort (a, b) ->
    String(a.step ? '').localeCompare String(b.step ? '')

readOverride = ->
  overridePath = path.join(CWD, 'override.yaml')
  return {} unless fs.existsSync overridePath
  try yaml.load(fs.readFileSync(overridePath, 'utf8')) ? {} catch then {}

readYaml = (p) ->
  return {} unless fs.existsSync p
  try yaml.load(fs.readFileSync(p, 'utf8')) ? {} catch then {}

buildControls = ->
  override = readOverride()
  libraryDoc = readYaml path.join(CWD, 'data', 'jim_story_library.yaml')
  library = libraryDoc?.library ? {}
  storyStep = override?.select_story_recipe ? {}

  makeOptions = (shelfName) ->
    shelf = library?[shelfName] ? {}
    rows = []
    for own key, value of shelf
      label = value?.text ? value?.character ? key
      rows.push { key, label }
    rows.sort (a, b) -> String(a.label).localeCompare String(b.label)
    rows

  {
    pipeline: override.pipeline ? ''
    story_id: storyStep.story_id ? ''
    scene: storyStep.scene ? ''
    arrival: storyStep.arrival ? ''
    disturbance: storyStep.disturbance ? ''
    reflection: storyStep.reflection ? ''
    realization: storyStep.realization ? ''
    pipelines: [
      'base_ite'
      'oracle_ite'
      'lora_ite'
      'diary_ite'
      'story_scan'
      'lora_scan'
    ]
    diary_story_ids: [
      'jim_0001'
      'jim_0002'
      'jim_0003'
    ]
    scene_options: makeOptions 'scenes'
    arrival_options: makeOptions 'characters'
    disturbance_options: makeOptions 'disturbances'
    reflection_options: makeOptions 'reflections'
    realization_options: makeOptions 'realizations'
  }

describeOutputFile = (relativePath, runStart = null) ->
  fullPath = path.join(CWD, relativePath)
  exists = fs.existsSync(fullPath)
  stat = if exists then fs.statSync(fullPath) else null
  mtime = if stat? then stat.mtime.toISOString() else null
  fresh = false
  if stat? and runStart?
    started = new Date(runStart)
    fresh = not Number.isNaN(started.getTime()) and stat.mtime.getTime() >= started.getTime()

  {
    name: path.basename(relativePath)
    path: relativePath
    exists: exists
    size: stat?.size ? null
    mtime: mtime
    is_fresh: fresh
  }

collectExpectedOutputs = (run) ->
  override = readOverride()
  pipeline = override.pipeline ? run?.pipeline ? null
  return { out_files: [], diary_files: [] } unless pipeline?

  configPath = path.join(CWD, 'config', "#{pipeline}.yaml")
  recipe = readYaml(configPath)
  artifacts = recipe?.artifacts ? {}
  runStart = run?.started_at ? null

  outFiles = []
  diaryFiles = []
  seen = new Set()

  for own artifactKey, spec of artifacts
    continue unless spec? and typeof spec is 'object' and typeof spec.target is 'string'
    target = String(spec.target)
    continue if seen.has(target)
    seen.add target
    row = describeOutputFile target, runStart
    if /^diary\//.test(target)
      diaryFiles.push row
    else
      outFiles.push row

  if pipeline is 'diary_ite' and run?.hh_mm?
    diaryBase = "diary/diary_#{run.hh_mm}.txt"
    diaryAdapter = "diary/diary_#{run.hh_mm}.adapter.txt"
    for target in [diaryBase, diaryAdapter] when not seen.has(target)
      seen.add target
      diaryFiles.push describeOutputFile target, runStart

  outFiles.sort (a, b) -> String(a.path).localeCompare String(b.path)
  diaryFiles.sort (a, b) -> String(a.path).localeCompare String(b.path)

  {
    out_files: outFiles
    diary_files: diaryFiles
  }

buildStatus = ->
  run = readJson path.join(CWD, 'state', 'ui-run.json'), {}
  expectedOutputs = collectExpectedOutputs(run)
  events = readJsonlTail path.join(CWD, 'state', 'ui-events.jsonl')
  steps = collectStepStates()
  stem = if run?.logdir? then String(run.logdir) else latestLogStem()
  latestLog = if stem? then readText(path.join(CWD, 'logs', "#{stem}.log")) else ''
  latestErr = if stem? then readText(path.join(CWD, 'logs', "#{stem}.err")) else ''

  {
    run: run
    controls: buildControls()
    steps: steps
    events: events
    latest_log_stem: stem
    latest_log: latestLog
    latest_err: latestErr
    out_files: expectedOutputs.out_files
    diary_files: expectedOutputs.diary_files
  }

isAllowedFilePath = (relativePath) ->
  return false unless typeof relativePath is 'string' and relativePath.length
  normalized = path.normalize(relativePath)
  return false if normalized.startsWith('..') or path.isAbsolute(normalized)
  /^logs\//.test(normalized) or /^out\//.test(normalized) or /^diary\//.test(normalized) or /^build\//.test(normalized)

readViewerFile = (relativePath) ->
  return null unless isAllowedFilePath(relativePath)
  fullPath = path.join(CWD, relativePath)
  return null unless fs.existsSync(fullPath)
  stat = fs.statSync(fullPath)
  return null unless stat.isFile()
  {
    path: relativePath
    size: stat.size
    mtime: stat.mtime.toISOString()
    text: readText(fullPath, '')
  }

sendJson = (res, code, payload) ->
  body = JSON.stringify(payload, null, 2)
  res.writeHead code,
    'Content-Type': 'application/json; charset=utf-8'
    'Content-Length': Buffer.byteLength(body)
    'Cache-Control': 'no-store'
  res.end body

sendHtml = (res, p) ->
  body = readText p, ''
  if not body.length
    res.writeHead 404, 'Content-Type': 'text/plain; charset=utf-8'
    res.end 'ui/index.html not found'
    return
  res.writeHead 200,
    'Content-Type': 'text/html; charset=utf-8'
    'Content-Length': Buffer.byteLength(body)
    'Cache-Control': 'no-store'
  res.end body

readRequestBody = (req) ->
  new Promise (resolve, reject) ->
    chunks = []
    req.on 'data', (chunk) -> chunks.push chunk
    req.on 'end', ->
      text = Buffer.concat(chunks).toString('utf8')
      resolve text
    req.on 'error', reject

clearStepState = ->
  stateDir = path.join(CWD, 'state')
  return unless fs.existsSync stateDir
  for name in fs.readdirSync(stateDir) when /^step-.*\.json$/.test(name) or /^ui-.*\.(json|jsonl)$/.test(name)
    fs.unlinkSync path.join(stateDir, name)

  pipelinePath = path.join(CWD, 'pipeline.json')
  fs.unlinkSync(pipelinePath) if fs.existsSync(pipelinePath)

seedUiRun = (launch, override) ->
  runPath = path.join(CWD, 'state', 'ui-run.json')
  current = readJson(runPath, {})
  current = {} unless current? and typeof current is 'object' and not Array.isArray(current)

  seeded =
    pipeline: current.pipeline ? override.pipeline ? null
    pid: current.pid ? launch.pid
    cwd: current.cwd ? CWD
    hh_mm: current.hh_mm ? launch.hh_mm
    logdir: current.logdir ? launch.logdir
    status: current.status ? 'launching'
    started_at: current.started_at ? new Date().toISOString()
    finished_at: current.finished_at ? null

  writeText runPath, JSON.stringify(seeded, null, 2)

markUiRunExited = (launch, patch = {}) ->
  runPath = path.join(CWD, 'state', 'ui-run.json')
  current = readJson(runPath, {})
  return unless current? and typeof current is 'object' and not Array.isArray(current)
  return unless current.pid is launch.pid
  return unless current.status in ['launching', 'running']

  next = Object.assign {}, current,
    status: patch.status ? 'exited'
    finished_at: patch.finished_at ? new Date().toISOString()
  , patch

  writeText runPath, JSON.stringify(next, null, 2)

stopRepeatLoop = ->
  if repeatLoop.timer?
    clearTimeout repeatLoop.timer
  repeatLoop.enabled = false
  repeatLoop.payload = null
  repeatLoop.timer = null
  repeatLoop.next_launch_at = null

scheduleRepeatLaunch = ->
  return unless repeatLoop.enabled and repeatLoop.payload?

  pipelineState = readJson path.join(CWD, 'pipeline.json'), null
  if pipelineState?.status is 'shutdown'
    stopRepeatLoop()
    writeUiRunPatch
      loop_enabled: false
      countdown_seconds: null
      next_launch_at: null
    return

  delayMs = 60 * 1000
  repeatLoop.next_launch_at = new Date(Date.now() + delayMs).toISOString()
  writeUiRunPatch
    status: 'cooldown'
    loop_enabled: true
    countdown_seconds: 60
    next_launch_at: repeatLoop.next_launch_at

  repeatLoop.timer = setTimeout ->
    return unless repeatLoop.enabled and repeatLoop.payload?
    pipelineStateNow = readJson path.join(CWD, 'pipeline.json'), null
    if pipelineStateNow?.status is 'shutdown'
      stopRepeatLoop()
      writeUiRunPatch
        loop_enabled: false
        countdown_seconds: null
        next_launch_at: null
      return

    override = writeOverride repeatLoop.payload
    clearStepState()
    launch = startRunner()
    seedUiRun launch, override
    writeUiRunPatch
      loop_enabled: true
      countdown_seconds: null
      next_launch_at: null
  , delayMs

writeOverride = (payload) ->
  override = readOverride()
  override.pipeline = String(payload.pipeline ? override.pipeline ? '')

  if override.pipeline is 'diary_ite'
    override.select_story_recipe ?= {}

  if override.pipeline is 'diary_ite' and payload.story_id?
    override.select_story_recipe.story_id = String(payload.story_id)

  if override.pipeline is 'diary_ite'
    for key in ['scene', 'arrival', 'disturbance', 'reflection', 'realization']
      if payload[key]?
        value = String(payload[key]).trim()
        if value.length
          override.select_story_recipe[key] = value
        else
          delete override.select_story_recipe[key]

  text = yaml.dump override,
    lineWidth: 120
    noRefs: true
  writeText path.join(CWD, 'override.yaml'), text
  override

startRunner = ->
  runTag = buildRunTag()
  logDir = path.join(CWD, 'logs')
  fs.mkdirSync logDir, { recursive: true }
  logPath = path.join(logDir, "#{runTag.logdir}.log")
  errPath = path.join(logDir, "#{runTag.logdir}.err")
  fs.writeFileSync logPath, '', 'utf8'
  fs.writeFileSync errPath, '', 'utf8'
  outFd = fs.openSync logPath, 'a'
  errFd = fs.openSync errPath, 'a'

  child = spawn 'coffee', [RUNNER],
    cwd: CWD
    detached: true
    stdio: ['ignore', outFd, errFd]
    env: Object.assign {}, process.env,
      EXEC: EXEC_ROOT
      PWD: CWD
      HH_MM: runTag.hh_mm
      LOGDIR: runTag.logdir

  child.unref()
  child.on 'error', (err) ->
    markUiRunExited {
      pid: child.pid
      hh_mm: runTag.hh_mm
      logdir: runTag.logdir
    },
      status: 'failed'
      error: String(err?.message ? err)

  child.on 'exit', (code, signal) ->
    status = if code is 0 then 'done' else 'failed'
    markUiRunExited {
      pid: child.pid
      hh_mm: runTag.hh_mm
      logdir: runTag.logdir
    },
      status: status
      exit_code: code
      signal: signal ? null

    if repeatLoop.enabled
      if status is 'done'
        scheduleRepeatLaunch()
      else
        stopRepeatLoop()
        writeUiRunPatch
          loop_enabled: false
          countdown_seconds: null
          next_launch_at: null

  {
    pid: child.pid
    hh_mm: runTag.hh_mm
    logdir: runTag.logdir
  }

handleLaunch = (req, res) ->
  bodyText = await readRequestBody req
  payload = {}
  try
    payload = JSON.parse(bodyText ? '{}')
  catch
    return sendJson res, 400, { ok: false, error: 'invalid json body' }

  pipeline = String(payload.pipeline ? '').trim()
  return sendJson(res, 400, { ok: false, error: 'pipeline is required' }) unless pipeline.length

  if payload.continuous is true
    repeatLoop.enabled = true
    repeatLoop.payload = Object.assign {}, payload
  else
    stopRepeatLoop()

  override = writeOverride payload
  clearStepState()
  launch = startRunner()
  seedUiRun launch, override
  writeUiRunPatch
    loop_enabled: repeatLoop.enabled
    countdown_seconds: null
    next_launch_at: null

  sendJson res, 200,
    ok: true
    pid: launch.pid
    hh_mm: launch.hh_mm
    logdir: launch.logdir
    override: override

handleKill = (req, res) ->
  stopRepeatLoop()
  runPath = path.join(CWD, 'state', 'ui-run.json')
  run = readJson(runPath, {})
  pid = Number(run?.pid ? 0)
  targetKind = 'run'

  if run?.status is 'skipped' and Array.isArray(run?.other_runners) and run.other_runners.length > 0
    first = String(run.other_runners[0] ? '')
    match = first.match(/^\s*(\d+)\b/)
    if match?
      pid = Number(match[1])
      targetKind = 'blocking_runner'

  return sendJson(res, 400, { ok: false, error: 'no active run pid recorded' }) unless pid > 0

  try
    process.kill pid, 'SIGTERM'
  catch err
    return sendJson res, 500,
      ok: false
      error: String(err?.message ? err)

  next = Object.assign {}, run,
    status: 'killing'
    kill_requested_at: new Date().toISOString()
    loop_enabled: false
    countdown_seconds: null
    next_launch_at: null
  writeText runPath, JSON.stringify(next, null, 2)

  sendJson res, 200,
    ok: true
    pid: pid
    target_kind: targetKind

server = http.createServer (req, res) ->
  url = req.url ? '/'
  if url is '/' or url is '/index.html'
    return sendHtml res, path.join(CWD, 'ui', 'index.html')
  if url is '/api/status'
    return sendJson res, 200, buildStatus()
  if url.startsWith('/api/file?')
    query = new URL(url, 'http://127.0.0.1').searchParams
    relativePath = query.get('path')
    payload = readViewerFile(relativePath)
    return sendJson(res, 404, { ok: false, error: 'file not found' }) unless payload?
    return sendJson res, 200, { ok: true, file: payload }
  if url is '/api/launch' and req.method is 'POST'
    return Promise.resolve(handleLaunch(req, res)).catch (err) ->
      sendJson res, 500,
        ok: false
        error: String(err?.message ? err)
  if url is '/api/kill' and req.method is 'POST'
    return Promise.resolve(handleKill(req, res)).catch (err) ->
      sendJson res, 500,
        ok: false
        error: String(err?.message ? err)
  res.writeHead 404, 'Content-Type': 'text/plain; charset=utf-8'
  res.end 'not found'

server.listen PORT, '127.0.0.1', ->
  console.log "[ui_server] listening on http://127.0.0.1:#{PORT}"

setInterval ->
  return unless repeatLoop.enabled and repeatLoop.next_launch_at?
  run = readJson path.join(CWD, 'state', 'ui-run.json'), {}
  return unless run?.status is 'cooldown'
  remainingMs = Math.max(0, new Date(repeatLoop.next_launch_at).getTime() - Date.now())
  seconds = Math.ceil(remainingMs / 1000)
  writeUiRunPatch
    loop_enabled: true
    countdown_seconds: seconds
    next_launch_at: repeatLoop.next_launch_at
, 1000
