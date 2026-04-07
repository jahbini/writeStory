#!/usr/bin/env coffee
fs = require 'fs'
path = require 'path'
http = require 'http'

CWD = process.cwd()
PORT = Number(process.env.UI_PORT ? 4311)

readJson = (p, fallback = null) ->
  return fallback unless fs.existsSync(p)
  try JSON.parse(fs.readFileSync(p, 'utf8')) catch then fallback

readText = (p, fallback = '') ->
  return fallback unless fs.existsSync(p)
  try fs.readFileSync(p, 'utf8') catch then fallback

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

buildStatus = ->
  run = readJson path.join(CWD, 'state', 'ui-run.json'), {}
  events = readJsonlTail path.join(CWD, 'state', 'ui-events.jsonl')
  steps = collectStepStates()
  stem = latestLogStem()
  latestLog = if stem? then tailText(path.join(CWD, 'logs', "#{stem}.log")) else ''
  latestErr = if stem? then tailText(path.join(CWD, 'logs', "#{stem}.err")) else ''

  {
    run: run
    steps: steps
    events: events
    latest_log_stem: stem
    latest_log: latestLog
    latest_err: latestErr
    out_files: listFiles path.join(CWD, 'out')
    diary_files: listFiles path.join(CWD, 'diary')
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

server = http.createServer (req, res) ->
  url = req.url ? '/'
  if url is '/' or url is '/index.html'
    return sendHtml res, path.join(CWD, 'ui', 'index.html')
  if url is '/api/status'
    return sendJson res, 200, buildStatus()
  res.writeHead 404, 'Content-Type': 'text/plain; charset=utf-8'
  res.end 'not found'

server.listen PORT, '127.0.0.1', ->
  console.log "[ui_server] listening on http://127.0.0.1:#{PORT}"
