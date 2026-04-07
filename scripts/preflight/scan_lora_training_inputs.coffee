fs = require 'fs'
path = require 'path'
yaml = require 'js-yaml'
{ DatabaseSync } = require 'node:sqlite'

MAX_SNIPPET = 220

makeRule = (reason, regex) ->
  { reason, regex }

RULES = [
  makeRule '<stop> marker', /<stop>/i
  makeRule '<end> marker', /<end>/i
  makeRule '<input> marker', /<input>/i
  makeRule 'Input/Output leakage', /\bInput:|\bOutput:/i
  makeRule 'html tag', /<\/?(html|body|p|br|sup|em|strong|div|span|blockquote|script|style|table|tr|td|ul|ol|li|a|img|h[1-6])\b/i
  makeRule 'xml-style tag', /<\/?[A-Za-z_][A-Za-z0-9:_-]*(\s+[^>]*)?>/
  makeRule 'broken tag fragment', /<[A-Za-z][^>\n]{0,80}$|^.{0,80}>/
  makeRule 'template fragment', /\{\{\{|\}\}\}|\{\{|\}\}|<<<|>>>/
  makeRule 'repeated symbol junk', /[<>{}=+]{5,}/
  makeRule 'prompt scaffolding', /\bBegin:\s*$|^You are writing in the narrative voice|^Expand the following story fragment/i
  makeRule 'machine text leakage', /\bPrompt:\s+\d+\s+tokens|\bGeneration:\s+\d+\s+tokens|\bPeak memory:/i
  makeRule 'markdown blockquote residue', /^\s*>\s+\S/
  makeRule 'markdown link residue', /\[[^\]]+\]\([^)]+\)|\[[^\]]+\]\[\d+\]/
  makeRule 'markdown footnote residue', /\[\d+\]/
  makeRule 'markdown emphasis residue', /(^|[^\w])(\*\*[^*]+\*\*|__[^_]+__|\*[^*\s][^*]*\*|_[^_\s][^_]*_)([^\w]|$)/
  makeRule 'escaped markup residue', /&lt;|&gt;|&amp;lt;|&amp;gt;|&#x?[0-9a-f]+;/i
  makeRule 'control-like junk', /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/,
  makeRule 'suspicious unicode junk', /[\u200B-\u200F\u2060\uFEFF\uFFFD]/
]

truncateSnippet = (value) ->
  text = String(value ? '').replace(/\s+/g, ' ').trim()
  return text unless text.length > MAX_SNIPPET
  text.slice(0, MAX_SNIPPET - 3) + '...'

addFinding = (findings, seen, sourcePath, lineNumber, location, snippet, reason) ->
  key = [sourcePath, lineNumber ? '', location ? '', reason, snippet].join ' | '
  return if seen.has key
  seen.add key
  findings.push
    source: sourcePath
    line: lineNumber ? null
    location: location ? null
    snippet: truncateSnippet snippet
    reason: reason

scanLine = (findings, seen, sourcePath, lineNumber, location, text) ->
  value = String(text ? '')
  return unless value.length

  for rule in RULES
    rule.regex.lastIndex = 0 if rule.regex.global
    if rule.regex.test value
      addFinding findings, seen, sourcePath, lineNumber, location, value, rule.reason

scanText = (findings, seen, sourcePath, location, text, baseLine = null) ->
  lines = String(text ? '').split /\r?\n/
  idx = 0
  while idx < lines.length
    lineText = lines[idx]
    lineNumber = if baseLine? then baseLine + idx else idx + 1
    scanLine findings, seen, sourcePath, lineNumber, location, lineText
    idx += 1

scanStructured = (findings, seen, sourcePath, value, location, baseLine = null) ->
  return unless value?

  if typeof value is 'string'
    scanText findings, seen, sourcePath, location, value, baseLine
    return

  if Array.isArray(value)
    idx = 0
    while idx < value.length
      nextLocation = if location? then "#{location}[#{idx}]" else "[#{idx}]"
      scanStructured findings, seen, sourcePath, value[idx], nextLocation, baseLine
      idx += 1
    return

  return unless typeof value is 'object'

  for own key, child of value
    nextLocation = if location? then "#{location}.#{key}" else key
    scanStructured findings, seen, sourcePath, child, nextLocation, baseLine

parseStructuredFile = (filePath, rawText) ->
  ext = path.extname(filePath).toLowerCase()
  if ext is '.json'
    return JSON.parse rawText
  if ext in ['.yaml', '.yml']
    return yaml.load rawText
  null

scanJsonlFile = (findings, seen, filePath, rawText) ->
  lines = rawText.split /\r?\n/
  idx = 0
  while idx < lines.length
    lineText = lines[idx]
    lineNumber = idx + 1
    if lineText.trim().length
      scanLine findings, seen, filePath, lineNumber, '$raw', lineText
      try
        row = JSON.parse lineText
        scanStructured findings, seen, filePath, row, '$jsonl', lineNumber
      catch
        addFinding findings, seen, filePath, lineNumber, '$raw', lineText, 'invalid jsonl line'
    idx += 1

scanFile = (findings, seen, filePath) ->
  rawText = fs.readFileSync filePath, 'utf8'
  scanText findings, seen, filePath, '$raw', rawText

  ext = path.extname(filePath).toLowerCase()
  if ext is '.jsonl'
    scanJsonlFile findings, seen, filePath, rawText
    return

  try
    parsed = parseStructuredFile filePath, rawText
    scanStructured findings, seen, filePath, parsed, '$parsed'
  catch
    addFinding findings, seen, filePath, null, '$raw', rawText.slice(0, 120), 'invalid structured file'

scanStoriesTable = (findings, seen, sqlitePath) ->
  return unless fs.existsSync sqlitePath

  db = new DatabaseSync sqlitePath
  try
    rows = db.prepare("""
      SELECT story_id, title, text
      FROM stories
      ORDER BY story_id ASC
    """).all()

    for row in rows
      sourcePath = "#{sqlitePath}#stories{#{row.story_id}}"
      scanStructured findings, seen, sourcePath, row, '$sqlite'
  finally
    db.close()

runSelfTest = ->
  samples = [
    { text: 'hello <stop>', expect: '<stop> marker' }
    { text: '<html>oops</html>', expect: 'html tag' }
    { text: 'Input: this leaked', expect: 'Input/Output leakage' }
    { text: '{{{comment:oops}}}', expect: 'template fragment' }
    { text: '<<<<<<', expect: 'repeated symbol junk' }
  ]

  for sample in samples
    matched = false
    for rule in RULES
      rule.regex.lastIndex = 0 if rule.regex.global
      if rule.regex.test sample.text
        if rule.reason is sample.expect
          matched = true
          break
    throw new Error "self-test failed for #{sample.expect}" unless matched

buildReport = (filesScanned, sqliteScanned, findings) ->
  lines = []
  lines.push 'LoRA Training Contamination Scan Report'
  lines.push "generated_at: #{new Date().toISOString()}"
  lines.push "files_scanned: #{filesScanned.length}"
  lines.push "sqlite_scanned: #{sqliteScanned}"
  lines.push "findings: #{findings.length}"
  lines.push ''

  if filesScanned.length
    lines.push 'SCANNED FILES'
    for filePath in filesScanned
      lines.push "- #{filePath}"
    lines.push ''

  if sqliteScanned
    lines.push "SCANNED SQLITE: #{sqliteScanned}"
    lines.push ''

  if findings.length is 0
    lines.push 'No suspicious content found.'
    return lines.join "\n"

  lines.push 'FINDINGS'
  idx = 1
  for finding in findings
    lines.push "#{idx}. source: #{finding.source}"
    lines.push "   line: #{finding.line}" if finding.line?
    lines.push "   location: #{finding.location}" if finding.location?
    lines.push "   reason: #{finding.reason}"
    lines.push "   snippet: #{finding.snippet}"
    lines.push ''
    idx += 1

  lines.join "\n"

persistReport = (L, artifactKey, text) ->
  L.make artifactKey, text
  experiment = L.theLowdown('experiment.yaml')?.value ? {}
  targetPath = experiment?.artifacts?[artifactKey]?.target
  if typeof targetPath is 'string'
    L.saveThis targetPath, text

@step =
  desc: "Scan LoRA source, intermediate, SQLite, and final training inputs for contamination"

  action: (L) ->
    runSelfTest()

    rawPaths = L.param 'raw_paths', []
    intermediatePaths = L.param 'intermediate_paths', []
    finalPaths = L.param 'final_paths', []
    sqlitePathParam = L.param 'sqlite_db', null
    skipMissing = L.param 'skip_missing', true
    failOnFindings = L.param 'fail_on_findings', true

    findings = []
    seenFindings = new Set()
    filesScanned = []
    sqliteScanned = null

    allPaths = []
    for filePath in rawPaths when filePath?
      allPaths.push path.resolve String(filePath)
    for filePath in intermediatePaths when filePath?
      allPaths.push path.resolve String(filePath)
    for filePath in finalPaths when filePath?
      allPaths.push path.resolve String(filePath)

    seenPaths = new Set()
    uniquePaths = []
    for filePath in allPaths
      continue if seenPaths.has filePath
      seenPaths.add filePath
      uniquePaths.push filePath

    for filePath in uniquePaths
      unless fs.existsSync filePath
        if skipMissing
          continue
        addFinding findings, seenFindings, filePath, null, null, '', 'missing file'
        continue
      stat = fs.statSync filePath
      continue unless stat.isFile()
      filesScanned.push filePath
      scanFile findings, seenFindings, filePath

    if sqlitePathParam?
      sqlitePath = path.resolve String(sqlitePathParam)
      if fs.existsSync sqlitePath
        sqliteScanned = sqlitePath
        scanStoriesTable findings, seenFindings, sqlitePath
      else if not skipMissing
        addFinding findings, seenFindings, sqlitePath, null, null, '', 'missing sqlite db'

    findings.sort (a, b) ->
      cmp = String(a.source).localeCompare String(b.source)
      return cmp unless cmp is 0
      lineA = a.line ? 0
      lineB = b.line ? 0
      return lineA - lineB unless lineA is lineB
      String(a.reason).localeCompare String(b.reason)

    reportText = buildReport filesScanned, sqliteScanned, findings

    console.log "[scan_lora_training_inputs] files scanned:", filesScanned.length
    console.log "[scan_lora_training_inputs] sqlite scanned:", sqliteScanned ? 'none'
    console.log "[scan_lora_training_inputs] findings:", findings.length

    persistReport L, 'scan_report', reportText

    if failOnFindings and findings.length > 0
      throw new Error "[#{L.stepName}] suspicious LoRA training contamination found: #{findings.length}"

    L.done()
    return
