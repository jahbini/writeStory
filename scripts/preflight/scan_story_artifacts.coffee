fs = require 'fs'
path = require 'path'
yaml = require 'js-yaml'

MAX_SNIPPET = 180

makePattern = (reason, regex) ->
  { reason, regex }

PATTERNS = [
  makePattern 'html tag', /<\/?(html|body|p|br|sup|em|strong|div|span|blockquote|script|style|table|tr|td|ul|ol|li|a|img|h[1-6])\b/i
  makePattern 'xml-style tag', /<\/?[A-Za-z_][A-Za-z0-9:_-]*(\s+[^>]*)?>/
  makePattern 'broken tag fragment', /<[A-Za-z][^>\n]{0,80}$|^.{0,80}>/
  makePattern 'markdown heading residue', /^\s{0,3}#{1,6}\s+\S/
  makePattern 'markdown link residue', /\[[^\]]+\]\([^)]+\)|\[[^\]]+\]\[\d+\]/
  makePattern 'markdown footnote residue', /\[\d+\]/
  makePattern 'markdown emphasis residue', /(^|[^\w])(\*\*[^*]+\*\*|__[^_]+__|\*[^*\s][^*]*\*|_[^_\s][^_]*_)([^\w]|$)/
  makePattern 'markdown blockquote residue', /^\s*>\s+\S/
  makePattern 'template residue', /\{\{\{|\}\}\}|\{\{|\}\}|<<<|>>>/
  makePattern 'escaped markup residue', /&lt;|&gt;|&amp;lt;|&amp;gt;|&#x?[0-9a-f]+;/i
  makePattern 'repeated symbolic junk', /[<>{}=+]{5,}/
  makePattern 'prompt or machine text leakage', /Input:|Output:|Prompt:|Generation:|Peak memory:|please confirm|<stop>|<end>|<\/stop>/i
  makePattern 'control-like junk', /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/,
  makePattern 'suspicious unicode junk', /[\u200B-\u200F\u2060\uFEFF\uFFFD]/
]

truncateSnippet = (text) ->
  value = String(text ? '').replace(/\s+/g, ' ').trim()
  return value unless value.length > MAX_SNIPPET
  value.slice(0, MAX_SNIPPET - 3) + '...'

addFinding = (findings, seen, filePath, lineNumber, location, snippet, reason) ->
  key = [filePath, lineNumber ? '', location ? '', reason, snippet].join ' | '
  return if seen.has key
  seen.add key
  findings.push
    file: filePath
    line: lineNumber ? null
    location: location ? null
    snippet: truncateSnippet snippet
    reason: reason

scanLine = (findings, seen, filePath, lineNumber, location, lineText) ->
  text = String(lineText ? '')
  return unless text.trim().length

  for pattern in PATTERNS
    pattern.regex.lastIndex = 0 if pattern.regex.global
    if pattern.regex.test text
      addFinding findings, seen, filePath, lineNumber, location, text, pattern.reason

scanTextBlock = (findings, seen, filePath, location, text, lineNumber = null) ->
  value = String(text ? '')
  lines = value.split /\r?\n/

  if lineNumber? and lines.length is 1
    scanLine findings, seen, filePath, lineNumber, location, value
    return

  idx = 0
  while idx < lines.length
    lineText = lines[idx]
    actualLine = if lineNumber? then lineNumber + idx else idx + 1
    scanLine findings, seen, filePath, actualLine, location, lineText
    idx += 1

scanStructured = (findings, seen, filePath, value, location, lineNumber = null) ->
  return unless value?

  if typeof value is 'string'
    scanTextBlock findings, seen, filePath, location, value, lineNumber
    return

  if Array.isArray(value)
    idx = 0
    while idx < value.length
      nextLocation = if location? then "#{location}[#{idx}]" else "[#{idx}]"
      scanStructured findings, seen, filePath, value[idx], nextLocation, lineNumber
      idx += 1
    return

  return unless typeof value is 'object'

  for own key, child of value
    nextLocation = if location? then "#{location}.#{key}" else key
    scanStructured findings, seen, filePath, child, nextLocation, lineNumber

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
  scanTextBlock findings, seen, filePath, '$raw', rawText

  ext = path.extname(filePath).toLowerCase()
  if ext is '.jsonl'
    scanJsonlFile findings, seen, filePath, rawText
    return

  try
    structured = parseStructuredFile filePath, rawText
    scanStructured findings, seen, filePath, structured, '$parsed'
  catch
    addFinding findings, seen, filePath, null, '$raw', rawText.slice(0, 120), 'invalid structured file'

runSelfTest = ->
  samples = [
    { text: '<p>bad html</p>', expect: 'html tag' }
    { text: 'Some text <<< injected >>>', expect: 'template residue' }
    { text: 'Prompt: 200 tokens', expect: 'prompt or machine text leakage' }
    { text: 'Look at this =======', expect: 'repeated symbolic junk' }
    { text: "Zero\u200Bwidth", expect: 'suspicious unicode junk' }
  ]

  for sample in samples
    matched = false
    for pattern in PATTERNS
      pattern.regex.lastIndex = 0 if pattern.regex.global
      if pattern.regex.test sample.text
        if pattern.reason is sample.expect
          matched = true
          break
    throw new Error "self-test failed for #{sample.expect}" unless matched

buildReport = (scannedFiles, findings) ->
  lines = []
  lines.push 'Story Artifact Scan Report'
  lines.push "generated_at: #{new Date().toISOString()}"
  lines.push "files_scanned: #{scannedFiles.length}"
  lines.push "findings: #{findings.length}"
  lines.push ''

  if scannedFiles.length
    lines.push 'SCANNED FILES'
    for filePath in scannedFiles
      lines.push "- #{filePath}"
    lines.push ''

  if findings.length is 0
    lines.push 'No suspicious content found.'
    return lines.join "\n"

  lines.push 'FINDINGS'
  idx = 1
  for finding in findings
    lines.push "#{idx}. file: #{finding.file}"
    lines.push "   line: #{finding.line}" if finding.line?
    lines.push "   location: #{finding.location}" if finding.location?
    lines.push "   reason: #{finding.reason}"
    lines.push "   snippet: #{finding.snippet}"
    lines.push ''
    idx += 1

  lines.join "\n"

persistReport = (L, artifactKey, reportText) ->
  L.make artifactKey, reportText

  experiment = L.theLowdown('experiment.yaml')?.value ? {}
  targetPath = experiment?.artifacts?[artifactKey]?.target
  if typeof targetPath is 'string'
    L.saveThis targetPath, reportText

@step =
  desc: "Scan story-related source and derived artifacts for markup residue and machine junk"

  action: (L) ->
    runSelfTest()

    rawPaths = L.param 'raw_paths', []
    derivedPaths = L.param 'derived_paths', []
    skipMissing = L.param 'skip_missing', true
    failOnFindings = L.param 'fail_on_findings', true

    allPaths = []
    for item in rawPaths when item?
      allPaths.push path.resolve String(item)
    for item in derivedPaths when item?
      allPaths.push path.resolve String(item)

    uniquePaths = []
    seenPaths = new Set()
    for filePath in allPaths
      continue if seenPaths.has filePath
      seenPaths.add filePath
      uniquePaths.push filePath

    findings = []
    seenFindings = new Set()
    scannedFiles = []

    for filePath in uniquePaths
      unless fs.existsSync filePath
        if skipMissing
          continue
        addFinding findings, seenFindings, filePath, null, null, '', 'missing file'
        continue

      stat = fs.statSync filePath
      continue unless stat.isFile()

      scannedFiles.push filePath
      scanFile findings, seenFindings, filePath

    findings.sort (a, b) ->
      fileCmp = String(a.file).localeCompare String(b.file)
      return fileCmp unless fileCmp is 0
      lineA = a.line ? 0
      lineB = b.line ? 0
      return lineA - lineB unless lineA is lineB
      String(a.reason).localeCompare String(b.reason)

    reportText = buildReport scannedFiles, findings

    console.log "[scan_story_artifacts] files scanned:", scannedFiles.length
    console.log "[scan_story_artifacts] findings:", findings.length

    persistReport L, 'scan_report', reportText

    if failOnFindings and findings.length > 0
      throw new Error "[#{L.stepName}] suspicious story artifacts found: #{findings.length}"

    L.done()
    return
