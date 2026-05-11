#!/usr/bin/env coffee

fs   = require 'fs'
path = require 'path'

WARN_TOL = 1e-5
FAIL_TOL = 1e-4

# -------------------------------------------------------------
# Utility: load "i: value" files into arrays of floats
# -------------------------------------------------------------
loadVec = (filePath) ->
  txt = fs.readFileSync(filePath, 'utf8')
  out = []
  for line in txt.split(/\r?\n/)
    continue unless line.includes(':')
    [_, rhs] = line.split(':', 2)
    v = parseFloat(rhs)
    continue if Number.isNaN(v)
    out.push v
  out

# -------------------------------------------------------------
# Compare two vectors
# -------------------------------------------------------------
compare = (pyPath, cppPath, name) ->
  unless fs.existsSync(pyPath)
    console.log "[#{name}] missing python file: #{pyPath}"
    return false

  unless fs.existsSync(cppPath)
    console.log "[#{name}] missing cpp file: #{cppPath}"
    return false

  a = loadVec pyPath
  b = loadVec cppPath

  if a.length isnt b.length
    console.log "[#{name}] LENGTH MISMATCH: #{a.length} vs #{b.length}"
    return false

  maxAbs = 0
  maxIdx = -1
  warnCount = 0
  failCount = 0

  for i in [0...a.length]
    diff = Math.abs(a[i] - b[i])
    if diff > maxAbs
      maxAbs = diff
      maxIdx = i
    if diff > WARN_TOL then warnCount++
    if diff > FAIL_TOL then failCount++

  status =
    if failCount > 0 then "FAIL"
    else if warnCount > 0 then "WARN"
    else "OK"

  console.log "[#{name}] #{status}  len=#{a.length}  max_abs=#{maxAbs} @ idx=#{maxIdx}  warn=#{warnCount} fail=#{failCount}"

  failCount is 0

# -------------------------------------------------------------
# Fixed comparisons (pre-RoPE)
# -------------------------------------------------------------
console.log "\n=== RAW PROJECTION CHECKS ==="

rawPairs = [
  ["debug_py/python_q_proj.log", "debug_cpp/first_layer0_q_proj.log", "Q raw"]
  ["debug_py/python_k_proj.log", "debug_cpp/first_layer0_k_proj.log", "K raw"]
  ["debug_py/python_v_proj.log", "debug_cpp/first_layer0_v_proj.log", "V raw"]
]

rawOK = true
for [py, cpp, name] in rawPairs
  ok = compare py, cpp, name
  rawOK = rawOK and ok

unless rawOK
  console.log "\nABORTING: raw Q/K/V mismatch — RoPE & attention comparisons skipped."
  process.exit(1)

# -------------------------------------------------------------
# RoPE comparisons — per position
# -------------------------------------------------------------
console.log "\n=== RoPE CHECKS ==="

ropeDirPy  = "debug_py"
ropeDirCpp = "debug_cpp"

positions = fs.readdirSync(ropeDirPy)
  .map (f) -> f.match(/first_layer0_q_after_rope_pos_(\d+)\.log/)
  .filter(Boolean)
  .map (m) -> parseInt(m[1])
  .sort (a,b) -> a - b

console.log "RoPE positions:", positions.join(", ")

for pos in positions
  compare(
    "#{ropeDirPy}/first_layer0_q_after_rope_pos_#{pos}.log",
    "#{ropeDirCpp}/first_layer0_q_after_rope_pos_#{pos}.log",
    "Q after RoPE pos=#{pos}"
  )

  compare(
    "#{ropeDirPy}/first_layer0_k_after_rope_pos_#{pos}.log",
    "#{ropeDirCpp}/first_layer0_k_after_rope_pos_#{pos}.log",
    "K after RoPE pos=#{pos}"
  )

# -------------------------------------------------------------
# Logits argmax inspection (unchanged)
# -------------------------------------------------------------
console.log "\n=== LOGITS ARGMAX ==="

loadVector = (p) ->
  txt = fs.readFileSync p, 'utf8'
  vals = []
  for line in txt.split(/\r?\n/)
    m = line.match /^(\d+):\s+([\-0-9.eE]+)/
    continue unless m
    vals.push parseFloat m[2]
  vals

inspectLogits = (p) ->
  return unless fs.existsSync(p)
  v = loadVector p
  bestVal = -1e30
  bestIdx = -1
  for val, i in v
    if val > bestVal
      bestVal = val
      bestIdx = i
  console.log "logits: len=#{v.length} argmax=#{bestIdx} value=#{bestVal}"

inspectLogits "debug_cpp/cpp_logits_last_token.log"
