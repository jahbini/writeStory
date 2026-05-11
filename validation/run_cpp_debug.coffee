#!/usr/bin/env coffee
# run_cpp_debug.coffee — drive metal_llm to emit debug_cpp/*.log
CoffeeScript = require 'coffeescript'
CoffeeScript.register()

fs   = require 'fs'
path = require 'path'

tokenizer = require '../tokenizer/tokenizer.coffee'
{ createSession } = require '../generator/generator.coffee'

OUTDIR = 'debug_cpp'
unless fs.existsSync OUTDIR
  fs.mkdirSync OUTDIR

# Make sure C++ debug is enabled
process.env.MLX_DUMP_LOGITS ?= '3'
console.log "[debug] MLX_DUMP_LOGITS=#{process.env.MLX_DUMP_LOGITS}"

# Prompt & tokens
prompt = process.argv.slice(2).join(' ') or 'one two three four'
tokens = tokenizer.encode prompt

console.log "prompt:", prompt
console.log "tokens:", tokens, tokenizer.decode tokens

# Create Metal session (same model path you use elsewhere).
# Warning: the legacy loadModel path is not yet memory-safe for Qwen-sized
# models; use this diagnostic only after the loader has been redesigned.
modelPath = path.resolve __dirname, '../pipes/Qwen_Qwen3-4B-Instruct-2507/build/model4'
session   = createSession modelPath: modelPath

# Drive the model one token at a time; C++ side should dump into debug_cpp
for token, pos in tokens
  logits = session.forward token, pos
  console.log "[node.forwardStep] tokenId=#{token} pos=#{pos} logits_len=#{logits.length}"

session.free()
console.log "[done] metal_llm.cpp should have written logs into #{OUTDIR}"
