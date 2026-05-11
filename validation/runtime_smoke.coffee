CoffeeScript = require 'coffeescript'
CoffeeScript.register()
path = require 'path'

gen = require '../generator/generator.coffee'
tok = require '../tokenizer/tokenizer.coffee'

session = gen.createSession
  modelPath: path.resolve __dirname, '../loraland/phi-1_5'

prompt = "one two three four five"
promptTokens = tok.encode prompt
console.log "prompt and tokens",prompt, promptTokens

ids = session.generate
  promptTokens: promptTokens
  maxTokens:  8
  temperature: 0.5 
  topP: 0.8

console.log tok.decode ids
