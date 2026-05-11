CoffeeScript = require 'coffeescript'
CoffeeScript.register()

path = require 'path'

addonPath = path.resolve __dirname, '../metal/metal_llm.node'
metal = require addonPath

expectedExports = [
  'loadModel'
  'applyLora'
  'resetKV'
  'forwardStep'
  'freeModel'
  'getVocabSize'
  'getHiddenSize'
  'getNumLayers'
]

actualExports = Object.keys(metal).sort()
missingExports = expectedExports.filter (name) ->
  typeof metal[name] isnt 'function'

result =
  ok: missingExports.length is 0
  smoke_type: 'addon_export_only'
  note: 'Safe smoke only loads the native addon; it must not load model weights.'
  addon_path: addonPath
  expected_exports: expectedExports
  actual_exports: actualExports
  missing_exports: missingExports

console.log JSON.stringify result, null, 2

if missingExports.length
  process.exit 1
