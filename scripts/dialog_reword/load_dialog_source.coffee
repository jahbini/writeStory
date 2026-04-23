fs = require 'fs'
path = require 'path'

@step =
  desc: "Load a dialog text file for adapter rewrite"

  action: (L) ->
    inputFile = L.param 'input_file'
    throw new Error "[#{L.stepName}] Missing input_file param" unless inputFile?

    fullPath = path.resolve process.cwd(), String(inputFile)
    throw new Error "[#{L.stepName}] Input file not found: #{fullPath}" unless fs.existsSync fullPath

    stat = fs.statSync fullPath
    throw new Error "[#{L.stepName}] Input path is not a file: #{fullPath}" unless stat.isFile()

    text = fs.readFileSync fullPath, 'utf8'
    throw new Error "[#{L.stepName}] Input file is empty: #{fullPath}" unless text.trim().length

    console.log "[load_dialog_source] input:", fullPath
    console.log "[load_dialog_source] chars:", text.length

    L.make 'dialog_source_text', text
    L.done()
    return
