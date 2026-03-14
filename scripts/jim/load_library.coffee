fs = require 'fs'
path = require 'path'
yaml = require 'js-yaml'

@step =
  desc: "Load keyed Jim story library YAML"

  action: (M, stepName) ->
    libraryFile = M.getStepParam(stepName, 'library_file')

    execDir = M.theLowdown('env/EXEC')?.value ? process.cwd()
    cwdDir  = M.theLowdown('env/CWD')?.value ? process.cwd()

    resolveInputPath = (p) ->
      return null unless typeof p is 'string' and p.length > 0
      return p if path.isAbsolute(p)
      fromExec = path.resolve(execDir, p)
      return fromExec if fs.existsSync(fromExec)
      path.resolve(cwdDir, p)

    libraryPath = resolveInputPath(libraryFile)

    unless libraryPath? and fs.existsSync(libraryPath)
      throw new Error "[#{stepName}] Missing library_file: #{libraryFile} (resolved: #{libraryPath})"

    raw = fs.readFileSync libraryPath, 'utf8'
    doc = yaml.load(raw)

    unless doc?.library?
      throw new Error "[#{stepName}] Library YAML missing top-level 'library'"
    unless doc?.stories?
      throw new Error "[#{stepName}] Library YAML missing top-level 'stories'"

    out =
      source_file: libraryPath
      library: doc.library
      stories: doc.stories

    M.saveThis "story_library", out
    M.saveThis "done:#{stepName}", true
    return
