# meta/txt.coffee
fs   = require 'fs'
path = require 'path'

module.exports = (M, opts={}) ->
    baseDir = opts.baseDir ? process.cwd()
    readText = (p) -> if fs.existsSync(p) then fs.readFileSync(p,'utf8') else undefined
    writeText = (p,s) -> fs.mkdirSync(path.dirname(p),{recursive:true}); fs.writeFileSync(p,s,'utf8')

    M.addMetaRule "txt",
      /\.txt$/i,
      (key, value) ->
        dest = path.join(baseDir, key)
        if value is undefined
          return readText(dest)

        text =
          if Array.isArray(value)
            value.join('\n')
          else
            String(value ? '')

        writeText(dest, text)
        value
