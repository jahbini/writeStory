fs = require 'fs'

clean = (txt) ->
  s = String(txt ? '')
  s = s.replace(/{{{First Name}}}/g, 'friend')
  s = s.replace(/&[a-zA-Z]+;/g, ' ')
  s = s.replace(/\[([^\]]+)\]\[\d+\]/g, '$1')
  s = s.replace(/\[\d+\]/g, '')
  s = s.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
  s = s.replace(/[_*]{1,3}([^*_]+)[_*]{1,3}/g, '$1')
  s = s.replace(/ {2,}/g, ' ')
  s.trim()

safe = (title) ->
  String(title ? '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') or 'untitled'

@step =
  desc: "Convert markdown stories into marshalled JSONL segments"

  action: (M, stepName) ->
    inPath = M.getStepParam stepName, 'stories_md'
    outKey = M.getStepParam stepName, 'marshalled_stories'
    mode   = M.getStepParam stepName, 'split_mode'

    existing = M.theLowdown(outKey).value
    if Array.isArray(existing) and existing.length > 0
      console.log "[md2segments] output already exists, skipping"
      M.saveThis "done:#{stepName}", true
      return

    mdEntry = M.theLowdown(inPath)
    raw = mdEntry.value
    raw = await mdEntry.notifier if raw is undefined
    throw new Error "Markdown not found: #{inPath}" unless raw?

    unless mode in ['story', 'paragraph']
      throw new Error "Unsupported split_mode: #{mode}"

    lines = raw.split /\r?\n/
    stories = []
    currentTitle = null
    buffer = []

    flushStory = ->
      return unless currentTitle? and buffer.length
      text = clean buffer.join("\n")
      stories.push title: currentTitle, text: text if text.length
      buffer = []

    for line in lines
      if line.startsWith '# '
        flushStory()
        currentTitle = line.slice(2).trim()
      else
        buffer.push line

    flushStory()

    rows = []
    if mode is 'story'
      for story in stories
        rows.push
          meta:
            doc_id: safe(story.title)
            paragraph_index: '001'
            title: story.title
          text: story.text
    else
      for story in stories
        baseId = safe story.title
        paragraphs = story.text.split(/\n/)
          .map(clean)
          .filter (p) -> p.length
        index = 1
        for paragraph in paragraphs
          rows.push
            meta:
              doc_id: baseId
              paragraph_index: String(index).padStart(3, '0')
              title: story.title
            text: paragraph
          index += 1

    console.log "[md2segments] stories:", stories.length, "segments:", rows.length
    M.saveThis outKey, rows
    M.saveThis "done:#{stepName}", true
    return
