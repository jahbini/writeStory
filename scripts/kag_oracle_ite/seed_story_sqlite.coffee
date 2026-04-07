clean = (txt) ->
  s = String(txt ? '')
  s = s.replace(/{{{\s*First Name\s*}}}/gi, 'friend')
  s = s.replace(/{{{\s*first Name\s*}}}/g, 'friend')
  s = s.replace(/{{{\s*first name\s*}}}/gi, 'friend')
  s = s.replace(/{{{\s*comment:[^}]*}}}/gi, ' ')
  s = s.replace(/{{{\s*roar:[^}]*}}}/gi, ' ')
  s = s.replace(/{{{\s*death:[^}]*}}}/gi, ' ')
  s = s.replace(/{{{\s*RogerObt\s*}}}/g, ' ')
  s = s.replace(/{{{\s*Roger\s*}}}/g, 'Roger')
  s = s.replace(/{{{[^}]+}}}/g, ' ')
  s = s.replace(/{{[^}]+}}/g, ' ')
  s = s.replace(/https?:\/\/\S+/g, '')
  s = s.replace(/&(rsquo|lsquo|apos|#39);/gi, "'")
  s = s.replace(/&(rdquo|ldquo|quot);/gi, '"')
  s = s.replace(/&(mdash|ndash|hellip);/gi, ' ')
  s = s.replace(/&(nbsp|amp);/gi, ' ')
  s = s.replace(/&[a-zA-Z#0-9]+;/g, ' ')
  s = s.replace(/<\/?(html|body|p|br|sup|em|strong|div|span|blockquote|script|style|table|tr|td|ul|ol|li|a|img|h[1-6])\b[^>]*>/gi, ' ')
  s = s.replace(/<\/?[A-Za-z_][A-Za-z0-9:_-]*(\s+[^>]*)?>/g, ' ')
  s = s.replace(/\[([^\]]+)\]\[\d+\]/g, '$1')
  s = s.replace(/\[\d+\]/g, '')
  s = s.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
  s = s.replace(/[_*]{1,3}([^*_]+)[_*]{1,3}/g, '$1')
  s = s.replace(/<<<|>>>/g, ' ')
  s = s.replace(/<stop>|<end>|<input>/gi, ' ')
  s = s.replace(/\bInput:|\bOutput:|\bPrompt:\s+\d+\s+tokens|\bGeneration:\s+\d+\s+tokens|\bPeak memory:/gi, ' ')
  s = s.replace(/[\u200B-\u200F\u2060\uFEFF\uFFFD]/g, '')
  s = s.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, ' ')
  s = s.replace(/[<>{}=+]{5,}/g, ' ')
  lines = s.split /\r?\n/
  outLines = []
  for line in lines
    current = String(line ? '')
    current = current.replace(/^\s*>\s?/, '')
    current = current.replace(/^\s*#{1,6}\s+/, '')
    current = current.replace(/\s{2,}/g, ' ').trim()
    continue if /^:\s*$/.test(current)
    continue if /^please confirm\b/i.test(current)
    continue if /^begin:\s*$/i.test(current)
    continue if /^story fragment:\s*$/i.test(current)
    outLines.push current
  lines = outLines
  while lines.length
    line = String(lines[lines.length - 1] ? '').trim()
    break unless /^:\s*https?:\/\/\S+\s*$/.test(line) or /^https?:\/\/\S+\s*$/.test(line)
    lines.pop()
  lines.join("\n").replace(/\n{3,}/g, "\n\n").trim()

safe = (title) ->
  String(title ? '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') or 'untitled'

@step =
  desc: "Seed storyByID sqlite records from markdown stories"

  action: (S) ->
    existingStories = S.theLowdown('allStories.jsonl')?.value
    throw new Error "[#{S.stepName}] allStories.jsonl must be an array" unless Array.isArray existingStories

    if existingStories.length > 0
      storyIDs = []
      for story in existingStories
        storyID = story?.story_id
        continue unless storyID?
        storyIDs.push storyID
      console.log "[seed_story_sqlite] sqlite already seeded, stories:", storyIDs.length
      S.make 'story_seed_ids', storyIDs
      S.done()
      return

    raw = await S.need 'stories_md'
    throw new Error "[#{S.stepName}] stories_md must be a string artifact" unless typeof raw is 'string'

    lines = raw.split /\r?\n/
    stories = []
    currentTitle = null
    buffer = []

    flushStory = ->
      return unless currentTitle? and buffer.length
      text = clean buffer.join("\n")
      if text.length
        stories.push
          story_id: safe(currentTitle)
          title: currentTitle
          text: text
      buffer = []

    for line in lines
      if line.startsWith '# '
        flushStory()
        currentTitle = line.slice(2).trim()
      else
        buffer.push line

    flushStory()

    storyIDs = []
    for story in stories
      S.saveThis "storyByID{#{story.story_id}}.json", story
      storyIDs.push story.story_id

    console.log "[seed_story_sqlite] stories seeded:", storyIDs.length
    S.make 'story_seed_ids', storyIDs
    S.done()
    return
