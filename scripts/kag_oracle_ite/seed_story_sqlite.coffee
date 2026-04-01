clean = (txt) ->
  s = String(txt ? '')
  s = s.replace(/{{{First Name}}}/g, 'friend')
  s = s.replace(/https?:\/\/\S+/g, '')
  s = s.replace(/&(rsquo|lsquo|apos|#39);/gi, "'")
  s = s.replace(/&(rdquo|ldquo|quot);/gi, '"')
  s = s.replace(/&[a-zA-Z#0-9]+;/g, ' ')
  s = s.replace(/\[([^\]]+)\]\[\d+\]/g, '$1')
  s = s.replace(/\[\d+\]/g, '')
  s = s.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
  s = s.replace(/[_*]{1,3}([^*_]+)[_*]{1,3}/g, '$1')
  s = s.replace(/ {2,}/g, ' ')
  lines = s.split /\r?\n/
  lines = (line for line in lines when not /^:\s*$/.test(String(line ? '').trim()))
  while lines.length
    line = String(lines[lines.length - 1] ? '').trim()
    break unless /^:\s*https?:\/\/\S+\s*$/.test(line) or /^https?:\/\/\S+\s*$/.test(line)
    lines.pop()
  lines.join("\n").trim()

safe = (title) ->
  String(title ? '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') or 'untitled'

@step =
  desc: "Seed storyByID sqlite records from markdown stories"

  action: (S) ->
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
          text: "#{text}\n\n<stop>"
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
