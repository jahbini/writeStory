fs = require 'fs'
path = require 'path'

@step =
  desc: "Build paragraph-level KAG input records from segments and emotion tags"

  action: (M, stepName) ->
    storiesKey = M.getStepParam stepName, 'marshalled_stories'
    emotionsKey = M.getStepParam stepName, 'kag_emotions'
    outputPath = M.getStepParam stepName, 'kag_segments'

    storiesEntry = M.theLowdown storiesKey
    stories = storiesEntry?.value
    stories = await storiesEntry.notifier if stories is undefined

    emotionsEntry = M.theLowdown emotionsKey
    emotionRows = emotionsEntry?.value
    emotionRows = await emotionsEntry.notifier if emotionRows is undefined

    throw new Error "[#{stepName}] #{storiesKey} must be an array" unless Array.isArray(stories)
    throw new Error "[#{stepName}] #{emotionsKey} must be an array" unless Array.isArray(emotionRows)

    lookup = Object.create null
    for row in emotionRows
      docId = row?.meta?.doc_id
      paragraphIndex = row?.meta?.paragraph_index
      emotions = row?.emotions
      continue unless docId?
      continue unless paragraphIndex?
      continue unless emotions?
      lookup["#{docId}|#{paragraphIndex}"] = emotions

    fs.mkdirSync path.dirname(outputPath), { recursive: true }
    fs.writeFileSync outputPath, '', 'utf8'

    matched = 0
    rowsWritten = 0

    for segment in stories
      docId = segment?.meta?.doc_id
      paragraphIndex = segment?.meta?.paragraph_index
      title = segment?.meta?.title
      text = segment?.text

      continue unless docId?
      continue unless paragraphIndex?
      continue unless title?
      continue unless text?

      emotions = lookup["#{docId}|#{paragraphIndex}"]
      continue unless emotions?

      matched += 1

      row =
        meta:
          doc_id: docId
          paragraph_index: paragraphIndex
          title: title
        text: text
        emotions: emotions

      fs.appendFileSync outputPath, JSON.stringify(row) + "\n", 'utf8'
      rowsWritten += 1

    console.log "[prepare_kag_segments] segments matched:", matched
    console.log "[prepare_kag_segments] rows written:", rowsWritten

    M.saveThis "done:#{stepName}", true
    return
