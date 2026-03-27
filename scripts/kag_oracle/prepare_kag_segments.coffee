@step =
  desc: "Build paragraph-level KAG input records from segments and emotion tags"

  action: (S) ->
    stories = await S.need 'marshalled_stories'
    emotionRows = await S.need 'kag_emotions'

    throw new Error "[#{S.stepName}] marshalled_stories must be an array" unless Array.isArray(stories)
    throw new Error "[#{S.stepName}] kag_emotions must be an array" unless Array.isArray(emotionRows)

    lookup = Object.create null
    for row in emotionRows
      docId = row?.meta?.doc_id
      paragraphIndex = row?.meta?.paragraph_index
      emotions = row?.emotions
      continue unless docId?
      continue unless paragraphIndex?
      continue unless emotions?
      lookup["#{docId}|#{paragraphIndex}"] = emotions

    matched = 0
    outRows = []

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

      outRows.push row

    console.log "[prepare_kag_segments] segments matched:", matched
    console.log "[prepare_kag_segments] rows written:", outRows.length

    S.make 'kag_segments', outRows
    S.done()
    return
