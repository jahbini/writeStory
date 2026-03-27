extractJSON = (raw) ->
  return {} unless raw?
  block = raw.match(/\{[\s\S]*\}/)?[0]
  try JSON.parse(block) catch then {}

@step =
  desc: "Classify a batch of untagged markdown segments with the emotion oracle"

  action: (S) ->
    promptPrefix = S.param 'prompt_prefix'
    promptSuffix = S.param 'prompt_suffix'
    batchSz = S.param 'batch_size'
    maxTok = S.param 'max_tokens'
    quantizedModelMemoKey = S.param 'quantized_model_memo_key', 'quantizedModelDir'
    modelDir = S.theLowdown(quantizedModelMemoKey)?.value ? S.param('model_dir') ? S.theLowdown('modelDir')?.value
    throw new Error "[oracle_ask] Missing model_dir/quantized model path" unless modelDir?
    segments = await S.need 'marshalled_stories'
    taggedRows = await S.peek 'kag_emotions', []

    throw new Error "[#{S.stepName}] marshalled_stories must be an array" unless Array.isArray(segments)
    taggedRows = [] unless Array.isArray(taggedRows)

    tagged = new Set()
    for row in taggedRows when row?.meta?
      tagged.add "#{row.meta.doc_id}|#{row.meta.paragraph_index}"

    remaining = 0
    for segment in segments
      key = "#{segment.meta?.doc_id}|#{segment.meta?.paragraph_index}"
      continue if tagged.has key
      remaining += 1

    pending = []
    for segment in segments
      key = "#{segment.meta?.doc_id}|#{segment.meta?.paragraph_index}"
      continue if tagged.has key
      pending.push segment
      break if pending.length >= batchSz

    console.log "[oracle_ask] pending:", pending.length
    console.log "[oracle_ask] stories left after this batch:", Math.max(remaining - pending.length, 0)
    S.make 'kag_viewed', pending

    newStoryIdSet = new Set()
    for segment in pending
      title = segment?.meta?.title
      continue unless title?
      newStoryIdSet.add title
    newStoryIds = Array.from newStoryIdSet

    if pending.length is 0
      console.error "JIM BAD EXIT"
      S.make 'new_story_ids', newStoryIds
      S.make 'kag_emotions', taggedRows
      S.done()
      return

    outRows = taggedRows.slice()

    for segment in pending
      text = segment.text ? ''
      meta = segment.meta ? {}
      prompt = "#{promptPrefix}#{text}#{promptSuffix}"

      raw = S.callMLX 'generate',
         model: modelDir
         prompt: prompt
         "max-tokens": String(maxTok)
         "max-kv-size": 1024
         temp: String S.param "temperature"
         "top-p": String S.param 'top_p'
         "top-k": String S.param 'top_k'

      outRows.push
        meta:
          doc_id: meta.doc_id
          paragraph_index: meta.paragraph_index
        emotions: extractJSON(raw)

      console.log "JIM Emotions from", meta.doc_id, raw
      console.log "[oracle_ask] tagged #{meta.doc_id} #{meta.paragraph_index}"

    S.make 'new_story_ids', newStoryIds
    S.make 'kag_emotions', outRows
    S.done()
    return
