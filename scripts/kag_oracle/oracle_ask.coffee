loadArray = (M, key) ->
  entry = M.theLowdown key
  value = entry?.value
  value = [] if value is undefined
  value

extractJSON = (raw) ->
  return {} unless raw?
  block = raw.match(/\{[\s\S]*\}/)?[0]
  try JSON.parse(block) catch then {}

@step =
  desc: "Classify a batch of untagged markdown segments with the emotion oracle"

  action: (M, stepName) ->
    segKey  = M.getStepParam stepName, 'marshalled_stories'
    emoKey  = M.getStepParam stepName, 'kag_emotions'
    newKey  = M.getStepParam stepName, 'new_story_ids'
    promptPrefix = M.getStepParam stepName, 'prompt_prefix'
    promptSuffix = M.getStepParam stepName, 'prompt_suffix'
    batchSz = M.getStepParam stepName, 'batch_size'
    maxTok  = M.getStepParam stepName, 'max_tokens'
    viewed  = M.getStepParam stepName,  'kag_viewed'
    quantizedModelMemoKey = M.getStepParam stepName, 'quantized_model_memo_key', 'quantizedModelDir'
    modelDir = M.theLowdown(quantizedModelMemoKey)?.value ? M.getStepParam(stepName, 'model_dir') ? M.theLowdown('modelDir')?.value
    throw new Error "[oracle_ask] Missing model_dir/quantized model path" unless modelDir?
    segments = await loadArray M, segKey
    taggedRows = await loadArray M, emoKey

    throw new Error "#{segKey} must be an array" unless Array.isArray(segments)
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
    M.saveThis viewed, pending

    newStoryIdSet = new Set()
    for segment in pending
      title = segment?.meta?.title
      continue unless title?
      newStoryIdSet.add title
    newStoryIds = Array.from newStoryIdSet

    if pending.length is 0
      console.error "JIM BAD EXIT"
      M.saveThis newKey, newStoryIds
      M.saveThis emoKey, taggedRows
      M.saveThis "done:#{stepName}", true
      return

    outRows = taggedRows.slice()

    for segment in pending
      text = segment.text ? ''
      meta = segment.meta ? {}
      prompt = "#{promptPrefix}#{text}#{promptSuffix}"

      raw = M.callMLX 'generate',
         model: modelDir
         prompt: prompt
         "max-tokens": String(maxTok)
         "max-kv-size": 1024
         temp: String M.getStepParam stepName, "temperature"
         "top-p": String M.getStepParam stepName, 'top_p'
         "top-k": String M.getStepParam stepName, 'top_k'

      outRows.push
        meta:
          doc_id: meta.doc_id
          paragraph_index: meta.paragraph_index
        emotions: extractJSON(raw)

      console.log "JIM Emotions from", meta.doc_id, raw
      console.log "[oracle_ask] tagged #{meta.doc_id} #{meta.paragraph_index}"

    M.saveThis newKey, newStoryIds
    M.saveThis emoKey, outRows
    M.saveThis "done:#{stepName}", true
    return
