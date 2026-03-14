loadArray = (M, key) ->
  entry = M.theLowdown key
  value = entry?.value
  value = await entry.notifier if value is undefined
  value

mkPrompt = (row) ->
  """
Continue in the same voice and mannner as the text below.
#{row.prompt}
""".trim()

isSequential = (a, b) ->
  return false unless a?.meta?.doc_id? and b?.meta?.doc_id?
  return false unless a.meta.doc_id is b.meta.doc_id
  ai = parseInt(a.meta.paragraph_index, 10)
  bi = parseInt(b.meta.paragraph_index, 10)
  bi is ai + 1

@step =
  desc: "Rotate merged markdown segments into LoRA train and valid sets"

  action: (M, stepName) ->
    mergedKey = M.getStepParam stepName, 'merged_segments'
    trainKey  = M.getStepParam stepName, 'train_file'
    validKey  = M.getStepParam stepName, 'valid_file'

    mergedRows = await loadArray M, mergedKey
    oldTrain   = await loadArray M, trainKey
    oldValid   = await loadArray M, validKey

    throw new Error "#{mergedKey} must be an array" unless Array.isArray(mergedRows)
    oldTrain = [] unless Array.isArray(oldTrain)
    oldValid = [] unless Array.isArray(oldValid)

    newTrain = []
    skipped = 0

    for index in [0...mergedRows.length - 1]
      current = mergedRows[index]
      nextRow = mergedRows[index + 1]
      unless isSequential(current, nextRow)
        skipped += 1
        continue
      continue unless current?.prompt?
      newTrain.push
        prompt: mkPrompt(current)
        completion: nextRow.prompt

    newValid = oldValid.concat oldTrain
    newValid = newTrain if newValid.length is 0

    console.log "[rotate_merged]"
    console.log "  merged rows:", mergedRows.length
    console.log "  new train pairs:", newTrain.length
    console.log "  skipped (non-seq):", skipped
    console.log "  old train:", oldTrain.length
    console.log "  old valid:", oldValid.length
    console.log "  -> new valid:", newValid.length

    M.saveThis trainKey, newTrain
    M.saveThis validKey, newValid
    M.saveThis "done:#{stepName}", true
    return
