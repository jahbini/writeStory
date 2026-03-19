mkPrompt = (row) ->
  """
Continue in the same voice and mannner as the text below.
#{row.prompt}
""".trim()

toTextExample = (row) ->
  return null unless row?
  return row if row.text?
  if row.prompt? and row.completion?
    return text: "#{row.prompt}\n\n#{row.completion}"
  null

isSequential = (a, b) ->
  return true
  return false unless a?.meta?.doc_id? and b?.meta?.doc_id?
  return false unless a.meta.doc_id is b.meta.doc_id
  ai = parseInt(a.meta.paragraph_index, 10)
  bi = parseInt(b.meta.paragraph_index, 10)
  bi is ai + 1

@step =
  desc: "Rotate merged markdown segments into LoRA train and valid sets"

  action: (M, stepName) ->
    viewedKey = M.getStepParam stepName, 'viewed_segments'
    trainKey  = M.getStepParam stepName, 'train_file'
    validKey  = M.getStepParam stepName, 'valid_file'
    testKey  = M.getStepParam stepName,  'test_file'

    viewedEntry = M.theLowdown viewedKey
    viewedRows = viewedEntry?.value
    viewedRows = await viewedEntry.notifier if viewedRows is undefined

    trainEntry = M.theLowdown trainKey.value
    validEntry = M.theLowdown validKey.value

    throw new Error "#{viewedKey} must be an array" unless Array.isArray(viewedRows)

    newTrain = []
    skipped = 0

    for index in [0...viewedRows.length - 1]
      current = viewedRows[index]
      nextRow = viewedRows[index + 1]
      unless isSequential(current, nextRow)
        skipped += 1
        continue
      console.error "current",current
      continue unless current?.text?
      newTrain.push
        text: "#{mkPrompt(current)}\n\n#{nextRow.text}"
      console.error "training key",newTrain



    M.saveThis trainKey, newTrain
    M.saveThis validKey, newTrain
    M.saveThis testKey, newTrain
    M.saveThis "done:#{stepName}", true
    return
