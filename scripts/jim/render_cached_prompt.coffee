renderField = (label, field) ->
  return null unless field?
  if typeof field isnt 'object' or Array.isArray(field)
    text = String(field).trim()
    return null unless text.length
    return "#{label}:\n#{text}"
  lines = []
  lines.push "#{label}:"
  for own key, value of field when value?
    text = String(value).trim()
    continue unless text.length
    lines.push "- #{key}: #{text}"
  return null unless lines.length > 1
  lines.join "\n"

@step =
  desc: "Render final cached prompt text from prompt-cache spec"

  action: (M, stepName) ->
    specEntry = M.theLowdown 'prompt_cache_spec'
    spec = specEntry?.value
    if spec is undefined
      if typeof specEntry?.waitFor is 'function'
        spec = await specEntry.waitFor()
      else if specEntry?.notifier?
        spec = await specEntry.notifier
    throw new Error "[#{stepName}] Missing input key 'prompt_cache_spec'" if spec is undefined

    sections = []
    sections.push spec.stable_instructions.join("\n") if Array.isArray(spec.stable_instructions)

    if Array.isArray(spec.rules) and spec.rules.length
      sections.push "Rules:\n" + (("- #{rule}" for rule in spec.rules).join "\n")

    fieldSections = []
    kagFields = spec.kag_fields ? {}
    for label in Object.keys(kagFields)
      rendered = renderField label, kagFields[label]
      fieldSections.push rendered if rendered?

    if fieldSections.length
      sections.push "KAG context:\n" + fieldSections.join "\n\n"

    template = spec.story_template ? {}
    sections.push "#{template.label ? 'Story fragment'}:\n\n#{template.text ? ''}"

    prompt = sections.join "\n\n"

    M.saveThis "cached_prompt_text", prompt
    M.saveThis "done:#{stepName}", true
    return
