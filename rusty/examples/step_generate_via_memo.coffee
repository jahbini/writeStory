# rusty/examples/step_generate_via_memo.coffee
#
# Pipeline-facing example only.
# This step is intentionally isolated under `rusty/` and not wired into any
# live recipe. It shows the shape of an ordinary step using Memo visibility to
# request generation through an `mlx/...` key.

resolvePrompt = (L) ->
  direct = String(L.param('prompt', '') ? '').trim()
  return direct if direct.length
  'hello from pipeline'

resolveModelHandle = (L) ->
  direct = String(L.param('model_handle', '') ? '').trim()
  return direct if direct.length
  'model:fake'

resolveSessionHandle = (L) ->
  direct = String(L.param('session_handle', '') ? '').trim()
  return direct if direct.length
  'sess:example'

waitForDefined = (entry, timeoutMs = 10000) ->
  Promise.race [
    Promise.resolve(entry.notifier)
    new Promise (_, reject) ->
      setTimeout (-> reject new Error("timeout waiting for Memo entry")), timeoutMs
  ]

@step =
  desc: "Example pipeline step that requests generation through an mlx/... Memo key"

  action: (L) ->
    model = resolveModelHandle L
    session = resolveSessionHandle L
    prompt = resolvePrompt L

    requestKey = 'mlx/generate/example.json'
    requestPayload =
      model: model
      session: session
      prompt: prompt

    # The step behaves like a normal pipeline step:
    # - write a visible ML request artifact into Memo
    # - wait on the matching Memo entry/notifier
    # - convert the bridge response into a normal output artifact
    requestEntry = L.saveThis requestKey, requestPayload
    response = await waitForDefined requestEntry

    throw new Error "[#{L.stepName}] bridge response must be an object" unless response? and typeof response is 'object'
    throw new Error "[#{L.stepName}] bridge request failed: #{response?.error?.message ? 'unknown error'}" unless response.ok is true

    text = String(response?.value?.text ? '').trim()
    throw new Error "[#{L.stepName}] bridge response missing generated text" unless text.length

    L.make 'rusty_story_text', text
    L.done()
    return
