#!/usr/bin/env coffee
###
init_hf_to_loraland.coffee
------------------------------------------------------------
Pipeline init step (HARDENED):
• Uses git + git-lfs (no HF CLI, no Python)
• Detects failures correctly
• Retries 3 times with 10-minute backoff
• Idempotent + restart-safe
• Memo is sole source of truth
###

path  = require 'path'
cp    = require 'child_process'

SLEEP_10_MIN = 10 * 60 * 1000
MAX_RETRIES  = 3

sleep = (ms) ->
  end = Date.now() + ms
  while Date.now() < end then null
  return

run = (cmd, args, cwd = null) ->
  cp.execFileSync cmd, args,
    cwd: cwd
    stdio: 'inherit'

runSh = (cmd, cwd = null) ->
  cp.execSync cmd,
    cwd: cwd
    stdio: 'pipe'
    encoding: 'utf8'

@step =
  desc: "Initialize base HF model into loraland (git + lfs, retry-hardened)"

  action: (M, stepName) ->

    throw new Error "Missing stepName" unless stepName?
    throw new Error "Memo missing getStepParam()" unless typeof M.getStepParam is 'function'

    # ------------------------------------------------------------
    # Read parameters from Memo
    # ------------------------------------------------------------

    hfModelId = M.getStepParam stepName, 'model'
    loraRoot  = M.getStepParam stepName, 'loraLand'

    throw new Error "Missing model param" unless hfModelId?
    throw new Error "Missing loraLand param" unless loraRoot?

    targetDir = path.resolve loraRoot
    repoUrl   = "https://huggingface.co/#{hfModelId}"

    M.saveThis 'modelDir', targetDir
    console.log "model directory",targetDir

    # ------------------------------------------------------------
    # Short-circuit if already present AND non-empty
    # ------------------------------------------------------------

    present = false
    try
      out = runSh "find #{JSON.stringify(targetDir)} -mindepth 1 -maxdepth 1 | head -n 1"
      present = out.trim().length > 0
    catch then present = false
    hasWeights = false
    try
      out = runSh "find #{JSON.stringify(targetDir)} -type f \\( -name '*.safetensors' -o -name '*.bin' \\) | head -n 1"
      hasWeights = out.trim().length > 0
    catch then hasWeights = false

    if present && hasWeights
      console.log "[init] Model already present, skipping."
      return

    run 'mkdir', ['-p', targetDir]

    # ------------------------------------------------------------
    # Retry loop
    # ------------------------------------------------------------

    lastError = null

    for attempt in [1..MAX_RETRIES]

      console.log "[init] Attempt #{attempt} of #{MAX_RETRIES}"

      try
        # Clean partial state before retry
        try run 'rm', ['-rf', targetDir] catch then null

        # Clone repo
        run 'git', ['clone', '--depth', '1', repoUrl, targetDir]

        # Pull LFS objects
        run 'git', ['lfs', 'pull'], targetDir

        # --------------------------------------------------------
        # Sanity check: repo must contain something real
        # --------------------------------------------------------

        repoHasFiles = false
        try
          out = runSh "find #{JSON.stringify(targetDir)} -mindepth 1 -maxdepth 1 | head -n 1"
          repoHasFiles = out.trim().length > 0
        catch then repoHasFiles = false
        throw new Error "Empty repo after clone" unless repoHasFiles

        hasWeights = false
        try
          out = runSh "find #{JSON.stringify(targetDir)} -type f \\( -name '*.safetensors' -o -name '*.bin' \\) | head -n 1"
          hasWeights = out.trim().length > 0
        catch then hasWeights = false
        throw new Error "No model weights found" unless hasWeights

        console.log "[init] Model successfully materialized."
        return

      catch err
        lastError = err
        console.log "[init] ERROR:", err.message

        if attempt < MAX_RETRIES
          console.log "[init] Waiting 10 minutes before retry…"
          sleep SLEEP_10_MIN
        else
          console.log "[init] Exhausted retries."

    # ------------------------------------------------------------
    # Final failure
    # ------------------------------------------------------------

    throw lastError
