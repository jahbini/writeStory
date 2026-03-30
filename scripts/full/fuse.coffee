path = require 'path'

@step =
  desc: "Fuse LoRA adapters and quantize models"

  action: (S) ->
    registry = await S.need 'artifacts_registry'

    doFuse = !!S.param 'do_fuse'
    dryRun = !!S.param 'dry_run'

    for entry in (registry.runs ? [])
      fusedDir = entry.fused_dir ? path.join(S.param('output_dir'), 'fused')
      quantDir = entry.quantized_dir ? path.join(S.param('output_dir'), 'quantized')

      if doFuse
        fuseArgs =
          model: entry.model_id
          "adapter-path": entry.adapter_dir
          "save-path": fusedDir
        console.log "[fuse] fuse #{entry.model_id} -> #{fusedDir}"
        S.callMLX 'fuse', fuseArgs unless dryRun
        entry.fused_dir = fusedDir

      convertArgs =
        "hf-path": entry.fused_dir ? fusedDir
        "mlx-path": quantDir
      console.log "[fuse] quantize #{entry.model_id} -> #{quantDir}"
      S.callMLX 'convert', convertArgs unless dryRun

      entry.quantized_dir = quantDir
      entry.quantize_bits = S.param 'q_bits'
      entry.q_group_size = S.param 'q_group'

    registry.updated_utc = new Date().toISOString().replace(/\.\d+Z$/, 'Z')
    S.make 'artifacts_registry', registry
    S.done()
    return
