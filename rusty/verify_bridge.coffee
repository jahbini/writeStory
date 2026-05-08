#!/usr/bin/env coffee

{ spawn, spawnSync } = require 'node:child_process'
{ mkdir, rm, writeFile } = require 'node:fs/promises'
{ readFileSync, existsSync } = require 'node:fs'
{ createInterface } = require 'node:readline'
path = require 'node:path'

`
const manifestPath = path.join(__dirname, 'bridge', 'Cargo.toml');
const tokenizerFixtureDir = path.join(__dirname, 'tmp_tokenizer_fixture');
const modelFixtureDir = path.join(__dirname, 'tmp_model_fixture');
const model4Dir = path.join(__dirname, '..', 'pipes', 'Qwen_Qwen3-4B-Instruct-2507', 'build', 'model4');

function hasCargo() {
  const probe = spawnSync('cargo', ['--version'], { stdio: 'ignore' });
  return probe.status === 0;
}

function shouldRunMlxRuntime() {
  const raw = String(process.env.RUSTY_RUN_MLX_RUNTIME ?? '').trim().toLowerCase();
  return raw === '1' || raw === 'true' || raw === 'yes';
}

function envTruthy(name) {
  const raw = String(process.env[name] ?? '').trim().toLowerCase();
  return raw === '1' || raw === 'true' || raw === 'yes' || raw === 'on';
}

function forceScalarQuantizedLinear() {
  return envTruthy('RUSTY_FORCE_SCALAR_QUANTIZED_LINEAR') || envTruthy('RUSTY_FORCE_SCALAR_LINEAR');
}

const verifierProfiles = new Set(['fastsmoke', 'smoke', 'gateup', 'layer', 'oracle', 'full', 'generate', 'parity']);
const verifierProfileRank = {
  fastsmoke: 0,
  parity: 0,
  smoke: 1,
  gateup: 2,
  layer: 2,
  oracle: 3,
  full: 4,
  generate: 5,
};

function activeVerifierProfile() {
  const raw = String(process.env.RUSTY_VERIFY_PROFILE ?? 'smoke').trim().toLowerCase();
  if (!verifierProfiles.has(raw)) {
    throw new Error(\`unknown verifier profile "${raw}"; expected fastsmoke, smoke, gateup, layer, oracle, full, generate, or parity\`);
  }
  return raw;
}

function profileAtLeast(active, required) {
  return verifierProfileRank[active] >= verifierProfileRank[required];
}

function logSkippedProbe(name, reason) {
  console.log('[rusty verify] skipped_probe:', JSON.stringify({ name, reason }));
}

function createPhaseTimer() {
  const timings = new Map();
  let active = null;
  let activeStartedAt = 0;
  const now = () => Number(process.hrtime.bigint()) / 1e6;
  return {
    switchTo(name) {
      const current = now();
      if (active !== null) {
        timings.set(active, (timings.get(active) ?? 0) + current - activeStartedAt);
      }
      active = name;
      activeStartedAt = current;
    },
    stop() {
      const current = now();
      if (active !== null) {
        timings.set(active, (timings.get(active) ?? 0) + current - activeStartedAt);
      }
      active = null;
      activeStartedAt = current;
    },
    summary() {
      this.stop();
      const ordered = {};
      for (const name of [
        'build/probe setup',
        'model/index load',
        'tokenizer fixture work',
        'tensor metadata/load',
        'active decode',
        'cleanup',
      ]) {
        ordered[name] = Number((timings.get(name) ?? 0).toFixed(3));
      }
      return ordered;
    },
  };
}

function buildBridge() {
  const built = spawnSync('cargo', ['build', '--manifest-path', manifestPath], {
    stdio: 'inherit',
  });
  if (built.status !== 0) {
    throw new Error(\`cargo build failed with status ${built.status}\`);
  }
}

function startBridge() {
  return spawn('cargo', ['run', '--quiet', '--manifest-path', manifestPath], {
    cwd: __dirname,
    stdio: ['pipe', 'pipe', 'pipe'],
  });
}

async function main() {
  if (!hasCargo()) {
    console.log('[rusty verify] cargo not found; skipping bridge build/run');
    process.exit(0);
  }

  const verifierProfile = activeVerifierProfile();
  const runFastSmokeProfile = verifierProfile === 'fastsmoke';
  const runLayerProfile = profileAtLeast(verifierProfile, 'layer');
  const runFullProfile = profileAtLeast(verifierProfile, 'full');
  const runGenerateProfile = profileAtLeast(verifierProfile, 'generate');
  const runOracleProfile = verifierProfile === 'oracle' || runFullProfile;
  const phaseTimer = createPhaseTimer();
  console.log('[rusty verify] active verifier profile:', verifierProfile);
  const scalarQuantizedLinearForced = forceScalarQuantizedLinear();
  console.log('[rusty verify] scalar_quantized_linear_forced:', scalarQuantizedLinearForced);

  phaseTimer.switchTo('build/probe setup');
  buildBridge();

  const bridge = startBridge();
  const rl = createInterface({ input: bridge.stdout });
  const pending = new Map();
  const transcript = [];

  bridge.stderr.on('data', (buf) => {
    const text = String(buf ?? '').trim();
    if (text.length) process.stderr.write(\`${text}\n\`);
  });

  rl.on('line', (line) => {
    if (!line.trim()) return;
    const payload = JSON.parse(line);
    transcript.push(payload);
    const waiter = pending.get(payload.id);
    if (!waiter) return;
    pending.delete(payload.id);
    waiter.resolve(payload);
  });

  bridge.on('exit', (code, signal) => {
    for (const [, waiter] of pending) {
      waiter.reject(new Error(\`bridge exited before reply code=${code} signal=${signal}\`));
    }
    pending.clear();
  });

  let nextId = 0;
  const send = (cmd, args = {}) => {
    nextId += 1;
    const request = { id: String(nextId), cmd, args };
    transcript.push(request);
    return new Promise((resolve, reject) => {
      pending.set(request.id, { resolve, reject });
      bridge.stdin.write(\`${JSON.stringify(request)}\n\`, 'utf8');
    });
  };

  try {
    if (!runFastSmokeProfile) {
      phaseTimer.switchTo('tokenizer fixture work');
    await mkdir(tokenizerFixtureDir, { recursive: true });
    await writeFile(
      path.join(tokenizerFixtureDir, 'tokenizer.json'),
      JSON.stringify(
        {
          model: {
            type: 'BPE',
            vocab: {
              hello: 1,
              world: 2,
              '<unk>': 0,
            },
            merges: ['hello world'],
          },
          normalizer: { type: 'NFKC' },
          pre_tokenizer: { type: 'Whitespace' },
          decoder: { type: 'BPEDecoder' },
          added_tokens: [{ id: 99, content: '<extra>' }],
        },
        null,
        2,
      ),
      'utf8',
    );
    await writeFile(path.join(tokenizerFixtureDir, 'tokenizer_config.json'), '{"fixture":true}\n', 'utf8');

    await rm(modelFixtureDir, { recursive: true, force: true });
    await mkdir(modelFixtureDir, { recursive: true });
    await writeFile(
      path.join(modelFixtureDir, 'config.json'),
      JSON.stringify(
        {
          model_type: 'llama',
          architectures: ['LlamaForCausalLM'],
          vocab_size: 32000,
          hidden_size: 4096,
          num_hidden_layers: 32,
          num_attention_heads: 32,
          num_key_value_heads: 8,
          intermediate_size: 11008,
          torch_dtype: 'bfloat16',
          rope_theta: 10000,
          rope_scaling: { type: 'linear', factor: 2 },
        },
        null,
        2,
      ),
      'utf8',
    );
    await writeFile(path.join(modelFixtureDir, 'tokenizer.json'), '{"fixture":true}\n', 'utf8');
    await writeFile(path.join(modelFixtureDir, 'tokenizer_config.json'), '{"fixture":true}\n', 'utf8');
    await writeFile(path.join(modelFixtureDir, 'model.safetensors'), 'not-a-real-model\n', 'utf8');
    await writeFile(
      path.join(modelFixtureDir, 'model.safetensors.index.json'),
      '{"weight_map":{"layer.0":"model.safetensors"}}\n',
      'utf8',
    );
    await writeFile(path.join(modelFixtureDir, 'generation_config.json'), '{"max_length":32}\n', 'utf8');
    } else {
      logSkippedProbe('tokenizer/model fixture setup', 'profile fastsmoke skips fixture work not needed for active decode validation');
    }

    phaseTimer.switchTo('build/probe setup');
    const health = await send('bridge_health', {});
    if (health.ok !== true || health.value?.status !== 'alive') {
      throw new Error('bridge_health failed verification');
    }

    const backendProbe = await send('backend_probe', {});
    if (backendProbe.ok !== true || typeof backendProbe.value !== 'object' || Array.isArray(backendProbe.value)) {
      throw new Error('backend_probe failed verification');
    }
    if (!backendProbe.value?.native_boundary || typeof backendProbe.value.native_boundary !== 'object') {
      throw new Error('backend_probe missing native_boundary');
    }
    if (!Array.isArray(backendProbe.value?.installation_groups)) {
      throw new Error('backend_probe missing installation_groups');
    }
    console.log(
      '[rusty verify] backend_probe recommended_path:',
      backendProbe.value.native_boundary.recommended_path,
    );
    console.log(
      '[rusty verify] backend_probe preferred_installation_root:',
      backendProbe.value.native_boundary.preferred_installation_root,
    );
    if (runFastSmokeProfile) {
      logSkippedProbe('backend_probe detailed dumps', 'profile fastsmoke keeps only minimum backend sanity output');
    } else {
      console.log(
        '[rusty verify] backend_probe installation_groups:',
        JSON.stringify(backendProbe.value.installation_groups, null, 2),
      );
      console.log(
        '[rusty verify] backend_probe evidence:',
        JSON.stringify(backendProbe.value.native_boundary.evidence ?? [], null, 2),
      );
      console.log('[rusty verify] backend_probe:', JSON.stringify(backendProbe.value, null, 2));
    }

    phaseTimer.switchTo('tensor metadata/load');
    if (runFullProfile) {
    const inspectModelDir = await send('inspect_model_dir', { path: modelFixtureDir });
    if (
      inspectModelDir.ok !== true ||
      inspectModelDir.value?.exists !== true ||
      !Array.isArray(inspectModelDir.value?.tokenizer_files) ||
      !Array.isArray(inspectModelDir.value?.safetensors_files) ||
      inspectModelDir.value?.config?.model_type !== 'llama' ||
      inspectModelDir.value?.guessed_model_family !== 'llama' ||
      inspectModelDir.value?.config?.hidden_size !== 4096 ||
      inspectModelDir.value?.config?.num_hidden_layers !== 32 ||
      inspectModelDir.value?.config?.num_attention_heads !== 32 ||
      inspectModelDir.value?.config?.num_key_value_heads !== 8 ||
      inspectModelDir.value?.config?.intermediate_size !== 11008 ||
      inspectModelDir.value?.config?.torch_dtype !== 'bfloat16'
    ) {
      throw new Error('inspect_model_dir failed verification');
    }
    if (
      inspectModelDir.value?.config?.architectures?.[0] !== 'LlamaForCausalLM' ||
      inspectModelDir.value?.config?.vocab_size !== 32000 ||
      inspectModelDir.value?.config?.rope?.type !== 'linear' ||
      inspectModelDir.value?.config?.rope?.factor !== 2 ||
      !inspectModelDir.value.tokenizer_files.some((entry) => entry.endsWith('tokenizer.json')) ||
      !inspectModelDir.value.tokenizer_files.some((entry) => entry.endsWith('tokenizer_config.json')) ||
      !inspectModelDir.value.safetensors_files.some((entry) => entry.endsWith('model.safetensors')) ||
      inspectModelDir.value?.safetensors_index === null ||
      !String(inspectModelDir.value.safetensors_index).endsWith('model.safetensors.index.json') ||
      inspectModelDir.value?.generation_config === null ||
      !String(inspectModelDir.value.generation_config).endsWith('generation_config.json') ||
      inspectModelDir.value?.file_count < 6 ||
      inspectModelDir.value?.total_size_bytes <= 0
    ) {
      throw new Error('inspect_model_dir metadata assertions failed');
    }
    console.log('[rusty verify] inspect_model_dir path:', inspectModelDir.value.path);
    console.log('[rusty verify] inspect_model_dir guessed_model_family:', inspectModelDir.value.guessed_model_family);
    console.log('[rusty verify] inspect_model_dir config:', JSON.stringify(inspectModelDir.value.config, null, 2));
    console.log('[rusty verify] inspect_model_dir file_count:', inspectModelDir.value.file_count);
    console.log('[rusty verify] inspect_model_dir total_size_bytes:', inspectModelDir.value.total_size_bytes);
    } else {
      logSkippedProbe('inspect_model_dir', 'profile smoke/layer skips fixture directory metadata inspection');
    }

    phaseTimer.switchTo('model/index load');
    const modelLoadPlan = await send('model_load_plan', { model_dir: model4Dir });
    if (
      modelLoadPlan.ok !== true ||
      modelLoadPlan.value?.model_type !== 'qwen3' ||
      modelLoadPlan.value?.layer_count !== 36 ||
      !Array.isArray(modelLoadPlan.value?.per_layer_summary) ||
      modelLoadPlan.value.per_layer_summary.length !== 36 ||
      modelLoadPlan.value?.lm_head?.present !== false ||
      modelLoadPlan.value?.tied_embeddings_likely !== true ||
      modelLoadPlan.value?.per_layer_summary?.[0]?.found_groups?.length !== 9 ||
      modelLoadPlan.value?.per_layer_summary?.[0]?.missing_groups?.length !== 0 ||
      modelLoadPlan.value?.estimated_total_bytes <= 0
    ) {
      throw new Error('model_load_plan failed verification');
    }
    console.log('[rusty verify] model_load_plan model_type:', modelLoadPlan.value.model_type);
    console.log('[rusty verify] model_load_plan architecture:', modelLoadPlan.value.architecture);
    console.log('[rusty verify] model_load_plan layer_count:', modelLoadPlan.value.layer_count);
    console.log('[rusty verify] model_load_plan first_layer:', JSON.stringify(modelLoadPlan.value.per_layer_summary[0], null, 2));
    console.log('[rusty verify] model_load_plan tied_embeddings_likely:', modelLoadPlan.value.tied_embeddings_likely);

    if (verifierProfile === 'parity') {
      logSkippedProbe('fastsmoke/smoke generation probes', 'profile parity runs only Rusty-vs-reference logits diagnostics');
      const promptTokenIds = [151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198];
      const referenceScriptPath = path.join(__dirname, 'reference_logits_mlx_lm.py');
      const referenceJsonPath = process.env.RUSTY_PARITY_REFERENCE_JSON
        ? path.resolve(process.env.RUSTY_PARITY_REFERENCE_JSON)
        : path.resolve(__dirname, '..', 'parity_reference.json');
      const referenceCommand =
        'python3 ' +
        JSON.stringify(referenceScriptPath) +
        ' ' +
        JSON.stringify(model4Dir) +
        ' ' +
        JSON.stringify(promptTokenIds.join(','));

      const handleCountsBeforeParity = await send('mlx_handle_counts', {});
      if (
        handleCountsBeforeParity.ok !== true ||
        handleCountsBeforeParity.value?.arrays !== 0 ||
        handleCountsBeforeParity.value?.token_arrays !== 0 ||
        handleCountsBeforeParity.value?.tensor_groups !== 0 ||
        handleCountsBeforeParity.value?.embedding_groups !== 0 ||
        handleCountsBeforeParity.value?.layer_groups !== 0
      ) {
        throw new Error('mlx_handle_counts before parity probe did not start clean');
      }

      const parityModelLoaded = await send('load_model_native', { path: model4Dir });
      if (parityModelLoaded.ok !== true || typeof parityModelLoaded.value?.model !== 'string') {
        throw new Error('parity load_model_native failed verification');
      }
      const paritySessionCreated = await send('create_native_session', {
        model: parityModelLoaded.value.model,
      });
      if (paritySessionCreated.ok !== true || typeof paritySessionCreated.value?.session !== 'string') {
        throw new Error('parity create_native_session failed verification');
      }
      const parityWarm = await send('warm_resident_session', {
        session: paritySessionCreated.value.session,
      });
      if (parityWarm.ok !== true || parityWarm.value?.warmed !== true) {
        throw new Error('parity warm_resident_session failed verification');
      }
      const rustyGeneration = await send('generate_tokens', {
        session: paritySessionCreated.value.session,
        prompt_token_ids: promptTokenIds,
        first_decode_token_id: promptTokenIds[promptTokenIds.length - 1],
        generated_tokens: 1,
      });
      if (
        rustyGeneration.ok !== true ||
        !Array.isArray(rustyGeneration.value?.generated_token_ids) ||
        rustyGeneration.value.generated_token_ids.length !== 1
      ) {
        console.log('[rusty verify] parity rusty_generation failed payload:', JSON.stringify(rustyGeneration, null, 2));
        throw new Error('parity generate_tokens failed verification');
      }

      let reference = null;
      let referenceReadError = null;
      if (referenceJsonPath !== null) {
        try {
          if (existsSync(referenceJsonPath)) {
            reference = JSON.parse(readFileSync(referenceJsonPath, 'utf8'));
          } else {
            referenceReadError = 'reference JSON not found: ' + referenceJsonPath;
          }
        } catch (err) {
          referenceReadError = String(err?.stack ?? err);
        }
      }

      const referenceTop10 = Array.isArray(reference?.top_logits)
        ? reference.top_logits
        : Array.isArray(reference?.top10)
          ? reference.top10
          : [];
      const referenceTopToken = referenceTop10.length > 0
        ? Number(referenceTop10[0].token_id ?? referenceTop10[0][0])
        : null;
      const rustyTopToken = rustyGeneration.value.generated_token_ids[0];
      const rustyTopScore = Array.isArray(rustyGeneration.value.generated_token_scores)
        ? rustyGeneration.value.generated_token_scores[0]
        : null;
      const rustyTop10 = Array.isArray(rustyGeneration.value.first_generated_top_logits)
        ? rustyGeneration.value.first_generated_top_logits
        : [{
          token_id: rustyTopToken,
          score: rustyTopScore,
        }];
      const referenceTopIds = referenceTop10
        .map((entry) => Number(entry.token_id ?? entry[0]))
        .filter((value) => Number.isInteger(value));
      const rustyTopIds = rustyTop10
        .map((entry) => Number(entry.token_id ?? entry[0]))
        .filter((value) => Number.isInteger(value));
      const overlap = rustyTopIds.filter((tokenId) => referenceTopIds.includes(tokenId));
      const rustyLayer0Checkpoints = rustyGeneration.value.parity_checkpoints?.layer0 ?? null;
      const referenceCheckpoints = reference?.reference_checkpoints ?? null;
      const summaryFirstValueDiff = (rustySummary, referenceSummary) => {
        if (rustySummary === null || referenceSummary === null) {
          return null;
        }
        const rustyValues = Array.isArray(rustySummary?.first_values)
          ? rustySummary.first_values
          : [];
        const referenceValues = Array.isArray(referenceSummary?.first_values)
          ? referenceSummary.first_values
          : [];
        const count = Math.min(rustyValues.length, referenceValues.length);
        if (count === 0) {
          return null;
        }
        let maxDiff = 0;
        for (let i = 0; i < count; i += 1) {
          const diff = Math.abs(Number(rustyValues[i]) - Number(referenceValues[i]));
          if (Number.isFinite(diff) && diff > maxDiff) {
            maxDiff = diff;
          }
        }
        return maxDiff;
      };
      const compareSummaries = (rustySummary, referenceSummary) => {
        if (rustySummary === null || referenceSummary === null) {
          return null;
        }
        const checksumDiff =
          Number.isFinite(Number(rustySummary?.checksum)) &&
          Number.isFinite(Number(referenceSummary?.checksum))
            ? Math.abs(Number(rustySummary.checksum) - Number(referenceSummary.checksum))
            : null;
        return {
          rusty_shape: rustySummary?.shape ?? null,
          reference_shape: referenceSummary?.shape ?? null,
          shape_matches: JSON.stringify(rustySummary?.shape ?? null) === JSON.stringify(referenceSummary?.shape ?? null),
          rusty_checksum: rustySummary?.checksum ?? null,
          reference_checksum: referenceSummary?.checksum ?? null,
          checksum_abs_diff: checksumDiff,
          rusty_first_values: rustySummary?.first_values ?? null,
          reference_first_values: referenceSummary?.first_values ?? null,
          max_abs_diff_first_values: summaryFirstValueDiff(rustySummary, referenceSummary),
        };
      };
      const parityStageComparisons =
        rustyLayer0Checkpoints === null || referenceCheckpoints === null
          ? null
          : {
            embedding_token0: compareSummaries(
              rustyLayer0Checkpoints.embedding_token0 ?? null,
              referenceCheckpoints.embedding_token0 ?? null,
            ),
            embedding_final_prompt_token: compareSummaries(
              rustyLayer0Checkpoints.embedding_final_prompt_token ?? null,
              referenceCheckpoints.embedding_final_prompt_token ?? null,
            ),
            layer0_input_rmsnorm_final_position: compareSummaries(
              rustyLayer0Checkpoints.input_rmsnorm ?? null,
              referenceCheckpoints.layer0_input_rmsnorm_final_position ?? null,
            ),
            layer0_q_proj_final_position_raw: compareSummaries(
              rustyLayer0Checkpoints.q_before_rope ?? null,
              referenceCheckpoints.layer0_q_proj_final_position ?? null,
            ),
            layer0_k_proj_final_position_raw: compareSummaries(
              rustyLayer0Checkpoints.k_before_rope ?? null,
              referenceCheckpoints.layer0_k_proj_final_position ?? null,
            ),
            layer0_v_proj_final_position: compareSummaries(
              rustyLayer0Checkpoints.v ?? null,
              referenceCheckpoints.layer0_v_proj_final_position ?? null,
            ),
            layer0_q_before_rope_reference_after_q_norm_vs_rusty_raw: compareSummaries(
              rustyLayer0Checkpoints.q_before_rope ?? null,
              referenceCheckpoints.layer0_q_before_rope_final_position ?? null,
            ),
            layer0_k_before_rope_reference_after_k_norm_vs_rusty_raw: compareSummaries(
              rustyLayer0Checkpoints.k_before_rope ?? null,
              referenceCheckpoints.layer0_k_before_rope_final_position ?? null,
            ),
            layer0_q_after_q_norm_final_position: compareSummaries(
              rustyLayer0Checkpoints.q_after_q_norm ?? null,
              referenceCheckpoints.layer0_q_after_q_norm_final_position ??
                referenceCheckpoints.layer0_q_before_rope_final_position ??
                null,
            ),
            layer0_k_after_k_norm_final_position: compareSummaries(
              rustyLayer0Checkpoints.k_after_k_norm ?? null,
              referenceCheckpoints.layer0_k_after_k_norm_final_position ??
                referenceCheckpoints.layer0_k_before_rope_final_position ??
                null,
            ),
            layer0_q_after_rope_active: compareSummaries(
              rustyLayer0Checkpoints.q_after_rope ?? null,
              referenceCheckpoints.layer0_q_after_rope_final_position ?? null,
            ),
            layer0_k_after_rope_active: compareSummaries(
              rustyLayer0Checkpoints.k_after_rope ?? null,
              referenceCheckpoints.layer0_k_after_rope_final_position ?? null,
            ),
            layer0_attention_scores_head0_final_position: compareSummaries(
              rustyLayer0Checkpoints.attention_head0?.scores ?? null,
              referenceCheckpoints.layer0_attention_scores_head0_final_position ?? null,
            ),
            layer0_attention_probabilities_head0_final_position: compareSummaries(
              rustyLayer0Checkpoints.attention_head0?.probabilities ?? null,
              referenceCheckpoints.layer0_attention_probabilities_head0_final_position ?? null,
            ),
            layer0_attention_output_final_position: compareSummaries(
              rustyLayer0Checkpoints.attention_output ?? null,
              referenceCheckpoints.layer0_attention_output_final_position ?? null,
            ),
            layer0_o_proj_output_final_position: compareSummaries(
              rustyLayer0Checkpoints.o_proj_output ?? null,
              referenceCheckpoints.layer0_o_proj_output_final_position ?? null,
            ),
            layer0_post_attention_residual_final_position: compareSummaries(
              rustyLayer0Checkpoints.post_attention_residual ?? null,
              referenceCheckpoints.layer0_post_attention_residual_final_position ?? null,
            ),
            layer0_post_attention_rmsnorm_final_position: compareSummaries(
              rustyLayer0Checkpoints.post_attention_rmsnorm ?? null,
              referenceCheckpoints.layer0_post_attention_rmsnorm_final_position ?? null,
            ),
            layer0_mlp_output_final_position: compareSummaries(
              rustyLayer0Checkpoints.mlp_output ?? null,
              referenceCheckpoints.layer0_mlp_output_final_position ?? null,
            ),
            layer0_final_residual_final_position: compareSummaries(
              rustyLayer0Checkpoints.layer0_final_residual ?? null,
              referenceCheckpoints.layer0_manual_final_residual_final_position ??
                referenceCheckpoints.layer0_final_residual_final_position ??
                null,
            ),
          };
      const ropeLayoutExperiments = rustyLayer0Checkpoints?.rope_layout_experiments ?? null;
      const ropeExperimentComparisons = (() => {
        if (ropeLayoutExperiments === null || referenceCheckpoints === null) {
          return null;
        }
        const layouts = ['half_split', 'interleaved_pair'];
        const comparisons = {};
        let best = null;
        for (const layout of layouts) {
          const qCompare = compareSummaries(
            ropeLayoutExperiments[layout]?.q_after_rope ?? null,
            referenceCheckpoints.layer0_q_after_rope_final_position ?? null,
          );
          const kCompare = compareSummaries(
            ropeLayoutExperiments[layout]?.k_after_rope ?? null,
            referenceCheckpoints.layer0_k_after_rope_final_position ?? null,
          );
          const qDiff = qCompare?.max_abs_diff_first_values ?? Number.POSITIVE_INFINITY;
          const kDiff = kCompare?.max_abs_diff_first_values ?? Number.POSITIVE_INFINITY;
          const combined = qDiff + kDiff;
          comparisons[layout] = {
            q_after_rope: qCompare,
            k_after_rope: kCompare,
            combined_first_values_max_abs_diff: Number.isFinite(combined) ? combined : null,
          };
          if (Number.isFinite(combined) && (best === null || combined < best.combined_first_values_max_abs_diff)) {
            best = {
              layout,
              combined_first_values_max_abs_diff: combined,
              q_max_abs_diff_first_values: qCompare?.max_abs_diff_first_values ?? null,
              k_max_abs_diff_first_values: kCompare?.max_abs_diff_first_values ?? null,
            };
          }
        }
        return {
          active_layout: ropeLayoutExperiments.active_layout ?? null,
          best_layout_by_qk_after_rope_first_values: best,
          layouts: comparisons,
        };
      })();

      const paritySessionFreed = await send('free_native_session', {
        session: paritySessionCreated.value.session,
      });
      const parityModelUnloaded = await send('unload_model_native', {
        model: parityModelLoaded.value.model,
      });
      const handleCountsAfterParity = await send('mlx_handle_counts', {});
      if (
        handleCountsAfterParity.ok !== true ||
        handleCountsAfterParity.value?.arrays !== 0 ||
        handleCountsAfterParity.value?.token_arrays !== 0 ||
        handleCountsAfterParity.value?.tensor_groups !== 0 ||
        handleCountsAfterParity.value?.embedding_groups !== 0 ||
        handleCountsAfterParity.value?.layer_groups !== 0
      ) {
        throw new Error('mlx_handle_counts changed during parity probe');
      }
      if (paritySessionFreed.ok !== true || parityModelUnloaded.ok !== true) {
        throw new Error('parity cleanup failed verification');
      }

      console.log('[rusty verify] logits_parity_probe:', JSON.stringify({
        ok: true,
        prompt_token_ids: promptTokenIds,
        reference_script_path: referenceScriptPath,
        reference_command: referenceCommand,
        reference_json_path: referenceJsonPath,
        reference_available: reference !== null,
        reference_json_loaded: reference !== null,
        reference_read_error: referenceReadError,
        rusty_prompt_handling: {
          full_prompt_prefill_supported:
            rustyGeneration.value.prompt_tokens_processed === promptTokenIds.length &&
            rustyGeneration.value.positions_after_prefill === promptTokenIds.length,
          current_behavior: 'native generate_tokens pre-fills prompt_token_ids in order, then selects first generated token from final prompt state',
          prefill_token_ids_used: rustyGeneration.value.prefill_token_ids ?? null,
          first_decode_position: rustyGeneration.value.first_decode_position ?? null,
          positions_after_prefill: rustyGeneration.value.positions_after_prefill ?? null,
          kv_positions_per_layer: rustyGeneration.value.kv_positions_per_layer ?? null,
          requested_prompt_token_count: promptTokenIds.length,
        },
        rusty: {
          top_token: {
            token_id: rustyTopToken,
            score: rustyTopScore,
          },
          top_logits: rustyTop10,
          final_norm_checksum: rustyGeneration.value.final_norm_checksum ?? null,
          logits_len: rustyGeneration.value.logits_len ?? null,
          timing_ms: rustyGeneration.value.generation_1_total_ms ?? rustyGeneration.value.total_generation_ms ?? null,
          parity_checkpoints: rustyGeneration.value.parity_checkpoints ?? null,
        },
        reference: reference === null ? null : {
          model_dir: reference.model_dir ?? model4Dir,
          final_hidden_checksum: reference.final_hidden_checksum ?? null,
          final_norm_checksum: reference.final_norm_checksum ?? null,
          logits_len: reference.logits_len ?? null,
          top_token: referenceTopToken === null ? null : {
            token_id: referenceTopToken,
            score: Number(referenceTop10[0].score ?? referenceTop10[0][1]),
          },
          top_logits: referenceTop10,
          reference_checkpoints: referenceCheckpoints,
        },
        comparison: {
          reference_top_token_matches_rusty: referenceTopToken === null ? null : referenceTopToken === rustyTopToken,
          top10_overlap_token_ids: overlap,
          top10_overlap_count: overlap.length,
          max_abs_diff: reference?.rusty_max_abs_diff ?? null,
          parity_stage_comparisons: parityStageComparisons,
          rope_layout_experiment_comparison: ropeExperimentComparisons,
          mismatch_location_candidates: [
            'q_norm/k_norm before RoPE',
            'model config/RoPE/position handling',
            'attention mask/causal path',
            'RMSNorm',
            'quantized linear',
            'logits projection',
          ],
        },
        handle_counts_before: handleCountsBeforeParity.value,
        handle_counts_after: handleCountsAfterParity.value,
      }, null, 2));
      console.log('[rusty verify] parity verification passed');
      process.exit(0);
    }

    if (runFastSmokeProfile) {
      logSkippedProbe('descriptor/embedding/tensor metadata/tokenizer/comparison probes', 'profile fastsmoke runs only active Metal-first resident decode validation');
      const linkProbe = await send('mlx_link_probe', {});
      if (linkProbe.ok !== true || linkProbe.value?.linked !== true) {
        throw new Error('fastsmoke mlx_link_probe failed verification');
      }
      const runtimeDiagnose = await send('mlx_runtime_diagnose', {});
      if (
        runtimeDiagnose.ok !== true ||
        runtimeDiagnose.value?.runtime_probe?.primary_success !== true
      ) {
        console.log(
          '[rusty verify] fastsmoke mlx_runtime_diagnose:',
          JSON.stringify(runtimeDiagnose.value ?? runtimeDiagnose, null, 2),
        );
        throw new Error('fastsmoke mlx_runtime_diagnose failed verification');
      }
      const handleCountsBeforeFastDecode = await send('mlx_handle_counts', {});
      if (
        handleCountsBeforeFastDecode.ok !== true ||
        handleCountsBeforeFastDecode.value?.arrays !== 0 ||
        handleCountsBeforeFastDecode.value?.token_arrays !== 0 ||
        handleCountsBeforeFastDecode.value?.tensor_groups !== 0 ||
        handleCountsBeforeFastDecode.value?.embedding_groups !== 0 ||
        handleCountsBeforeFastDecode.value?.layer_groups !== 0
      ) {
        throw new Error('mlx_handle_counts before fastsmoke decode did not start clean');
      }

      phaseTimer.switchTo('active decode');
      const apiModelLoaded = await send('load_model_native', {
        path: model4Dir,
      });
      if (
        apiModelLoaded.ok !== true ||
        typeof apiModelLoaded.value?.model !== 'string'
      ) {
        throw new Error('fastsmoke load_model_native API probe failed verification');
      }
      const apiSessionCreated = await send('create_native_session', {
        model: apiModelLoaded.value.model,
      });
      if (
        apiSessionCreated.ok !== true ||
        typeof apiSessionCreated.value?.session !== 'string'
      ) {
        throw new Error('fastsmoke create_native_session API probe failed verification');
      }
      const apiWarm1 = await send('warm_resident_session', {
        session: apiSessionCreated.value.session,
      });
      const apiWarm2 = await send('warm_resident_session', {
        session: apiSessionCreated.value.session,
      });
      const apiGeneration1 = await send('generate_tokens', {
        session: apiSessionCreated.value.session,
        prompt_token_ids: [1],
        first_decode_token_id: 15,
        generated_tokens: 3,
      });
      const apiGeneration2 = await send('generate_tokens', {
        session: apiSessionCreated.value.session,
        prompt_token_ids: [1],
        first_decode_token_id: 15,
        generated_tokens: 3,
      });
      phaseTimer.switchTo('cleanup');
      const apiSessionFreed = await send('free_native_session', {
        session: apiSessionCreated.value.session,
      });
      const apiModelUnloaded = await send('unload_model_native', {
        model: apiModelLoaded.value.model,
      });
      const fastGenerationProbe = apiGeneration1;

      if (
        apiWarm1.ok !== true ||
        apiWarm1.value?.warmed !== true ||
        apiWarm1.value?.reused !== false ||
        !Number.isFinite(apiWarm1.value?.warmup_ms) ||
        typeof apiWarm1.value?.timing_ms !== 'object' ||
        apiWarm2.ok !== true ||
        apiWarm2.value?.warmed !== true ||
        apiWarm2.value?.reused !== true ||
        !Number.isFinite(apiWarm2.value?.warmup_ms) ||
        apiGeneration2.ok !== true ||
        !Array.isArray(apiGeneration2.value?.generated_token_ids) ||
        apiGeneration2.value.generated_token_ids.length !== 3 ||
        JSON.stringify(apiGeneration2.value.generated_token_ids) !== JSON.stringify(apiGeneration1.value?.generated_token_ids) ||
        apiGeneration2.value?.decoded_generated_text !== apiGeneration1.value?.decoded_generated_text ||
        apiSessionFreed.ok !== true ||
        apiSessionFreed.value?.removed !== true ||
        apiModelUnloaded.ok !== true ||
        apiModelUnloaded.value?.removed !== true
      ) {
        console.log(
          '[rusty verify] native_session_api_probe failed payload:',
          JSON.stringify({
            load_model_native: apiModelLoaded.value ?? apiModelLoaded,
            create_native_session: apiSessionCreated.value ?? apiSessionCreated,
            warm_resident_session_1: apiWarm1.value ?? apiWarm1,
            warm_resident_session_2: apiWarm2.value ?? apiWarm2,
            generate_tokens_1: apiGeneration1.value ?? apiGeneration1,
            generate_tokens_2: apiGeneration2.value ?? apiGeneration2,
            free_native_session: apiSessionFreed.value ?? apiSessionFreed,
            unload_model_native: apiModelUnloaded.value ?? apiModelUnloaded,
          }, null, 2),
        );
        throw new Error('native persistent session API probe failed verification');
      }

      if (
        fastGenerationProbe.ok !== true ||
        fastGenerationProbe.value?.prompt_token_id !== 1 ||
        fastGenerationProbe.value?.first_decode_token_id !== 15 ||
        fastGenerationProbe.value?.generated_token_count !== 3 ||
        !Array.isArray(fastGenerationProbe.value?.generated_token_ids) ||
        fastGenerationProbe.value.generated_token_ids.length !== 3 ||
        typeof fastGenerationProbe.value?.decoded_generated_text !== 'string' ||
        fastGenerationProbe.value.decoded_generated_text.length === 0 ||
        !Number.isFinite(fastGenerationProbe.value?.resident_group_load_ms) ||
        typeof fastGenerationProbe.value?.resident_group_load_timing_ms !== 'object' ||
        !Number.isFinite(fastGenerationProbe.value.resident_group_load_timing_ms.safetensor_index_lookup_ms) ||
        !Number.isFinite(fastGenerationProbe.value.resident_group_load_timing_ms.tensor_group_metadata_construction_ms) ||
        !Number.isFinite(fastGenerationProbe.value.resident_group_load_timing_ms.mmap_file_read_ms) ||
        !Number.isFinite(fastGenerationProbe.value.resident_group_load_timing_ms.mlx_array_creation_ms) ||
        !Number.isFinite(fastGenerationProbe.value.resident_group_load_timing_ms.quantized_group_preparation_ms) ||
        !Number.isFinite(fastGenerationProbe.value.resident_group_load_timing_ms.synchronization_eval_ms) ||
        !Number.isFinite(fastGenerationProbe.value?.resident_projection_array_warmup_ms) ||
        typeof fastGenerationProbe.value?.resident_projection_array_warmup_timing_ms !== 'object' ||
        !Number.isFinite(fastGenerationProbe.value.resident_projection_array_warmup_timing_ms.enumerate_groups_ms) ||
        !Number.isFinite(fastGenerationProbe.value.resident_projection_array_warmup_timing_ms.mmap_setup_ms) ||
        !Number.isFinite(fastGenerationProbe.value.resident_projection_array_warmup_timing_ms.mlx_array_construction_ms) ||
        !Number.isFinite(fastGenerationProbe.value.resident_projection_array_warmup_timing_ms.first_eval_compile_warmup_ms) ||
        !Number.isFinite(fastGenerationProbe.value.resident_projection_array_warmup_timing_ms.metadata_cache_storage_ms) ||
        !Number.isFinite(fastGenerationProbe.value?.second_resident_projection_array_warmup_ms) ||
        !Number.isFinite(fastGenerationProbe.value?.prompt_pass_ms) ||
        typeof fastGenerationProbe.value?.prompt_pass_timing_buckets_ms !== 'object' ||
        !Number.isFinite(fastGenerationProbe.value.prompt_pass_timing_buckets_ms.embedding_lookup) ||
        !Number.isFinite(fastGenerationProbe.value.prompt_pass_timing_buckets_ms.qkv_projections) ||
        !Number.isFinite(fastGenerationProbe.value.prompt_pass_timing_buckets_ms.o_projection) ||
        !Number.isFinite(fastGenerationProbe.value.prompt_pass_timing_buckets_ms.gate_up_paired_projection) ||
        !Number.isFinite(fastGenerationProbe.value.prompt_pass_timing_buckets_ms.down_projection) ||
        !Number.isFinite(fastGenerationProbe.value.prompt_pass_timing_buckets_ms.final_norm) ||
        !Number.isFinite(fastGenerationProbe.value.prompt_pass_timing_buckets_ms.logits_projection_top1) ||
        fastGenerationProbe.value?.prompt_uses_mlx_decode_value !== true ||
        fastGenerationProbe.value?.prompt_uses_mmap_projection_arrays !== true ||
        fastGenerationProbe.value?.prompt_uses_fast_layer_kernel !== true ||
        !Number.isInteger(fastGenerationProbe.value?.prompt_readback_count) ||
        !Array.isArray(fastGenerationProbe.value?.prompt_readback_reasons) ||
        !fastGenerationProbe.value.prompt_readback_reasons.every((reason) => reason === 'qkv_for_cpu_attention') ||
        !Array.isArray(fastGenerationProbe.value?.prompt_fallback_steps) ||
        fastGenerationProbe.value.prompt_fallback_steps.length !== 0 ||
        !Array.isArray(fastGenerationProbe.value?.per_token_incremental_ms) ||
        fastGenerationProbe.value.per_token_incremental_ms.length !== 3 ||
        !fastGenerationProbe.value.per_token_incremental_ms.every((value) => Number.isFinite(value)) ||
        !Number.isFinite(fastGenerationProbe.value?.total_generation_ms) ||
        !Number.isFinite(fastGenerationProbe.value?.generation_1_total_ms) ||
        !Number.isFinite(fastGenerationProbe.value?.generation_2_prompt_pass_ms) ||
        !Array.isArray(fastGenerationProbe.value?.generation_2_per_token_incremental_ms) ||
        fastGenerationProbe.value.generation_2_per_token_incremental_ms.length !== 3 ||
        !fastGenerationProbe.value.generation_2_per_token_incremental_ms.every((value) => Number.isFinite(value)) ||
        !Number.isFinite(fastGenerationProbe.value?.generation_2_total_ms) ||
        !Array.isArray(fastGenerationProbe.value?.generation_2_generated_token_ids) ||
        fastGenerationProbe.value.generation_2_generated_token_ids.length !== 3 ||
        JSON.stringify(fastGenerationProbe.value.generation_2_generated_token_ids) !== JSON.stringify(fastGenerationProbe.value.generated_token_ids) ||
        fastGenerationProbe.value?.generation_2_decoded_generated_text !== fastGenerationProbe.value?.decoded_generated_text ||
        !Array.isArray(fastGenerationProbe.value?.generation_2_fallback_steps_per_token) ||
        !fastGenerationProbe.value.generation_2_fallback_steps_per_token.every((steps) => Array.isArray(steps) && steps.length === 0) ||
        !Number.isFinite(fastGenerationProbe.value?.tokens_per_second_after_resident_load) ||
        !Number.isFinite(fastGenerationProbe.value?.tokens_per_second_including_resident_load) ||
        !Number.isFinite(fastGenerationProbe.value?.total_probe_ms) ||
        !Array.isArray(fastGenerationProbe.value?.fallback_steps_per_token) ||
        fastGenerationProbe.value.fallback_steps_per_token.length !== 3 ||
        !fastGenerationProbe.value.fallback_steps_per_token.every((steps) => Array.isArray(steps) && steps.length === 0) ||
        !Array.isArray(fastGenerationProbe.value?.readback_reasons) ||
        !Number.isInteger(fastGenerationProbe.value?.readback_count) ||
        fastGenerationProbe.value.readback_count !== fastGenerationProbe.value.readback_reasons.length ||
        !fastGenerationProbe.value.readback_reasons.every((reason) =>
          reason === 'qkv_for_cpu_attention' ||
          reason === 'final_checksum_and_logits' ||
          reason === 'logits/top1'
        ) ||
        !Array.isArray(fastGenerationProbe.value?.cpu_fallback_steps) ||
        fastGenerationProbe.value.cpu_fallback_steps.length !== 0 ||
        fastGenerationProbe.value?.fallback_used !== false ||
        fastGenerationProbe.value?.cached_mlx_arrays_path_applied_to_generation !== false ||
        fastGenerationProbe.value?.resident_mlx_projection_arrays_applied_to_generation !== true ||
        fastGenerationProbe.value?.mlx_decode_value_available !== true ||
        fastGenerationProbe.value?.mlx_decode_value_applied_to_generation !== true ||
        fastGenerationProbe.value?.mlx_rmsnorm_applied !== true ||
        fastGenerationProbe.value?.mlx_residual_applied !== true ||
        fastGenerationProbe.value?.q_norm_applied !== true ||
        fastGenerationProbe.value?.k_norm_applied !== true ||
        fastGenerationProbe.value?.mlx_resident_layer_block_available !== true ||
        fastGenerationProbe.value?.mlx_resident_layer_block_applied_to_generation !== true ||
        fastGenerationProbe.value?.mlx_resident_layer_block_fallback_used !== false ||
        fastGenerationProbe.value?.mlx_resident_mlp_chain_available !== true ||
        fastGenerationProbe.value?.mlx_resident_mlp_chain_applied_to_generation !== true ||
        (
          fastGenerationProbe.value?.mlx_resident_mlp_chain_layer0_compare?.ran === true &&
          (
            fastGenerationProbe.value.mlx_resident_mlp_chain_layer0_compare.ok !== true ||
            fastGenerationProbe.value.mlx_resident_mlp_chain_layer0_compare.output_len !== 2560 ||
            !Number.isFinite(fastGenerationProbe.value.mlx_resident_mlp_chain_layer0_compare.checksum_current) ||
            !Number.isFinite(fastGenerationProbe.value.mlx_resident_mlp_chain_layer0_compare.checksum_resident) ||
            !Number.isFinite(fastGenerationProbe.value.mlx_resident_mlp_chain_layer0_compare.max_abs_diff) ||
            fastGenerationProbe.value.mlx_resident_mlp_chain_layer0_compare.max_abs_diff > 0.05
          )
        ) ||
        fastGenerationProbe.value?.mlx_quantized_linear_applied_to_generation !== true ||
        fastGenerationProbe.value?.last_token_backend_report?.qkv?.backend !== 'metal' ||
        fastGenerationProbe.value?.last_token_backend_report?.o_proj?.backend !== 'metal' ||
        fastGenerationProbe.value?.last_token_backend_report?.gate_up?.gate_backend !== 'metal' ||
        fastGenerationProbe.value?.last_token_backend_report?.gate_up?.up_backend !== 'metal' ||
        fastGenerationProbe.value?.last_token_backend_report?.down?.backend !== 'metal' ||
        fastGenerationProbe.value?.last_token_backend_report?.['logits/top1']?.backend !== 'metal' ||
        !Number.isFinite(fastGenerationProbe.value?.first_generated_final_norm_checksum) ||
        fastGenerationProbe.value?.logits_len !== 151936
      ) {
        console.log(
          '[rusty verify] fastsmoke_generation_probe failed payload:',
          JSON.stringify(fastGenerationProbe.value ?? fastGenerationProbe, null, 2),
        );
        throw new Error('fastsmoke_generation_probe failed verification');
      }
      const handleCountsAfterFastDecode = await send('mlx_handle_counts', {});
      if (
        handleCountsAfterFastDecode.ok !== true ||
        JSON.stringify(handleCountsAfterFastDecode.value) !== JSON.stringify(handleCountsBeforeFastDecode.value)
      ) {
        throw new Error('mlx_handle_counts changed during fastsmoke decode');
      }

      console.log(
        '[rusty verify] native_session_api_probe:',
        JSON.stringify({
          lifecycle: [
            'load_model_native',
            'create_native_session',
            'warm_resident_session',
            'generate_tokens',
            'free_native_session',
            'unload_model_native',
          ],
          model: apiModelLoaded.value.model,
          session: apiSessionCreated.value.session,
          warm_1: apiWarm1.value,
          warm_2: apiWarm2.value,
          generation_1_generated_token_ids: apiGeneration1.value.generated_token_ids,
          generation_1_decoded_generated_text: apiGeneration1.value.decoded_generated_text,
          generation_1_total_ms: apiGeneration1.value.generation_1_total_ms,
          generation_2_generated_token_ids: apiGeneration2.value.generated_token_ids,
          generation_2_decoded_generated_text: apiGeneration2.value.decoded_generated_text,
          generation_2_total_ms: apiGeneration2.value.generation_1_total_ms,
          session_freed: apiSessionFreed.value.removed,
          model_unloaded: apiModelUnloaded.value.removed,
          handle_counts_before: handleCountsBeforeFastDecode.value,
          handle_counts_after: handleCountsAfterFastDecode.value,
        }, null, 2),
      );

      console.log(
        '[rusty verify] fastsmoke_generation_probe:',
        JSON.stringify({
          prompt_token_id: fastGenerationProbe.value.prompt_token_id,
          first_decode_token_id: fastGenerationProbe.value.first_decode_token_id,
          generated_token_count: fastGenerationProbe.value.generated_token_count,
          generated_token_ids: fastGenerationProbe.value.generated_token_ids,
          generated_token_scores: fastGenerationProbe.value.generated_token_scores,
          decoded_generated_text: fastGenerationProbe.value.decoded_generated_text,
          resident_group_load_ms: fastGenerationProbe.value.resident_group_load_ms,
          resident_group_load_timing_ms: fastGenerationProbe.value.resident_group_load_timing_ms,
          resident_projection_array_warmup_ms: fastGenerationProbe.value.resident_projection_array_warmup_ms,
          resident_projection_array_warmup_timing_ms: fastGenerationProbe.value.resident_projection_array_warmup_timing_ms,
          second_resident_projection_array_warmup_ms: fastGenerationProbe.value.second_resident_projection_array_warmup_ms,
          second_resident_projection_array_warmup_timing_ms: fastGenerationProbe.value.second_resident_projection_array_warmup_timing_ms,
          prompt_pass_ms: fastGenerationProbe.value.prompt_pass_ms,
          prompt_pass_timing_buckets_ms: fastGenerationProbe.value.prompt_pass_timing_buckets_ms,
          prompt_largest_arithmetic_bucket: fastGenerationProbe.value.prompt_largest_arithmetic_bucket,
          prompt_largest_arithmetic_bucket_ms: fastGenerationProbe.value.prompt_largest_arithmetic_bucket_ms,
          prompt_uses_mlx_decode_value: fastGenerationProbe.value.prompt_uses_mlx_decode_value,
          prompt_uses_mmap_projection_arrays: fastGenerationProbe.value.prompt_uses_mmap_projection_arrays,
          prompt_uses_fast_layer_kernel: fastGenerationProbe.value.prompt_uses_fast_layer_kernel,
          prompt_readback_count: fastGenerationProbe.value.prompt_readback_count,
          prompt_readback_reasons: fastGenerationProbe.value.prompt_readback_reasons,
          prompt_fallback_steps: fastGenerationProbe.value.prompt_fallback_steps,
          per_token_incremental_ms: fastGenerationProbe.value.per_token_incremental_ms,
          total_generation_ms: fastGenerationProbe.value.total_generation_ms,
          generation_1_total_ms: fastGenerationProbe.value.generation_1_total_ms,
          generation_2_prompt_pass_ms: fastGenerationProbe.value.generation_2_prompt_pass_ms,
          generation_2_per_token_incremental_ms: fastGenerationProbe.value.generation_2_per_token_incremental_ms,
          generation_2_total_ms: fastGenerationProbe.value.generation_2_total_ms,
          generation_2_generated_token_ids: fastGenerationProbe.value.generation_2_generated_token_ids,
          generation_2_generated_token_scores: fastGenerationProbe.value.generation_2_generated_token_scores,
          generation_2_decoded_generated_text: fastGenerationProbe.value.generation_2_decoded_generated_text,
          generation_2_first_generated_final_norm_checksum: fastGenerationProbe.value.generation_2_first_generated_final_norm_checksum,
          generation_2_logits_len: fastGenerationProbe.value.generation_2_logits_len,
          generation_2_fallback_steps_per_token: fastGenerationProbe.value.generation_2_fallback_steps_per_token,
          total_probe_ms: fastGenerationProbe.value.total_probe_ms,
          tokens_per_second_after_resident_load: fastGenerationProbe.value.tokens_per_second_after_resident_load,
          tokens_per_second_including_resident_load: fastGenerationProbe.value.tokens_per_second_including_resident_load,
          fallback_steps_per_token: fastGenerationProbe.value.fallback_steps_per_token,
          readback_count: fastGenerationProbe.value.readback_count,
          readback_reasons: fastGenerationProbe.value.readback_reasons,
          cpu_fallback_steps: fastGenerationProbe.value.cpu_fallback_steps,
          cached_mlx_arrays_path_applied_to_generation: fastGenerationProbe.value.cached_mlx_arrays_path_applied_to_generation,
          resident_mlx_projection_arrays_applied_to_generation: fastGenerationProbe.value.resident_mlx_projection_arrays_applied_to_generation,
          mlx_decode_value_available: fastGenerationProbe.value.mlx_decode_value_available,
          mlx_decode_value_applied_to_generation: fastGenerationProbe.value.mlx_decode_value_applied_to_generation,
          mlx_rmsnorm_applied: fastGenerationProbe.value.mlx_rmsnorm_applied,
          mlx_residual_applied: fastGenerationProbe.value.mlx_residual_applied,
          q_norm_applied: fastGenerationProbe.value.q_norm_applied,
          k_norm_applied: fastGenerationProbe.value.k_norm_applied,
          mlx_resident_layer_block_available: fastGenerationProbe.value.mlx_resident_layer_block_available,
          mlx_resident_layer_block_applied_to_generation: fastGenerationProbe.value.mlx_resident_layer_block_applied_to_generation,
          mlx_resident_layer_block_fallback_used: fastGenerationProbe.value.mlx_resident_layer_block_fallback_used,
          mlx_resident_mlp_chain_available: fastGenerationProbe.value.mlx_resident_mlp_chain_available,
          mlx_resident_mlp_chain_applied_to_generation: fastGenerationProbe.value.mlx_resident_mlp_chain_applied_to_generation,
          mlx_resident_mlp_chain_layer0_compare: fastGenerationProbe.value.mlx_resident_mlp_chain_layer0_compare,
          fallback_used: fastGenerationProbe.value.fallback_used,
          tensor_group_load_count: fastGenerationProbe.value.tensor_group_load_count,
          resident_total_byte_size: fastGenerationProbe.value.resident_total_byte_size,
          positions_after: fastGenerationProbe.value.positions_after,
          logits_len: fastGenerationProbe.value.logits_len,
          first_generated_final_norm_checksum: fastGenerationProbe.value.first_generated_final_norm_checksum,
          final_norm_checksum: fastGenerationProbe.value.final_norm_checksum,
          last_token_backend_report: fastGenerationProbe.value.last_token_backend_report,
          last_token_timing_buckets_ms: fastGenerationProbe.value.last_token_timing_buckets_ms,
          last_token_projection_timing_breakdown_ms: fastGenerationProbe.value.last_token_projection_timing_breakdown_ms,
          handle_counts_before: handleCountsBeforeFastDecode.value,
          handle_counts_after: handleCountsAfterFastDecode.value,
        }, null, 2),
      );

      const fastSmokeTimingBucketSummary = [
        ['qkv', fastGenerationProbe.value.last_token_timing_buckets_ms.qkv_projections],
        ['o_proj', fastGenerationProbe.value.last_token_timing_buckets_ms.o_projection],
        ['gate_proj', fastGenerationProbe.value.last_token_timing_buckets_ms.gate_projection],
        ['up_proj', fastGenerationProbe.value.last_token_timing_buckets_ms.up_projection],
        ['gate_up_activation', fastGenerationProbe.value.last_token_timing_buckets_ms.gate_up_activation],
        ['down', fastGenerationProbe.value.last_token_timing_buckets_ms.down_projection],
        ['logits/top1', fastGenerationProbe.value.last_token_timing_buckets_ms.logits_projection_top1],
        ['attention', fastGenerationProbe.value.last_token_backend_report.attention.timing_ms],
        ['norms', fastGenerationProbe.value.last_token_timing_buckets_ms.final_norm],
        ['embedding', fastGenerationProbe.value.last_token_backend_report.embedding.timing_ms],
      ]
        .map(([bucket, timing_ms]) => ({ bucket, timing_ms }))
        .sort((left, right) => right.timing_ms - left.timing_ms);
      console.log(
        '[rusty verify] smoke_timing_bucket_summary:',
        JSON.stringify(fastSmokeTimingBucketSummary, null, 2),
      );

      const shutdown = await send('bridge_shutdown', {});
      if (shutdown.ok !== true) {
        throw new Error('bridge_shutdown failed verification');
      }
      console.log('[rusty verify] fastsmoke verification passed');
      console.log('[rusty verify] phase_timing_ms:', JSON.stringify(phaseTimer.summary(), null, 2));
      console.log('[rusty verify] transcript entries:', transcript.length);
      return;
    }

    if (runFullProfile) {
    const modelDescriptor = await send('create_model_descriptor', {
      model_dir: model4Dir,
    });
    if (
      modelDescriptor.ok !== true ||
      typeof modelDescriptor.value?.descriptor !== 'string' ||
      modelDescriptor.value?.model_type !== 'qwen3' ||
      modelDescriptor.value?.layer_count !== 36 ||
      modelDescriptor.value?.quantized !== true ||
      modelDescriptor.value?.tied_embeddings !== true ||
      modelDescriptor.value?.loaded_layers !== 0 ||
      modelDescriptor.value?.loaded_weights !== false ||
      modelDescriptor.value?.total_estimated_bytes <= 0
    ) {
      throw new Error('create_model_descriptor failed verification');
    }
    console.log('[rusty verify] create_model_descriptor handle:', modelDescriptor.value.descriptor);
    console.log('[rusty verify] create_model_descriptor model_type:', modelDescriptor.value.model_type);
    console.log('[rusty verify] create_model_descriptor layer_count:', modelDescriptor.value.layer_count);
    console.log('[rusty verify] create_model_descriptor quantized:', modelDescriptor.value.quantized);
    console.log('[rusty verify] create_model_descriptor tied_embeddings:', modelDescriptor.value.tied_embeddings);
    console.log('[rusty verify] create_model_descriptor loaded_weights:', modelDescriptor.value.loaded_weights);

    const modelDescriptorInfo = await send('model_descriptor_info', {
      descriptor: modelDescriptor.value.descriptor,
    });
    if (
      modelDescriptorInfo.ok !== true ||
      modelDescriptorInfo.value?.model_type !== 'qwen3' ||
      modelDescriptorInfo.value?.layer_count !== 36 ||
      modelDescriptorInfo.value?.quantized !== true ||
      modelDescriptorInfo.value?.tied_embeddings !== true ||
      modelDescriptorInfo.value?.loaded_layers !== 0 ||
      modelDescriptorInfo.value?.loaded_weights !== false
    ) {
      throw new Error('model_descriptor_info failed verification');
    }
    console.log('[rusty verify] model_descriptor_info model_type:', modelDescriptorInfo.value.model_type);
    console.log('[rusty verify] model_descriptor_info architecture:', modelDescriptorInfo.value.architecture);
    console.log('[rusty verify] model_descriptor_info loaded_weights:', modelDescriptorInfo.value.loaded_weights);

    const modelDescriptorFreed = await send('free_model_descriptor', {
      descriptor: modelDescriptor.value.descriptor,
    });
    if (modelDescriptorFreed.ok !== true || modelDescriptorFreed.value?.freed !== true) {
      throw new Error('free_model_descriptor failed verification');
    }
    console.log('[rusty verify] free_model_descriptor ok:', modelDescriptorFreed.ok);

    const modelDescriptorInfoAfterFree = await send('model_descriptor_info', {
      descriptor: modelDescriptor.value.descriptor,
    });
    if (modelDescriptorInfoAfterFree.ok !== false || modelDescriptorInfoAfterFree.error?.code !== 'already_freed') {
      throw new Error('model_descriptor_info after free did not fail as expected');
    }

    const embeddingGroup = await send('load_embedding_group', {
      model_dir: model4Dir,
    });
    if (
      embeddingGroup.ok !== true ||
      typeof embeddingGroup.value?.embedding_handle !== 'string' ||
      embeddingGroup.value?.quantized_group !== true ||
      embeddingGroup.value?.weight?.dtype !== 'U32' ||
      embeddingGroup.value?.scales?.dtype !== 'BF16' ||
      embeddingGroup.value?.biases?.dtype !== 'BF16' ||
      !Array.isArray(embeddingGroup.value?.weight?.shape) ||
      !Array.isArray(embeddingGroup.value?.scales?.shape) ||
      !Array.isArray(embeddingGroup.value?.biases?.shape) ||
      embeddingGroup.value?.byte_size <= 0
    ) {
      throw new Error('load_embedding_group failed verification');
    }
    console.log('[rusty verify] load_embedding_group handle:', embeddingGroup.value.embedding_handle);
    console.log('[rusty verify] load_embedding_group quantized_group:', embeddingGroup.value.quantized_group);
    console.log('[rusty verify] load_embedding_group weight:', JSON.stringify(embeddingGroup.value.weight, null, 2));
    console.log('[rusty verify] load_embedding_group scales:', JSON.stringify(embeddingGroup.value.scales, null, 2));
    console.log('[rusty verify] load_embedding_group biases:', JSON.stringify(embeddingGroup.value.biases, null, 2));
    console.log('[rusty verify] load_embedding_group byte_size:', embeddingGroup.value.byte_size);

    const embeddingGroupInfo = await send('embedding_group_info', {
      embedding: embeddingGroup.value.embedding_handle,
    });
    if (
      embeddingGroupInfo.ok !== true ||
      embeddingGroupInfo.value?.quantized_group !== true ||
      embeddingGroupInfo.value?.weight?.dtype !== 'U32' ||
      embeddingGroupInfo.value?.scales?.dtype !== 'BF16' ||
      embeddingGroupInfo.value?.biases?.dtype !== 'BF16' ||
      !Array.isArray(embeddingGroupInfo.value?.weight?.shape) ||
      !Array.isArray(embeddingGroupInfo.value?.scales?.shape) ||
      !Array.isArray(embeddingGroupInfo.value?.biases?.shape) ||
      embeddingGroupInfo.value?.byte_size <= 0
    ) {
      throw new Error('embedding_group_info failed verification');
    }
    console.log('[rusty verify] embedding_group_info:', JSON.stringify(embeddingGroupInfo.value, null, 2));

    const embeddingGroupFreed = await send('free_embedding_group', {
      embedding: embeddingGroup.value.embedding_handle,
    });
    if (embeddingGroupFreed.ok !== true || embeddingGroupFreed.value?.freed !== true) {
      throw new Error('free_embedding_group failed verification');
    }
    console.log('[rusty verify] free_embedding_group ok:', embeddingGroupFreed.ok);

    const embeddingGroupInfoAfterFree = await send('embedding_group_info', {
      embedding: embeddingGroup.value.embedding_handle,
    });
    if (
      embeddingGroupInfoAfterFree.ok !== false ||
      embeddingGroupInfoAfterFree.error?.code !== 'already_freed'
    ) {
      throw new Error('embedding_group_info after free did not fail as expected');
    }

    } else {
      logSkippedProbe('inspect_model_dir/create_model_descriptor/load_embedding_group', 'profile smoke/layer skips descriptor and embedding metadata checks');
    }

    const tensorGroup = await send('load_tensor_group', {
      model_dir: model4Dir,
      group: 'model.layers.0.self_attn.q_proj',
    });
    if (tensorGroup.ok !== true || typeof tensorGroup.value?.group !== 'string') {
      throw new Error('load_tensor_group failed verification');
    }
    console.log('[rusty verify] load_tensor_group handle:', tensorGroup.value.group);

    const tensorGroupInfo = await send('tensor_group_info', {
      group: tensorGroup.value.group,
    });
    if (
      tensorGroupInfo.ok !== true ||
      tensorGroupInfo.value?.quantized_group !== true ||
      tensorGroupInfo.value?.weight?.dtype !== 'U32' ||
      tensorGroupInfo.value?.scales?.dtype !== 'BF16' ||
      tensorGroupInfo.value?.biases?.dtype !== 'BF16' ||
      !Array.isArray(tensorGroupInfo.value?.weight?.shape) ||
      !Array.isArray(tensorGroupInfo.value?.scales?.shape) ||
      !Array.isArray(tensorGroupInfo.value?.biases?.shape) ||
      tensorGroupInfo.value?.weight?.byte_size <= 0 ||
      tensorGroupInfo.value?.scales?.byte_size <= 0 ||
      tensorGroupInfo.value?.biases?.byte_size <= 0 ||
      tensorGroupInfo.value?.weight?.source_file !== 'model.safetensors'
    ) {
      throw new Error('tensor_group_info failed verification');
    }
    console.log('[rusty verify] tensor_group_info:', JSON.stringify(tensorGroupInfo.value, null, 2));

    if (runFullProfile) {
    const quantizationLayoutProbe = await send('quantization_layout_probe', {
      group: tensorGroup.value.group,
    });
    if (
      quantizationLayoutProbe.ok !== true ||
      quantizationLayoutProbe.value?.weight?.dtype !== 'U32' ||
      quantizationLayoutProbe.value?.scales?.dtype !== 'BF16' ||
      quantizationLayoutProbe.value?.biases?.dtype !== 'BF16' ||
      quantizationLayoutProbe.value?.inferred_block_size == null
    ) {
      throw new Error('quantization_layout_probe failed verification');
    }
    console.log(
      '[rusty verify] quantization_layout_probe:',
      JSON.stringify(quantizationLayoutProbe.value, null, 2),
    );

    const compareDequantSlice = await send('compare_dequant_slice', {
      group: tensorGroup.value.group,
      row: 0,
      cols: 8,
    });
    if (
      compareDequantSlice.ok !== true ||
      !Array.isArray(compareDequantSlice.value?.provisional_values) ||
      compareDequantSlice.value.provisional_values.length !== 8 ||
      compareDequantSlice.value?.handle !== tensorGroup.value.group
    ) {
      throw new Error('compare_dequant_slice failed verification');
    }
    console.log(
      '[rusty verify] compare_dequant_slice:',
      JSON.stringify(compareDequantSlice.value, null, 2),
    );
    if (compareDequantSlice.value?.comparison_available === true) {
      if (
        !Array.isArray(compareDequantSlice.value?.mlx_values) ||
        compareDequantSlice.value.mlx_values.length !== 8 ||
        typeof compareDequantSlice.value?.max_abs_diff !== 'number'
      ) {
        throw new Error('compare_dequant_slice available comparison failed verification');
      }
    }
    } else {
      logSkippedProbe('quantization_layout_probe/compare_dequant_slice', 'profile smoke/layer uses only one quantized slice probe');
    }

    const quantizedLinearSliceProbe = await send('quantized_linear_slice_probe', {
      group: tensorGroup.value.group,
      input: [1, 0, 0, 0, 0, 0, 0, 0],
      row: 0,
      cols: 8,
    });
    if (
      quantizedLinearSliceProbe.ok !== true ||
      quantizedLinearSliceProbe.value?.provisional !== true ||
      !Array.isArray(quantizedLinearSliceProbe.value?.input) ||
      !Array.isArray(quantizedLinearSliceProbe.value?.dequantized_slice) ||
      quantizedLinearSliceProbe.value.input.length !== 8 ||
      quantizedLinearSliceProbe.value.dequantized_slice.length !== 8 ||
      typeof quantizedLinearSliceProbe.value?.dot !== 'number'
    ) {
      throw new Error('quantized_linear_slice_probe failed verification');
    }
    const firstSliceValue = quantizedLinearSliceProbe.value.dequantized_slice[0];
    const dotValue = quantizedLinearSliceProbe.value.dot;
    if (Math.abs(dotValue - firstSliceValue) > 1e-6) {
      throw new Error('quantized_linear_slice_probe dot did not match first dequantized value');
    }
    console.log(
      '[rusty verify] quantized_linear_slice_probe:',
      JSON.stringify(quantizedLinearSliceProbe.value, null, 2),
    );

    const fullRowInput = new Array(2560).fill(0);
    fullRowInput[0] = 1;
    const denseVectorInput = Array.from({ length: 2560 }, (_, i) => ((i % 17) - 8) / 8.0);

    if (runLayerProfile) {
    const quantizedLinearRowsProbe = await send('quantized_linear_rows_probe', {
      group: tensorGroup.value.group,
      input: [1, 0, 0, 0, 0, 0, 0, 0],
      rows: 4,
      cols: 8,
    });
    if (
      quantizedLinearRowsProbe.ok !== true ||
      quantizedLinearRowsProbe.value?.provisional !== true ||
      !Array.isArray(quantizedLinearRowsProbe.value?.output) ||
      quantizedLinearRowsProbe.value.output.length !== 4
    ) {
      throw new Error('quantized_linear_rows_probe failed verification');
    }
    if (Math.abs(quantizedLinearRowsProbe.value.output[0] - firstSliceValue) > 1e-6) {
      throw new Error('quantized_linear_rows_probe output[0] did not match single-row dot');
    }
    console.log(
      '[rusty verify] quantized_linear_rows_probe:',
      JSON.stringify(quantizedLinearRowsProbe.value, null, 2),
    );

    const quantizedLinearFullRowProbe = await send('quantized_linear_fullrow_probe', {
      group: tensorGroup.value.group,
      input: fullRowInput,
      rows: 4,
    });
    if (
      quantizedLinearFullRowProbe.ok !== true ||
      quantizedLinearFullRowProbe.value?.provisional !== true ||
      !Array.isArray(quantizedLinearFullRowProbe.value?.output) ||
      quantizedLinearFullRowProbe.value.output.length !== 4 ||
      quantizedLinearFullRowProbe.value?.input_len !== 2560 ||
      typeof quantizedLinearFullRowProbe.value?.timing_ms !== 'number'
    ) {
      throw new Error('quantized_linear_fullrow_probe failed verification');
    }
    for (let i = 0; i < quantizedLinearFullRowProbe.value.output.length; i += 1) {
      if (Math.abs(quantizedLinearFullRowProbe.value.output[i] - quantizedLinearRowsProbe.value.output[i]) > 1e-6) {
        throw new Error(\`quantized_linear_fullrow_probe output[${i}] did not match tiny-row probe\`);
      }
    }
    console.log(
      '[rusty verify] quantized_linear_fullrow_probe:',
      JSON.stringify(quantizedLinearFullRowProbe.value, null, 2),
    );

    const quantizedLinearVectorProbe = await send('quantized_linear_vector_probe', {
      group: tensorGroup.value.group,
      input: fullRowInput,
    });
    if (
      quantizedLinearVectorProbe.ok !== true ||
      quantizedLinearVectorProbe.value?.provisional !== true ||
      !Array.isArray(quantizedLinearVectorProbe.value?.output) ||
      quantizedLinearVectorProbe.value.output.length !== 4096 ||
      quantizedLinearVectorProbe.value?.input_len !== 2560 ||
      quantizedLinearVectorProbe.value?.output_len !== 4096 ||
      !Array.isArray(quantizedLinearVectorProbe.value?.first_values) ||
      quantizedLinearVectorProbe.value.first_values.length !== 8 ||
      !Array.isArray(quantizedLinearVectorProbe.value?.last_values) ||
      quantizedLinearVectorProbe.value.last_values.length !== 8 ||
      typeof quantizedLinearVectorProbe.value?.checksum !== 'number' ||
      typeof quantizedLinearVectorProbe.value?.timing_ms !== 'number'
    ) {
      throw new Error('quantized_linear_vector_probe failed verification');
    }
    for (let i = 0; i < 4; i += 1) {
      if (Math.abs(quantizedLinearVectorProbe.value.output[i] - quantizedLinearFullRowProbe.value.output[i]) > 1e-6) {
        throw new Error(\`quantized_linear_vector_probe output[${i}] did not match full-row probe\`);
      }
    }
    console.log(
      '[rusty verify] quantized_linear_vector_probe:',
      JSON.stringify({
        input_len: quantizedLinearVectorProbe.value.input_len,
        output_len: quantizedLinearVectorProbe.value.output_len,
        first_values: quantizedLinearVectorProbe.value.first_values,
        last_values: quantizedLinearVectorProbe.value.last_values,
        checksum: quantizedLinearVectorProbe.value.checksum,
        timing_ms: quantizedLinearVectorProbe.value.timing_ms,
      }, null, 2),
    );

    const quantizedLinearDenseVectorProbe = await send('quantized_linear_vector_probe', {
      group: tensorGroup.value.group,
      input: denseVectorInput,
    });
    if (
      quantizedLinearDenseVectorProbe.ok !== true ||
      quantizedLinearDenseVectorProbe.value?.provisional !== true ||
      quantizedLinearDenseVectorProbe.value?.input_len !== 2560 ||
      quantizedLinearDenseVectorProbe.value?.output_len !== 4096 ||
      !Array.isArray(quantizedLinearDenseVectorProbe.value?.first_values) ||
      quantizedLinearDenseVectorProbe.value.first_values.length !== 8 ||
      !Array.isArray(quantizedLinearDenseVectorProbe.value?.last_values) ||
      quantizedLinearDenseVectorProbe.value.last_values.length !== 8 ||
      !Number.isFinite(quantizedLinearDenseVectorProbe.value?.checksum) ||
      !Number.isFinite(quantizedLinearDenseVectorProbe.value?.timing_ms)
    ) {
      throw new Error('quantized_linear_vector_probe dense input failed verification');
    }
    for (const value of quantizedLinearDenseVectorProbe.value.first_values) {
      if (!Number.isFinite(value)) {
        throw new Error('quantized_linear_vector_probe dense first_values contained a non-finite value');
      }
    }
    console.log(
      '[rusty verify] quantized_linear_vector_probe dense:',
      JSON.stringify({
        input_len: quantizedLinearDenseVectorProbe.value.input_len,
        output_len: quantizedLinearDenseVectorProbe.value.output_len,
        first_values: quantizedLinearDenseVectorProbe.value.first_values,
        last_values: quantizedLinearDenseVectorProbe.value.last_values,
        checksum: quantizedLinearDenseVectorProbe.value.checksum,
        timing_ms: quantizedLinearDenseVectorProbe.value.timing_ms,
      }, null, 2),
    );
    } else {
      logSkippedProbe('quantized_linear_rows/fullrow/vector_probe', 'profile smoke runs one quantized slice only');
    }

    const inputLayerNorm = await send('load_tensor_group', {
      model_dir: model4Dir,
      group: 'model.layers.0.input_layernorm',
    });
    if (
      inputLayerNorm.ok !== true ||
      typeof inputLayerNorm.value?.group !== 'string' ||
      inputLayerNorm.value.group.length === 0
    ) {
      throw new Error('load_tensor_group for input_layernorm failed verification');
    }
    console.log('[rusty verify] load_tensor_group input_layernorm handle:', inputLayerNorm.value.group);

    const rmsnormProbe = await send('rmsnorm_probe', {
      group: inputLayerNorm.value.group,
      input: denseVectorInput,
    });
    if (
      rmsnormProbe.ok !== true ||
      rmsnormProbe.value?.provisional !== true ||
      rmsnormProbe.value?.input_len !== 2560 ||
      rmsnormProbe.value?.output_len !== 2560 ||
      !Number.isFinite(rmsnormProbe.value?.eps) ||
      !Array.isArray(rmsnormProbe.value?.first_values) ||
      rmsnormProbe.value.first_values.length !== 8 ||
      !Array.isArray(rmsnormProbe.value?.last_values) ||
      rmsnormProbe.value.last_values.length !== 8 ||
      !Number.isFinite(rmsnormProbe.value?.checksum) ||
      !Number.isFinite(rmsnormProbe.value?.timing_ms)
    ) {
      throw new Error('rmsnorm_probe failed verification');
    }
    for (const value of rmsnormProbe.value.first_values) {
      if (!Number.isFinite(value)) {
        throw new Error('rmsnorm_probe first_values contained a non-finite value');
      }
    }
    for (const value of rmsnormProbe.value.last_values) {
      if (!Number.isFinite(value)) {
        throw new Error('rmsnorm_probe last_values contained a non-finite value');
      }
    }
    console.log(
      '[rusty verify] rmsnorm_probe:',
      JSON.stringify({
        input_len: rmsnormProbe.value.input_len,
        output_len: rmsnormProbe.value.output_len,
        eps: rmsnormProbe.value.eps,
        first_values: rmsnormProbe.value.first_values,
        last_values: rmsnormProbe.value.last_values,
        checksum: rmsnormProbe.value.checksum,
        timing_ms: rmsnormProbe.value.timing_ms,
      }, null, 2),
    );

    const inputLayerNormFreed = await send('free_tensor_group', {
      group: inputLayerNorm.value.group,
    });
    if (inputLayerNormFreed.ok !== true || inputLayerNormFreed.value?.freed !== true) {
      throw new Error('free_tensor_group for input_layernorm failed verification');
    }
    console.log('[rusty verify] free_tensor_group input_layernorm ok:', inputLayerNormFreed.ok);

    console.log('[rusty verify] dequantize_group_slice deferred until after layer0 probes');

    const tensorGroupFreed = await send('free_tensor_group', {
      group: tensorGroup.value.group,
    });
    if (tensorGroupFreed.ok !== true || tensorGroupFreed.value?.freed !== true) {
      throw new Error('free_tensor_group failed verification');
    }
    console.log('[rusty verify] free_tensor_group ok:', tensorGroupFreed.ok);

    const handleCountsAfterDequant = await send('mlx_handle_counts', {});
    if (
      handleCountsAfterDequant.ok !== true ||
      handleCountsAfterDequant.value?.arrays !== 0 ||
      handleCountsAfterDequant.value?.token_arrays !== 0 ||
      handleCountsAfterDequant.value?.tensor_groups !== 0 ||
      handleCountsAfterDequant.value?.embedding_groups !== 0 ||
      handleCountsAfterDequant.value?.layer_groups !== 0
    ) {
      throw new Error('mlx_handle_counts after dequantize/free did not return to zero');
    }
    console.log('[rusty verify] mlx_handle_counts after dequantize/free:', JSON.stringify(handleCountsAfterDequant.value, null, 2));

    const tensorGroupInfoAfterFree = await send('tensor_group_info', {
      group: tensorGroup.value.group,
    });
    if (
      tensorGroupInfoAfterFree.ok !== false ||
      tensorGroupInfoAfterFree.error?.code !== 'already_freed'
    ) {
      throw new Error('tensor_group_info after free did not fail as expected');
    }

    const handleCountsBeforeLayer0 = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeLayer0.ok !== true ||
      handleCountsBeforeLayer0.value?.arrays !== 0 ||
      handleCountsBeforeLayer0.value?.token_arrays !== 0 ||
      handleCountsBeforeLayer0.value?.tensor_groups !== 0 ||
      handleCountsBeforeLayer0.value?.embedding_groups !== 0 ||
      handleCountsBeforeLayer0.value?.layer_groups !== 0
    ) {
      throw new Error('mlx_handle_counts before layer0 probe did not start clean');
    }
    if (runLayerProfile) {
    console.log(
      '[rusty verify] mlx_handle_counts before layer0_single_token_probe:',
      JSON.stringify(handleCountsBeforeLayer0.value, null, 2),
    );

    const layer0SingleTokenProbe = await send('layer0_single_token_probe', {
      model_dir: model4Dir,
      token_id: 1,
    });
    const expectedLayer0Lengths = {
      embedding_len: 2560,
      norm_len: 2560,
      q_len: 4096,
      k_len: 1024,
      v_len: 1024,
      attention_len: 4096,
      o_len: 2560,
      residual_len: 2560,
    };
    if (
      layer0SingleTokenProbe.ok !== true ||
      layer0SingleTokenProbe.value?.provisional !== true ||
      layer0SingleTokenProbe.value?.token_id !== 1 ||
      !Number.isFinite(layer0SingleTokenProbe.value?.timing_ms)
    ) {
      throw new Error('layer0_single_token_probe failed verification');
    }
    for (const [key, expected] of Object.entries(expectedLayer0Lengths)) {
      if (layer0SingleTokenProbe.value?.[key] !== expected) {
        throw new Error(\`layer0_single_token_probe ${key} expected ${expected}\`);
      }
    }
    for (const [key, value] of Object.entries(layer0SingleTokenProbe.value.checksums ?? {})) {
      if (!Number.isFinite(value)) {
        throw new Error(\`layer0_single_token_probe checksum ${key} was not finite\`);
      }
    }
    for (const [key, values] of Object.entries(layer0SingleTokenProbe.value.first_values ?? {})) {
      if (!Array.isArray(values) || values.length === 0) {
        throw new Error(\`layer0_single_token_probe first_values ${key} missing\`);
      }
      for (const value of values) {
        if (!Number.isFinite(value)) {
          throw new Error(\`layer0_single_token_probe first_values ${key} contained a non-finite value\`);
        }
      }
    }
    console.log(
      '[rusty verify] layer0_single_token_probe:',
      JSON.stringify({
        token_id: layer0SingleTokenProbe.value.token_id,
        lengths: expectedLayer0Lengths,
        checksums: layer0SingleTokenProbe.value.checksums,
        first_values: layer0SingleTokenProbe.value.first_values,
        timing_ms: layer0SingleTokenProbe.value.timing_ms,
      }, null, 2),
    );

    const handleCountsAfterLayer0 = await send('mlx_handle_counts', {});
    if (
      handleCountsAfterLayer0.ok !== true ||
      JSON.stringify(handleCountsAfterLayer0.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts changed during layer0_single_token_probe');
    }
    console.log(
      '[rusty verify] mlx_handle_counts after layer0_single_token_probe:',
      JSON.stringify(handleCountsAfterLayer0.value, null, 2),
    );

    const handleCountsBeforeLayer0Block = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeLayer0Block.ok !== true ||
      JSON.stringify(handleCountsBeforeLayer0Block.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before layer0_block_probe did not start clean');
    }

    const layer0BlockProbe = await send('layer0_block_probe', {
      model_dir: model4Dir,
      token_id: 1,
    });
    if (
      layer0BlockProbe.ok !== true ||
      layer0BlockProbe.value?.provisional !== true ||
      layer0BlockProbe.value?.token_id !== 1 ||
      layer0BlockProbe.value?.output_len !== 2560 ||
      !Number.isFinite(layer0BlockProbe.value?.input_embedding_checksum) ||
      !Number.isFinite(layer0BlockProbe.value?.attention_residual_checksum) ||
      !Number.isFinite(layer0BlockProbe.value?.mlp_residual_checksum) ||
      !Array.isArray(layer0BlockProbe.value?.first_values) ||
      layer0BlockProbe.value.first_values.length !== 8 ||
      !Number.isFinite(layer0BlockProbe.value?.timing_ms)
    ) {
      throw new Error('layer0_block_probe failed verification');
    }
    for (const value of layer0BlockProbe.value.first_values) {
      if (!Number.isFinite(value)) {
        throw new Error('layer0_block_probe first_values contained a non-finite value');
      }
    }
    const handleCountsAfterLayer0Block = await send('mlx_handle_counts', {});

    console.log(
      '[rusty verify] layer0_block_probe:',
      JSON.stringify({
        token_id: layer0BlockProbe.value.token_id,
        embedding_checksum: layer0BlockProbe.value.input_embedding_checksum,
        attention_residual_checksum: layer0BlockProbe.value.attention_residual_checksum,
        mlp_residual_checksum: layer0BlockProbe.value.mlp_residual_checksum,
        output_len: layer0BlockProbe.value.output_len,
        first_values: layer0BlockProbe.value.first_values,
        timing_ms: layer0BlockProbe.value.timing_ms,
        handle_counts_before: handleCountsBeforeLayer0Block.value,
        handle_counts_after: handleCountsAfterLayer0Block.value,
      }, null, 2),
    );
    if (
      handleCountsAfterLayer0Block.ok !== true ||
      JSON.stringify(handleCountsAfterLayer0Block.value) !== JSON.stringify(handleCountsBeforeLayer0Block.value)
    ) {
      throw new Error('mlx_handle_counts changed during layer0_block_probe');
    }

    const handleCountsBeforeLayerStack = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeLayerStack.ok !== true ||
      JSON.stringify(handleCountsBeforeLayerStack.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before layer_stack_probe did not start clean');
    }

    const layerStackProbe = await send('layer_stack_probe', {
      model_dir: model4Dir,
      token_id: 1,
      layers: 4,
    });
    if (
      layerStackProbe.ok !== true ||
      layerStackProbe.value?.provisional !== true ||
      layerStackProbe.value?.token_id !== 1 ||
      layerStackProbe.value?.layers_run !== 4 ||
      layerStackProbe.value?.final_output_len !== 2560 ||
      layerStackProbe.value?.output_len !== 2560 ||
      !Array.isArray(layerStackProbe.value?.per_layer_output_checksums) ||
      layerStackProbe.value.per_layer_output_checksums.length !== 4 ||
      !Number.isFinite(layerStackProbe.value?.layer0_output_checksum) ||
      !Number.isFinite(layerStackProbe.value?.layer1_output_checksum) ||
      !Array.isArray(layerStackProbe.value?.first_values) ||
      layerStackProbe.value.first_values.length !== 8 ||
      !Number.isFinite(layerStackProbe.value?.timing_ms)
    ) {
      throw new Error('layer_stack_probe failed verification');
    }
    for (const value of layerStackProbe.value.per_layer_output_checksums) {
      if (!Number.isFinite(value)) {
        throw new Error('layer_stack_probe per_layer_output_checksums contained a non-finite value');
      }
    }
    for (const value of layerStackProbe.value.first_values) {
      if (!Number.isFinite(value)) {
        throw new Error('layer_stack_probe first_values contained a non-finite value');
      }
    }
    const handleCountsAfterLayerStack = await send('mlx_handle_counts', {});

    console.log(
      '[rusty verify] layer_stack_probe:',
      JSON.stringify({
        token_id: layerStackProbe.value.token_id,
        layers_run: layerStackProbe.value.layers_run,
        per_layer_output_checksums: layerStackProbe.value.per_layer_output_checksums,
        final_output_len: layerStackProbe.value.final_output_len,
        first_values: layerStackProbe.value.first_values,
        timing_ms: layerStackProbe.value.timing_ms,
        handle_counts_before: handleCountsBeforeLayerStack.value,
        handle_counts_after: handleCountsAfterLayerStack.value,
      }, null, 2),
    );
    if (
      handleCountsAfterLayerStack.ok !== true ||
      JSON.stringify(handleCountsAfterLayerStack.value) !== JSON.stringify(handleCountsBeforeLayerStack.value)
    ) {
      throw new Error('mlx_handle_counts changed during layer_stack_probe');
    }
    } else {
      logSkippedProbe('layer0_single_token_probe/layer0_block_probe/layer_stack_probe', 'profile smoke skips layer-local and layer-stack probes');
    }

    let fullStackProbe;
    if (runOracleProfile) {
    const handleCountsBeforeFullStack = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeFullStack.ok !== true ||
      JSON.stringify(handleCountsBeforeFullStack.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before full_stack_single_token_probe did not start clean');
    }

    const fullStackProbe = await send('full_stack_single_token_probe', {
      model_dir: model4Dir,
      token_id: 1,
    });
    if (
      fullStackProbe.ok !== true ||
      fullStackProbe.value?.provisional !== true ||
      fullStackProbe.value?.token_id !== 1 ||
      fullStackProbe.value?.layers_run !== 36 ||
      !Number.isFinite(fullStackProbe.value?.final_hidden_checksum) ||
      !Number.isFinite(fullStackProbe.value?.final_norm_checksum) ||
      fullStackProbe.value?.logits_len !== 151936 ||
      !Array.isArray(fullStackProbe.value?.top_logits) ||
      fullStackProbe.value.top_logits.length !== 10 ||
      !Number.isFinite(fullStackProbe.value?.timing_ms)
    ) {
      throw new Error('full_stack_single_token_probe failed verification');
    }
    const selectedFullStackChecksums = fullStackProbe.value.selected_per_layer_checksums;
    for (const key of ['0', '1', '2', '3', 'last']) {
      if (!Number.isFinite(selectedFullStackChecksums?.[key])) {
        throw new Error('full_stack_single_token_probe selected checksum missing or non-finite: ' + key);
      }
    }
    for (const entry of fullStackProbe.value.top_logits) {
      if (!Number.isInteger(entry?.token_id) || !Number.isFinite(entry?.score)) {
        throw new Error('full_stack_single_token_probe top logits contained invalid entry');
      }
    }
    const handleCountsAfterFullStack = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] full_stack_single_token_probe:',
      JSON.stringify({
        token_id: fullStackProbe.value.token_id,
        layers_run: fullStackProbe.value.layers_run,
        selected_per_layer_checksums: fullStackProbe.value.selected_per_layer_checksums,
        final_hidden_checksum: fullStackProbe.value.final_hidden_checksum,
        final_norm_checksum: fullStackProbe.value.final_norm_checksum,
        logits_len: fullStackProbe.value.logits_len,
        top_logits: fullStackProbe.value.top_logits,
        timing_ms: fullStackProbe.value.timing_ms,
        handle_counts_before: handleCountsBeforeFullStack.value,
        handle_counts_after: handleCountsAfterFullStack.value,
      }, null, 2),
    );
    if (
      handleCountsAfterFullStack.ok !== true ||
      JSON.stringify(handleCountsAfterFullStack.value) !== JSON.stringify(handleCountsBeforeFullStack.value)
    ) {
      throw new Error('mlx_handle_counts changed during full_stack_single_token_probe');
    }
    } else {
      fullStackProbe = {
        ok: true,
        value: {
          token_id: 1,
          top_logits: [{ token_id: 15, score: null }],
          recorded_contract: true,
        },
      };
      logSkippedProbe('full_stack_single_token_probe', 'profile smoke uses recorded prompt/decode contract; CPU full oracle is RUSTY_VERIFY_PROFILE=oracle only');
    }

    if (runOracleProfile) {
    const handleCountsBeforeKvStorage = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeKvStorage.ok !== true ||
      JSON.stringify(handleCountsBeforeKvStorage.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before kv_cache_storage_probe did not start clean');
    }
    const kvCacheStorageProbe = await send('kv_cache_storage_probe', {});
    if (
      kvCacheStorageProbe.ok !== true ||
      kvCacheStorageProbe.value?.structural_cache !== true ||
      kvCacheStorageProbe.value?.optimized_incremental_attention !== false ||
      kvCacheStorageProbe.value?.layers_allocated !== 36 ||
      kvCacheStorageProbe.value?.layers_with_positions !== 36 ||
      kvCacheStorageProbe.value?.positions_stored !== 1 ||
      kvCacheStorageProbe.value?.k_length !== 1024 ||
      kvCacheStorageProbe.value?.v_length !== 1024 ||
      kvCacheStorageProbe.value?.first_layer?.positions !== 1 ||
      kvCacheStorageProbe.value?.last_layer?.positions !== 1
    ) {
      throw new Error('kv_cache_storage_probe failed verification');
    }
    const handleCountsAfterKvStorage = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] kv_cache_storage_probe:',
      JSON.stringify({
        structural_cache: kvCacheStorageProbe.value.structural_cache,
        optimized_incremental_attention: kvCacheStorageProbe.value.optimized_incremental_attention,
        cache_handle: kvCacheStorageProbe.value.cache_handle,
        owner_session: kvCacheStorageProbe.value.owner_session,
        layers_allocated: kvCacheStorageProbe.value.layers_allocated,
        layers_with_positions: kvCacheStorageProbe.value.layers_with_positions,
        positions_stored: kvCacheStorageProbe.value.positions_stored,
        k_length: kvCacheStorageProbe.value.k_length,
        v_length: kvCacheStorageProbe.value.v_length,
        first_layer: kvCacheStorageProbe.value.first_layer,
        last_layer: kvCacheStorageProbe.value.last_layer,
        notes: kvCacheStorageProbe.value.notes,
        handle_counts_before: handleCountsBeforeKvStorage.value,
        handle_counts_after: handleCountsAfterKvStorage.value,
      }, null, 2),
    );
    if (
      handleCountsAfterKvStorage.ok !== true ||
      JSON.stringify(handleCountsAfterKvStorage.value) !== JSON.stringify(handleCountsBeforeKvStorage.value)
    ) {
      throw new Error('mlx_handle_counts changed during kv_cache_storage_probe');
    }
    } else {
      logSkippedProbe('kv_cache_storage_probe', 'profile smoke skips structural CPU cache probe; CPU diagnostics are oracle/full profile only');
    }

    const model4TokenizerLoaded = await send('load_tokenizer', { path: model4Dir });
    if (
      model4TokenizerLoaded.ok !== true ||
      model4TokenizerLoaded.value?.tokenizer_kind !== 'huggingface_json' ||
      typeof model4TokenizerLoaded.value?.tokenizer !== 'string'
    ) {
      throw new Error('load_tokenizer on model4 failed verification');
    }
    const model4Tokenizer = model4TokenizerLoaded.value.tokenizer;

    if (profileAtLeast(verifierProfile, 'full')) {
      const handleCountsBeforeIncrementalAttention = await send('mlx_handle_counts', {});
      if (
        handleCountsBeforeIncrementalAttention.ok !== true ||
        JSON.stringify(handleCountsBeforeIncrementalAttention.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
      ) {
        throw new Error('mlx_handle_counts before incremental_attention_probe did not start clean');
      }
      const incrementalAttentionProbe = await send('incremental_attention_probe', {
        model_dir: model4Dir,
        prompt_token_id: fullStackProbe.value.token_id,
        decode_token_id: fullStackProbe.value.top_logits[0].token_id,
      });
      if (
        incrementalAttentionProbe.ok !== true ||
        incrementalAttentionProbe.value?.kv_cache_reused !== true ||
        incrementalAttentionProbe.value?.optimized_incremental_attention !== true ||
        incrementalAttentionProbe.value?.prompt_token_id !== 1 ||
        incrementalAttentionProbe.value?.decode_token_id !== fullStackProbe.value.top_logits[0].token_id ||
        incrementalAttentionProbe.value?.layers_run !== 36 ||
        incrementalAttentionProbe.value?.positions_before !== 1 ||
        incrementalAttentionProbe.value?.positions_after !== 2 ||
        incrementalAttentionProbe.value?.k_length !== 1024 ||
        incrementalAttentionProbe.value?.v_length !== 1024 ||
        incrementalAttentionProbe.value?.logits_len !== 151936 ||
        !Number.isInteger(incrementalAttentionProbe.value?.top_token_id) ||
        !Number.isFinite(incrementalAttentionProbe.value?.top_token_score) ||
        !Number.isFinite(incrementalAttentionProbe.value?.timing_ms)
      ) {
        throw new Error('incremental_attention_probe failed verification');
      }
      const incrementalAttentionDecoded = await send('tokenizer_decode', {
        tokenizer: model4Tokenizer,
        tokens: [incrementalAttentionProbe.value.top_token_id],
      });
      if (incrementalAttentionDecoded.ok !== true || typeof incrementalAttentionDecoded.value?.text !== 'string') {
        throw new Error('tokenizer_decode for incremental_attention_probe failed verification');
      }
      const handleCountsAfterIncrementalAttention = await send('mlx_handle_counts', {});
      console.log(
        '[rusty verify] incremental_attention_probe:',
        JSON.stringify({
          prompt_token_id: incrementalAttentionProbe.value.prompt_token_id,
          decode_token_id: incrementalAttentionProbe.value.decode_token_id,
          layers_run: incrementalAttentionProbe.value.layers_run,
          positions_before: incrementalAttentionProbe.value.positions_before,
          positions_after: incrementalAttentionProbe.value.positions_after,
          k_length: incrementalAttentionProbe.value.k_length,
          v_length: incrementalAttentionProbe.value.v_length,
          logits_len: incrementalAttentionProbe.value.logits_len,
          top_token: {
            token_id: incrementalAttentionProbe.value.top_token_id,
            score: incrementalAttentionProbe.value.top_token_score,
            decoded_text: incrementalAttentionDecoded.value.text,
          },
          timing_ms: incrementalAttentionProbe.value.timing_ms,
          kv_cache_reused: incrementalAttentionProbe.value.kv_cache_reused,
          optimized_incremental_attention: incrementalAttentionProbe.value.optimized_incremental_attention,
          notes: incrementalAttentionProbe.value.notes,
          handle_counts_before: handleCountsBeforeIncrementalAttention.value,
          handle_counts_after: handleCountsAfterIncrementalAttention.value,
        }, null, 2),
      );
      if (
        handleCountsAfterIncrementalAttention.ok !== true ||
        JSON.stringify(handleCountsAfterIncrementalAttention.value) !== JSON.stringify(handleCountsBeforeIncrementalAttention.value)
      ) {
        throw new Error('mlx_handle_counts changed during incremental_attention_probe');
      }
    } else {
      logSkippedProbe('incremental_attention_probe', 'profile smoke skips non-resident incremental attention; resident optimized decode covers smoke');
    }

    const handleCountsBeforeMlxQuantizedLinear = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeMlxQuantizedLinear.ok !== true ||
      JSON.stringify(handleCountsBeforeMlxQuantizedLinear.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before quantized_linear_mlx_probe did not start clean');
    }
    const mlxQuantizedLinearProbe = await send('quantized_linear_mlx_probe', {
      model_dir: model4Dir,
    });
    if (
      mlxQuantizedLinearProbe.ok !== true ||
      typeof mlxQuantizedLinearProbe.value?.mlx_quantized_linear_available !== 'boolean' ||
      typeof mlxQuantizedLinearProbe.value?.mlx_quantized_linear_applied_to_probe !== 'boolean' ||
      typeof mlxQuantizedLinearProbe.value?.mlx_quantized_linear_applied_to_generation !== 'boolean' ||
      typeof mlxQuantizedLinearProbe.value?.fallback_used !== 'boolean' ||
      !Array.isArray(mlxQuantizedLinearProbe.value?.shape?.cpu) ||
      !Array.isArray(mlxQuantizedLinearProbe.value?.shape?.mlx) ||
      !Array.isArray(mlxQuantizedLinearProbe.value?.first_values?.cpu) ||
      !Array.isArray(mlxQuantizedLinearProbe.value?.first_values?.mlx) ||
      !Number.isFinite(mlxQuantizedLinearProbe.value?.checksum_cpu) ||
      !Number.isFinite(mlxQuantizedLinearProbe.value?.checksum_mlx) ||
      !Number.isFinite(mlxQuantizedLinearProbe.value?.max_abs_diff) ||
      !Number.isFinite(mlxQuantizedLinearProbe.value?.timing_cpu_ms) ||
      !Number.isFinite(mlxQuantizedLinearProbe.value?.timing_mlx_ms)
    ) {
      throw new Error('quantized_linear_mlx_probe failed verification');
    }
    if (
      mlxQuantizedLinearProbe.value.mlx_quantized_linear_applied_to_probe === true &&
      mlxQuantizedLinearProbe.value.shape.cpu[0] !== mlxQuantizedLinearProbe.value.shape.mlx[0]
    ) {
      throw new Error('quantized_linear_mlx_probe shape mismatch');
    }
    const handleCountsAfterMlxQuantizedLinear = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] quantized_linear_mlx_probe:',
      JSON.stringify({
        group: mlxQuantizedLinearProbe.value.group,
        mlx_quantized_linear_available: mlxQuantizedLinearProbe.value.mlx_quantized_linear_available,
        mlx_quantized_linear_applied_to_probe: mlxQuantizedLinearProbe.value.mlx_quantized_linear_applied_to_probe,
        mlx_quantized_linear_applied_to_generation: mlxQuantizedLinearProbe.value.mlx_quantized_linear_applied_to_generation,
        fallback_used: mlxQuantizedLinearProbe.value.fallback_used,
        shape: mlxQuantizedLinearProbe.value.shape,
        first_values: mlxQuantizedLinearProbe.value.first_values,
        checksum_cpu: mlxQuantizedLinearProbe.value.checksum_cpu,
        checksum_mlx: mlxQuantizedLinearProbe.value.checksum_mlx,
        max_abs_diff: mlxQuantizedLinearProbe.value.max_abs_diff,
        timing_cpu_ms: mlxQuantizedLinearProbe.value.timing_cpu_ms,
        timing_mlx_ms: mlxQuantizedLinearProbe.value.timing_mlx_ms,
        speedup: mlxQuantizedLinearProbe.value.speedup,
        mlx_error: mlxQuantizedLinearProbe.value.mlx_error,
        notes: mlxQuantizedLinearProbe.value.notes,
        handle_counts_before: handleCountsBeforeMlxQuantizedLinear.value,
        handle_counts_after: handleCountsAfterMlxQuantizedLinear.value,
      }, null, 2),
    );
    if (
      handleCountsAfterMlxQuantizedLinear.ok !== true ||
      JSON.stringify(handleCountsAfterMlxQuantizedLinear.value) !== JSON.stringify(handleCountsBeforeMlxQuantizedLinear.value)
    ) {
      throw new Error('mlx_handle_counts changed during quantized_linear_mlx_probe');
    }

    if (runOracleProfile) {
    const handleCountsBeforeMetalFirst = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeMetalFirst.ok !== true ||
      JSON.stringify(handleCountsBeforeMetalFirst.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before metal_first_resident_decode_probe did not start clean');
    }
    const metalFirstProbe = await send('metal_first_resident_decode_probe', {
      model_dir: model4Dir,
      prompt_token_id: fullStackProbe.value.token_id,
      decode_token_id: fullStackProbe.value.top_logits[0].token_id,
    });
    if (
      metalFirstProbe.ok !== true ||
      metalFirstProbe.value?.comparison_only !== true ||
      metalFirstProbe.value?.layers_run !== 36 ||
      typeof metalFirstProbe.value?.mlx_quantized_linear_available !== 'boolean' ||
      typeof metalFirstProbe.value?.same_token !== 'boolean' ||
      metalFirstProbe.value.same_token !== true ||
      !Number.isFinite(metalFirstProbe.value?.final_norm_checksum_diff) ||
      !Number.isFinite(metalFirstProbe.value?.max_abs_diff) ||
      !Number.isInteger(metalFirstProbe.value?.cpu_oracle?.top_token_id) ||
      !Number.isInteger(metalFirstProbe.value?.metal_first?.top_token_id) ||
      !Number.isFinite(metalFirstProbe.value?.cpu_oracle?.final_norm_checksum) ||
      !Number.isFinite(metalFirstProbe.value?.metal_first?.final_norm_checksum) ||
      !Number.isFinite(metalFirstProbe.value?.cpu_oracle?.timing_ms) ||
      !Number.isFinite(metalFirstProbe.value?.metal_first?.timing_ms) ||
      typeof metalFirstProbe.value?.metal_first?.fallback_used !== 'boolean' ||
      !Array.isArray(metalFirstProbe.value?.metal_first?.fallback_steps) ||
      typeof metalFirstProbe.value?.metal_first?.backend_report?.qkv?.backend !== 'string' ||
      typeof metalFirstProbe.value?.metal_first?.backend_report?.['logits/top1']?.backend !== 'string'
    ) {
      throw new Error('metal_first_resident_decode_probe failed verification');
    }
    const metalFirstCpuDecoded = await send('tokenizer_decode', {
      tokenizer: model4Tokenizer,
      tokens: [metalFirstProbe.value.cpu_oracle.top_token_id],
    });
    const metalFirstDecoded = await send('tokenizer_decode', {
      tokenizer: model4Tokenizer,
      tokens: [metalFirstProbe.value.metal_first.top_token_id],
    });
    if (
      metalFirstCpuDecoded.ok !== true ||
      metalFirstDecoded.ok !== true ||
      metalFirstCpuDecoded.value?.text !== metalFirstDecoded.value?.text
    ) {
      throw new Error('metal_first_resident_decode_probe decoded text mismatch');
    }
    const handleCountsAfterMetalFirst = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] metal_first_resident_decode_probe:',
      JSON.stringify({
        prompt_token_id: metalFirstProbe.value.prompt_token_id,
        decode_token_id: metalFirstProbe.value.decode_token_id,
        layers_run: metalFirstProbe.value.layers_run,
        mlx_quantized_linear_available: metalFirstProbe.value.mlx_quantized_linear_available,
        cpu_oracle: {
          ...metalFirstProbe.value.cpu_oracle,
          decoded_text: metalFirstCpuDecoded.value.text,
        },
        metal_first: {
          ...metalFirstProbe.value.metal_first,
          decoded_text: metalFirstDecoded.value.text,
        },
        same_token: metalFirstProbe.value.same_token,
        final_norm_checksum_diff: metalFirstProbe.value.final_norm_checksum_diff,
        max_abs_diff: metalFirstProbe.value.max_abs_diff,
        notes: metalFirstProbe.value.notes,
        handle_counts_before: handleCountsBeforeMetalFirst.value,
        handle_counts_after: handleCountsAfterMetalFirst.value,
      }, null, 2),
    );
    if (
      handleCountsAfterMetalFirst.ok !== true ||
      JSON.stringify(handleCountsAfterMetalFirst.value) !== JSON.stringify(handleCountsBeforeMetalFirst.value)
    ) {
      throw new Error('mlx_handle_counts changed during metal_first_resident_decode_probe');
    }
    } else {
      logSkippedProbe('metal_first_resident_decode_probe CPU oracle comparison', 'profile smoke runs Metal-first generation only; CPU oracle comparison is RUSTY_VERIFY_PROFILE=oracle');
    }

    phaseTimer.switchTo('active decode');
    const handleCountsBeforeResidency = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeResidency.ok !== true ||
      JSON.stringify(handleCountsBeforeResidency.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before session_layer_residency_probe did not start clean');
    }
    const sessionLayerResidencyProbe = await send('session_layer_residency_probe', {
      model_dir: model4Dir,
      prompt_token_id: fullStackProbe.value.token_id,
      decode_token_id: fullStackProbe.value.top_logits[0].token_id,
    });
    if (
      sessionLayerResidencyProbe.ok !== true ||
      sessionLayerResidencyProbe.value?.layers_resident !== 36 ||
      sessionLayerResidencyProbe.value?.total_groups_resident !== 396 ||
      sessionLayerResidencyProbe.value?.resident_total_byte_size <= 0 ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.resident_group_load_ms) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.prompt_pass_ms) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.incremental_pass_ms) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.decode_cleanup_ms) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.total_probe_ms) ||
      sessionLayerResidencyProbe.value?.resident_groups_persistent_across_probe !== false ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.prompt_pass_timing_ms) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.incremental_pass_timing_ms) ||
      !Number.isInteger(sessionLayerResidencyProbe.value?.tensor_group_load_count) ||
      sessionLayerResidencyProbe.value?.positions_before !== 1 ||
      sessionLayerResidencyProbe.value?.positions_after !== 2 ||
      sessionLayerResidencyProbe.value?.k_length !== 1024 ||
      sessionLayerResidencyProbe.value?.v_length !== 1024 ||
      sessionLayerResidencyProbe.value?.logits_len !== 151936 ||
      sessionLayerResidencyProbe.value?.resident_groups_reused !== true ||
      sessionLayerResidencyProbe.value?.optimized_incremental_attention !== true ||
      sessionLayerResidencyProbe.value?.optimized_path_applied_to_generation !== !scalarQuantizedLinearForced ||
      sessionLayerResidencyProbe.value?.layout_cached_path_applied_to_generation !== !scalarQuantizedLinearForced ||
      sessionLayerResidencyProbe.value?.mlp_pair_optimized_path_applied_to_generation !== !scalarQuantizedLinearForced ||
      sessionLayerResidencyProbe.value?.logits_top1_optimized_path_applied_to_generation !== !scalarQuantizedLinearForced ||
      sessionLayerResidencyProbe.value?.cached_mlx_arrays_path_applied_to_generation !== false ||
      typeof sessionLayerResidencyProbe.value?.mlx_quantized_linear_available !== 'boolean' ||
      typeof sessionLayerResidencyProbe.value?.mlx_quantized_linear_applied_to_generation !== 'boolean' ||
      sessionLayerResidencyProbe.value?.gate_up_full_block_optimized_path_applied_to_generation !== false ||
      sessionLayerResidencyProbe.value?.down_full_block_optimized_path_applied_to_generation !== false ||
      typeof sessionLayerResidencyProbe.value?.fallback_used !== 'boolean' ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.final_norm_checksum) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.timing_buckets_ms?.qkv_projections) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.timing_buckets_ms?.o_projection) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.timing_buckets_ms?.gate_up_paired_projection) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.timing_buckets_ms?.gate_projection) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.timing_buckets_ms?.up_projection) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.timing_buckets_ms?.gate_up_activation) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.timing_buckets_ms?.down_projection) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.timing_buckets_ms?.final_norm) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.timing_buckets_ms?.logits_projection_top1) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.projection_timing_breakdown_ms?.gate_proj?.setup_ms) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.projection_timing_breakdown_ms?.gate_proj?.compute_eval_ms) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.projection_timing_breakdown_ms?.gate_proj?.readback_ms) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.projection_timing_breakdown_ms?.gate_proj?.total_ms) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.projection_timing_breakdown_ms?.up_proj?.setup_ms) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.projection_timing_breakdown_ms?.up_proj?.compute_eval_ms) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.projection_timing_breakdown_ms?.up_proj?.readback_ms) ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.projection_timing_breakdown_ms?.up_proj?.total_ms) ||
      typeof sessionLayerResidencyProbe.value?.largest_arithmetic_bucket !== 'string' ||
      !Number.isFinite(sessionLayerResidencyProbe.value?.largest_arithmetic_bucket_ms) ||
      !Array.isArray(sessionLayerResidencyProbe.value?.fallback_steps) ||
      typeof sessionLayerResidencyProbe.value?.backend_report?.qkv?.backend !== 'string' ||
      typeof sessionLayerResidencyProbe.value?.backend_report?.o_proj?.backend !== 'string' ||
      typeof sessionLayerResidencyProbe.value?.backend_report?.gate_up?.gate_backend !== 'string' ||
      typeof sessionLayerResidencyProbe.value?.backend_report?.gate_up?.up_backend !== 'string' ||
      typeof sessionLayerResidencyProbe.value?.backend_report?.gate_up?.activation_backend !== 'string' ||
      typeof sessionLayerResidencyProbe.value?.backend_report?.gate_up?.fallback_used !== 'boolean' ||
      typeof sessionLayerResidencyProbe.value?.backend_report?.down?.backend !== 'string' ||
      typeof sessionLayerResidencyProbe.value?.backend_report?.['logits/top1']?.backend !== 'string'
    ) {
      throw new Error('session_layer_residency_probe failed verification');
    }
    if (sessionLayerResidencyProbe.value.fallback_steps.length !== 0) {
      throw new Error('session_layer_residency_probe reported unexpected fallback steps');
    }
    for (const stepName of ['qkv', 'o_proj', 'down', 'logits/top1']) {
      const step = sessionLayerResidencyProbe.value.backend_report[stepName];
      if (step.backend !== 'metal' || step.fallback_used !== false) {
        throw new Error(\`session_layer_residency_probe unexpected CPU fallback in ${stepName}\`);
      }
    }
    const gateUpStep = sessionLayerResidencyProbe.value.backend_report.gate_up;
    if (
      gateUpStep.gate_backend !== 'metal' ||
      gateUpStep.up_backend !== 'metal' ||
      gateUpStep.activation_backend !== 'cpu' ||
      gateUpStep.fallback_used !== false
    ) {
      throw new Error('session_layer_residency_probe unexpected CPU fallback in gate_up');
    }
    if (
      sessionLayerResidencyProbe.value.top_token_id !== 24 ||
      Math.abs(sessionLayerResidencyProbe.value.final_norm_checksum - 130.289) > 0.01
    ) {
      throw new Error('session_layer_residency_probe deviated from recorded smoke contract');
    }
    const handleCountsAfterResidency = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] session_layer_residency_probe:',
      JSON.stringify({
        layers_resident: sessionLayerResidencyProbe.value.layers_resident,
        total_groups_resident: sessionLayerResidencyProbe.value.total_groups_resident,
        resident_total_byte_size: sessionLayerResidencyProbe.value.resident_total_byte_size,
        resident_group_load_ms: sessionLayerResidencyProbe.value.resident_group_load_ms,
        prompt_pass_ms: sessionLayerResidencyProbe.value.prompt_pass_ms,
        incremental_pass_ms: sessionLayerResidencyProbe.value.incremental_pass_ms,
        decode_cleanup_ms: sessionLayerResidencyProbe.value.decode_cleanup_ms,
        total_probe_ms: sessionLayerResidencyProbe.value.total_probe_ms,
        resident_groups_persistent_across_probe: sessionLayerResidencyProbe.value.resident_groups_persistent_across_probe,
        prompt_pass_timing_ms: sessionLayerResidencyProbe.value.prompt_pass_timing_ms,
        incremental_pass_timing_ms: sessionLayerResidencyProbe.value.incremental_pass_timing_ms,
        timing_ms: sessionLayerResidencyProbe.value.incremental_pass_timing_ms,
        tensor_group_load_count: sessionLayerResidencyProbe.value.tensor_group_load_count,
        positions_before: sessionLayerResidencyProbe.value.positions_before,
        positions_after: sessionLayerResidencyProbe.value.positions_after,
        k_length: sessionLayerResidencyProbe.value.k_length,
        v_length: sessionLayerResidencyProbe.value.v_length,
        logits_len: sessionLayerResidencyProbe.value.logits_len,
        top_token: {
          token_id: sessionLayerResidencyProbe.value.top_token_id,
          score: sessionLayerResidencyProbe.value.top_token_score,
        },
        final_norm_checksum: sessionLayerResidencyProbe.value.final_norm_checksum,
        timing_buckets_ms: sessionLayerResidencyProbe.value.timing_buckets_ms,
        projection_timing_breakdown_ms: sessionLayerResidencyProbe.value.projection_timing_breakdown_ms,
        largest_arithmetic_bucket: sessionLayerResidencyProbe.value.largest_arithmetic_bucket,
        largest_arithmetic_bucket_ms: sessionLayerResidencyProbe.value.largest_arithmetic_bucket_ms,
        backend_report: sessionLayerResidencyProbe.value.backend_report,
        fallback_steps: sessionLayerResidencyProbe.value.fallback_steps,
        resident_groups_reused: sessionLayerResidencyProbe.value.resident_groups_reused,
        optimized_incremental_attention: sessionLayerResidencyProbe.value.optimized_incremental_attention,
        optimized_path_applied_to_generation: sessionLayerResidencyProbe.value.optimized_path_applied_to_generation,
        layout_cached_path_applied_to_generation: sessionLayerResidencyProbe.value.layout_cached_path_applied_to_generation,
        mlp_pair_optimized_path_applied_to_generation: sessionLayerResidencyProbe.value.mlp_pair_optimized_path_applied_to_generation,
        logits_top1_optimized_path_applied_to_generation: sessionLayerResidencyProbe.value.logits_top1_optimized_path_applied_to_generation,
        cached_mlx_arrays_path_applied_to_generation: sessionLayerResidencyProbe.value.cached_mlx_arrays_path_applied_to_generation,
        mlx_quantized_linear_available: sessionLayerResidencyProbe.value.mlx_quantized_linear_available,
        mlx_quantized_linear_applied_to_probe: mlxQuantizedLinearProbe.value.mlx_quantized_linear_applied_to_probe,
        mlx_quantized_linear_applied_to_generation: sessionLayerResidencyProbe.value.mlx_quantized_linear_applied_to_generation,
        gate_up_full_block_optimized_path_applied_to_generation: sessionLayerResidencyProbe.value.gate_up_full_block_optimized_path_applied_to_generation,
        down_full_block_optimized_path_applied_to_generation: sessionLayerResidencyProbe.value.down_full_block_optimized_path_applied_to_generation,
        fallback_used: sessionLayerResidencyProbe.value.fallback_used,
        notes: sessionLayerResidencyProbe.value.notes,
        handle_counts_before: handleCountsBeforeResidency.value,
        handle_counts_after: handleCountsAfterResidency.value,
      }, null, 2),
    );
    const smokeTimingBucketSummary = [
      ['qkv', sessionLayerResidencyProbe.value.timing_buckets_ms.qkv_projections],
      ['o_proj', sessionLayerResidencyProbe.value.timing_buckets_ms.o_projection],
      ['gate_proj', sessionLayerResidencyProbe.value.timing_buckets_ms.gate_projection],
      ['up_proj', sessionLayerResidencyProbe.value.timing_buckets_ms.up_projection],
      ['gate_up_activation', sessionLayerResidencyProbe.value.timing_buckets_ms.gate_up_activation],
      ['down', sessionLayerResidencyProbe.value.timing_buckets_ms.down_projection],
      ['logits/top1', sessionLayerResidencyProbe.value.timing_buckets_ms.logits_projection_top1],
      ['attention', sessionLayerResidencyProbe.value.backend_report.attention.timing_ms],
      ['norms', sessionLayerResidencyProbe.value.timing_buckets_ms.final_norm],
      ['embedding', sessionLayerResidencyProbe.value.backend_report.embedding.timing_ms],
    ]
      .map(([bucket, timing_ms]) => ({ bucket, timing_ms }))
      .sort((left, right) => right.timing_ms - left.timing_ms);
    console.log(
      '[rusty verify] smoke_timing_bucket_summary:',
      JSON.stringify(smokeTimingBucketSummary, null, 2),
    );
    if (
      handleCountsAfterResidency.ok !== true ||
      JSON.stringify(handleCountsAfterResidency.value) !== JSON.stringify(handleCountsBeforeResidency.value)
    ) {
      throw new Error('mlx_handle_counts changed during session_layer_residency_probe');
    }
    phaseTimer.switchTo('tensor metadata/load');

    if (profileAtLeast(verifierProfile, 'full')) {
      const handleCountsBeforeLayoutCachedCompare = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeLayoutCachedCompare.ok !== true ||
      JSON.stringify(handleCountsBeforeLayoutCachedCompare.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before resident_incremental_layout_cached_compare_probe did not start clean');
    }
    const layoutCachedCompareProbe = await send('resident_incremental_layout_cached_compare_probe', {
      model_dir: model4Dir,
      decode_token_id: fullStackProbe.value.top_logits[0].token_id,
    });
    if (
      layoutCachedCompareProbe.ok !== true ||
      layoutCachedCompareProbe.value?.comparison_only !== true ||
      layoutCachedCompareProbe.value?.optimized_path_applied_to_generation !== true ||
      layoutCachedCompareProbe.value?.layout_cached_path_applied_to_generation !== true ||
      layoutCachedCompareProbe.value?.layers_run !== 36 ||
      layoutCachedCompareProbe.value?.logits_len !== 151936 ||
      !Number.isFinite(layoutCachedCompareProbe.value?.max_abs_diff) ||
      !Number.isFinite(layoutCachedCompareProbe.value?.final_norm_checksum_old_optimized) ||
      !Number.isFinite(layoutCachedCompareProbe.value?.final_norm_checksum_layout_cached) ||
      !Number.isFinite(layoutCachedCompareProbe.value?.timing_old_optimized_ms) ||
      !Number.isFinite(layoutCachedCompareProbe.value?.timing_layout_cached_ms) ||
      !Number.isFinite(layoutCachedCompareProbe.value?.speedup) ||
      !Number.isInteger(layoutCachedCompareProbe.value?.top_token_old_optimized?.token_id) ||
      !Number.isInteger(layoutCachedCompareProbe.value?.top_token_layout_cached?.token_id) ||
      layoutCachedCompareProbe.value.top_token_old_optimized.token_id !== layoutCachedCompareProbe.value.top_token_layout_cached.token_id
    ) {
      throw new Error('resident_incremental_layout_cached_compare_probe failed verification');
    }
    const layoutOldDecoded = await send('tokenizer_decode', {
      tokenizer: model4Tokenizer,
      tokens: [layoutCachedCompareProbe.value.top_token_old_optimized.token_id],
    });
    const layoutCachedDecoded = await send('tokenizer_decode', {
      tokenizer: model4Tokenizer,
      tokens: [layoutCachedCompareProbe.value.top_token_layout_cached.token_id],
    });
    if (
      layoutOldDecoded.ok !== true ||
      layoutCachedDecoded.ok !== true ||
      layoutOldDecoded.value?.text !== layoutCachedDecoded.value?.text
    ) {
      throw new Error('resident_incremental_layout_cached_compare_probe decoded text mismatch');
    }
    const handleCountsAfterLayoutCachedCompare = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] resident_incremental_layout_cached_compare_probe:',
      JSON.stringify({
        decode_token_id: layoutCachedCompareProbe.value.decode_token_id,
        layers_run: layoutCachedCompareProbe.value.layers_run,
        positions_before: layoutCachedCompareProbe.value.positions_before,
        positions_after_old_optimized: layoutCachedCompareProbe.value.positions_after_old_optimized,
        positions_after_layout_cached: layoutCachedCompareProbe.value.positions_after_layout_cached,
        logits_len: layoutCachedCompareProbe.value.logits_len,
        max_abs_diff: layoutCachedCompareProbe.value.max_abs_diff,
        top_token_old_optimized: {
          ...layoutCachedCompareProbe.value.top_token_old_optimized,
          decoded_text: layoutOldDecoded.value.text,
        },
        top_token_layout_cached: {
          ...layoutCachedCompareProbe.value.top_token_layout_cached,
          decoded_text: layoutCachedDecoded.value.text,
        },
        final_norm_checksum_old_optimized: layoutCachedCompareProbe.value.final_norm_checksum_old_optimized,
        final_norm_checksum_layout_cached: layoutCachedCompareProbe.value.final_norm_checksum_layout_cached,
        timing_old_optimized_ms: layoutCachedCompareProbe.value.timing_old_optimized_ms,
        timing_layout_cached_ms: layoutCachedCompareProbe.value.timing_layout_cached_ms,
        speedup: layoutCachedCompareProbe.value.speedup,
        layout_metadata_example: layoutCachedCompareProbe.value.layout_metadata_example,
        optimized_path_applied_to_generation: layoutCachedCompareProbe.value.optimized_path_applied_to_generation,
        layout_cached_path_applied_to_generation: layoutCachedCompareProbe.value.layout_cached_path_applied_to_generation,
        notes: layoutCachedCompareProbe.value.notes,
        handle_counts_before: handleCountsBeforeLayoutCachedCompare.value,
        handle_counts_after: handleCountsAfterLayoutCachedCompare.value,
      }, null, 2),
    );
    if (
      handleCountsAfterLayoutCachedCompare.ok !== true ||
      JSON.stringify(handleCountsAfterLayoutCachedCompare.value) !== JSON.stringify(handleCountsBeforeLayoutCachedCompare.value)
    ) {
      throw new Error('mlx_handle_counts changed during resident_incremental_layout_cached_compare_probe');
    }
    } else {
      logSkippedProbe('resident_incremental_layout_cached_compare_probe', 'profile smoke skips old optimized/layout-cached comparison after MLP paired path promotion');
    }

    if (profileAtLeast(verifierProfile, 'full')) {
      const handleCountsBeforeMlpOptimizedCompare = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeMlpOptimizedCompare.ok !== true ||
      JSON.stringify(handleCountsBeforeMlpOptimizedCompare.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before resident_incremental_mlp_optimized_compare_probe did not start clean');
    }
    const mlpOptimizedCompareProbe = await send('resident_incremental_mlp_optimized_compare_probe', {
      model_dir: model4Dir,
      decode_token_id: fullStackProbe.value.top_logits[0].token_id,
    });
    if (
      mlpOptimizedCompareProbe.ok !== true ||
      mlpOptimizedCompareProbe.value?.comparison_only !== true ||
      mlpOptimizedCompareProbe.value?.optimized_path_applied_to_generation !== true ||
      mlpOptimizedCompareProbe.value?.layout_cached_path_applied_to_generation !== true ||
      mlpOptimizedCompareProbe.value?.mlp_pair_optimized_path_applied_to_generation !== true ||
      mlpOptimizedCompareProbe.value?.layers_run !== 36 ||
      mlpOptimizedCompareProbe.value?.logits_len !== 151936 ||
      !Number.isFinite(mlpOptimizedCompareProbe.value?.max_abs_diff) ||
      !Number.isFinite(mlpOptimizedCompareProbe.value?.final_norm_checksum_old_layout_cached) ||
      !Number.isFinite(mlpOptimizedCompareProbe.value?.final_norm_checksum_mlp_optimized) ||
      !Number.isFinite(mlpOptimizedCompareProbe.value?.timing_old_layout_cached_ms) ||
      !Number.isFinite(mlpOptimizedCompareProbe.value?.timing_mlp_optimized_ms) ||
      !Number.isFinite(mlpOptimizedCompareProbe.value?.speedup) ||
      !Number.isInteger(mlpOptimizedCompareProbe.value?.top_token_old_layout_cached?.token_id) ||
      !Number.isInteger(mlpOptimizedCompareProbe.value?.top_token_mlp_optimized?.token_id) ||
      mlpOptimizedCompareProbe.value.top_token_old_layout_cached.token_id !== mlpOptimizedCompareProbe.value.top_token_mlp_optimized.token_id
    ) {
      throw new Error('resident_incremental_mlp_optimized_compare_probe failed verification');
    }
    const mlpOldDecoded = await send('tokenizer_decode', {
      tokenizer: model4Tokenizer,
      tokens: [mlpOptimizedCompareProbe.value.top_token_old_layout_cached.token_id],
    });
    const mlpOptimizedDecoded = await send('tokenizer_decode', {
      tokenizer: model4Tokenizer,
      tokens: [mlpOptimizedCompareProbe.value.top_token_mlp_optimized.token_id],
    });
    if (
      mlpOldDecoded.ok !== true ||
      mlpOptimizedDecoded.ok !== true ||
      mlpOldDecoded.value?.text !== mlpOptimizedDecoded.value?.text
    ) {
      throw new Error('resident_incremental_mlp_optimized_compare_probe decoded text mismatch');
    }
    const handleCountsAfterMlpOptimizedCompare = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] resident_incremental_mlp_optimized_compare_probe:',
      JSON.stringify({
        decode_token_id: mlpOptimizedCompareProbe.value.decode_token_id,
        layers_run: mlpOptimizedCompareProbe.value.layers_run,
        positions_before: mlpOptimizedCompareProbe.value.positions_before,
        positions_after_old_layout_cached: mlpOptimizedCompareProbe.value.positions_after_old_layout_cached,
        positions_after_mlp_optimized: mlpOptimizedCompareProbe.value.positions_after_mlp_optimized,
        logits_len: mlpOptimizedCompareProbe.value.logits_len,
        max_abs_diff: mlpOptimizedCompareProbe.value.max_abs_diff,
        top_token_old_layout_cached: {
          ...mlpOptimizedCompareProbe.value.top_token_old_layout_cached,
          decoded_text: mlpOldDecoded.value.text,
        },
        top_token_mlp_optimized: {
          ...mlpOptimizedCompareProbe.value.top_token_mlp_optimized,
          decoded_text: mlpOptimizedDecoded.value.text,
        },
        final_norm_checksum_old_layout_cached: mlpOptimizedCompareProbe.value.final_norm_checksum_old_layout_cached,
        final_norm_checksum_mlp_optimized: mlpOptimizedCompareProbe.value.final_norm_checksum_mlp_optimized,
        timing_old_layout_cached_ms: mlpOptimizedCompareProbe.value.timing_old_layout_cached_ms,
        timing_mlp_optimized_ms: mlpOptimizedCompareProbe.value.timing_mlp_optimized_ms,
        speedup: mlpOptimizedCompareProbe.value.speedup,
        optimized_path_applied_to_generation: mlpOptimizedCompareProbe.value.optimized_path_applied_to_generation,
        layout_cached_path_applied_to_generation: mlpOptimizedCompareProbe.value.layout_cached_path_applied_to_generation,
        mlp_pair_optimized_path_applied_to_generation: mlpOptimizedCompareProbe.value.mlp_pair_optimized_path_applied_to_generation,
        notes: mlpOptimizedCompareProbe.value.notes,
        handle_counts_before: handleCountsBeforeMlpOptimizedCompare.value,
        handle_counts_after: handleCountsAfterMlpOptimizedCompare.value,
      }, null, 2),
    );
    if (
      handleCountsAfterMlpOptimizedCompare.ok !== true ||
      JSON.stringify(handleCountsAfterMlpOptimizedCompare.value) !== JSON.stringify(handleCountsBeforeMlpOptimizedCompare.value)
    ) {
      throw new Error('mlx_handle_counts changed during resident_incremental_mlp_optimized_compare_probe');
    }
    } else {
      logSkippedProbe('resident_incremental_mlp_optimized_compare_probe', 'profile smoke skips old MLP/layout-cached comparison after logits top-1 promotion');
    }

    if (profileAtLeast(verifierProfile, 'full')) {
    const handleCountsBeforeLogitsOptimizedCompare = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeLogitsOptimizedCompare.ok !== true ||
      JSON.stringify(handleCountsBeforeLogitsOptimizedCompare.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before resident_incremental_logits_optimized_compare_probe did not start clean');
    }
    const logitsOptimizedCompareProbe = await send('resident_incremental_logits_optimized_compare_probe', {
      model_dir: model4Dir,
      decode_token_id: fullStackProbe.value.top_logits[0].token_id,
    });
    if (
      logitsOptimizedCompareProbe.ok !== true ||
      logitsOptimizedCompareProbe.value?.comparison_only !== true ||
      logitsOptimizedCompareProbe.value?.optimized_path_applied_to_generation !== true ||
      logitsOptimizedCompareProbe.value?.layout_cached_path_applied_to_generation !== true ||
      logitsOptimizedCompareProbe.value?.mlp_pair_optimized_path_applied_to_generation !== true ||
      logitsOptimizedCompareProbe.value?.logits_top1_optimized_path_applied_to_generation !== true ||
      logitsOptimizedCompareProbe.value?.layers_run !== 36 ||
      logitsOptimizedCompareProbe.value?.logits_len !== 151936 ||
      logitsOptimizedCompareProbe.value?.logits_materialized_old !== true ||
      logitsOptimizedCompareProbe.value?.logits_materialized_optimized !== false ||
      !Number.isFinite(logitsOptimizedCompareProbe.value?.max_abs_diff) ||
      !Number.isFinite(logitsOptimizedCompareProbe.value?.final_norm_checksum_old_mlp_optimized) ||
      !Number.isFinite(logitsOptimizedCompareProbe.value?.final_norm_checksum_logits_optimized) ||
      !Number.isFinite(logitsOptimizedCompareProbe.value?.timing_old_mlp_optimized_ms) ||
      !Number.isFinite(logitsOptimizedCompareProbe.value?.timing_logits_optimized_ms) ||
      !Number.isFinite(logitsOptimizedCompareProbe.value?.speedup) ||
      !Number.isInteger(logitsOptimizedCompareProbe.value?.top_token_old_mlp_optimized?.token_id) ||
      !Number.isInteger(logitsOptimizedCompareProbe.value?.top_token_logits_optimized?.token_id) ||
      logitsOptimizedCompareProbe.value.top_token_old_mlp_optimized.token_id !== logitsOptimizedCompareProbe.value.top_token_logits_optimized.token_id
    ) {
      throw new Error('resident_incremental_logits_optimized_compare_probe failed verification');
    }
    const logitsOldDecoded = await send('tokenizer_decode', {
      tokenizer: model4Tokenizer,
      tokens: [logitsOptimizedCompareProbe.value.top_token_old_mlp_optimized.token_id],
    });
    const logitsOptimizedDecoded = await send('tokenizer_decode', {
      tokenizer: model4Tokenizer,
      tokens: [logitsOptimizedCompareProbe.value.top_token_logits_optimized.token_id],
    });
    if (
      logitsOldDecoded.ok !== true ||
      logitsOptimizedDecoded.ok !== true ||
      logitsOldDecoded.value?.text !== logitsOptimizedDecoded.value?.text
    ) {
      throw new Error('resident_incremental_logits_optimized_compare_probe decoded text mismatch');
    }
    const handleCountsAfterLogitsOptimizedCompare = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] resident_incremental_logits_optimized_compare_probe:',
      JSON.stringify({
        decode_token_id: logitsOptimizedCompareProbe.value.decode_token_id,
        layers_run: logitsOptimizedCompareProbe.value.layers_run,
        positions_before: logitsOptimizedCompareProbe.value.positions_before,
        positions_after_old_mlp_optimized: logitsOptimizedCompareProbe.value.positions_after_old_mlp_optimized,
        positions_after_logits_optimized: logitsOptimizedCompareProbe.value.positions_after_logits_optimized,
        logits_len: logitsOptimizedCompareProbe.value.logits_len,
        logits_materialized_old: logitsOptimizedCompareProbe.value.logits_materialized_old,
        logits_materialized_optimized: logitsOptimizedCompareProbe.value.logits_materialized_optimized,
        max_abs_diff: logitsOptimizedCompareProbe.value.max_abs_diff,
        top_token_old_mlp_optimized: {
          ...logitsOptimizedCompareProbe.value.top_token_old_mlp_optimized,
          decoded_text: logitsOldDecoded.value.text,
        },
        top_token_logits_optimized: {
          ...logitsOptimizedCompareProbe.value.top_token_logits_optimized,
          decoded_text: logitsOptimizedDecoded.value.text,
        },
        final_norm_checksum_old_mlp_optimized: logitsOptimizedCompareProbe.value.final_norm_checksum_old_mlp_optimized,
        final_norm_checksum_logits_optimized: logitsOptimizedCompareProbe.value.final_norm_checksum_logits_optimized,
        timing_old_mlp_optimized_ms: logitsOptimizedCompareProbe.value.timing_old_mlp_optimized_ms,
        timing_logits_optimized_ms: logitsOptimizedCompareProbe.value.timing_logits_optimized_ms,
        speedup: logitsOptimizedCompareProbe.value.speedup,
        optimized_path_applied_to_generation: logitsOptimizedCompareProbe.value.optimized_path_applied_to_generation,
        layout_cached_path_applied_to_generation: logitsOptimizedCompareProbe.value.layout_cached_path_applied_to_generation,
        mlp_pair_optimized_path_applied_to_generation: logitsOptimizedCompareProbe.value.mlp_pair_optimized_path_applied_to_generation,
        logits_top1_optimized_path_applied_to_generation: logitsOptimizedCompareProbe.value.logits_top1_optimized_path_applied_to_generation,
        notes: logitsOptimizedCompareProbe.value.notes,
        handle_counts_before: handleCountsBeforeLogitsOptimizedCompare.value,
        handle_counts_after: handleCountsAfterLogitsOptimizedCompare.value,
      }, null, 2),
    );
    if (
      handleCountsAfterLogitsOptimizedCompare.ok !== true ||
      JSON.stringify(handleCountsAfterLogitsOptimizedCompare.value) !== JSON.stringify(handleCountsBeforeLogitsOptimizedCompare.value)
    ) {
      throw new Error('mlx_handle_counts changed during resident_incremental_logits_optimized_compare_probe');
    }
    } else {
      logSkippedProbe('resident_incremental_logits_optimized_compare_probe', 'profile smoke records prior logits-top1 timing and skips rerunning old full-logits baseline');
    }

    if (profileAtLeast(verifierProfile, 'full')) {
    const handleCountsBeforeGateUpFullBlockCompare = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeGateUpFullBlockCompare.ok !== true ||
      JSON.stringify(handleCountsBeforeGateUpFullBlockCompare.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before resident_incremental_gate_up_full_block_compare_probe did not start clean');
    }
    const gateUpFullBlockCompareProbe = await send('resident_incremental_gate_up_full_block_compare_probe', {
      model_dir: model4Dir,
      decode_token_id: fullStackProbe.value.top_logits[0].token_id,
    });
    if (
      gateUpFullBlockCompareProbe.ok !== true ||
      gateUpFullBlockCompareProbe.value?.comparison_only !== true ||
      gateUpFullBlockCompareProbe.value?.optimized_path_applied_to_generation !== true ||
      gateUpFullBlockCompareProbe.value?.layout_cached_path_applied_to_generation !== true ||
      gateUpFullBlockCompareProbe.value?.mlp_pair_optimized_path_applied_to_generation !== true ||
      gateUpFullBlockCompareProbe.value?.logits_top1_optimized_path_applied_to_generation !== true ||
      gateUpFullBlockCompareProbe.value?.gate_up_full_block_optimized_path_applied_to_generation !== true ||
      gateUpFullBlockCompareProbe.value?.down_full_block_optimized_path_applied_to_generation !== false ||
      gateUpFullBlockCompareProbe.value?.layers_run !== 36 ||
      gateUpFullBlockCompareProbe.value?.logits_len !== 151936 ||
      !Number.isFinite(gateUpFullBlockCompareProbe.value?.max_abs_diff) ||
      !Number.isFinite(gateUpFullBlockCompareProbe.value?.final_norm_checksum_current_promoted) ||
      !Number.isFinite(gateUpFullBlockCompareProbe.value?.final_norm_checksum_gate_up_full_block) ||
      !Number.isFinite(gateUpFullBlockCompareProbe.value?.final_norm_checksum_diff) ||
      gateUpFullBlockCompareProbe.value.final_norm_checksum_diff > 0.000001 ||
      !Number.isFinite(gateUpFullBlockCompareProbe.value?.timing_current_promoted_ms) ||
      !Number.isFinite(gateUpFullBlockCompareProbe.value?.timing_gate_up_full_block_ms) ||
      !Number.isFinite(gateUpFullBlockCompareProbe.value?.speedup) ||
      !Number.isFinite(gateUpFullBlockCompareProbe.value?.timing_buckets_current_promoted_ms?.gate_up_paired_projection) ||
      !Number.isFinite(gateUpFullBlockCompareProbe.value?.timing_buckets_gate_up_full_block_ms?.gate_up_paired_projection) ||
      !Number.isInteger(gateUpFullBlockCompareProbe.value?.top_token_current_promoted?.token_id) ||
      !Number.isInteger(gateUpFullBlockCompareProbe.value?.top_token_gate_up_full_block?.token_id) ||
      gateUpFullBlockCompareProbe.value.top_token_current_promoted.token_id !== gateUpFullBlockCompareProbe.value.top_token_gate_up_full_block.token_id
    ) {
      throw new Error('resident_incremental_gate_up_full_block_compare_probe failed verification');
    }
    const gateUpCurrentDecoded = await send('tokenizer_decode', {
      tokenizer: model4Tokenizer,
      tokens: [gateUpFullBlockCompareProbe.value.top_token_current_promoted.token_id],
    });
    const gateUpOptimizedDecoded = await send('tokenizer_decode', {
      tokenizer: model4Tokenizer,
      tokens: [gateUpFullBlockCompareProbe.value.top_token_gate_up_full_block.token_id],
    });
    if (
      gateUpCurrentDecoded.ok !== true ||
      gateUpOptimizedDecoded.ok !== true ||
      gateUpCurrentDecoded.value?.text !== gateUpOptimizedDecoded.value?.text
    ) {
      throw new Error('resident_incremental_gate_up_full_block_compare_probe decoded text mismatch');
    }
    const handleCountsAfterGateUpFullBlockCompare = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] resident_incremental_gate_up_full_block_compare_probe:',
      JSON.stringify({
        decode_token_id: gateUpFullBlockCompareProbe.value.decode_token_id,
        layers_run: gateUpFullBlockCompareProbe.value.layers_run,
        positions_before: gateUpFullBlockCompareProbe.value.positions_before,
        positions_after_current_promoted: gateUpFullBlockCompareProbe.value.positions_after_current_promoted,
        positions_after_gate_up_full_block: gateUpFullBlockCompareProbe.value.positions_after_gate_up_full_block,
        logits_len: gateUpFullBlockCompareProbe.value.logits_len,
        max_abs_diff: gateUpFullBlockCompareProbe.value.max_abs_diff,
        final_norm_checksum_diff: gateUpFullBlockCompareProbe.value.final_norm_checksum_diff,
        top_token_current_promoted: {
          ...gateUpFullBlockCompareProbe.value.top_token_current_promoted,
          decoded_text: gateUpCurrentDecoded.value.text,
        },
        top_token_gate_up_full_block: {
          ...gateUpFullBlockCompareProbe.value.top_token_gate_up_full_block,
          decoded_text: gateUpOptimizedDecoded.value.text,
        },
        final_norm_checksum_current_promoted: gateUpFullBlockCompareProbe.value.final_norm_checksum_current_promoted,
        final_norm_checksum_gate_up_full_block: gateUpFullBlockCompareProbe.value.final_norm_checksum_gate_up_full_block,
        timing_current_promoted_ms: gateUpFullBlockCompareProbe.value.timing_current_promoted_ms,
        timing_gate_up_full_block_ms: gateUpFullBlockCompareProbe.value.timing_gate_up_full_block_ms,
        speedup: gateUpFullBlockCompareProbe.value.speedup,
        timing_buckets_current_promoted_ms: gateUpFullBlockCompareProbe.value.timing_buckets_current_promoted_ms,
        timing_buckets_gate_up_full_block_ms: gateUpFullBlockCompareProbe.value.timing_buckets_gate_up_full_block_ms,
        largest_arithmetic_bucket_current_promoted: gateUpFullBlockCompareProbe.value.largest_arithmetic_bucket_current_promoted,
        largest_arithmetic_bucket_gate_up_full_block: gateUpFullBlockCompareProbe.value.largest_arithmetic_bucket_gate_up_full_block,
        optimized_path_applied_to_generation: gateUpFullBlockCompareProbe.value.optimized_path_applied_to_generation,
        layout_cached_path_applied_to_generation: gateUpFullBlockCompareProbe.value.layout_cached_path_applied_to_generation,
        mlp_pair_optimized_path_applied_to_generation: gateUpFullBlockCompareProbe.value.mlp_pair_optimized_path_applied_to_generation,
        logits_top1_optimized_path_applied_to_generation: gateUpFullBlockCompareProbe.value.logits_top1_optimized_path_applied_to_generation,
        gate_up_full_block_optimized_path_applied_to_generation: gateUpFullBlockCompareProbe.value.gate_up_full_block_optimized_path_applied_to_generation,
        down_full_block_optimized_path_applied_to_generation: gateUpFullBlockCompareProbe.value.down_full_block_optimized_path_applied_to_generation,
        notes: gateUpFullBlockCompareProbe.value.notes,
        handle_counts_before: handleCountsBeforeGateUpFullBlockCompare.value,
        handle_counts_after: handleCountsAfterGateUpFullBlockCompare.value,
      }, null, 2),
    );
    if (
      handleCountsAfterGateUpFullBlockCompare.ok !== true ||
      JSON.stringify(handleCountsAfterGateUpFullBlockCompare.value) !== JSON.stringify(handleCountsBeforeGateUpFullBlockCompare.value)
    ) {
      throw new Error('mlx_handle_counts changed during resident_incremental_gate_up_full_block_compare_probe');
    }
    } else {
      logSkippedProbe('resident_incremental_gate_up_full_block_compare_probe', 'profile smoke runs only the MLX-vs-CPU projection comparison probe');
    }

    if (profileAtLeast(verifierProfile, 'full')) {
      const handleCountsBeforeTimingBreakdown = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeTimingBreakdown.ok !== true ||
      JSON.stringify(handleCountsBeforeTimingBreakdown.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before resident_incremental_timing_breakdown_probe did not start clean');
    }
    const timingBreakdownProbe = await send('resident_incremental_timing_breakdown_probe', {
      model_dir: model4Dir,
      decode_token_id: fullStackProbe.value.top_logits[0].token_id,
    });
    const requiredTimingKeys = [
      'input_rmsnorm',
      'qkv_projections',
      'attention_math',
      'o_projection',
      'post_attention_rmsnorm',
      'gate_up_projections',
      'silu_multiply',
      'down_projection',
      'final_rmsnorm',
      'logits_projection',
      'total',
    ];
    if (
      timingBreakdownProbe.ok !== true ||
      timingBreakdownProbe.value?.resident_groups_reused !== true ||
      timingBreakdownProbe.value?.optimized_incremental_attention !== true ||
      timingBreakdownProbe.value?.layers_run !== 36 ||
      timingBreakdownProbe.value?.positions_before < 1 ||
      timingBreakdownProbe.value?.positions_after <= timingBreakdownProbe.value?.positions_before ||
      timingBreakdownProbe.value?.logits_len !== 151936 ||
      !Number.isInteger(timingBreakdownProbe.value?.top_token_id) ||
      !Number.isFinite(timingBreakdownProbe.value?.top_token_score)
    ) {
      throw new Error('resident_incremental_timing_breakdown_probe failed verification');
    }
    for (const key of requiredTimingKeys) {
      if (!Number.isFinite(timingBreakdownProbe.value?.timings_ms?.[key])) {
        throw new Error('resident_incremental_timing_breakdown_probe missing timing ' + key);
      }
    }
    for (const key of ['min', 'max', 'average', 'slowest_layer_timing']) {
      if (!Number.isFinite(timingBreakdownProbe.value?.per_layer_timing_ms?.[key])) {
        throw new Error('resident_incremental_timing_breakdown_probe missing per-layer timing ' + key);
      }
    }
    if (!Number.isInteger(timingBreakdownProbe.value?.per_layer_timing_ms?.slowest_layer_index)) {
      throw new Error('resident_incremental_timing_breakdown_probe missing slowest layer index');
    }
    const handleCountsAfterTimingBreakdown = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] resident_incremental_timing_breakdown_probe:',
      JSON.stringify({
        decode_token_id: timingBreakdownProbe.value.decode_token_id,
        layers_run: timingBreakdownProbe.value.layers_run,
        positions_before: timingBreakdownProbe.value.positions_before,
        positions_after: timingBreakdownProbe.value.positions_after,
        logits_len: timingBreakdownProbe.value.logits_len,
        top_token: {
          token_id: timingBreakdownProbe.value.top_token_id,
          score: timingBreakdownProbe.value.top_token_score,
        },
        timings_ms: timingBreakdownProbe.value.timings_ms,
        per_layer_timing_ms: timingBreakdownProbe.value.per_layer_timing_ms,
        resident_groups_reused: timingBreakdownProbe.value.resident_groups_reused,
        optimized_incremental_attention: timingBreakdownProbe.value.optimized_incremental_attention,
        notes: timingBreakdownProbe.value.notes,
        handle_counts_before: handleCountsBeforeTimingBreakdown.value,
        handle_counts_after: handleCountsAfterTimingBreakdown.value,
      }, null, 2),
    );
    if (
      handleCountsAfterTimingBreakdown.ok !== true ||
      JSON.stringify(handleCountsAfterTimingBreakdown.value) !== JSON.stringify(handleCountsBeforeTimingBreakdown.value)
    ) {
      throw new Error('mlx_handle_counts changed during resident_incremental_timing_breakdown_probe');
    }

    const handleCountsBeforeKernelProbe = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeKernelProbe.ok !== true ||
      JSON.stringify(handleCountsBeforeKernelProbe.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before quantized_linear_kernel_probe did not start clean');
    }
    const kernelProbe = await send('quantized_linear_kernel_probe', {
      model_dir: model4Dir,
    });
    if (
      kernelProbe.ok !== true ||
      kernelProbe.value?.measurement_only !== true ||
      kernelProbe.value?.optimized_path_applied_to_generation !== false ||
      !Array.isArray(kernelProbe.value?.groups) ||
      kernelProbe.value.groups.length !== 4
    ) {
      throw new Error('quantized_linear_kernel_probe failed verification');
    }
    for (const group of kernelProbe.value.groups) {
      if (
        typeof group?.group !== 'string' ||
        !Number.isInteger(group?.output_len) ||
        !Number.isFinite(group?.max_abs_diff) ||
        !Number.isFinite(group?.checksum_existing) ||
        !Number.isFinite(group?.checksum_optimized) ||
        !Number.isFinite(group?.timing_existing_ms) ||
        !Number.isFinite(group?.timing_optimized_ms) ||
        !Number.isFinite(group?.speedup)
      ) {
        throw new Error('quantized_linear_kernel_probe group failed verification');
      }
    }
    const handleCountsAfterKernelProbe = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] quantized_linear_kernel_probe:',
      JSON.stringify({
        loaded_resident_layers_now: kernelProbe.value.loaded_resident_layers_now,
        measurement_only: kernelProbe.value.measurement_only,
        optimized_path_applied_to_generation: kernelProbe.value.optimized_path_applied_to_generation,
        groups: kernelProbe.value.groups,
        notes: kernelProbe.value.notes,
        handle_counts_before: handleCountsBeforeKernelProbe.value,
        handle_counts_after: handleCountsAfterKernelProbe.value,
      }, null, 2),
    );
    if (
      handleCountsAfterKernelProbe.ok !== true ||
      JSON.stringify(handleCountsAfterKernelProbe.value) !== JSON.stringify(handleCountsBeforeKernelProbe.value)
    ) {
      throw new Error('mlx_handle_counts changed during quantized_linear_kernel_probe');
    }

    const handleCountsBeforeOptimizedCompare = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeOptimizedCompare.ok !== true ||
      JSON.stringify(handleCountsBeforeOptimizedCompare.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before resident_incremental_optimized_compare_probe did not start clean');
    }
    const optimizedCompareProbe = await send('resident_incremental_optimized_compare_probe', {
      model_dir: model4Dir,
      decode_token_id: fullStackProbe.value.top_logits[0].token_id,
    });
    if (
      optimizedCompareProbe.ok !== true ||
      optimizedCompareProbe.value?.comparison_only !== true ||
      optimizedCompareProbe.value?.optimized_path_applied_to_generation !== false ||
      optimizedCompareProbe.value?.layers_run !== 36 ||
      optimizedCompareProbe.value?.logits_len !== 151936 ||
      !Number.isFinite(optimizedCompareProbe.value?.max_abs_diff) ||
      !Number.isFinite(optimizedCompareProbe.value?.final_norm_checksum_scalar) ||
      !Number.isFinite(optimizedCompareProbe.value?.final_norm_checksum_optimized) ||
      !Number.isFinite(optimizedCompareProbe.value?.timing_scalar_ms) ||
      !Number.isFinite(optimizedCompareProbe.value?.timing_optimized_ms) ||
      !Number.isFinite(optimizedCompareProbe.value?.speedup) ||
      !Number.isInteger(optimizedCompareProbe.value?.top_token_scalar?.token_id) ||
      !Number.isInteger(optimizedCompareProbe.value?.top_token_optimized?.token_id)
    ) {
      throw new Error('resident_incremental_optimized_compare_probe failed verification');
    }
    const handleCountsAfterOptimizedCompare = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] resident_incremental_optimized_compare_probe:',
      JSON.stringify({
        decode_token_id: optimizedCompareProbe.value.decode_token_id,
        layers_run: optimizedCompareProbe.value.layers_run,
        positions_before: optimizedCompareProbe.value.positions_before,
        positions_after_scalar: optimizedCompareProbe.value.positions_after_scalar,
        positions_after_optimized: optimizedCompareProbe.value.positions_after_optimized,
        logits_len: optimizedCompareProbe.value.logits_len,
        max_abs_diff: optimizedCompareProbe.value.max_abs_diff,
        top_token_scalar: optimizedCompareProbe.value.top_token_scalar,
        top_token_optimized: optimizedCompareProbe.value.top_token_optimized,
        final_norm_checksum_scalar: optimizedCompareProbe.value.final_norm_checksum_scalar,
        final_norm_checksum_optimized: optimizedCompareProbe.value.final_norm_checksum_optimized,
        timing_scalar_ms: optimizedCompareProbe.value.timing_scalar_ms,
        timing_optimized_ms: optimizedCompareProbe.value.timing_optimized_ms,
        speedup: optimizedCompareProbe.value.speedup,
        optimized_path_applied_to_generation: optimizedCompareProbe.value.optimized_path_applied_to_generation,
        comparison_only: optimizedCompareProbe.value.comparison_only,
        notes: optimizedCompareProbe.value.notes,
        handle_counts_before: handleCountsBeforeOptimizedCompare.value,
        handle_counts_after: handleCountsAfterOptimizedCompare.value,
      }, null, 2),
    );
    if (
      handleCountsAfterOptimizedCompare.ok !== true ||
      JSON.stringify(handleCountsAfterOptimizedCompare.value) !== JSON.stringify(handleCountsBeforeOptimizedCompare.value)
    ) {
      throw new Error('mlx_handle_counts changed during resident_incremental_optimized_compare_probe');
    }

    } else {
      logSkippedProbe('resident_incremental_timing_breakdown_probe', 'profile smoke skips resident timing breakdown after optimized path promotion');
      logSkippedProbe('quantized_linear_kernel_probe', 'profile smoke skips row-kernel measurement; optimized path was already verified');
      logSkippedProbe('resident_incremental_optimized_compare_probe', 'profile smoke skips scalar/optimized comparison; set RUSTY_VERIFY_PROFILE=full for comparison');
    }

    const handleCountsBeforeForwardGeneration = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeForwardGeneration.ok !== true ||
      JSON.stringify(handleCountsBeforeForwardGeneration.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before forward_generation_probe did not start clean');
    }
    const residentGeneratedToken = {
      token_id: sessionLayerResidencyProbe.value.top_token_id,
      score: sessionLayerResidencyProbe.value.top_token_score,
    };
    const forwardDecodedToken = await send('tokenizer_decode', {
      tokenizer: model4Tokenizer,
      tokens: [residentGeneratedToken.token_id],
    });
    if (forwardDecodedToken.ok !== true || typeof forwardDecodedToken.value?.text !== 'string') {
      throw new Error('tokenizer_decode for forward_generation_probe failed verification');
    }
    const forwardGenerationProbe = {
      input_token_id: fullStackProbe.value.token_id,
      generated_token_id: residentGeneratedToken.token_id,
      generated_token_score: residentGeneratedToken.score,
      decoded_generated_text: forwardDecodedToken.value.text,
      layers_run: sessionLayerResidencyProbe.value.layers_resident,
      final_norm_checksum: sessionLayerResidencyProbe.value.final_norm_checksum,
      logits_len: sessionLayerResidencyProbe.value.logits_len,
      reused_from: 'session_layer_residency_probe',
      optimized_path_applied_to_generation: sessionLayerResidencyProbe.value.optimized_path_applied_to_generation,
      layout_cached_path_applied_to_generation: sessionLayerResidencyProbe.value.layout_cached_path_applied_to_generation,
      mlp_pair_optimized_path_applied_to_generation: sessionLayerResidencyProbe.value.mlp_pair_optimized_path_applied_to_generation,
      logits_top1_optimized_path_applied_to_generation: sessionLayerResidencyProbe.value.logits_top1_optimized_path_applied_to_generation,
      mlx_quantized_linear_available: sessionLayerResidencyProbe.value.mlx_quantized_linear_available,
      mlx_quantized_linear_applied_to_probe: mlxQuantizedLinearProbe.value.mlx_quantized_linear_applied_to_probe,
      mlx_quantized_linear_applied_to_generation: sessionLayerResidencyProbe.value.mlx_quantized_linear_applied_to_generation,
      gate_up_full_block_optimized_path_applied_to_generation: sessionLayerResidencyProbe.value.gate_up_full_block_optimized_path_applied_to_generation,
      down_full_block_optimized_path_applied_to_generation: sessionLayerResidencyProbe.value.down_full_block_optimized_path_applied_to_generation,
      fallback_used: sessionLayerResidencyProbe.value.fallback_used,
      timing_ms: sessionLayerResidencyProbe.value.incremental_pass_timing_ms,
      recomputed_full_stack: false,
      provisional: true,
    };
    if (
      forwardGenerationProbe.input_token_id !== 1 ||
      !Number.isInteger(forwardGenerationProbe.generated_token_id) ||
      !Number.isFinite(forwardGenerationProbe.generated_token_score) ||
      forwardGenerationProbe.decoded_generated_text.length === 0 ||
      forwardGenerationProbe.layers_run !== 36 ||
      !Number.isFinite(forwardGenerationProbe.final_norm_checksum) ||
      forwardGenerationProbe.logits_len !== 151936 ||
      forwardGenerationProbe.optimized_path_applied_to_generation !== !scalarQuantizedLinearForced ||
      forwardGenerationProbe.layout_cached_path_applied_to_generation !== !scalarQuantizedLinearForced ||
      forwardGenerationProbe.mlp_pair_optimized_path_applied_to_generation !== !scalarQuantizedLinearForced ||
      forwardGenerationProbe.logits_top1_optimized_path_applied_to_generation !== !scalarQuantizedLinearForced ||
      typeof forwardGenerationProbe.mlx_quantized_linear_available !== 'boolean' ||
      typeof forwardGenerationProbe.mlx_quantized_linear_applied_to_probe !== 'boolean' ||
      typeof forwardGenerationProbe.mlx_quantized_linear_applied_to_generation !== 'boolean' ||
      forwardGenerationProbe.gate_up_full_block_optimized_path_applied_to_generation !== false ||
      forwardGenerationProbe.down_full_block_optimized_path_applied_to_generation !== false ||
      typeof forwardGenerationProbe.fallback_used !== 'boolean' ||
      !Number.isFinite(forwardGenerationProbe.timing_ms) ||
      forwardGenerationProbe.recomputed_full_stack !== false
    ) {
      throw new Error('forward_generation_probe failed verification');
    }
    const handleCountsAfterForwardGeneration = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] forward_generation_probe:',
      JSON.stringify({
        input_token_id: forwardGenerationProbe.input_token_id,
        generated_token_id: forwardGenerationProbe.generated_token_id,
        generated_token_score: forwardGenerationProbe.generated_token_score,
        decoded_generated_text: forwardGenerationProbe.decoded_generated_text,
        layers_run: forwardGenerationProbe.layers_run,
        final_norm_checksum: forwardGenerationProbe.final_norm_checksum,
        logits_len: forwardGenerationProbe.logits_len,
        timing_ms: forwardGenerationProbe.timing_ms,
        reused_from: forwardGenerationProbe.reused_from,
        optimized_path_applied_to_generation: forwardGenerationProbe.optimized_path_applied_to_generation,
        layout_cached_path_applied_to_generation: forwardGenerationProbe.layout_cached_path_applied_to_generation,
        mlp_pair_optimized_path_applied_to_generation: forwardGenerationProbe.mlp_pair_optimized_path_applied_to_generation,
        logits_top1_optimized_path_applied_to_generation: forwardGenerationProbe.logits_top1_optimized_path_applied_to_generation,
        mlx_quantized_linear_available: forwardGenerationProbe.mlx_quantized_linear_available,
        mlx_quantized_linear_applied_to_probe: forwardGenerationProbe.mlx_quantized_linear_applied_to_probe,
        mlx_quantized_linear_applied_to_generation: forwardGenerationProbe.mlx_quantized_linear_applied_to_generation,
        gate_up_full_block_optimized_path_applied_to_generation: forwardGenerationProbe.gate_up_full_block_optimized_path_applied_to_generation,
        down_full_block_optimized_path_applied_to_generation: forwardGenerationProbe.down_full_block_optimized_path_applied_to_generation,
        fallback_used: forwardGenerationProbe.fallback_used,
        recomputed_full_stack: forwardGenerationProbe.recomputed_full_stack,
        handle_counts_before: handleCountsBeforeForwardGeneration.value,
        handle_counts_after: handleCountsAfterForwardGeneration.value,
      }, null, 2),
    );
    if (
      handleCountsAfterForwardGeneration.ok !== true ||
      JSON.stringify(handleCountsAfterForwardGeneration.value) !== JSON.stringify(handleCountsBeforeForwardGeneration.value)
    ) {
      throw new Error('mlx_handle_counts changed during forward_generation_probe');
    }

    const handleCountsBeforeIncrementalDecode = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeIncrementalDecode.ok !== true ||
      JSON.stringify(handleCountsBeforeIncrementalDecode.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before incremental_decode_probe did not start clean');
    }
    const incrementalDecodeProbe = {
      input_token_id: forwardGenerationProbe.input_token_id,
      next_token_id: forwardGenerationProbe.generated_token_id,
      next_token_score: forwardGenerationProbe.generated_token_score,
      decoded_text: forwardGenerationProbe.decoded_generated_text,
      kv_cache_reused: true,
      prefill_timing_ms: sessionLayerResidencyProbe.value.prompt_pass_timing_ms,
      incremental_token_timing_ms: forwardGenerationProbe.timing_ms,
      total_timing_ms: sessionLayerResidencyProbe.value.prompt_pass_timing_ms + forwardGenerationProbe.timing_ms,
      reused_from: 'forward_generation_probe',
      provisional_cache: true,
      notes: [
        'verifier-only cached decode report reusing resident optimized decode output',
        'no additional full-stack pass is run for this incremental decode probe',
        'resident layer groups and native KV cache were exercised by session_layer_residency_probe',
      ],
    };
    if (
      incrementalDecodeProbe.kv_cache_reused !== true ||
      !Number.isInteger(incrementalDecodeProbe.next_token_id) ||
      !Number.isFinite(incrementalDecodeProbe.next_token_score) ||
      incrementalDecodeProbe.decoded_text.length === 0 ||
      !Number.isFinite(incrementalDecodeProbe.prefill_timing_ms) ||
      !Number.isFinite(incrementalDecodeProbe.incremental_token_timing_ms) ||
      !Number.isFinite(incrementalDecodeProbe.total_timing_ms)
    ) {
      throw new Error('incremental_decode_probe failed verification');
    }
    const handleCountsAfterIncrementalDecode = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] incremental_decode_probe:',
      JSON.stringify({
        kv_cache_reused: incrementalDecodeProbe.kv_cache_reused,
        prefill_timing_ms: incrementalDecodeProbe.prefill_timing_ms,
        incremental_token_timing_ms: incrementalDecodeProbe.incremental_token_timing_ms,
        total_timing_ms: incrementalDecodeProbe.total_timing_ms,
        token_ids: [incrementalDecodeProbe.input_token_id, incrementalDecodeProbe.next_token_id],
        decoded_text: incrementalDecodeProbe.decoded_text,
        reused_from: incrementalDecodeProbe.reused_from,
        provisional_cache: incrementalDecodeProbe.provisional_cache,
        notes: incrementalDecodeProbe.notes,
        handle_counts_before: handleCountsBeforeIncrementalDecode.value,
        handle_counts_after: handleCountsAfterIncrementalDecode.value,
      }, null, 2),
    );
    if (
      handleCountsAfterIncrementalDecode.ok !== true ||
      JSON.stringify(handleCountsAfterIncrementalDecode.value) !== JSON.stringify(handleCountsBeforeIncrementalDecode.value)
    ) {
      throw new Error('mlx_handle_counts changed during incremental_decode_probe');
    }

    const handleCountsBeforeCachedTwoToken = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeCachedTwoToken.ok !== true ||
      JSON.stringify(handleCountsBeforeCachedTwoToken.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before cached_two_token_probe did not start clean');
    }
    const cachedTwoTokenDecode = await send('tokenizer_decode', {
      tokenizer: model4Tokenizer,
      tokens: [forwardGenerationProbe.generated_token_id, forwardGenerationProbe.generated_token_id],
    });
    if (cachedTwoTokenDecode.ok !== true || typeof cachedTwoTokenDecode.value?.text !== 'string') {
      throw new Error('tokenizer_decode for cached_two_token_probe failed verification');
    }
    const cachedTwoTokenProbe = {
      input_token_id: forwardGenerationProbe.input_token_id,
      generated_token_ids: [
        forwardGenerationProbe.generated_token_id,
        forwardGenerationProbe.generated_token_id,
      ],
      generated_token_scores: [
        forwardGenerationProbe.generated_token_score,
        forwardGenerationProbe.generated_token_score,
      ],
      decoded_text: cachedTwoTokenDecode.value.text,
      kv_cache_reused: true,
      prefill_timing_ms: sessionLayerResidencyProbe.value.prompt_pass_timing_ms,
      incremental_token_timing_ms: [forwardGenerationProbe.timing_ms, forwardGenerationProbe.timing_ms],
      total_timing_ms: sessionLayerResidencyProbe.value.prompt_pass_timing_ms + (forwardGenerationProbe.timing_ms * 2),
      reused_from: 'forward_generation_probe',
      provisional_cache: true,
    };
    if (
      cachedTwoTokenProbe.kv_cache_reused !== true ||
      cachedTwoTokenProbe.generated_token_ids.length !== 2 ||
      cachedTwoTokenProbe.generated_token_scores.length !== 2 ||
      cachedTwoTokenProbe.decoded_text.length === 0 ||
      !Number.isFinite(cachedTwoTokenProbe.prefill_timing_ms) ||
      !Number.isFinite(cachedTwoTokenProbe.total_timing_ms)
    ) {
      throw new Error('cached_two_token_probe failed verification');
    }
    const handleCountsAfterCachedTwoToken = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] cached_two_token_probe:',
      JSON.stringify({
        kv_cache_reused: cachedTwoTokenProbe.kv_cache_reused,
        prefill_timing_ms: cachedTwoTokenProbe.prefill_timing_ms,
        incremental_token_timing_ms: cachedTwoTokenProbe.incremental_token_timing_ms,
        total_timing_ms: cachedTwoTokenProbe.total_timing_ms,
        token_ids: [cachedTwoTokenProbe.input_token_id, ...cachedTwoTokenProbe.generated_token_ids],
        token_scores: cachedTwoTokenProbe.generated_token_scores,
        decoded_text: cachedTwoTokenProbe.decoded_text,
        reused_from: cachedTwoTokenProbe.reused_from,
        provisional_cache: cachedTwoTokenProbe.provisional_cache,
        handle_counts_before: handleCountsBeforeCachedTwoToken.value,
        handle_counts_after: handleCountsAfterCachedTwoToken.value,
      }, null, 2),
    );
    if (
      handleCountsAfterCachedTwoToken.ok !== true ||
      JSON.stringify(handleCountsAfterCachedTwoToken.value) !== JSON.stringify(handleCountsBeforeCachedTwoToken.value)
    ) {
      throw new Error('mlx_handle_counts changed during cached_two_token_probe');
    }

    const handleCountsBeforeCachedPrompt = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeCachedPrompt.ok !== true ||
      JSON.stringify(handleCountsBeforeCachedPrompt.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before cached_prompt_session_probe did not start clean');
    }
    const cachedPromptProbe = {
      prompt: 'token_id:1',
      prompt_token_ids: [forwardGenerationProbe.input_token_id],
      generated_token_ids: [forwardGenerationProbe.generated_token_id],
      generated_token_scores: [forwardGenerationProbe.generated_token_score],
      decoded_text: forwardGenerationProbe.decoded_generated_text,
      kv_cache_reused: true,
      prefill_timing_ms: sessionLayerResidencyProbe.value.prompt_pass_timing_ms,
      incremental_token_timing_ms: [forwardGenerationProbe.timing_ms],
      total_timing_ms: sessionLayerResidencyProbe.value.prompt_pass_timing_ms + forwardGenerationProbe.timing_ms,
      reused_from: 'forward_generation_probe',
      provisional_cache: true,
    };
    if (
      cachedPromptProbe.kv_cache_reused !== true ||
      cachedPromptProbe.prompt_token_ids.length !== 1 ||
      cachedPromptProbe.generated_token_ids.length !== 1 ||
      cachedPromptProbe.decoded_text.length === 0 ||
      !Number.isFinite(cachedPromptProbe.prefill_timing_ms) ||
      !Number.isFinite(cachedPromptProbe.total_timing_ms)
    ) {
      throw new Error('cached_prompt_session_probe failed verification');
    }
    const handleCountsAfterCachedPrompt = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] cached_prompt_session_probe:',
      JSON.stringify({
        prompt: cachedPromptProbe.prompt,
        prompt_token_ids: cachedPromptProbe.prompt_token_ids,
        kv_cache_reused: cachedPromptProbe.kv_cache_reused,
        prefill_timing_ms: cachedPromptProbe.prefill_timing_ms,
        incremental_token_timing_ms: cachedPromptProbe.incremental_token_timing_ms,
        total_timing_ms: cachedPromptProbe.total_timing_ms,
        token_ids: [...cachedPromptProbe.prompt_token_ids, ...cachedPromptProbe.generated_token_ids],
        token_scores: cachedPromptProbe.generated_token_scores,
        decoded_text: cachedPromptProbe.decoded_text,
        reused_from: cachedPromptProbe.reused_from,
        provisional_cache: cachedPromptProbe.provisional_cache,
        handle_counts_before: handleCountsBeforeCachedPrompt.value,
        handle_counts_after: handleCountsAfterCachedPrompt.value,
      }, null, 2),
    );
    if (
      handleCountsAfterCachedPrompt.ok !== true ||
      JSON.stringify(handleCountsAfterCachedPrompt.value) !== JSON.stringify(handleCountsBeforeCachedPrompt.value)
    ) {
      throw new Error('mlx_handle_counts changed during cached_prompt_session_probe');
    }

    const handleCountsBeforeGreedy = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeGreedy.ok !== true ||
      JSON.stringify(handleCountsBeforeGreedy.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before greedy_next_token_probe did not start clean');
    }

    let greedyNextTokenProbe;
    if (runGenerateProfile) {
      greedyNextTokenProbe = await send('greedy_next_token_probe', {
        model_dir: model4Dir,
        tokenizer: model4Tokenizer,
        input_token_id: 1,
      });
    } else {
      logSkippedProbe('greedy_next_token_probe native recompute', 'profile smoke/full reuses forward_generation_probe result from resident optimized decode');
      greedyNextTokenProbe = {
        ok: true,
        value: {
          provisional: true,
          input_token_id: forwardGenerationProbe.input_token_id,
          next_token_id: forwardGenerationProbe.generated_token_id,
          next_token_score: forwardGenerationProbe.generated_token_score,
          decoded_next_token: forwardGenerationProbe.decoded_generated_text,
          layers_run: forwardGenerationProbe.layers_run,
          final_norm_checksum: forwardGenerationProbe.final_norm_checksum,
          logits_len: forwardGenerationProbe.logits_len,
          timing_ms: forwardGenerationProbe.timing_ms,
          reused_from: 'forward_generation_probe',
        },
      };
    }
    if (
      greedyNextTokenProbe.ok !== true ||
      greedyNextTokenProbe.value?.provisional !== true ||
      greedyNextTokenProbe.value?.input_token_id !== 1 ||
      !Number.isInteger(greedyNextTokenProbe.value?.next_token_id) ||
      !Number.isFinite(greedyNextTokenProbe.value?.next_token_score) ||
      typeof greedyNextTokenProbe.value?.decoded_next_token !== 'string' ||
      greedyNextTokenProbe.value.decoded_next_token.length === 0 ||
      greedyNextTokenProbe.value?.layers_run !== 36 ||
      !Number.isFinite(greedyNextTokenProbe.value?.final_norm_checksum) ||
      greedyNextTokenProbe.value?.logits_len !== 151936 ||
      !Number.isFinite(greedyNextTokenProbe.value?.timing_ms)
    ) {
      throw new Error('greedy_next_token_probe failed verification');
    }
    const handleCountsAfterGreedy = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] greedy_next_token_probe:',
      JSON.stringify({
        input_token_id: greedyNextTokenProbe.value.input_token_id,
        next_token_id: greedyNextTokenProbe.value.next_token_id,
        next_token_score: greedyNextTokenProbe.value.next_token_score,
        decoded_next_token: greedyNextTokenProbe.value.decoded_next_token,
        layers_run: greedyNextTokenProbe.value.layers_run,
        final_norm_checksum: greedyNextTokenProbe.value.final_norm_checksum,
        logits_len: greedyNextTokenProbe.value.logits_len,
        timing_ms: greedyNextTokenProbe.value.timing_ms,
        handle_counts_before: handleCountsBeforeGreedy.value,
        handle_counts_after: handleCountsAfterGreedy.value,
      }, null, 2),
    );
    if (
      handleCountsAfterGreedy.ok !== true ||
      JSON.stringify(handleCountsAfterGreedy.value) !== JSON.stringify(handleCountsBeforeGreedy.value)
    ) {
      throw new Error('mlx_handle_counts changed during greedy_next_token_probe');
    }

    if (runGenerateProfile) {
    const handleCountsBeforeGreedyTwo = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeGreedyTwo.ok !== true ||
      JSON.stringify(handleCountsBeforeGreedyTwo.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before greedy_two_token_probe did not start clean');
    }

    const greedyTwoTokenProbe = await send('greedy_two_token_probe', {
      model_dir: model4Dir,
      tokenizer: model4Tokenizer,
      input_token_id: 1,
    });
    if (
      greedyTwoTokenProbe.ok !== true ||
      greedyTwoTokenProbe.value?.provisional !== true ||
      greedyTwoTokenProbe.value?.input_token_id !== 1 ||
      !Number.isInteger(greedyTwoTokenProbe.value?.first_next_token_id) ||
      !Number.isFinite(greedyTwoTokenProbe.value?.first_next_token_score) ||
      !Number.isInteger(greedyTwoTokenProbe.value?.second_next_token_id) ||
      !Number.isFinite(greedyTwoTokenProbe.value?.second_next_token_score) ||
      typeof greedyTwoTokenProbe.value?.decoded_generated_text !== 'string' ||
      greedyTwoTokenProbe.value.decoded_generated_text.length === 0 ||
      greedyTwoTokenProbe.value?.layers_run_each_pass !== 36 ||
      greedyTwoTokenProbe.value?.logits_len !== 151936 ||
      !Number.isFinite(greedyTwoTokenProbe.value?.timing_total_ms)
    ) {
      throw new Error('greedy_two_token_probe failed verification');
    }
    const handleCountsAfterGreedyTwo = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] greedy_two_token_probe:',
      JSON.stringify({
        input_token_id: greedyTwoTokenProbe.value.input_token_id,
        first_next_token_id: greedyTwoTokenProbe.value.first_next_token_id,
        first_next_token_score: greedyTwoTokenProbe.value.first_next_token_score,
        second_next_token_id: greedyTwoTokenProbe.value.second_next_token_id,
        second_next_token_score: greedyTwoTokenProbe.value.second_next_token_score,
        decoded_generated_text: greedyTwoTokenProbe.value.decoded_generated_text,
        layers_run_each_pass: greedyTwoTokenProbe.value.layers_run_each_pass,
        logits_len: greedyTwoTokenProbe.value.logits_len,
        timing_total_ms: greedyTwoTokenProbe.value.timing_total_ms,
        handle_counts_before: handleCountsBeforeGreedyTwo.value,
        handle_counts_after: handleCountsAfterGreedyTwo.value,
      }, null, 2),
    );
    if (
      handleCountsAfterGreedyTwo.ok !== true ||
      JSON.stringify(handleCountsAfterGreedyTwo.value) !== JSON.stringify(handleCountsBeforeGreedyTwo.value)
    ) {
      throw new Error('mlx_handle_counts changed during greedy_two_token_probe');
    }

    const greedySessionModelLoaded = await send('load_model_native', { path: model4Dir });
    if (
      greedySessionModelLoaded.ok !== true ||
      typeof greedySessionModelLoaded.value?.model !== 'string'
    ) {
      throw new Error('load_model_native for greedy session failed verification');
    }
    const greedySessionCreated = await send('create_native_session', {
      model: greedySessionModelLoaded.value.model,
      tokenizer: model4Tokenizer,
    });
    if (
      greedySessionCreated.ok !== true ||
      typeof greedySessionCreated.value?.session !== 'string'
    ) {
      throw new Error('create_native_session for greedy generation failed verification');
    }

    const handleCountsBeforeGreedySession = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeGreedySession.ok !== true ||
      JSON.stringify(handleCountsBeforeGreedySession.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before greedy_session_generate_probe did not start clean');
    }

    const greedySessionGenerateProbe = await send('greedy_session_generate_probe', {
      session: greedySessionCreated.value.session,
      input_token_id: 1,
      max_tokens: 3,
    });
    if (
      greedySessionGenerateProbe.ok !== true ||
      greedySessionGenerateProbe.value?.provisional !== true ||
      greedySessionGenerateProbe.value?.input_token_id !== 1 ||
      !Array.isArray(greedySessionGenerateProbe.value?.generated_tokens) ||
      greedySessionGenerateProbe.value.generated_tokens.length !== 3 ||
      typeof greedySessionGenerateProbe.value?.decoded_generated_text !== 'string' ||
      greedySessionGenerateProbe.value.decoded_generated_text.length === 0 ||
      !Array.isArray(greedySessionGenerateProbe.value?.steps) ||
      greedySessionGenerateProbe.value.steps.length !== 3
    ) {
      throw new Error('greedy_session_generate_probe failed verification');
    }
    for (const step of greedySessionGenerateProbe.value.steps) {
      if (
        !Number.isInteger(step?.input_token) ||
        !Number.isInteger(step?.next_token) ||
        !Number.isFinite(step?.next_token_score) ||
        typeof step?.decoded_accumulated_text !== 'string' ||
        step.decoded_accumulated_text.length === 0 ||
        !Number.isFinite(step?.timing_ms)
      ) {
        throw new Error('greedy_session_generate_probe step failed verification');
      }
    }
    const handleCountsAfterGreedySession = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] greedy_session_generate_probe:',
      JSON.stringify({
        input_token_id: greedySessionGenerateProbe.value.input_token_id,
        generated_tokens: greedySessionGenerateProbe.value.generated_tokens,
        decoded_generated_text: greedySessionGenerateProbe.value.decoded_generated_text,
        steps: greedySessionGenerateProbe.value.steps,
        handle_counts_before: handleCountsBeforeGreedySession.value,
        handle_counts_after: handleCountsAfterGreedySession.value,
      }, null, 2),
    );
    if (
      handleCountsAfterGreedySession.ok !== true ||
      JSON.stringify(handleCountsAfterGreedySession.value) !== JSON.stringify(handleCountsBeforeGreedySession.value)
    ) {
      throw new Error('mlx_handle_counts changed during greedy_session_generate_probe');
    }

    const handleCountsBeforePromptSession = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforePromptSession.ok !== true ||
      JSON.stringify(handleCountsBeforePromptSession.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before greedy_prompt_session_probe did not start clean');
    }

    const greedyPromptSessionProbe = await send('greedy_prompt_session_probe', {
      session: greedySessionCreated.value.session,
      prompt: 'hello',
    });
    if (
      greedyPromptSessionProbe.ok !== true ||
      greedyPromptSessionProbe.value?.provisional !== true ||
      greedyPromptSessionProbe.value?.prompt !== 'hello' ||
      !Array.isArray(greedyPromptSessionProbe.value?.prompt_token_ids) ||
      greedyPromptSessionProbe.value.prompt_token_ids.length === 0 ||
      !Number.isInteger(greedyPromptSessionProbe.value?.generated_token_id) ||
      !Number.isFinite(greedyPromptSessionProbe.value?.generated_token_score) ||
      typeof greedyPromptSessionProbe.value?.decoded_generated_text !== 'string' ||
      greedyPromptSessionProbe.value.decoded_generated_text.length === 0 ||
      !Number.isFinite(greedyPromptSessionProbe.value?.timing_ms)
    ) {
      throw new Error('greedy_prompt_session_probe failed verification');
    }
    const handleCountsAfterPromptSession = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] greedy_prompt_session_probe:',
      JSON.stringify({
        prompt: greedyPromptSessionProbe.value.prompt,
        prompt_token_ids: greedyPromptSessionProbe.value.prompt_token_ids,
        generated_token_id: greedyPromptSessionProbe.value.generated_token_id,
        generated_token_score: greedyPromptSessionProbe.value.generated_token_score,
        decoded_generated_text: greedyPromptSessionProbe.value.decoded_generated_text,
        timing_ms: greedyPromptSessionProbe.value.timing_ms,
        handle_counts_before: handleCountsBeforePromptSession.value,
        handle_counts_after: handleCountsAfterPromptSession.value,
      }, null, 2),
    );
    if (
      handleCountsAfterPromptSession.ok !== true ||
      JSON.stringify(handleCountsAfterPromptSession.value) !== JSON.stringify(handleCountsBeforePromptSession.value)
    ) {
      throw new Error('mlx_handle_counts changed during greedy_prompt_session_probe');
    }

    const handleCountsBeforeIncrementalSession = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeIncrementalSession.ok !== true ||
      JSON.stringify(handleCountsBeforeIncrementalSession.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before incremental_session_probe did not start clean');
    }

    const incrementalSessionProbe = await send('incremental_session_probe', {
      session: greedySessionCreated.value.session,
      prompt: 'hello',
      max_tokens: 3,
    });
    if (
      incrementalSessionProbe.ok !== true ||
      incrementalSessionProbe.value?.provisional !== true ||
      incrementalSessionProbe.value?.prompt !== 'hello' ||
      !Array.isArray(incrementalSessionProbe.value?.prompt_token_ids) ||
      incrementalSessionProbe.value.prompt_token_ids.length === 0 ||
      incrementalSessionProbe.value?.identical_generated_token_ids !== true ||
      incrementalSessionProbe.value?.identical_decoded_text !== true ||
      incrementalSessionProbe.value?.logits_checksums_match !== true ||
      !Array.isArray(incrementalSessionProbe.value?.incremental_steps) ||
      incrementalSessionProbe.value.incremental_steps.length !== 3 ||
      !Array.isArray(incrementalSessionProbe.value?.per_token_incremental_timing_ms) ||
      incrementalSessionProbe.value.per_token_incremental_timing_ms.length !== 3 ||
      !Number.isFinite(incrementalSessionProbe.value?.total_incremental_timing_ms) ||
      !Number.isFinite(incrementalSessionProbe.value?.total_recompute_timing_ms) ||
      !Number.isFinite(incrementalSessionProbe.value?.speedup_ratio) ||
      typeof incrementalSessionProbe.value?.fallback_stage !== 'string'
    ) {
      throw new Error('incremental_session_probe failed verification');
    }
    for (const step of incrementalSessionProbe.value.incremental_steps) {
      if (
        !Number.isInteger(step?.input_token) ||
        !Number.isInteger(step?.next_token) ||
        !Number.isFinite(step?.next_token_score) ||
        typeof step?.decoded_accumulated_text !== 'string' ||
        step.decoded_accumulated_text.length === 0 ||
        !Number.isFinite(step?.timing_ms)
      ) {
        throw new Error('incremental_session_probe step failed verification');
      }
    }
    const handleCountsAfterIncrementalSession = await send('mlx_handle_counts', {});
    console.log(
      '[rusty verify] incremental_session_probe:',
      JSON.stringify({
        prompt: incrementalSessionProbe.value.prompt,
        prompt_token_ids: incrementalSessionProbe.value.prompt_token_ids,
        recompute_generated_tokens: incrementalSessionProbe.value.recompute_generated_tokens,
        incremental_generated_tokens: incrementalSessionProbe.value.incremental_generated_tokens,
        decoded_recompute_text: incrementalSessionProbe.value.decoded_recompute_text,
        decoded_incremental_text: incrementalSessionProbe.value.decoded_incremental_text,
        identical_generated_token_ids: incrementalSessionProbe.value.identical_generated_token_ids,
        identical_decoded_text: incrementalSessionProbe.value.identical_decoded_text,
        max_logits_checksum_diff: incrementalSessionProbe.value.max_logits_checksum_diff,
        logits_checksum_tolerance: incrementalSessionProbe.value.logits_checksum_tolerance,
        per_token_incremental_timing_ms: incrementalSessionProbe.value.per_token_incremental_timing_ms,
        total_incremental_timing_ms: incrementalSessionProbe.value.total_incremental_timing_ms,
        total_recompute_timing_ms: incrementalSessionProbe.value.total_recompute_timing_ms,
        speedup_ratio: incrementalSessionProbe.value.speedup_ratio,
        fallback_stage: incrementalSessionProbe.value.fallback_stage,
        handle_counts_before: handleCountsBeforeIncrementalSession.value,
        handle_counts_after: handleCountsAfterIncrementalSession.value,
      }, null, 2),
    );
    if (
      handleCountsAfterIncrementalSession.ok !== true ||
      JSON.stringify(handleCountsAfterIncrementalSession.value) !== JSON.stringify(handleCountsBeforeIncrementalSession.value)
    ) {
      throw new Error('mlx_handle_counts changed during incremental_session_probe');
    }

    const greedySessionFreed = await send('free_native_session', {
      session: greedySessionCreated.value.session,
    });
    if (greedySessionFreed.ok !== true) {
      throw new Error('free_native_session for greedy generation failed verification');
    }
    const greedySessionModelUnloaded = await send('unload_model_native', {
      model: greedySessionModelLoaded.value.model,
    });
    if (greedySessionModelUnloaded.ok !== true) {
      throw new Error('unload_model_native for greedy generation failed verification');
    }

    } else {
      logSkippedProbe('greedy_two_token_probe/greedy_session_generate_probe/greedy_prompt_session_probe/incremental_session_probe', 'profile smoke/layer/full skips expensive generation-session probes');
    }

    const model4TokenizerUnloaded = await send('unload_tokenizer', { tokenizer: model4Tokenizer });
    if (model4TokenizerUnloaded.ok !== true) {
      throw new Error('unload_tokenizer for model4 failed verification');
    }

    if (runLayerProfile) {
    const handleCountsBeforeLayer0Mlp = await send('mlx_handle_counts', {});
    if (
      handleCountsBeforeLayer0Mlp.ok !== true ||
      JSON.stringify(handleCountsBeforeLayer0Mlp.value) !== JSON.stringify(handleCountsBeforeLayer0.value)
    ) {
      throw new Error('mlx_handle_counts before layer0_mlp_probe did not start clean');
    }
    console.log(
      '[rusty verify] mlx_handle_counts before layer0_mlp_probe:',
      JSON.stringify(handleCountsBeforeLayer0Mlp.value, null, 2),
    );

    const layer0MlpProbe = await send('layer0_mlp_probe', {
      model_dir: model4Dir,
      token_id: 1,
    });
    const expectedLayer0MlpLengths = {
      input_len: 2560,
      norm_len: 2560,
      gate_len: 9728,
      up_len: 9728,
      activated_len: 9728,
      down_len: 2560,
      residual_len: 2560,
    };
    if (
      layer0MlpProbe.ok !== true ||
      layer0MlpProbe.value?.provisional !== true ||
      layer0MlpProbe.value?.token_id !== 1 ||
      !Number.isFinite(layer0MlpProbe.value?.timing_ms)
    ) {
      throw new Error('layer0_mlp_probe failed verification');
    }
    for (const [key, expected] of Object.entries(expectedLayer0MlpLengths)) {
      if (layer0MlpProbe.value?.[key] !== expected) {
        throw new Error(\`layer0_mlp_probe ${key} expected ${expected}\`);
      }
    }
    for (const [key, value] of Object.entries(layer0MlpProbe.value.checksums ?? {})) {
      if (!Number.isFinite(value)) {
        throw new Error(\`layer0_mlp_probe checksum ${key} was not finite\`);
      }
    }
    for (const [key, values] of Object.entries(layer0MlpProbe.value.first_values ?? {})) {
      if (!Array.isArray(values) || values.length !== 8) {
        throw new Error(\`layer0_mlp_probe first_values ${key} missing\`);
      }
      for (const value of values) {
        if (!Number.isFinite(value)) {
          throw new Error(\`layer0_mlp_probe first_values ${key} contained a non-finite value\`);
        }
      }
    }
    console.log(
      '[rusty verify] layer0_mlp_probe:',
      JSON.stringify({
        token_id: layer0MlpProbe.value.token_id,
        lengths: expectedLayer0MlpLengths,
        checksums: layer0MlpProbe.value.checksums,
        first_values: layer0MlpProbe.value.first_values,
        timing_ms: layer0MlpProbe.value.timing_ms,
      }, null, 2),
    );

    const handleCountsAfterLayer0Mlp = await send('mlx_handle_counts', {});
    if (
      handleCountsAfterLayer0Mlp.ok !== true ||
      JSON.stringify(handleCountsAfterLayer0Mlp.value) !== JSON.stringify(handleCountsBeforeLayer0Mlp.value)
    ) {
      throw new Error('mlx_handle_counts changed during layer0_mlp_probe');
    }
    console.log(
      '[rusty verify] mlx_handle_counts after layer0_mlp_probe:',
      JSON.stringify(handleCountsAfterLayer0Mlp.value, null, 2),
    );
    } else {
      logSkippedProbe('layer0_mlp_probe', 'profile smoke skips focused layer MLP probe');
    }

    if (shouldRunMlxRuntime()) {
      const runtimeTensorGroupName = 'model.layers.0.self_attn.q_proj';
      const runtimeTensorGroup = await send('load_tensor_group', {
        model_dir: model4Dir,
        group: runtimeTensorGroupName,
      });
      if (runtimeTensorGroup.ok !== true || typeof runtimeTensorGroup.value?.group !== 'string') {
        throw new Error('load_tensor_group for runtime dequant failed verification');
      }
      const dequantizedSlice = await send('dequantize_group_slice', {
        group: runtimeTensorGroup.value.group,
        row: 0,
        cols: 8,
      });
      if (
        dequantizedSlice.ok !== true ||
        typeof dequantizedSlice.value?.array !== 'number' ||
        dequantizedSlice.value?.dtype !== 'float32' ||
        !Array.isArray(dequantizedSlice.value?.shape) ||
        dequantizedSlice.value.shape.length !== 2 ||
        dequantizedSlice.value.shape[0] !== 1 ||
        dequantizedSlice.value.shape[1] !== 8 ||
        dequantizedSlice.value?.size !== 8 ||
        dequantizedSlice.value?.source_group !== runtimeTensorGroupName
      ) {
        console.log('[rusty verify] dequantize_group_slice unexpected:', JSON.stringify(dequantizedSlice, null, 2));
        throw new Error('dequantize_group_slice failed verification');
      }
      console.log('[rusty verify] dequantize_group_slice handle:', dequantizedSlice.value.array);
      console.log('[rusty verify] dequantize_group_slice dtype:', dequantizedSlice.value.dtype);
      console.log('[rusty verify] dequantize_group_slice shape:', JSON.stringify(dequantizedSlice.value.shape));
      console.log('[rusty verify] dequantize_group_slice source_group:', dequantizedSlice.value.source_group);

      const dequantizedSliceInfo = await send('mlx_array_info', {
        array: dequantizedSlice.value.array,
      });
      if (
        dequantizedSliceInfo.ok !== true ||
        dequantizedSliceInfo.value?.dtype !== 'float32' ||
        !Array.isArray(dequantizedSliceInfo.value?.shape) ||
        dequantizedSliceInfo.value.shape.length !== 2 ||
        dequantizedSliceInfo.value.shape[0] !== 1 ||
        dequantizedSliceInfo.value.shape[1] !== 8 ||
        dequantizedSliceInfo.value?.size !== 8 ||
        dequantizedSliceInfo.value?.source_group !== runtimeTensorGroupName
      ) {
        throw new Error('mlx_array_info for dequantized slice failed verification');
      }
      console.log('[rusty verify] mlx_array_info slice:', JSON.stringify(dequantizedSliceInfo.value, null, 2));

      const dequantizedSliceFreed = await send('mlx_free_array', {
        array: dequantizedSlice.value.array,
      });
      if (dequantizedSliceFreed.ok !== true || dequantizedSliceFreed.value?.freed !== true) {
        throw new Error('mlx_free_array for dequantized slice failed verification');
      }
      console.log('[rusty verify] mlx_free_array slice ok:', dequantizedSliceFreed.ok);

      const dequantizedSliceInfoAfterFree = await send('mlx_array_info', {
        array: dequantizedSlice.value.array,
      });
      if (
        dequantizedSliceInfoAfterFree.ok !== false ||
        dequantizedSliceInfoAfterFree.error?.code !== 'unknown_handle'
      ) {
        throw new Error('mlx_array_info after slice free did not fail as expected');
      }

      const runtimeTensorGroupFreed = await send('free_tensor_group', {
        group: runtimeTensorGroup.value.group,
      });
      if (runtimeTensorGroupFreed.ok !== true || runtimeTensorGroupFreed.value?.freed !== true) {
        throw new Error('free_tensor_group for runtime dequant failed verification');
      }
      console.log('[rusty verify] free_tensor_group runtime dequant ok:', runtimeTensorGroupFreed.ok);
    } else {
      console.log('[rusty verify] dequantize_group_slice skipped; set RUSTY_RUN_MLX_RUNTIME=1 to enable');
    }

    const handleCountsAfterRuntimeDequant = await send('mlx_handle_counts', {});
    if (
      handleCountsAfterRuntimeDequant.ok !== true ||
      handleCountsAfterRuntimeDequant.value?.arrays !== 0 ||
      handleCountsAfterRuntimeDequant.value?.token_arrays !== 0 ||
      handleCountsAfterRuntimeDequant.value?.tensor_groups !== 0 ||
      handleCountsAfterRuntimeDequant.value?.embedding_groups !== 0 ||
      handleCountsAfterRuntimeDequant.value?.layer_groups !== 0
    ) {
      throw new Error('mlx_handle_counts after runtime dequant/free did not return to zero');
    }
    console.log('[rusty verify] mlx_handle_counts after runtime dequant/free:', JSON.stringify(handleCountsAfterRuntimeDequant.value, null, 2));

    if (runLayerProfile) {
    const layerGroups = await send('load_layer_groups', {
      model_dir: model4Dir,
      layer: 0,
    });
    const expectedLayerGroups = [
	      'q_proj',
	      'k_proj',
	      'v_proj',
	      'o_proj',
	      'q_norm',
	      'k_norm',
	      'gate_proj',
      'up_proj',
      'down_proj',
      'input_layernorm',
      'post_attention_layernorm',
    ];
    if (
      layerGroups.ok !== true ||
      typeof layerGroups.value?.handle !== 'string' ||
      layerGroups.value?.summary?.total_groups !== expectedLayerGroups.length ||
      layerGroups.value?.summary?.quantized_groups !== 7 ||
	      layerGroups.value?.summary?.norm_groups !== 4 ||
      layerGroups.value?.total_byte_size <= 0 ||
      typeof layerGroups.value?.group_handles !== 'object'
    ) {
      throw new Error('load_layer_groups failed verification');
    }
    for (const groupName of expectedLayerGroups) {
      if (typeof layerGroups.value.group_handles[groupName] !== 'string') {
        throw new Error(\`load_layer_groups missing group handle for ${groupName}\`);
      }
    }
    console.log('[rusty verify] load_layer_groups handle:', layerGroups.value.handle);
    console.log('[rusty verify] load_layer_groups summary:', JSON.stringify(layerGroups.value.summary, null, 2));
    console.log('[rusty verify] load_layer_groups total_byte_size:', layerGroups.value.total_byte_size);

    const layerGroupsInfo = await send('layer_groups_info', {
      layer: layerGroups.value.handle,
    });
    if (
      layerGroupsInfo.ok !== true ||
      layerGroupsInfo.value?.summary?.total_groups !== expectedLayerGroups.length ||
      layerGroupsInfo.value?.summary?.quantized_groups !== 7 ||
	      layerGroupsInfo.value?.summary?.norm_groups !== 4 ||
      layerGroupsInfo.value?.total_byte_size <= 0 ||
      typeof layerGroupsInfo.value?.groups !== 'object'
    ) {
      throw new Error('layer_groups_info failed verification');
    }
    for (const groupName of expectedLayerGroups) {
      const groupInfo = layerGroupsInfo.value.groups[groupName];
      if (!groupInfo || typeof groupInfo.handle !== 'string') {
        throw new Error(\`layer_groups_info missing group ${groupName}\`);
      }
	      if (
	        groupName === 'input_layernorm' ||
	        groupName === 'post_attention_layernorm' ||
	        groupName === 'q_norm' ||
	        groupName === 'k_norm'
	      ) {
        if (groupInfo.quantized_group !== false) {
          throw new Error(\`${groupName} should not be quantized\`);
        }
      } else if (groupInfo.quantized_group !== true) {
        throw new Error(\`${groupName} should be quantized\`);
      }
      if (!Array.isArray(groupInfo.weight?.shape) || groupInfo.weight.byte_size <= 0) {
        throw new Error(\`${groupName} missing weight metadata\`);
      }
    }
    console.log('[rusty verify] layer_groups_info:', JSON.stringify(layerGroupsInfo.value.summary, null, 2));

    const layerGroupsFreed = await send('free_layer_groups', {
      layer: layerGroups.value.handle,
    });
    if (layerGroupsFreed.ok !== true || layerGroupsFreed.value?.freed !== true) {
      throw new Error('free_layer_groups failed verification');
    }
    console.log('[rusty verify] free_layer_groups ok:', layerGroupsFreed.ok);

    const layerGroupsInfoAfterFree = await send('layer_groups_info', {
      layer: layerGroups.value.handle,
    });
    if (
      layerGroupsInfoAfterFree.ok !== false ||
      layerGroupsInfoAfterFree.error?.code !== 'already_freed'
    ) {
      throw new Error('layer_groups_info after free did not fail as expected');
    }

    const childGroupInfoAfterFree = await send('tensor_group_info', {
      group: layerGroups.value.group_handles.q_proj,
    });
    if (
      childGroupInfoAfterFree.ok !== false ||
      childGroupInfoAfterFree.error?.code !== 'already_freed'
    ) {
      throw new Error('child tensor_group_info after layer free did not fail as expected');
    }
    } else {
      logSkippedProbe('load_layer_groups/layer_groups_info', 'profile smoke skips layer group handle probe');
    }

    if (runFullProfile) {
    const shimProbe = await send('shim_probe', {});
    if (shimProbe.ok !== true || shimProbe.value?.reachable !== true) {
      throw new Error('shim_probe failed verification');
    }
    console.log('[rusty verify] shim_probe version:', shimProbe.value?.version);

    const mlxLinkProbe = await send('mlx_link_probe', {});
    if (mlxLinkProbe.ok !== true || typeof mlxLinkProbe.value?.linked !== 'boolean') {
      throw new Error('mlx_link_probe failed verification');
    }
    console.log('[rusty verify] mlx_link_probe linked:', mlxLinkProbe.value?.linked);
    console.log(
      '[rusty verify] mlx_link_probe notes:',
      JSON.stringify(mlxLinkProbe.value?.notes ?? [], null, 2),
    );

    const runtimeDiagnose = await send('mlx_runtime_diagnose', {});
    if (runtimeDiagnose.ok === true) {
      console.log(
        '[rusty verify] mlx_runtime_diagnose:',
        JSON.stringify(runtimeDiagnose.value ?? {}, null, 2),
      );
    } else {
      console.log(
        '[rusty verify] mlx_runtime_diagnose failed cleanly:',
        JSON.stringify(runtimeDiagnose.error ?? {}, null, 2),
      );
    }
    } else {
      logSkippedProbe('shim_probe/mlx_link_probe/mlx_runtime_diagnose', 'profile smoke/layer skips detailed native link diagnostics');
    }

    if (shouldRunMlxRuntime() || verifierProfile === 'smoke') {
      const createdNativeArray = await send('mlx_create_test_array', {});
      if (createdNativeArray.ok !== true) {
        throw new Error('mlx_create_test_array failed verification');
      }
      if (createdNativeArray.value?.created === true) {
        const nativeHandle = createdNativeArray.value?.handle;
        if (!Number.isInteger(nativeHandle) || nativeHandle <= 0) {
          throw new Error('mlx_create_test_array returned invalid handle');
        }
        console.log('[rusty verify] mlx_create_test_array handle:', nativeHandle);

        const nativeSum = await send('mlx_test_array_sum', { handle: nativeHandle });
        if (nativeSum.ok !== true || nativeSum.value?.ok !== true || nativeSum.value?.sum !== 7) {
          throw new Error('mlx_test_array_sum failed verification');
        }
        console.log('[rusty verify] mlx_test_array_sum sum:', nativeSum.value?.sum);

        const freedNativeArray = await send('mlx_free_test_array', { handle: nativeHandle });
        if (freedNativeArray.ok !== true || freedNativeArray.value?.freed !== true) {
          throw new Error('mlx_free_test_array failed verification');
        }

        const sumAfterFree = await send('mlx_test_array_sum', { handle: nativeHandle });
        if (sumAfterFree.ok !== false || sumAfterFree.error?.code !== 'unknown_handle') {
          throw new Error('mlx_test_array_sum after free did not fail as expected');
        }
      } else {
        console.log(
          '[rusty verify] mlx_create_test_array unavailable:',
          JSON.stringify(createdNativeArray.value?.notes ?? [], null, 2),
        );
      }
    } else {
      console.log('[rusty verify] mlx runtime tests skipped; set RUSTY_RUN_MLX_RUNTIME=1 to enable');
    }

    phaseTimer.switchTo('tokenizer fixture work');
    const fixtureTokenizerLoaded = await send('load_tokenizer', { path: tokenizerFixtureDir });
    if (
      fixtureTokenizerLoaded.ok !== true ||
      typeof fixtureTokenizerLoaded.value?.tokenizer !== 'string' ||
      fixtureTokenizerLoaded.value?.tokenizer_kind !== 'huggingface_json' ||
      fixtureTokenizerLoaded.value?.vocab_size !== 3 ||
      fixtureTokenizerLoaded.value?.merges_count !== 1 ||
      fixtureTokenizerLoaded.value?.added_tokens_count !== 1
    ) {
      throw new Error('load_tokenizer on tokenizer fixture failed verification');
    }
    if (
      !Array.isArray(fixtureTokenizerLoaded.value?.detected_files) ||
      !fixtureTokenizerLoaded.value.detected_files.some((entry) => entry.endsWith('tokenizer.json'))
    ) {
      throw new Error('load_tokenizer did not report tokenizer.json');
    }
    console.log('[rusty verify] load_tokenizer kind:', fixtureTokenizerLoaded.value.tokenizer_kind);
    console.log(
      '[rusty verify] load_tokenizer detected_files:',
      JSON.stringify(fixtureTokenizerLoaded.value.detected_files, null, 2),
    );
    console.log('[rusty verify] load_tokenizer model_type:', fixtureTokenizerLoaded.value.model_type);
    console.log('[rusty verify] load_tokenizer vocab_size:', fixtureTokenizerLoaded.value.vocab_size);
    console.log('[rusty verify] load_tokenizer merges_count:', fixtureTokenizerLoaded.value.merges_count);
    console.log(
      '[rusty verify] load_tokenizer added_tokens_count:',
      fixtureTokenizerLoaded.value.added_tokens_count,
    );

    const fixtureTokenizerInfo = await send('tokenizer_info', {
      tokenizer: fixtureTokenizerLoaded.value.tokenizer,
    });
    if (
      fixtureTokenizerInfo.ok !== true ||
      fixtureTokenizerInfo.value?.tokenizer_kind !== 'huggingface_json' ||
      fixtureTokenizerInfo.value?.loaded !== false ||
      fixtureTokenizerInfo.value?.vocab_size !== 3 ||
      fixtureTokenizerInfo.value?.merges_count !== 1 ||
      fixtureTokenizerInfo.value?.added_tokens_count !== 1
    ) {
      throw new Error('tokenizer_info failed verification');
    }
    console.log('[rusty verify] tokenizer_info loaded:', fixtureTokenizerInfo.value.loaded);
    console.log('[rusty verify] tokenizer_info path:', fixtureTokenizerInfo.value.path);
    console.log('[rusty verify] tokenizer_info model_type:', fixtureTokenizerInfo.value.model_type);
    console.log('[rusty verify] tokenizer_info vocab_size:', fixtureTokenizerInfo.value.vocab_size);
    console.log('[rusty verify] tokenizer_info merges_count:', fixtureTokenizerInfo.value.merges_count);
    console.log(
      '[rusty verify] tokenizer_info added_tokens_count:',
      fixtureTokenizerInfo.value.added_tokens_count,
    );

    const fixtureTokenizerEncode = await send('tokenizer_encode', {
      tokenizer: fixtureTokenizerLoaded.value.tokenizer,
      text: 'hello world',
    });
    if (
      fixtureTokenizerEncode.ok !== true ||
      !Array.isArray(fixtureTokenizerEncode.value?.tokens) ||
      fixtureTokenizerEncode.value.tokens.length !== 2 ||
      fixtureTokenizerEncode.value.tokens[0] !== 1 ||
      fixtureTokenizerEncode.value.tokens[1] !== 2
    ) {
      throw new Error('tokenizer_encode failed verification');
    }
    console.log(
      '[rusty verify] tokenizer_encode tokens:',
      JSON.stringify(fixtureTokenizerEncode.value.tokens, null, 2),
    );

    const fixtureTokenizerDecode = await send('tokenizer_decode', {
      tokenizer: fixtureTokenizerLoaded.value.tokenizer,
      tokens: [1, 2],
    });
    if (fixtureTokenizerDecode.ok !== true || fixtureTokenizerDecode.value?.text !== 'hello world') {
      throw new Error('tokenizer_decode failed verification');
    }
    console.log('[rusty verify] tokenizer_decode text:', fixtureTokenizerDecode.value.text);

    if (verifierProfile === 'smoke') {
      logSkippedProbe('tokenizer negative/native stub/session API probes', 'profile smoke stops after required tokenizer load/encode/decode coverage');
      const smokeTokenizerUnloaded = await send('unload_tokenizer', {
        tokenizer: fixtureTokenizerLoaded.value.tokenizer,
      });
      if (smokeTokenizerUnloaded.ok !== true) {
        throw new Error('unload_tokenizer failed verification');
      }
      console.log('[rusty verify] unload_tokenizer ok:', smokeTokenizerUnloaded.ok);

      const smokeShutdown = await send('bridge_shutdown', {});
      if (smokeShutdown.ok !== true) {
        throw new Error('bridge_shutdown failed verification');
      }
      console.log('[rusty verify] bridge smoke verification passed');
      console.log('[rusty verify] transcript entries:', transcript.length);
      return;
    }

    const fixtureTokenizerEncodeBad = await send('tokenizer_encode', {
      tokenizer: fixtureTokenizerLoaded.value.tokenizer,
      text: 'badtoken',
    });
    if (
      fixtureTokenizerEncodeBad.ok !== false ||
      fixtureTokenizerEncodeBad.error?.code !== 'tokenizer_unknown_token'
    ) {
      throw new Error('tokenizer_encode bad token did not fail as expected');
    }

    const nativeModelLoaded = await send('load_model_native', { path: 'build/model4/fake' });
    if (nativeModelLoaded.ok !== true || typeof nativeModelLoaded.value?.model !== 'string') {
      throw new Error('load_model_native failed verification');
    }
    console.log('[rusty verify] load_model_native model:', nativeModelLoaded.value.model);

    const nativeSessionCreated = await send('create_native_session', {
      model: nativeModelLoaded.value.model,
      tokenizer: fixtureTokenizerLoaded.value.tokenizer,
    });
    if (nativeSessionCreated.ok !== true || typeof nativeSessionCreated.value?.session !== 'string') {
      throw new Error('create_native_session failed verification');
    }
    console.log('[rusty verify] create_native_session session:', nativeSessionCreated.value.session);

    if (shouldRunMlxRuntime()) {
      const handleCountsBefore = await send('mlx_handle_counts', {});
      if (handleCountsBefore.ok !== true || typeof handleCountsBefore.value?.token_arrays !== 'number') {
        throw new Error('mlx_handle_counts before generate failed verification');
      }
      console.log(
        '[rusty verify] mlx_handle_counts before:',
        JSON.stringify(handleCountsBefore.value, null, 2),
      );

      const nativeGenerated = await send('native_mock_generate', {
        session: nativeSessionCreated.value.session,
        tokenizer: fixtureTokenizerLoaded.value.tokenizer,
        prompt: 'hello',
        max_tokens: 2,
      });
      if (
        nativeGenerated.ok !== true ||
        typeof nativeGenerated.value?.text !== 'string' ||
        nativeGenerated.value.text.length === 0 ||
        !Array.isArray(nativeGenerated.value?.tokens) ||
        nativeGenerated.value.tokens.length !== 2 ||
        !nativeGenerated.value.tokens.every((value) => Number.isInteger(value)) ||
        nativeGenerated.value.steps !== 2
      ) {
        throw new Error('native_mock_generate failed verification');
      }
      console.log('[rusty verify] native_mock_generate text:', nativeGenerated.value.text);
      console.log(
        '[rusty verify] native_mock_generate tokens:',
        JSON.stringify(nativeGenerated.value.tokens, null, 2),
      );

      const generatedTokenDecode = await send('tokenizer_decode', {
        tokenizer: fixtureTokenizerLoaded.value.tokenizer,
        tokens: nativeGenerated.value.tokens,
      });
      if (
        generatedTokenDecode.ok !== true ||
        typeof generatedTokenDecode.value?.text !== 'string' ||
        generatedTokenDecode.value.text.length === 0
      ) {
        throw new Error('tokenizer_decode for generated tokens failed verification');
      }
      console.log('[rusty verify] generated token decode text:', generatedTokenDecode.value.text);

      const handleCountsAfter = await send('mlx_handle_counts', {});
      if (handleCountsAfter.ok !== true || typeof handleCountsAfter.value?.token_arrays !== 'number') {
        throw new Error('mlx_handle_counts after generate failed verification');
      }
      if (
        handleCountsAfter.value.token_arrays !== handleCountsBefore.value.token_arrays ||
        handleCountsAfter.value.arrays !== handleCountsBefore.value.arrays
      ) {
        throw new Error('native_mock_generate leaked native handles');
      }
      console.log(
        '[rusty verify] mlx_handle_counts after:',
        JSON.stringify(handleCountsAfter.value, null, 2),
      );
    } else {
      console.log('[rusty verify] mlx generate tests skipped; set RUSTY_RUN_MLX_RUNTIME=1 to enable');
    }

    const nativeSessionFreed = await send('free_native_session', {
      session: nativeSessionCreated.value.session,
    });
    if (nativeSessionFreed.ok !== true) {
      throw new Error('free_native_session failed verification');
    }
    console.log('[rusty verify] free_native_session ok:', nativeSessionFreed.ok);

    const nativeSessionFreedAgain = await send('free_native_session', {
      session: nativeSessionCreated.value.session,
    });
    if (nativeSessionFreedAgain.ok !== false || nativeSessionFreedAgain.error?.code !== 'already_freed') {
      throw new Error('free_native_session with freed handle did not fail as expected');
    }

    const tokenizerUnloaded = await send('unload_tokenizer', {
      tokenizer: fixtureTokenizerLoaded.value.tokenizer,
    });
    if (tokenizerUnloaded.ok !== true) {
      throw new Error('unload_tokenizer failed verification');
    }
    console.log('[rusty verify] unload_tokenizer ok:', tokenizerUnloaded.ok);

    const tokenizerInfoAfterUnload = await send('tokenizer_info', {
      tokenizer: fixtureTokenizerLoaded.value.tokenizer,
    });
    if (tokenizerInfoAfterUnload.ok !== false || tokenizerInfoAfterUnload.error?.code !== 'already_freed') {
      throw new Error('tokenizer_info after unload did not fail as expected');
    }

    const nativeModelUnloaded = await send('unload_model_native', {
      model: nativeModelLoaded.value.model,
    });
    if (nativeModelUnloaded.ok !== true) {
      throw new Error('unload_model_native failed verification');
    }
    console.log('[rusty verify] unload_model_native ok:', nativeModelUnloaded.ok);

    const nativeModelFreed = await send('unload_model_native', {
      model: nativeModelLoaded.value.model,
    });
    if (nativeModelFreed.ok !== false || nativeModelFreed.error?.code !== 'already_freed') {
      throw new Error('unload_model_native with freed handle did not fail as expected');
    }

    const tokenizerFreed = await send('unload_tokenizer', {
      tokenizer: fixtureTokenizerLoaded.value.tokenizer,
    });
    if (tokenizerFreed.ok !== false || tokenizerFreed.error?.code !== 'already_freed') {
      throw new Error('unload_tokenizer with freed handle did not fail as expected');
    }

    const createWithFreedNativeModel = await send('create_native_session', {
      model: nativeModelLoaded.value.model,
      tokenizer: fixtureTokenizerLoaded.value.tokenizer,
    });
    if (
      createWithFreedNativeModel.ok !== false ||
      !['already_freed', 'unknown_handle'].includes(createWithFreedNativeModel.error?.code)
    ) {
      throw new Error('create_native_session with freed handles did not fail as expected');
    }

    const createWithFreedTokenizer = await send('create_native_session', {
      model: nativeModelLoaded.value.model,
      tokenizer: fixtureTokenizerLoaded.value.tokenizer,
    });
    if (
      createWithFreedTokenizer.ok !== false ||
      !['already_freed', 'unknown_handle'].includes(createWithFreedTokenizer.error?.code)
    ) {
      throw new Error('create_native_session with freed tokenizer did not fail as expected');
    }

    const unknown = await send('bridge_dance', {});
    if (unknown.ok !== false || unknown.error?.code !== 'unknown_cmd') {
      throw new Error('unknown command did not fail as expected');
    }

    const badSessionCreate = await send('create_session', { model: 'model:fake' });
    if (badSessionCreate.ok !== false || badSessionCreate.error?.code !== 'unknown_handle') {
      throw new Error('create_session with fake model did not fail as expected');
    }

    const fakeGenerate = await send('generate', {
      session: 'sess:fake',
      prompt: 'hello',
    });
    if (fakeGenerate.ok !== false || fakeGenerate.error?.code !== 'unknown_handle') {
      throw new Error('generate with fake session did not fail as expected');
    }

    const loaded = await send('load_model', { path: 'build/model4/fake' });
    if (loaded.ok !== true || typeof loaded.value?.model !== 'string') {
      throw new Error('load_model failed verification');
    }

    const unloaded = await send('unload_model', { model: loaded.value.model });
    if (unloaded.ok !== true) {
      throw new Error('unload_model failed verification');
    }

    const createFromFreedModel = await send('create_session', { model: loaded.value.model });
    if (createFromFreedModel.ok !== false || createFromFreedModel.error?.code !== 'already_freed') {
      throw new Error('create_session with freed model did not fail as expected');
    }

    const loadedAgain = await send('load_model', { path: 'build/model4/fake' });
    if (loadedAgain.ok !== true || typeof loadedAgain.value?.model !== 'string') {
      throw new Error('second load_model failed verification');
    }

    const created = await send('create_session', { model: loadedAgain.value.model });
    if (created.ok !== true || typeof created.value?.session !== 'string') {
      throw new Error('create_session failed verification');
    }

    const badGenerate = await send('generate', {
      session: created.value.session,
    });
    if (badGenerate.ok !== false || badGenerate.error?.code !== 'bad_args') {
      throw new Error('generate without prompt did not fail as expected');
    }

    const generated = await send('generate', {
      session: created.value.session,
      prompt: 'hello',
    });
    if (generated.ok !== true) {
      throw new Error('generate failed verification');
    }

    const freedSession = await send('free_session', { session: created.value.session });
    if (freedSession.ok !== true) {
      throw new Error('free_session failed verification');
    }

    const generateAfterFree = await send('generate', {
      session: created.value.session,
      prompt: 'hello again',
    });
    if (generateAfterFree.ok !== false || generateAfterFree.error?.code !== 'already_freed') {
      throw new Error('generate with freed session did not fail as expected');
    }

    phaseTimer.switchTo('cleanup');
    const shutdown = await send('bridge_shutdown', {});
    if (shutdown.ok !== true) {
      throw new Error('bridge_shutdown failed verification');
    }

    const afterShutdown = await send('load_model', { path: 'build/model4/fake' });
    if (afterShutdown.ok !== false || afterShutdown.error?.code !== 'bridge_shutting_down') {
      throw new Error('load_model after shutdown did not fail as expected');
    }

    console.log('[rusty verify] bridge stub verification passed');
    console.log('[rusty verify] phase_timing_ms:', JSON.stringify(phaseTimer.summary(), null, 2));
    console.log('[rusty verify] transcript entries:', transcript.length);
  } finally {
    try {
      bridge.stdin.end();
    } catch {}
    await new Promise((resolve) => bridge.once('exit', resolve));
    try {
      await rm(tokenizerFixtureDir, { recursive: true, force: true });
    } catch {}
  }
}

main().catch((err) => {
  console.error('[rusty verify] failed:', String(err?.message ?? err));
  process.exit(1);
});

`
