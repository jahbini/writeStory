#!/usr/bin/env node

import { spawn, spawnSync } from 'node:child_process';
import { createInterface } from 'node:readline';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const manifestPath = path.join(__dirname, 'bridge', 'Cargo.toml');

function hasCargo() {
  const probe = spawnSync('cargo', ['--version'], { stdio: 'ignore' });
  return probe.status === 0;
}

function buildBridge() {
  const built = spawnSync('cargo', ['build', '--manifest-path', manifestPath], {
    stdio: 'inherit',
  });
  if (built.status !== 0) {
    throw new Error(`cargo build failed with status ${built.status}`);
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

  buildBridge();

  const bridge = startBridge();
  const rl = createInterface({ input: bridge.stdout });
  const pending = new Map();
  const transcript = [];

  bridge.stderr.on('data', (buf) => {
    const text = String(buf ?? '').trim();
    if (text.length) process.stderr.write(`${text}\n`);
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
      waiter.reject(new Error(`bridge exited before reply code=${code} signal=${signal}`));
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
      bridge.stdin.write(`${JSON.stringify(request)}\n`, 'utf8');
    });
  };

  try {
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
    console.log(
      '[rusty verify] backend_probe installation_groups:',
      JSON.stringify(backendProbe.value.installation_groups, null, 2),
    );
    console.log(
      '[rusty verify] backend_probe evidence:',
      JSON.stringify(backendProbe.value.native_boundary.evidence ?? [], null, 2),
    );
    console.log('[rusty verify] backend_probe:', JSON.stringify(backendProbe.value, null, 2));

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
      if (nativeSum.ok !== true || nativeSum.value?.ok !== true || typeof nativeSum.value?.sum !== 'number') {
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

    const shutdown = await send('bridge_shutdown', {});
    if (shutdown.ok !== true) {
      throw new Error('bridge_shutdown failed verification');
    }

    const afterShutdown = await send('load_model', { path: 'build/model4/fake' });
    if (afterShutdown.ok !== false || afterShutdown.error?.code !== 'bridge_shutting_down') {
      throw new Error('load_model after shutdown did not fail as expected');
    }

    console.log('[rusty verify] bridge stub verification passed');
    console.log('[rusty verify] transcript entries:', transcript.length);
  } finally {
    try {
      bridge.stdin.end();
    } catch {}
    await new Promise((resolve) => bridge.once('exit', resolve));
  }
}

main().catch((err) => {
  console.error('[rusty verify] failed:', String(err?.message ?? err));
  process.exit(1);
});
