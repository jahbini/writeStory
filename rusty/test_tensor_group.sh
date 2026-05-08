#!/usr/bin/env bash
set -euo pipefail

node <<'NODE'
const { spawn } = require('node:child_process');
const { createInterface } = require('node:readline');

const bridge = spawn('cargo', ['run', '--quiet', '--manifest-path', 'rusty/bridge/Cargo.toml'], {
  stdio: ['pipe', 'pipe', 'inherit'],
});

const rl = createInterface({ input: bridge.stdout });
const pending = new Map();
let nextId = 0;

rl.on('line', (line) => {
  if (!line.trim()) return;
  const msg = JSON.parse(line);
  const waiter = pending.get(msg.id);
  if (waiter) {
    pending.delete(msg.id);
    waiter(msg);
  }
});

function send(cmd, args = {}) {
  const id = String(++nextId);
  const request = { id, cmd, args };
  console.log(`Q: ${cmd} ${JSON.stringify(args)}`);
  bridge.stdin.write(JSON.stringify(request) + '\n');
  return new Promise((resolve) => pending.set(id, resolve)).then((msg) => {
    console.log(`A: ${cmd} -> ${JSON.stringify(msg)}`);
    return msg;
  });
}

(async () => {
  console.log('Q: Can the bridge load one tensor group from model4?');
  const load = await send('load_tensor_group', {
    model_dir: 'pipes/Qwen_Qwen3-4B-Instruct-2507/build/model4',
    group: 'model.layers.0.self_attn.q_proj',
  });
  if (!load.ok) throw new Error(`load_tensor_group failed: ${JSON.stringify(load)}`);
  console.log(`A: yes, handle = ${load.value.group}`);

  console.log('Q: Does tensor_group_info report the expected q_proj metadata?');
  const info = await send('tensor_group_info', { group: load.value.group });
  if (!info.ok) throw new Error(`tensor_group_info failed: ${JSON.stringify(info)}`);
  if (info.value.weight?.dtype !== 'U32') throw new Error('expected weight dtype U32');
  if (info.value.scales?.dtype !== 'BF16') throw new Error('expected scales dtype BF16');
  if (info.value.biases?.dtype !== 'BF16') throw new Error('expected biases dtype BF16');
  console.log('A: yes, q_proj metadata matches expectations');

  console.log('Q: Can the tensor group be freed cleanly?');
  const freed = await send('free_tensor_group', { group: load.value.group });
  if (!freed.ok || freed.value?.freed !== true) {
    throw new Error(`free_tensor_group failed: ${JSON.stringify(freed)}`);
  }
  console.log('A: yes, free_tensor_group returned freed:true');

  console.log('Q: Does the freed handle fail cleanly?');
  const after = await send('tensor_group_info', { group: load.value.group });
  if (after.ok !== false || after.error?.code !== 'already_freed') {
    throw new Error(`expected already_freed after free: ${JSON.stringify(after)}`);
  }
  console.log('A: yes, tensor_group_info returned already_freed');

  console.log('Q: Can the bridge shut down cleanly?');
  await send('bridge_shutdown', {});
  console.log('A: yes, bridge_shutdown accepted');
  bridge.stdin.end();
})().catch((err) => {
  console.error(err.stack || String(err));
  bridge.stdin.end();
  process.exitCode = 1;
});
NODE
