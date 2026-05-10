# Rusty LLM Chain

This note explains where each major Rusty section fits in the language-model
generation chain. It is intentionally short and user-facing; implementation
status and benchmark details live in `STATUS.md`.

## 1. Pipeline Recipe And UI

The writeStory recipe and UI decide when generation should run and which prompt,
model, adapter, and sampling controls to use.

In the LLM chain this is the orchestration layer. It does not own tensors,
model weights, KV cache memory, or Metal buffers. It selects a step, passes
ordinary scalar settings, and receives text plus metadata.

Key files:

- `scripts/prompt_ite/generate_prompt_rusty_ite.coffee`
- recipe YAML entries that call the Rusty prompt step

## 2. CoffeeScript Native Session API

`native_session_api.coffee` is the JavaScript/CoffeeScript client wrapper around
the resident Rusty bridge process.

In the LLM chain this is the control-plane adapter. It starts or talks to the
bridge, sends JSONL commands, and tracks opaque handles such as model, tokenizer,
and session handles. It must not inspect or own tensor payloads.

Key file:

- `rusty/native_session_api.coffee`

## 3. Rust JSONL Bridge

`rusty/bridge/` is the resident native process. It receives JSONL commands from
CoffeeScript, validates arguments, manages handle tables, and calls the native
MLX shim.

In the LLM chain this is the ownership boundary. JavaScript remains an
orchestrator; Rust owns native lifetime, handle cleanup, and command dispatch.

Key files:

- `rusty/bridge/src/main.rs`
- `rusty/bridge/src/commands.rs`
- `rusty/bridge/src/handles.rs`
- `rusty/bridge/src/state.rs`
- `rusty/bridge/src/backend/shim.rs`

## 4. C++ MLX Shim

`mlx_shim.cpp` is where the model math and MLX/Metal work currently happen.

In the LLM chain this is the execution layer. It loads model groups, keeps
resident MLX arrays, manages session state, applies adapters, runs prompt
prefill, runs token decode, updates KV cache, samples/selects tokens, and
returns scalar metadata.

Key files:

- `rusty/bridge/native/mlx_shim.cpp`
- `rusty/bridge/native/mlx_shim.h`

## 5. Tokenizer

The tokenizer converts UI prompt text into model token IDs and converts generated
token IDs back into cleaned text.

In the LLM chain this is the text/token boundary. Rusty must preserve Qwen chat
special tokens and newlines, hide special stop tokens in cleaned output, and
avoid lossy whitespace tokenization.

Relevant behavior:

- raw prompt mode remains available
- Qwen chat formatting can add `<|im_start|>`, `<|im_end|>`, role text, and
  assistant generation prefix
- output exposes both raw and cleaned decoded text where useful

## 6. Model And Resident Session

The native model/session objects keep the expensive generation state alive for a
request.

In the LLM chain this is the model-residency layer. It owns model metadata,
resident layer groups, resident projection arrays, norm weights, tokenizer
state, optional adapter state, and session-local KV cache.

Important distinction:

- persistent model weights are separate from
- persistent KV cache, which is separate from
- transient per-token/layer scratch values

## 7. Adapter / LoRA

An adapter, when configured, is loaded natively from `adapter_config.json` and
`adapters.safetensors`.

In the LLM chain this modifies projection outputs. Rusty applies LoRA deltas as:

```text
base_output + scale * ((input @ lora_a) @ lora_b)
```

The delta is applied only to layers and projections with matching adapter
tensors. For Qwen attention projections, LoRA is added before `q_norm`/`k_norm`
and before RoPE.

Adapter tensors are not owned by CoffeeScript and are not sent over JSONL.

## 8. Prompt Prefill

Prompt prefill processes every prompt token in order before the first generated
token is selected.

In the LLM chain this builds the initial context. Each prompt token updates the
session KV cache, advances the position, and leaves the model state ready to
predict the next token after the full prompt, not just after the first prompt
token.

This matters for correctness: Qwen parity requires full prompt prefill plus
correct position handling.

## 9. Per-Token Decode

Decode generates one new token at a time.

For each token Rusty runs the current hidden state through all model layers:

1. input RMSNorm
2. q/k/v projections
3. q_norm/k_norm
4. RoPE using the actual token position
5. KV cache append/update
6. causal attention over cached context
7. o projection and residual add
8. post-attention RMSNorm
9. MLP gate/up, SiLU multiply, down projection
10. final residual
11. final norm and logits/top-k
12. token selection, stop checks, and decode metadata

The current promoted backend keeps as much of this path MLX-resident as
practical while preserving scalar metadata for the UI.

## 10. KV Cache

The KV cache stores prior keys and values so decode does not recompute all prior
tokens.

In the LLM chain this is the attention memory. The current default backend is
`chunked_compact_mlx`: compact K/V chunks are stored at key/value-head count and
used by chunk-aware attention. This reduces memory relative to expanded q-head
KV storage while preserving full-context semantics.

Current policy:

- default: full-context chunked compact MLX cache
- CPU mirror: disabled by default
- rotating, quantized, swapped, and recompute cache modes: future or diagnostic
  ideas, not the normal generation path

## 11. Logits, Sampling, And Stop Handling

After the final hidden state, Rusty projects to logits and selects the next
token.

In the LLM chain this is the decision layer. Greedy generation uses top-1.
Sampled generation can use temperature and top-k. EOS and configured stop tokens
end generation. Rusty returns token IDs, raw decoded text, cleaned decoded text,
timing, backend, cache, adapter, and cleanup metadata.

The normal path should avoid full-logit CPU materialization except when a
diagnostic flag asks for it.

## 12. Metadata Back To The UI

Rusty returns generated text plus scalar metadata to CoffeeScript, which writes
the prompt outputs used by the UI.

In the LLM chain this is observability, not model execution. Useful fields
include active attention backend, cache stats, adapter status, timing, tokens
per second, readback reasons, fallback flags, stop reason, and cleanup/session
counts.

The UI can use this metadata to confirm that the intended path ran without
owning or seeing native tensor data.

