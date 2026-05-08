#!/usr/bin/env python3
"""Reference next-token logits for a fixed token-id prompt using mlx_lm.

This script is intentionally standalone and is not run by the normal verifier.
It prints JSON so `RUSTY_VERIFY_PROFILE=parity` can compare Rusty against the
reference by setting `RUSTY_PARITY_REFERENCE_JSON` to the saved output path.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _parse_token_ids(raw: str) -> list[int]:
    tokens: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        tokens.append(int(part))
    if not tokens:
        raise ValueError("token id list is empty")
    return tokens


def _checksum(values) -> float:
    flat = values.reshape(-1)
    count = min(int(flat.size), 1024)
    if count <= 0:
        return 0.0
    return float(flat[:count].sum().item())


def _summary(mx_array) -> dict:
    import mlx.core as mx

    mx.eval(mx_array)
    flat = mx_array.reshape(-1)
    first_count = min(int(flat.size), 8)
    first_values = [float(flat[i].item()) for i in range(first_count)]
    return {
        "shape": list(mx_array.shape),
        "checksum": _checksum(mx_array),
        "first_values": first_values,
    }


def _attr_path(obj, *names):
    current = obj
    for name in names:
        if current is None or not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current


def _reference_stage_dump(model, token_ids: list[int]) -> dict:
    import mlx.core as mx
    import mlx.nn as nn

    inner = _attr_path(model, "model") or model
    embed = (
        _attr_path(inner, "embed_tokens")
        or _attr_path(inner, "wte")
        or _attr_path(inner, "tok_embeddings")
    )
    layers = _attr_path(inner, "layers")
    if embed is None or layers is None:
        return {
            "available": False,
            "error": "could not locate model embed_tokens/layers attributes",
            "model_type": str(type(model)),
            "inner_type": str(type(inner)),
        }

    tokens = mx.array([token_ids], dtype=mx.uint32)
    hidden = embed(tokens)
    mx.eval(hidden)
    result = {
        "available": True,
        "embedding_token0": _summary(hidden[:, 0, :]),
        "embedding_final_prompt_token": _summary(hidden[:, -1, :]),
    }

    try:
        layer0 = layers[0]
        input_layernorm = _attr_path(layer0, "input_layernorm")
        self_attn = _attr_path(layer0, "self_attn") or _attr_path(layer0, "attention")
        if input_layernorm is None or self_attn is None:
            result["layer0_available"] = False
            result["layer0_error"] = "could not locate layer0 input_layernorm/self_attn"
            return result
        norm = input_layernorm(hidden)
        result["layer0_input_rmsnorm_final_position"] = _summary(norm[:, -1, :])
        q_proj = _attr_path(self_attn, "q_proj")
        k_proj = _attr_path(self_attn, "k_proj")
        v_proj = _attr_path(self_attn, "v_proj")
        o_proj = _attr_path(self_attn, "o_proj")
        q_norm = _attr_path(self_attn, "q_norm")
        k_norm = _attr_path(self_attn, "k_norm")
        rope = _attr_path(self_attn, "rope")
        if q_proj is not None and k_proj is not None and v_proj is not None:
            bsz, seq_len, _dim = norm.shape
            n_heads = int(getattr(self_attn, "n_heads"))
            n_kv_heads = int(getattr(self_attn, "n_kv_heads"))
            q_raw = q_proj(norm)
            k_raw = k_proj(norm)
            v_raw = v_proj(norm)
            result["layer0_q_proj_final_position"] = _summary(q_raw[:, -1, :])
            result["layer0_k_proj_final_position"] = _summary(k_raw[:, -1, :])
            result["layer0_v_proj_final_position"] = _summary(v_raw[:, -1, :])

            q_heads = q_raw.reshape(bsz, seq_len, n_heads, -1)
            k_heads = k_raw.reshape(bsz, seq_len, n_kv_heads, -1)
            v_heads = v_raw.reshape(bsz, seq_len, n_kv_heads, -1).transpose(0, 2, 1, 3)
            if q_norm is not None:
                q_heads = q_norm(q_heads)
                result["layer0_q_after_q_norm_final_position"] = _summary(q_heads[:, -1, :, :].reshape(1, -1))
            if k_norm is not None:
                k_heads = k_norm(k_heads)
                result["layer0_k_after_k_norm_final_position"] = _summary(k_heads[:, -1, :, :].reshape(1, -1))

            q_pre_rope = q_heads.transpose(0, 2, 1, 3)
            k_pre_rope = k_heads.transpose(0, 2, 1, 3)
            result["layer0_q_before_rope_final_position"] = _summary(q_pre_rope[:, :, -1, :].reshape(1, -1))
            result["layer0_k_before_rope_final_position"] = _summary(k_pre_rope[:, :, -1, :].reshape(1, -1))

            if rope is not None:
                q_rope = rope(q_pre_rope)
                k_rope = rope(k_pre_rope)
                result["layer0_q_after_rope_final_position"] = _summary(q_rope[:, :, -1, :].reshape(1, -1))
                result["layer0_k_after_rope_final_position"] = _summary(k_rope[:, :, -1, :].reshape(1, -1))
            else:
                q_rope = q_pre_rope
                k_rope = k_pre_rope

            q0 = q_rope[0, 0, -1, :]
            k0 = k_rope[0, 0, :, :]
            scale = float(getattr(self_attn, "scale", q0.shape[-1] ** -0.5))
            scores0 = mx.sum(k0 * q0[None, :], axis=-1) * scale
            probs0 = mx.softmax(scores0, axis=-1)
            result["layer0_attention_scores_head0_final_position"] = _summary(scores0)
            result["layer0_attention_probabilities_head0_final_position"] = _summary(probs0)

            # Manual GQA attention for final prompt position.
            repeat = n_heads // n_kv_heads
            per_head = []
            for h in range(n_heads):
                kv_h = h // repeat
                qh = q_rope[0, h, -1, :]
                kh = k_rope[0, kv_h, :, :]
                vh = v_heads[0, kv_h, :, :]
                scores = mx.sum(kh * qh[None, :], axis=-1) * scale
                probs = mx.softmax(scores, axis=-1)
                per_head.append(mx.sum(vh * probs[:, None], axis=0))
            attn_final = mx.stack(per_head, axis=0).reshape(1, 1, -1)
            result["layer0_attention_output_final_position"] = _summary(attn_final.reshape(1, -1))
            if o_proj is not None:
                o_out = o_proj(attn_final)
                result["layer0_o_proj_output_final_position"] = _summary(o_out[:, -1, :])
                post_attn_residual = hidden[:, -1:, :] + o_out
                result["layer0_post_attention_residual_final_position"] = _summary(post_attn_residual[:, -1, :])
                post_norm_layer = _attr_path(layer0, "post_attention_layernorm")
                mlp = _attr_path(layer0, "mlp")
                if post_norm_layer is not None and mlp is not None:
                    post_norm = post_norm_layer(post_attn_residual)
                    result["layer0_post_attention_rmsnorm_final_position"] = _summary(post_norm[:, -1, :])
                    mlp_out = mlp(post_norm)
                    result["layer0_mlp_output_final_position"] = _summary(mlp_out[:, -1, :])
                    result["layer0_manual_final_residual_final_position"] = _summary((post_attn_residual + mlp_out)[:, -1, :])
        # Full layer call signatures vary across mlx_lm releases. Try the simple
        # form only; failure is diagnostic, not fatal.
        try:
            layer_output = layer0(hidden)
            if isinstance(layer_output, tuple):
                layer_output = layer_output[0]
            result["layer0_final_residual_final_position"] = _summary(layer_output[:, -1, :])
        except Exception as err:
            result["layer0_final_residual_error"] = str(err)
    except Exception as err:
        result["layer0_available"] = False
        result["layer0_error"] = str(err)
    return result


def main() -> int:
    if len(sys.argv) < 3:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "usage",
                    "message": "usage: reference_logits_mlx_lm.py MODEL_DIR TOKEN_IDS_CSV",
                }
            )
        )
        return 2

    model_dir = str(Path(sys.argv[1]).expanduser().resolve())
    token_ids = _parse_token_ids(sys.argv[2])

    started = time.perf_counter()
    import mlx.core as mx
    from mlx_lm import load

    model, _tokenizer = load(model_dir)
    load_ms = (time.perf_counter() - started) * 1000.0

    input_ids = mx.array([token_ids], dtype=mx.uint32)
    forward_started = time.perf_counter()
    output = model(input_ids)
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output
    mx.eval(logits)
    forward_ms = (time.perf_counter() - forward_started) * 1000.0

    last_logits = logits[0, -1, :]
    try:
        indices = mx.argpartition(-last_logits, kth=10)[:10]
        unsorted_values = last_logits[indices]
        order = mx.argsort(-unsorted_values)
        indices = indices[order]
        values = unsorted_values[order]
        mx.eval(values, indices)
        top_logits = [
            {"token_id": int(indices[i].item()), "score": float(values[i].item())}
            for i in range(10)
        ]
    except Exception:
        # Compatibility fallback for older MLX Python APIs with limited top-k
        # helpers. This is reference-only and intentionally not used in Rusty.
        import numpy as np

        logits_np = np.array(last_logits)
        top_indices = np.argpartition(-logits_np, 10)[:10]
        top_indices = top_indices[np.argsort(-logits_np[top_indices])]
        top_logits = [
            {"token_id": int(index), "score": float(logits_np[index])}
            for index in top_indices
        ]

    print(
        json.dumps(
            {
                "ok": True,
                "model_dir": model_dir,
                "prompt_token_ids": token_ids,
                "logits_len": int(last_logits.size),
                "logits_checksum": _checksum(last_logits),
                "top_logits": top_logits,
                "reference_checkpoints": _reference_stage_dump(model, token_ids),
                "timing_ms": {
                    "load": load_ms,
                    "forward": forward_ms,
                },
                "notes": [
                    "Reference uses mlx_lm.load and runs the full prompt token sequence.",
                    "final_hidden_checksum is not exposed by this script.",
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
