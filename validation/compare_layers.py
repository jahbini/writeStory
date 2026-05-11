#!/usr/bin/env python3
import argparse
import os
import numpy as np
import mlx.core as mx
from mlx_lm.utils import load as mlx_load

# ------------------------------------------------------------
# Fixed output directory
# ------------------------------------------------------------
OUTDIR = "debug_py"
os.makedirs(OUTDIR, exist_ok=True)


# ------------------------------------------------------------
# Utility: write array to file as "i: value"
# ------------------------------------------------------------
def write_vector(path, vec, header=None):
    with open(path, "w") as f:
        if header:
            f.write(header + "\n")
        for i, v in enumerate(vec):
            f.write(f"{i}: {float(v):.12f}\n")


# ------------------------------------------------------------
# RoPE (NeoX-style, pair-wise, matching C++ apply_rope_neox)
# ------------------------------------------------------------
def build_rope_neox(rotary_dim, pos):
    """
    Mirror the C++ version:

        for (int i = 0; i < rotary_dim; i += 2) {
            float inv_freq = std::pow(1000000.0f, -((float)i) / rotary_dim);
            float angle = position * inv_freq;
            cos[i]   = cos[i+1] = std::cos(angle);
            sin[i]   = sin[i+1] = std::sin(angle);
        }
    """
    cos = np.zeros(rotary_dim, dtype=np.float32)
    sin = np.zeros(rotary_dim, dtype=np.float32)

    for i in range(0, rotary_dim, 2):
        inv_freq = 10000.0 ** -( float(i) / rotary_dim)
        angle = float(pos) * inv_freq
        c = np.cos(angle).astype(np.float32)
        s = np.sin(angle).astype(np.float32)
        cos[i] = cos[i + 1] = c
        sin[i] = sin[i + 1] = s

    return cos, sin


def apply_rope_to_vec(x, cos_cache, sin_cache, rotary_dim):
    """
    In-place style, but we return a new vector for clarity.
    x is a 1D numpy slice for a single head (length >= rotary_dim).
    """
    x = x.copy()
    for i in range(0, rotary_dim, 2):
        x0 = x[i]
        x1 = x[i + 1]
        c = cos_cache[i]
        s = sin_cache[i]
        x[i]     = x0 * c - x1 * s
        x[i + 1] = x1 * c + x0 * s
    return x


# ------------------------------------------------------------
# Q/K/V + RoPE (Python-side reproduction of the C++ path)
# ------------------------------------------------------------
def compute_qkv_rope(model, layer_id, h_ln_np, pos):
    """
    h_ln_np : numpy 1D, LN output (hidden_dim)
    pos     : integer position (e.g. len(tokens) - 1)
    """
    layer = model.model.layers[layer_id]

    # Projections as float32 numpy
    Wq = np.array(layer.self_attn.q_proj.weight, dtype=np.float32)  # [hidden, hidden]
    bq = np.array(layer.self_attn.q_proj.bias,   dtype=np.float32)

    Wk = np.array(layer.self_attn.k_proj.weight, dtype=np.float32)
    bk = np.array(layer.self_attn.k_proj.bias,   dtype=np.float32)

    Wv = np.array(layer.self_attn.v_proj.weight, dtype=np.float32)
    bv = np.array(layer.self_attn.v_proj.bias,   dtype=np.float32)

    # Raw Q/K/V (same math as C++)
    q = h_ln_np @ Wq.T + bq
    k = h_ln_np @ Wk.T + bk
    v = h_ln_np @ Wv.T + bv

    write_vector(f"{OUTDIR}/python_q_proj.log", q, "[python] raw Q")
    write_vector(f"{OUTDIR}/python_k_proj.log", k, "[python] raw K")
    write_vector(f"{OUTDIR}/python_v_proj.log", v, "[python] raw V")

    # RoPE parameters from the MLX attention module
    attn = layer.self_attn
    n_heads = int(attn.num_heads)
    head_dim = int(attn.head_dim)
    rotary_dim = int( head_dim * attn.partial_rotary_factor )

    cos_cache, sin_cache = build_rope_neox(rotary_dim, pos)

    q2 = q.copy()
    k2 = k.copy()

    print("DEBUG rope:", "pos=", pos, "rotary_dim=", rotary_dim, "theta=", getattr(attn, "rope_theta", None),
          "rope_obj=", getattr(attn, "rope", None))
    print("DEBUG cos/sin[0..3] =", cos_cache[:4], sin_cache[:4])
    # Apply RoPE per head, exactly like C++ loop over heads
    for h in range(n_heads):
        start = h * head_dim
        end = start + head_dim

        q2[start:end] = apply_rope_to_vec(q2[start:end], cos_cache, sin_cache, rotary_dim)
        k2[start:end] = apply_rope_to_vec(k2[start:end], cos_cache, sin_cache, rotary_dim)

    write_vector( f"{OUTDIR}/first_layer0_q_after_rope_pos_{pos}.log", q2, "[python] Q after RoPE")

    write_vector( f"{OUTDIR}/first_layer0_k_after_rope_pos_{pos}.log", k2, "[python] K after RoPE")

    # Combined QKV file (raw only — same style as earlier harness)
    with open(f"{OUTDIR}/python_qkv_layer0.log", "w") as f:
        f.write("[python] Q projection\n")
        for i, x in enumerate(q):
            f.write(f"{i}: {x:.12f}\n")

        f.write("\n[python] K projection\n")
        for i, x in enumerate(k):
            f.write(f"{i}: {x:.12f}\n")

        f.write("\n[python] V projection\n")
        for i, x in enumerate(v):
            f.write(f"{i}: {x:.12f}\n")

    return q2, k2, v


# ------------------------------------------------------------
# Main CLI workflow (fixed CLI, fixed OUTDIR)
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--dump_logits", action="store_true")
    args = parser.parse_args()

    # Load local MLX model + tokenizer
    model, tokenizer = mlx_load(args.model)

    # Tokenize prompt with tokenizer.encode (TokenizerWrapper is NOT callable)
    tokens_list = tokenizer.encode(args.prompt)
    print("Python token IDs:", tokens_list, args.prompt)

    # Last token ID as a plain Python int (no numpy yet)
    last_id = int(tokens_list[-1])

    # --------------------------------------------------------
    # Raw embedding (row from embed_tokens.weight)
    # --------------------------------------------------------
    # embed_tokens.weight: MLX array [vocab, hidden]
    embed_weight = np.array(model.model.embed_tokens.weight, dtype=np.float32)
    embed_raw = embed_weight[last_id]          # shape [hidden]

    write_vector(
        f"{OUTDIR}/python_embed_raw.log",
        embed_raw,
        "[python] raw embedding",
    )

    # --------------------------------------------------------
    # Input LayerNorm (numpy-only, matches C++ manual LN)
    # --------------------------------------------------------
    layer = model.model.layers[args.layer]
    ln_w = np.array(layer.input_layernorm.weight, dtype=np.float32)  # [hidden]
    ln_b = np.array(layer.input_layernorm.bias,   dtype=np.float32)  # [hidden]

    # Standard LN over embed_raw, then scale/shift
    mean = embed_raw.mean()
    var = ((embed_raw - mean) ** 2).mean()
    inv_std = 1.0 / np.sqrt(var + 1e-5)

    norm = (embed_raw - mean) * inv_std
    h_ln_np = norm * ln_w + ln_b

    write_vector(
        f"{OUTDIR}/python_input_ln.log",
        h_ln_np,
        "[python] input LN",
    )

    # --------------------------------------------------------
    # Q/K/V + RoPE for the chosen layer
    # --------------------------------------------------------
    pos = len(tokens_list) - 1
    compute_qkv_rope(model, args.layer, h_ln_np, pos)

    # --------------------------------------------------------
    # Full-model logits for last token (optional)
    # --------------------------------------------------------
    if args.dump_logits:
        # Shape: (1, seq)
        tokens_mx = mx.array([tokens_list], dtype=mx.int32)

        out = model(tokens_mx)
        # Some MLX models return (logits, cache), some just logits
        if isinstance(out, (tuple, list)):
            logits_mx = out[0]
        else:
            logits_mx = out

        # logits: [batch, seq, vocab]
        logits_np = np.array(logits_mx[0, -1], dtype=np.float32)
        write_vector(
            f"{OUTDIR}/python_logits.log",
            logits_np,
            "[python] logits",
        )


if __name__ == "__main__":
    main()
