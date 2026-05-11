#!/usr/bin/env python3
import argparse
import os
import numpy as np
import mlx.core as mx
from mlx_lm.utils import load as mlx_load

OUTDIR = "debug_py"
os.makedirs(OUTDIR, exist_ok=True)

def write_vector(path, vec_1d, header=None):
    arr = np.array(vec_1d, dtype=np.float32).reshape(-1)
    with open(path, "w") as f:
        if header:
            f.write(header + "\n")
        for i, v in enumerate(arr):
            f.write(f"{i}: {float(v):.12f}\n")

def linear_out(linear, x_2d):
    """
    Use MLX module call so we match MLX exactly.
    x_2d: [1, hidden]
    returns: [1, hidden]
    """
    y = linear(x_2d)
    mx.eval(y)
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pos", type=int, default=0, help="position offset to apply to RoPE")
    args = ap.parse_args()

    model, tok = mlx_load(args.model)

    tokens = tok.encode(args.prompt)
    print("tokens:", tokens, args.prompt)

    # single-token trace (use last token); for multi-token, loop and pass pos=t
    token_id = int(tokens[-1])
    pos = int(args.pos)

    layer = model.model.layers[args.layer]
    attn  = layer.self_attn

    # ---- embed token: [hidden]
    emb_w = model.model.embed_tokens.weight  # [vocab, hidden]
    x0 = emb_w[token_id]                     # [hidden]
    mx.eval(x0)
    write_vector(f"{OUTDIR}/mlx_embed_raw.log", x0, "[mlx] embed raw")

    # ---- make it batchy: [1, hidden]
    x = x0.reshape(1, -1)

    # ---- input LN (use module, not numpy)
    x_ln = layer.input_layernorm(x)          # [1, hidden]
    mx.eval(x_ln)
    write_vector(f"{OUTDIR}/mlx_input_ln.log", x_ln[0], "[mlx] input LN")

    # ---- Q/K/V projections via module calls (matches MLX)
    q = linear_out(attn.q_proj, x_ln)        # [1, hidden]
    k = linear_out(attn.k_proj, x_ln)        # [1, hidden]
    v = linear_out(attn.v_proj, x_ln)        # [1, hidden]

    write_vector(f"{OUTDIR}/mlx_q_proj.log", q[0], "[mlx] raw Q proj")
    write_vector(f"{OUTDIR}/mlx_k_proj.log", k[0], "[mlx] raw K proj")
    write_vector(f"{OUTDIR}/mlx_v_proj.log", v[0], "[mlx] raw V proj")

    # ---- reshape to what RoPE expects
    q_heads  = int(attn.num_heads)
    head_dim = int(attn.head_dim)
    kv_heads = int(getattr(attn, "num_kv_heads", q_heads))

    q4 = q.reshape(1, 1, q_heads, head_dim)
    k4 = k.reshape(1, 1, kv_heads, head_dim)

    # ---- apply MLX RoPE operator exactly
    q_rope = attn.rope(q4, pos)
    k_rope = attn.rope(k4, pos)
    mx.eval(q_rope)
    mx.eval(k_rope)

    # flatten back to [hidden] for file-compat with your comparer
    q_rope_flat = q_rope.reshape(-1)
    k_rope_flat = k_rope.reshape(-1)

    write_vector(f"{OUTDIR}/mlx_q_after_rope_pos_{pos}.log", q_rope_flat, "[mlx] Q after RoPE")
    write_vector(f"{OUTDIR}/mlx_k_after_rope_pos_{pos}.log", k_rope_flat, "[mlx] K after RoPE")

    # optional quick print of the rope config as MLX sees it
    print("MLX attn:",
          "num_heads=", q_heads,
          "num_kv_heads=", kv_heads,
          "head_dim=", head_dim,
          "rotary_dim=", getattr(attn, "rotary_dim", None),
          "partial_rotary_factor=", getattr(attn, "partial_rotary_factor", None),
          "rope_theta=", getattr(attn, "rope_theta", None),
          "rope=", getattr(attn, "rope", None))

if __name__ == "__main__":
    main()
