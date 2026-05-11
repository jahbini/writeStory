import mlx.core as mx
import numpy as np
from mlx_lm.utils import load as mlx_load
from pathlib import Path

# Load model
model, _ = mlx_load(str(Path(__file__).resolve().parent.parent / "loraland" / "phi-1_5"))
attn = model.model.layers[0].self_attn
rope = attn.rope

head_dim = attn.head_dim

# Salt: 1..6 then zeros
salt = np.zeros(head_dim, dtype=np.float32)
salt[:6] = [1, 2, 3, 4, 5, 6]

# Shape required by MLX RoPE: [B, T, H, D]
x = mx.array(
    salt.reshape(1, 1, 1, head_dim),
    dtype=mx.float32
)

for pos in range(4):
    y = rope(x, pos)
    mx.eval(y)

    print(f"\npos={pos}")
    for i, v in enumerate(y.reshape(-1).tolist()):
        print(f"{i}: {v}")
