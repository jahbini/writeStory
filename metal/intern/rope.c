
inline void apply_rope_neox(
    float* x,
    const std::vector<float>& cos,
    const std::vector<float>& sin,
    int rotary_dim)
{
    int half = rotary_dim / 2;

    // rotate the FIRST rotary_dim dims
    for (int i = 0; i < half; ++i) {
        float xi = x[2*i];
        float xj = x[2*i + 1];

        float c = cos[2*i];
        float s = sin[2*i];

        float xr =  xi * c - xj * s;
        float yr =  xi * s + xj * c;

        x[2*i]     = xr;
        x[2*i + 1] = yr;
    }

    // DO NOT TOUCH dims rotary_dim .. head_dim-1
    // they pass through unchanged
}

// ------------------------------------------------------------
// Apply RoPE (MLX / LLaMA style, split-half rotation)
// ------------------------------------------------------------
static void apply_rope_mlx(
    float* q,
    float* k,
    int pos
)
{
    // q: [num_heads * head_dim]
    // k: [num_kv_heads * head_dim]
    mx::array q_mx(q,
        {1, 1,  cfg.num_heads, cfg.head_dim},
        mx::float32
    );

    mx::array k_mx(k,
        {1, 1,  cfg.num_kv_heads, cfg.head_dim},
        mx::float32
    );

    // Apply RoPE to Q
    mx::array q_out = mx::fast::rope(
        q_mx,
        cfg.rotary_dim,
        /*traditional=*/false,
        cfg.rope_theta,
        /*scale=*/1.0f,
        pos
    );

    // Apply RoPE to K
    mx::array k_out = mx::fast::rope(
        k_mx,
        cfg.rotary_dim,
        /*traditional=*/false,
        cfg.rope_theta,
        /*scale=*/1.0f,
        pos
    );

    mx::eval(q_out);
    mx::eval(k_out);

    // Copy back (critical)
    std::memcpy(
        q,
        q_out.data<float>(),
        sizeof(float) * cfg.num_heads * cfg.head_dim
    );

    std::memcpy(
        k,
        k_out.data<float>(),
        sizeof(float) * cfg.num_kv_heads * cfg.head_dim
    );
}
