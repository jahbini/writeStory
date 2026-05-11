// Phi-1.5 RoPE params (from config.json)
// NeoX-style rotary embedding used by Phi-1.5
// Build RoPE cache (NeoX-style, Phi-1.5: partial_rotary_factor * head_dim)
void build_rope_cache_neox(
    int rotary_dim,
    int pos,
    std::vector<float>& cos_cache,
    std::vector<float>& sin_cache)
{
    // Ensure both caches have exactly rotary_dim entries
    cos_cache.resize(rotary_dim);
    sin_cache.resize(rotary_dim);

    const int half = rotary_dim / 2;

    // We treat every *pair* (2*i, 2*i+1) as one complex component.
    // GPT-NeoX / Phi-1.5 convention:
    //   inv_freq[i] = rope_theta^(-(2*i / rotary_dim))
    for (int i = 0; i < half; ++i) {
        float exponent = 2.0f * static_cast<float>(i)
                       / static_cast<float>(rotary_dim);
        float inv_freq = std::pow(cfg.rope_theta, -exponent);
        float angle    = static_cast<float>(pos) * inv_freq;

        float c = std::cos(angle);
        float s = std::sin(angle);

        int j = 2 * i;
        cos_cache[j]     = c;
        cos_cache[j + 1] = c;
        sin_cache[j]     = s;
        sin_cache[j + 1] = s;
    }
}

// ------------------------------------------------------------
// Build RoPE cache (MLX / LLaMA style, split-half)
// ------------------------------------------------------------
void build_rope_cache_mlx(
    int rotary_dim,
    int pos,
    std::vector<float>& cos_cache,
    std::vector<float>& sin_cache)
{
    const int half = rotary_dim / 2;

    // One cos/sin per half-dimension
    cos_cache.resize(half);
    sin_cache.resize(half);

    const float log_base = std::log(cfg.rope_theta);
    const float fpos = static_cast<float>(pos);

    for (int i = 0; i < half; ++i) {
        // inv_freq = base ** (-i / half)
        float exponent = -static_cast<float>(i) / static_cast<float>(half);
        float inv_freq = std::exp(log_base * exponent);

        float angle = fpos * inv_freq;

        cos_cache[i] = std::cos(angle);
        sin_cache[i] = std::sin(angle);
    }
}
