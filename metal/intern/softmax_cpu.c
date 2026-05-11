void apply_attention_softmax_cpu_cached(
    const QKV& qkv,     // current token Q,K,V (after RoPE)
    int pos,
    int layer,
    std::vector<float>& attn_out)
{
    const int q_heads  = cfg.num_heads;
    const int kv_heads =
        (cfg.num_kv_heads > 0) ? cfg.num_kv_heads : cfg.num_heads;
    const int head_dim = cfg.head_dim;
    const int max_pos  = cfg.max_position_embeddings;

    if (pos >= max_pos) {
        std::fprintf(stderr,
            "[metal_llm] ERROR: KV cache overflow (pos=%d, max=%d)\n",
            pos, max_pos);
        attn_out.clear();
        return;
    }

    attn_out.assign(cfg.hidden_size, 0.0f);

    // ---- Layer slice of KV cache
    float* Kcache = model->kv_cache_k.data()
        + layer * kv_heads * max_pos * head_dim;

    float* Vcache = model->kv_cache_v.data()
        + layer * kv_heads * max_pos * head_dim;
    // ------------------------------------------------------------
    // 1. Write current K,V into cache at position pos
    // ------------------------------------------------------------
    for (int h = 0; h < kv_heads; ++h) {
        float* kdst = Kcache + ((h * max_pos + pos) * head_dim);
        float* vdst = Vcache + ((h * max_pos + pos) * head_dim);

        const float* ksrc = qkv.K.data() + h * head_dim;
        const float* vsrc = qkv.V.data() + h * head_dim;

        std::memcpy(kdst, ksrc, head_dim * sizeof(float));
        std::memcpy(vdst, vsrc, head_dim * sizeof(float));
    }

    if (layer == 0 && pos == 1) {
        std::fprintf(stderr,
            "[KV CHECK] L0 K[0][1][0]=%.6f\n",
            Kcache[(0 * max_pos + 1) * head_dim + 0]);
    }

    if (layer == 1 && pos == 1) {
        std::fprintf(stderr,
            "[KV CHECK] L1 K[0][1][0]=%.6f\n",
            Kcache[(0 * max_pos + 1) * head_dim + 0]);
    }

    if (pos == 1 && (layer == 0 || layer == 1)) {
        const float* ksrc0 = qkv.K.data() + 0 * head_dim;
        const float  kdst0 = Kcache[(0 * max_pos + pos) * head_dim + 0];

        std::fprintf(stderr,
            "[KV CHECK] L%d pos=%d  ksrc0=%.6f  kdst0=%.6f\n",
            layer, pos, ksrc0[0], kdst0);
    }
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const int   T     = pos + 1;

    // ------------------------------------------------------------
    // 2. Attention per query head
    // ------------------------------------------------------------
    for (int h = 0; h < q_heads; ++h) {
        const int kv_h = (h < kv_heads) ? h : (h % kv_heads);

        const float* qh = qkv.Q.data() + h * head_dim;

        std::vector<float> scores(T);

        float maxScore = -1e30f;
        for (int t = 0; t < T; ++t) {
            const float* kh =
                Kcache + ((kv_h * max_pos + t) * head_dim);

            double dot = 0.0;
            for (int i = 0; i < head_dim; ++i)
                dot += (double)qh[i] * (double)kh[i];

            float s = (float)dot * scale;
            scores[t] = s;
            if (s > maxScore) maxScore = s;
        }

        float sumExp = 0.0f;
        for (int t = 0; t < T; ++t) {
            float e = std::exp(scores[t] - maxScore);
            scores[t] = e;
            sumExp += e;
        }

        const float invSum = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;

        float* out_h = attn_out.data() + h * head_dim;
        for (int i = 0; i < head_dim; ++i) {
            double acc = 0.0;
            for (int t = 0; t < T; ++t) {
                const float* vh =
                    Vcache + ((kv_h * max_pos + t) * head_dim);
                acc += (double)(scores[t] * invSum) * (double)vh[i];
            }
            out_h[i] = (float)acc;
        }
    }
}
