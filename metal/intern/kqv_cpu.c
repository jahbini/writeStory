// ------------------------------------------------------------
// compute_qkv_layer_cpu
//
// Q = hvec * Wq^T + bq
// K = hvec * Wk^T + bk
// V = hvec * Wv^T + bv
//
// Wq/Wk/Wv are float32 [cfg.hidden_size, cfg.hidden_size]
// biases (optional) are float32 [cfg.hidden_size]
// ------------------------------------------------------------
bool dumped_first_qkv = false;
QKV compute_qkv_layer_cpu(
    metal_llm_model* model,
    int layer,
    const std::vector<float>& hvec)
{
    QKV out;
    if (!model || hvec.empty()) return out;


    // keys like "model.layers.<layer>.self_attn.q_proj.weight"
    std::string k_qw = layer_key(layer, "self_attn.q_proj.weight");
    std::string k_kw = layer_key(layer, "self_attn.k_proj.weight");
    std::string k_vw = layer_key(layer, "self_attn.v_proj.weight");

    auto it_qw = model->weights.find(k_qw);
    auto it_kw = model->weights.find(k_kw);
    auto it_vw = model->weights.find(k_vw);

    if (it_qw == model->weights.end() ||
        it_kw == model->weights.end() ||
        it_vw == model->weights.end())
    {
        std::fprintf(stderr,
            "[metal_llm] ERROR: missing Q/K/V projection weights for %s / %s / %s\n",
            k_qw.c_str(), k_kw.c_str(), k_vw.c_str());
        return out;
    }

    mx::array Wq = it_qw->second;
    mx::array Wk = it_kw->second;
    mx::array Wv = it_vw->second;

    mx::eval(Wq);
    mx::eval(Wk);
    mx::eval(Wv);

    if (Wq.ndim() != 2 || Wk.ndim() != 2 || Wv.ndim() != 2) {
        std::fprintf(stderr,
            "[metal_llm] ERROR: QKV ndim != 2 (q=%d k=%d v=%d)\n",
            (int)Wq.ndim(), (int)Wk.ndim(), (int)Wv.ndim());
        return out;
    }

    if (Wq.shape(1) != cfg.hidden_size ||
        Wk.shape(1) != cfg.hidden_size ||
        Wv.shape(1) != cfg.hidden_size)
    {
        std::fprintf(stderr,
            "[metal_llm] ERROR: QKV weight cols != cfg.hidden_size (q=%d k=%d v=%d hidden=%d)\n",
            (int)Wq.shape(1), (int)Wk.shape(1), (int)Wv.shape(1),
            cfg.hidden_size);
        return out;
    }

    if (Wq.dtype() != mx::float32 ||
        Wk.dtype() != mx::float32 ||
        Wv.dtype() != mx::float32)
    {
        std::fprintf(stderr,
            "[metal_llm] ERROR: QKV weights not float32 after upcast\n");
        return out;
    }

    const float* Wq_ptr = Wq.data<float>();
    const float* Wk_ptr = Wk.data<float>();
    const float* Wv_ptr = Wv.data<float>();

    // ------------------------------------------------
    // Load biases (optional)
    // ------------------------------------------------
    std::vector<float> Bq(cfg.hidden_size, 0.0f);
    std::vector<float> Bk(cfg.hidden_size, 0.0f);
    std::vector<float> Bv(cfg.hidden_size, 0.0f);

    auto load_bias = [&](const std::string& key, std::vector<float>& dst) {
        auto it = model->weights.find(key);
        if (it == model->weights.end()) return; // optional

        mx::array B = it->second;
        mx::eval(B);

        if (B.ndim() != 1 || (int)B.shape(0) != cfg.hidden_size ||
            B.dtype() != mx::float32)
        {
            std::fprintf(stderr,
                "[metal_llm] ERROR: %s bias shape/dtype mismatch (size=%d, hidden=%d)\n",
                key.c_str(), (int)B.shape(0), cfg.hidden_size);
            return;
        }

        const float* p = B.data<float>();
        for (int i = 0; i < cfg.hidden_size; ++i) dst[i] = p[i];
    };

    load_bias(layer_key(layer, "self_attn.q_proj.bias"), Bq);
    load_bias(layer_key(layer, "self_attn.k_proj.bias"), Bk);
    load_bias(layer_key(layer, "self_attn.v_proj.bias"), Bv);

    // ------------------------------------------------
    // Allocate outputs
    // ------------------------------------------------
    out.Q.resize(cfg.hidden_size);
    out.K.resize(cfg.hidden_size);
    out.V.resize(cfg.hidden_size);

    // ------------------------------------------------
    // CPU matmul loop
    // ------------------------------------------------
    for (int o = 0; o < cfg.hidden_size; ++o) {
        double acc_q = 0.0;
        double acc_k = 0.0;
        double acc_v = 0.0;

        const float* row_q = Wq_ptr + (size_t)o * cfg.hidden_size;
        const float* row_k = Wk_ptr + (size_t)o * cfg.hidden_size;
        const float* row_v = Wv_ptr + (size_t)o * cfg.hidden_size;

        for (int i = 0; i < cfg.hidden_size; ++i) {
            float h = hvec[i];
            acc_q += h * row_q[i];
            acc_k += h * row_k[i];
            acc_v += h * row_v[i];
        }

        acc_q += Bq[o];
        acc_k += Bk[o];
        acc_v += Bv[o];

        out.Q[o] = (float)acc_q;
        out.K[o] = (float)acc_k;
        out.V[o] = (float)acc_v;
    }

// ----------------------------------------------------------
// DEBUG: Dump only for the *first* token, only once.
// Requires that metal.forwardStep passes in `pos`.
// ----------------------------------------------------------
if (!dumped_first_qkv ) {
    dumped_first_qkv = true;

    char fname[256];

    // dump Q
    std::snprintf(fname, sizeof(fname), "debug_cpp/first_layer%d_q_proj.log", layer);
    std::FILE* fq = std::fopen(fname, "w");
    for (int i = 0; i < cfg.hidden_size; ++i)
        std::fprintf(fq, "%d: %.12f\n", i, out.Q[i]);
    std::fclose(fq);

    // dump K
    std::snprintf(fname, sizeof(fname), "debug_cpp/first_layer%d_k_proj.log", layer);
    std::FILE* fk = std::fopen(fname, "w");
    for (int i = 0; i < cfg.hidden_size; ++i)
        std::fprintf(fk, "%d: %.12f\n", i, out.K[i]);
    std::fclose(fk);

    // dump V
    std::snprintf(fname, sizeof(fname), "debug_cpp/first_layer%d_v_proj.log", layer);
    std::FILE* fv = std::fopen(fname, "w");
    for (int i = 0; i < cfg.hidden_size; ++i)
        std::fprintf(fv, "%d: %.12f\n", i, out.V[i]);
    std::fclose(fv);

    std::fprintf(stderr, "[metal_llm][debug] Wrote FIRST QKV dump for layer %d\n", layer);
}
    return out;   // unchanged
}

