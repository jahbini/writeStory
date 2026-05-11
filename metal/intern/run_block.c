// ------------------------------------------------------------
// Helper: attention block (QKV + RoPE + self-attn + O-proj)
// ------------------------------------------------------------
bool run_attention_block(
    metal_llm_model* model,
    int layer,
    const std::vector<float>& h_in_raw,
    const std::vector<float>& h_in_ln,
    const std::vector<float>& cos_cache,
    const std::vector<float>& sin_cache,
    std::vector<float>& h_after_attn_out,
    int pos)
{
    if ((int)h_in_ln.size() != cfg.hidden_size) {
        std::fprintf(stderr,
            "[metal_llm] ERROR: run_attention_block: LN size mismatch\n");
        return false;
    }

    // ---- QKV from normalized input ----
    QKV qkv = compute_qkv_layer_cpu(model, layer, h_in_ln);
    if (qkv.Q.empty()) return false;

    if ((int)qkv.Q.size() != cfg.hidden_size ||
        (int)qkv.K.size() != cfg.hidden_size ||
        (int)qkv.V.size() != cfg.hidden_size)
    {
        std::fprintf(stderr,
            "[metal_llm] ERROR: QKV size mismatch at layer %d\n", layer);
        return false;
    }

    // ----------------------------
    // RoPE: Q rotates over *all attention heads*
    //       K rotates over *KV heads only*
    // Apply RoPE once to the full Q and K blocks (single token, this layer)
    // ----------------------------
    //apply_rope_mlx(qkv.Q.data(), qkv.K.data(), pos);
    // RoPE debug:accomplished: prove we rotated (and only the rotary slice)
    // ----------------------------
    apply_rope_mlx(qkv.Q.data(), qkv.K.data(), pos);

    if (layer == 0) {
         dump_debug_vector(
             "debug_cpp/first_layer0_q_after_rope_pos_" + std::to_string(pos) + ".log",
             qkv.Q,
             "Q after RoPE (cpp)"
         );
         dump_debug_vector(
             "debug_cpp/first_layer0_k_after_rope_pos_" + std::to_string(pos) + ".log",
             qkv.K,
             "K after RoPE (cpp)"
         );
     }
    // ----------------------------
    // Self-attention
    // ----------------------------
    std::vector<float> attn_out(cfg.hidden_size, 0.0f);
    //apply_attention_softmax_cpu(qkv, cfg.num_heads, cfg.head_dim, attn_out);
    apply_attention_softmax_cpu_cached(
        qkv,
        pos,
	layer,
        attn_out
    );

        // ----------------------------
    // O-projection: o = dense(attn_out)
    // ----------------------------
    std::string wname =
        "model.layers." + std::to_string(layer) +
        ".self_attn.dense.weight";

    std::string bname =
        "model.layers." + std::to_string(layer) +
        ".self_attn.dense.bias";

    auto itW = model->weights.find(wname);
    if (itW == model->weights.end()) {
        std::fprintf(stderr, "[metal_llm] ERROR: missing %s\n", wname.c_str());
        return false;
    }

    auto itB = model->weights.find(bname);
    if (itB == model->weights.end()) {
        std::fprintf(stderr, "[metal_llm] ERROR: missing %s\n", bname.c_str());
        return false;
    }

    mx::array& Wmx = itW->second;
    mx::array& Bmx = itB->second;
    mx::eval(Wmx);
    mx::eval(Bmx);

    if (Wmx.ndim() != 2 ||
        (int)Wmx.shape(0) != cfg.hidden_size ||
        (int)Wmx.shape(1) != cfg.hidden_size ||
        Wmx.dtype() != mx::float32)
    {
        std::fprintf(stderr, "[metal_llm] ERROR: dense.weight shape/dtype mismatch\n");
        return false;
    }

    if (Bmx.ndim() != 1 ||
        (int)Bmx.shape(0) != cfg.hidden_size ||
        Bmx.dtype() != mx::float32)
    {
        std::fprintf(stderr, "[metal_llm] ERROR: dense.bias shape/dtype mismatch\n");
        return false;
    }

    const float* Wdense = Wmx.data<float>();   // [out, in]
    const float* Bdense = Bmx.data<float>();   // [out]

    std::vector<float> o_proj(cfg.hidden_size, 0.0f);

    // o_proj[i] = sum_j W[i,j] * attn_out[j] + b[i]
    for (int i = 0; i < cfg.hidden_size; ++i) {
        double sum = (double)Bdense[i];
        const float* Wi = Wdense + i * cfg.hidden_size;
        for (int j = 0; j < cfg.hidden_size; ++j) {
            sum += (double)Wi[j] * (double)attn_out[j];
        }
        o_proj[i] = (float)sum;
    }

    // ----------------------------
    // Residual: h_after_attn = h_in_raw + o_proj
    // ----------------------------
    std::vector<float> h_after_attn(cfg.hidden_size);
    for (int i = 0; i < cfg.hidden_size; ++i) {
        h_after_attn[i] = h_in_raw[i] + o_proj[i];
    }

    h_after_attn_out = h_after_attn;
    return true;
#if 0
 // Phi residual rule: raw input + attention output
    std::vector<float> h_residual(cfg.hidden_size);
    for (int i = 0; i < cfg.hidden_size; ++i)
        h_residual[i] = h_in_raw[i] + attn_out[i];

    // ----------------------------
    // O-projection
    // ----------------------------
    std::string wname =
        "model.layers." + std::to_string(layer) +
        ".self_attn.dense.weight";

    auto it_dense = model->weights.find(wname);
    if (it_dense == model->weights.end()) {
        std::fprintf(stderr,
            "[metal_llm] ERROR: missing %s\n", wname.c_str());
        return false;
    }

    mx::array& Wmx = it_dense->second;
    mx::eval(Wmx);

    if (Wmx.ndim() != 2 ||
        (int)Wmx.shape(0) != cfg.hidden_size ||
        (int)Wmx.shape(1) != cfg.hidden_size ||
        Wmx.dtype() != mx::float32)
    {
        std::fprintf(stderr,
            "[metal_llm] ERROR: dense.weight shape/dtype mismatch\n");
        return false;
    }

    const float* Wdense = Wmx.data<float>();

    std::vector<float> attn_proj(cfg.hidden_size, 0.0f);
    for (int i = 0; i < cfg.hidden_size; ++i) {
        float sum = 0.0f;
        const float* Wi = Wdense + i * cfg.hidden_size;
        for (int j = 0; j < cfg.hidden_size; ++j)
            sum += Wi[j] * h_residual[j];
        attn_proj[i] = sum;
    }

    h_after_attn_out = attn_proj;
    return true;
#endif
}
