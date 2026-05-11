// ------------------------------------------------------------
// Generic LayerNorm on CPU with learned gamma/beta
// ------------------------------------------------------------
bool apply_layernorm_cpu(
    metal_llm_model* model,
    const char* weight_key,
    const char* bias_key,
    const std::vector<float>& h_in,
    std::vector<float>& h_out)
{
    if (!model) {
        log_err("apply_layernorm_cpu", "null model");
        return false;
    }

    auto it_w = model->weights.find(weight_key);
    auto it_b = model->weights.find(bias_key);

    if (it_w == model->weights.end() || it_b == model->weights.end()) {
        std::fprintf(stderr,
                "[metal_llm] ERROR: missing layernorm weights/bias: %s / %s\n",
                weight_key, bias_key);
        return false;
    }

    mx::array& ln_w_arr = it_w->second;
    mx::array& ln_b_arr = it_b->second;

    F32VecRef gamma, beta;
    if (!get_f32_vector_1d(ln_w_arr, weight_key, gamma) ||
        !get_f32_vector_1d(ln_b_arr, bias_key, beta))
    {
        return false;
    }

    if (gamma.size != cfg.hidden_size || beta.size != cfg.hidden_size) {
        std::fprintf(stderr,
            "[metal_llm] ERROR: layernorm size mismatch (%s,%s) "
            "gamma=%d beta=%d hidden=%d\n",
            weight_key, bias_key,
            gamma.size, beta.size, cfg.hidden_size);
        return false;
    }

    h_out.resize(cfg.hidden_size);

    // mean
    double mean = 0.0;
    for (int i = 0; i < cfg.hidden_size; ++i) {
        mean += static_cast<double>(h_in[i]);
    }
    mean /= static_cast<double>(cfg.hidden_size);

    // variance
    double var = 0.0;
    for (int i = 0; i < cfg.hidden_size; ++i) {
        double diff = static_cast<double>(h_in[i]) - mean;
        var += diff * diff;
    }
    var /= static_cast<double>(cfg.hidden_size);

    double inv_std = 1.0 / std::sqrt(var + 1e-5);

    for (int i = 0; i < cfg.hidden_size; ++i) {
        float norm = static_cast<float>(
            (static_cast<double>(h_in[i]) - mean) * inv_std
        );
        h_out[i] = norm * gamma.ptr[i] + beta.ptr[i];
    }

    return true;
}

