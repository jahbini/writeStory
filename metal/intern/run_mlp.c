// ------------------------------------------------------------
// Helper: run MLP + residual
// h_next = h_after_attn + MLP(h_norm)
// ------------------------------------------------------------
bool run_mlp_with_residual(
    metal_llm_model* model,
    int layer,
    const std::vector<float>& h_after_attn,
    const std::vector<float>& h_norm,
    std::vector<float>& h_next_out)
{
    std::vector<float> h_mlp;
    if (!apply_mlp_layer_cpu(model, layer, h_norm, h_mlp))
        return false;

    if (h_mlp.size() != h_after_attn.size()) {
        std::fprintf(stderr,
            "[metal_llm] ERROR: MLP output size mismatch at layer %d\n",
            layer);
        return false;
    }

    h_next_out.resize(cfg.hidden_size);

    for (int i = 0; i < cfg.hidden_size; ++i)
        h_next_out[i] = h_after_attn[i] + h_mlp[i];

    return true;
}

