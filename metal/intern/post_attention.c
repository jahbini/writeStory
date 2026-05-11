// ------------------------------------------------------------
// Helper: optional post-attention layernorm
// ------------------------------------------------------------
bool run_post_attention_layernorm(
    metal_llm_model* model,
    int layer,
    const std::vector<float>& h_after_attn,
    std::vector<float>& h_norm_out)
{
    std::string lnW = layer_key(layer, "post_attention_layernorm.weight");
    std::string lnB = layer_key(layer, "post_attention_layernorm.bias");

    bool has_lnW = (model->weights.find(lnW) != model->weights.end());
    bool has_lnB = (model->weights.find(lnB) != model->weights.end());

    if (has_lnW && has_lnB) {
        if (!apply_layernorm_cpu(model,
                                 lnW.c_str(),
                                 lnB.c_str(),
                                 h_after_attn,
                                 h_norm_out))
        {
            std::fprintf(stderr,
                "[metal_llm] ERROR: failed post-attention LN for layer %d\n",
                layer);
            return false;
        }
    } else {
        // Phi style: no post-attention LN
        h_norm_out = h_after_attn;
    }

    return true;
}

