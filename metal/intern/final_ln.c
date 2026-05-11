// ------------------------------------------------------------
// Helper: final layernorm (if present)
// Tries common Phi-style names and falls back to identity.
// ------------------------------------------------------------
bool apply_final_layernorm_if_present(
    metal_llm_model* model,
    const std::vector<float>& h_in,
    std::vector<float>& h_out)
{
    // Try Phi-style "model.final_layernorm"
    const char* keys[][2] = {
        {"model.final_layernorm.weight", "model.final_layernorm.bias"},
        {"model.norm.weight",            "model.norm.bias"}
    };

    for (auto &pair : keys) {
        if (model->weights.find(pair[0]) != model->weights.end() &&
            model->weights.find(pair[1]) != model->weights.end())
        {
            if (!apply_layernorm_cpu(model,
                                     pair[0],
                                     pair[1],
                                     h_in,
                                     h_out))
            {
                std::fprintf(stderr,
                    "[metal_llm] ERROR: final layernorm %s/%s failed\n",
                    pair[0], pair[1]);
                return false;
            }
            return true;
        }
    }

    // No final norm → identity
    h_out = h_in;
    return true;
}

