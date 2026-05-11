// ------------------------------------------------------------
// Helper: embedding only (no layernorm), returns float32
// ------------------------------------------------------------
bool compute_embedding_only(
    metal_llm_model* model,
    int32_t token_id,
    std::vector<float>& hvec_out,
    int& hidden_size_out)
{
    auto it_emb = model->weights.find("model.embed_tokens.weight");
    if (it_emb == model->weights.end()) {
        std::fprintf(stderr,
            "[metal_llm] ERROR: model.embed_tokens.weight missing\n");
        return false;
    }

    mx::array& embed_mat = it_emb->second;
    if (embed_mat.ndim() != 2) {
        std::fprintf(stderr,
            "[metal_llm] ERROR: embed_tokens.weight ndim=%d (expected 2)\n",
            (int)embed_mat.ndim());
        return false;
    }

    int emb_vocab  = (int)embed_mat.shape(0);
    hidden_size_out = cfg.hidden_size;

    int row = (token_id < 0) ? 0 :
              (token_id >= emb_vocab ? emb_vocab - 1 : token_id);

    mx::array embed_vec = mx::slice(embed_mat,
                                    {row, 0},
                                    {row + 1, cfg.hidden_size});
    // upcast to float32 so the CPU side is always float32
    mx::array embed_f32 = mx::astype(embed_vec, mx::float32);
    mx::eval(embed_f32);

    hvec_out.resize(cfg.hidden_size);
    std::memcpy(hvec_out.data(),
                embed_f32.data<float>(),
                sizeof(float) * cfg.hidden_size);

    // optional: keep your debug dump if you like
    return true;
}

