// ------------------------------------------------------------
// Helper: embedding + input_layernorm for layer 0
// ------------------------------------------------------------
bool compute_embedding_and_input_ln(
    metal_llm_model* model,
    int32_t token_id,
    std::vector<float>& hvec_out,
    int& hidden_size_out)
{
    // --- DEBUG OVERRIDE: force a specific token for comparison ---
    if (const char* dbg_tok = std::getenv("MLX_DEBUG_FORCE_TOKEN")) {
        int forced = std::atoi(dbg_tok);
        if (forced >= 0) {
            std::fprintf(stderr,
                "[metal_llm][debug] Forcing token_id=%d instead of %d\n",
                forced, token_id);
            token_id = forced;
        }
    }
    
    auto it_emb = model->weights.find("model.embed_tokens.weight");
    if (it_emb == model->weights.end()) {
        std::fprintf(stderr,
            "[metal_llm] ERROR: model.embed_tokens.weight missing\n");
        return false;
    }

    mx::array& embed_mat = it_emb->second;
    if (embed_mat.ndim() != 2) {
        std::fprintf(stderr,                                  // ChitLin 000020
            "[metal_llm] ERROR: embed_tokens.weight ndim=%d (expected 2)\n",
            (int)embed_mat.ndim());
        return false;
    }

    int emb_vocab  = (int)embed_mat.shape(0);
    hidden_size_out = cfg.hidden_size;

    int row = token_id < 0 ? 0 :                              // ChitLin 000030
              token_id >= emb_vocab ? emb_vocab - 1 :
              token_id;

    // slice out [1, cfg.hidden_size]
    mx::array embed_vec = mx::slice(embed_mat,
                                    {row, 0},
                                    {row + 1, cfg.hidden_size});
    mx::eval(embed_vec);

    // --- DEBUG: dump raw embedding for this token (pre-LN) ---
    try {
        std::filesystem::create_directories("debug_cpp");
        mx::array embed_f32 = mx::astype(embed_vec, mx::float32);
        mx::eval(embed_f32);

        {
            std::ofstream f("debug_cpp/cpp_embed_last_token.log",
                            std::ios::trunc);
            if (f.is_open()) {
                const float* e = embed_f32.data<float>();
                for (int i = 0; i < cfg.hidden_size; ++i) {
                    // embed_vec has shape [1, cfg.hidden_size], so just dump row 0
                    f << i << ": " << std::setprecision(12)
                      << e[i] << "\n";
                }
            }
        }
    } catch (...) {
        // debug only; don't kill the run
        std::fprintf(stderr,
            "[metal_llm] WARNING: failed to write cpp_embed_last_token.log\n");
    }

    // ---- layernorm weights/bias ----
    auto it_lnw = model->weights.find("model.layers.0.input_layernorm.weight");
    auto it_lnb = model->weights.find("model.layers.0.input_layernorm.bias");

    if (it_lnw == model->weights.end() || it_lnb == model->weights.end()) {  // ChitLin 000040
        std::fprintf(stderr,
            "[metal_llm] ERROR: model.layers.0.input_layernorm.{weight,bias} missing\n");
        return false;
    }

    mx::array ln_w = it_lnw->second;
    mx::array ln_b = it_lnb->second;

    mx::array x0 = mx::fast::layer_norm(
        embed_vec,        // [1, hidden]
        ln_w,             // [hidden]
        ln_b,             // [hidden]
        1e-5f
    );
    mx::eval(x0);

    mx::array x0_f32 = mx::astype(x0, mx::float32);
    mx::eval(x0_f32);

    // --- DEBUG: dump LN output (post-LN) to mirror Python log ---
    try {
        std::filesystem::create_directories("debug_cpp");
        std::ofstream f("debug_cpp/cpp_layer0_input_ln_out.log",
                        std::ios::trunc);
        if (f.is_open()) {
            const float* x = x0_f32.data<float>();
            for (int i = 0; i < cfg.hidden_size; ++i) {
                f << i << ": " << std::setprecision(12)
                  << x[i] << "\n";
            }
        }
    } catch (...) {
        std::fprintf(stderr,
            "[metal_llm] WARNING: failed to write cpp_layer0_input_ln_out.log\n");
    }


    // Copy LN output to hvec_out as before
    hvec_out.resize(cfg.hidden_size);
    std::memcpy(hvec_out.data(),
                x0_f32.data<float>(),
                sizeof(float) * cfg.hidden_size);                  // ChitLin 000060

    return true;
}

