// ---------------------------------------------------------------------------
//  FORWARD STEP (Option A: full layer-0 block)
// ---------------------------------------------------------------------------
//
// Pipeline:
//
//   1) embedding lookup
//   2) layer-0 input_layernorm (MLX)
//   3) QKV projection (CPU)
//   4) single-token self-attention (V pass-through)
//   5) residual: h_after_attn = h0 + attn_out
//   6) post-attention LN (CPU helper)
//   7) MLP (CPU)
//   8) residual: h_final = h_after_attn + mlp_out
//   9) lm_head matmul (CPU)
// ---------------------------------------------------------------------------

extern "C"
llm_status llm_forward_step(
    metal_llm_model* model,
    int32_t token_id,
    int32_t pos,
    float* out_logits)
{
    if (!model || !out_logits) return LLM_ERR_INVALID;
    if (cfg.vocab_size <= 0) return LLM_ERR_INVALID;       

    const int32_t vocab = cfg.vocab_size;
    for (int32_t i = 0; i < vocab; ++i) {
        out_logits[i] = 0.0f;
    }

    // 1. Embedding only (no LN)
    std::vector<float> hvec;
    int hidden_size = 0;

    std::fprintf(stderr, "[compute_embeddingis], pos= %d, token_id= %d\n",pos,token_id);
    if (!compute_embedding_only(model, token_id, hvec, hidden_size))
        return LLM_ERR_INVALID;                               

    // 2. RoPE cache (per position)
    std::vector<float> cos_cache;
    std::vector<float> sin_cache;

    std::fprintf(stderr, "[buildrope_cache], pos= %d, token_id= %d\n",pos,token_id);
    build_rope_cache_mlx(cfg.rotary_dim, pos, cos_cache, sin_cache); 
								      

    // 3. Transformer layers
    int dbgLayer = -1;
    if (debugFlagEnabled("MLX_DEBUG_LAYER")) {
        const char* s = std::getenv("MLX_DEBUG_LAYER");
        if (s) dbgLayer = std::atoi(s);
        std::fprintf(stderr, "[layer_debug] Isolating layer %d\n", dbgLayer);
    }

    for (int layer = 0; layer < cfg.num_layers; ++layer) { 
        if (dbgLayer >= 0 && layer != dbgLayer) { continue; }

        // 3.1 per-layer input_layernorm: h_ln = LN_L(hvec)
        std::string lnW_key = layer_key(layer, "input_layernorm.weight");
        std::string lnB_key = layer_key(layer, "input_layernorm.bias");

        std::vector<float> h_ln;                              
        if (!apply_layernorm_cpu(model, lnW_key.c_str(), lnB_key.c_str(), hvec, h_ln))
        {
            std::fprintf(stderr,
                "[metal_llm] ERROR: input_layernorm failed at layer %d\n",
                layer);
            return LLM_ERR_INVALID;
        }

        // 3.2 attention block: uses hvec (raw) + h_ln (normalized)
        std::vector<float> h_after_attn;
        if (!run_attention_block(model, layer, hvec, h_ln, cos_cache, sin_cache, h_after_attn,pos)){ return LLM_ERR_INVALID; }

        // 3.3 optional post-attention LN
        std::vector<float> h_norm;
        if (!run_post_attention_layernorm(model, layer, h_after_attn, h_norm)) { return LLM_ERR_INVALID;                           }
        // 3.4 MLP + residual
        std::vector<float> h_next;
        if (!run_mlp_with_residual(model, layer, h_after_attn, h_norm, h_next)) { return LLM_ERR_INVALID; }

        hvec.swap(h_next);

        if (dbgLayer == layer) { std::fprintf(stderr, "[layer_debug] Completed isolated layer %d\n", layer); break; }
    }


    // 4. Optional final layernorm before head
    std::vector<float> h_final;
    if (!apply_final_layernorm_if_present(model, hvec, h_final))
        return LLM_ERR_INVALID;                               

    // 5. Final lm_head
    if (!apply_lm_head(model, h_final, out_logits))
        return LLM_ERR_INVALID;                               

    // 6. Optional logits dump for debugging
    if (debugFlagEnabled("MLX_DUMP_LOGITS") || pos == 4) {
        std::filesystem::create_directories("debug_cpp");
        FILE* f = std::fopen("debug_cpp/cpp_logits_last_token.log", "w");
        if (f) {
            for (int i = 0; i < cfg.vocab_size; ++i) {
                std::fprintf(f, "%d: %.12f\n", i, out_logits[i]);
            }
            std::fclose(f);
            std::fprintf(stderr,
                "[metal_llm][debug] wrote cpp_logits_last_token.log\n");
        }
    }

    return LLM_OK;
}

