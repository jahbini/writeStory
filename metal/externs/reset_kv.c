// ---------------------------------------------------------------------------
//  RESET KV CACHE
// ---------------------------------------------------------------------------

extern "C"
llm_status llm_reset_kv_cache(metal_llm_model* m)
{
    if (!m) return LLM_ERR_INVALID;

    const int kv_heads = cfg.num_kv_heads;
    const int max_pos  = cfg.max_position_embeddings;
    const int head_dim = cfg.head_dim;
    const size_t n = (size_t)kv_heads * (size_t)max_pos * (size_t)head_dim * (size_t)cfg.num_layers;
    m->kv_cache_k.assign(n, 0.0f);
    m->kv_cache_v.assign(n, 0.0f);
    m->kv_initialized = true;

    std::cout << "[metal_llm] KV cache reset" << std::endl;
    return LLM_OK;
}

