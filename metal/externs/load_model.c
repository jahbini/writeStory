
// ---------------------------------------------------------------------------
//  LOAD MODEL (directory-style, HF local cache)
// ---------------------------------------------------------------------------
//
// Expects:
//   <directory>/config.json
//   <directory>/*.safetensors
//
// Uses MLX 0.30.0's load_safetensors, which returns:
//   std::pair<unordered_map<string, mx::array>, unordered_map<string, string>>
//
// Populates:
//   m->vocab_size / cfg.hidden_size / num_layers
//   m->weights[name] = mx::array
//   m->kv_cache_k / m->kv_cache_v
// ---------------------------------------------------------------------------


extern "C"
llm_status llm_load_model(metal_llm_model* m, const char* directory)
{
    if (!m || !directory) {
        log_err("llm_load_model", "null argument");
        return LLM_ERR_INVALID;
    }



// Load text
std::string cfg_path = std::string(directory) + "/config.json";
std::string cfg_txt  = read_file(cfg_path);

    // Load JSON into variant map
// j is your map<string, variant<...>>
//std::unordered_map<std::string, std::variant<int, int64_t, float, double, bool, std::string>> j;
auto    j = json_parse_flat_object(cfg_txt);

// Helper lambda
auto get = [&](const std::string& k, auto default_v) {
    using T = decltype(default_v);

    auto it = j.find(k);
    if (it == j.end())
        return default_v;

    // Safe variant access: no substitution failure
    if (auto p = std::get_if<T>(&it->second)) {
        return *p;
    }

    // Type mismatch: fall back to default
    return default_v;
};

// Required
cfg.vocab_size         = get("vocab_size", -1);
cfg.hidden_size        = get("hidden_size", -1);
cfg.num_layers         = get("num_hidden_layers", get("num_layers", -1));
cfg.num_heads          = get("num_attention_heads", get("num_heads", -1));
cfg.max_position_embeddings = get("max_position_embeddings",-1);

// Optional
int raw_kv_heads = get("num_key_value_heads", 0);
// If the config has null or 0 for num_key_value_heads (Phi-1.5 case),
// fall back to full multi-head attention.
if (raw_kv_heads <= 0) {
    cfg.num_kv_heads = cfg.num_heads;
} else {
    cfg.num_kv_heads = raw_kv_heads;
}

cfg.rope_theta         = get("rope_theta", 10000.0f);
cfg.partial_rotary_factor = get("partial_rotary_factor", 1.0f);
cfg.qk_layernorm       = get("qk_layernorm", false);

// If num_key_value_heads is null/0/missing in config.json,
// our tiny parser turns it into 0. Treat <= 0 as "use full heads".
if (cfg.num_kv_heads <= 0) {
    cfg.num_kv_heads = cfg.num_heads;
}

// Derived
cfg.head_dim   = cfg.hidden_size / cfg.num_heads;
cfg.rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor);

// Debug
fprintf(stderr,
    "[metal_llm] config: vocab=%d hidden=%d layers=%d heads=%d kv=%d head_dim=%d rotary_dim=%d\n",
    cfg.vocab_size,
    cfg.hidden_size,
    cfg.num_layers,
    cfg.num_heads,
    cfg.num_kv_heads,
    cfg.head_dim,
    cfg.rotary_dim);


    if (cfg.vocab_size <= 0 || cfg.hidden_size <= 0 || cfg.num_layers <= 0) {
        std::fprintf(stderr,
                     "[metal_llm] ERROR: invalid config values "
                     "(vocab=%d hidden=%d layers=%d)\n",
                     cfg.vocab_size, cfg.hidden_size, cfg.num_layers);
        return LLM_ERR_INVALID;
    }

    // 1) Find all .safetensors files in directory
    std::vector<std::string> safes;

    for (auto& ent : fs::directory_iterator(directory)) {
        if (!ent.is_regular_file()) continue;
        auto path = ent.path().string();

        const std::string ext = ".safetensors";
        if (path.size() >= ext.size() &&
            path.compare(path.size() - ext.size(), ext.size(), ext) == 0)
        {
            safes.push_back(path);
        }
    }

    if (safes.empty()) {
        std::fprintf(stderr,
                     "[metal_llm] ERROR: no .safetensors found in %s\n",
                     directory);
        return LLM_ERR_LOAD;
    }

    // Ensure deterministic order if multiple shards
    std::sort(safes.begin(), safes.end());

    std::printf("[metal_llm] Found %zu safetensors files\n",
                safes.size());

    // 2) Load each safetensor and merge into the weights map
    m->weights.clear();

    for (const auto& path : safes) {
        std::printf("[metal_llm] Loading %s\n", path.c_str());

        auto st_pair = mx::load_safetensors(path);
        auto& tensor_map = st_pair.first;

        for (auto& kv : tensor_map) {
            mx::eval(kv.second);  // ensure materialized
            m->weights.insert_or_assign(kv.first, kv.second);
        }
    }
    // Immediately upcast all weights to float32 for CPU math
    {
        for (auto &kv2 : m->weights) {
            mx::array &A = kv2.second;
            // Skip if already float32
            if (A.dtype() == mx::float32) continue;
            // Only upcast float16 → float32 (ignore int weights if any)
            if (A.dtype() == mx::float16) {
                kv2.second = mx::astype(A, mx::float32);
                mx::eval(kv2.second);
            }
        }
        std::printf("[metal_llm] Upcast all weights to float32\n");
    }
// TEMPORARY DEBUG: list all weights
fprintf(stderr, "=== MODEL WEIGHTS? ===\n");
//for (auto &kv : m->weights) { fprintf(stderr, "%s\n", kv.first.c_str()); }
//fprintf(stderr, "=== END WEIGHTS ===\n");

    std::printf("[metal_llm] Loaded %zu total weight tensors.\n",
                m->weights.size());

    // 3) Initialize KV cache on the default device (currently unused)
    const int kv_heads = cfg.num_kv_heads;
    const int max_pos  = cfg.max_position_embeddings;
    const int head_dim = cfg.head_dim;

    const size_t n = (size_t)kv_heads * (size_t)max_pos * (size_t)head_dim * (size_t)cfg.num_layers;
    m->kv_cache_k.assign(n, 0.0f);
    m->kv_cache_v.assign(n, 0.0f);
    m->kv_initialized = true;

    m->kv_initialized = false;

    model = m;
    std::printf("[metal_llm] KV cache allocated.\n");

    return LLM_OK;
}

