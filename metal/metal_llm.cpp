// metal_llm.cpp — minimal MLX/Metal-backed impl, deduped, with dtype string helper
// ------------------------------------------------------------------------------

#include "metal_llm.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <variant>

#include <mlx/mlx.h>   // full MLX core, including load_safetensors
namespace fs = std::filesystem;
namespace mx = mlx::core;
// ---------------------------------------------------------------------------
//  metal_llm_model constructor
// ---------------------------------------------------------------------------

typedef struct LLMConfig {
    int   vocab_size;            // e.g. 51200
    int   hidden_size;           // e.g. 2048
    int   num_layers;            // e.g. 24
    int   num_heads;             // e.g. 32
    int   num_kv_heads;          // e.g. 32 or null->heads

    int   max_position_embeddings;  // e.g. 2048 (from config.json)

    int   head_dim;              // derived: hidden_size / num_heads
    int   rotary_dim;            // derived: head_dim * partial_rotary_factor

    float rope_theta;            // e.g. 10000.0
    float partial_rotary_factor; // e.g. 0.5 for Phi-1.5

    bool  qk_layernorm;          // from config.json (usually false)
} LLMConfig;
LLMConfig cfg;
metal_llm_model::metal_llm_model()
    : lora_A(mx::zeros({0}, mx::float32)),
      lora_B(mx::zeros({0}, mx::float32)),
      kv_initialized(false),
      kv_cache_k(),
      kv_cache_v(),
      vocab_size(0),
      hidden_size(0),
      num_layers(0),
      device(mx::default_device())
{}
metal_llm_model* model;

// ------------------------------------------------------------
// QKV struct (CPU)
// ------------------------------------------------------------
struct QKV {
    std::vector<float> Q;
    std::vector<float> K;
    std::vector<float> V;
};

// Debug: dump Q/K/V only once, for the first token of the first forward step
static bool dumped_first_qkv = false;

static bool debugFlagEnabled(const char* name) {
    const char* val = ::getenv(name);

    if (!val) {
        // variable does NOT exist in this process
        // std::fprintf(stderr, "[env] %s = (null)\n", name);
        return false;
    }

    // Print the environment value EXACTLY as received:
    std::fprintf(stderr, "[env] %s = \"%s\"\n", name, val);

    // Empty string → false
    if (val[0] == '\0') return false;

    // If it's purely numeric, use numeric semantics: 0 → off, nonzero → on
    char* end = nullptr;
    long num = std::strtol(val, &end, 10);
    if (end != val && *end == '\0') {
        return (num != 0);
    }

    // Otherwise, treat some words as "true"
    if (strcasecmp(val, "true") == 0) return true;
    if (strcasecmp(val, "on")   == 0) return true;
    if (strcasecmp(val, "yes")  == 0) return true;
    if (strcasecmp(val, "y")    == 0) return true;
    if (strcasecmp(val, "t")    == 0) return true;

    // Explicit "1" handled by numeric path above; everything else → false
    return false;
}
// ---------------------------------------------------------------------------
//  LOGGING
// ---------------------------------------------------------------------------
static void dump_debug_vector(
    const std::string &path,
    const std::vector<float> &vec,
    const char *header = nullptr)
{
    std::ofstream ofs(path);
    if (!ofs) {
        std::fprintf(stderr,"[metal_llm][debug] Cannot open %s\n", path.c_str());
        return;
    }

    if (header) ofs << "--- " << header << "\n";

    ofs << std::setprecision(12);
    for (size_t i=0; i < vec.size(); ++i)
        ofs << i << ": " << vec[i] << "\n";
}
static void log_err(const char* where, const char* msg) {
    std::cerr << "[metal_llm][" << where << "] " << msg << std::endl;
}

static void log_warn(const char* where, const char* msg) {
    std::cerr << "[metal_llm][WARN][" << where << "] " << msg << std::endl;
}

// ---------------------------------------------------------------------------
//  INTERNAL HELPERS (anonymous namespace)
// ---------------------------------------------------------------------------

namespace {
#include "intern/json_parse.c"
#include "intern/shape.c"
#include "intern/softmax.c"
#include "intern/kqv_cpu.c"
#include "intern/layernorm_cpu.c"
#include "intern/single_attention.c"
#include "intern/mlp_cpu.c"
#include "intern/softmax_cpu.c"
#include "intern/embedding.c"
#include "intern/embedding_input_ln.c"
// #include "intern/embedding_only.c"
#include "intern/rope_cache.c"
#include "intern/rope.c"
#include "intern/run_block.c"
#include "intern/post_attention.c"
#include "intern/run_mlp.c"
#include "intern/final_ln.c"
#include "intern/lm_head.c"
} // anonymous namespace
#include "externs/load_model.c"
#include "externs/apply_lora.c"
#include "externs/reset_kv.c"
#include "externs/forward_step.c"
// ---------------------------------------------------------------------------
//  FREE MODEL
// ---------------------------------------------------------------------------

extern "C"
void llm_free_model(metal_llm_model* model)
{
    if (!model) return;

    delete model;

    std::cout << "[metal_llm] Model freed" << std::endl;
}

// ---------------------------------------------------------------------------
//  METADATA ACCESSORS
// ---------------------------------------------------------------------------

extern "C"
int32_t llm_get_vocab_size(metal_llm_model* model) {
    return model ? cfg.vocab_size : -1;
}

extern "C"
int32_t llm_get_hidden_size(metal_llm_model* model) {
    return model ? cfg.hidden_size : -1;
}

extern "C"
int32_t llm_get_num_layers(metal_llm_model* model) {
    return model ? cfg.num_layers : -1;
}


