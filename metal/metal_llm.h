#ifndef METAL_LLM_H
#define METAL_LLM_H

// metal_llm.h — C-facing interface + model struct
// ------------------------------------------------

#include <cstdint>
#include <string>
#include <unordered_map>

#include <mlx/mlx.h>   // brings in mlx::core::array, Device, etc.

namespace mx = mlx::core;

// Status codes from your original version
typedef enum {
    LLM_OK          = 0,
    LLM_ERR_LOAD    = -1,
    LLM_ERR_LORA    = -2,
    LLM_ERR_STEP    = -3,
    LLM_ERR_ALLOC   = -4,
    LLM_ERR_INVALID = -5
} llm_status;

// ---------------------------------------------------------------------------
// INTERNAL MODEL STRUCTURE
// ---------------------------------------------------------------------------
// This must be visible here so metal_llm_node.cpp can access m->weights.
// It remains logically "opaque" to JS land.
// ---------------------------------------------------------------------------
struct metal_llm_model {
    // All model parameters from safetensors, keyed by name
    std::unordered_map<std::string, mx::array> weights;

    // LoRA storage placeholders
    mx::array lora_A;
    mx::array lora_B;

    // KV cache
    std::vector<float> kv_cache_k; // [kv_heads * max_pos * head_dim]
    std::vector<float> kv_cache_v; // same
    bool      kv_initialized;

    // Metadata
    int32_t   vocab_size;
    int32_t   hidden_size;
    int32_t   num_layers;

    // Device
    mx::Device device;

    metal_llm_model();
};

// ---------------------------------------------------------------------------
// C-LINKAGE API (called from the Node addon)
// ---------------------------------------------------------------------------

extern "C" {

llm_status llm_load_model(
    metal_llm_model* m,
    const char* directory_c
);

llm_status llm_apply_lora(
    metal_llm_model* model,
    const char* lora_path,
    float alpha,
    bool merge
);

llm_status llm_reset_kv_cache(
    metal_llm_model* model
);

llm_status llm_forward_step(
    metal_llm_model* model,
    int32_t token_id,
    int32_t pos,
    float* out_logits
);

void llm_free_model(
    metal_llm_model* model
);

int32_t llm_get_vocab_size(
    metal_llm_model* model
);

int32_t llm_get_hidden_size(
    metal_llm_model* model
);

int32_t llm_get_num_layers(
    metal_llm_model* model
);

} // extern "C"

#endif // METAL_LLM_H
