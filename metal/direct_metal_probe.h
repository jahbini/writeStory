#pragma once

#include <cstdint>

struct GypsyDirectMetalProbeResult {
    bool ok = false;
    bool device_available = false;
    bool library_compiled = false;
    bool pipeline_created = false;
    bool command_completed = false;
    float checksum = 0.0f;
    float max_abs_diff = 0.0f;
    float q_checksum = 0.0f;
    float k_checksum = 0.0f;
    float v_checksum = 0.0f;
    float attention_checksum = 0.0f;
    std::uint32_t top_id = 0;
    float top_score = 0.0f;
    double elapsed_ms = 0.0;
    const char* error = nullptr;
};

struct GypsyDirectMetalLayerParams {
    const std::uint16_t* input_norm_weight = nullptr;
    const std::uint32_t* q_weight = nullptr;
    const std::uint16_t* q_scales = nullptr;
    const std::uint16_t* q_biases = nullptr;
    const std::uint32_t* k_weight = nullptr;
    const std::uint16_t* k_scales = nullptr;
    const std::uint16_t* k_biases = nullptr;
    const std::uint32_t* v_weight = nullptr;
    const std::uint16_t* v_scales = nullptr;
    const std::uint16_t* v_biases = nullptr;
    const std::uint16_t* q_norm_weight = nullptr;
    const std::uint16_t* k_norm_weight = nullptr;
    const std::uint32_t* o_weight = nullptr;
    const std::uint16_t* o_scales = nullptr;
    const std::uint16_t* o_biases = nullptr;
    const std::uint16_t* post_norm_weight = nullptr;
    const std::uint32_t* gate_weight = nullptr;
    const std::uint16_t* gate_scales = nullptr;
    const std::uint16_t* gate_biases = nullptr;
    const std::uint32_t* up_weight = nullptr;
    const std::uint16_t* up_scales = nullptr;
    const std::uint16_t* up_biases = nullptr;
    const std::uint32_t* down_weight = nullptr;
    const std::uint16_t* down_scales = nullptr;
    const std::uint16_t* down_biases = nullptr;
    std::uint32_t layer = 0;
    std::uint32_t seq_len = 1;
    std::uint32_t position = 0;
};

GypsyDirectMetalProbeResult GypsyRunDirectMetalAddProbe(std::uint32_t n);
GypsyDirectMetalProbeResult GypsyRunDirectMetalAttentionProbe(std::uint32_t seq_len);
GypsyDirectMetalProbeResult GypsyRunDirectMetalAttentionCompare(
    const float* q,
    const float* k,
    const float* v,
    const float* expected,
    std::uint32_t seq_len);
GypsyDirectMetalProbeResult GypsyRunDirectMetalAttentionToHost(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    std::uint32_t seq_len);
void GypsyResetDirectMetalAttentionKvCache();
GypsyDirectMetalProbeResult GypsyRunDirectMetalAttentionAppendToHost(
    std::uint32_t layer,
    const float* q,
    const float* current_k,
    const float* current_v,
    float* out,
    std::uint32_t seq_len);
GypsyDirectMetalProbeResult GypsyRunDirectMetalAttentionOProjectionAppendToHost(
    std::uint32_t layer,
    const float* q,
    const float* current_k,
    const float* current_v,
    const std::uint32_t* o_weight,
    const std::uint16_t* o_scales,
    const std::uint16_t* o_biases,
    float* out,
    std::uint32_t seq_len);
GypsyDirectMetalProbeResult GypsyRunDirectMetalAttentionOResidualAppendToHost(
    std::uint32_t layer,
    const float* q,
    const float* current_k,
    const float* current_v,
    const std::uint32_t* o_weight,
    const std::uint16_t* o_scales,
    const std::uint16_t* o_biases,
    const float* residual,
    float* out,
    std::uint32_t seq_len);
GypsyDirectMetalProbeResult GypsyRunDirectMetalQkvAttentionOAppendToHost(
    std::uint32_t layer,
    const float* input,
    const std::uint32_t* q_weight,
    const std::uint16_t* q_scales,
    const std::uint16_t* q_biases,
    const std::uint32_t* k_weight,
    const std::uint16_t* k_scales,
    const std::uint16_t* k_biases,
    const std::uint32_t* v_weight,
    const std::uint16_t* v_scales,
    const std::uint16_t* v_biases,
    const std::uint16_t* q_norm_weight,
    const std::uint16_t* k_norm_weight,
    const std::uint32_t* o_weight,
    const std::uint16_t* o_scales,
    const std::uint16_t* o_biases,
    float* out,
    std::uint32_t seq_len,
    std::uint32_t position);
GypsyDirectMetalProbeResult GypsyRunDirectMetalQuantizedProjectionCompare(
    const float* input,
    const std::uint32_t* weight,
    const std::uint16_t* scales,
    const std::uint16_t* biases,
    const float* expected,
    float* out,
    std::uint32_t rows,
    std::uint32_t packed_cols,
    std::uint32_t input_len);
GypsyDirectMetalProbeResult GypsyRunDirectMetalQuantizedProjectionTiledCompare(
    const float* input,
    const std::uint32_t* weight,
    const std::uint16_t* scales,
    const std::uint16_t* biases,
    const float* expected,
    float* out,
    std::uint32_t rows,
    std::uint32_t packed_cols,
    std::uint32_t input_len);
GypsyDirectMetalProbeResult GypsyRunDirectMetalQuantizedProjectionTiledToHost(
    const float* input,
    const std::uint32_t* weight,
    const std::uint16_t* scales,
    const std::uint16_t* biases,
    float* out,
    std::uint32_t rows,
    std::uint32_t packed_cols,
    std::uint32_t input_len);
GypsyDirectMetalProbeResult GypsyRunDirectMetalQuantizedProjectionTiledTop1(
    const float* input,
    const std::uint32_t* weight,
    const std::uint16_t* scales,
    const std::uint16_t* biases,
    std::uint32_t rows,
    std::uint32_t packed_cols,
    std::uint32_t input_len);
GypsyDirectMetalProbeResult GypsyRunDirectMetalRmsNormToHost(
    const float* input,
    const std::uint16_t* weight,
    float* out,
    std::uint32_t input_len);
GypsyDirectMetalProbeResult GypsyRunDirectMetalSiluMultiplyToHost(
    const float* gate,
    const float* up,
    float* out,
    std::uint32_t input_len);
GypsyDirectMetalProbeResult GypsyRunDirectMetalAddResidualToHost(
    const float* residual,
    const float* values,
    float* out,
    std::uint32_t input_len);
GypsyDirectMetalProbeResult GypsyRunDirectMetalMlpTailToHost(
    const float* residual,
    const float* o_projection,
    const std::uint16_t* post_norm_weight,
    const std::uint32_t* gate_weight,
    const std::uint16_t* gate_scales,
    const std::uint16_t* gate_biases,
    const std::uint32_t* up_weight,
    const std::uint16_t* up_scales,
    const std::uint16_t* up_biases,
    const std::uint32_t* down_weight,
    const std::uint16_t* down_scales,
    const std::uint16_t* down_biases,
    float* out);
GypsyDirectMetalProbeResult GypsyRunDirectMetalFullLayerToHost(
    std::uint32_t layer,
    const float* input,
    const std::uint16_t* input_norm_weight,
    const std::uint32_t* q_weight,
    const std::uint16_t* q_scales,
    const std::uint16_t* q_biases,
    const std::uint32_t* k_weight,
    const std::uint16_t* k_scales,
    const std::uint16_t* k_biases,
    const std::uint32_t* v_weight,
    const std::uint16_t* v_scales,
    const std::uint16_t* v_biases,
    const std::uint16_t* q_norm_weight,
    const std::uint16_t* k_norm_weight,
    const std::uint32_t* o_weight,
    const std::uint16_t* o_scales,
    const std::uint16_t* o_biases,
    const std::uint16_t* post_norm_weight,
    const std::uint32_t* gate_weight,
    const std::uint16_t* gate_scales,
    const std::uint16_t* gate_biases,
    const std::uint32_t* up_weight,
    const std::uint16_t* up_scales,
    const std::uint16_t* up_biases,
    const std::uint32_t* down_weight,
    const std::uint16_t* down_scales,
    const std::uint16_t* down_biases,
    float* out,
    std::uint32_t seq_len,
    std::uint32_t position);
GypsyDirectMetalProbeResult GypsyRunDirectMetalFullLayerResident(
    std::uint32_t layer,
    const float* input,
    const std::uint16_t* input_norm_weight,
    const std::uint32_t* q_weight,
    const std::uint16_t* q_scales,
    const std::uint16_t* q_biases,
    const std::uint32_t* k_weight,
    const std::uint16_t* k_scales,
    const std::uint16_t* k_biases,
    const std::uint32_t* v_weight,
    const std::uint16_t* v_scales,
    const std::uint16_t* v_biases,
    const std::uint16_t* q_norm_weight,
    const std::uint16_t* k_norm_weight,
    const std::uint32_t* o_weight,
    const std::uint16_t* o_scales,
    const std::uint16_t* o_biases,
    const std::uint16_t* post_norm_weight,
    const std::uint32_t* gate_weight,
    const std::uint16_t* gate_scales,
    const std::uint16_t* gate_biases,
    const std::uint32_t* up_weight,
    const std::uint16_t* up_scales,
    const std::uint16_t* up_biases,
    const std::uint32_t* down_weight,
    const std::uint16_t* down_scales,
    const std::uint16_t* down_biases,
    float* out,
    std::uint32_t seq_len,
    std::uint32_t position,
    std::uint32_t load_input,
    std::uint32_t read_output);
GypsyDirectMetalProbeResult GypsyRunDirectMetalTerminalStackResident(
    const GypsyDirectMetalLayerParams* layers,
    std::uint32_t layer_count,
    const float* input,
    float* out);
