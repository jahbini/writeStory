#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "direct_metal_probe.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

const char* PersistError(const std::string& message) {
    static thread_local std::string storage;
    storage = message;
    return storage.c_str();
}

std::string NSErrorMessage(NSError* error) {
    if (error == nil) return "";
    NSString* description = [error localizedDescription];
    if (description == nil) return "unknown NSError";
    return std::string([description UTF8String]);
}

NSString* AttentionKernelSource() {
    return @R"METAL(
        #include <metal_stdlib>
        using namespace metal;

        #define Q_HEADS 32
        #define KV_HEADS 8
        #define Q_PER_KV 4
        #define HEAD_DIM 128
        #define MAX_SEQ 256
        #define SCALE 0.08838834764831845f

        kernel void gypsy_attention_probe(
            const device float* q [[buffer(0)]],
            const device float* k [[buffer(1)]],
            const device float* v [[buffer(2)]],
            device float* out [[buffer(3)]],
            constant uint& seq_len [[buffer(4)]],
            uint h [[thread_position_in_grid]]) {
            if (h >= Q_HEADS) return;
            const uint n = min(seq_len, (uint)MAX_SEQ);
            const uint kvh = h / Q_PER_KV;
            float scores[MAX_SEQ];
            float max_score = -3.4028234663852886e38f;
            for (uint p = 0; p < n; ++p) {
                float score = 0.0f;
                for (uint d = 0; d < HEAD_DIM; ++d) {
                    score += q[h * HEAD_DIM + d] * k[(p * KV_HEADS + kvh) * HEAD_DIM + d];
                }
                score *= SCALE;
                scores[p] = score;
                max_score = metal::max(max_score, score);
            }
            float denom = 0.0f;
            for (uint p = 0; p < n; ++p) {
                scores[p] = metal::exp(scores[p] - max_score);
                denom += scores[p];
            }
            denom = metal::max(denom, 1.0e-20f);
            for (uint d = 0; d < HEAD_DIM; ++d) {
                float acc = 0.0f;
                for (uint p = 0; p < n; ++p) {
                    const float prob = scores[p] / denom;
                    acc += prob * v[(p * KV_HEADS + kvh) * HEAD_DIM + d];
                }
                out[h * HEAD_DIM + d] = acc;
            }
        }

        static inline float gypsy_bf16_to_float(uint16_t x) {
            uint bits = ((uint)x) << 16;
            return as_type<float>(bits);
        }

        kernel void gypsy_o_projection_from_attention(
            const device float* input [[buffer(0)]],
            const device uint* weight [[buffer(1)]],
            const device ushort* scales [[buffer(2)]],
            const device ushort* biases [[buffer(3)]],
            device float* out [[buffer(4)]],
            uint row [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]],
            uint threads [[threads_per_threadgroup]]) {
            if (row >= 2560) return;
            constexpr uint group_size = 64;
            constexpr uint values_per_word = 8;
            constexpr uint words_per_group = group_size / values_per_word;
            constexpr uint groups = 4096 / group_size;
            constexpr uint packed_cols = 4096 / values_per_word;
            const uint total_words = groups * words_per_group;
            threadgroup float partials[256];
            float acc = 0.0f;
            for (uint word_index = tid; word_index < total_words; word_index += threads) {
                const uint g = word_index / words_per_group;
                const uint w = word_index - (g * words_per_group);
                const float scale = gypsy_bf16_to_float(scales[row * groups + g]);
                const float bias = gypsy_bf16_to_float(biases[row * groups + g]);
                const uint packed = weight[row * packed_cols + word_index];
                for (uint n = 0; n < values_per_word; ++n) {
                    const uint qv = (packed >> (n * 4)) & 0xFu;
                    const uint input_index = g * group_size + w * values_per_word + n;
                    acc += input[input_index] * (((float)qv * scale) + bias);
                }
            }
            partials[tid] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    partials[tid] += partials[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                out[row] = partials[0];
            }
        }

        kernel void gypsy_qkv_projection_2560_int4(
            const device float* input [[buffer(0)]],
            const device uint* weight [[buffer(1)]],
            const device ushort* scales [[buffer(2)]],
            const device ushort* biases [[buffer(3)]],
            device float* out [[buffer(4)]],
            constant uint& rows [[buffer(5)]],
            uint row [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]],
            uint threads [[threads_per_threadgroup]]) {
            if (row >= rows) return;
            constexpr uint group_size = 64;
            constexpr uint values_per_word = 8;
            constexpr uint words_per_group = group_size / values_per_word;
            constexpr uint groups = 2560 / group_size;
            constexpr uint packed_cols = 2560 / values_per_word;
            const uint total_words = groups * words_per_group;
            threadgroup float partials[256];
            float acc = 0.0f;
            for (uint word_index = tid; word_index < total_words; word_index += threads) {
                const uint g = word_index / words_per_group;
                const uint w = word_index - (g * words_per_group);
                const float scale = gypsy_bf16_to_float(scales[row * groups + g]);
                const float bias = gypsy_bf16_to_float(biases[row * groups + g]);
                const uint packed = weight[row * packed_cols + word_index];
                for (uint n = 0; n < values_per_word; ++n) {
                    const uint qv = (packed >> (n * 4)) & 0xFu;
                    const uint input_index = g * group_size + w * values_per_word + n;
                    acc += input[input_index] * (((float)qv * scale) + bias);
                }
            }
            partials[tid] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    partials[tid] += partials[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                out[row] = partials[0];
            }
        }

        kernel void gypsy_qk_norm_rope_qwen(
            const device float* q_raw [[buffer(0)]],
            const device float* k_raw [[buffer(1)]],
            const device ushort* q_norm_weight [[buffer(2)]],
            const device ushort* k_norm_weight [[buffer(3)]],
            device float* q_rope [[buffer(4)]],
            device float* k_rope [[buffer(5)]],
            constant uint& position [[buffer(6)]],
            uint h [[thread_position_in_grid]]) {
            if (h >= Q_HEADS) return;
            constexpr float eps = 1.0e-6f;
            constexpr float theta = 5000000.0f;
            float q_ss = 0.0f;
            for (uint d = 0; d < HEAD_DIM; ++d) {
                const float x = q_raw[h * HEAD_DIM + d];
                q_ss += x * x;
            }
            const float q_r = metal::rsqrt(q_ss / float(HEAD_DIM) + eps);
            for (uint i = 0; i < HEAD_DIM / 2; ++i) {
                const float freq = metal::pow(theta, -float(i * 2) / float(HEAD_DIM));
                const float angle = float(position) * freq;
                const float c = metal::cos(angle);
                const float s = metal::sin(angle);
                const uint a = h * HEAD_DIM + i;
                const uint b = h * HEAD_DIM + i + HEAD_DIM / 2;
                const float x1 = q_raw[a] * q_r * gypsy_bf16_to_float(q_norm_weight[i]);
                const float x2 = q_raw[b] * q_r * gypsy_bf16_to_float(q_norm_weight[i + HEAD_DIM / 2]);
                q_rope[a] = x1 * c - x2 * s;
                q_rope[b] = x2 * c + x1 * s;
            }
            if (h < KV_HEADS) {
                float k_ss = 0.0f;
                for (uint d = 0; d < HEAD_DIM; ++d) {
                    const float x = k_raw[h * HEAD_DIM + d];
                    k_ss += x * x;
                }
                const float k_r = metal::rsqrt(k_ss / float(HEAD_DIM) + eps);
                for (uint i = 0; i < HEAD_DIM / 2; ++i) {
                    const float freq = metal::pow(theta, -float(i * 2) / float(HEAD_DIM));
                    const float angle = float(position) * freq;
                    const float c = metal::cos(angle);
                    const float s = metal::sin(angle);
                    const uint a = h * HEAD_DIM + i;
                    const uint b = h * HEAD_DIM + i + HEAD_DIM / 2;
                    const float x1 = k_raw[a] * k_r * gypsy_bf16_to_float(k_norm_weight[i]);
                    const float x2 = k_raw[b] * k_r * gypsy_bf16_to_float(k_norm_weight[i + HEAD_DIM / 2]);
                    k_rope[a] = x1 * c - x2 * s;
                    k_rope[b] = x2 * c + x1 * s;
                }
            }
        }

        kernel void gypsy_add_residual_2560(
            const device float* residual [[buffer(0)]],
            device float* values [[buffer(1)]],
            uint i [[thread_position_in_grid]]) {
            if (i >= 2560) return;
            values[i] += residual[i];
        }

        kernel void gypsy_rmsnorm_2560(
            const device float* input [[buffer(0)]],
            const device ushort* weight [[buffer(1)]],
            device float* out [[buffer(2)]],
            uint tid [[thread_index_in_threadgroup]]) {
            constexpr uint hidden = 2560;
            constexpr uint threads = 256;
            constexpr float eps = 1.0e-6f;
            threadgroup float partials[threads];
            float sum = 0.0f;
            for (uint i = tid; i < hidden; i += threads) {
                const float x = input[i];
                sum += x * x;
            }
            partials[tid] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    partials[tid] += partials[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const float inv_rms = metal::rsqrt(partials[0] / float(hidden) + eps);
            for (uint i = tid; i < hidden; i += threads) {
                out[i] = input[i] * inv_rms * gypsy_bf16_to_float(weight[i]);
            }
        }

        kernel void gypsy_silu_multiply(
            const device float* gate [[buffer(0)]],
            const device float* up [[buffer(1)]],
            device float* out [[buffer(2)]],
            constant uint& n [[buffer(3)]],
            uint i [[thread_position_in_grid]]) {
            if (i >= n) return;
            const float g = gate[i];
            out[i] = (g / (1.0f + metal::exp(-g))) * up[i];
        }

        kernel void gypsy_down_projection_9728_int4(
            const device float* input [[buffer(0)]],
            const device uint* weight [[buffer(1)]],
            const device ushort* scales [[buffer(2)]],
            const device ushort* biases [[buffer(3)]],
            device float* out [[buffer(4)]],
            uint row [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]],
            uint threads [[threads_per_threadgroup]]) {
            if (row >= 2560) return;
            constexpr uint group_size = 64;
            constexpr uint values_per_word = 8;
            constexpr uint words_per_group = group_size / values_per_word;
            constexpr uint groups = 9728 / group_size;
            constexpr uint packed_cols = 9728 / values_per_word;
            constexpr uint total_words = groups * words_per_group;
            threadgroup float partials[256];
            float acc = 0.0f;
            for (uint word_index = tid; word_index < total_words; word_index += threads) {
                const uint g = word_index / words_per_group;
                const uint w = word_index - (g * words_per_group);
                const float scale = gypsy_bf16_to_float(scales[row * groups + g]);
                const float bias = gypsy_bf16_to_float(biases[row * groups + g]);
                const uint packed = weight[row * packed_cols + word_index];
                for (uint n = 0; n < values_per_word; ++n) {
                    const uint q = (packed >> (n * 4)) & 0xFu;
                    const uint input_index = g * group_size + w * values_per_word + n;
                    acc += input[input_index] * (((float)q * scale) + bias);
                }
            }
            partials[tid] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    partials[tid] += partials[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                out[row] = partials[0];
            }
        }
    )METAL";
}

NSString* QuantizedProjectionKernelSource() {
    return @R"METAL(
        #include <metal_stdlib>
        using namespace metal;

        static inline float gypsy_bf16_to_float(uint16_t x) {
            uint bits = ((uint)x) << 16;
            return as_type<float>(bits);
        }

        kernel void gypsy_quantized_projection_probe(
            const device float* input [[buffer(0)]],
            const device uint* weight [[buffer(1)]],
            const device ushort* scales [[buffer(2)]],
            const device ushort* biases [[buffer(3)]],
            device float* out [[buffer(4)]],
            constant uint& rows [[buffer(5)]],
            constant uint& packed_cols [[buffer(6)]],
            constant uint& input_len [[buffer(7)]],
            uint row [[thread_position_in_grid]]) {
            if (row >= rows) return;
            constexpr uint group_size = 64;
            constexpr uint values_per_word = 8;
            constexpr uint words_per_group = group_size / values_per_word;
            const uint groups = input_len / group_size;
            float acc = 0.0f;
            for (uint g = 0; g < groups; ++g) {
                const float scale = gypsy_bf16_to_float(scales[row * groups + g]);
                const float bias = gypsy_bf16_to_float(biases[row * groups + g]);
                for (uint w = 0; w < words_per_group; ++w) {
                    const uint packed = weight[row * packed_cols + g * words_per_group + w];
                    for (uint n = 0; n < values_per_word; ++n) {
                        const uint q = (packed >> (n * 4)) & 0xFu;
                        const uint input_index = g * group_size + w * values_per_word + n;
                        acc += input[input_index] * (((float)q * scale) + bias);
                    }
                }
            }
            out[row] = acc;
        }
    )METAL";
}

NSString* QuantizedProjectionTiledKernelSource(std::uint32_t rows, std::uint32_t packed_cols, std::uint32_t input_len) {
    std::string source = R"METAL(
        #include <metal_stdlib>
        using namespace metal;

        #define ROWS __ROWS__
        #define PACKED_COLS __PACKED_COLS__
        #define INPUT_LEN __INPUT_LEN__

        static inline float gypsy_bf16_to_float(uint16_t x) {
            uint bits = ((uint)x) << 16;
            return as_type<float>(bits);
        }

        kernel void gypsy_quantized_projection_tiled_probe(
            const device float* input [[buffer(0)]],
            const device uint* weight [[buffer(1)]],
            const device ushort* scales [[buffer(2)]],
            const device ushort* biases [[buffer(3)]],
            device float* out [[buffer(4)]],
            uint row [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]],
            uint threads [[threads_per_threadgroup]]) {
            if (row >= ROWS) return;
            constexpr uint group_size = 64;
            constexpr uint values_per_word = 8;
            constexpr uint words_per_group = group_size / values_per_word;
            constexpr uint groups = INPUT_LEN / group_size;
            const uint total_words = groups * words_per_group;
            threadgroup float partials[256];
            float acc = 0.0f;
            for (uint word_index = tid; word_index < total_words; word_index += threads) {
                const uint g = word_index / words_per_group;
                const uint w = word_index - (g * words_per_group);
                const float scale = gypsy_bf16_to_float(scales[row * groups + g]);
                const float bias = gypsy_bf16_to_float(biases[row * groups + g]);
                const uint packed = weight[row * PACKED_COLS + word_index];
                for (uint n = 0; n < values_per_word; ++n) {
                    const uint q = (packed >> (n * 4)) & 0xFu;
                    const uint input_index = g * group_size + w * values_per_word + n;
                    acc += input[input_index] * (((float)q * scale) + bias);
                }
            }
            partials[tid] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    partials[tid] += partials[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                out[row] = partials[0];
            }
        }

        kernel void gypsy_top1_reduce_probe(
            const device float* logits [[buffer(0)]],
            device float* partial_scores [[buffer(1)]],
            device uint* partial_ids [[buffer(2)]],
            uint group [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]]) {
            constexpr uint threads = 256;
            threadgroup float scores[threads];
            threadgroup uint ids[threads];
            const uint idx = group * threads + tid;
            if (idx < ROWS) {
                scores[tid] = logits[idx];
                ids[tid] = idx;
            } else {
                scores[tid] = -3.4028234663852886e38f;
                ids[tid] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
                if (tid < stride && scores[tid + stride] > scores[tid]) {
                    scores[tid] = scores[tid + stride];
                    ids[tid] = ids[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                partial_scores[group] = scores[0];
                partial_ids[group] = ids[0];
            }
        }

        kernel void gypsy_top1_serial_probe(
            const device float* logits [[buffer(0)]],
            device float* top_score [[buffer(1)]],
            device uint* top_id [[buffer(2)]],
            uint tid [[thread_position_in_grid]]) {
            if (tid != 0) return;
            float best = logits[0];
            uint best_id = 0;
            for (uint i = 1; i < ROWS; ++i) {
                const float value = logits[i];
                if (value > best) {
                    best = value;
                    best_id = i;
                }
            }
            top_score[0] = best;
            top_id[0] = best_id;
        }
    )METAL";
    auto replace_all = [](std::string& s, const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    replace_all(source, "__ROWS__", std::to_string(rows));
    replace_all(source, "__PACKED_COLS__", std::to_string(packed_cols));
    replace_all(source, "__INPUT_LEN__", std::to_string(input_len));
    return [NSString stringWithUTF8String:source.c_str()];
}

struct DirectMetalAttentionContext {
    struct LayerKvBuffers {
        id<MTLBuffer> k_buffer = nil;
        id<MTLBuffer> v_buffer = nil;
        id<MTLBuffer> o_weight_buffer = nil;
        id<MTLBuffer> o_scales_buffer = nil;
        id<MTLBuffer> o_biases_buffer = nil;
        id<MTLBuffer> q_weight_buffer = nil;
        id<MTLBuffer> q_scales_buffer = nil;
        id<MTLBuffer> q_biases_buffer = nil;
        id<MTLBuffer> k_weight_buffer = nil;
        id<MTLBuffer> k_scales_buffer = nil;
        id<MTLBuffer> k_biases_buffer = nil;
        id<MTLBuffer> v_weight_buffer = nil;
        id<MTLBuffer> v_scales_buffer = nil;
        id<MTLBuffer> v_biases_buffer = nil;
        id<MTLBuffer> input_norm_weight_buffer = nil;
        id<MTLBuffer> q_norm_weight_buffer = nil;
        id<MTLBuffer> k_norm_weight_buffer = nil;
        id<MTLBuffer> post_norm_weight_buffer = nil;
        id<MTLBuffer> gate_weight_buffer = nil;
        id<MTLBuffer> gate_scales_buffer = nil;
        id<MTLBuffer> gate_biases_buffer = nil;
        id<MTLBuffer> up_weight_buffer = nil;
        id<MTLBuffer> up_scales_buffer = nil;
        id<MTLBuffer> up_biases_buffer = nil;
        id<MTLBuffer> down_weight_buffer = nil;
        id<MTLBuffer> down_scales_buffer = nil;
        id<MTLBuffer> down_biases_buffer = nil;
        size_t capacity_bytes = 0;
        size_t o_weight_capacity_bytes = 0;
        size_t o_scale_capacity_bytes = 0;
        size_t q_weight_capacity_bytes = 0;
        size_t q_scale_capacity_bytes = 0;
        size_t kv_weight_capacity_bytes = 0;
        size_t kv_scale_capacity_bytes = 0;
        size_t input_norm_weight_capacity_bytes = 0;
        size_t head_norm_weight_capacity_bytes = 0;
        size_t post_norm_weight_capacity_bytes = 0;
        size_t gate_weight_capacity_bytes = 0;
        size_t gate_scale_capacity_bytes = 0;
        size_t up_weight_capacity_bytes = 0;
        size_t up_scale_capacity_bytes = 0;
        size_t down_weight_capacity_bytes = 0;
        size_t down_scale_capacity_bytes = 0;
        const void* o_weight_source = nullptr;
        const void* o_scales_source = nullptr;
        const void* o_biases_source = nullptr;
        const void* q_weight_source = nullptr;
        const void* q_scales_source = nullptr;
        const void* q_biases_source = nullptr;
        const void* k_weight_source = nullptr;
        const void* k_scales_source = nullptr;
        const void* k_biases_source = nullptr;
        const void* v_weight_source = nullptr;
        const void* v_scales_source = nullptr;
        const void* v_biases_source = nullptr;
        const void* input_norm_weight_source = nullptr;
        const void* q_norm_weight_source = nullptr;
        const void* k_norm_weight_source = nullptr;
        const void* post_norm_weight_source = nullptr;
        const void* gate_weight_source = nullptr;
        const void* gate_scales_source = nullptr;
        const void* gate_biases_source = nullptr;
        const void* up_weight_source = nullptr;
        const void* up_scales_source = nullptr;
        const void* up_biases_source = nullptr;
        const void* down_weight_source = nullptr;
        const void* down_scales_source = nullptr;
        const void* down_biases_source = nullptr;
    };
    id<MTLDevice> device = nil;
    id<MTLLibrary> library = nil;
    id<MTLFunction> function = nil;
    id<MTLFunction> o_projection_function = nil;
    id<MTLFunction> qkv_projection_function = nil;
    id<MTLFunction> qk_norm_rope_function = nil;
    id<MTLFunction> residual_add_function = nil;
    id<MTLFunction> rmsnorm_function = nil;
    id<MTLFunction> silu_multiply_function = nil;
    id<MTLFunction> down_projection_function = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLComputePipelineState> o_projection_pipeline = nil;
    id<MTLComputePipelineState> qkv_projection_pipeline = nil;
    id<MTLComputePipelineState> qk_norm_rope_pipeline = nil;
    id<MTLComputePipelineState> residual_add_pipeline = nil;
    id<MTLComputePipelineState> rmsnorm_pipeline = nil;
    id<MTLComputePipelineState> silu_multiply_pipeline = nil;
    id<MTLComputePipelineState> down_projection_pipeline = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLBuffer> q_buffer = nil;
    id<MTLBuffer> k_buffer = nil;
    id<MTLBuffer> v_buffer = nil;
    id<MTLBuffer> out_buffer = nil;
    id<MTLBuffer> o_out_buffer = nil;
    id<MTLBuffer> residual_buffer = nil;
    id<MTLBuffer> rmsnorm_weight_buffer = nil;
    id<MTLBuffer> post_norm_weight_buffer = nil;
    id<MTLBuffer> rmsnorm_out_buffer = nil;
    id<MTLBuffer> hidden_input_buffer = nil;
    id<MTLBuffer> hidden_next_buffer = nil;
    id<MTLBuffer> q_raw_buffer = nil;
    id<MTLBuffer> k_raw_buffer = nil;
    id<MTLBuffer> v_raw_buffer = nil;
    id<MTLBuffer> q_rope_buffer = nil;
    id<MTLBuffer> k_rope_buffer = nil;
    id<MTLBuffer> q_norm_weight_buffer = nil;
    id<MTLBuffer> k_norm_weight_buffer = nil;
    id<MTLBuffer> rows_buffer = nil;
    id<MTLBuffer> position_buffer = nil;
    id<MTLBuffer> o_weight_buffer = nil;
    id<MTLBuffer> o_scales_buffer = nil;
    id<MTLBuffer> o_biases_buffer = nil;
    id<MTLBuffer> n_buffer = nil;
    id<MTLBuffer> gate_buffer = nil;
    id<MTLBuffer> up_buffer = nil;
    id<MTLBuffer> activation_out_buffer = nil;
    id<MTLBuffer> gate_weight_buffer = nil;
    id<MTLBuffer> gate_scales_buffer = nil;
    id<MTLBuffer> gate_biases_buffer = nil;
    id<MTLBuffer> up_weight_buffer = nil;
    id<MTLBuffer> up_scales_buffer = nil;
    id<MTLBuffer> up_biases_buffer = nil;
    id<MTLBuffer> down_weight_buffer = nil;
    id<MTLBuffer> down_scales_buffer = nil;
    id<MTLBuffer> down_biases_buffer = nil;
    std::array<LayerKvBuffers, 36> layer_kv;
    size_t q_capacity_bytes = 0;
    size_t kv_capacity_bytes = 0;
    size_t out_capacity_bytes = 0;
    size_t o_out_capacity_bytes = 0;
    size_t residual_capacity_bytes = 0;
    size_t rmsnorm_weight_capacity_bytes = 0;
    size_t post_norm_weight_capacity_bytes = 0;
    size_t rmsnorm_out_capacity_bytes = 0;
    size_t hidden_input_capacity_bytes = 0;
    size_t hidden_next_capacity_bytes = 0;
    size_t q_raw_capacity_bytes = 0;
    size_t k_raw_capacity_bytes = 0;
    size_t v_raw_capacity_bytes = 0;
    size_t q_rope_capacity_bytes = 0;
    size_t k_rope_capacity_bytes = 0;
    size_t norm_weight_capacity_bytes = 0;
    size_t o_weight_capacity_bytes = 0;
    size_t o_scale_capacity_bytes = 0;
    size_t gate_capacity_bytes = 0;
    size_t up_capacity_bytes = 0;
    size_t activation_out_capacity_bytes = 0;
    size_t gate_weight_capacity_bytes = 0;
    size_t gate_scale_capacity_bytes = 0;
    size_t up_weight_capacity_bytes = 0;
    size_t up_scale_capacity_bytes = 0;
    size_t down_weight_capacity_bytes = 0;
    size_t down_scale_capacity_bytes = 0;
    const void* gate_weight_source = nullptr;
    const void* gate_scales_source = nullptr;
    const void* gate_biases_source = nullptr;
    const void* up_weight_source = nullptr;
    const void* up_scales_source = nullptr;
    const void* up_biases_source = nullptr;
    const void* down_weight_source = nullptr;
    const void* down_scales_source = nullptr;
    const void* down_biases_source = nullptr;
    std::string error;
};

bool UseResidentOProjectionBuffers() {
    const char* value = std::getenv("GYPSY_RESIDENT_O_PROJECTION_BUFFERS");
    return value != nullptr && std::string(value) == "1";
}

DirectMetalAttentionContext& AttentionContext() {
    static DirectMetalAttentionContext context;
    static bool initialized = false;
    if (initialized) return context;
    initialized = true;

    @autoreleasepool {
        context.device = MTLCreateSystemDefaultDevice();
        if (context.device == nil) {
            context.error = "MTLCreateSystemDefaultDevice returned nil";
            return context;
        }

        NSError* error = nil;
        context.library = [context.device newLibraryWithSource:AttentionKernelSource() options:nil error:&error];
        if (context.library == nil) {
            context.error = "newLibraryWithSource failed: " + NSErrorMessage(error);
            return context;
        }

        context.function = [context.library newFunctionWithName:@"gypsy_attention_probe"];
        if (context.function == nil) {
            context.error = "newFunctionWithName gypsy_attention_probe failed";
            return context;
        }

        context.pipeline = [context.device newComputePipelineStateWithFunction:context.function error:&error];
        if (context.pipeline == nil) {
            context.error = "newComputePipelineStateWithFunction failed: " + NSErrorMessage(error);
            return context;
        }

        context.o_projection_function = [context.library newFunctionWithName:@"gypsy_o_projection_from_attention"];
        if (context.o_projection_function == nil) {
            context.error = "newFunctionWithName gypsy_o_projection_from_attention failed";
            return context;
        }

        context.o_projection_pipeline = [context.device newComputePipelineStateWithFunction:context.o_projection_function error:&error];
        if (context.o_projection_pipeline == nil) {
            context.error = "newComputePipelineStateWithFunction o projection failed: " + NSErrorMessage(error);
            return context;
        }

        context.qkv_projection_function = [context.library newFunctionWithName:@"gypsy_qkv_projection_2560_int4"];
        if (context.qkv_projection_function == nil) {
            context.error = "newFunctionWithName gypsy_qkv_projection_2560_int4 failed";
            return context;
        }

        context.qkv_projection_pipeline = [context.device newComputePipelineStateWithFunction:context.qkv_projection_function error:&error];
        if (context.qkv_projection_pipeline == nil) {
            context.error = "newComputePipelineStateWithFunction qkv projection failed: " + NSErrorMessage(error);
            return context;
        }

        context.qk_norm_rope_function = [context.library newFunctionWithName:@"gypsy_qk_norm_rope_qwen"];
        if (context.qk_norm_rope_function == nil) {
            context.error = "newFunctionWithName gypsy_qk_norm_rope_qwen failed";
            return context;
        }

        context.qk_norm_rope_pipeline = [context.device newComputePipelineStateWithFunction:context.qk_norm_rope_function error:&error];
        if (context.qk_norm_rope_pipeline == nil) {
            context.error = "newComputePipelineStateWithFunction qk norm rope failed: " + NSErrorMessage(error);
            return context;
        }

        context.residual_add_function = [context.library newFunctionWithName:@"gypsy_add_residual_2560"];
        if (context.residual_add_function == nil) {
            context.error = "newFunctionWithName gypsy_add_residual_2560 failed";
            return context;
        }

        context.residual_add_pipeline = [context.device newComputePipelineStateWithFunction:context.residual_add_function error:&error];
        if (context.residual_add_pipeline == nil) {
            context.error = "newComputePipelineStateWithFunction residual add failed: " + NSErrorMessage(error);
            return context;
        }

        context.rmsnorm_function = [context.library newFunctionWithName:@"gypsy_rmsnorm_2560"];
        if (context.rmsnorm_function == nil) {
            context.error = "newFunctionWithName gypsy_rmsnorm_2560 failed";
            return context;
        }

        context.rmsnorm_pipeline = [context.device newComputePipelineStateWithFunction:context.rmsnorm_function error:&error];
        if (context.rmsnorm_pipeline == nil) {
            context.error = "newComputePipelineStateWithFunction rmsnorm failed: " + NSErrorMessage(error);
            return context;
        }

        context.silu_multiply_function = [context.library newFunctionWithName:@"gypsy_silu_multiply"];
        if (context.silu_multiply_function == nil) {
            context.error = "newFunctionWithName gypsy_silu_multiply failed";
            return context;
        }

        context.silu_multiply_pipeline = [context.device newComputePipelineStateWithFunction:context.silu_multiply_function error:&error];
        if (context.silu_multiply_pipeline == nil) {
            context.error = "newComputePipelineStateWithFunction silu multiply failed: " + NSErrorMessage(error);
            return context;
        }

        context.down_projection_function = [context.library newFunctionWithName:@"gypsy_down_projection_9728_int4"];
        if (context.down_projection_function == nil) {
            context.error = "newFunctionWithName gypsy_down_projection_9728_int4 failed";
            return context;
        }

        context.down_projection_pipeline = [context.device newComputePipelineStateWithFunction:context.down_projection_function error:&error];
        if (context.down_projection_pipeline == nil) {
            context.error = "newComputePipelineStateWithFunction down projection failed: " + NSErrorMessage(error);
            return context;
        }

        context.queue = [context.device newCommandQueue];
        if (context.queue == nil) {
            context.error = "newCommandQueue failed";
            return context;
        }
    }

    return context;
}

GypsyDirectMetalProbeResult RunDirectAttention(
    const float* q,
    const float* k,
    const float* v,
    const float* expected,
    float* host_out,
    std::uint32_t seq_len) {
    GypsyDirectMetalProbeResult result;
    const auto start = std::chrono::steady_clock::now();

    @autoreleasepool {
        DirectMetalAttentionContext& context = AttentionContext();
        result.device_available = context.device != nil;
        result.library_compiled = context.library != nil;
        result.pipeline_created = context.pipeline != nil;
        if (!context.error.empty()) {
            result.error = PersistError(context.error);
            return result;
        }

        constexpr std::uint32_t q_heads = 32;
        constexpr std::uint32_t kv_heads = 8;
        constexpr std::uint32_t head_dim = 128;
        const std::uint32_t n = std::max<std::uint32_t>(1, std::min<std::uint32_t>(seq_len, 256));
        const size_t q_bytes = static_cast<size_t>(q_heads) * head_dim * sizeof(float);
        const size_t kv_bytes = static_cast<size_t>(n) * kv_heads * head_dim * sizeof(float);
        const size_t out_bytes = static_cast<size_t>(q_heads) * head_dim * sizeof(float);
        if (context.q_buffer == nil || context.q_capacity_bytes < q_bytes) {
            context.q_buffer = [context.device newBufferWithLength:q_bytes options:MTLResourceStorageModeShared];
            context.q_capacity_bytes = q_bytes;
        }
        if (context.k_buffer == nil || context.kv_capacity_bytes < kv_bytes) {
            context.k_buffer = [context.device newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];
            context.v_buffer = [context.device newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];
            context.kv_capacity_bytes = kv_bytes;
        }
        if (context.out_buffer == nil || context.out_capacity_bytes < out_bytes) {
            context.out_buffer = [context.device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
            context.out_capacity_bytes = out_bytes;
        }
        if (context.n_buffer == nil) {
            context.n_buffer = [context.device newBufferWithLength:sizeof(n) options:MTLResourceStorageModeShared];
        }
        if (context.q_buffer == nil || context.k_buffer == nil || context.v_buffer == nil ||
            context.out_buffer == nil || context.n_buffer == nil) {
            result.error = PersistError("newBuffer allocation failed");
            return result;
        }
        std::memcpy(context.q_buffer.contents, q, q_bytes);
        std::memcpy(context.k_buffer.contents, k, kv_bytes);
        std::memcpy(context.v_buffer.contents, v, kv_bytes);
        std::memcpy(context.n_buffer.contents, &n, sizeof(n));

        id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            result.error = PersistError("command buffer or encoder creation failed");
            return result;
        }

        [encoder setComputePipelineState:context.pipeline];
        [encoder setBuffer:context.q_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.k_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.v_buffer offset:0 atIndex:2];
        [encoder setBuffer:context.out_buffer offset:0 atIndex:3];
        [encoder setBuffer:context.n_buffer offset:0 atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(q_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(q_heads, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            result.error = PersistError("command buffer did not complete: " + NSErrorMessage(command_buffer.error));
            return result;
        }
        result.command_completed = true;

        float* out = static_cast<float*>(context.out_buffer.contents);
        double checksum = 0.0;
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < static_cast<size_t>(q_heads) * head_dim; ++i) {
            if (host_out != nullptr) {
                host_out[i] = out[i];
            }
            if (expected != nullptr) {
                const float diff = std::fabs(out[i] - expected[i]);
                max_abs_diff = std::max(max_abs_diff, diff);
            }
            checksum += static_cast<double>(out[i]);
        }
        result.checksum = static_cast<float>(checksum);
        result.max_abs_diff = max_abs_diff;
        result.ok = expected == nullptr || max_abs_diff < 1.0e-5f;
    }

    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

GypsyDirectMetalProbeResult RunDirectAttentionWithResidentLayerKv(
    std::uint32_t layer,
    const float* q,
    const float* current_k,
    const float* current_v,
    float* host_out,
    std::uint32_t seq_len) {
    GypsyDirectMetalProbeResult result;
    const auto start = std::chrono::steady_clock::now();

    @autoreleasepool {
        DirectMetalAttentionContext& context = AttentionContext();
        result.device_available = context.device != nil;
        result.library_compiled = context.library != nil;
        result.pipeline_created = context.pipeline != nil;
        if (!context.error.empty()) {
            result.error = PersistError(context.error);
            return result;
        }
        if (layer >= context.layer_kv.size()) {
            result.error = PersistError("direct Metal attention layer index out of range");
            return result;
        }

        constexpr std::uint32_t q_heads = 32;
        constexpr std::uint32_t kv_heads = 8;
        constexpr std::uint32_t head_dim = 128;
        const std::uint32_t n = std::max<std::uint32_t>(1, std::min<std::uint32_t>(seq_len, 256));
        const size_t q_bytes = static_cast<size_t>(q_heads) * head_dim * sizeof(float);
        const size_t current_kv_bytes = static_cast<size_t>(kv_heads) * head_dim * sizeof(float);
        const size_t kv_capacity_bytes = static_cast<size_t>(256) * kv_heads * head_dim * sizeof(float);
        const size_t out_bytes = static_cast<size_t>(q_heads) * head_dim * sizeof(float);

        if (context.q_buffer == nil || context.q_capacity_bytes < q_bytes) {
            context.q_buffer = [context.device newBufferWithLength:q_bytes options:MTLResourceStorageModeShared];
            context.q_capacity_bytes = q_bytes;
        }
        DirectMetalAttentionContext::LayerKvBuffers& layer_buffers = context.layer_kv[layer];
        if (layer_buffers.k_buffer == nil || layer_buffers.capacity_bytes < kv_capacity_bytes) {
            layer_buffers.k_buffer = [context.device newBufferWithLength:kv_capacity_bytes options:MTLResourceStorageModeShared];
            layer_buffers.v_buffer = [context.device newBufferWithLength:kv_capacity_bytes options:MTLResourceStorageModeShared];
            layer_buffers.capacity_bytes = kv_capacity_bytes;
        }
        if (context.out_buffer == nil || context.out_capacity_bytes < out_bytes) {
            context.out_buffer = [context.device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
            context.out_capacity_bytes = out_bytes;
        }
        if (context.n_buffer == nil) {
            context.n_buffer = [context.device newBufferWithLength:sizeof(n) options:MTLResourceStorageModeShared];
        }
        if (context.q_buffer == nil || layer_buffers.k_buffer == nil || layer_buffers.v_buffer == nil ||
            context.out_buffer == nil || context.n_buffer == nil) {
            result.error = PersistError("newBuffer allocation failed");
            return result;
        }

        std::memcpy(context.q_buffer.contents, q, q_bytes);
        const size_t write_offset = static_cast<size_t>(n - 1) * current_kv_bytes;
        std::memcpy(static_cast<char*>(layer_buffers.k_buffer.contents) + write_offset, current_k, current_kv_bytes);
        std::memcpy(static_cast<char*>(layer_buffers.v_buffer.contents) + write_offset, current_v, current_kv_bytes);
        std::memcpy(context.n_buffer.contents, &n, sizeof(n));

        id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            result.error = PersistError("command buffer or encoder creation failed");
            return result;
        }

        [encoder setComputePipelineState:context.pipeline];
        [encoder setBuffer:context.q_buffer offset:0 atIndex:0];
        [encoder setBuffer:layer_buffers.k_buffer offset:0 atIndex:1];
        [encoder setBuffer:layer_buffers.v_buffer offset:0 atIndex:2];
        [encoder setBuffer:context.out_buffer offset:0 atIndex:3];
        [encoder setBuffer:context.n_buffer offset:0 atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(q_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(q_heads, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            result.error = PersistError("command buffer did not complete: " + NSErrorMessage(command_buffer.error));
            return result;
        }
        result.command_completed = true;

        float* out = static_cast<float*>(context.out_buffer.contents);
        double checksum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(q_heads) * head_dim; ++i) {
            if (host_out != nullptr) {
                host_out[i] = out[i];
            }
            checksum += static_cast<double>(out[i]);
        }
        result.checksum = static_cast<float>(checksum);
        result.max_abs_diff = 0.0f;
        result.ok = true;
    }

    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

GypsyDirectMetalProbeResult RunDirectAttentionOProjectionWithResidentLayerKv(
    std::uint32_t layer,
    const float* q,
    const float* current_k,
    const float* current_v,
    const std::uint32_t* o_weight,
    const std::uint16_t* o_scales,
    const std::uint16_t* o_biases,
    float* host_out,
    std::uint32_t seq_len) {
    GypsyDirectMetalProbeResult result;
    const auto start = std::chrono::steady_clock::now();

    @autoreleasepool {
        DirectMetalAttentionContext& context = AttentionContext();
        result.device_available = context.device != nil;
        result.library_compiled = context.library != nil;
        result.pipeline_created = context.pipeline != nil && context.o_projection_pipeline != nil;
        if (!context.error.empty()) {
            result.error = PersistError(context.error);
            return result;
        }
        if (layer >= context.layer_kv.size()) {
            result.error = PersistError("direct Metal attention+o layer index out of range");
            return result;
        }

        constexpr std::uint32_t q_heads = 32;
        constexpr std::uint32_t kv_heads = 8;
        constexpr std::uint32_t head_dim = 128;
        constexpr std::uint32_t hidden_size = 2560;
        constexpr std::uint32_t o_rows = 2560;
        constexpr std::uint32_t o_packed_cols = 512;
        constexpr std::uint32_t o_groups = 64;
        const bool use_resident_o_projection_buffers = UseResidentOProjectionBuffers();
        const std::uint32_t n = std::max<std::uint32_t>(1, std::min<std::uint32_t>(seq_len, 256));
        const size_t q_bytes = static_cast<size_t>(q_heads) * head_dim * sizeof(float);
        const size_t current_kv_bytes = static_cast<size_t>(kv_heads) * head_dim * sizeof(float);
        const size_t kv_capacity_bytes = static_cast<size_t>(256) * kv_heads * head_dim * sizeof(float);
        const size_t attention_out_bytes = static_cast<size_t>(q_heads) * head_dim * sizeof(float);
        const size_t o_weight_bytes = static_cast<size_t>(o_rows) * o_packed_cols * sizeof(std::uint32_t);
        const size_t o_scale_bytes = static_cast<size_t>(o_rows) * o_groups * sizeof(std::uint16_t);
        const size_t o_out_bytes = static_cast<size_t>(hidden_size) * sizeof(float);

        if (context.q_buffer == nil || context.q_capacity_bytes < q_bytes) {
            context.q_buffer = [context.device newBufferWithLength:q_bytes options:MTLResourceStorageModeShared];
            context.q_capacity_bytes = q_bytes;
        }
        DirectMetalAttentionContext::LayerKvBuffers& layer_buffers = context.layer_kv[layer];
        if (layer_buffers.k_buffer == nil || layer_buffers.capacity_bytes < kv_capacity_bytes) {
            layer_buffers.k_buffer = [context.device newBufferWithLength:kv_capacity_bytes options:MTLResourceStorageModeShared];
            layer_buffers.v_buffer = [context.device newBufferWithLength:kv_capacity_bytes options:MTLResourceStorageModeShared];
            layer_buffers.capacity_bytes = kv_capacity_bytes;
        }
        if (context.out_buffer == nil || context.out_capacity_bytes < attention_out_bytes) {
            context.out_buffer = [context.device newBufferWithLength:attention_out_bytes options:MTLResourceStorageModeShared];
            context.out_capacity_bytes = attention_out_bytes;
        }
        if (context.o_out_buffer == nil || context.o_out_capacity_bytes < o_out_bytes) {
            context.o_out_buffer = [context.device newBufferWithLength:o_out_bytes options:MTLResourceStorageModeShared];
            context.o_out_capacity_bytes = o_out_bytes;
        }
        if (context.n_buffer == nil) {
            context.n_buffer = [context.device newBufferWithLength:sizeof(n) options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> o_weight_buffer = nil;
        id<MTLBuffer> o_scales_buffer = nil;
        id<MTLBuffer> o_biases_buffer = nil;
        if (use_resident_o_projection_buffers) {
            if (layer_buffers.o_weight_buffer == nil || layer_buffers.o_weight_capacity_bytes < o_weight_bytes) {
                layer_buffers.o_weight_buffer = [context.device newBufferWithLength:o_weight_bytes options:MTLResourceStorageModeShared];
                layer_buffers.o_weight_capacity_bytes = o_weight_bytes;
                layer_buffers.o_weight_source = nullptr;
            }
            if (layer_buffers.o_scales_buffer == nil || layer_buffers.o_scale_capacity_bytes < o_scale_bytes) {
                layer_buffers.o_scales_buffer = [context.device newBufferWithLength:o_scale_bytes options:MTLResourceStorageModeShared];
                layer_buffers.o_biases_buffer = [context.device newBufferWithLength:o_scale_bytes options:MTLResourceStorageModeShared];
                layer_buffers.o_scale_capacity_bytes = o_scale_bytes;
                layer_buffers.o_scales_source = nullptr;
                layer_buffers.o_biases_source = nullptr;
            }
            o_weight_buffer = layer_buffers.o_weight_buffer;
            o_scales_buffer = layer_buffers.o_scales_buffer;
            o_biases_buffer = layer_buffers.o_biases_buffer;
        } else {
            if (context.o_weight_buffer == nil || context.o_weight_capacity_bytes < o_weight_bytes) {
                context.o_weight_buffer = [context.device newBufferWithLength:o_weight_bytes options:MTLResourceStorageModeShared];
                context.o_weight_capacity_bytes = o_weight_bytes;
            }
            if (context.o_scales_buffer == nil || context.o_scale_capacity_bytes < o_scale_bytes) {
                context.o_scales_buffer = [context.device newBufferWithLength:o_scale_bytes options:MTLResourceStorageModeShared];
                context.o_biases_buffer = [context.device newBufferWithLength:o_scale_bytes options:MTLResourceStorageModeShared];
                context.o_scale_capacity_bytes = o_scale_bytes;
            }
            o_weight_buffer = context.o_weight_buffer;
            o_scales_buffer = context.o_scales_buffer;
            o_biases_buffer = context.o_biases_buffer;
        }
        if (context.q_buffer == nil || layer_buffers.k_buffer == nil || layer_buffers.v_buffer == nil ||
            context.out_buffer == nil || context.o_out_buffer == nil || context.n_buffer == nil ||
            o_weight_buffer == nil ||
            o_scales_buffer == nil ||
            o_biases_buffer == nil) {
            result.error = PersistError("newBuffer allocation failed for attention+o");
            return result;
        }

        if (use_resident_o_projection_buffers) {
            if (layer_buffers.o_weight_source != o_weight) {
                std::memcpy(o_weight_buffer.contents, o_weight, o_weight_bytes);
                layer_buffers.o_weight_source = o_weight;
            }
            if (layer_buffers.o_scales_source != o_scales) {
                std::memcpy(o_scales_buffer.contents, o_scales, o_scale_bytes);
                layer_buffers.o_scales_source = o_scales;
            }
            if (layer_buffers.o_biases_source != o_biases) {
                std::memcpy(o_biases_buffer.contents, o_biases, o_scale_bytes);
                layer_buffers.o_biases_source = o_biases;
            }
        } else {
            std::memcpy(o_weight_buffer.contents, o_weight, o_weight_bytes);
            std::memcpy(o_scales_buffer.contents, o_scales, o_scale_bytes);
            std::memcpy(o_biases_buffer.contents, o_biases, o_scale_bytes);
        }
        std::memcpy(context.q_buffer.contents, q, q_bytes);
        const size_t write_offset = static_cast<size_t>(n - 1) * current_kv_bytes;
        std::memcpy(static_cast<char*>(layer_buffers.k_buffer.contents) + write_offset, current_k, current_kv_bytes);
        std::memcpy(static_cast<char*>(layer_buffers.v_buffer.contents) + write_offset, current_v, current_kv_bytes);
        std::memcpy(context.n_buffer.contents, &n, sizeof(n));

        id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            result.error = PersistError("command buffer or encoder creation failed");
            return result;
        }
        [encoder setComputePipelineState:context.pipeline];
        [encoder setBuffer:context.q_buffer offset:0 atIndex:0];
        [encoder setBuffer:layer_buffers.k_buffer offset:0 atIndex:1];
        [encoder setBuffer:layer_buffers.v_buffer offset:0 atIndex:2];
        [encoder setBuffer:context.out_buffer offset:0 atIndex:3];
        [encoder setBuffer:context.n_buffer offset:0 atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(q_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(q_heads, 1, 1)];

        [encoder setComputePipelineState:context.o_projection_pipeline];
        [encoder setBuffer:context.out_buffer offset:0 atIndex:0];
        [encoder setBuffer:o_weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:o_scales_buffer offset:0 atIndex:2];
        [encoder setBuffer:o_biases_buffer offset:0 atIndex:3];
        [encoder setBuffer:context.o_out_buffer offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(o_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            result.error = PersistError("command buffer did not complete: " + NSErrorMessage(command_buffer.error));
            return result;
        }
        result.command_completed = true;

        float* out = static_cast<float*>(context.o_out_buffer.contents);
        double checksum = 0.0;
        for (size_t i = 0; i < hidden_size; ++i) {
            if (host_out != nullptr) {
                host_out[i] = out[i];
            }
            checksum += static_cast<double>(out[i]);
        }
        result.checksum = static_cast<float>(checksum);
        result.max_abs_diff = 0.0f;
        result.ok = true;
    }

    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

GypsyDirectMetalProbeResult RunDirectQuantizedProjection(
    const float* input,
    const std::uint32_t* weight,
    const std::uint16_t* scales,
    const std::uint16_t* biases,
    const float* expected,
    float* host_out,
    std::uint32_t rows,
    std::uint32_t packed_cols,
    std::uint32_t input_len) {
    GypsyDirectMetalProbeResult result;
    const auto start = std::chrono::steady_clock::now();

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        result.device_available = device != nil;
        if (device == nil) {
            result.error = PersistError("MTLCreateSystemDefaultDevice returned nil");
            return result;
        }

        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:QuantizedProjectionKernelSource() options:nil error:&error];
        result.library_compiled = library != nil;
        if (library == nil) {
            result.error = PersistError("newLibraryWithSource quantized projection failed: " + NSErrorMessage(error));
            return result;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"gypsy_quantized_projection_probe"];
        if (function == nil) {
            result.error = PersistError("newFunctionWithName gypsy_quantized_projection_probe failed");
            return result;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        result.pipeline_created = pipeline != nil;
        if (pipeline == nil) {
            result.error = PersistError("newComputePipelineStateWithFunction quantized projection failed: " + NSErrorMessage(error));
            return result;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            result.error = PersistError("newCommandQueue failed");
            return result;
        }

        const std::uint32_t groups = input_len / 64;
        const size_t input_bytes = static_cast<size_t>(input_len) * sizeof(float);
        const size_t weight_bytes = static_cast<size_t>(rows) * packed_cols * sizeof(std::uint32_t);
        const size_t scale_bytes = static_cast<size_t>(rows) * groups * sizeof(std::uint16_t);
        const size_t out_bytes = static_cast<size_t>(rows) * sizeof(float);
        id<MTLBuffer> input_buffer = [device newBufferWithBytes:input length:input_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> weight_buffer = [device newBufferWithBytes:weight length:weight_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> scales_buffer = [device newBufferWithBytes:scales length:scale_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> biases_buffer = [device newBufferWithBytes:biases length:scale_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buffer = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> rows_buffer = [device newBufferWithBytes:&rows length:sizeof(rows) options:MTLResourceStorageModeShared];
        id<MTLBuffer> packed_cols_buffer = [device newBufferWithBytes:&packed_cols length:sizeof(packed_cols) options:MTLResourceStorageModeShared];
        id<MTLBuffer> input_len_buffer = [device newBufferWithBytes:&input_len length:sizeof(input_len) options:MTLResourceStorageModeShared];
        if (input_buffer == nil || weight_buffer == nil || scales_buffer == nil || biases_buffer == nil ||
            out_buffer == nil || rows_buffer == nil || packed_cols_buffer == nil || input_len_buffer == nil) {
            result.error = PersistError("newBuffer allocation failed for quantized projection");
            return result;
        }

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            result.error = PersistError("command buffer or encoder creation failed");
            return result;
        }
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:scales_buffer offset:0 atIndex:2];
        [encoder setBuffer:biases_buffer offset:0 atIndex:3];
        [encoder setBuffer:out_buffer offset:0 atIndex:4];
        [encoder setBuffer:rows_buffer offset:0 atIndex:5];
        [encoder setBuffer:packed_cols_buffer offset:0 atIndex:6];
        [encoder setBuffer:input_len_buffer offset:0 atIndex:7];
        const NSUInteger width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
        [encoder dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            result.error = PersistError("command buffer did not complete: " + NSErrorMessage(command_buffer.error));
            return result;
        }
        result.command_completed = true;

        float* out = static_cast<float*>(out_buffer.contents);
        double checksum = 0.0;
        float max_abs_diff = 0.0f;
        for (std::uint32_t i = 0; i < rows; ++i) {
            if (host_out != nullptr) host_out[i] = out[i];
            if (expected != nullptr) {
                max_abs_diff = std::max(max_abs_diff, std::fabs(out[i] - expected[i]));
            }
            checksum += static_cast<double>(out[i]);
        }
        result.checksum = static_cast<float>(checksum);
        result.max_abs_diff = max_abs_diff;
        result.ok = expected == nullptr || max_abs_diff < 1.0e-3f;
    }

    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

GypsyDirectMetalProbeResult RunDirectQuantizedProjectionTiled(
    const float* input,
    const std::uint32_t* weight,
    const std::uint16_t* scales,
    const std::uint16_t* biases,
    const float* expected,
    float* host_out,
    std::uint32_t rows,
    std::uint32_t packed_cols,
    std::uint32_t input_len,
    bool top1_only = false) {
    GypsyDirectMetalProbeResult result;
    const auto start = std::chrono::steady_clock::now();

    struct TiledProjectionContext {
        id<MTLDevice> device = nil;
        id<MTLLibrary> library = nil;
        id<MTLFunction> function = nil;
        id<MTLFunction> top1_function = nil;
        id<MTLFunction> top1_serial_function = nil;
        id<MTLComputePipelineState> pipeline = nil;
        id<MTLComputePipelineState> top1_pipeline = nil;
        id<MTLComputePipelineState> top1_serial_pipeline = nil;
        id<MTLCommandQueue> queue = nil;
        id<MTLBuffer> input_buffer = nil;
        id<MTLBuffer> weight_buffer = nil;
        id<MTLBuffer> scales_buffer = nil;
        id<MTLBuffer> biases_buffer = nil;
        id<MTLBuffer> out_buffer = nil;
        id<MTLBuffer> partial_scores_buffer = nil;
        id<MTLBuffer> partial_ids_buffer = nil;
        size_t input_capacity_bytes = 0;
        size_t weight_capacity_bytes = 0;
        size_t scale_capacity_bytes = 0;
        size_t out_capacity_bytes = 0;
        size_t partial_scores_capacity_bytes = 0;
        size_t partial_ids_capacity_bytes = 0;
        const void* weight_source = nullptr;
        const void* scales_source = nullptr;
        const void* biases_source = nullptr;
        std::string error;
    };
    auto context_key = std::to_string(rows) + "x" + std::to_string(input_len) + "p" + std::to_string(packed_cols);
    static std::map<std::string, std::unique_ptr<TiledProjectionContext>> contexts;

    @autoreleasepool {
        TiledProjectionContext* context = nullptr;
        auto it = contexts.find(context_key);
        if (it == contexts.end()) {
            auto created = std::make_unique<TiledProjectionContext>();
            created->device = MTLCreateSystemDefaultDevice();
            if (created->device == nil) {
                created->error = "MTLCreateSystemDefaultDevice returned nil";
            } else {
                NSError* error = nil;
                created->library = [created->device newLibraryWithSource:QuantizedProjectionTiledKernelSource(rows, packed_cols, input_len) options:nil error:&error];
                if (created->library == nil) {
                    created->error = "newLibraryWithSource tiled quantized projection failed: " + NSErrorMessage(error);
                } else {
                    created->function = [created->library newFunctionWithName:@"gypsy_quantized_projection_tiled_probe"];
                    if (created->function == nil) {
                        created->error = "newFunctionWithName gypsy_quantized_projection_tiled_probe failed";
                    } else {
                        created->pipeline = [created->device newComputePipelineStateWithFunction:created->function error:&error];
                        if (created->pipeline == nil) {
                            created->error = "newComputePipelineStateWithFunction tiled quantized projection failed: " + NSErrorMessage(error);
                        } else {
                            created->top1_function = [created->library newFunctionWithName:@"gypsy_top1_reduce_probe"];
                            if (created->top1_function == nil) {
                                created->error = "newFunctionWithName gypsy_top1_reduce_probe failed";
                            } else {
                                created->top1_pipeline = [created->device newComputePipelineStateWithFunction:created->top1_function error:&error];
                                if (created->top1_pipeline == nil) {
                                    created->error = "newComputePipelineStateWithFunction top1 reduce failed: " + NSErrorMessage(error);
                                } else {
                                    created->top1_serial_function = [created->library newFunctionWithName:@"gypsy_top1_serial_probe"];
                                    if (created->top1_serial_function == nil) {
                                        created->error = "newFunctionWithName gypsy_top1_serial_probe failed";
                                    } else {
                                        created->top1_serial_pipeline = [created->device newComputePipelineStateWithFunction:created->top1_serial_function error:&error];
                                        if (created->top1_serial_pipeline == nil) {
                                            created->error = "newComputePipelineStateWithFunction top1 serial failed: " + NSErrorMessage(error);
                                        } else {
                                            created->queue = [created->device newCommandQueue];
                                            if (created->queue == nil) {
                                                created->error = "newCommandQueue failed";
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            context = created.get();
            contexts.emplace(context_key, std::move(created));
        } else {
            context = it->second.get();
        }

        result.device_available = context->device != nil;
        result.library_compiled = context->library != nil;
        result.pipeline_created = context->pipeline != nil;
        if (!context->error.empty()) {
            result.error = PersistError(context->error);
            return result;
        }

        const std::uint32_t groups = input_len / 64;
        const size_t input_bytes = static_cast<size_t>(input_len) * sizeof(float);
        const size_t weight_bytes = static_cast<size_t>(rows) * packed_cols * sizeof(std::uint32_t);
        const size_t scale_bytes = static_cast<size_t>(rows) * groups * sizeof(std::uint16_t);
        const size_t out_bytes = static_cast<size_t>(rows) * sizeof(float);
        if (context->input_buffer == nil || context->input_capacity_bytes < input_bytes) {
            context->input_buffer = [context->device newBufferWithLength:input_bytes options:MTLResourceStorageModeShared];
            context->input_capacity_bytes = input_bytes;
        }
        if (context->weight_buffer == nil || context->weight_capacity_bytes < weight_bytes) {
            context->weight_buffer = [context->device newBufferWithLength:weight_bytes options:MTLResourceStorageModeShared];
            context->weight_capacity_bytes = weight_bytes;
        }
        if (context->scales_buffer == nil || context->scale_capacity_bytes < scale_bytes) {
            context->scales_buffer = [context->device newBufferWithLength:scale_bytes options:MTLResourceStorageModeShared];
            context->biases_buffer = [context->device newBufferWithLength:scale_bytes options:MTLResourceStorageModeShared];
            context->scale_capacity_bytes = scale_bytes;
        }
        if (context->out_buffer == nil || context->out_capacity_bytes < out_bytes) {
            context->out_buffer = [context->device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
            context->out_capacity_bytes = out_bytes;
        }
        const std::uint32_t partial_count = (rows + 255) / 256;
        const size_t partial_scores_bytes = static_cast<size_t>(partial_count) * sizeof(float);
        const size_t partial_ids_bytes = static_cast<size_t>(partial_count) * sizeof(std::uint32_t);
        if (context->partial_scores_buffer == nil || context->partial_scores_capacity_bytes < partial_scores_bytes) {
            context->partial_scores_buffer = [context->device newBufferWithLength:partial_scores_bytes options:MTLResourceStorageModeShared];
            context->partial_scores_capacity_bytes = partial_scores_bytes;
        }
        if (context->partial_ids_buffer == nil || context->partial_ids_capacity_bytes < partial_ids_bytes) {
            context->partial_ids_buffer = [context->device newBufferWithLength:partial_ids_bytes options:MTLResourceStorageModeShared];
            context->partial_ids_capacity_bytes = partial_ids_bytes;
        }
        if (context->input_buffer == nil || context->weight_buffer == nil || context->scales_buffer == nil ||
            context->biases_buffer == nil || context->out_buffer == nil ||
            context->partial_scores_buffer == nil || context->partial_ids_buffer == nil) {
            result.error = PersistError("newBuffer allocation failed for tiled quantized projection");
            return result;
        }
        std::memcpy(context->input_buffer.contents, input, input_bytes);
        if (context->weight_source != weight) {
            std::memcpy(context->weight_buffer.contents, weight, weight_bytes);
            context->weight_source = weight;
        }
        if (context->scales_source != scales) {
            std::memcpy(context->scales_buffer.contents, scales, scale_bytes);
            context->scales_source = scales;
        }
        if (context->biases_source != biases) {
            std::memcpy(context->biases_buffer.contents, biases, scale_bytes);
            context->biases_source = biases;
        }

        id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            result.error = PersistError("command buffer or encoder creation failed");
            return result;
        }
        [encoder setComputePipelineState:context->pipeline];
        [encoder setBuffer:context->input_buffer offset:0 atIndex:0];
        [encoder setBuffer:context->weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:context->scales_buffer offset:0 atIndex:2];
        [encoder setBuffer:context->biases_buffer offset:0 atIndex:3];
        [encoder setBuffer:context->out_buffer offset:0 atIndex:4];
        constexpr NSUInteger threads_per_row = 64;
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads_per_row, 1, 1)];
        if (top1_only) {
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [encoder setComputePipelineState:context->top1_serial_pipeline];
            [encoder setBuffer:context->out_buffer offset:0 atIndex:0];
            [encoder setBuffer:context->partial_scores_buffer offset:0 atIndex:1];
            [encoder setBuffer:context->partial_ids_buffer offset:0 atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        }
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            result.error = PersistError("command buffer did not complete: " + NSErrorMessage(command_buffer.error));
            return result;
        }
        result.command_completed = true;

        if (top1_only) {
            const float* partial_scores = static_cast<const float*>(context->partial_scores_buffer.contents);
            const std::uint32_t* partial_ids = static_cast<const std::uint32_t*>(context->partial_ids_buffer.contents);
            std::uint32_t top_id = partial_ids[0];
            float top_score = partial_scores[0];
            double checksum = 0.0;
            checksum = static_cast<double>(top_score);
            result.top_id = top_id;
            result.top_score = top_score;
            result.checksum = static_cast<float>(checksum);
            result.max_abs_diff = 0.0f;
            result.ok = true;
            result.elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            return result;
        }

        float* out = static_cast<float*>(context->out_buffer.contents);
        double checksum = 0.0;
        float max_abs_diff = 0.0f;
        for (std::uint32_t i = 0; i < rows; ++i) {
            if (host_out != nullptr) host_out[i] = out[i];
            if (expected != nullptr) {
                max_abs_diff = std::max(max_abs_diff, std::fabs(out[i] - expected[i]));
            }
            checksum += static_cast<double>(out[i]);
        }
        result.checksum = static_cast<float>(checksum);
        result.max_abs_diff = max_abs_diff;
        result.ok = expected == nullptr || max_abs_diff < 1.0e-3f;
    }

    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

} // namespace

GypsyDirectMetalProbeResult GypsyRunDirectMetalAddProbe(std::uint32_t n) {
    GypsyDirectMetalProbeResult result;
    const auto start = std::chrono::steady_clock::now();

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        result.device_available = device != nil;
        if (device == nil) {
            result.error = PersistError("MTLCreateSystemDefaultDevice returned nil");
            return result;
        }

        static NSString* source = @R"METAL(
            #include <metal_stdlib>
            using namespace metal;

            kernel void gypsy_add_probe(
                const device float* a [[buffer(0)]],
                const device float* b [[buffer(1)]],
                device float* out [[buffer(2)]],
                constant uint& n [[buffer(3)]],
                uint gid [[thread_position_in_grid]]) {
                if (gid >= n) return;
                out[gid] = a[gid] + b[gid];
            }
        )METAL";

        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        result.library_compiled = library != nil;
        if (library == nil) {
            result.error = PersistError("newLibraryWithSource failed: " + NSErrorMessage(error));
            return result;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"gypsy_add_probe"];
        if (function == nil) {
            result.error = PersistError("newFunctionWithName gypsy_add_probe failed");
            return result;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        result.pipeline_created = pipeline != nil;
        if (pipeline == nil) {
            result.error = PersistError("newComputePipelineStateWithFunction failed: " + NSErrorMessage(error));
            return result;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            result.error = PersistError("newCommandQueue failed");
            return result;
        }

        const std::uint32_t count = std::max<std::uint32_t>(1, n);
        const size_t bytes = static_cast<size_t>(count) * sizeof(float);
        std::vector<float> a(count);
        std::vector<float> b(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            a[i] = static_cast<float>(i % 97) * 0.125f;
            b[i] = static_cast<float>((i * 3) % 89) * -0.0625f;
        }

        id<MTLBuffer> a_buffer = [device newBufferWithBytes:a.data() length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_buffer = [device newBufferWithBytes:b.data() length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> n_buffer = [device newBufferWithBytes:&count length:sizeof(count) options:MTLResourceStorageModeShared];
        if (a_buffer == nil || b_buffer == nil || out_buffer == nil || n_buffer == nil) {
            result.error = PersistError("newBuffer allocation failed");
            return result;
        }

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            result.error = PersistError("command buffer or encoder creation failed");
            return result;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:b_buffer offset:0 atIndex:1];
        [encoder setBuffer:out_buffer offset:0 atIndex:2];
        [encoder setBuffer:n_buffer offset:0 atIndex:3];

        const NSUInteger width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
        MTLSize grid_size = MTLSizeMake(count, 1, 1);
        MTLSize threadgroup_size = MTLSizeMake(width, 1, 1);
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            result.error = PersistError("command buffer did not complete: " + NSErrorMessage(command_buffer.error));
            return result;
        }
        result.command_completed = true;

        float* out = static_cast<float*>(out_buffer.contents);
        double checksum = 0.0;
        float max_abs_diff = 0.0f;
        for (std::uint32_t i = 0; i < count; ++i) {
            const float expected = a[i] + b[i];
            const float diff = std::fabs(out[i] - expected);
            max_abs_diff = std::max(max_abs_diff, diff);
            checksum += static_cast<double>(out[i]);
        }

        result.checksum = static_cast<float>(checksum);
        result.max_abs_diff = max_abs_diff;
        result.ok = max_abs_diff <= 0.0f;
    }

    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

GypsyDirectMetalProbeResult GypsyRunDirectMetalAttentionProbe(std::uint32_t seq_len) {
    GypsyDirectMetalProbeResult result;
    const auto start = std::chrono::steady_clock::now();

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        result.device_available = device != nil;
        if (device == nil) {
            result.error = PersistError("MTLCreateSystemDefaultDevice returned nil");
            return result;
        }

        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:AttentionKernelSource() options:nil error:&error];
        result.library_compiled = library != nil;
        if (library == nil) {
            result.error = PersistError("newLibraryWithSource failed: " + NSErrorMessage(error));
            return result;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"gypsy_attention_probe"];
        if (function == nil) {
            result.error = PersistError("newFunctionWithName gypsy_attention_probe failed");
            return result;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        result.pipeline_created = pipeline != nil;
        if (pipeline == nil) {
            result.error = PersistError("newComputePipelineStateWithFunction failed: " + NSErrorMessage(error));
            return result;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            result.error = PersistError("newCommandQueue failed");
            return result;
        }

        constexpr std::uint32_t q_heads = 32;
        constexpr std::uint32_t kv_heads = 8;
        constexpr std::uint32_t q_per_kv = 4;
        constexpr std::uint32_t head_dim = 128;
        const std::uint32_t n = std::max<std::uint32_t>(1, std::min<std::uint32_t>(seq_len, 256));
        std::vector<float> q(static_cast<size_t>(q_heads) * head_dim);
        std::vector<float> k(static_cast<size_t>(n) * kv_heads * head_dim);
        std::vector<float> v(static_cast<size_t>(n) * kv_heads * head_dim);
        std::vector<float> expected(static_cast<size_t>(q_heads) * head_dim, 0.0f);
        for (size_t i = 0; i < q.size(); ++i) {
            q[i] = (static_cast<float>((i * 17) % 101) - 50.0f) * 0.01f;
        }
        for (size_t i = 0; i < k.size(); ++i) {
            k[i] = (static_cast<float>((i * 19) % 97) - 48.0f) * 0.01f;
            v[i] = (static_cast<float>((i * 23) % 89) - 44.0f) * 0.01f;
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        std::vector<float> scores(n);
        for (std::uint32_t h = 0; h < q_heads; ++h) {
            const std::uint32_t kvh = h / q_per_kv;
            float max_score = -std::numeric_limits<float>::infinity();
            for (std::uint32_t p = 0; p < n; ++p) {
                float score = 0.0f;
                for (std::uint32_t d = 0; d < head_dim; ++d) {
                    score += q[static_cast<size_t>(h) * head_dim + d] *
                        k[(static_cast<size_t>(p) * kv_heads + kvh) * head_dim + d];
                }
                score *= scale;
                scores[p] = score;
                max_score = std::max(max_score, score);
            }
            float denom = 0.0f;
            for (std::uint32_t p = 0; p < n; ++p) {
                scores[p] = std::exp(scores[p] - max_score);
                denom += scores[p];
            }
            denom = std::max(denom, 1.0e-20f);
            for (std::uint32_t p = 0; p < n; ++p) {
                const float prob = scores[p] / denom;
                for (std::uint32_t d = 0; d < head_dim; ++d) {
                    expected[static_cast<size_t>(h) * head_dim + d] +=
                        prob * v[(static_cast<size_t>(p) * kv_heads + kvh) * head_dim + d];
                }
            }
        }

        const size_t q_bytes = q.size() * sizeof(float);
        const size_t kv_bytes = k.size() * sizeof(float);
        const size_t out_bytes = expected.size() * sizeof(float);
        id<MTLBuffer> q_buffer = [device newBufferWithBytes:q.data() length:q_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> k_buffer = [device newBufferWithBytes:k.data() length:kv_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> v_buffer = [device newBufferWithBytes:v.data() length:kv_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buffer = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> n_buffer = [device newBufferWithBytes:&n length:sizeof(n) options:MTLResourceStorageModeShared];
        if (q_buffer == nil || k_buffer == nil || v_buffer == nil || out_buffer == nil || n_buffer == nil) {
            result.error = PersistError("newBuffer allocation failed");
            return result;
        }

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            result.error = PersistError("command buffer or encoder creation failed");
            return result;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:q_buffer offset:0 atIndex:0];
        [encoder setBuffer:k_buffer offset:0 atIndex:1];
        [encoder setBuffer:v_buffer offset:0 atIndex:2];
        [encoder setBuffer:out_buffer offset:0 atIndex:3];
        [encoder setBuffer:n_buffer offset:0 atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(q_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(q_heads, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            result.error = PersistError("command buffer did not complete: " + NSErrorMessage(command_buffer.error));
            return result;
        }
        result.command_completed = true;

        float* out = static_cast<float*>(out_buffer.contents);
        double checksum = 0.0;
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < expected.size(); ++i) {
            const float diff = std::fabs(out[i] - expected[i]);
            max_abs_diff = std::max(max_abs_diff, diff);
            checksum += static_cast<double>(out[i]);
        }
        result.checksum = static_cast<float>(checksum);
        result.max_abs_diff = max_abs_diff;
        result.ok = max_abs_diff < 1.0e-5f;
    }

    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

GypsyDirectMetalProbeResult GypsyRunDirectMetalAttentionCompare(
    const float* q,
    const float* k,
    const float* v,
    const float* expected,
    std::uint32_t seq_len) {
    if (q == nullptr || k == nullptr || v == nullptr || expected == nullptr) {
        GypsyDirectMetalProbeResult result;
        result.error = PersistError("GypsyRunDirectMetalAttentionCompare received null input");
        return result;
    }
    return RunDirectAttention(q, k, v, expected, nullptr, seq_len);
}

GypsyDirectMetalProbeResult GypsyRunDirectMetalAttentionToHost(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    std::uint32_t seq_len) {
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr) {
        GypsyDirectMetalProbeResult result;
        result.error = PersistError("GypsyRunDirectMetalAttentionToHost received null input");
        return result;
    }
    return RunDirectAttention(q, k, v, nullptr, out, seq_len);
}

void GypsyResetDirectMetalAttentionKvCache() {
    @autoreleasepool {
        DirectMetalAttentionContext& context = AttentionContext();
        for (auto& layer : context.layer_kv) {
            layer.k_buffer = nil;
            layer.v_buffer = nil;
            layer.capacity_bytes = 0;
        }
    }
}

GypsyDirectMetalProbeResult GypsyRunDirectMetalAttentionAppendToHost(
    std::uint32_t layer,
    const float* q,
    const float* current_k,
    const float* current_v,
    float* out,
    std::uint32_t seq_len) {
    if (q == nullptr || current_k == nullptr || current_v == nullptr || out == nullptr) {
        GypsyDirectMetalProbeResult result;
        result.error = PersistError("GypsyRunDirectMetalAttentionAppendToHost received null input");
        return result;
    }
    return RunDirectAttentionWithResidentLayerKv(layer, q, current_k, current_v, out, seq_len);
}

GypsyDirectMetalProbeResult GypsyRunDirectMetalAttentionOProjectionAppendToHost(
    std::uint32_t layer,
    const float* q,
    const float* current_k,
    const float* current_v,
    const std::uint32_t* o_weight,
    const std::uint16_t* o_scales,
    const std::uint16_t* o_biases,
    float* out,
    std::uint32_t seq_len) {
    if (q == nullptr || current_k == nullptr || current_v == nullptr ||
        o_weight == nullptr || o_scales == nullptr || o_biases == nullptr || out == nullptr) {
        GypsyDirectMetalProbeResult result;
        result.error = PersistError("GypsyRunDirectMetalAttentionOProjectionAppendToHost received null input");
        return result;
    }
    return RunDirectAttentionOProjectionWithResidentLayerKv(layer, q, current_k, current_v, o_weight, o_scales, o_biases, out, seq_len);
}

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
    std::uint32_t seq_len) {
    if (q == nullptr || current_k == nullptr || current_v == nullptr ||
        o_weight == nullptr || o_scales == nullptr || o_biases == nullptr ||
        residual == nullptr || out == nullptr) {
        GypsyDirectMetalProbeResult result;
        result.error = PersistError("GypsyRunDirectMetalAttentionOResidualAppendToHost received null input");
        return result;
    }
    GypsyDirectMetalProbeResult result;
    const auto start = std::chrono::steady_clock::now();

    @autoreleasepool {
        DirectMetalAttentionContext& context = AttentionContext();
        result.device_available = context.device != nil;
        result.library_compiled = context.library != nil;
        result.pipeline_created = context.pipeline != nil &&
            context.o_projection_pipeline != nil &&
            context.residual_add_pipeline != nil;
        if (!context.error.empty()) {
            result.error = PersistError(context.error);
            return result;
        }
        if (layer >= context.layer_kv.size()) {
            result.error = PersistError("direct Metal attention+o+residual layer index out of range");
            return result;
        }

        constexpr std::uint32_t q_heads = 32;
        constexpr std::uint32_t kv_heads = 8;
        constexpr std::uint32_t head_dim = 128;
        constexpr std::uint32_t hidden_size = 2560;
        constexpr std::uint32_t o_rows = 2560;
        constexpr std::uint32_t o_packed_cols = 512;
        constexpr std::uint32_t o_groups = 64;
        const std::uint32_t n = std::max<std::uint32_t>(1, std::min<std::uint32_t>(seq_len, 256));
        const size_t q_bytes = static_cast<size_t>(q_heads) * head_dim * sizeof(float);
        const size_t current_kv_bytes = static_cast<size_t>(kv_heads) * head_dim * sizeof(float);
        const size_t kv_capacity_bytes = static_cast<size_t>(256) * kv_heads * head_dim * sizeof(float);
        const size_t attention_out_bytes = static_cast<size_t>(q_heads) * head_dim * sizeof(float);
        const size_t o_weight_bytes = static_cast<size_t>(o_rows) * o_packed_cols * sizeof(std::uint32_t);
        const size_t o_scale_bytes = static_cast<size_t>(o_rows) * o_groups * sizeof(std::uint16_t);
        const size_t hidden_bytes = static_cast<size_t>(hidden_size) * sizeof(float);

        if (context.q_buffer == nil || context.q_capacity_bytes < q_bytes) {
            context.q_buffer = [context.device newBufferWithLength:q_bytes options:MTLResourceStorageModeShared];
            context.q_capacity_bytes = q_bytes;
        }
        DirectMetalAttentionContext::LayerKvBuffers& layer_buffers = context.layer_kv[layer];
        if (layer_buffers.k_buffer == nil || layer_buffers.capacity_bytes < kv_capacity_bytes) {
            layer_buffers.k_buffer = [context.device newBufferWithLength:kv_capacity_bytes options:MTLResourceStorageModeShared];
            layer_buffers.v_buffer = [context.device newBufferWithLength:kv_capacity_bytes options:MTLResourceStorageModeShared];
            layer_buffers.capacity_bytes = kv_capacity_bytes;
        }
        if (context.out_buffer == nil || context.out_capacity_bytes < attention_out_bytes) {
            context.out_buffer = [context.device newBufferWithLength:attention_out_bytes options:MTLResourceStorageModeShared];
            context.out_capacity_bytes = attention_out_bytes;
        }
        if (context.o_out_buffer == nil || context.o_out_capacity_bytes < hidden_bytes) {
            context.o_out_buffer = [context.device newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared];
            context.o_out_capacity_bytes = hidden_bytes;
        }
        if (context.residual_buffer == nil || context.residual_capacity_bytes < hidden_bytes) {
            context.residual_buffer = [context.device newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared];
            context.residual_capacity_bytes = hidden_bytes;
        }
        if (context.n_buffer == nil) {
            context.n_buffer = [context.device newBufferWithLength:sizeof(n) options:MTLResourceStorageModeShared];
        }
        if (context.o_weight_buffer == nil || context.o_weight_capacity_bytes < o_weight_bytes) {
            context.o_weight_buffer = [context.device newBufferWithLength:o_weight_bytes options:MTLResourceStorageModeShared];
            context.o_weight_capacity_bytes = o_weight_bytes;
        }
        if (context.o_scales_buffer == nil || context.o_scale_capacity_bytes < o_scale_bytes) {
            context.o_scales_buffer = [context.device newBufferWithLength:o_scale_bytes options:MTLResourceStorageModeShared];
            context.o_biases_buffer = [context.device newBufferWithLength:o_scale_bytes options:MTLResourceStorageModeShared];
            context.o_scale_capacity_bytes = o_scale_bytes;
        }
        if (context.q_buffer == nil || layer_buffers.k_buffer == nil || layer_buffers.v_buffer == nil ||
            context.out_buffer == nil || context.o_out_buffer == nil || context.residual_buffer == nil ||
            context.n_buffer == nil || context.o_weight_buffer == nil ||
            context.o_scales_buffer == nil || context.o_biases_buffer == nil) {
            result.error = PersistError("newBuffer allocation failed for attention+o+residual");
            return result;
        }

        std::memcpy(context.o_weight_buffer.contents, o_weight, o_weight_bytes);
        std::memcpy(context.o_scales_buffer.contents, o_scales, o_scale_bytes);
        std::memcpy(context.o_biases_buffer.contents, o_biases, o_scale_bytes);
        std::memcpy(context.q_buffer.contents, q, q_bytes);
        std::memcpy(context.residual_buffer.contents, residual, hidden_bytes);
        const size_t write_offset = static_cast<size_t>(n - 1) * current_kv_bytes;
        std::memcpy(static_cast<char*>(layer_buffers.k_buffer.contents) + write_offset, current_k, current_kv_bytes);
        std::memcpy(static_cast<char*>(layer_buffers.v_buffer.contents) + write_offset, current_v, current_kv_bytes);
        std::memcpy(context.n_buffer.contents, &n, sizeof(n));

        id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            result.error = PersistError("command buffer or encoder creation failed");
            return result;
        }
        [encoder setComputePipelineState:context.pipeline];
        [encoder setBuffer:context.q_buffer offset:0 atIndex:0];
        [encoder setBuffer:layer_buffers.k_buffer offset:0 atIndex:1];
        [encoder setBuffer:layer_buffers.v_buffer offset:0 atIndex:2];
        [encoder setBuffer:context.out_buffer offset:0 atIndex:3];
        [encoder setBuffer:context.n_buffer offset:0 atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(q_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(q_heads, 1, 1)];

        [encoder setComputePipelineState:context.o_projection_pipeline];
        [encoder setBuffer:context.out_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.o_weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.o_scales_buffer offset:0 atIndex:2];
        [encoder setBuffer:context.o_biases_buffer offset:0 atIndex:3];
        [encoder setBuffer:context.o_out_buffer offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(o_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        [encoder setComputePipelineState:context.residual_add_pipeline];
        [encoder setBuffer:context.residual_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.o_out_buffer offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(hidden_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            result.error = PersistError("command buffer did not complete: " + NSErrorMessage(command_buffer.error));
            return result;
        }
        result.command_completed = true;

        float* residual_out = static_cast<float*>(context.o_out_buffer.contents);
        double checksum = 0.0;
        for (size_t i = 0; i < hidden_size; ++i) {
            out[i] = residual_out[i];
            checksum += static_cast<double>(residual_out[i]);
        }
        result.checksum = static_cast<float>(checksum);
        result.max_abs_diff = 0.0f;
        result.ok = true;
    }

    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

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
    std::uint32_t position) {
    if (input == nullptr ||
        q_weight == nullptr || q_scales == nullptr || q_biases == nullptr ||
        k_weight == nullptr || k_scales == nullptr || k_biases == nullptr ||
        v_weight == nullptr || v_scales == nullptr || v_biases == nullptr ||
        q_norm_weight == nullptr || k_norm_weight == nullptr ||
        o_weight == nullptr || o_scales == nullptr || o_biases == nullptr ||
        out == nullptr) {
        GypsyDirectMetalProbeResult result;
        result.error = PersistError("GypsyRunDirectMetalQkvAttentionOAppendToHost received null input");
        return result;
    }
    GypsyDirectMetalProbeResult result;
    const auto start = std::chrono::steady_clock::now();

    @autoreleasepool {
        DirectMetalAttentionContext& context = AttentionContext();
        result.device_available = context.device != nil;
        result.library_compiled = context.library != nil;
        result.pipeline_created = context.qkv_projection_pipeline != nil &&
            context.qk_norm_rope_pipeline != nil &&
            context.pipeline != nil &&
            context.o_projection_pipeline != nil;
        if (!context.error.empty()) {
            result.error = PersistError(context.error);
            return result;
        }
        if (layer >= context.layer_kv.size()) {
            result.error = PersistError("direct Metal qkv+attention+o layer index out of range");
            return result;
        }

        constexpr std::uint32_t hidden_size = 2560;
        constexpr std::uint32_t q_rows = 4096;
        constexpr std::uint32_t kv_rows = 1024;
        constexpr std::uint32_t q_heads = 32;
        constexpr std::uint32_t kv_heads = 8;
        constexpr std::uint32_t head_dim = 128;
        constexpr std::uint32_t in_packed_cols = 320;
        constexpr std::uint32_t in_groups = 40;
        constexpr std::uint32_t o_rows = 2560;
        constexpr std::uint32_t o_packed_cols = 512;
        constexpr std::uint32_t o_groups = 64;
        const std::uint32_t n = std::max<std::uint32_t>(1, std::min<std::uint32_t>(seq_len, 256));
        const size_t hidden_bytes = static_cast<size_t>(hidden_size) * sizeof(float);
        const size_t q_bytes = static_cast<size_t>(q_rows) * sizeof(float);
        const size_t kv_bytes = static_cast<size_t>(kv_rows) * sizeof(float);
        const size_t current_kv_bytes = static_cast<size_t>(kv_heads) * head_dim * sizeof(float);
        const size_t kv_capacity_bytes = static_cast<size_t>(256) * kv_heads * head_dim * sizeof(float);
        const size_t q_weight_bytes = static_cast<size_t>(q_rows) * in_packed_cols * sizeof(std::uint32_t);
        const size_t kv_weight_bytes = static_cast<size_t>(kv_rows) * in_packed_cols * sizeof(std::uint32_t);
        const size_t q_scale_bytes = static_cast<size_t>(q_rows) * in_groups * sizeof(std::uint16_t);
        const size_t kv_scale_bytes = static_cast<size_t>(kv_rows) * in_groups * sizeof(std::uint16_t);
        const size_t o_weight_bytes = static_cast<size_t>(o_rows) * o_packed_cols * sizeof(std::uint32_t);
        const size_t o_scale_bytes = static_cast<size_t>(o_rows) * o_groups * sizeof(std::uint16_t);
        const size_t norm_weight_bytes = static_cast<size_t>(head_dim) * sizeof(std::uint16_t);

        auto ensure_buffer = [&](id<MTLBuffer>& buffer, size_t& capacity, size_t bytes) {
            if (buffer == nil || capacity < bytes) {
                buffer = [context.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                capacity = bytes;
            }
        };

        ensure_buffer(context.hidden_input_buffer, context.hidden_input_capacity_bytes, hidden_bytes);
        ensure_buffer(context.q_raw_buffer, context.q_raw_capacity_bytes, q_bytes);
        ensure_buffer(context.k_raw_buffer, context.k_raw_capacity_bytes, kv_bytes);
        ensure_buffer(context.v_raw_buffer, context.v_raw_capacity_bytes, kv_bytes);
        ensure_buffer(context.q_rope_buffer, context.q_rope_capacity_bytes, q_bytes);
        ensure_buffer(context.k_rope_buffer, context.k_rope_capacity_bytes, kv_bytes);
        ensure_buffer(context.q_norm_weight_buffer, context.norm_weight_capacity_bytes, norm_weight_bytes);
        if (context.k_norm_weight_buffer == nil) {
            context.k_norm_weight_buffer = [context.device newBufferWithLength:norm_weight_bytes options:MTLResourceStorageModeShared];
        }
        DirectMetalAttentionContext::LayerKvBuffers& layer_buffers = context.layer_kv[layer];
        if (layer_buffers.k_buffer == nil || layer_buffers.capacity_bytes < kv_capacity_bytes) {
            layer_buffers.k_buffer = [context.device newBufferWithLength:kv_capacity_bytes options:MTLResourceStorageModeShared];
            layer_buffers.v_buffer = [context.device newBufferWithLength:kv_capacity_bytes options:MTLResourceStorageModeShared];
            layer_buffers.capacity_bytes = kv_capacity_bytes;
        }
        ensure_buffer(context.out_buffer, context.out_capacity_bytes, q_bytes);
        ensure_buffer(context.o_out_buffer, context.o_out_capacity_bytes, hidden_bytes);
        if (context.o_weight_buffer == nil || context.o_weight_capacity_bytes < o_weight_bytes) {
            context.o_weight_buffer = [context.device newBufferWithLength:o_weight_bytes options:MTLResourceStorageModeShared];
            context.o_weight_capacity_bytes = o_weight_bytes;
        }
        if (context.o_scales_buffer == nil || context.o_scale_capacity_bytes < o_scale_bytes) {
            context.o_scales_buffer = [context.device newBufferWithLength:o_scale_bytes options:MTLResourceStorageModeShared];
            context.o_biases_buffer = [context.device newBufferWithLength:o_scale_bytes options:MTLResourceStorageModeShared];
            context.o_scale_capacity_bytes = o_scale_bytes;
        }
        if (context.rows_buffer == nil) {
            context.rows_buffer = [context.device newBufferWithLength:sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
        }
        if (context.n_buffer == nil) {
            context.n_buffer = [context.device newBufferWithLength:sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
        }
        if (context.position_buffer == nil) {
            context.position_buffer = [context.device newBufferWithLength:sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
        }
        if (context.hidden_input_buffer == nil || context.q_raw_buffer == nil ||
            context.k_raw_buffer == nil || context.v_raw_buffer == nil ||
            context.q_rope_buffer == nil || context.k_rope_buffer == nil ||
            context.q_norm_weight_buffer == nil || context.k_norm_weight_buffer == nil ||
            layer_buffers.k_buffer == nil || layer_buffers.v_buffer == nil ||
            context.out_buffer == nil || context.o_out_buffer == nil ||
            context.o_weight_buffer == nil || context.o_scales_buffer == nil ||
            context.o_biases_buffer == nil || context.rows_buffer == nil ||
            context.n_buffer == nil || context.position_buffer == nil) {
            result.error = PersistError("newBuffer allocation failed for qkv+attention+o");
            return result;
        }

        std::memcpy(context.hidden_input_buffer.contents, input, hidden_bytes);
        std::memcpy(context.q_norm_weight_buffer.contents, q_norm_weight, norm_weight_bytes);
        std::memcpy(context.k_norm_weight_buffer.contents, k_norm_weight, norm_weight_bytes);
        std::memcpy(context.o_weight_buffer.contents, o_weight, o_weight_bytes);
        std::memcpy(context.o_scales_buffer.contents, o_scales, o_scale_bytes);
        std::memcpy(context.o_biases_buffer.contents, o_biases, o_scale_bytes);
        std::memcpy(context.n_buffer.contents, &n, sizeof(n));
        std::memcpy(context.position_buffer.contents, &position, sizeof(position));

        auto copy_if_changed = [&](id<MTLBuffer> buffer, const void*& source, const void* data, size_t bytes) {
            if (source != data) {
                std::memcpy(buffer.contents, data, bytes);
                source = data;
            }
        };
        auto ensure_layer_buffer = [&](id<MTLBuffer>& buffer, size_t& capacity, size_t bytes) {
            if (buffer == nil || capacity < bytes) {
                buffer = [context.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                capacity = bytes;
            }
        };
        ensure_layer_buffer(layer_buffers.q_weight_buffer, layer_buffers.q_weight_capacity_bytes, q_weight_bytes);
        ensure_layer_buffer(layer_buffers.q_scales_buffer, layer_buffers.q_scale_capacity_bytes, q_scale_bytes);
        ensure_layer_buffer(layer_buffers.q_biases_buffer, layer_buffers.q_scale_capacity_bytes, q_scale_bytes);
        ensure_layer_buffer(layer_buffers.k_weight_buffer, layer_buffers.kv_weight_capacity_bytes, kv_weight_bytes);
        ensure_layer_buffer(layer_buffers.k_scales_buffer, layer_buffers.kv_scale_capacity_bytes, kv_scale_bytes);
        ensure_layer_buffer(layer_buffers.k_biases_buffer, layer_buffers.kv_scale_capacity_bytes, kv_scale_bytes);
        ensure_layer_buffer(layer_buffers.v_weight_buffer, layer_buffers.kv_weight_capacity_bytes, kv_weight_bytes);
        ensure_layer_buffer(layer_buffers.v_scales_buffer, layer_buffers.kv_scale_capacity_bytes, kv_scale_bytes);
        ensure_layer_buffer(layer_buffers.v_biases_buffer, layer_buffers.kv_scale_capacity_bytes, kv_scale_bytes);
        if (layer_buffers.q_weight_buffer != nil) copy_if_changed(layer_buffers.q_weight_buffer, layer_buffers.q_weight_source, q_weight, q_weight_bytes);
        if (layer_buffers.q_scales_buffer != nil) copy_if_changed(layer_buffers.q_scales_buffer, layer_buffers.q_scales_source, q_scales, q_scale_bytes);
        if (layer_buffers.q_biases_buffer != nil) copy_if_changed(layer_buffers.q_biases_buffer, layer_buffers.q_biases_source, q_biases, q_scale_bytes);
        if (layer_buffers.k_weight_buffer != nil) copy_if_changed(layer_buffers.k_weight_buffer, layer_buffers.k_weight_source, k_weight, kv_weight_bytes);
        if (layer_buffers.k_scales_buffer != nil) copy_if_changed(layer_buffers.k_scales_buffer, layer_buffers.k_scales_source, k_scales, kv_scale_bytes);
        if (layer_buffers.k_biases_buffer != nil) copy_if_changed(layer_buffers.k_biases_buffer, layer_buffers.k_biases_source, k_biases, kv_scale_bytes);
        if (layer_buffers.v_weight_buffer != nil) copy_if_changed(layer_buffers.v_weight_buffer, layer_buffers.v_weight_source, v_weight, kv_weight_bytes);
        if (layer_buffers.v_scales_buffer != nil) copy_if_changed(layer_buffers.v_scales_buffer, layer_buffers.v_scales_source, v_scales, kv_scale_bytes);
        if (layer_buffers.v_biases_buffer != nil) copy_if_changed(layer_buffers.v_biases_buffer, layer_buffers.v_biases_source, v_biases, kv_scale_bytes);
        id<MTLBuffer> q_rows_buffer = [context.device newBufferWithBytes:&q_rows length:sizeof(q_rows) options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_rows_buffer = [context.device newBufferWithBytes:&kv_rows length:sizeof(kv_rows) options:MTLResourceStorageModeShared];
        if (layer_buffers.q_weight_buffer == nil || layer_buffers.q_scales_buffer == nil || layer_buffers.q_biases_buffer == nil ||
            layer_buffers.k_weight_buffer == nil || layer_buffers.k_scales_buffer == nil || layer_buffers.k_biases_buffer == nil ||
            layer_buffers.v_weight_buffer == nil || layer_buffers.v_scales_buffer == nil || layer_buffers.v_biases_buffer == nil ||
            q_rows_buffer == nil || kv_rows_buffer == nil) {
            result.error = PersistError("newBuffer allocation failed for qkv weights");
            return result;
        }

        id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            result.error = PersistError("command buffer or encoder creation failed");
            return result;
        }
        auto dispatch_projection = [&](id<MTLBuffer> weight,
                                       id<MTLBuffer> scales,
                                       id<MTLBuffer> biases,
                                       id<MTLBuffer> output,
                                       id<MTLBuffer> rows_buffer,
                                       std::uint32_t rows) {
            [encoder setComputePipelineState:context.qkv_projection_pipeline];
            [encoder setBuffer:context.hidden_input_buffer offset:0 atIndex:0];
            [encoder setBuffer:weight offset:0 atIndex:1];
            [encoder setBuffer:scales offset:0 atIndex:2];
            [encoder setBuffer:biases offset:0 atIndex:3];
            [encoder setBuffer:output offset:0 atIndex:4];
            [encoder setBuffer:rows_buffer offset:0 atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        };
        dispatch_projection(layer_buffers.q_weight_buffer, layer_buffers.q_scales_buffer, layer_buffers.q_biases_buffer, context.q_raw_buffer, q_rows_buffer, q_rows);
        dispatch_projection(layer_buffers.k_weight_buffer, layer_buffers.k_scales_buffer, layer_buffers.k_biases_buffer, context.k_raw_buffer, kv_rows_buffer, kv_rows);
        dispatch_projection(layer_buffers.v_weight_buffer, layer_buffers.v_scales_buffer, layer_buffers.v_biases_buffer, context.v_raw_buffer, kv_rows_buffer, kv_rows);

        [encoder setComputePipelineState:context.qk_norm_rope_pipeline];
        [encoder setBuffer:context.q_raw_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.k_raw_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.q_norm_weight_buffer offset:0 atIndex:2];
        [encoder setBuffer:context.k_norm_weight_buffer offset:0 atIndex:3];
        [encoder setBuffer:context.q_rope_buffer offset:0 atIndex:4];
        [encoder setBuffer:context.k_rope_buffer offset:0 atIndex:5];
        [encoder setBuffer:context.position_buffer offset:0 atIndex:6];
        [encoder dispatchThreads:MTLSizeMake(q_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(q_heads, 1, 1)];

        const size_t write_offset = static_cast<size_t>(n - 1) * current_kv_bytes;
        [encoder endEncoding];

        id<MTLBlitCommandEncoder> copy_encoder = [command_buffer blitCommandEncoder];
        [copy_encoder copyFromBuffer:context.k_rope_buffer sourceOffset:0 toBuffer:layer_buffers.k_buffer destinationOffset:write_offset size:current_kv_bytes];
        [copy_encoder copyFromBuffer:context.v_raw_buffer sourceOffset:0 toBuffer:layer_buffers.v_buffer destinationOffset:write_offset size:current_kv_bytes];
        [copy_encoder endEncoding];

        encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:context.pipeline];
        [encoder setBuffer:context.q_rope_buffer offset:0 atIndex:0];
        [encoder setBuffer:layer_buffers.k_buffer offset:0 atIndex:1];
        [encoder setBuffer:layer_buffers.v_buffer offset:0 atIndex:2];
        [encoder setBuffer:context.out_buffer offset:0 atIndex:3];
        [encoder setBuffer:context.n_buffer offset:0 atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(q_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(q_heads, 1, 1)];

        [encoder setComputePipelineState:context.o_projection_pipeline];
        [encoder setBuffer:context.out_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.o_weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.o_scales_buffer offset:0 atIndex:2];
        [encoder setBuffer:context.o_biases_buffer offset:0 atIndex:3];
        [encoder setBuffer:context.o_out_buffer offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(o_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            result.error = PersistError("command buffer did not complete: " + NSErrorMessage(command_buffer.error));
            return result;
        }
        result.command_completed = true;

        float* q_debug = static_cast<float*>(context.q_rope_buffer.contents);
        float* k_debug = static_cast<float*>(context.k_rope_buffer.contents);
        float* v_debug = static_cast<float*>(context.v_raw_buffer.contents);
        float* attention_debug = static_cast<float*>(context.out_buffer.contents);
        double q_checksum = 0.0;
        double k_checksum = 0.0;
        double v_checksum = 0.0;
        double attention_checksum = 0.0;
        for (size_t i = 0; i < q_bytes / sizeof(float); ++i) q_checksum += static_cast<double>(q_debug[i]);
        for (size_t i = 0; i < kv_bytes / sizeof(float); ++i) {
            k_checksum += static_cast<double>(k_debug[i]);
            v_checksum += static_cast<double>(v_debug[i]);
        }
        for (size_t i = 0; i < q_bytes / sizeof(float); ++i) attention_checksum += static_cast<double>(attention_debug[i]);
        result.q_checksum = static_cast<float>(q_checksum);
        result.k_checksum = static_cast<float>(k_checksum);
        result.v_checksum = static_cast<float>(v_checksum);
        result.attention_checksum = static_cast<float>(attention_checksum);

        float* o_out = static_cast<float*>(context.o_out_buffer.contents);
        double checksum = 0.0;
        for (size_t i = 0; i < hidden_size; ++i) {
            out[i] = o_out[i];
            checksum += static_cast<double>(o_out[i]);
        }
        result.checksum = static_cast<float>(checksum);
        result.max_abs_diff = 0.0f;
        result.ok = true;
    }

    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

GypsyDirectMetalProbeResult GypsyRunDirectMetalQuantizedProjectionCompare(
    const float* input,
    const std::uint32_t* weight,
    const std::uint16_t* scales,
    const std::uint16_t* biases,
    const float* expected,
    float* out,
    std::uint32_t rows,
    std::uint32_t packed_cols,
    std::uint32_t input_len) {
    if (input == nullptr || weight == nullptr || scales == nullptr || biases == nullptr ||
        expected == nullptr || out == nullptr) {
        GypsyDirectMetalProbeResult result;
        result.error = PersistError("GypsyRunDirectMetalQuantizedProjectionCompare received null input");
        return result;
    }
    if (input_len % 64 != 0 || packed_cols != input_len / 8) {
        GypsyDirectMetalProbeResult result;
        result.error = PersistError("quantized projection shape guard failed: input_len must be divisible by 64 and packed_cols must equal input_len / 8");
        return result;
    }
    return RunDirectQuantizedProjection(input, weight, scales, biases, expected, out, rows, packed_cols, input_len);
}

GypsyDirectMetalProbeResult GypsyRunDirectMetalQuantizedProjectionTiledCompare(
    const float* input,
    const std::uint32_t* weight,
    const std::uint16_t* scales,
    const std::uint16_t* biases,
    const float* expected,
    float* out,
    std::uint32_t rows,
    std::uint32_t packed_cols,
    std::uint32_t input_len) {
    if (input == nullptr || weight == nullptr || scales == nullptr || biases == nullptr ||
        expected == nullptr || out == nullptr) {
        GypsyDirectMetalProbeResult result;
        result.error = PersistError("GypsyRunDirectMetalQuantizedProjectionTiledCompare received null input");
        return result;
    }
    if (input_len % 64 != 0 || packed_cols != input_len / 8) {
        GypsyDirectMetalProbeResult result;
        result.error = PersistError("tiled quantized projection shape guard failed: input_len must be divisible by 64 and packed_cols must equal input_len / 8");
        return result;
    }
    return RunDirectQuantizedProjectionTiled(input, weight, scales, biases, expected, out, rows, packed_cols, input_len);
}

GypsyDirectMetalProbeResult GypsyRunDirectMetalQuantizedProjectionTiledToHost(
    const float* input,
    const std::uint32_t* weight,
    const std::uint16_t* scales,
    const std::uint16_t* biases,
    float* out,
    std::uint32_t rows,
    std::uint32_t packed_cols,
    std::uint32_t input_len) {
    if (input == nullptr || weight == nullptr || scales == nullptr || biases == nullptr || out == nullptr) {
        GypsyDirectMetalProbeResult result;
        result.error = PersistError("GypsyRunDirectMetalQuantizedProjectionTiledToHost received null input");
        return result;
    }
    if (input_len % 64 != 0 || packed_cols != input_len / 8) {
        GypsyDirectMetalProbeResult result;
        result.error = PersistError("tiled quantized projection shape guard failed: input_len must be divisible by 64 and packed_cols must equal input_len / 8");
        return result;
    }
    return RunDirectQuantizedProjectionTiled(input, weight, scales, biases, nullptr, out, rows, packed_cols, input_len);
}

GypsyDirectMetalProbeResult GypsyRunDirectMetalQuantizedProjectionTiledTop1(
    const float* input,
    const std::uint32_t* weight,
    const std::uint16_t* scales,
    const std::uint16_t* biases,
    std::uint32_t rows,
    std::uint32_t packed_cols,
    std::uint32_t input_len) {
    if (input == nullptr || weight == nullptr || scales == nullptr || biases == nullptr) {
        GypsyDirectMetalProbeResult result;
        result.error = PersistError("GypsyRunDirectMetalQuantizedProjectionTiledTop1 received null input");
        return result;
    }
    if (input_len % 64 != 0 || packed_cols != input_len / 8) {
        GypsyDirectMetalProbeResult result;
        result.error = PersistError("tiled top1 projection shape guard failed: input_len must be divisible by 64 and packed_cols must equal input_len / 8");
        return result;
    }
    return RunDirectQuantizedProjectionTiled(
        input,
        weight,
        scales,
        biases,
        nullptr,
        nullptr,
        rows,
        packed_cols,
        input_len,
        true);
}

GypsyDirectMetalProbeResult GypsyRunDirectMetalRmsNormToHost(
    const float* input,
    const std::uint16_t* weight,
    float* out,
    std::uint32_t input_len) {
    GypsyDirectMetalProbeResult result;
    const auto start = std::chrono::steady_clock::now();
    if (input == nullptr || weight == nullptr || out == nullptr) {
        result.error = PersistError("GypsyRunDirectMetalRmsNormToHost received null input");
        return result;
    }
    if (input_len != 2560) {
        result.error = PersistError("GypsyRunDirectMetalRmsNormToHost supports only fixed input_len 2560");
        return result;
    }
    @autoreleasepool {
        DirectMetalAttentionContext& context = AttentionContext();
        result.device_available = context.device != nil;
        result.library_compiled = context.library != nil;
        result.pipeline_created = context.rmsnorm_pipeline != nil;
        if (!context.error.empty()) {
            result.error = PersistError(context.error);
            return result;
        }
        const size_t bytes = static_cast<size_t>(input_len) * sizeof(float);
        const size_t weight_bytes = static_cast<size_t>(input_len) * sizeof(std::uint16_t);
        if (context.hidden_input_buffer == nil || context.hidden_input_capacity_bytes < bytes) {
            context.hidden_input_buffer = [context.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            context.hidden_input_capacity_bytes = bytes;
        }
        if (context.rmsnorm_weight_buffer == nil || context.rmsnorm_weight_capacity_bytes < weight_bytes) {
            context.rmsnorm_weight_buffer = [context.device newBufferWithLength:weight_bytes options:MTLResourceStorageModeShared];
            context.rmsnorm_weight_capacity_bytes = weight_bytes;
        }
        if (context.rmsnorm_out_buffer == nil || context.rmsnorm_out_capacity_bytes < bytes) {
            context.rmsnorm_out_buffer = [context.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            context.rmsnorm_out_capacity_bytes = bytes;
        }
        if (context.hidden_input_buffer == nil || context.rmsnorm_weight_buffer == nil || context.rmsnorm_out_buffer == nil) {
            result.error = PersistError("newBuffer allocation failed for rmsnorm");
            return result;
        }
        std::memcpy(context.hidden_input_buffer.contents, input, bytes);
        std::memcpy(context.rmsnorm_weight_buffer.contents, weight, weight_bytes);
        id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            result.error = PersistError("command buffer or encoder creation failed");
            return result;
        }
        [encoder setComputePipelineState:context.rmsnorm_pipeline];
        [encoder setBuffer:context.hidden_input_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.rmsnorm_weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.rmsnorm_out_buffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            result.error = PersistError("command buffer did not complete: " + NSErrorMessage(command_buffer.error));
            return result;
        }
        result.command_completed = true;
        const float* values = static_cast<const float*>(context.rmsnorm_out_buffer.contents);
        double checksum = 0.0;
        for (std::uint32_t i = 0; i < input_len; ++i) {
            out[i] = values[i];
            checksum += static_cast<double>(values[i]);
        }
        result.checksum = static_cast<float>(checksum);
        result.ok = true;
    }
    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

GypsyDirectMetalProbeResult GypsyRunDirectMetalSiluMultiplyToHost(
    const float* gate,
    const float* up,
    float* out,
    std::uint32_t input_len) {
    GypsyDirectMetalProbeResult result;
    const auto start = std::chrono::steady_clock::now();
    if (gate == nullptr || up == nullptr || out == nullptr) {
        result.error = PersistError("GypsyRunDirectMetalSiluMultiplyToHost received null input");
        return result;
    }
    @autoreleasepool {
        DirectMetalAttentionContext& context = AttentionContext();
        result.device_available = context.device != nil;
        result.library_compiled = context.library != nil;
        result.pipeline_created = context.silu_multiply_pipeline != nil;
        if (!context.error.empty()) {
            result.error = PersistError(context.error);
            return result;
        }
        const size_t bytes = static_cast<size_t>(input_len) * sizeof(float);
        if (context.gate_buffer == nil || context.gate_capacity_bytes < bytes) {
            context.gate_buffer = [context.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            context.gate_capacity_bytes = bytes;
        }
        if (context.up_buffer == nil || context.up_capacity_bytes < bytes) {
            context.up_buffer = [context.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            context.up_capacity_bytes = bytes;
        }
        if (context.activation_out_buffer == nil || context.activation_out_capacity_bytes < bytes) {
            context.activation_out_buffer = [context.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            context.activation_out_capacity_bytes = bytes;
        }
        if (context.n_buffer == nil) {
            context.n_buffer = [context.device newBufferWithLength:sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
        }
        if (context.gate_buffer == nil || context.up_buffer == nil || context.activation_out_buffer == nil || context.n_buffer == nil) {
            result.error = PersistError("newBuffer allocation failed for silu multiply");
            return result;
        }
        std::memcpy(context.gate_buffer.contents, gate, bytes);
        std::memcpy(context.up_buffer.contents, up, bytes);
        std::memcpy(context.n_buffer.contents, &input_len, sizeof(std::uint32_t));
        id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            result.error = PersistError("command buffer or encoder creation failed");
            return result;
        }
        [encoder setComputePipelineState:context.silu_multiply_pipeline];
        [encoder setBuffer:context.gate_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.up_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.activation_out_buffer offset:0 atIndex:2];
        [encoder setBuffer:context.n_buffer offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(input_len, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            result.error = PersistError("command buffer did not complete: " + NSErrorMessage(command_buffer.error));
            return result;
        }
        result.command_completed = true;
        const float* values = static_cast<const float*>(context.activation_out_buffer.contents);
        double checksum = 0.0;
        for (std::uint32_t i = 0; i < input_len; ++i) {
            out[i] = values[i];
            checksum += static_cast<double>(values[i]);
        }
        result.checksum = static_cast<float>(checksum);
        result.ok = true;
    }
    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

GypsyDirectMetalProbeResult GypsyRunDirectMetalAddResidualToHost(
    const float* residual,
    const float* values,
    float* out,
    std::uint32_t input_len) {
    GypsyDirectMetalProbeResult result;
    const auto start = std::chrono::steady_clock::now();
    if (residual == nullptr || values == nullptr || out == nullptr) {
        result.error = PersistError("GypsyRunDirectMetalAddResidualToHost received null input");
        return result;
    }
    if (input_len != 2560) {
        result.error = PersistError("GypsyRunDirectMetalAddResidualToHost supports only fixed input_len 2560");
        return result;
    }
    @autoreleasepool {
        DirectMetalAttentionContext& context = AttentionContext();
        result.device_available = context.device != nil;
        result.library_compiled = context.library != nil;
        result.pipeline_created = context.residual_add_pipeline != nil;
        if (!context.error.empty()) {
            result.error = PersistError(context.error);
            return result;
        }
        const size_t bytes = static_cast<size_t>(input_len) * sizeof(float);
        if (context.residual_buffer == nil || context.residual_capacity_bytes < bytes) {
            context.residual_buffer = [context.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            context.residual_capacity_bytes = bytes;
        }
        if (context.o_out_buffer == nil || context.o_out_capacity_bytes < bytes) {
            context.o_out_buffer = [context.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            context.o_out_capacity_bytes = bytes;
        }
        if (context.residual_buffer == nil || context.o_out_buffer == nil) {
            result.error = PersistError("newBuffer allocation failed for residual add");
            return result;
        }
        std::memcpy(context.residual_buffer.contents, residual, bytes);
        std::memcpy(context.o_out_buffer.contents, values, bytes);
        id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            result.error = PersistError("command buffer or encoder creation failed");
            return result;
        }
        [encoder setComputePipelineState:context.residual_add_pipeline];
        [encoder setBuffer:context.residual_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.o_out_buffer offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(input_len, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            result.error = PersistError("command buffer did not complete: " + NSErrorMessage(command_buffer.error));
            return result;
        }
        result.command_completed = true;
        const float* output_values = static_cast<const float*>(context.o_out_buffer.contents);
        double checksum = 0.0;
        for (std::uint32_t i = 0; i < input_len; ++i) {
            out[i] = output_values[i];
            checksum += static_cast<double>(output_values[i]);
        }
        result.checksum = static_cast<float>(checksum);
        result.ok = true;
    }
    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

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
    float* out) {
    GypsyDirectMetalProbeResult result;
    const auto start = std::chrono::steady_clock::now();
    if (residual == nullptr || o_projection == nullptr || post_norm_weight == nullptr ||
        gate_weight == nullptr || gate_scales == nullptr || gate_biases == nullptr ||
        up_weight == nullptr || up_scales == nullptr || up_biases == nullptr ||
        down_weight == nullptr || down_scales == nullptr || down_biases == nullptr ||
        out == nullptr) {
        result.error = PersistError("GypsyRunDirectMetalMlpTailToHost received null input");
        return result;
    }
    @autoreleasepool {
        DirectMetalAttentionContext& context = AttentionContext();
        result.device_available = context.device != nil;
        result.library_compiled = context.library != nil;
        result.pipeline_created = context.residual_add_pipeline != nil &&
            context.rmsnorm_pipeline != nil &&
            context.qkv_projection_pipeline != nil &&
            context.silu_multiply_pipeline != nil &&
            context.down_projection_pipeline != nil;
        if (!context.error.empty()) {
            result.error = PersistError(context.error);
            return result;
        }

        constexpr std::uint32_t hidden_size = 2560;
        constexpr std::uint32_t intermediate_size = 9728;
        constexpr std::uint32_t gate_up_packed_cols = hidden_size / 8;
        constexpr std::uint32_t gate_up_groups = hidden_size / 64;
        constexpr std::uint32_t down_packed_cols = intermediate_size / 8;
        constexpr std::uint32_t down_groups = intermediate_size / 64;
        const size_t hidden_bytes = static_cast<size_t>(hidden_size) * sizeof(float);
        const size_t intermediate_bytes = static_cast<size_t>(intermediate_size) * sizeof(float);
        const size_t norm_weight_bytes = static_cast<size_t>(hidden_size) * sizeof(std::uint16_t);
        const size_t gate_up_weight_bytes = static_cast<size_t>(intermediate_size) * gate_up_packed_cols * sizeof(std::uint32_t);
        const size_t gate_up_scale_bytes = static_cast<size_t>(intermediate_size) * gate_up_groups * sizeof(std::uint16_t);
        const size_t down_weight_bytes = static_cast<size_t>(hidden_size) * down_packed_cols * sizeof(std::uint32_t);
        const size_t down_scale_bytes = static_cast<size_t>(hidden_size) * down_groups * sizeof(std::uint16_t);

        auto ensure_buffer = [&](id<MTLBuffer>& buffer, size_t& capacity, size_t bytes) {
            if (buffer == nil || capacity < bytes) {
                buffer = [context.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                capacity = bytes;
            }
        };
        auto copy_if_changed = [&](id<MTLBuffer> buffer, const void*& source, const void* data, size_t bytes) {
            if (source != data) {
                std::memcpy(buffer.contents, data, bytes);
                source = data;
            }
        };

        ensure_buffer(context.residual_buffer, context.residual_capacity_bytes, hidden_bytes);
        ensure_buffer(context.o_out_buffer, context.o_out_capacity_bytes, hidden_bytes);
        ensure_buffer(context.rmsnorm_weight_buffer, context.rmsnorm_weight_capacity_bytes, norm_weight_bytes);
        ensure_buffer(context.rmsnorm_out_buffer, context.rmsnorm_out_capacity_bytes, hidden_bytes);
        ensure_buffer(context.gate_buffer, context.gate_capacity_bytes, intermediate_bytes);
        ensure_buffer(context.up_buffer, context.up_capacity_bytes, intermediate_bytes);
        ensure_buffer(context.activation_out_buffer, context.activation_out_capacity_bytes, intermediate_bytes);
        ensure_buffer(context.hidden_input_buffer, context.hidden_input_capacity_bytes, hidden_bytes);
        ensure_buffer(context.gate_weight_buffer, context.gate_weight_capacity_bytes, gate_up_weight_bytes);
        ensure_buffer(context.gate_scales_buffer, context.gate_scale_capacity_bytes, gate_up_scale_bytes);
        ensure_buffer(context.gate_biases_buffer, context.gate_scale_capacity_bytes, gate_up_scale_bytes);
        ensure_buffer(context.up_weight_buffer, context.up_weight_capacity_bytes, gate_up_weight_bytes);
        ensure_buffer(context.up_scales_buffer, context.up_scale_capacity_bytes, gate_up_scale_bytes);
        ensure_buffer(context.up_biases_buffer, context.up_scale_capacity_bytes, gate_up_scale_bytes);
        ensure_buffer(context.down_weight_buffer, context.down_weight_capacity_bytes, down_weight_bytes);
        ensure_buffer(context.down_scales_buffer, context.down_scale_capacity_bytes, down_scale_bytes);
        ensure_buffer(context.down_biases_buffer, context.down_scale_capacity_bytes, down_scale_bytes);
        if (context.rows_buffer == nil) {
            context.rows_buffer = [context.device newBufferWithLength:sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
        }
        if (context.n_buffer == nil) {
            context.n_buffer = [context.device newBufferWithLength:sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
        }
        if (context.residual_buffer == nil || context.o_out_buffer == nil || context.rmsnorm_weight_buffer == nil ||
            context.rmsnorm_out_buffer == nil || context.gate_buffer == nil || context.up_buffer == nil ||
            context.activation_out_buffer == nil || context.hidden_input_buffer == nil ||
            context.gate_weight_buffer == nil || context.gate_scales_buffer == nil || context.gate_biases_buffer == nil ||
            context.up_weight_buffer == nil || context.up_scales_buffer == nil || context.up_biases_buffer == nil ||
            context.down_weight_buffer == nil || context.down_scales_buffer == nil || context.down_biases_buffer == nil ||
            context.rows_buffer == nil || context.n_buffer == nil) {
            result.error = PersistError("newBuffer allocation failed for MLPTail");
            return result;
        }

        std::memcpy(context.residual_buffer.contents, residual, hidden_bytes);
        std::memcpy(context.o_out_buffer.contents, o_projection, hidden_bytes);
        std::memcpy(context.rmsnorm_weight_buffer.contents, post_norm_weight, norm_weight_bytes);
        copy_if_changed(context.gate_weight_buffer, context.gate_weight_source, gate_weight, gate_up_weight_bytes);
        copy_if_changed(context.gate_scales_buffer, context.gate_scales_source, gate_scales, gate_up_scale_bytes);
        copy_if_changed(context.gate_biases_buffer, context.gate_biases_source, gate_biases, gate_up_scale_bytes);
        copy_if_changed(context.up_weight_buffer, context.up_weight_source, up_weight, gate_up_weight_bytes);
        copy_if_changed(context.up_scales_buffer, context.up_scales_source, up_scales, gate_up_scale_bytes);
        copy_if_changed(context.up_biases_buffer, context.up_biases_source, up_biases, gate_up_scale_bytes);
        copy_if_changed(context.down_weight_buffer, context.down_weight_source, down_weight, down_weight_bytes);
        copy_if_changed(context.down_scales_buffer, context.down_scales_source, down_scales, down_scale_bytes);
        copy_if_changed(context.down_biases_buffer, context.down_biases_source, down_biases, down_scale_bytes);

        std::uint32_t gate_up_rows = intermediate_size;
        std::uint32_t activation_n = intermediate_size;
        std::memcpy(context.rows_buffer.contents, &gate_up_rows, sizeof(gate_up_rows));
        std::memcpy(context.n_buffer.contents, &activation_n, sizeof(activation_n));

        id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            result.error = PersistError("command buffer or encoder creation failed");
            return result;
        }

        [encoder setComputePipelineState:context.residual_add_pipeline];
        [encoder setBuffer:context.residual_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.o_out_buffer offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(hidden_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [encoder setComputePipelineState:context.rmsnorm_pipeline];
        [encoder setBuffer:context.o_out_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.rmsnorm_weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.rmsnorm_out_buffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        auto dispatch_gate_up_projection = [&](id<MTLBuffer> weight, id<MTLBuffer> scales, id<MTLBuffer> biases, id<MTLBuffer> output) {
            [encoder setComputePipelineState:context.qkv_projection_pipeline];
            [encoder setBuffer:context.rmsnorm_out_buffer offset:0 atIndex:0];
            [encoder setBuffer:weight offset:0 atIndex:1];
            [encoder setBuffer:scales offset:0 atIndex:2];
            [encoder setBuffer:biases offset:0 atIndex:3];
            [encoder setBuffer:output offset:0 atIndex:4];
            [encoder setBuffer:context.rows_buffer offset:0 atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake(intermediate_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        };
        dispatch_gate_up_projection(context.gate_weight_buffer, context.gate_scales_buffer, context.gate_biases_buffer, context.gate_buffer);
        dispatch_gate_up_projection(context.up_weight_buffer, context.up_scales_buffer, context.up_biases_buffer, context.up_buffer);

        [encoder setComputePipelineState:context.silu_multiply_pipeline];
        [encoder setBuffer:context.gate_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.up_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.activation_out_buffer offset:0 atIndex:2];
        [encoder setBuffer:context.n_buffer offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(intermediate_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [encoder setComputePipelineState:context.down_projection_pipeline];
        [encoder setBuffer:context.activation_out_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.down_weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.down_scales_buffer offset:0 atIndex:2];
        [encoder setBuffer:context.down_biases_buffer offset:0 atIndex:3];
        [encoder setBuffer:context.hidden_input_buffer offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(hidden_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        [encoder setComputePipelineState:context.residual_add_pipeline];
        [encoder setBuffer:context.o_out_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.hidden_input_buffer offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(hidden_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            result.error = PersistError("command buffer did not complete: " + NSErrorMessage(command_buffer.error));
            return result;
        }
        result.command_completed = true;
        const float* values = static_cast<const float*>(context.hidden_input_buffer.contents);
        double checksum = 0.0;
        for (std::uint32_t i = 0; i < hidden_size; ++i) {
            out[i] = values[i];
            checksum += static_cast<double>(values[i]);
        }
        result.checksum = static_cast<float>(checksum);
        result.ok = true;
    }
    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

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
    std::uint32_t read_output) {
    GypsyDirectMetalProbeResult result;
    const auto start = std::chrono::steady_clock::now();
    if ((load_input != 0 && input == nullptr) || input_norm_weight == nullptr ||
        q_weight == nullptr || q_scales == nullptr || q_biases == nullptr ||
        k_weight == nullptr || k_scales == nullptr || k_biases == nullptr ||
        v_weight == nullptr || v_scales == nullptr || v_biases == nullptr ||
        q_norm_weight == nullptr || k_norm_weight == nullptr ||
        o_weight == nullptr || o_scales == nullptr || o_biases == nullptr ||
        post_norm_weight == nullptr ||
        gate_weight == nullptr || gate_scales == nullptr || gate_biases == nullptr ||
        up_weight == nullptr || up_scales == nullptr || up_biases == nullptr ||
        down_weight == nullptr || down_scales == nullptr || down_biases == nullptr ||
        (read_output != 0 && out == nullptr)) {
        result.error = PersistError("GypsyRunDirectMetalFullLayerResident received null input");
        return result;
    }
    @autoreleasepool {
        DirectMetalAttentionContext& context = AttentionContext();
        result.device_available = context.device != nil;
        result.library_compiled = context.library != nil;
        result.pipeline_created = context.rmsnorm_pipeline != nil &&
            context.qkv_projection_pipeline != nil &&
            context.qk_norm_rope_pipeline != nil &&
            context.pipeline != nil &&
            context.o_projection_pipeline != nil &&
            context.residual_add_pipeline != nil &&
            context.silu_multiply_pipeline != nil &&
            context.down_projection_pipeline != nil;
        if (!context.error.empty()) {
            result.error = PersistError(context.error);
            return result;
        }
        if (layer >= context.layer_kv.size()) {
            result.error = PersistError("direct Metal full layer index out of range");
            return result;
        }

        constexpr std::uint32_t hidden_size = 2560;
        constexpr std::uint32_t intermediate_size = 9728;
        constexpr std::uint32_t q_rows = 4096;
        constexpr std::uint32_t kv_rows = 1024;
        constexpr std::uint32_t q_heads = 32;
        constexpr std::uint32_t kv_heads = 8;
        constexpr std::uint32_t head_dim = 128;
        constexpr std::uint32_t in_packed_cols = 320;
        constexpr std::uint32_t in_groups = 40;
        constexpr std::uint32_t o_packed_cols = 512;
        constexpr std::uint32_t o_groups = 64;
        constexpr std::uint32_t gate_up_packed_cols = hidden_size / 8;
        constexpr std::uint32_t gate_up_groups = hidden_size / 64;
        constexpr std::uint32_t down_packed_cols = intermediate_size / 8;
        constexpr std::uint32_t down_groups = intermediate_size / 64;
        const std::uint32_t n = std::max<std::uint32_t>(1, std::min<std::uint32_t>(seq_len, 256));
        const size_t hidden_bytes = static_cast<size_t>(hidden_size) * sizeof(float);
        const size_t q_bytes = static_cast<size_t>(q_rows) * sizeof(float);
        const size_t kv_bytes = static_cast<size_t>(kv_rows) * sizeof(float);
        const size_t current_kv_bytes = static_cast<size_t>(kv_heads) * head_dim * sizeof(float);
        const size_t kv_capacity_bytes = static_cast<size_t>(256) * kv_heads * head_dim * sizeof(float);
        const size_t intermediate_bytes = static_cast<size_t>(intermediate_size) * sizeof(float);
        const size_t hidden_norm_weight_bytes = static_cast<size_t>(hidden_size) * sizeof(std::uint16_t);
        const size_t head_norm_weight_bytes = static_cast<size_t>(head_dim) * sizeof(std::uint16_t);
        const size_t q_weight_bytes = static_cast<size_t>(q_rows) * in_packed_cols * sizeof(std::uint32_t);
        const size_t kv_weight_bytes = static_cast<size_t>(kv_rows) * in_packed_cols * sizeof(std::uint32_t);
        const size_t q_scale_bytes = static_cast<size_t>(q_rows) * in_groups * sizeof(std::uint16_t);
        const size_t kv_scale_bytes = static_cast<size_t>(kv_rows) * in_groups * sizeof(std::uint16_t);
        const size_t o_weight_bytes = static_cast<size_t>(hidden_size) * o_packed_cols * sizeof(std::uint32_t);
        const size_t o_scale_bytes = static_cast<size_t>(hidden_size) * o_groups * sizeof(std::uint16_t);
        const size_t gate_up_weight_bytes = static_cast<size_t>(intermediate_size) * gate_up_packed_cols * sizeof(std::uint32_t);
        const size_t gate_up_scale_bytes = static_cast<size_t>(intermediate_size) * gate_up_groups * sizeof(std::uint16_t);
        const size_t down_weight_bytes = static_cast<size_t>(hidden_size) * down_packed_cols * sizeof(std::uint32_t);
        const size_t down_scale_bytes = static_cast<size_t>(hidden_size) * down_groups * sizeof(std::uint16_t);

        auto ensure_buffer = [&](id<MTLBuffer>& buffer, size_t& capacity, size_t bytes) {
            if (buffer == nil || capacity < bytes) {
                buffer = [context.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                capacity = bytes;
            }
        };
        auto copy_if_changed = [&](id<MTLBuffer> buffer, const void*& source, const void* data, size_t bytes) {
            if (source != data) {
                std::memcpy(buffer.contents, data, bytes);
                source = data;
            }
        };

        ensure_buffer(context.hidden_input_buffer, context.hidden_input_capacity_bytes, hidden_bytes);
        ensure_buffer(context.residual_buffer, context.residual_capacity_bytes, hidden_bytes);
        ensure_buffer(context.rmsnorm_weight_buffer, context.rmsnorm_weight_capacity_bytes, hidden_norm_weight_bytes);
        ensure_buffer(context.post_norm_weight_buffer, context.post_norm_weight_capacity_bytes, hidden_norm_weight_bytes);
        ensure_buffer(context.rmsnorm_out_buffer, context.rmsnorm_out_capacity_bytes, hidden_bytes);
        ensure_buffer(context.q_raw_buffer, context.q_raw_capacity_bytes, q_bytes);
        ensure_buffer(context.k_raw_buffer, context.k_raw_capacity_bytes, kv_bytes);
        ensure_buffer(context.v_raw_buffer, context.v_raw_capacity_bytes, kv_bytes);
        ensure_buffer(context.q_rope_buffer, context.q_rope_capacity_bytes, q_bytes);
        ensure_buffer(context.k_rope_buffer, context.k_rope_capacity_bytes, kv_bytes);
        ensure_buffer(context.q_norm_weight_buffer, context.norm_weight_capacity_bytes, head_norm_weight_bytes);
        if (context.k_norm_weight_buffer == nil) {
            context.k_norm_weight_buffer = [context.device newBufferWithLength:head_norm_weight_bytes options:MTLResourceStorageModeShared];
        }
        DirectMetalAttentionContext::LayerKvBuffers& layer_buffers = context.layer_kv[layer];
        if (layer_buffers.k_buffer == nil || layer_buffers.capacity_bytes < kv_capacity_bytes) {
            layer_buffers.k_buffer = [context.device newBufferWithLength:kv_capacity_bytes options:MTLResourceStorageModeShared];
            layer_buffers.v_buffer = [context.device newBufferWithLength:kv_capacity_bytes options:MTLResourceStorageModeShared];
            layer_buffers.capacity_bytes = kv_capacity_bytes;
        }
        ensure_buffer(context.out_buffer, context.out_capacity_bytes, q_bytes);
        ensure_buffer(context.o_out_buffer, context.o_out_capacity_bytes, hidden_bytes);
        ensure_buffer(context.o_weight_buffer, context.o_weight_capacity_bytes, o_weight_bytes);
        ensure_buffer(context.o_scales_buffer, context.o_scale_capacity_bytes, o_scale_bytes);
        if (context.o_biases_buffer == nil || context.o_scale_capacity_bytes < o_scale_bytes) {
            context.o_biases_buffer = [context.device newBufferWithLength:o_scale_bytes options:MTLResourceStorageModeShared];
        }
        ensure_buffer(context.gate_buffer, context.gate_capacity_bytes, intermediate_bytes);
        ensure_buffer(context.up_buffer, context.up_capacity_bytes, intermediate_bytes);
        ensure_buffer(context.activation_out_buffer, context.activation_out_capacity_bytes, intermediate_bytes);
        ensure_buffer(context.gate_weight_buffer, context.gate_weight_capacity_bytes, gate_up_weight_bytes);
        ensure_buffer(context.gate_scales_buffer, context.gate_scale_capacity_bytes, gate_up_scale_bytes);
        ensure_buffer(context.gate_biases_buffer, context.gate_scale_capacity_bytes, gate_up_scale_bytes);
        ensure_buffer(context.up_weight_buffer, context.up_weight_capacity_bytes, gate_up_weight_bytes);
        ensure_buffer(context.up_scales_buffer, context.up_scale_capacity_bytes, gate_up_scale_bytes);
        ensure_buffer(context.up_biases_buffer, context.up_scale_capacity_bytes, gate_up_scale_bytes);
        ensure_buffer(context.down_weight_buffer, context.down_weight_capacity_bytes, down_weight_bytes);
        ensure_buffer(context.down_scales_buffer, context.down_scale_capacity_bytes, down_scale_bytes);
        ensure_buffer(context.down_biases_buffer, context.down_scale_capacity_bytes, down_scale_bytes);

        if (context.hidden_input_buffer == nil || context.residual_buffer == nil || context.rmsnorm_weight_buffer == nil ||
            context.post_norm_weight_buffer == nil ||
            context.rmsnorm_out_buffer == nil || context.q_raw_buffer == nil || context.k_raw_buffer == nil ||
            context.v_raw_buffer == nil || context.q_rope_buffer == nil || context.k_rope_buffer == nil ||
            context.q_norm_weight_buffer == nil || context.k_norm_weight_buffer == nil ||
            layer_buffers.k_buffer == nil || layer_buffers.v_buffer == nil || context.out_buffer == nil ||
            context.o_out_buffer == nil || context.o_weight_buffer == nil || context.o_scales_buffer == nil ||
            context.o_biases_buffer == nil || context.gate_buffer == nil || context.up_buffer == nil ||
            context.activation_out_buffer == nil || context.gate_weight_buffer == nil || context.gate_scales_buffer == nil ||
            context.gate_biases_buffer == nil || context.up_weight_buffer == nil || context.up_scales_buffer == nil ||
            context.up_biases_buffer == nil || context.down_weight_buffer == nil || context.down_scales_buffer == nil ||
            context.down_biases_buffer == nil) {
            result.error = PersistError("newBuffer allocation failed for full layer");
            return result;
        }

        auto ensure_layer_buffer = [&](id<MTLBuffer>& buffer, size_t& capacity, size_t bytes) {
            if (buffer == nil || capacity < bytes) {
                buffer = [context.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                capacity = bytes;
            }
        };
        ensure_layer_buffer(layer_buffers.q_weight_buffer, layer_buffers.q_weight_capacity_bytes, q_weight_bytes);
        ensure_layer_buffer(layer_buffers.q_scales_buffer, layer_buffers.q_scale_capacity_bytes, q_scale_bytes);
        ensure_layer_buffer(layer_buffers.q_biases_buffer, layer_buffers.q_scale_capacity_bytes, q_scale_bytes);
        ensure_layer_buffer(layer_buffers.k_weight_buffer, layer_buffers.kv_weight_capacity_bytes, kv_weight_bytes);
        ensure_layer_buffer(layer_buffers.k_scales_buffer, layer_buffers.kv_scale_capacity_bytes, kv_scale_bytes);
        ensure_layer_buffer(layer_buffers.k_biases_buffer, layer_buffers.kv_scale_capacity_bytes, kv_scale_bytes);
        ensure_layer_buffer(layer_buffers.v_weight_buffer, layer_buffers.kv_weight_capacity_bytes, kv_weight_bytes);
        ensure_layer_buffer(layer_buffers.v_scales_buffer, layer_buffers.kv_scale_capacity_bytes, kv_scale_bytes);
        ensure_layer_buffer(layer_buffers.v_biases_buffer, layer_buffers.kv_scale_capacity_bytes, kv_scale_bytes);

        if (load_input != 0) {
            std::memcpy(context.hidden_input_buffer.contents, input, hidden_bytes);
        }
        std::memcpy(context.rmsnorm_weight_buffer.contents, input_norm_weight, hidden_norm_weight_bytes);
        std::memcpy(context.post_norm_weight_buffer.contents, post_norm_weight, hidden_norm_weight_bytes);
        std::memcpy(context.q_norm_weight_buffer.contents, q_norm_weight, head_norm_weight_bytes);
        std::memcpy(context.k_norm_weight_buffer.contents, k_norm_weight, head_norm_weight_bytes);
        copy_if_changed(layer_buffers.q_weight_buffer, layer_buffers.q_weight_source, q_weight, q_weight_bytes);
        copy_if_changed(layer_buffers.q_scales_buffer, layer_buffers.q_scales_source, q_scales, q_scale_bytes);
        copy_if_changed(layer_buffers.q_biases_buffer, layer_buffers.q_biases_source, q_biases, q_scale_bytes);
        copy_if_changed(layer_buffers.k_weight_buffer, layer_buffers.k_weight_source, k_weight, kv_weight_bytes);
        copy_if_changed(layer_buffers.k_scales_buffer, layer_buffers.k_scales_source, k_scales, kv_scale_bytes);
        copy_if_changed(layer_buffers.k_biases_buffer, layer_buffers.k_biases_source, k_biases, kv_scale_bytes);
        copy_if_changed(layer_buffers.v_weight_buffer, layer_buffers.v_weight_source, v_weight, kv_weight_bytes);
        copy_if_changed(layer_buffers.v_scales_buffer, layer_buffers.v_scales_source, v_scales, kv_scale_bytes);
        copy_if_changed(layer_buffers.v_biases_buffer, layer_buffers.v_biases_source, v_biases, kv_scale_bytes);
        std::memcpy(context.o_weight_buffer.contents, o_weight, o_weight_bytes);
        std::memcpy(context.o_scales_buffer.contents, o_scales, o_scale_bytes);
        std::memcpy(context.o_biases_buffer.contents, o_biases, o_scale_bytes);
        std::memcpy(context.gate_weight_buffer.contents, gate_weight, gate_up_weight_bytes);
        std::memcpy(context.gate_scales_buffer.contents, gate_scales, gate_up_scale_bytes);
        std::memcpy(context.gate_biases_buffer.contents, gate_biases, gate_up_scale_bytes);
        std::memcpy(context.up_weight_buffer.contents, up_weight, gate_up_weight_bytes);
        std::memcpy(context.up_scales_buffer.contents, up_scales, gate_up_scale_bytes);
        std::memcpy(context.up_biases_buffer.contents, up_biases, gate_up_scale_bytes);
        std::memcpy(context.down_weight_buffer.contents, down_weight, down_weight_bytes);
        std::memcpy(context.down_scales_buffer.contents, down_scales, down_scale_bytes);
        std::memcpy(context.down_biases_buffer.contents, down_biases, down_scale_bytes);

        std::uint32_t q_rows_value = q_rows;
        std::uint32_t kv_rows_value = kv_rows;
        std::uint32_t gate_up_rows_value = intermediate_size;
        std::uint32_t seq_value = n;
        std::uint32_t activation_n = intermediate_size;
        id<MTLBuffer> q_rows_buffer = [context.device newBufferWithBytes:&q_rows_value length:sizeof(q_rows_value) options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_rows_buffer = [context.device newBufferWithBytes:&kv_rows_value length:sizeof(kv_rows_value) options:MTLResourceStorageModeShared];
        id<MTLBuffer> gate_up_rows_buffer = [context.device newBufferWithBytes:&gate_up_rows_value length:sizeof(gate_up_rows_value) options:MTLResourceStorageModeShared];
        id<MTLBuffer> seq_buffer = [context.device newBufferWithBytes:&seq_value length:sizeof(seq_value) options:MTLResourceStorageModeShared];
        id<MTLBuffer> activation_n_buffer = [context.device newBufferWithBytes:&activation_n length:sizeof(activation_n) options:MTLResourceStorageModeShared];
        id<MTLBuffer> position_buffer = [context.device newBufferWithBytes:&position length:sizeof(position) options:MTLResourceStorageModeShared];
        if (q_rows_buffer == nil || kv_rows_buffer == nil || gate_up_rows_buffer == nil ||
            seq_buffer == nil || activation_n_buffer == nil || position_buffer == nil) {
            result.error = PersistError("newBuffer allocation failed for full layer scalar buffers");
            return result;
        }

        id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
        if (command_buffer == nil) {
            result.error = PersistError("command buffer creation failed");
            return result;
        }
        id<MTLBlitCommandEncoder> initial_copy_encoder = [command_buffer blitCommandEncoder];
        if (initial_copy_encoder == nil) {
            result.error = PersistError("initial blit encoder creation failed");
            return result;
        }
        [initial_copy_encoder copyFromBuffer:context.hidden_input_buffer sourceOffset:0 toBuffer:context.residual_buffer destinationOffset:0 size:hidden_bytes];
        [initial_copy_encoder endEncoding];

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            result.error = PersistError("compute encoder creation failed");
            return result;
        }

        [encoder setComputePipelineState:context.rmsnorm_pipeline];
        [encoder setBuffer:context.hidden_input_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.rmsnorm_weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.rmsnorm_out_buffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        auto dispatch_input_projection = [&](id<MTLBuffer> weight, id<MTLBuffer> scales, id<MTLBuffer> biases, id<MTLBuffer> output, id<MTLBuffer> rows_buffer, std::uint32_t rows) {
            [encoder setComputePipelineState:context.qkv_projection_pipeline];
            [encoder setBuffer:context.rmsnorm_out_buffer offset:0 atIndex:0];
            [encoder setBuffer:weight offset:0 atIndex:1];
            [encoder setBuffer:scales offset:0 atIndex:2];
            [encoder setBuffer:biases offset:0 atIndex:3];
            [encoder setBuffer:output offset:0 atIndex:4];
            [encoder setBuffer:rows_buffer offset:0 atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        };
        dispatch_input_projection(layer_buffers.q_weight_buffer, layer_buffers.q_scales_buffer, layer_buffers.q_biases_buffer, context.q_raw_buffer, q_rows_buffer, q_rows);
        dispatch_input_projection(layer_buffers.k_weight_buffer, layer_buffers.k_scales_buffer, layer_buffers.k_biases_buffer, context.k_raw_buffer, kv_rows_buffer, kv_rows);
        dispatch_input_projection(layer_buffers.v_weight_buffer, layer_buffers.v_scales_buffer, layer_buffers.v_biases_buffer, context.v_raw_buffer, kv_rows_buffer, kv_rows);

        [encoder setComputePipelineState:context.qk_norm_rope_pipeline];
        [encoder setBuffer:context.q_raw_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.k_raw_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.q_norm_weight_buffer offset:0 atIndex:2];
        [encoder setBuffer:context.k_norm_weight_buffer offset:0 atIndex:3];
        [encoder setBuffer:context.q_rope_buffer offset:0 atIndex:4];
        [encoder setBuffer:context.k_rope_buffer offset:0 atIndex:5];
        [encoder setBuffer:position_buffer offset:0 atIndex:6];
        [encoder dispatchThreads:MTLSizeMake(q_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(q_heads, 1, 1)];
        [encoder endEncoding];

        const size_t write_offset = static_cast<size_t>(n - 1) * current_kv_bytes;
        id<MTLBlitCommandEncoder> copy_encoder = [command_buffer blitCommandEncoder];
        [copy_encoder copyFromBuffer:context.k_rope_buffer sourceOffset:0 toBuffer:layer_buffers.k_buffer destinationOffset:write_offset size:current_kv_bytes];
        [copy_encoder copyFromBuffer:context.v_raw_buffer sourceOffset:0 toBuffer:layer_buffers.v_buffer destinationOffset:write_offset size:current_kv_bytes];
        [copy_encoder endEncoding];

        encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:context.pipeline];
        [encoder setBuffer:context.q_rope_buffer offset:0 atIndex:0];
        [encoder setBuffer:layer_buffers.k_buffer offset:0 atIndex:1];
        [encoder setBuffer:layer_buffers.v_buffer offset:0 atIndex:2];
        [encoder setBuffer:context.out_buffer offset:0 atIndex:3];
        [encoder setBuffer:seq_buffer offset:0 atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(q_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(q_heads, 1, 1)];

        [encoder setComputePipelineState:context.o_projection_pipeline];
        [encoder setBuffer:context.out_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.o_weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.o_scales_buffer offset:0 atIndex:2];
        [encoder setBuffer:context.o_biases_buffer offset:0 atIndex:3];
        [encoder setBuffer:context.o_out_buffer offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(hidden_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        [encoder setComputePipelineState:context.residual_add_pipeline];
        [encoder setBuffer:context.residual_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.o_out_buffer offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(hidden_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [encoder setComputePipelineState:context.rmsnorm_pipeline];
        [encoder setBuffer:context.o_out_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.post_norm_weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.rmsnorm_out_buffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        auto dispatch_mlp_projection = [&](id<MTLBuffer> weight, id<MTLBuffer> scales, id<MTLBuffer> biases, id<MTLBuffer> output) {
            [encoder setComputePipelineState:context.qkv_projection_pipeline];
            [encoder setBuffer:context.rmsnorm_out_buffer offset:0 atIndex:0];
            [encoder setBuffer:weight offset:0 atIndex:1];
            [encoder setBuffer:scales offset:0 atIndex:2];
            [encoder setBuffer:biases offset:0 atIndex:3];
            [encoder setBuffer:output offset:0 atIndex:4];
            [encoder setBuffer:gate_up_rows_buffer offset:0 atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake(intermediate_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        };
        dispatch_mlp_projection(context.gate_weight_buffer, context.gate_scales_buffer, context.gate_biases_buffer, context.gate_buffer);
        dispatch_mlp_projection(context.up_weight_buffer, context.up_scales_buffer, context.up_biases_buffer, context.up_buffer);

        [encoder setComputePipelineState:context.silu_multiply_pipeline];
        [encoder setBuffer:context.gate_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.up_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.activation_out_buffer offset:0 atIndex:2];
        [encoder setBuffer:activation_n_buffer offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(intermediate_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [encoder setComputePipelineState:context.down_projection_pipeline];
        [encoder setBuffer:context.activation_out_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.down_weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.down_scales_buffer offset:0 atIndex:2];
        [encoder setBuffer:context.down_biases_buffer offset:0 atIndex:3];
        [encoder setBuffer:context.hidden_input_buffer offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(hidden_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        [encoder setComputePipelineState:context.residual_add_pipeline];
        [encoder setBuffer:context.o_out_buffer offset:0 atIndex:0];
        [encoder setBuffer:context.hidden_input_buffer offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(hidden_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            result.error = PersistError("command buffer did not complete: " + NSErrorMessage(command_buffer.error));
            return result;
        }
        result.command_completed = true;
        double checksum = 0.0;
        if (read_output != 0) {
            const float* output_values = static_cast<const float*>(context.hidden_input_buffer.contents);
            for (std::uint32_t i = 0; i < hidden_size; ++i) {
                out[i] = output_values[i];
                checksum += static_cast<double>(output_values[i]);
            }
        }
        result.checksum = static_cast<float>(checksum);
        result.ok = true;
    }
    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

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
    std::uint32_t position) {
    return GypsyRunDirectMetalFullLayerResident(
        layer,
        input,
        input_norm_weight,
        q_weight,
        q_scales,
        q_biases,
        k_weight,
        k_scales,
        k_biases,
        v_weight,
        v_scales,
        v_biases,
        q_norm_weight,
        k_norm_weight,
        o_weight,
        o_scales,
        o_biases,
        post_norm_weight,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        down_weight,
        down_scales,
        down_biases,
        out,
        seq_len,
        position,
        1,
        1);
}

GypsyDirectMetalProbeResult GypsyRunDirectMetalTerminalStackResident(
    const GypsyDirectMetalLayerParams* layers,
    std::uint32_t layer_count,
    const float* input,
    float* out) {
    GypsyDirectMetalProbeResult result;
    const auto start = std::chrono::steady_clock::now();
    if (layers == nullptr || layer_count == 0 || input == nullptr || out == nullptr) {
        result.error = PersistError("GypsyRunDirectMetalTerminalStackResident received null input");
        return result;
    }
    @autoreleasepool {
        DirectMetalAttentionContext& context = AttentionContext();
        result.device_available = context.device != nil;
        result.library_compiled = context.library != nil;
        result.pipeline_created = context.rmsnorm_pipeline != nil &&
            context.qkv_projection_pipeline != nil &&
            context.qk_norm_rope_pipeline != nil &&
            context.pipeline != nil &&
            context.o_projection_pipeline != nil &&
            context.residual_add_pipeline != nil &&
            context.silu_multiply_pipeline != nil &&
            context.down_projection_pipeline != nil;
        if (!context.error.empty()) {
            result.error = PersistError(context.error);
            return result;
        }

        constexpr std::uint32_t hidden_size = 2560;
        constexpr std::uint32_t intermediate_size = 9728;
        constexpr std::uint32_t q_rows = 4096;
        constexpr std::uint32_t kv_rows = 1024;
        constexpr std::uint32_t q_heads = 32;
        constexpr std::uint32_t kv_heads = 8;
        constexpr std::uint32_t head_dim = 128;
        constexpr std::uint32_t in_packed_cols = 320;
        constexpr std::uint32_t in_groups = 40;
        constexpr std::uint32_t o_packed_cols = 512;
        constexpr std::uint32_t o_groups = 64;
        constexpr std::uint32_t gate_up_packed_cols = hidden_size / 8;
        constexpr std::uint32_t gate_up_groups = hidden_size / 64;
        constexpr std::uint32_t down_packed_cols = intermediate_size / 8;
        constexpr std::uint32_t down_groups = intermediate_size / 64;
        const size_t hidden_bytes = static_cast<size_t>(hidden_size) * sizeof(float);
        const size_t q_bytes = static_cast<size_t>(q_rows) * sizeof(float);
        const size_t kv_bytes = static_cast<size_t>(kv_rows) * sizeof(float);
        const size_t current_kv_bytes = static_cast<size_t>(kv_heads) * head_dim * sizeof(float);
        const size_t kv_capacity_bytes = static_cast<size_t>(256) * kv_heads * head_dim * sizeof(float);
        const size_t intermediate_bytes = static_cast<size_t>(intermediate_size) * sizeof(float);
        const size_t hidden_norm_weight_bytes = static_cast<size_t>(hidden_size) * sizeof(std::uint16_t);
        const size_t head_norm_weight_bytes = static_cast<size_t>(head_dim) * sizeof(std::uint16_t);
        const size_t q_weight_bytes = static_cast<size_t>(q_rows) * in_packed_cols * sizeof(std::uint32_t);
        const size_t kv_weight_bytes = static_cast<size_t>(kv_rows) * in_packed_cols * sizeof(std::uint32_t);
        const size_t q_scale_bytes = static_cast<size_t>(q_rows) * in_groups * sizeof(std::uint16_t);
        const size_t kv_scale_bytes = static_cast<size_t>(kv_rows) * in_groups * sizeof(std::uint16_t);
        const size_t o_weight_bytes = static_cast<size_t>(hidden_size) * o_packed_cols * sizeof(std::uint32_t);
        const size_t o_scale_bytes = static_cast<size_t>(hidden_size) * o_groups * sizeof(std::uint16_t);
        const size_t gate_up_weight_bytes = static_cast<size_t>(intermediate_size) * gate_up_packed_cols * sizeof(std::uint32_t);
        const size_t gate_up_scale_bytes = static_cast<size_t>(intermediate_size) * gate_up_groups * sizeof(std::uint16_t);
        const size_t down_weight_bytes = static_cast<size_t>(hidden_size) * down_packed_cols * sizeof(std::uint32_t);
        const size_t down_scale_bytes = static_cast<size_t>(hidden_size) * down_groups * sizeof(std::uint16_t);

        auto ensure_buffer = [&](id<MTLBuffer>& buffer, size_t& capacity, size_t bytes) {
            if (buffer == nil || capacity < bytes) {
                buffer = [context.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                capacity = bytes;
            }
        };
        auto ensure_layer_buffer = [&](id<MTLBuffer>& buffer, size_t& capacity, size_t bytes) {
            if (buffer == nil || capacity < bytes) {
                buffer = [context.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                capacity = bytes;
            }
        };
        auto copy_layer_if_changed = [&](id<MTLBuffer> buffer, const void*& source, const void* data, size_t bytes) {
            if (source != data) {
                std::memcpy(buffer.contents, data, bytes);
                source = data;
            }
        };

        ensure_buffer(context.hidden_input_buffer, context.hidden_input_capacity_bytes, hidden_bytes);
        ensure_buffer(context.hidden_next_buffer, context.hidden_next_capacity_bytes, hidden_bytes);
        ensure_buffer(context.residual_buffer, context.residual_capacity_bytes, hidden_bytes);
        ensure_buffer(context.rmsnorm_out_buffer, context.rmsnorm_out_capacity_bytes, hidden_bytes);
        ensure_buffer(context.q_raw_buffer, context.q_raw_capacity_bytes, q_bytes);
        ensure_buffer(context.k_raw_buffer, context.k_raw_capacity_bytes, kv_bytes);
        ensure_buffer(context.v_raw_buffer, context.v_raw_capacity_bytes, kv_bytes);
        ensure_buffer(context.q_rope_buffer, context.q_rope_capacity_bytes, q_bytes);
        ensure_buffer(context.k_rope_buffer, context.k_rope_capacity_bytes, kv_bytes);
        ensure_buffer(context.out_buffer, context.out_capacity_bytes, q_bytes);
        ensure_buffer(context.o_out_buffer, context.o_out_capacity_bytes, hidden_bytes);
        ensure_buffer(context.gate_buffer, context.gate_capacity_bytes, intermediate_bytes);
        ensure_buffer(context.up_buffer, context.up_capacity_bytes, intermediate_bytes);
        ensure_buffer(context.activation_out_buffer, context.activation_out_capacity_bytes, intermediate_bytes);
        if (context.hidden_input_buffer == nil || context.hidden_next_buffer == nil || context.residual_buffer == nil ||
            context.rmsnorm_out_buffer == nil || context.q_raw_buffer == nil || context.k_raw_buffer == nil ||
            context.v_raw_buffer == nil || context.q_rope_buffer == nil || context.k_rope_buffer == nil ||
            context.out_buffer == nil || context.o_out_buffer == nil || context.gate_buffer == nil ||
            context.up_buffer == nil || context.activation_out_buffer == nil) {
            result.error = PersistError("newBuffer allocation failed for terminal stack shared buffers");
            return result;
        }

        std::memcpy(context.hidden_input_buffer.contents, input, hidden_bytes);

        for (std::uint32_t i = 0; i < layer_count; ++i) {
            const GypsyDirectMetalLayerParams& lp = layers[i];
            if (lp.layer >= context.layer_kv.size() ||
                lp.input_norm_weight == nullptr || lp.q_weight == nullptr || lp.q_scales == nullptr || lp.q_biases == nullptr ||
                lp.k_weight == nullptr || lp.k_scales == nullptr || lp.k_biases == nullptr ||
                lp.v_weight == nullptr || lp.v_scales == nullptr || lp.v_biases == nullptr ||
                lp.q_norm_weight == nullptr || lp.k_norm_weight == nullptr ||
                lp.o_weight == nullptr || lp.o_scales == nullptr || lp.o_biases == nullptr ||
                lp.post_norm_weight == nullptr ||
                lp.gate_weight == nullptr || lp.gate_scales == nullptr || lp.gate_biases == nullptr ||
                lp.up_weight == nullptr || lp.up_scales == nullptr || lp.up_biases == nullptr ||
                lp.down_weight == nullptr || lp.down_scales == nullptr || lp.down_biases == nullptr) {
                result.error = PersistError("terminal stack layer has missing tensor");
                return result;
            }
            DirectMetalAttentionContext::LayerKvBuffers& lb = context.layer_kv[lp.layer];
            ensure_layer_buffer(lb.k_buffer, lb.capacity_bytes, kv_capacity_bytes);
            ensure_layer_buffer(lb.v_buffer, lb.capacity_bytes, kv_capacity_bytes);
            ensure_layer_buffer(lb.q_weight_buffer, lb.q_weight_capacity_bytes, q_weight_bytes);
            ensure_layer_buffer(lb.q_scales_buffer, lb.q_scale_capacity_bytes, q_scale_bytes);
            ensure_layer_buffer(lb.q_biases_buffer, lb.q_scale_capacity_bytes, q_scale_bytes);
            ensure_layer_buffer(lb.k_weight_buffer, lb.kv_weight_capacity_bytes, kv_weight_bytes);
            ensure_layer_buffer(lb.k_scales_buffer, lb.kv_scale_capacity_bytes, kv_scale_bytes);
            ensure_layer_buffer(lb.k_biases_buffer, lb.kv_scale_capacity_bytes, kv_scale_bytes);
            ensure_layer_buffer(lb.v_weight_buffer, lb.kv_weight_capacity_bytes, kv_weight_bytes);
            ensure_layer_buffer(lb.v_scales_buffer, lb.kv_scale_capacity_bytes, kv_scale_bytes);
            ensure_layer_buffer(lb.v_biases_buffer, lb.kv_scale_capacity_bytes, kv_scale_bytes);
            ensure_layer_buffer(lb.o_weight_buffer, lb.o_weight_capacity_bytes, o_weight_bytes);
            ensure_layer_buffer(lb.o_scales_buffer, lb.o_scale_capacity_bytes, o_scale_bytes);
            ensure_layer_buffer(lb.o_biases_buffer, lb.o_scale_capacity_bytes, o_scale_bytes);
            ensure_layer_buffer(lb.input_norm_weight_buffer, lb.input_norm_weight_capacity_bytes, hidden_norm_weight_bytes);
            ensure_layer_buffer(lb.q_norm_weight_buffer, lb.head_norm_weight_capacity_bytes, head_norm_weight_bytes);
            ensure_layer_buffer(lb.k_norm_weight_buffer, lb.head_norm_weight_capacity_bytes, head_norm_weight_bytes);
            ensure_layer_buffer(lb.post_norm_weight_buffer, lb.post_norm_weight_capacity_bytes, hidden_norm_weight_bytes);
            ensure_layer_buffer(lb.gate_weight_buffer, lb.gate_weight_capacity_bytes, gate_up_weight_bytes);
            ensure_layer_buffer(lb.gate_scales_buffer, lb.gate_scale_capacity_bytes, gate_up_scale_bytes);
            ensure_layer_buffer(lb.gate_biases_buffer, lb.gate_scale_capacity_bytes, gate_up_scale_bytes);
            ensure_layer_buffer(lb.up_weight_buffer, lb.up_weight_capacity_bytes, gate_up_weight_bytes);
            ensure_layer_buffer(lb.up_scales_buffer, lb.up_scale_capacity_bytes, gate_up_scale_bytes);
            ensure_layer_buffer(lb.up_biases_buffer, lb.up_scale_capacity_bytes, gate_up_scale_bytes);
            ensure_layer_buffer(lb.down_weight_buffer, lb.down_weight_capacity_bytes, down_weight_bytes);
            ensure_layer_buffer(lb.down_scales_buffer, lb.down_scale_capacity_bytes, down_scale_bytes);
            ensure_layer_buffer(lb.down_biases_buffer, lb.down_scale_capacity_bytes, down_scale_bytes);
            if (lb.k_buffer == nil || lb.v_buffer == nil || lb.q_weight_buffer == nil || lb.q_scales_buffer == nil ||
                lb.q_biases_buffer == nil || lb.k_weight_buffer == nil || lb.k_scales_buffer == nil ||
                lb.k_biases_buffer == nil || lb.v_weight_buffer == nil || lb.v_scales_buffer == nil ||
                lb.v_biases_buffer == nil || lb.o_weight_buffer == nil || lb.o_scales_buffer == nil ||
                lb.o_biases_buffer == nil || lb.input_norm_weight_buffer == nil || lb.q_norm_weight_buffer == nil ||
                lb.k_norm_weight_buffer == nil || lb.post_norm_weight_buffer == nil || lb.gate_weight_buffer == nil ||
                lb.gate_scales_buffer == nil || lb.gate_biases_buffer == nil || lb.up_weight_buffer == nil ||
                lb.up_scales_buffer == nil || lb.up_biases_buffer == nil || lb.down_weight_buffer == nil ||
                lb.down_scales_buffer == nil || lb.down_biases_buffer == nil) {
                result.error = PersistError("newBuffer allocation failed for terminal stack layer buffers");
                return result;
            }
            copy_layer_if_changed(lb.input_norm_weight_buffer, lb.input_norm_weight_source, lp.input_norm_weight, hidden_norm_weight_bytes);
            copy_layer_if_changed(lb.q_norm_weight_buffer, lb.q_norm_weight_source, lp.q_norm_weight, head_norm_weight_bytes);
            copy_layer_if_changed(lb.k_norm_weight_buffer, lb.k_norm_weight_source, lp.k_norm_weight, head_norm_weight_bytes);
            copy_layer_if_changed(lb.post_norm_weight_buffer, lb.post_norm_weight_source, lp.post_norm_weight, hidden_norm_weight_bytes);
            copy_layer_if_changed(lb.q_weight_buffer, lb.q_weight_source, lp.q_weight, q_weight_bytes);
            copy_layer_if_changed(lb.q_scales_buffer, lb.q_scales_source, lp.q_scales, q_scale_bytes);
            copy_layer_if_changed(lb.q_biases_buffer, lb.q_biases_source, lp.q_biases, q_scale_bytes);
            copy_layer_if_changed(lb.k_weight_buffer, lb.k_weight_source, lp.k_weight, kv_weight_bytes);
            copy_layer_if_changed(lb.k_scales_buffer, lb.k_scales_source, lp.k_scales, kv_scale_bytes);
            copy_layer_if_changed(lb.k_biases_buffer, lb.k_biases_source, lp.k_biases, kv_scale_bytes);
            copy_layer_if_changed(lb.v_weight_buffer, lb.v_weight_source, lp.v_weight, kv_weight_bytes);
            copy_layer_if_changed(lb.v_scales_buffer, lb.v_scales_source, lp.v_scales, kv_scale_bytes);
            copy_layer_if_changed(lb.v_biases_buffer, lb.v_biases_source, lp.v_biases, kv_scale_bytes);
            copy_layer_if_changed(lb.o_weight_buffer, lb.o_weight_source, lp.o_weight, o_weight_bytes);
            copy_layer_if_changed(lb.o_scales_buffer, lb.o_scales_source, lp.o_scales, o_scale_bytes);
            copy_layer_if_changed(lb.o_biases_buffer, lb.o_biases_source, lp.o_biases, o_scale_bytes);
            copy_layer_if_changed(lb.gate_weight_buffer, lb.gate_weight_source, lp.gate_weight, gate_up_weight_bytes);
            copy_layer_if_changed(lb.gate_scales_buffer, lb.gate_scales_source, lp.gate_scales, gate_up_scale_bytes);
            copy_layer_if_changed(lb.gate_biases_buffer, lb.gate_biases_source, lp.gate_biases, gate_up_scale_bytes);
            copy_layer_if_changed(lb.up_weight_buffer, lb.up_weight_source, lp.up_weight, gate_up_weight_bytes);
            copy_layer_if_changed(lb.up_scales_buffer, lb.up_scales_source, lp.up_scales, gate_up_scale_bytes);
            copy_layer_if_changed(lb.up_biases_buffer, lb.up_biases_source, lp.up_biases, gate_up_scale_bytes);
            copy_layer_if_changed(lb.down_weight_buffer, lb.down_weight_source, lp.down_weight, down_weight_bytes);
            copy_layer_if_changed(lb.down_scales_buffer, lb.down_scales_source, lp.down_scales, down_scale_bytes);
            copy_layer_if_changed(lb.down_biases_buffer, lb.down_biases_source, lp.down_biases, down_scale_bytes);
        }

        id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
        if (command_buffer == nil) {
            result.error = PersistError("terminal stack command buffer creation failed");
            return result;
        }

        for (std::uint32_t i = 0; i < layer_count; ++i) {
            const GypsyDirectMetalLayerParams& lp = layers[i];
            DirectMetalAttentionContext::LayerKvBuffers& lb = context.layer_kv[lp.layer];
            const std::uint32_t q_rows_value = q_rows;
            const std::uint32_t kv_rows_value = kv_rows;
            const std::uint32_t gate_up_rows_value = intermediate_size;
            const std::uint32_t seq_value = std::max<std::uint32_t>(1, std::min<std::uint32_t>(lp.seq_len, 256));
            const std::uint32_t activation_n = intermediate_size;
            id<MTLBuffer> q_rows_buffer = [context.device newBufferWithBytes:&q_rows_value length:sizeof(q_rows_value) options:MTLResourceStorageModeShared];
            id<MTLBuffer> kv_rows_buffer = [context.device newBufferWithBytes:&kv_rows_value length:sizeof(kv_rows_value) options:MTLResourceStorageModeShared];
            id<MTLBuffer> gate_up_rows_buffer = [context.device newBufferWithBytes:&gate_up_rows_value length:sizeof(gate_up_rows_value) options:MTLResourceStorageModeShared];
            id<MTLBuffer> seq_buffer = [context.device newBufferWithBytes:&seq_value length:sizeof(seq_value) options:MTLResourceStorageModeShared];
            id<MTLBuffer> activation_n_buffer = [context.device newBufferWithBytes:&activation_n length:sizeof(activation_n) options:MTLResourceStorageModeShared];
            id<MTLBuffer> position_buffer = [context.device newBufferWithBytes:&lp.position length:sizeof(lp.position) options:MTLResourceStorageModeShared];
            if (q_rows_buffer == nil || kv_rows_buffer == nil || gate_up_rows_buffer == nil ||
                seq_buffer == nil || activation_n_buffer == nil || position_buffer == nil) {
                result.error = PersistError("terminal stack scalar buffer allocation failed");
                return result;
            }

            id<MTLBlitCommandEncoder> copy_encoder = [command_buffer blitCommandEncoder];
            if (copy_encoder == nil) {
                result.error = PersistError("terminal stack residual copy encoder creation failed");
                return result;
            }
            [copy_encoder copyFromBuffer:context.hidden_input_buffer sourceOffset:0 toBuffer:context.residual_buffer destinationOffset:0 size:hidden_bytes];
            [copy_encoder endEncoding];

            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (encoder == nil) {
                result.error = PersistError("terminal stack compute encoder creation failed");
                return result;
            }

            [encoder setComputePipelineState:context.rmsnorm_pipeline];
            [encoder setBuffer:context.hidden_input_buffer offset:0 atIndex:0];
            [encoder setBuffer:lb.input_norm_weight_buffer offset:0 atIndex:1];
            [encoder setBuffer:context.rmsnorm_out_buffer offset:0 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            auto dispatch_input_projection = [&](id<MTLBuffer> weight, id<MTLBuffer> scales, id<MTLBuffer> biases, id<MTLBuffer> output, id<MTLBuffer> rows_buffer, std::uint32_t rows) {
                [encoder setComputePipelineState:context.qkv_projection_pipeline];
                [encoder setBuffer:context.rmsnorm_out_buffer offset:0 atIndex:0];
                [encoder setBuffer:weight offset:0 atIndex:1];
                [encoder setBuffer:scales offset:0 atIndex:2];
                [encoder setBuffer:biases offset:0 atIndex:3];
                [encoder setBuffer:output offset:0 atIndex:4];
                [encoder setBuffer:rows_buffer offset:0 atIndex:5];
                [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            };
            dispatch_input_projection(lb.q_weight_buffer, lb.q_scales_buffer, lb.q_biases_buffer, context.q_raw_buffer, q_rows_buffer, q_rows);
            dispatch_input_projection(lb.k_weight_buffer, lb.k_scales_buffer, lb.k_biases_buffer, context.k_raw_buffer, kv_rows_buffer, kv_rows);
            dispatch_input_projection(lb.v_weight_buffer, lb.v_scales_buffer, lb.v_biases_buffer, context.v_raw_buffer, kv_rows_buffer, kv_rows);

            [encoder setComputePipelineState:context.qk_norm_rope_pipeline];
            [encoder setBuffer:context.q_raw_buffer offset:0 atIndex:0];
            [encoder setBuffer:context.k_raw_buffer offset:0 atIndex:1];
            [encoder setBuffer:lb.q_norm_weight_buffer offset:0 atIndex:2];
            [encoder setBuffer:lb.k_norm_weight_buffer offset:0 atIndex:3];
            [encoder setBuffer:context.q_rope_buffer offset:0 atIndex:4];
            [encoder setBuffer:context.k_rope_buffer offset:0 atIndex:5];
            [encoder setBuffer:position_buffer offset:0 atIndex:6];
            [encoder dispatchThreads:MTLSizeMake(q_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(q_heads, 1, 1)];
            [encoder endEncoding];

            const size_t write_offset = static_cast<size_t>(seq_value - 1) * current_kv_bytes;
            copy_encoder = [command_buffer blitCommandEncoder];
            [copy_encoder copyFromBuffer:context.k_rope_buffer sourceOffset:0 toBuffer:lb.k_buffer destinationOffset:write_offset size:current_kv_bytes];
            [copy_encoder copyFromBuffer:context.v_raw_buffer sourceOffset:0 toBuffer:lb.v_buffer destinationOffset:write_offset size:current_kv_bytes];
            [copy_encoder endEncoding];

            encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:context.pipeline];
            [encoder setBuffer:context.q_rope_buffer offset:0 atIndex:0];
            [encoder setBuffer:lb.k_buffer offset:0 atIndex:1];
            [encoder setBuffer:lb.v_buffer offset:0 atIndex:2];
            [encoder setBuffer:context.out_buffer offset:0 atIndex:3];
            [encoder setBuffer:seq_buffer offset:0 atIndex:4];
            [encoder dispatchThreads:MTLSizeMake(q_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(q_heads, 1, 1)];

            [encoder setComputePipelineState:context.o_projection_pipeline];
            [encoder setBuffer:context.out_buffer offset:0 atIndex:0];
            [encoder setBuffer:lb.o_weight_buffer offset:0 atIndex:1];
            [encoder setBuffer:lb.o_scales_buffer offset:0 atIndex:2];
            [encoder setBuffer:lb.o_biases_buffer offset:0 atIndex:3];
            [encoder setBuffer:context.o_out_buffer offset:0 atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(hidden_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

            [encoder setComputePipelineState:context.residual_add_pipeline];
            [encoder setBuffer:context.residual_buffer offset:0 atIndex:0];
            [encoder setBuffer:context.o_out_buffer offset:0 atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(hidden_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [encoder setComputePipelineState:context.rmsnorm_pipeline];
            [encoder setBuffer:context.o_out_buffer offset:0 atIndex:0];
            [encoder setBuffer:lb.post_norm_weight_buffer offset:0 atIndex:1];
            [encoder setBuffer:context.rmsnorm_out_buffer offset:0 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            auto dispatch_mlp_projection = [&](id<MTLBuffer> weight, id<MTLBuffer> scales, id<MTLBuffer> biases, id<MTLBuffer> output) {
                [encoder setComputePipelineState:context.qkv_projection_pipeline];
                [encoder setBuffer:context.rmsnorm_out_buffer offset:0 atIndex:0];
                [encoder setBuffer:weight offset:0 atIndex:1];
                [encoder setBuffer:scales offset:0 atIndex:2];
                [encoder setBuffer:biases offset:0 atIndex:3];
                [encoder setBuffer:output offset:0 atIndex:4];
                [encoder setBuffer:gate_up_rows_buffer offset:0 atIndex:5];
                [encoder dispatchThreadgroups:MTLSizeMake(intermediate_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            };
            dispatch_mlp_projection(lb.gate_weight_buffer, lb.gate_scales_buffer, lb.gate_biases_buffer, context.gate_buffer);
            dispatch_mlp_projection(lb.up_weight_buffer, lb.up_scales_buffer, lb.up_biases_buffer, context.up_buffer);

            [encoder setComputePipelineState:context.silu_multiply_pipeline];
            [encoder setBuffer:context.gate_buffer offset:0 atIndex:0];
            [encoder setBuffer:context.up_buffer offset:0 atIndex:1];
            [encoder setBuffer:context.activation_out_buffer offset:0 atIndex:2];
            [encoder setBuffer:activation_n_buffer offset:0 atIndex:3];
            [encoder dispatchThreads:MTLSizeMake(intermediate_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [encoder setComputePipelineState:context.down_projection_pipeline];
            [encoder setBuffer:context.activation_out_buffer offset:0 atIndex:0];
            [encoder setBuffer:lb.down_weight_buffer offset:0 atIndex:1];
            [encoder setBuffer:lb.down_scales_buffer offset:0 atIndex:2];
            [encoder setBuffer:lb.down_biases_buffer offset:0 atIndex:3];
            [encoder setBuffer:context.hidden_next_buffer offset:0 atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(hidden_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

            [encoder setComputePipelineState:context.residual_add_pipeline];
            [encoder setBuffer:context.o_out_buffer offset:0 atIndex:0];
            [encoder setBuffer:context.hidden_next_buffer offset:0 atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(hidden_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];

            copy_encoder = [command_buffer blitCommandEncoder];
            [copy_encoder copyFromBuffer:context.hidden_next_buffer sourceOffset:0 toBuffer:context.hidden_input_buffer destinationOffset:0 size:hidden_bytes];
            [copy_encoder endEncoding];
        }

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            result.error = PersistError("terminal stack command buffer did not complete: " + NSErrorMessage(command_buffer.error));
            return result;
        }

        result.command_completed = true;
        const float* output_values = static_cast<const float*>(context.hidden_input_buffer.contents);
        double checksum = 0.0;
        for (std::uint32_t i = 0; i < hidden_size; ++i) {
            out[i] = output_values[i];
            checksum += static_cast<double>(output_values[i]);
        }
        result.checksum = static_cast<float>(checksum);
        result.ok = true;
    }
    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}
