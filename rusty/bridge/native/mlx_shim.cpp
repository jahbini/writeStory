#include "mlx_shim.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <utility>

#include <fcntl.h>
#if defined(__APPLE__)
#include <mach/mach.h>
#endif
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"
#endif

namespace {

std::uint64_t read_u64_le(const std::uint8_t* data);

#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
struct TestArrayRecord {
  mlx::core::array value;
};

struct TokenArrayRecord {
  mlx::core::array value;
  std::string source_group;
};

struct ArrayRecord {
  mlx::core::array value;
  std::string source_group;
};

struct MappedFile {
  int fd = -1;
  std::size_t size = 0;
  const std::uint8_t* data = nullptr;

  MappedFile(int fd_in, std::size_t size_in, const std::uint8_t* data_in)
      : fd(fd_in), size(size_in), data(data_in) {}

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  ~MappedFile() {
    if (data != nullptr && data != MAP_FAILED && size > 0) {
      munmap(const_cast<std::uint8_t*>(data), size);
    }
    if (fd >= 0) {
      close(fd);
    }
  }
};

struct GroupTensorRecord {
  std::string kind;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::uint64_t byte_start = 0;
  std::uint64_t byte_end = 0;
  std::string source_file;
  std::vector<std::uint8_t> payload;
  const std::uint8_t* payload_view = nullptr;
  std::size_t payload_view_size = 0;
  std::shared_ptr<const MappedFile> mapped_file;
};

struct TensorGroupRecord {
  std::string group;
  std::string source_dir;
  std::string index_path;
  bool loaded = true;
  bool quantized_group = false;
  std::uint64_t total_byte_size = 0;
  std::unordered_map<std::string, GroupTensorRecord> tensors;
  bool quantized_layout_cached = false;
  std::size_t quantized_rows = 0;
  std::size_t quantized_packed_cols = 0;
  std::size_t quantized_scale_cols = 0;
  std::size_t quantized_logical_width = 0;
  std::size_t quantized_block_size = 8;
  std::size_t quantized_output_len = 0;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  mutable bool mlx_quantized_arrays_cached = false;
  mutable std::vector<std::uint32_t> mlx_weight_words;
  mutable std::vector<mlx::core::bfloat16_t> mlx_scale_values;
  mutable std::vector<mlx::core::bfloat16_t> mlx_bias_values;
  mutable std::optional<mlx::core::array> mlx_weight_array;
  mutable std::optional<mlx::core::array> mlx_scales_array;
  mutable std::optional<mlx::core::array> mlx_biases_array;
#endif
};

struct LayerGroupsRecord {
  int layer = 0;
  std::string source_dir;
  std::string model_dir;
  std::string index_path;
  bool loaded = true;
  std::unordered_map<std::string, std::string> group_handles;
  std::uint64_t total_byte_size = 0;
  std::uint64_t quantized_group_count = 0;
  std::uint64_t norm_group_count = 0;
};

struct EmbeddingGroupRecord {
  std::string group;
  std::string source_dir;
  std::string index_path;
  bool loaded = true;
  bool quantized_group = false;
  std::uint64_t total_byte_size = 0;
  std::unordered_map<std::string, GroupTensorRecord> tensors;
};

struct RuntimeProbeResult {
  bool primary_success = false;
  std::string primary_target = "default_device";
  std::string primary_failure_stage;
  std::string primary_exception;
  bool cpu_attempted = false;
  bool cpu_success = false;
  std::string cpu_failure_stage;
  std::string cpu_exception;
};

struct RopeRuntimeConfig {
  bool enabled = false;
  double theta = 10000.0;
  std::uint64_t max_position_embeddings = 0;
  std::size_t head_dim = 128;
  std::size_t num_attention_heads = 0;
  std::size_t num_key_value_heads = 0;
};

RopeRuntimeConfig& active_rope_config() {
  static RopeRuntimeConfig config;
  return config;
}

struct NativeLayerKvCache {
  std::vector<std::vector<float>> keys;
  std::vector<std::vector<float>> values;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  std::optional<mlx::core::array> expanded_keys;
  std::optional<mlx::core::array> expanded_values;
  std::size_t expanded_current_len = 0;
  std::vector<mlx::core::array> expanded_key_chunks;
  std::vector<mlx::core::array> expanded_value_chunks;
  std::size_t expanded_chunk_current_len = 0;
  std::size_t expanded_chunk_size = 0;
  std::vector<mlx::core::array> compact_key_chunks;
  std::vector<mlx::core::array> compact_value_chunks;
  std::size_t compact_chunk_current_len = 0;
  std::size_t compact_chunk_size = 0;
#endif
};

enum class CachePolicyKind {
  Full,
  Rotating,
  Quantized,
  Recompute,
};

struct CachePolicy {
  CachePolicyKind kind = CachePolicyKind::Full;
  std::uint64_t step = 1;
  std::uint64_t max_size = 0;
  std::uint64_t keep = 0;
  std::uint64_t bits = 0;
  std::uint64_t group_size = 0;
  std::uint64_t start_at = 0;
  std::uint64_t window = 0;
  std::string base;
};

class LayerKvCache {
 public:
  virtual ~LayerKvCache() = default;
  virtual void update_and_fetch() = 0;
  virtual void trim(std::uint64_t max_len) = 0;
  virtual std::uint64_t len() const = 0;
  virtual std::uint64_t nbytes() const = 0;
  virtual bool is_empty() const = 0;
};

class ChunkedExpandedKvCache final : public LayerKvCache {
 public:
  explicit ChunkedExpandedKvCache(const NativeLayerKvCache& layer)
      : layer_(layer) {}

  void update_and_fetch() override {
    // The current implementation updates/fetches through the existing
    // mlx_expanded_kv_cache_* helpers. This concrete cache object names the
    // active policy without changing execution behavior yet.
  }

  void trim(std::uint64_t /*max_len*/) override {
    // Full-style cache keeps all positions currently in scope.
  }

  std::uint64_t len() const override {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
    if (layer_.compact_chunk_size > 0) {
      return static_cast<std::uint64_t>(layer_.compact_chunk_current_len);
    }
    if (layer_.expanded_chunk_size > 0) {
      return static_cast<std::uint64_t>(layer_.expanded_chunk_current_len);
    }
    if (layer_.expanded_current_len > 0) {
      return static_cast<std::uint64_t>(layer_.expanded_current_len);
    }
#endif
    return static_cast<std::uint64_t>(layer_.keys.size());
  }

  std::uint64_t nbytes() const override {
    std::uint64_t bytes = 0;
    for (const auto& key : layer_.keys) {
      bytes += static_cast<std::uint64_t>(key.size() * sizeof(float));
    }
    for (const auto& value : layer_.values) {
      bytes += static_cast<std::uint64_t>(value.size() * sizeof(float));
    }
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
    for (const auto& chunk : layer_.expanded_key_chunks) {
      bytes += static_cast<std::uint64_t>(chunk.nbytes());
    }
    for (const auto& chunk : layer_.expanded_value_chunks) {
      bytes += static_cast<std::uint64_t>(chunk.nbytes());
    }
    for (const auto& chunk : layer_.compact_key_chunks) {
      bytes += static_cast<std::uint64_t>(chunk.nbytes());
    }
    for (const auto& chunk : layer_.compact_value_chunks) {
      bytes += static_cast<std::uint64_t>(chunk.nbytes());
    }
    if (layer_.expanded_keys.has_value()) {
      bytes += static_cast<std::uint64_t>(layer_.expanded_keys->nbytes());
    }
    if (layer_.expanded_values.has_value()) {
      bytes += static_cast<std::uint64_t>(layer_.expanded_values->nbytes());
    }
#endif
    return bytes;
  }

  bool is_empty() const override {
    return len() == 0;
  }

 private:
  const NativeLayerKvCache& layer_;
};

std::uint64_t layer_kv_cache_len(const NativeLayerKvCache& layer) {
  return ChunkedExpandedKvCache(layer).len();
}

struct NativeSessionKvCache {
  std::string owner_session;
  std::uint64_t layers_allocated = 0;
  std::uint64_t positions_stored = 0;
  std::uint64_t max_seq = 0;
  std::vector<NativeLayerKvCache> layers;
};

struct ResidentLoadTiming {
  double safetensor_index_lookup_ms = 0.0;
  double tensor_group_metadata_construction_ms = 0.0;
  double mmap_file_read_ms = 0.0;
  double mlx_array_creation_ms = 0.0;
  double quantized_group_preparation_ms = 0.0;
  double synchronization_eval_ms = 0.0;
};

struct ResidentProjectionWarmupTiming {
  double enumerate_groups_ms = 0.0;
  double mmap_setup_ms = 0.0;
  double mlx_array_construction_ms = 0.0;
  double first_eval_compile_warmup_ms = 0.0;
  double metadata_cache_storage_ms = 0.0;
};

struct ResidentLayerGroups {
  TensorGroupRecord input_norm;
  TensorGroupRecord q_proj;
  TensorGroupRecord k_proj;
  TensorGroupRecord v_proj;
  TensorGroupRecord o_proj;
  TensorGroupRecord q_norm;
  TensorGroupRecord k_norm;
  TensorGroupRecord post_attention_norm;
  TensorGroupRecord gate_proj;
  TensorGroupRecord up_proj;
  TensorGroupRecord down_proj;
  std::uint64_t total_byte_size = 0;
};

struct ResidentModelLayers {
  std::string model_dir;
  std::vector<ResidentLayerGroups> layers;
  std::uint64_t total_groups = 0;
  std::uint64_t total_byte_size = 0;
  std::uint64_t tensor_group_load_count = 0;
  ResidentLoadTiming load_timing;
};

struct NativeResidentSessionRecord {
  std::string session;
  std::string model_dir;
  std::string adapter_dir;
  TensorGroupRecord final_norm;
  TensorGroupRecord embedding;
  bool warmed = false;
  double first_warmup_ms = 0.0;
  ResidentProjectionWarmupTiming first_warmup_timing;
};

struct LoraProjectionRecord {
  bool present = false;
  std::string name;
  std::vector<std::uint64_t> a_shape;
  std::vector<std::uint64_t> b_shape;
  std::optional<mlx::core::array> a;
  std::optional<mlx::core::array> b;
};

struct LoraLayerRecord {
  int layer = -1;
  LoraProjectionRecord q_proj;
  LoraProjectionRecord k_proj;
  LoraProjectionRecord v_proj;
  LoraProjectionRecord o_proj;
  LoraProjectionRecord gate_proj;
  LoraProjectionRecord up_proj;
  LoraProjectionRecord down_proj;
};

struct NativeAdapterRecord {
  std::string adapter_dir;
  std::string dtype = "F32";
  std::uint64_t rank = 0;
  double scale = 0.0;
  std::uint64_t tensor_count = 0;
  std::vector<int> layers;
  std::vector<std::string> targets;
  std::vector<std::string> missing_expected_tensors;
  std::vector<std::string> unexpected_tensors;
  std::shared_ptr<const MappedFile> mapped_file;
  std::unordered_map<int, LoraLayerRecord> layer_records;
};

std::unordered_map<std::string, std::shared_ptr<NativeAdapterRecord>>& native_adapter_cache() {
  static std::unordered_map<std::string, std::shared_ptr<NativeAdapterRecord>> table;
  return table;
}

std::mutex& native_adapter_mutex() {
  static std::mutex mutex;
  return mutex;
}

thread_local std::shared_ptr<NativeAdapterRecord> active_generation_adapter;

std::unordered_map<std::string, NativeResidentSessionRecord>& native_resident_session_table() {
  static std::unordered_map<std::string, NativeResidentSessionRecord> table;
  return table;
}

std::mutex& native_resident_session_mutex() {
  static std::mutex mutex;
  return mutex;
}

struct ResidentDecodeResult {
  std::vector<float> logits;
  std::size_t logits_len = 0;
  double final_norm_checksum = 0.0;
  std::uint64_t top_token_id = 0;
  float top_token_score = 0.0f;
  double timing_ms = 0.0;
  double embedding_ms = 0.0;
  double qkv_projection_ms = 0.0;
  double qk_norm_rope_ms = 0.0;
  double o_projection_ms = 0.0;
  double gate_up_projection_ms = 0.0;
  double gate_projection_ms = 0.0;
  double up_projection_ms = 0.0;
  double gate_up_activation_ms = 0.0;
  double gate_projection_setup_ms = 0.0;
  double gate_projection_eval_ms = 0.0;
  double gate_projection_readback_ms = 0.0;
  double up_projection_setup_ms = 0.0;
  double up_projection_eval_ms = 0.0;
  double up_projection_readback_ms = 0.0;
  double down_projection_ms = 0.0;
  double final_norm_ms = 0.0;
  double logits_projection_ms = 0.0;
  double kv_append_ms = 0.0;
  double attention_score_ms = 0.0;
  double attention_softmax_ms = 0.0;
  double attention_value_mix_ms = 0.0;
  double attention_kv_view_assembly_ms = 0.0;
  double attention_score_matmul_ms = 0.0;
  double attention_softmax_detail_ms = 0.0;
  double attention_value_mix_matmul_ms = 0.0;
  double attention_reshape_flatten_ms = 0.0;
  double attention_eval_sync_ms = 0.0;
  std::vector<double> per_layer_attention_ms;
  std::vector<std::string> per_layer_attention_backends;
  std::vector<std::string> per_layer_attention_fallback_reasons;
  std::string gate_backend = "cpu";
  std::string up_backend = "cpu";
  std::string activation_backend = "cpu";
  bool gate_up_fallback_used = false;
  bool mlx_resident_mlp_chain_applied = false;
  bool mlx_resident_mlp_chain_fallback_used = false;
  bool mlx_resident_layer_block_applied = false;
  bool mlx_resident_layer_block_fallback_used = false;
  std::string largest_arithmetic_bucket;
  double largest_arithmetic_bucket_ms = 0.0;
  std::uint64_t positions_before = 0;
  std::uint64_t positions_after = 0;
};

struct DecodeTimingSnapshot {
  std::uint64_t token_position = 0;
  std::uint64_t sequence_length_at_token = 0;
  std::uint64_t kv_layers_allocated = 0;
  std::uint64_t kv_positions_stored = 0;
  std::size_t k_length = 0;
  std::size_t v_length = 0;
  ResidentDecodeResult result;
  std::vector<std::string> fallback_steps;
  std::vector<std::string> readback_reasons;
};

struct CachedAttentionTiming {
  double kv_append_ms = 0.0;
  double score_ms = 0.0;
  double softmax_ms = 0.0;
  double value_mix_ms = 0.0;

  double total_attention_ms() const {
    return kv_append_ms + score_ms + softmax_ms + value_mix_ms;
  }
};

struct MlxAttentionDetailTiming {
  double kv_view_assembly_ms = 0.0;
  double score_matmul_ms = 0.0;
  double softmax_ms = 0.0;
  double value_mix_matmul_ms = 0.0;
  double reshape_flatten_ms = 0.0;
  double eval_sync_ms = 0.0;
};

struct GenerationTimingBuckets {
  double qkv_ms = 0.0;
  double qk_norm_rope_ms = 0.0;
  double attention_append_ms = 0.0;
  double attention_math_ms = 0.0;
  double attention_kv_view_assembly_ms = 0.0;
  double attention_score_matmul_ms = 0.0;
  double attention_softmax_detail_ms = 0.0;
  double attention_value_mix_matmul_ms = 0.0;
  double attention_reshape_flatten_ms = 0.0;
  double attention_eval_sync_ms = 0.0;
  double o_proj_ms = 0.0;
  double mlp_gate_up_ms = 0.0;
  double mlp_activation_ms = 0.0;
  double mlp_down_ms = 0.0;
  double logits_sampling_ms = 0.0;
  double sync_readback_ms = 0.0;
  double accounted_ms = 0.0;
  std::uint64_t qkv_sync_count = 0;
  std::uint64_t attention_append_sync_count = 0;
  std::uint64_t attention_eval_sync_count = 0;
  std::uint64_t o_residual_sync_count = 0;
  std::uint64_t mlp_residual_sync_count = 0;
  std::uint64_t logits_sync_count = 0;
  std::uint64_t final_readback_sync_count = 0;
  std::uint64_t tokens = 0;

  void add(const ResidentDecodeResult& result) {
    qkv_ms += result.qkv_projection_ms;
    qk_norm_rope_ms += result.qk_norm_rope_ms;
    attention_append_ms += result.kv_append_ms;
    attention_math_ms +=
        result.attention_score_ms +
        result.attention_softmax_ms +
        result.attention_value_mix_ms;
    attention_kv_view_assembly_ms += result.attention_kv_view_assembly_ms;
    attention_score_matmul_ms += result.attention_score_matmul_ms;
    attention_softmax_detail_ms += result.attention_softmax_detail_ms;
    attention_value_mix_matmul_ms += result.attention_value_mix_matmul_ms;
    attention_reshape_flatten_ms += result.attention_reshape_flatten_ms;
    attention_eval_sync_ms += result.attention_eval_sync_ms;
    o_proj_ms += result.o_projection_ms;
    mlp_gate_up_ms += result.gate_up_projection_ms;
    mlp_activation_ms += result.gate_up_activation_ms;
    mlp_down_ms += result.down_projection_ms;
    logits_sampling_ms += result.logits_projection_ms;
    // The current path synchronizes/readbacks inside MLX projection helpers and
    // final logits selection; until those are separately plumbed, track visible
    // final norm/checksum readback here so this bucket is not hidden.
    sync_readback_ms += result.final_norm_ms;
    accounted_ms +=
        result.qkv_projection_ms +
        result.qk_norm_rope_ms +
        result.kv_append_ms +
        result.attention_score_ms +
        result.attention_softmax_ms +
        result.attention_value_mix_ms +
        result.o_projection_ms +
        result.gate_up_projection_ms +
        result.gate_up_activation_ms +
        result.down_projection_ms +
        result.logits_projection_ms +
        result.final_norm_ms;
    const std::uint64_t layer_steps =
        static_cast<std::uint64_t>(result.per_layer_attention_ms.size());
    qkv_sync_count += layer_steps;
    // Chunk append updates are now deferred into the following attention eval.
    attention_append_sync_count += 0;
    attention_eval_sync_count += result.attention_eval_sync_ms > 0.0 ? layer_steps : 0;
    o_residual_sync_count += layer_steps;
    mlp_residual_sync_count += layer_steps;
    logits_sync_count += result.logits_projection_ms > 0.0 ? 1 : 0;
    final_readback_sync_count += result.final_norm_ms > 0.0 ? 1 : 0;
    tokens += 1;
  }
};

std::string generation_timing_bucket_summary_json(
    const GenerationTimingBuckets& buckets,
    double generation_ms) {
  auto item = [&](const char* name, double ms, const char* backend) {
    std::ostringstream out;
    const double pct = generation_ms > 0.0 ? (ms * 100.0) / generation_ms : 0.0;
    const double avg = buckets.tokens > 0
        ? ms / static_cast<double>(buckets.tokens)
        : 0.0;
    out << "{"
        << "\"name\":\"" << name << "\","
        << "\"cumulative_ms\":" << ms << ","
        << "\"average_ms_per_token\":" << avg << ","
        << "\"percent_of_generation_time\":" << pct << ","
        << "\"backend\":\"" << backend << "\""
        << "}";
    return out.str();
  };
  struct SyncSite {
    const char* name;
    std::uint64_t count;
    double ms;
    const char* reason;
  };
  std::vector<SyncSite> sync_sites{
      {"attention_eval_sync", buckets.attention_eval_sync_count, buckets.attention_eval_sync_ms, "materialize chunked attention output for CPU vector boundary"},
      {"qkv_projection_eval_sync", buckets.qkv_sync_count, buckets.qkv_ms, "materialize q/k/v for q_norm/k_norm/RoPE and KV append"},
      {"mlp_down_residual_eval_sync", buckets.mlp_residual_sync_count, buckets.mlp_down_ms, "materialize MLP residual output as next layer state"},
      {"o_projection_residual_eval_sync", buckets.o_residual_sync_count, buckets.o_proj_ms, "materialize attention residual before post-attention norm"},
      {"logits_eval_sync", buckets.logits_sync_count, buckets.logits_sampling_ms, "materialize logits for token selection"},
      {"final_norm_readback_sync", buckets.final_readback_sync_count, buckets.sync_readback_ms, "final checksum/logits input readback"},
      {"attention_append_sync", buckets.attention_append_sync_count, 0.0, "deferred into attention eval for chunked_expanded_kv"}
  };
  std::sort(sync_sites.begin(), sync_sites.end(), [](const SyncSite& a, const SyncSite& b) {
    return a.ms > b.ms;
  });
  const std::uint64_t sync_count_total =
      buckets.qkv_sync_count +
      buckets.attention_append_sync_count +
      buckets.attention_eval_sync_count +
      buckets.o_residual_sync_count +
      buckets.mlp_residual_sync_count +
      buckets.logits_sync_count +
      buckets.final_readback_sync_count;
  auto sync_sites_json = [&]() {
    std::ostringstream sites;
    sites << "[";
    for (std::size_t i = 0; i < sync_sites.size(); ++i) {
      if (i > 0) {
        sites << ",";
      }
      sites << "{"
            << "\"site\":\"" << sync_sites[i].name << "\","
            << "\"count\":" << sync_sites[i].count << ","
            << "\"cumulative_ms\":" << sync_sites[i].ms << ","
            << "\"reason\":\"" << sync_sites[i].reason << "\""
            << "}";
    }
    sites << "]";
    return sites.str();
  };
  std::ostringstream out;
  out << "{"
      << "\"tokens_measured\":" << buckets.tokens << ","
      << "\"generation_time_ms\":" << generation_ms << ","
      << "\"accounted_ms\":" << buckets.accounted_ms << ","
      << "\"unaccounted_ms\":" << (generation_ms - buckets.accounted_ms) << ","
      << "\"sync_count_total\":" << sync_count_total << ","
      << "\"sync_count_per_token_avg\":"
      << (buckets.tokens > 0 ? static_cast<double>(sync_count_total) / static_cast<double>(buckets.tokens) : 0.0)
      << ","
      << "\"top_sync_sites_by_cumulative_ms\":" << sync_sites_json() << ","
      << "\"sync_sites_inside_attention\":["
      << "\"attention_append_sync\","
      << "\"attention_eval_sync\""
      << "],"
      << "\"sync_sites_between_qkv_o_mlp_logits\":["
      << "\"qkv_projection_eval_sync\","
      << "\"o_projection_residual_eval_sync\","
      << "\"mlp_down_residual_eval_sync\","
      << "\"logits_eval_sync\","
      << "\"final_norm_readback_sync\""
      << "],"
      << "\"buckets\":["
      << item("qkv", buckets.qkv_ms, "metal") << ","
      << item("q_norm/k_norm/RoPE", buckets.qk_norm_rope_ms, "cpu") << ","
      << item("attention_append_update", buckets.attention_append_ms, "mixed") << ","
      << item("attention_matmul_softmax_value_mix", buckets.attention_math_ms, "metal") << ","
      << item("attention_kv_chunk_view_assembly", buckets.attention_kv_view_assembly_ms, "metal") << ","
      << item("attention_score_matmul", buckets.attention_score_matmul_ms, "metal") << ","
      << item("attention_softmax", buckets.attention_softmax_detail_ms, "metal") << ","
      << item("attention_value_mix_matmul", buckets.attention_value_mix_matmul_ms, "metal") << ","
      << item("attention_reshape_flatten", buckets.attention_reshape_flatten_ms, "cpu") << ","
      << item("attention_eval_sync", buckets.attention_eval_sync_ms, "mixed") << ","
      << item("o_proj", buckets.o_proj_ms, "metal") << ","
      << item("MLP_gate_up", buckets.mlp_gate_up_ms, "metal") << ","
      << item("MLP_activation", buckets.mlp_activation_ms, "metal") << ","
      << item("MLP_down", buckets.mlp_down_ms, "metal") << ","
      << item("logits_sampling", buckets.logits_sampling_ms, "metal") << ","
      << item("sync_readback", buckets.sync_readback_ms, "mixed")
      << "]"
      << "}";
  return out.str();
}

std::string json_escape(const std::string& value);

void add_elapsed_ms(double& bucket, std::chrono::steady_clock::time_point start) {
  auto end = std::chrono::steady_clock::now();
  bucket += std::chrono::duration<double, std::milli>(end - start).count();
}

void update_largest_arithmetic_bucket(ResidentDecodeResult& result) {
  const std::pair<const char*, double> buckets[] = {
      {"embedding_lookup", result.embedding_ms},
      {"qkv_projections", result.qkv_projection_ms},
      {"q_norm_k_norm_rope", result.qk_norm_rope_ms},
      {"o_projection", result.o_projection_ms},
      {"gate_up_paired_projection", result.gate_up_projection_ms},
      {"down_projection", result.down_projection_ms},
      {"final_norm", result.final_norm_ms},
      {"logits_projection_top1", result.logits_projection_ms},
      {"attention_cpu_total", result.kv_append_ms + result.attention_score_ms +
          result.attention_softmax_ms + result.attention_value_mix_ms},
  };
  result.largest_arithmetic_bucket = buckets[0].first;
  result.largest_arithmetic_bucket_ms = buckets[0].second;
  for (const auto& bucket : buckets) {
    if (bucket.second > result.largest_arithmetic_bucket_ms) {
      result.largest_arithmetic_bucket = bucket.first;
      result.largest_arithmetic_bucket_ms = bucket.second;
    }
  }
}

std::string resident_decode_timing_buckets_json(const ResidentDecodeResult& result) {
  std::ostringstream out;
  out << "{"
      << "\"embedding_lookup\":" << result.embedding_ms << ","
      << "\"qkv_projections\":" << result.qkv_projection_ms << ","
      << "\"q_norm_k_norm_rope\":" << result.qk_norm_rope_ms << ","
      << "\"o_projection\":" << result.o_projection_ms << ","
      << "\"gate_up_paired_projection\":" << result.gate_up_projection_ms << ","
      << "\"gate_projection\":" << result.gate_projection_ms << ","
      << "\"up_projection\":" << result.up_projection_ms << ","
      << "\"gate_up_activation\":" << result.gate_up_activation_ms << ","
      << "\"down_projection\":" << result.down_projection_ms << ","
      << "\"final_norm\":" << result.final_norm_ms << ","
      << "\"logits_projection_top1\":" << result.logits_projection_ms << ","
      << "\"kv_append\":" << result.kv_append_ms << ","
      << "\"attention_score_computation\":" << result.attention_score_ms << ","
      << "\"softmax\":" << result.attention_softmax_ms << ","
      << "\"attention_value_mix\":" << result.attention_value_mix_ms << ","
      << "\"total_attention_cpu\":" << (result.kv_append_ms + result.attention_score_ms +
          result.attention_softmax_ms + result.attention_value_mix_ms)
      << "}";
  return out.str();
}

std::string attention_layer_timing_summary_json(const ResidentDecodeResult& result) {
  if (result.per_layer_attention_ms.empty()) {
    return "{\"available\":false,\"min_ms\":0,\"avg_ms\":0,\"max_ms\":0,\"slowest_layer\":0,\"slowest_layer_ms\":0,\"sequence_length\":0}";
  }
  double min_ms = result.per_layer_attention_ms[0];
  double max_ms = result.per_layer_attention_ms[0];
  double sum_ms = 0.0;
  std::size_t slowest_layer = 0;
  for (std::size_t i = 0; i < result.per_layer_attention_ms.size(); ++i) {
    const double value = result.per_layer_attention_ms[i];
    min_ms = std::min(min_ms, value);
    if (value > max_ms) {
      max_ms = value;
      slowest_layer = i;
    }
    sum_ms += value;
  }
  std::ostringstream out;
  out << "{"
      << "\"available\":true,"
      << "\"min_ms\":" << min_ms << ","
      << "\"avg_ms\":" << (sum_ms / static_cast<double>(result.per_layer_attention_ms.size())) << ","
      << "\"max_ms\":" << max_ms << ","
      << "\"slowest_layer\":" << slowest_layer << ","
      << "\"slowest_layer_ms\":" << max_ms << ","
      << "\"sequence_length\":" << result.positions_after
      << "}";
  return out.str();
}

std::string attention_backends_json(const ResidentDecodeResult& result) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < result.per_layer_attention_backends.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << "{"
        << "\"layer\":" << i << ","
        << "\"backend\":\"" << json_escape(result.per_layer_attention_backends[i]) << "\","
        << "\"timing_ms\":";
    if (i < result.per_layer_attention_ms.size()) {
      out << result.per_layer_attention_ms[i];
    } else {
      out << 0;
    }
    out << ",\"fallback_reason\":\"";
    if (i < result.per_layer_attention_fallback_reasons.size()) {
      out << json_escape(result.per_layer_attention_fallback_reasons[i]);
    }
    out << "\"}";
  }
  out << "]";
  return out.str();
}

std::string projection_timing_breakdown_json(const ResidentDecodeResult& result) {
  auto projection_json = [](double setup_ms, double eval_ms, double readback_ms, double total_ms) {
    std::ostringstream out;
    out << "{"
        << "\"setup_ms\":" << setup_ms << ","
        << "\"compute_eval_ms\":" << eval_ms << ","
        << "\"readback_ms\":" << readback_ms << ","
        << "\"total_ms\":" << total_ms
        << "}";
    return out.str();
  };
  std::ostringstream out;
  out << "{"
      << "\"gate_proj\":" << projection_json(
             result.gate_projection_setup_ms,
             result.gate_projection_eval_ms,
             result.gate_projection_readback_ms,
             result.gate_projection_ms) << ","
      << "\"up_proj\":" << projection_json(
             result.up_projection_setup_ms,
             result.up_projection_eval_ms,
             result.up_projection_readback_ms,
             result.up_projection_ms)
      << "}";
  return out.str();
}

extern thread_local std::vector<std::string> mlx_quantized_linear_fallback_steps;

std::string fallback_steps_json() {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < mlx_quantized_linear_fallback_steps.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << "\"" << json_escape(mlx_quantized_linear_fallback_steps[i]) << "\"";
  }
  out << "]";
  return out.str();
}

std::string backend_step_json(
    const char* backend,
    bool fallback_used,
    double timing_ms,
    std::optional<double> checksum = std::nullopt) {
  std::ostringstream out;
  out << "{"
      << "\"backend\":\"" << backend << "\","
      << "\"fallback_used\":" << (fallback_used ? "true" : "false") << ","
      << "\"timing_ms\":" << timing_ms << ","
      << "\"checksum\":";
  if (checksum.has_value()) {
    out << *checksum;
  } else {
    out << "null";
  }
  out << "}";
  return out.str();
}

std::string gate_up_backend_step_json(const ResidentDecodeResult& result) {
  std::ostringstream out;
  out << "{"
      << "\"backend\":\"" << ((result.gate_backend == "metal" && result.up_backend == "metal") ? "metal" : "cpu") << "\","
      << "\"gate_backend\":\"" << json_escape(result.gate_backend) << "\","
      << "\"up_backend\":\"" << json_escape(result.up_backend) << "\","
      << "\"activation_backend\":\"" << json_escape(result.activation_backend) << "\","
      << "\"fallback_used\":" << (result.gate_up_fallback_used ? "true" : "false") << ","
      << "\"mlx_resident_mlp_chain_applied\":" << (result.mlx_resident_mlp_chain_applied ? "true" : "false") << ","
      << "\"mlx_resident_mlp_chain_fallback_used\":" << (result.mlx_resident_mlp_chain_fallback_used ? "true" : "false") << ","
      << "\"timing_ms\":" << result.gate_up_projection_ms << ","
      << "\"checksum\":null"
      << "}";
  return out.str();
}

std::string backend_report_json(
    const ResidentDecodeResult& result,
    bool metal_available,
    const std::vector<std::string>& fallback_steps) {
  auto fell_back = [&](const char* step) {
    return std::find(fallback_steps.begin(), fallback_steps.end(), step) != fallback_steps.end();
  };
  auto metal_or_cpu = [&](const char* step) {
    return (metal_available && !fell_back(step)) ? "metal" : "cpu";
  };
  std::ostringstream out;
  out << "{"
      << "\"embedding\":" << backend_step_json("cpu", false, 0.0) << ","
      << "\"qkv\":" << backend_step_json(metal_or_cpu("qkv"), fell_back("qkv"), result.qkv_projection_ms) << ","
      << "\"attention\":" << backend_step_json(
             (!result.per_layer_attention_backends.empty() &&
              std::all_of(
                  result.per_layer_attention_backends.begin(),
                  result.per_layer_attention_backends.end(),
                  [](const std::string& backend) {
                    return backend == "mlx" || backend == "mlx_batched" ||
                           backend == "mlx_expanded_kv" ||
                           backend == "mlx_chunked_expanded_kv" ||
                           backend == "mlx_chunked_compact_mlx" ||
                           backend == "mlx_chunked_compact_mlx_chunk_aware" ||
                           backend == "mlx_resident_block_chunked_expanded_kv";
                  }))
                 ? result.per_layer_attention_backends.front().c_str()
                 : "cpu",
             std::any_of(
                 result.per_layer_attention_fallback_reasons.begin(),
                 result.per_layer_attention_fallback_reasons.end(),
                 [](const std::string& reason) { return !reason.empty(); }),
             result.kv_append_ms + result.attention_score_ms + result.attention_softmax_ms +
                 result.attention_value_mix_ms) << ","
      << "\"o_proj\":" << backend_step_json(metal_or_cpu("o_proj"), fell_back("o_proj"), result.o_projection_ms) << ","
      << "\"post_attn_norm\":" << backend_step_json("cpu", false, 0.0) << ","
      << "\"gate_up\":" << gate_up_backend_step_json(result) << ","
      << "\"down\":" << backend_step_json(metal_or_cpu("down"), fell_back("down"), result.down_projection_ms) << ","
      << "\"final_norm\":" << backend_step_json("cpu", false, result.final_norm_ms, result.final_norm_checksum) << ","
      << "\"logits/top1\":" << backend_step_json(metal_or_cpu("logits/top1"), fell_back("logits/top1"), result.logits_projection_ms)
      << "}";
  return out.str();
}

std::string decode_timing_snapshot_json(
    const DecodeTimingSnapshot& snapshot,
    bool metal_available) {
  auto fell_back = [&](const char* step) {
    return std::find(
        snapshot.fallback_steps.begin(),
        snapshot.fallback_steps.end(),
        step) != snapshot.fallback_steps.end();
  };
  auto metal_or_cpu = [&](const char* step) {
    return (metal_available && !fell_back(step)) ? "metal" : "cpu";
  };
  auto bucket = [](double timing_ms, const char* backend) {
    std::ostringstream out;
    out << "{"
        << "\"timing_ms\":" << timing_ms << ","
        << "\"backend\":\"" << backend << "\""
        << "}";
    return out.str();
  };
  auto string_array = [](const std::vector<std::string>& values) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (i > 0) {
        out << ",";
      }
      out << "\"" << json_escape(values[i]) << "\"";
    }
    out << "]";
    return out.str();
  };

  const ResidentDecodeResult& r = snapshot.result;
  std::ostringstream out;
  out << "{"
      << "\"token_position\":" << snapshot.token_position << ","
      << "\"sequence_length_at_token\":" << snapshot.sequence_length_at_token << ","
      << "\"total_timing_ms\":" << r.timing_ms << ","
      << "\"buckets\":{"
      << "\"q_norm/k_norm/RoPE\":" << bucket(r.qk_norm_rope_ms, "cpu") << ","
      << "\"qkv_projection\":" << bucket(r.qkv_projection_ms, metal_or_cpu("qkv")) << ","
      << "\"KV_append\":" << bucket(
             r.kv_append_ms,
             (!r.per_layer_attention_backends.empty() &&
              r.per_layer_attention_backends.front().find("mlx_") == 0) ? "mixed" : "cpu") << ","
      << "\"attention_score_computation\":" << bucket(
             r.attention_score_ms,
             (!r.per_layer_attention_backends.empty() &&
              r.per_layer_attention_backends.front().find("mlx_") == 0) ? "metal" : "cpu") << ","
      << "\"softmax\":" << bucket(
             r.attention_softmax_ms,
             (!r.per_layer_attention_backends.empty() &&
              r.per_layer_attention_backends.front().find("mlx_") == 0) ? "metal" : "cpu") << ","
      << "\"attention_value_mix\":" << bucket(
             r.attention_value_mix_ms,
             (!r.per_layer_attention_backends.empty() &&
              r.per_layer_attention_backends.front().find("mlx_") == 0) ? "metal" : "cpu") << ","
      << "\"attention_kv_chunk_view_assembly\":" << bucket(r.attention_kv_view_assembly_ms, "metal") << ","
      << "\"attention_score_matmul\":" << bucket(r.attention_score_matmul_ms, "metal") << ","
      << "\"attention_softmax_detail\":" << bucket(r.attention_softmax_detail_ms, "metal") << ","
      << "\"attention_value_mix_matmul\":" << bucket(r.attention_value_mix_matmul_ms, "metal") << ","
      << "\"attention_reshape_flatten\":" << bucket(r.attention_reshape_flatten_ms, "cpu") << ","
      << "\"attention_eval_sync\":" << bucket(r.attention_eval_sync_ms, "mixed") << ","
      << "\"o_proj\":" << bucket(r.o_projection_ms, metal_or_cpu("o_proj")) << ","
      << "\"MLP_gate_up\":" << bucket(r.gate_up_projection_ms, r.gate_backend == "metal" && r.up_backend == "metal" ? "metal" : "mixed") << ","
      << "\"MLP_activation\":" << bucket(r.gate_up_activation_ms, r.activation_backend.c_str()) << ","
      << "\"MLP_down\":" << bucket(r.down_projection_ms, metal_or_cpu("down")) << ","
      << "\"final_norm\":" << bucket(r.final_norm_ms, "mixed") << ","
      << "\"logits/top1_or_sampling\":" << bucket(r.logits_projection_ms, metal_or_cpu("logits/top1")) << ","
      << "\"readback/sync\":" << bucket(0.0, snapshot.readback_reasons.empty() ? "metal" : "mixed")
      << "},"
      << "\"backend_report\":" << backend_report_json(r, metal_available, snapshot.fallback_steps) << ","
      << "\"readback_count_per_token\":" << snapshot.readback_reasons.size() << ","
      << "\"readback_reasons_per_token\":" << string_array(snapshot.readback_reasons) << ","
      << "\"fallback_steps_per_token\":" << string_array(snapshot.fallback_steps) << ","
      << "\"kv_cache_backend\":\""
      << ((!r.per_layer_attention_backends.empty() &&
           r.per_layer_attention_backends.front().find("mlx_") == 0) ? "mlx_expanded_q_heads" : "cpu")
      << "\","
      << "\"kv_cache_shape\":{"
      << "\"layers_allocated\":" << snapshot.kv_layers_allocated << ","
      << "\"positions_stored\":" << snapshot.kv_positions_stored << ","
      << "\"k_length\":" << snapshot.k_length << ","
      << "\"v_length\":" << snapshot.v_length << ","
      << "\"layout\":\""
      << ((!r.per_layer_attention_backends.empty() &&
           r.per_layer_attention_backends.front() == "mlx_expanded_kv")
              ? "per_layer_expanded_q_head_preallocated_mlx_float"
              : "per_layer_vector_per_position_cpu_float")
      << "\""
      << "},"
      << "\"attention_backend\":\""
      << ((!r.per_layer_attention_backends.empty()) ? json_escape(r.per_layer_attention_backends.front()) : "cpu")
      << "\""
      << ",\"attention_layer_timing_summary\":"
      << attention_layer_timing_summary_json(r)
      << ",\"attention_backends_per_layer\":"
      << attention_backends_json(r)
      << "}";
  return out.str();
}

std::mutex& test_array_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::uint64_t, TestArrayRecord>& test_array_table() {
  static std::unordered_map<std::uint64_t, TestArrayRecord> table;
  return table;
}

std::mutex& token_array_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::uint64_t, TokenArrayRecord>& token_array_table() {
  static std::unordered_map<std::uint64_t, TokenArrayRecord> table;
  return table;
}

std::mutex& array_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::uint64_t, ArrayRecord>& array_table() {
  static std::unordered_map<std::uint64_t, ArrayRecord> table;
  return table;
}

std::vector<ArrayRecord>& retired_array_records() {
  static std::vector<ArrayRecord> records;
  return records;
}

std::mutex& tensor_group_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, TensorGroupRecord>& tensor_group_table() {
  static std::unordered_map<std::string, TensorGroupRecord> table;
  return table;
}

std::mutex& embedding_group_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, EmbeddingGroupRecord>& embedding_group_table() {
  static std::unordered_map<std::string, EmbeddingGroupRecord> table;
  return table;
}

std::mutex& layer_group_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, LayerGroupsRecord>& layer_group_table() {
  static std::unordered_map<std::string, LayerGroupsRecord> table;
  return table;
}

std::mutex& native_kv_cache_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, NativeSessionKvCache>& native_kv_cache_table() {
  static std::unordered_map<std::string, NativeSessionKvCache> table;
  return table;
}

std::mutex& resident_model_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, ResidentModelLayers>& resident_model_table() {
  static std::unordered_map<std::string, ResidentModelLayers> table;
  return table;
}

std::uint64_t next_test_array_handle() {
  static std::uint64_t next = 1;
  return next++;
}

std::uint64_t next_array_handle() {
  static std::uint64_t next = 1001;
  return next++;
}

std::string next_tensor_group_handle() {
  static std::uint64_t next = 1;
  return "ltgrp:" + std::to_string(next++);
}

std::string json_escape(const std::string& input) {
  std::ostringstream out;
  for (char ch : input) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

mlx::core::array make_test_array() {
  return mlx::core::array(7.0f);
}

mlx::core::array make_token_array(const std::uint64_t* tokens, std::size_t length) {
  if (tokens == nullptr && length != 0) {
    throw std::runtime_error("token buffer is null");
  }
  return mlx::core::array(
      reinterpret_cast<const int64_t*>(tokens),
      mlx::core::Shape{static_cast<int>(length)});
}

std::string shape_to_json(const mlx::core::Shape& shape) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << shape[i];
  }
  out << "]";
  return out.str();
}

std::string json_array_u64_to_string(const std::vector<std::uint64_t>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << values[i];
  }
  out << "]";
  return out.str();
}

std::string json_array_int_to_string(const std::vector<int>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << values[i];
  }
  out << "]";
  return out.str();
}

std::string json_array_float_to_string(const std::vector<float>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << values[i];
  }
  out << "]";
  return out.str();
}

std::string json_array_double_to_string(const std::vector<double>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << values[i];
  }
  out << "]";
  return out.str();
}

std::string json_array_string_to_string(const std::vector<std::string>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << "\"" << json_escape(values[i]) << "\"";
  }
  out << "]";
  return out.str();
}

std::string cache_policy_to_json(const CachePolicy& policy) {
  std::ostringstream out;
  out << "{";
  switch (policy.kind) {
    case CachePolicyKind::Full:
      out << "\"kind\":\"Full\","
          << "\"step\":" << policy.step;
      break;
    case CachePolicyKind::Rotating:
      out << "\"kind\":\"Rotating\","
          << "\"max_size\":" << policy.max_size << ","
          << "\"keep\":" << policy.keep << ","
          << "\"step\":" << policy.step << ","
          << "\"implemented\":false";
      break;
    case CachePolicyKind::Quantized:
      out << "\"kind\":\"Quantized\","
          << "\"bits\":" << policy.bits << ","
          << "\"group_size\":" << policy.group_size << ","
          << "\"start_at\":" << policy.start_at << ","
          << "\"base\":\"" << json_escape(policy.base) << "\","
          << "\"implemented\":false";
      break;
    case CachePolicyKind::Recompute:
      out << "\"kind\":\"Recompute\","
          << "\"window\":" << policy.window << ","
          << "\"diagnostic_only\":true,"
          << "\"implemented\":false";
      break;
  }
  out << "}";
  return out.str();
}

std::string json_top_logits_to_string(const std::vector<std::pair<std::uint64_t, float>>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << "{\"token_id\":" << values[i].first << ",\"score\":" << values[i].second << "}";
  }
  out << "]";
  return out.str();
}

std::string json_array_u8_to_string(const std::vector<std::uint8_t>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << static_cast<unsigned>(values[i]);
  }
  out << "]";
  return out.str();
}

std::uint32_t read_u32_le_from_bytes(const std::vector<std::uint8_t>& bytes, std::size_t index) {
  std::size_t offset = index * 4;
  if (offset + 4 > bytes.size()) {
    return 0;
  }
  return static_cast<std::uint32_t>(bytes[offset]) |
         (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
         (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
         (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

std::uint16_t read_u16_le_from_bytes(const std::vector<std::uint8_t>& bytes, std::size_t index) {
  std::size_t offset = index * 2;
  if (offset + 2 > bytes.size()) {
    return 0;
  }
  return static_cast<std::uint16_t>(bytes[offset]) |
         (static_cast<std::uint16_t>(bytes[offset + 1]) << 8);
}

std::uint32_t read_u32_le_unchecked(const std::uint8_t* bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8) |
         (static_cast<std::uint32_t>(bytes[2]) << 16) |
         (static_cast<std::uint32_t>(bytes[3]) << 24);
}

std::uint16_t read_u16_le_unchecked(const std::uint8_t* bytes) {
  return static_cast<std::uint16_t>(bytes[0]) |
         (static_cast<std::uint16_t>(bytes[1]) << 8);
}

std::string json_escape_newlines(const std::string& input) {
  std::ostringstream out;
  for (char ch : input) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

bool read_file_to_string(const std::string& path, std::string& output) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  output = buffer.str();
  return true;
}

std::unordered_map<std::string, std::string>& file_text_cache() {
  static std::unordered_map<std::string, std::string> cache;
  return cache;
}

std::mutex& file_text_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

bool read_file_to_string_cached(const std::string& path, std::string& output) {
  {
    std::lock_guard<std::mutex> lock(file_text_cache_mutex());
    auto it = file_text_cache().find(path);
    if (it != file_text_cache().end()) {
      output = it->second;
      return true;
    }
  }
  std::string loaded;
  if (!read_file_to_string(path, loaded)) {
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(file_text_cache_mutex());
    auto inserted = file_text_cache().emplace(path, std::move(loaded));
    output = inserted.first->second;
  }
  return true;
}

struct SafetensorHeaderCacheEntry {
  std::uint64_t header_len = 0;
  std::string header_json;
};

std::unordered_map<std::string, SafetensorHeaderCacheEntry>& safetensor_header_cache() {
  static std::unordered_map<std::string, SafetensorHeaderCacheEntry> cache;
  return cache;
}

std::mutex& safetensor_header_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

bool read_safetensor_header_cached(
    const std::string& shard_path,
    std::uint64_t& header_len,
    std::string& header_json) {
  {
    std::lock_guard<std::mutex> lock(safetensor_header_cache_mutex());
    auto it = safetensor_header_cache().find(shard_path);
    if (it != safetensor_header_cache().end()) {
      header_len = it->second.header_len;
      header_json = it->second.header_json;
      return true;
    }
  }
  std::ifstream shard_file(shard_path, std::ios::binary);
  if (!shard_file.is_open()) {
    return false;
  }
  std::uint8_t len_bytes[8] = {0};
  shard_file.read(reinterpret_cast<char*>(len_bytes), 8);
  if (shard_file.gcount() != 8) {
    return false;
  }
  header_len = read_u64_le(len_bytes);
  std::string loaded_header(header_len, '\0');
  shard_file.read(loaded_header.data(), static_cast<std::streamsize>(header_len));
  if (static_cast<std::uint64_t>(shard_file.gcount()) != header_len) {
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(safetensor_header_cache_mutex());
    auto inserted = safetensor_header_cache().emplace(
        shard_path,
        SafetensorHeaderCacheEntry{header_len, std::move(loaded_header)});
    header_json = inserted.first->second.header_json;
  }
  return true;
}

std::unordered_map<std::string, std::weak_ptr<const MappedFile>>& mapped_file_cache() {
  static std::unordered_map<std::string, std::weak_ptr<const MappedFile>> cache;
  return cache;
}

std::mutex& mapped_file_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::shared_ptr<const MappedFile> mapped_file_for_path(const std::string& path) {
  {
    std::lock_guard<std::mutex> lock(mapped_file_cache_mutex());
    auto it = mapped_file_cache().find(path);
    if (it != mapped_file_cache().end()) {
      auto existing = it->second.lock();
      if (existing) {
        return existing;
      }
    }
  }

  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("failed to open safetensors shard for mmap");
  }
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    close(fd);
    throw std::runtime_error("failed to stat safetensors shard for mmap");
  }
  void* mapped = mmap(nullptr, static_cast<std::size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) {
    close(fd);
    throw std::runtime_error("failed to mmap safetensors shard");
  }
  auto file = std::make_shared<MappedFile>(
      fd,
      static_cast<std::size_t>(st.st_size),
      static_cast<const std::uint8_t*>(mapped));
  {
    std::lock_guard<std::mutex> lock(mapped_file_cache_mutex());
    mapped_file_cache()[path] = file;
  }
  return file;
}

std::string join_path(const std::string& base, const std::string& leaf);
bool extract_json_string_value(
    const std::string& object,
    const std::string& key,
    std::string& output);
bool extract_json_u64_value(
    const std::string& object,
    const std::string& key,
    std::uint64_t& output);
bool extract_json_double_value(
    const std::string& object,
    const std::string& key,
    double& output);
bool extract_json_array_u64(
    const std::string& object,
    const std::string& key,
    std::vector<std::uint64_t>& output);
std::string tensor_object_for_key(const std::string& json, const std::string& key);

GroupTensorRecord load_single_safetensor_tensor(
    const std::string& safetensor_path,
    const std::string& tensor_name,
    const std::shared_ptr<const MappedFile>& mapped_file) {
  std::uint64_t header_len = 0;
  std::string header_json;
  if (!read_safetensor_header_cached(safetensor_path, header_len, header_json)) {
    throw std::runtime_error("failed to read adapter safetensors header");
  }
  const std::string object = tensor_object_for_key(header_json, tensor_name);
  if (object.empty()) {
    throw std::runtime_error("adapter tensor missing: " + tensor_name);
  }
  GroupTensorRecord tensor;
  tensor.kind = tensor_name;
  tensor.source_file = safetensor_path;
  tensor.mapped_file = mapped_file;
  if (!extract_json_string_value(object, "dtype", tensor.dtype)) {
    throw std::runtime_error("adapter tensor missing dtype: " + tensor_name);
  }
  if (!extract_json_array_u64(object, "shape", tensor.shape)) {
    throw std::runtime_error("adapter tensor missing shape: " + tensor_name);
  }
  std::vector<std::uint64_t> offsets;
  if (!extract_json_array_u64(object, "data_offsets", offsets) || offsets.size() != 2) {
    throw std::runtime_error("adapter tensor missing data_offsets: " + tensor_name);
  }
  tensor.byte_start = offsets[0];
  tensor.byte_end = offsets[1];
  const std::uint64_t absolute_start = 8 + header_len + tensor.byte_start;
  const std::uint64_t absolute_end = 8 + header_len + tensor.byte_end;
  if (absolute_end > mapped_file->size || absolute_start > absolute_end) {
    throw std::runtime_error("adapter tensor offsets out of range: " + tensor_name);
  }
  tensor.payload_view = mapped_file->data + absolute_start;
  tensor.payload_view_size = static_cast<std::size_t>(absolute_end - absolute_start);
  return tensor;
}

mlx::core::array mlx_f32_array_from_adapter_tensor(const GroupTensorRecord& tensor) {
  if (tensor.dtype != "F32" || tensor.shape.size() != 2) {
    throw std::runtime_error("adapter tensor must be F32 matrix: " + tensor.kind);
  }
  const std::size_t expected_bytes =
      static_cast<std::size_t>(tensor.shape[0] * tensor.shape[1] * sizeof(float));
  if (tensor.payload_view_size != expected_bytes || tensor.payload_view == nullptr) {
    throw std::runtime_error("adapter tensor byte size mismatch: " + tensor.kind);
  }
  return mlx::core::array(
      reinterpret_cast<const float*>(tensor.payload_view),
      mlx::core::Shape{static_cast<int>(tensor.shape[0]), static_cast<int>(tensor.shape[1])},
      mlx::core::float32);
}

LoraProjectionRecord load_lora_projection(
    const std::string& safetensor_path,
    const std::shared_ptr<const MappedFile>& mapped_file,
    const std::string& prefix,
    std::uint64_t expected_in,
    std::uint64_t expected_out,
    std::uint64_t rank) {
  LoraProjectionRecord projection;
  projection.name = prefix;
  GroupTensorRecord a_tensor = load_single_safetensor_tensor(
      safetensor_path, prefix + ".lora_a", mapped_file);
  GroupTensorRecord b_tensor = load_single_safetensor_tensor(
      safetensor_path, prefix + ".lora_b", mapped_file);
  if (a_tensor.dtype != "F32" || b_tensor.dtype != "F32") {
    throw std::runtime_error("adapter LoRA tensors must be F32: " + prefix);
  }
  if (a_tensor.shape.size() != 2 || b_tensor.shape.size() != 2 ||
      a_tensor.shape[0] != expected_in ||
      a_tensor.shape[1] != rank ||
      b_tensor.shape[0] != rank ||
      b_tensor.shape[1] != expected_out) {
    throw std::runtime_error("adapter LoRA tensor shape mismatch: " + prefix);
  }
  projection.a_shape = a_tensor.shape;
  projection.b_shape = b_tensor.shape;
  projection.a.emplace(mlx_f32_array_from_adapter_tensor(a_tensor));
  projection.b.emplace(mlx_f32_array_from_adapter_tensor(b_tensor));
  projection.present = true;
  return projection;
}

std::shared_ptr<NativeAdapterRecord> load_native_adapter_record(const std::string& adapter_dir) {
  if (adapter_dir.empty()) {
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(native_adapter_mutex());
    auto it = native_adapter_cache().find(adapter_dir);
    if (it != native_adapter_cache().end()) {
      return it->second;
    }
  }
  const std::string config_path = join_path(adapter_dir, "adapter_config.json");
  const std::string safetensor_path = join_path(adapter_dir, "adapters.safetensors");
  std::string config_json;
  if (!read_file_to_string(config_path, config_json)) {
    throw std::runtime_error("adapter_config.json not readable");
  }
  auto adapter = std::make_shared<NativeAdapterRecord>();
  adapter->adapter_dir = adapter_dir;
  if (!extract_json_u64_value(config_json, "rank", adapter->rank) || adapter->rank == 0) {
    throw std::runtime_error("adapter rank missing");
  }
  if (!extract_json_double_value(config_json, "scale", adapter->scale)) {
    throw std::runtime_error("adapter scale missing");
  }
  adapter->mapped_file = mapped_file_for_path(safetensor_path);

  const std::array<std::string, 7> targets = {
      "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"};
  adapter->targets.assign(targets.begin(), targets.end());
  adapter->layers.clear();
  for (int layer = 20; layer <= 35; ++layer) {
    LoraLayerRecord layer_record;
    layer_record.layer = layer;
    const std::string base = "model.layers." + std::to_string(layer) + ".";
    layer_record.q_proj = load_lora_projection(
        safetensor_path, adapter->mapped_file, base + "self_attn.q_proj", 2560, 4096, adapter->rank);
    layer_record.k_proj = load_lora_projection(
        safetensor_path, adapter->mapped_file, base + "self_attn.k_proj", 2560, 1024, adapter->rank);
    layer_record.v_proj = load_lora_projection(
        safetensor_path, adapter->mapped_file, base + "self_attn.v_proj", 2560, 1024, adapter->rank);
    layer_record.o_proj = load_lora_projection(
        safetensor_path, adapter->mapped_file, base + "self_attn.o_proj", 4096, 2560, adapter->rank);
    layer_record.gate_proj = load_lora_projection(
        safetensor_path, adapter->mapped_file, base + "mlp.gate_proj", 2560, 9728, adapter->rank);
    layer_record.up_proj = load_lora_projection(
        safetensor_path, adapter->mapped_file, base + "mlp.up_proj", 2560, 9728, adapter->rank);
    layer_record.down_proj = load_lora_projection(
        safetensor_path, adapter->mapped_file, base + "mlp.down_proj", 9728, 2560, adapter->rank);
    adapter->layer_records[layer] = std::move(layer_record);
    adapter->layers.push_back(layer);
  }
  adapter->tensor_count = 224;
  {
    std::lock_guard<std::mutex> lock(native_adapter_mutex());
    native_adapter_cache()[adapter_dir] = adapter;
  }
  return adapter;
}

const LoraLayerRecord* adapter_layer_for(std::size_t layer) {
  if (!active_generation_adapter) {
    return nullptr;
  }
  auto it = active_generation_adapter->layer_records.find(static_cast<int>(layer));
  if (it == active_generation_adapter->layer_records.end()) {
    return nullptr;
  }
  return &it->second;
}

mlx::core::array apply_lora_delta_if_present(
    const mlx::core::array& base_output,
    const mlx::core::array& input_array,
    const LoraProjectionRecord* projection) {
  if (projection == nullptr || !projection->present) {
    return base_output;
  }
  auto low_rank = mlx::core::matmul(input_array, *projection->a);
  auto delta = mlx::core::matmul(low_rank, *projection->b) *
      static_cast<float>(active_generation_adapter ? active_generation_adapter->scale : 1.0);
  return base_output + delta;
}

bool read_file_range_to_bytes(
    const std::string& path,
    std::uint64_t offset,
    std::uint64_t length,
    std::vector<std::uint8_t>& output) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }
  file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  if (!file.good()) {
    return false;
  }
  output.resize(static_cast<std::size_t>(length));
  file.read(reinterpret_cast<char*>(output.data()), static_cast<std::streamsize>(length));
  return static_cast<std::uint64_t>(file.gcount()) == length;
}

const std::uint8_t* tensor_payload_data(const GroupTensorRecord& tensor) {
  if (tensor.payload_view != nullptr) {
    return tensor.payload_view;
  }
  return tensor.payload.data();
}

std::size_t tensor_payload_size(const GroupTensorRecord& tensor) {
  if (tensor.payload_view != nullptr) {
    return tensor.payload_view_size;
  }
  return tensor.payload.size();
}

std::string strip_quotes(const std::string& input) {
  if (input.size() >= 2 && input.front() == '"' && input.back() == '"') {
    return input.substr(1, input.size() - 2);
  }
  return input;
}

bool file_exists(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  return file.good();
}

std::string parent_dir(const std::string& path) {
  auto pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return ".";
  }
  return path.substr(0, pos);
}

std::string join_path(const std::string& base, const std::string& leaf) {
  if (base.empty()) {
    return leaf;
  }
  if (base.back() == '/' || base.back() == '\\') {
    return base + leaf;
  }
  return base + "/" + leaf;
}

bool string_ends_with(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool extract_json_string_value(
    const std::string& object,
    const std::string& key,
    std::string& output) {
  auto key_pos = object.find("\"" + key + "\"");
  if (key_pos == std::string::npos) {
    return false;
  }
  auto colon = object.find(':', key_pos);
  if (colon == std::string::npos) {
    return false;
  }
  auto first_quote = object.find('"', colon + 1);
  if (first_quote == std::string::npos) {
    return false;
  }
  auto second_quote = object.find('"', first_quote + 1);
  if (second_quote == std::string::npos) {
    return false;
  }
  output = object.substr(first_quote + 1, second_quote - first_quote - 1);
  return true;
}

bool extract_json_u64_value(
    const std::string& object,
    const std::string& key,
    std::uint64_t& output) {
  auto key_pos = object.find("\"" + key + "\"");
  if (key_pos == std::string::npos) {
    return false;
  }
  auto colon = object.find(':', key_pos);
  if (colon == std::string::npos) {
    return false;
  }
  auto start = object.find_first_of("0123456789", colon + 1);
  if (start == std::string::npos) {
    return false;
  }
  auto end = object.find_first_not_of("0123456789", start);
  try {
    output = std::stoull(object.substr(start, end - start));
    return true;
  } catch (...) {
    return false;
  }
}

bool extract_json_double_value(
    const std::string& object,
    const std::string& key,
    double& output) {
  auto key_pos = object.find("\"" + key + "\"");
  if (key_pos == std::string::npos) {
    return false;
  }
  auto colon = object.find(':', key_pos);
  if (colon == std::string::npos) {
    return false;
  }
  auto start = object.find_first_of("0123456789-+.", colon + 1);
  if (start == std::string::npos) {
    return false;
  }
  auto end = object.find_first_not_of("0123456789eE-+.", start);
  try {
    output = std::stod(object.substr(start, end - start));
    return true;
  } catch (...) {
    return false;
  }
}

bool extract_json_array_u64(
    const std::string& object,
    const std::string& key,
    std::vector<std::uint64_t>& output) {
  auto key_pos = object.find("\"" + key + "\"");
  if (key_pos == std::string::npos) {
    return false;
  }
  auto open = object.find('[', key_pos);
  if (open == std::string::npos) {
    return false;
  }
  auto close = object.find(']', open);
  if (close == std::string::npos) {
    return false;
  }
  std::string body = object.substr(open + 1, close - open - 1);
  std::istringstream input(body);
  std::string token;
  output.clear();
  while (std::getline(input, token, ',')) {
    if (token.empty()) continue;
    try {
      output.push_back(std::stoull(token));
    } catch (...) {
      return false;
    }
  }
  return true;
}

std::string tensor_object_for_key(const std::string& json, const std::string& key) {
  auto key_pos = json.find("\"" + key + "\"");
  if (key_pos == std::string::npos) {
    return {};
  }
  auto open = json.find('{', key_pos);
  if (open == std::string::npos) {
    return {};
  }
  int depth = 0;
  for (std::size_t i = open; i < json.size(); ++i) {
    char ch = json[i];
    if (ch == '{') {
      ++depth;
    } else if (ch == '}') {
      --depth;
      if (depth == 0) {
        return json.substr(open, i - open + 1);
      }
    }
  }
  return {};
}

std::string index_shard_for_tensor(
    const std::string& index_json,
    const std::string& tensor_name) {
  auto key_pos = index_json.find("\"" + tensor_name + "\"");
  if (key_pos == std::string::npos) {
    return {};
  }
  auto colon = index_json.find(':', key_pos);
  if (colon == std::string::npos) {
    return {};
  }
  auto first_quote = index_json.find('"', colon + 1);
  if (first_quote == std::string::npos) {
    return {};
  }
  auto second_quote = index_json.find('"', first_quote + 1);
  if (second_quote == std::string::npos) {
    return {};
  }
  return index_json.substr(first_quote + 1, second_quote - first_quote - 1);
}

mlx::core::array make_logits_array(std::size_t token_count) {
  return mlx::core::array({
      static_cast<float>(token_count),
      static_cast<float>(token_count + 1),
  });
}

std::vector<float> provisional_dequant_slice_values(
    const TensorGroupRecord& record,
    std::size_t row,
    std::size_t cols) {
  const auto weight_it = record.tensors.find("weight");
  const auto scales_it = record.tensors.find("scales");
  const auto biases_it = record.tensors.find("biases");
  if (weight_it == record.tensors.end() ||
      scales_it == record.tensors.end() ||
      biases_it == record.tensors.end()) {
    throw std::runtime_error("quantized tensor group is missing weight/scales/biases");
  }

  const auto& weight = weight_it->second.payload;
  const auto& scales = scales_it->second.payload;
  const auto& biases = biases_it->second.payload;
  const std::size_t weight_rows = weight_it->second.shape.size() > 0
      ? static_cast<std::size_t>(weight_it->second.shape[0])
      : 0;
  const std::size_t weight_cols = weight_it->second.shape.size() > 1
      ? static_cast<std::size_t>(weight_it->second.shape[1])
      : 0;
  const std::size_t scale_cols = scales_it->second.shape.size() > 1
      ? static_cast<std::size_t>(scales_it->second.shape[1])
      : 0;
  if (weight_rows == 0 || weight_cols == 0 || scale_cols == 0) {
    throw std::runtime_error("quantized tensor group has invalid shape metadata");
  }
  if (row >= weight_rows) {
    throw std::runtime_error("requested row is out of range for quantized tensor group");
  }
  if (cols == 0) {
    throw std::runtime_error("requested slice must have at least one column");
  }

  auto bf16_to_float = [](std::uint16_t word) {
    std::uint32_t bits = static_cast<std::uint32_t>(word) << 16;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };

  const std::size_t words_per_row = weight_cols;
  const std::size_t scales_per_row = scale_cols;
  const std::size_t biases_per_row = scale_cols;
  std::vector<float> values(cols, 0.0f);

  for (std::size_t col = 0; col < cols; ++col) {
    const std::size_t word_index = col / 8;
    const std::size_t nibble_index = col % 8;
    const std::size_t weight_index = row * words_per_row + word_index;
    const std::size_t scale_index = row * scales_per_row + (word_index / 8);
    const std::size_t bias_index = row * biases_per_row + (word_index / 8);

    const std::uint32_t weight_word = read_u32_le_from_bytes(weight, weight_index);
    const std::uint16_t scale_word = read_u16_le_from_bytes(scales, scale_index);
    const std::uint16_t bias_word = read_u16_le_from_bytes(biases, bias_index);
    const std::uint32_t nibble = (weight_word >> (static_cast<std::uint32_t>(nibble_index) * 4U)) & 0xFU;
    const float scale = bf16_to_float(scale_word);
    const float bias = bf16_to_float(bias_word);
    values[col] = static_cast<float>(nibble) * scale + bias;
  }

  return values;
}

std::vector<float> load_bf16_vector_from_tensor(const GroupTensorRecord& tensor) {
  std::vector<float> values;
  const std::uint8_t* payload = tensor_payload_data(tensor);
  const std::size_t payload_size = tensor_payload_size(tensor);
  if (payload == nullptr || payload_size == 0) {
    return values;
  }
  if (payload_size % 2 != 0) {
    return values;
  }
  values.reserve(payload_size / 2);
  for (std::size_t i = 0; i + 1 < payload_size; i += 2) {
    const std::uint16_t word = static_cast<std::uint16_t>(payload[i]) |
                               (static_cast<std::uint16_t>(payload[i + 1]) << 8);
    std::uint32_t bits = static_cast<std::uint32_t>(word) << 16;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    values.push_back(value);
  }
  return values;
}

double provisional_fullrow_dot_value(
    const TensorGroupRecord& record,
    std::size_t row,
    const std::vector<float>& input_values) {
  const auto weight_it = record.tensors.find("weight");
  const auto scales_it = record.tensors.find("scales");
  const auto biases_it = record.tensors.find("biases");
  if (weight_it == record.tensors.end() ||
      scales_it == record.tensors.end() ||
      biases_it == record.tensors.end()) {
    throw std::runtime_error("quantized tensor group is missing weight/scales/biases");
  }

  const auto& weight = weight_it->second.payload;
  const auto& scales = scales_it->second.payload;
  const auto& biases = biases_it->second.payload;
  const std::size_t weight_rows = weight_it->second.shape.size() > 0
      ? static_cast<std::size_t>(weight_it->second.shape[0])
      : 0;
  const std::size_t weight_cols = weight_it->second.shape.size() > 1
      ? static_cast<std::size_t>(weight_it->second.shape[1])
      : 0;
  const std::size_t scale_cols = scales_it->second.shape.size() > 1
      ? static_cast<std::size_t>(scales_it->second.shape[1])
      : 0;
  if (weight_rows == 0 || weight_cols == 0 || scale_cols == 0) {
    throw std::runtime_error("quantized tensor group has invalid shape metadata");
  }
  if (row >= weight_rows) {
    throw std::runtime_error("requested row is out of range for quantized tensor group");
  }

  const std::size_t logical_width = weight_cols * 8;
  if (input_values.size() != logical_width) {
    throw std::runtime_error("quantized full-row probe requires input length to equal logical input width");
  }

  auto bf16_to_float = [](std::uint16_t word) {
    std::uint32_t bits = static_cast<std::uint32_t>(word) << 16;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };

  const std::size_t words_per_row = weight_cols;
  const std::size_t scale_blocks_per_row = scale_cols;
  double dot = 0.0;

  for (std::size_t word_index = 0; word_index < words_per_row; ++word_index) {
    const std::size_t block_index = word_index / 8;
    const std::size_t weight_index = row * words_per_row + word_index;
    const std::size_t scale_index = row * scale_blocks_per_row + block_index;
    const std::size_t bias_index = row * scale_blocks_per_row + block_index;

    const std::uint32_t weight_word = read_u32_le_from_bytes(weight, weight_index);
    const std::uint16_t scale_word = read_u16_le_from_bytes(scales, scale_index);
    const std::uint16_t bias_word = read_u16_le_from_bytes(biases, bias_index);
    const float scale = bf16_to_float(scale_word);
    const float bias = bf16_to_float(bias_word);

    const std::size_t input_base = word_index * 8;
    for (std::size_t nibble_index = 0; nibble_index < 8; ++nibble_index) {
      const std::size_t input_index = input_base + nibble_index;
      const std::uint32_t nibble =
          (weight_word >> (static_cast<std::uint32_t>(nibble_index) * 4U)) & 0xFU;
      const float value = static_cast<float>(nibble) * scale + bias;
      dot += static_cast<double>(input_values[input_index]) * static_cast<double>(value);
    }
  }

  return dot;
}

double vector_checksum(const std::vector<float>& values) {
  double checksum = 0.0;
  for (float value : values) {
    checksum += static_cast<double>(value);
  }
  return checksum;
}

double vector_max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  const std::size_t count = std::min(lhs.size(), rhs.size());
  double diff = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    diff = std::max(
        diff,
        std::abs(static_cast<double>(lhs[i]) - static_cast<double>(rhs[i])));
  }
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<double>::infinity();
  }
  return diff;
}

float bf16_to_float(std::uint16_t word) {
  std::uint32_t bits = static_cast<std::uint32_t>(word) << 16;
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::vector<float> first_values_of(const std::vector<float>& values, std::size_t count = 8) {
  const std::size_t first_count = std::min<std::size_t>(count, values.size());
  return std::vector<float>(values.begin(), values.begin() + first_count);
}

std::string parity_vector_summary_json(const std::vector<float>& values) {
  std::ostringstream out;
  out << "{"
      << "\"shape\":[" << values.size() << "],"
      << "\"checksum\":" << vector_checksum(values) << ","
      << "\"first_values\":" << json_array_float_to_string(first_values_of(values))
      << "}";
  return out.str();
}

void cache_quantized_linear_layout(TensorGroupRecord& record) {
  record.quantized_layout_cached = false;
  record.quantized_rows = 0;
  record.quantized_packed_cols = 0;
  record.quantized_scale_cols = 0;
  record.quantized_logical_width = 0;
  record.quantized_block_size = 8;
  record.quantized_output_len = 0;
  if (!record.quantized_group) {
    return;
  }
  const auto weight_it = record.tensors.find("weight");
  const auto scales_it = record.tensors.find("scales");
  const auto biases_it = record.tensors.find("biases");
  if (weight_it == record.tensors.end() ||
      scales_it == record.tensors.end() ||
      biases_it == record.tensors.end() ||
      weight_it->second.shape.size() < 2 ||
      scales_it->second.shape.size() < 2 ||
      biases_it->second.shape.size() < 2) {
    return;
  }
  record.quantized_rows = static_cast<std::size_t>(weight_it->second.shape[0]);
  record.quantized_packed_cols = static_cast<std::size_t>(weight_it->second.shape[1]);
  record.quantized_scale_cols = static_cast<std::size_t>(scales_it->second.shape[1]);
  record.quantized_logical_width = record.quantized_packed_cols * record.quantized_block_size;
  record.quantized_output_len = record.quantized_rows;
  record.quantized_layout_cached =
      record.quantized_rows > 0 &&
      record.quantized_packed_cols > 0 &&
      record.quantized_scale_cols > 0 &&
      scales_it->second.shape[0] == record.quantized_rows &&
      biases_it->second.shape[0] == record.quantized_rows &&
      biases_it->second.shape[1] == record.quantized_scale_cols;
}

std::vector<float> quantized_linear_vector_values(
    const TensorGroupRecord& record,
    const std::vector<float>& input_values) {
  const auto weight_it = record.tensors.find("weight");
  if (weight_it == record.tensors.end() || weight_it->second.shape.empty()) {
    throw std::runtime_error("quantized linear vector probe requires weight metadata");
  }
  const std::size_t rows = static_cast<std::size_t>(weight_it->second.shape[0]);
  std::vector<float> output_values;
  output_values.reserve(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    output_values.push_back(
        static_cast<float>(provisional_fullrow_dot_value(record, row, input_values)));
  }
  return output_values;
}

std::vector<float> quantized_linear_vector_values_row_optimized(
    const TensorGroupRecord& record,
    const std::vector<float>& input_values) {
  const auto weight_it = record.tensors.find("weight");
  const auto scales_it = record.tensors.find("scales");
  const auto biases_it = record.tensors.find("biases");
  if (weight_it == record.tensors.end() ||
      scales_it == record.tensors.end() ||
      biases_it == record.tensors.end()) {
    throw std::runtime_error("optimized quantized linear requires weight/scales/biases");
  }

  const auto& weight = weight_it->second.payload;
  const auto& scales = scales_it->second.payload;
  const auto& biases = biases_it->second.payload;
  const std::size_t rows = weight_it->second.shape.size() > 0
      ? static_cast<std::size_t>(weight_it->second.shape[0])
      : 0;
  const std::size_t weight_cols = weight_it->second.shape.size() > 1
      ? static_cast<std::size_t>(weight_it->second.shape[1])
      : 0;
  const std::size_t scale_cols = scales_it->second.shape.size() > 1
      ? static_cast<std::size_t>(scales_it->second.shape[1])
      : 0;
  if (rows == 0 || weight_cols == 0 || scale_cols == 0) {
    throw std::runtime_error("optimized quantized linear has invalid shape metadata");
  }
  const std::size_t logical_width = weight_cols * 8;
  if (input_values.size() != logical_width) {
    throw std::runtime_error("optimized quantized linear input length mismatch");
  }

  auto bf16_to_float = [](std::uint16_t word) {
    std::uint32_t bits = static_cast<std::uint32_t>(word) << 16;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };

  std::vector<float> output_values(rows, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    double dot = 0.0;
    const std::size_t row_weight_base = row * weight_cols;
    const std::size_t row_scale_base = row * scale_cols;
    for (std::size_t block = 0; block < scale_cols; ++block) {
      const std::uint16_t scale_word = read_u16_le_from_bytes(scales, row_scale_base + block);
      const std::uint16_t bias_word = read_u16_le_from_bytes(biases, row_scale_base + block);
      const double scale = static_cast<double>(bf16_to_float(scale_word));
      const double bias = static_cast<double>(bf16_to_float(bias_word));
      const std::size_t word_begin = block * 8;
      const std::size_t word_end = std::min<std::size_t>(word_begin + 8, weight_cols);
      for (std::size_t word_index = word_begin; word_index < word_end; ++word_index) {
        const std::uint32_t weight_word =
            read_u32_le_from_bytes(weight, row_weight_base + word_index);
        const std::size_t input_base = word_index * 8;
        dot += static_cast<double>(input_values[input_base + 0]) *
               (static_cast<double>((weight_word >> 0U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 1]) *
               (static_cast<double>((weight_word >> 4U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 2]) *
               (static_cast<double>((weight_word >> 8U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 3]) *
               (static_cast<double>((weight_word >> 12U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 4]) *
               (static_cast<double>((weight_word >> 16U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 5]) *
               (static_cast<double>((weight_word >> 20U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 6]) *
               (static_cast<double>((weight_word >> 24U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 7]) *
               (static_cast<double>((weight_word >> 28U) & 0xFU) * scale + bias);
      }
    }
    output_values[row] = static_cast<float>(dot);
  }
  return output_values;
}

std::vector<float> quantized_linear_vector_values_layout_cached(
    const TensorGroupRecord& record,
    const std::vector<float>& input_values) {
  if (!record.quantized_layout_cached) {
    return quantized_linear_vector_values_row_optimized(record, input_values);
  }
  const auto weight_it = record.tensors.find("weight");
  const auto scales_it = record.tensors.find("scales");
  const auto biases_it = record.tensors.find("biases");
  if (weight_it == record.tensors.end() ||
      scales_it == record.tensors.end() ||
      biases_it == record.tensors.end()) {
    throw std::runtime_error("layout-cached quantized linear requires weight/scales/biases");
  }

  const std::size_t rows = record.quantized_rows;
  const std::size_t weight_cols = record.quantized_packed_cols;
  const std::size_t scale_cols = record.quantized_scale_cols;
  if (input_values.size() != record.quantized_logical_width) {
    throw std::runtime_error("layout-cached quantized linear input length mismatch");
  }

  const std::uint8_t* weight = tensor_payload_data(weight_it->second);
  const std::uint8_t* scales = tensor_payload_data(scales_it->second);
  const std::uint8_t* biases = tensor_payload_data(biases_it->second);
  std::vector<float> output_values(rows, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    double dot = 0.0;
    const std::size_t row_weight_byte_base = row * weight_cols * 4;
    const std::size_t row_scale_byte_base = row * scale_cols * 2;
    for (std::size_t block = 0; block < scale_cols; ++block) {
      const std::uint16_t scale_word =
          read_u16_le_unchecked(scales + row_scale_byte_base + (block * 2));
      const std::uint16_t bias_word =
          read_u16_le_unchecked(biases + row_scale_byte_base + (block * 2));
      const double scale = static_cast<double>(bf16_to_float(scale_word));
      const double bias = static_cast<double>(bf16_to_float(bias_word));
      const std::size_t word_begin = block * 8;
      const std::size_t word_limit = std::min<std::size_t>(word_begin + 8, weight_cols);
      for (std::size_t word_index = word_begin; word_index < word_limit; ++word_index) {
        const std::uint32_t weight_word =
            read_u32_le_unchecked(weight + row_weight_byte_base + (word_index * 4));
        const std::size_t input_base = word_index * 8;
        dot += static_cast<double>(input_values[input_base + 0]) *
               (static_cast<double>((weight_word >> 0U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 1]) *
               (static_cast<double>((weight_word >> 4U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 2]) *
               (static_cast<double>((weight_word >> 8U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 3]) *
               (static_cast<double>((weight_word >> 12U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 4]) *
               (static_cast<double>((weight_word >> 16U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 5]) *
               (static_cast<double>((weight_word >> 20U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 6]) *
               (static_cast<double>((weight_word >> 24U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 7]) *
               (static_cast<double>((weight_word >> 28U) & 0xFU) * scale + bias);
      }
    }
    output_values[row] = static_cast<float>(dot);
  }
  return output_values;
}

std::vector<float> quantized_linear_vector_values_layout_cached_full_blocks(
    const TensorGroupRecord& record,
    const std::vector<float>& input_values) {
  if (!record.quantized_layout_cached ||
      record.quantized_scale_cols * 8 != record.quantized_packed_cols) {
    return quantized_linear_vector_values_layout_cached(record, input_values);
  }
  const auto weight_it = record.tensors.find("weight");
  const auto scales_it = record.tensors.find("scales");
  const auto biases_it = record.tensors.find("biases");
  if (weight_it == record.tensors.end() ||
      scales_it == record.tensors.end() ||
      biases_it == record.tensors.end()) {
    throw std::runtime_error("full-block layout-cached quantized linear requires weight/scales/biases");
  }
  if (input_values.size() != record.quantized_logical_width) {
    throw std::runtime_error("full-block layout-cached quantized linear input length mismatch");
  }

  const std::size_t rows = record.quantized_rows;
  const std::size_t weight_cols = record.quantized_packed_cols;
  const std::size_t scale_cols = record.quantized_scale_cols;
  const std::uint8_t* weight = tensor_payload_data(weight_it->second);
  const std::uint8_t* scales = tensor_payload_data(scales_it->second);
  const std::uint8_t* biases = tensor_payload_data(biases_it->second);
  std::vector<float> output_values(rows, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    double dot = 0.0;
    const std::size_t row_weight_byte_base = row * weight_cols * 4;
    const std::size_t row_scale_byte_base = row * scale_cols * 2;
    for (std::size_t block = 0; block < scale_cols; ++block) {
      const std::uint16_t scale_word =
          read_u16_le_unchecked(scales + row_scale_byte_base + (block * 2));
      const std::uint16_t bias_word =
          read_u16_le_unchecked(biases + row_scale_byte_base + (block * 2));
      const double scale = static_cast<double>(bf16_to_float(scale_word));
      const double bias = static_cast<double>(bf16_to_float(bias_word));
      const std::size_t word_begin = block * 8;
      const std::size_t word_end = word_begin + 8;
      for (std::size_t word_index = word_begin; word_index < word_end; ++word_index) {
        const std::uint32_t weight_word =
            read_u32_le_unchecked(weight + row_weight_byte_base + (word_index * 4));
        const std::size_t input_base = word_index * 8;
        dot += static_cast<double>(input_values[input_base + 0]) *
               (static_cast<double>((weight_word >> 0U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 1]) *
               (static_cast<double>((weight_word >> 4U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 2]) *
               (static_cast<double>((weight_word >> 8U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 3]) *
               (static_cast<double>((weight_word >> 12U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 4]) *
               (static_cast<double>((weight_word >> 16U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 5]) *
               (static_cast<double>((weight_word >> 20U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 6]) *
               (static_cast<double>((weight_word >> 24U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 7]) *
               (static_cast<double>((weight_word >> 28U) & 0xFU) * scale + bias);
      }
    }
    output_values[row] = static_cast<float>(dot);
  }
  return output_values;
}

std::pair<std::vector<float>, std::vector<float>> quantized_linear_pair_values_layout_cached(
    const TensorGroupRecord& left,
    const TensorGroupRecord& right,
    const std::vector<float>& input_values) {
  if (!left.quantized_layout_cached || !right.quantized_layout_cached ||
      left.quantized_rows != right.quantized_rows ||
      left.quantized_packed_cols != right.quantized_packed_cols ||
      left.quantized_scale_cols != right.quantized_scale_cols ||
      left.quantized_logical_width != right.quantized_logical_width) {
    return {
        quantized_linear_vector_values_layout_cached(left, input_values),
        quantized_linear_vector_values_layout_cached(right, input_values),
    };
  }
  if (input_values.size() != left.quantized_logical_width) {
    throw std::runtime_error("layout-cached paired quantized linear input length mismatch");
  }

  const auto left_weight_it = left.tensors.find("weight");
  const auto left_scales_it = left.tensors.find("scales");
  const auto left_biases_it = left.tensors.find("biases");
  const auto right_weight_it = right.tensors.find("weight");
  const auto right_scales_it = right.tensors.find("scales");
  const auto right_biases_it = right.tensors.find("biases");
  if (left_weight_it == left.tensors.end() ||
      left_scales_it == left.tensors.end() ||
      left_biases_it == left.tensors.end() ||
      right_weight_it == right.tensors.end() ||
      right_scales_it == right.tensors.end() ||
      right_biases_it == right.tensors.end()) {
    throw std::runtime_error("layout-cached paired quantized linear requires weight/scales/biases");
  }

  const std::size_t rows = left.quantized_rows;
  const std::size_t weight_cols = left.quantized_packed_cols;
  const std::size_t scale_cols = left.quantized_scale_cols;
  const std::uint8_t* left_weight = tensor_payload_data(left_weight_it->second);
  const std::uint8_t* left_scales = tensor_payload_data(left_scales_it->second);
  const std::uint8_t* left_biases = tensor_payload_data(left_biases_it->second);
  const std::uint8_t* right_weight = tensor_payload_data(right_weight_it->second);
  const std::uint8_t* right_scales = tensor_payload_data(right_scales_it->second);
  const std::uint8_t* right_biases = tensor_payload_data(right_biases_it->second);

  std::vector<float> left_output(rows, 0.0f);
  std::vector<float> right_output(rows, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    double left_dot = 0.0;
    double right_dot = 0.0;
    const std::size_t row_weight_byte_base = row * weight_cols * 4;
    const std::size_t row_scale_byte_base = row * scale_cols * 2;
    for (std::size_t block = 0; block < scale_cols; ++block) {
      const std::size_t scale_byte_offset = row_scale_byte_base + (block * 2);
      const double left_scale = static_cast<double>(
          bf16_to_float(read_u16_le_unchecked(left_scales + scale_byte_offset)));
      const double left_bias = static_cast<double>(
          bf16_to_float(read_u16_le_unchecked(left_biases + scale_byte_offset)));
      const double right_scale = static_cast<double>(
          bf16_to_float(read_u16_le_unchecked(right_scales + scale_byte_offset)));
      const double right_bias = static_cast<double>(
          bf16_to_float(read_u16_le_unchecked(right_biases + scale_byte_offset)));
      const std::size_t word_begin = block * 8;
      const std::size_t word_limit = std::min<std::size_t>(word_begin + 8, weight_cols);
      for (std::size_t word_index = word_begin; word_index < word_limit; ++word_index) {
        const std::size_t weight_byte_offset = row_weight_byte_base + (word_index * 4);
        const std::uint32_t left_word =
            read_u32_le_unchecked(left_weight + weight_byte_offset);
        const std::uint32_t right_word =
            read_u32_le_unchecked(right_weight + weight_byte_offset);
        const std::size_t input_base = word_index * 8;
        left_dot += static_cast<double>(input_values[input_base + 0]) *
                    (static_cast<double>((left_word >> 0U) & 0xFU) * left_scale + left_bias);
        left_dot += static_cast<double>(input_values[input_base + 1]) *
                    (static_cast<double>((left_word >> 4U) & 0xFU) * left_scale + left_bias);
        left_dot += static_cast<double>(input_values[input_base + 2]) *
                    (static_cast<double>((left_word >> 8U) & 0xFU) * left_scale + left_bias);
        left_dot += static_cast<double>(input_values[input_base + 3]) *
                    (static_cast<double>((left_word >> 12U) & 0xFU) * left_scale + left_bias);
        left_dot += static_cast<double>(input_values[input_base + 4]) *
                    (static_cast<double>((left_word >> 16U) & 0xFU) * left_scale + left_bias);
        left_dot += static_cast<double>(input_values[input_base + 5]) *
                    (static_cast<double>((left_word >> 20U) & 0xFU) * left_scale + left_bias);
        left_dot += static_cast<double>(input_values[input_base + 6]) *
                    (static_cast<double>((left_word >> 24U) & 0xFU) * left_scale + left_bias);
        left_dot += static_cast<double>(input_values[input_base + 7]) *
                    (static_cast<double>((left_word >> 28U) & 0xFU) * left_scale + left_bias);

        right_dot += static_cast<double>(input_values[input_base + 0]) *
                     (static_cast<double>((right_word >> 0U) & 0xFU) * right_scale + right_bias);
        right_dot += static_cast<double>(input_values[input_base + 1]) *
                     (static_cast<double>((right_word >> 4U) & 0xFU) * right_scale + right_bias);
        right_dot += static_cast<double>(input_values[input_base + 2]) *
                     (static_cast<double>((right_word >> 8U) & 0xFU) * right_scale + right_bias);
        right_dot += static_cast<double>(input_values[input_base + 3]) *
                     (static_cast<double>((right_word >> 12U) & 0xFU) * right_scale + right_bias);
        right_dot += static_cast<double>(input_values[input_base + 4]) *
                     (static_cast<double>((right_word >> 16U) & 0xFU) * right_scale + right_bias);
        right_dot += static_cast<double>(input_values[input_base + 5]) *
                     (static_cast<double>((right_word >> 20U) & 0xFU) * right_scale + right_bias);
        right_dot += static_cast<double>(input_values[input_base + 6]) *
                     (static_cast<double>((right_word >> 24U) & 0xFU) * right_scale + right_bias);
        right_dot += static_cast<double>(input_values[input_base + 7]) *
                     (static_cast<double>((right_word >> 28U) & 0xFU) * right_scale + right_bias);
      }
    }
    left_output[row] = static_cast<float>(left_dot);
    right_output[row] = static_cast<float>(right_dot);
  }
  return {std::move(left_output), std::move(right_output)};
}

std::pair<std::vector<float>, std::vector<float>> quantized_linear_pair_values_layout_cached_full_blocks(
    const TensorGroupRecord& left,
    const TensorGroupRecord& right,
    const std::vector<float>& input_values) {
  if (!left.quantized_layout_cached || !right.quantized_layout_cached ||
      left.quantized_rows != right.quantized_rows ||
      left.quantized_packed_cols != right.quantized_packed_cols ||
      left.quantized_scale_cols != right.quantized_scale_cols ||
      left.quantized_logical_width != right.quantized_logical_width ||
      left.quantized_scale_cols * 8 != left.quantized_packed_cols) {
    return quantized_linear_pair_values_layout_cached(left, right, input_values);
  }
  if (input_values.size() != left.quantized_logical_width) {
    throw std::runtime_error("full-block layout-cached paired quantized linear input length mismatch");
  }

  const auto left_weight_it = left.tensors.find("weight");
  const auto left_scales_it = left.tensors.find("scales");
  const auto left_biases_it = left.tensors.find("biases");
  const auto right_weight_it = right.tensors.find("weight");
  const auto right_scales_it = right.tensors.find("scales");
  const auto right_biases_it = right.tensors.find("biases");
  if (left_weight_it == left.tensors.end() ||
      left_scales_it == left.tensors.end() ||
      left_biases_it == left.tensors.end() ||
      right_weight_it == right.tensors.end() ||
      right_scales_it == right.tensors.end() ||
      right_biases_it == right.tensors.end()) {
    throw std::runtime_error("full-block layout-cached paired quantized linear requires weight/scales/biases");
  }

  const std::size_t rows = left.quantized_rows;
  const std::size_t weight_cols = left.quantized_packed_cols;
  const std::size_t scale_cols = left.quantized_scale_cols;
  const std::uint8_t* left_weight = tensor_payload_data(left_weight_it->second);
  const std::uint8_t* left_scales = tensor_payload_data(left_scales_it->second);
  const std::uint8_t* left_biases = tensor_payload_data(left_biases_it->second);
  const std::uint8_t* right_weight = tensor_payload_data(right_weight_it->second);
  const std::uint8_t* right_scales = tensor_payload_data(right_scales_it->second);
  const std::uint8_t* right_biases = tensor_payload_data(right_biases_it->second);

  std::vector<float> left_output(rows, 0.0f);
  std::vector<float> right_output(rows, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    double left_dot = 0.0;
    double right_dot = 0.0;
    const std::size_t row_weight_byte_base = row * weight_cols * 4;
    const std::size_t row_scale_byte_base = row * scale_cols * 2;
    for (std::size_t block = 0; block < scale_cols; ++block) {
      const std::size_t scale_byte_offset = row_scale_byte_base + (block * 2);
      const double left_scale = static_cast<double>(
          bf16_to_float(read_u16_le_unchecked(left_scales + scale_byte_offset)));
      const double left_bias = static_cast<double>(
          bf16_to_float(read_u16_le_unchecked(left_biases + scale_byte_offset)));
      const double right_scale = static_cast<double>(
          bf16_to_float(read_u16_le_unchecked(right_scales + scale_byte_offset)));
      const double right_bias = static_cast<double>(
          bf16_to_float(read_u16_le_unchecked(right_biases + scale_byte_offset)));
      const std::size_t word_begin = block * 8;
      const std::size_t word_end = word_begin + 8;
      for (std::size_t word_index = word_begin; word_index < word_end; ++word_index) {
        const std::size_t weight_byte_offset = row_weight_byte_base + (word_index * 4);
        const std::uint32_t left_word =
            read_u32_le_unchecked(left_weight + weight_byte_offset);
        const std::uint32_t right_word =
            read_u32_le_unchecked(right_weight + weight_byte_offset);
        const std::size_t input_base = word_index * 8;
        left_dot += static_cast<double>(input_values[input_base + 0]) *
                    (static_cast<double>((left_word >> 0U) & 0xFU) * left_scale + left_bias);
        left_dot += static_cast<double>(input_values[input_base + 1]) *
                    (static_cast<double>((left_word >> 4U) & 0xFU) * left_scale + left_bias);
        left_dot += static_cast<double>(input_values[input_base + 2]) *
                    (static_cast<double>((left_word >> 8U) & 0xFU) * left_scale + left_bias);
        left_dot += static_cast<double>(input_values[input_base + 3]) *
                    (static_cast<double>((left_word >> 12U) & 0xFU) * left_scale + left_bias);
        left_dot += static_cast<double>(input_values[input_base + 4]) *
                    (static_cast<double>((left_word >> 16U) & 0xFU) * left_scale + left_bias);
        left_dot += static_cast<double>(input_values[input_base + 5]) *
                    (static_cast<double>((left_word >> 20U) & 0xFU) * left_scale + left_bias);
        left_dot += static_cast<double>(input_values[input_base + 6]) *
                    (static_cast<double>((left_word >> 24U) & 0xFU) * left_scale + left_bias);
        left_dot += static_cast<double>(input_values[input_base + 7]) *
                    (static_cast<double>((left_word >> 28U) & 0xFU) * left_scale + left_bias);

        right_dot += static_cast<double>(input_values[input_base + 0]) *
                     (static_cast<double>((right_word >> 0U) & 0xFU) * right_scale + right_bias);
        right_dot += static_cast<double>(input_values[input_base + 1]) *
                     (static_cast<double>((right_word >> 4U) & 0xFU) * right_scale + right_bias);
        right_dot += static_cast<double>(input_values[input_base + 2]) *
                     (static_cast<double>((right_word >> 8U) & 0xFU) * right_scale + right_bias);
        right_dot += static_cast<double>(input_values[input_base + 3]) *
                     (static_cast<double>((right_word >> 12U) & 0xFU) * right_scale + right_bias);
        right_dot += static_cast<double>(input_values[input_base + 4]) *
                     (static_cast<double>((right_word >> 16U) & 0xFU) * right_scale + right_bias);
        right_dot += static_cast<double>(input_values[input_base + 5]) *
                     (static_cast<double>((right_word >> 20U) & 0xFU) * right_scale + right_bias);
        right_dot += static_cast<double>(input_values[input_base + 6]) *
                     (static_cast<double>((right_word >> 24U) & 0xFU) * right_scale + right_bias);
        right_dot += static_cast<double>(input_values[input_base + 7]) *
                     (static_cast<double>((right_word >> 28U) & 0xFU) * right_scale + right_bias);
      }
    }
    left_output[row] = static_cast<float>(left_dot);
    right_output[row] = static_cast<float>(right_dot);
  }
  return {std::move(left_output), std::move(right_output)};
}

std::vector<std::pair<std::uint64_t, float>> top_logits(
    const std::vector<float>& logits,
    std::size_t count);

std::pair<std::uint64_t, float> quantized_linear_top1_layout_cached(
    const TensorGroupRecord& record,
    const std::vector<float>& input_values) {
  if (!record.quantized_layout_cached) {
    std::vector<float> logits = quantized_linear_vector_values_row_optimized(record, input_values);
    auto top = top_logits(logits, 1);
    if (top.empty()) {
      return {0, 0.0f};
    }
    return top[0];
  }
  const auto weight_it = record.tensors.find("weight");
  const auto scales_it = record.tensors.find("scales");
  const auto biases_it = record.tensors.find("biases");
  if (weight_it == record.tensors.end() ||
      scales_it == record.tensors.end() ||
      biases_it == record.tensors.end()) {
    throw std::runtime_error("layout-cached top1 quantized linear requires weight/scales/biases");
  }
  if (input_values.size() != record.quantized_logical_width) {
    throw std::runtime_error("layout-cached top1 quantized linear input length mismatch");
  }

  const std::size_t rows = record.quantized_rows;
  const std::size_t weight_cols = record.quantized_packed_cols;
  const std::size_t scale_cols = record.quantized_scale_cols;
  const std::uint8_t* weight = tensor_payload_data(weight_it->second);
  const std::uint8_t* scales = tensor_payload_data(scales_it->second);
  const std::uint8_t* biases = tensor_payload_data(biases_it->second);
  std::uint64_t top_token = 0;
  float top_score = -std::numeric_limits<float>::infinity();
  for (std::size_t row = 0; row < rows; ++row) {
    double dot = 0.0;
    const std::size_t row_weight_byte_base = row * weight_cols * 4;
    const std::size_t row_scale_byte_base = row * scale_cols * 2;
    for (std::size_t block = 0; block < scale_cols; ++block) {
      const std::size_t scale_byte_offset = row_scale_byte_base + (block * 2);
      const double scale = static_cast<double>(
          bf16_to_float(read_u16_le_unchecked(scales + scale_byte_offset)));
      const double bias = static_cast<double>(
          bf16_to_float(read_u16_le_unchecked(biases + scale_byte_offset)));
      const std::size_t word_begin = block * 8;
      const std::size_t word_limit = std::min<std::size_t>(word_begin + 8, weight_cols);
      for (std::size_t word_index = word_begin; word_index < word_limit; ++word_index) {
        const std::uint32_t weight_word =
            read_u32_le_unchecked(weight + row_weight_byte_base + (word_index * 4));
        const std::size_t input_base = word_index * 8;
        dot += static_cast<double>(input_values[input_base + 0]) *
               (static_cast<double>((weight_word >> 0U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 1]) *
               (static_cast<double>((weight_word >> 4U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 2]) *
               (static_cast<double>((weight_word >> 8U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 3]) *
               (static_cast<double>((weight_word >> 12U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 4]) *
               (static_cast<double>((weight_word >> 16U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 5]) *
               (static_cast<double>((weight_word >> 20U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 6]) *
               (static_cast<double>((weight_word >> 24U) & 0xFU) * scale + bias);
        dot += static_cast<double>(input_values[input_base + 7]) *
               (static_cast<double>((weight_word >> 28U) & 0xFU) * scale + bias);
      }
    }
    const float score = static_cast<float>(dot);
    if (row == 0 || score > top_score) {
      top_score = score;
      top_token = static_cast<std::uint64_t>(row);
    }
  }
  return {top_token, top_score};
}

bool disable_mlx_quantized_linear();
bool& mlx_quantized_linear_available_flag();
extern thread_local bool mlx_quantized_linear_runtime_fallback_used;
void record_mlx_quantized_linear_fallback_step();
extern thread_local std::vector<std::string> mlx_decode_readback_reasons;
void record_mlx_decode_readback(const std::string& reason);
struct MlxProjectionTiming {
  double setup_ms = 0.0;
  double eval_ms = 0.0;
  double readback_ms = 0.0;
};
struct MlxResidentMlpChainTiming {
  double setup_ms = 0.0;
  double gate_up_eval_ms = 0.0;
  double activation_eval_ms = 0.0;
  double down_eval_ms = 0.0;
  double readback_ms = 0.0;
};

#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
struct ResidentDecodeValue {
  mutable mlx::core::array mlx;
  mutable std::vector<float> cpu_values;
  mutable bool cpu_valid = false;

  explicit ResidentDecodeValue(mlx::core::array value)
      : mlx(std::move(value)) {}

  const std::vector<float>& cpu(const std::string& reason = "decode_value_cpu_view") const {
    if (!cpu_valid) {
      record_mlx_decode_readback(reason);
      mlx.eval();
      mlx::core::synchronize();
      const float* data = mlx.data<float>();
      cpu_values.assign(data, data + mlx.size());
      cpu_valid = true;
    }
    return cpu_values;
  }
};

mlx::core::array mlx_array_from_vector(const std::vector<float>& values) {
  return mlx::core::array(
      values.data(),
      mlx::core::Shape{1, static_cast<int>(values.size())},
      mlx::core::float32);
}
#endif
std::vector<float> quantized_linear_vector_values_mlx(
    const TensorGroupRecord& record,
    const std::vector<float>& input_values,
    MlxProjectionTiming* timing = nullptr);

std::vector<float> quantized_linear_vector_values_selected(
    const TensorGroupRecord& record,
    const std::vector<float>& input_values,
    bool use_optimized,
    bool use_layout_cached = false,
    MlxProjectionTiming* mlx_timing = nullptr) {
  if (use_optimized) {
    if (use_layout_cached &&
        mlx_quantized_linear_available_flag() &&
        !disable_mlx_quantized_linear()) {
      try {
        return quantized_linear_vector_values_mlx(record, input_values, mlx_timing);
      } catch (...) {
        record_mlx_quantized_linear_fallback_step();
      }
    }
    if (use_layout_cached) {
      return quantized_linear_vector_values_layout_cached(record, input_values);
    }
    return quantized_linear_vector_values_row_optimized(record, input_values);
  }
  return quantized_linear_vector_values(record, input_values);
}

bool env_truthy(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) {
    return false;
  }
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

bool force_scalar_quantized_linear() {
  return env_truthy("RUSTY_FORCE_SCALAR_QUANTIZED_LINEAR") ||
         env_truthy("RUSTY_FORCE_SCALAR_LINEAR");
}

bool disable_mlx_quantized_linear() {
  return env_truthy("RUSTY_DISABLE_MLX_QUANTIZED_LINEAR") ||
         env_truthy("RUSTY_FORCE_CPU_QUANTIZED_LINEAR");
}

bool disable_mlx_resident_mlp_chain() {
  return env_truthy("RUSTY_DISABLE_MLX_RESIDENT_MLP_CHAIN");
}

bool enable_mlx_resident_mlp_chain_compare() {
  return env_truthy("RUSTY_VERIFY_MLP_CHAIN_COMPARE");
}

bool enable_cached_mlx_quantized_arrays() {
  return env_truthy("RUSTY_DEBUG_CACHED_MLX_QUANTIZED_ARRAYS");
}

bool use_resident_mlx_projection_arrays() {
  return !env_truthy("RUSTY_DISABLE_RESIDENT_MLX_PROJECTION_ARRAYS");
}

bool disable_mlx_resident_layer_block() {
  return env_truthy("RUSTY_DISABLE_MLX_RESIDENT_LAYER_BLOCK");
}

bool experimental_mlx_resident_block() {
  return env_truthy("RUSTY_EXPERIMENTAL_MLX_RESIDENT_BLOCK");
}

bool enable_layer0_diagnostic_probe() {
  return env_truthy("RUSTY_DEBUG_LAYER0_PROBE") ||
         env_truthy("RUSTY_VERIFY_LAYER0_PROBE");
}

bool resident_attention_to_o_enabled() {
  return env_truthy("RUSTY_RESIDENT_ATTENTION_TO_O");
}

bool resident_o_residual_enabled() {
  return env_truthy("RUSTY_RESIDENT_O_RESIDUAL");
}

bool resident_mlp_only_requested() {
  return env_truthy("RUSTY_RESIDENT_MLP_ONLY");
}

bool experimental_mlx_logits_topk_enabled() {
  return !env_truthy("RUSTY_DISABLE_MLX_LOGITS_TOPK");
}

bool experimental_mlx_qk_norm_rope_enabled() {
  return !env_truthy("RUSTY_DISABLE_MLX_QK_NORM_ROPE");
}

bool verify_mlx_qk_norm_rope_enabled() {
  return env_truthy("RUSTY_VERIFY_MLX_QK_NORM_ROPE");
}

bool generation_checksum_diagnostics_enabled() {
  return env_truthy("RUSTY_DEBUG_CHECKSUMS") ||
         env_truthy("RUSTY_DEBUG_READBACKS") ||
         env_truthy("RUSTY_VERIFY_CHECKSUMS");
}

std::string experimental_mlx_attention_mode() {
  const char* raw = std::getenv("RUSTY_ATTENTION_BACKEND");
  if (raw == nullptr) {
    raw = std::getenv("RUSTY_EXPERIMENTAL_MLX_ATTENTION");
  }
  if (raw == nullptr) {
    return "chunked_compact_mlx";
  }
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (value == "cpu" || value == "compact_cpu" || value == "none" || value == "disabled") {
    return "0";
  }
  if (value == "1" || value == "true" || value == "yes" || value == "on") {
    return "chunked_compact_mlx";
  }
  if (value == "chunked" || value == "chunked_expanded" || value == "mlx_chunked") {
    return "chunked_expanded_kv";
  }
  if (value == "chunked_compact" || value == "compact_mlx" ||
      value == "mlx_compact" || value == "mlx_chunked_compact") {
    return "chunked_compact_mlx";
  }
  return value;
}

std::size_t expanded_kv_chunk_size() {
  const char* raw = std::getenv("RUSTY_KV_CHUNK_SIZE");
  if (raw == nullptr || std::string(raw).empty()) {
    return 256;
  }
  try {
    return std::max<std::size_t>(1, static_cast<std::size_t>(std::stoull(raw)));
  } catch (...) {
    return 256;
  }
}

bool defer_chunked_kv_append_sync() {
  return env_truthy("RUSTY_DEFER_CHUNKED_KV_APPEND_SYNC");
}

bool chunk_aware_attention_enabled() {
  if (env_truthy("RUSTY_FORCE_COMPACT_CONCAT_ATTENTION")) {
    return false;
  }
  if (env_truthy("RUSTY_DISABLE_CHUNK_AWARE_ATTENTION")) {
    return false;
  }
  return true;
}

bool keep_cpu_kv_mirror() {
  if (env_truthy("RUSTY_KEEP_CPU_KV_MIRROR")) {
    return true;
  }
  const std::string mode = experimental_mlx_attention_mode();
  return mode != "chunked_expanded_kv" &&
         mode != "chunked_compact_mlx" &&
         mode != "expanded_kv";
}

std::uint64_t current_process_thread_count() {
#if defined(__APPLE__)
  thread_act_array_t threads = nullptr;
  mach_msg_type_number_t count = 0;
  kern_return_t kr = task_threads(mach_task_self(), &threads, &count);
  if (kr == KERN_SUCCESS) {
    if (threads != nullptr) {
      vm_deallocate(
          mach_task_self(),
          reinterpret_cast<vm_address_t>(threads),
          static_cast<vm_size_t>(count * sizeof(thread_t)));
    }
    return static_cast<std::uint64_t>(count);
  }
#endif
  return 0;
}

std::string runtime_thread_env_json() {
  const char* names[] = {
      "MLX_NUM_THREADS",
      "OMP_NUM_THREADS",
      "VECLIB_MAXIMUM_THREADS",
      "RAYON_NUM_THREADS",
      "RUSTY_ATTENTION_BACKEND",
      "RUSTY_KV_CHUNK_SIZE"};
  std::ostringstream out;
  out << "{";
  for (std::size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
    if (i > 0) {
      out << ",";
    }
    const char* value = std::getenv(names[i]);
    out << "\"" << names[i] << "\":";
    if (value == nullptr) {
      out << "null";
    } else {
      out << "\"" << json_escape(value) << "\"";
    }
  }
  out << "}";
  return out.str();
}

bool experimental_mlx_attention() {
  const std::string mode = experimental_mlx_attention_mode();
  return mode == "chunked_expanded_kv" ||
         mode == "chunked_compact_mlx" ||
         mode == "expanded_kv" ||
         mode == "batched" ||
         mode == "mlx";
}

bool& mlx_quantized_linear_available_flag() {
  static bool available = false;
  return available;
}

thread_local bool mlx_quantized_linear_runtime_fallback_used = false;
thread_local std::string mlx_quantized_linear_step_context;
thread_local std::vector<std::string> mlx_quantized_linear_fallback_steps;
thread_local std::vector<std::string> mlx_decode_readback_reasons;
thread_local bool mlx_qk_norm_rope_applied_to_generation = false;
thread_local bool mlx_qk_norm_rope_fallback_used = false;
thread_local bool mlx_qk_norm_rope_verify_ran = false;
thread_local double mlx_qk_norm_rope_verify_max_abs_diff = 0.0;

void record_mlx_decode_readback(const std::string& reason) {
  mlx_decode_readback_reasons.push_back(reason);
}

void record_mlx_quantized_linear_fallback_step() {
  mlx_quantized_linear_runtime_fallback_used = true;
  const std::string step = mlx_quantized_linear_step_context.empty()
      ? "unknown"
      : mlx_quantized_linear_step_context;
  if (std::find(
          mlx_quantized_linear_fallback_steps.begin(),
          mlx_quantized_linear_fallback_steps.end(),
          step) == mlx_quantized_linear_fallback_steps.end()) {
    mlx_quantized_linear_fallback_steps.push_back(step);
  }
}

struct MlxQuantizedLinearStepScope {
  explicit MlxQuantizedLinearStepScope(std::string step)
      : previous(mlx_quantized_linear_step_context) {
    mlx_quantized_linear_step_context = std::move(step);
  }

  ~MlxQuantizedLinearStepScope() {
    mlx_quantized_linear_step_context = previous;
  }

  std::string previous;
};

#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
struct LocalQuantizedMlxArrays {
  std::vector<std::uint32_t> weight_words;
  std::vector<mlx::core::bfloat16_t> scale_values;
  std::vector<mlx::core::bfloat16_t> bias_values;
  std::optional<mlx::core::array> weight_array;
  std::optional<mlx::core::array> scales_array;
  std::optional<mlx::core::array> biases_array;
};

void ensure_mlx_quantized_arrays_cached(
    const TensorGroupRecord& record,
    ResidentProjectionWarmupTiming* timing = nullptr) {
  if (record.mlx_quantized_arrays_cached) {
    return;
  }
  if (!record.quantized_layout_cached) {
    throw std::runtime_error("MLX quantized linear requires cached quantized layout metadata");
  }
  const auto weight_it = record.tensors.find("weight");
  const auto scales_it = record.tensors.find("scales");
  const auto biases_it = record.tensors.find("biases");
  if (weight_it == record.tensors.end() ||
      scales_it == record.tensors.end() ||
      biases_it == record.tensors.end()) {
    throw std::runtime_error("MLX quantized linear requires weight/scales/biases");
  }
  const std::size_t rows = record.quantized_rows;
  const std::size_t packed_cols = record.quantized_packed_cols;
  const std::size_t scale_cols = record.quantized_scale_cols;
  if (rows == 0 || packed_cols == 0 || scale_cols == 0) {
    throw std::runtime_error("MLX quantized linear has invalid layout metadata");
  }

  auto construction_start = std::chrono::steady_clock::now();
  mlx::core::array weight_array(
      reinterpret_cast<const std::uint32_t*>(tensor_payload_data(weight_it->second)),
      mlx::core::Shape{static_cast<int>(rows), static_cast<int>(packed_cols)},
      mlx::core::uint32);
  mlx::core::array scales_array(
      reinterpret_cast<const mlx::core::bfloat16_t*>(tensor_payload_data(scales_it->second)),
      mlx::core::Shape{static_cast<int>(rows), static_cast<int>(scale_cols)},
      mlx::core::bfloat16);
  mlx::core::array biases_array(
      reinterpret_cast<const mlx::core::bfloat16_t*>(tensor_payload_data(biases_it->second)),
      mlx::core::Shape{static_cast<int>(rows), static_cast<int>(scale_cols)},
      mlx::core::bfloat16);
  if (timing != nullptr) {
    timing->mlx_array_construction_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - construction_start).count();
  }

  auto eval_start = std::chrono::steady_clock::now();
  auto weight_copy = mlx::core::copy(weight_array);
  auto scales_copy = mlx::core::copy(scales_array);
  auto biases_copy = mlx::core::copy(biases_array);
  mlx::core::eval(weight_copy, scales_copy, biases_copy);
  mlx::core::synchronize();
  if (timing != nullptr) {
    timing->first_eval_compile_warmup_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - eval_start).count();
  }

  auto storage_start = std::chrono::steady_clock::now();
  record.mlx_weight_array.emplace(std::move(weight_copy));
  record.mlx_scales_array.emplace(std::move(scales_copy));
  record.mlx_biases_array.emplace(std::move(biases_copy));
  record.mlx_quantized_arrays_cached = true;
  if (timing != nullptr) {
    timing->metadata_cache_storage_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - storage_start).count();
  }
}

void warm_resident_mlx_projection_arrays(
    const ResidentModelLayers& resident,
    const TensorGroupRecord& embedding,
    ResidentProjectionWarmupTiming* timing = nullptr) {
  if (!use_resident_mlx_projection_arrays()) {
    return;
  }
  auto enumerate_start = std::chrono::steady_clock::now();
  std::vector<const TensorGroupRecord*> groups;
  groups.reserve(resident.layers.size() * 7 + 1);
  for (const auto& layer : resident.layers) {
    groups.push_back(&layer.q_proj);
    groups.push_back(&layer.k_proj);
    groups.push_back(&layer.v_proj);
    groups.push_back(&layer.o_proj);
    groups.push_back(&layer.gate_proj);
    groups.push_back(&layer.up_proj);
    groups.push_back(&layer.down_proj);
  }
  groups.push_back(&embedding);
  if (timing != nullptr) {
    timing->enumerate_groups_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - enumerate_start).count();
  }
  for (const auto* group : groups) {
    ensure_mlx_quantized_arrays_cached(*group, timing);
  }
}

mlx::core::array make_quantized_matmul_from_mlx_input(
    const TensorGroupRecord& record,
    const mlx::core::array& input_array,
    LocalQuantizedMlxArrays& local,
    const std::size_t expected_input_width) {
  if (!record.quantized_layout_cached) {
    throw std::runtime_error("MLX quantized linear requires cached quantized layout metadata");
  }
  if (expected_input_width != record.quantized_logical_width) {
    throw std::runtime_error("MLX quantized linear input length mismatch");
  }
  const std::size_t rows = record.quantized_rows;
  const std::size_t packed_cols = record.quantized_packed_cols;
  const std::size_t scale_cols = record.quantized_scale_cols;
  if (rows == 0 || packed_cols == 0 || scale_cols == 0) {
    throw std::runtime_error("MLX quantized linear has invalid layout metadata");
  }
  const bool use_cached_arrays =
      use_resident_mlx_projection_arrays() || enable_cached_mlx_quantized_arrays();
  if (use_cached_arrays) {
    ensure_mlx_quantized_arrays_cached(record);
  } else {
    const auto weight_it = record.tensors.find("weight");
    const auto scales_it = record.tensors.find("scales");
    const auto biases_it = record.tensors.find("biases");
    if (weight_it == record.tensors.end() ||
        scales_it == record.tensors.end() ||
        biases_it == record.tensors.end()) {
      throw std::runtime_error("MLX quantized linear requires weight/scales/biases");
    }
    local.weight_words.resize(rows * packed_cols);
    std::memcpy(
        local.weight_words.data(),
        tensor_payload_data(weight_it->second),
        local.weight_words.size() * sizeof(std::uint32_t));
    local.scale_values.resize(rows * scale_cols);
    local.bias_values.resize(rows * scale_cols);
    std::memcpy(
        local.scale_values.data(),
        tensor_payload_data(scales_it->second),
        local.scale_values.size() * sizeof(mlx::core::bfloat16_t));
    std::memcpy(
        local.bias_values.data(),
        tensor_payload_data(biases_it->second),
        local.bias_values.size() * sizeof(mlx::core::bfloat16_t));
    local.weight_array.emplace(
        local.weight_words.data(),
        mlx::core::Shape{static_cast<int>(rows), static_cast<int>(packed_cols)},
        mlx::core::uint32);
    local.scales_array.emplace(
        local.scale_values.data(),
        mlx::core::Shape{static_cast<int>(rows), static_cast<int>(scale_cols)},
        mlx::core::bfloat16);
    local.biases_array.emplace(
        local.bias_values.data(),
        mlx::core::Shape{static_cast<int>(rows), static_cast<int>(scale_cols)},
        mlx::core::bfloat16);
  }
  const mlx::core::array& weight_ref = use_cached_arrays ? *record.mlx_weight_array : *local.weight_array;
  const mlx::core::array& scales_ref = use_cached_arrays ? *record.mlx_scales_array : *local.scales_array;
  const mlx::core::array& biases_ref = use_cached_arrays ? *record.mlx_biases_array : *local.biases_array;
  return mlx::core::quantized_matmul(
      input_array,
      weight_ref,
      scales_ref,
      std::optional<mlx::core::array>(biases_ref),
      true,
      64,
      4,
      "affine");
}
#endif

std::vector<float> quantized_linear_vector_values_mlx(
    const TensorGroupRecord& record,
    const std::vector<float>& input_values,
    MlxProjectionTiming* timing) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  auto setup_start = std::chrono::steady_clock::now();
  if (!record.quantized_layout_cached) {
    throw std::runtime_error("MLX quantized linear requires cached quantized layout metadata");
  }
  if (input_values.size() != record.quantized_logical_width) {
    throw std::runtime_error("MLX quantized linear input length mismatch");
  }
  const std::size_t rows = record.quantized_rows;
  const std::size_t packed_cols = record.quantized_packed_cols;
  const std::size_t scale_cols = record.quantized_scale_cols;
  if (rows == 0 || packed_cols == 0 || scale_cols == 0) {
    throw std::runtime_error("MLX quantized linear has invalid layout metadata");
  }
  const bool use_cached_arrays =
      use_resident_mlx_projection_arrays() || enable_cached_mlx_quantized_arrays();
  std::vector<std::uint32_t> weight_words;
  std::vector<mlx::core::bfloat16_t> scale_values;
  std::vector<mlx::core::bfloat16_t> bias_values;
  std::optional<mlx::core::array> weight_array;
  std::optional<mlx::core::array> scales_array;
  std::optional<mlx::core::array> biases_array;
  if (use_cached_arrays) {
    ensure_mlx_quantized_arrays_cached(record);
  } else {
    const auto weight_it = record.tensors.find("weight");
    const auto scales_it = record.tensors.find("scales");
    const auto biases_it = record.tensors.find("biases");
    if (weight_it == record.tensors.end() ||
        scales_it == record.tensors.end() ||
        biases_it == record.tensors.end()) {
      throw std::runtime_error("MLX quantized linear requires weight/scales/biases");
    }
    weight_words.resize(rows * packed_cols);
    std::memcpy(
        weight_words.data(),
        tensor_payload_data(weight_it->second),
        weight_words.size() * sizeof(std::uint32_t));
    scale_values.resize(rows * scale_cols);
    bias_values.resize(rows * scale_cols);
    std::memcpy(
        scale_values.data(),
        tensor_payload_data(scales_it->second),
        scale_values.size() * sizeof(mlx::core::bfloat16_t));
    std::memcpy(
        bias_values.data(),
        tensor_payload_data(biases_it->second),
        bias_values.size() * sizeof(mlx::core::bfloat16_t));
    weight_array.emplace(
        weight_words.data(),
        mlx::core::Shape{static_cast<int>(rows), static_cast<int>(packed_cols)},
        mlx::core::uint32);
    scales_array.emplace(
        scale_values.data(),
        mlx::core::Shape{static_cast<int>(rows), static_cast<int>(scale_cols)},
        mlx::core::bfloat16);
    biases_array.emplace(
        bias_values.data(),
        mlx::core::Shape{static_cast<int>(rows), static_cast<int>(scale_cols)},
        mlx::core::bfloat16);
  }
  mlx::core::array input_array(
      input_values.data(),
      mlx::core::Shape{1, static_cast<int>(input_values.size())},
      mlx::core::float32);
  const mlx::core::array& weight_ref = use_cached_arrays ? *record.mlx_weight_array : *weight_array;
  const mlx::core::array& scales_ref = use_cached_arrays ? *record.mlx_scales_array : *scales_array;
  const mlx::core::array& biases_ref = use_cached_arrays ? *record.mlx_biases_array : *biases_array;
  if (timing != nullptr) {
    timing->setup_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup_start).count();
  }

  auto eval_start = std::chrono::steady_clock::now();
  auto result = mlx::core::quantized_matmul(
      input_array,
      weight_ref,
      scales_ref,
      std::optional<mlx::core::array>(biases_ref),
      true,
      64,
      4,
      "affine");
  result.eval();
  mlx::core::synchronize();
  if (timing != nullptr) {
    timing->eval_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - eval_start).count();
  }

  auto readback_start = std::chrono::steady_clock::now();
  if (result.size() != static_cast<int>(rows)) {
    throw std::runtime_error("MLX quantized linear output length mismatch");
  }
  record_mlx_decode_readback(mlx_quantized_linear_step_context.empty()
      ? "quantized_linear_vector"
      : mlx_quantized_linear_step_context);
  const float* data = result.data<float>();
  std::vector<float> values(data, data + rows);
  if (timing != nullptr) {
    timing->readback_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - readback_start).count();
  }
  return values;
#else
  (void)record;
  (void)input_values;
  (void)timing;
  throw std::runtime_error("MLX native link is unavailable");
#endif
}

std::pair<std::vector<float>, std::vector<float>> quantized_linear_pair_values_mlx(
    const TensorGroupRecord& gate_record,
    const TensorGroupRecord& up_record,
    const std::vector<float>& input_values,
    MlxProjectionTiming* gate_timing = nullptr,
    MlxProjectionTiming* up_timing = nullptr) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  struct LocalQuantizedMlxArrays {
    std::vector<std::uint32_t> weight_words;
    std::vector<mlx::core::bfloat16_t> scale_values;
    std::vector<mlx::core::bfloat16_t> bias_values;
    std::optional<mlx::core::array> weight_array;
    std::optional<mlx::core::array> scales_array;
    std::optional<mlx::core::array> biases_array;
  };
  LocalQuantizedMlxArrays gate_local;
  LocalQuantizedMlxArrays up_local;
  const bool use_cached_arrays =
      use_resident_mlx_projection_arrays() || enable_cached_mlx_quantized_arrays();
  auto setup_start = std::chrono::steady_clock::now();
  auto make_quantized_matmul = [&](const TensorGroupRecord& record, LocalQuantizedMlxArrays& local) {
    if (!record.quantized_layout_cached) {
      throw std::runtime_error("MLX quantized pair requires cached quantized layout metadata");
    }
    if (input_values.size() != record.quantized_logical_width) {
      throw std::runtime_error("MLX quantized pair input length mismatch");
    }
    const std::size_t rows = record.quantized_rows;
    const std::size_t packed_cols = record.quantized_packed_cols;
    const std::size_t scale_cols = record.quantized_scale_cols;
    if (rows == 0 || packed_cols == 0 || scale_cols == 0) {
      throw std::runtime_error("MLX quantized pair has invalid layout metadata");
    }
    if (use_cached_arrays) {
      ensure_mlx_quantized_arrays_cached(record);
    } else {
      const auto weight_it = record.tensors.find("weight");
      const auto scales_it = record.tensors.find("scales");
      const auto biases_it = record.tensors.find("biases");
      if (weight_it == record.tensors.end() ||
          scales_it == record.tensors.end() ||
          biases_it == record.tensors.end()) {
        throw std::runtime_error("MLX quantized pair requires weight/scales/biases");
      }
      local.weight_words.resize(rows * packed_cols);
      std::memcpy(
          local.weight_words.data(),
          tensor_payload_data(weight_it->second),
          local.weight_words.size() * sizeof(std::uint32_t));
      local.scale_values.resize(rows * scale_cols);
      local.bias_values.resize(rows * scale_cols);
      std::memcpy(
          local.scale_values.data(),
          tensor_payload_data(scales_it->second),
          local.scale_values.size() * sizeof(mlx::core::bfloat16_t));
      std::memcpy(
          local.bias_values.data(),
          tensor_payload_data(biases_it->second),
          local.bias_values.size() * sizeof(mlx::core::bfloat16_t));
      local.weight_array.emplace(
          local.weight_words.data(),
          mlx::core::Shape{static_cast<int>(rows), static_cast<int>(packed_cols)},
          mlx::core::uint32);
      local.scales_array.emplace(
          local.scale_values.data(),
          mlx::core::Shape{static_cast<int>(rows), static_cast<int>(scale_cols)},
          mlx::core::bfloat16);
      local.biases_array.emplace(
          local.bias_values.data(),
          mlx::core::Shape{static_cast<int>(rows), static_cast<int>(scale_cols)},
          mlx::core::bfloat16);
    }
    const mlx::core::array& weight_ref = use_cached_arrays ? *record.mlx_weight_array : *local.weight_array;
    const mlx::core::array& scales_ref = use_cached_arrays ? *record.mlx_scales_array : *local.scales_array;
    const mlx::core::array& biases_ref = use_cached_arrays ? *record.mlx_biases_array : *local.biases_array;

    mlx::core::array input_array(
        input_values.data(),
        mlx::core::Shape{1, static_cast<int>(input_values.size())},
        mlx::core::float32);

    return mlx::core::quantized_matmul(
        input_array,
        weight_ref,
        scales_ref,
        std::optional<mlx::core::array>(biases_ref),
        true,
        64,
        4,
        "affine");
  };

  auto gate_result = make_quantized_matmul(gate_record, gate_local);
  auto up_result = make_quantized_matmul(up_record, up_local);
  const double setup_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - setup_start).count();
  if (gate_timing != nullptr) {
    gate_timing->setup_ms += setup_ms / 2.0;
  }
  if (up_timing != nullptr) {
    up_timing->setup_ms += setup_ms / 2.0;
  }

  auto eval_start = std::chrono::steady_clock::now();
  mlx::core::eval(gate_result, up_result);
  mlx::core::synchronize();
  const double eval_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - eval_start).count();
  if (gate_timing != nullptr) {
    gate_timing->eval_ms += eval_ms / 2.0;
  }
  if (up_timing != nullptr) {
    up_timing->eval_ms += eval_ms / 2.0;
  }

  auto readback_start = std::chrono::steady_clock::now();
  if (gate_result.size() != static_cast<int>(gate_record.quantized_rows) ||
      up_result.size() != static_cast<int>(up_record.quantized_rows)) {
    throw std::runtime_error("MLX quantized pair output length mismatch");
  }
  const float* gate_data = gate_result.data<float>();
  const float* up_data = up_result.data<float>();
  std::pair<std::vector<float>, std::vector<float>> values{
      std::vector<float>(gate_data, gate_data + gate_record.quantized_rows),
      std::vector<float>(up_data, up_data + up_record.quantized_rows)};
  const double readback_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - readback_start).count();
  if (gate_timing != nullptr) {
    gate_timing->readback_ms += readback_ms / 2.0;
  }
  if (up_timing != nullptr) {
    up_timing->readback_ms += readback_ms / 2.0;
  }
  return values;
#else
  (void)gate_record;
  (void)up_record;
  (void)input_values;
  throw std::runtime_error("MLX native link is unavailable");
#endif
}

std::vector<float> quantized_linear_mlp_chain_down_values_mlx(
    const TensorGroupRecord& gate_record,
    const TensorGroupRecord& up_record,
    const TensorGroupRecord& down_record,
    const std::vector<float>& input_values,
    MlxResidentMlpChainTiming* timing = nullptr) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  if (input_values.empty()) {
    throw std::runtime_error("MLX resident MLP chain input is empty");
  }
  LocalQuantizedMlxArrays gate_local;
  LocalQuantizedMlxArrays up_local;
  LocalQuantizedMlxArrays down_local;
  auto setup_start = std::chrono::steady_clock::now();
  mlx::core::array input_array(
      input_values.data(),
      mlx::core::Shape{1, static_cast<int>(input_values.size())},
      mlx::core::float32);
  auto gate_result = make_quantized_matmul_from_mlx_input(
      gate_record,
      input_array,
      gate_local,
      input_values.size());
  auto up_result = make_quantized_matmul_from_mlx_input(
      up_record,
      input_array,
      up_local,
      input_values.size());
  if (timing != nullptr) {
    timing->setup_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup_start).count();
  }

  auto gate_up_eval_start = std::chrono::steady_clock::now();
  mlx::core::eval(gate_result, up_result);
  if (timing != nullptr) {
    timing->gate_up_eval_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - gate_up_eval_start).count();
  }

  auto activation_start = std::chrono::steady_clock::now();
  auto activated = gate_result * mlx::core::sigmoid(gate_result) * up_result;
  if (timing != nullptr) {
    timing->activation_eval_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - activation_start).count();
  }

  auto down_eval_start = std::chrono::steady_clock::now();
  auto down_result = make_quantized_matmul_from_mlx_input(
      down_record,
      activated,
      down_local,
      gate_record.quantized_rows);
  down_result.eval();
  mlx::core::synchronize();
  if (timing != nullptr) {
    timing->down_eval_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - down_eval_start).count();
  }

  auto readback_start = std::chrono::steady_clock::now();
  if (down_result.size() != static_cast<int>(down_record.quantized_rows)) {
    throw std::runtime_error("MLX resident MLP chain down output length mismatch");
  }
  const float* down_data = down_result.data<float>();
  std::vector<float> down_values(down_data, down_data + down_record.quantized_rows);
  if (timing != nullptr) {
    timing->readback_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - readback_start).count();
  }
  return down_values;
#else
  (void)gate_record;
  (void)up_record;
  (void)down_record;
  (void)input_values;
  (void)timing;
  throw std::runtime_error("MLX native link is unavailable");
#endif
}

mlx::core::array quantized_linear_mlp_chain_residual_mlx(
    const TensorGroupRecord& gate_record,
    const TensorGroupRecord& up_record,
    const TensorGroupRecord& down_record,
    const mlx::core::array& projection_input_array,
    const mlx::core::array& residual_base_array,
    MlxResidentMlpChainTiming* timing = nullptr,
    const LoraProjectionRecord* gate_lora = nullptr,
    const LoraProjectionRecord* up_lora = nullptr,
    const LoraProjectionRecord* down_lora = nullptr) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  LocalQuantizedMlxArrays gate_local;
  LocalQuantizedMlxArrays up_local;
  LocalQuantizedMlxArrays down_local;
  auto setup_start = std::chrono::steady_clock::now();
  const std::size_t input_width = static_cast<std::size_t>(projection_input_array.size());
  auto gate_result = apply_lora_delta_if_present(
      make_quantized_matmul_from_mlx_input(
          gate_record, projection_input_array, gate_local, input_width),
      projection_input_array,
      gate_lora);
  auto up_result = apply_lora_delta_if_present(
      make_quantized_matmul_from_mlx_input(
          up_record, projection_input_array, up_local, input_width),
      projection_input_array,
      up_lora);
  if (timing != nullptr) {
    timing->setup_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup_start).count();
  }

  auto gate_up_eval_start = std::chrono::steady_clock::now();
  mlx::core::eval(gate_result, up_result);
  if (timing != nullptr) {
    timing->gate_up_eval_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - gate_up_eval_start).count();
  }

  auto activation_start = std::chrono::steady_clock::now();
  auto activated = gate_result * mlx::core::sigmoid(gate_result) * up_result;
  if (timing != nullptr) {
    timing->activation_eval_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - activation_start).count();
  }

  auto down_eval_start = std::chrono::steady_clock::now();
  auto down_result = apply_lora_delta_if_present(
      make_quantized_matmul_from_mlx_input(
          down_record,
          activated,
          down_local,
          gate_record.quantized_rows),
      activated,
      down_lora);
  auto residual = down_result + residual_base_array;
  residual.eval();
  mlx::core::synchronize();
  if (timing != nullptr) {
    timing->down_eval_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - down_eval_start).count();
  }
  return residual;
#else
  (void)gate_record;
  (void)up_record;
  (void)down_record;
  (void)projection_input_array;
  (void)residual_base_array;
  (void)timing;
  (void)gate_lora;
  (void)up_lora;
  (void)down_lora;
  throw std::runtime_error("MLX native link is unavailable");
#endif
}

std::vector<float> quantized_linear_residual_values_mlx(
    const TensorGroupRecord& record,
    const std::vector<float>& input_values,
    const std::vector<float>& residual_values,
    MlxProjectionTiming* timing = nullptr) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  if (input_values.size() != record.quantized_logical_width) {
    throw std::runtime_error("MLX residual quantized linear input length mismatch");
  }
  LocalQuantizedMlxArrays local;
  auto setup_start = std::chrono::steady_clock::now();
  mlx::core::array input_array(
      input_values.data(),
      mlx::core::Shape{1, static_cast<int>(input_values.size())},
      mlx::core::float32);
  auto projection = make_quantized_matmul_from_mlx_input(
      record,
      input_array,
      local,
      input_values.size());
  mlx::core::array residual_array(
      residual_values.data(),
      mlx::core::Shape{1, static_cast<int>(residual_values.size())},
      mlx::core::float32);
  if (timing != nullptr) {
    timing->setup_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup_start).count();
  }

  auto eval_start = std::chrono::steady_clock::now();
  auto result = projection + residual_array;
  result.eval();
  mlx::core::synchronize();
  if (timing != nullptr) {
    timing->eval_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - eval_start).count();
  }

  auto readback_start = std::chrono::steady_clock::now();
  if (result.size() != static_cast<int>(residual_values.size())) {
    throw std::runtime_error("MLX residual quantized linear output length mismatch");
  }
  const float* data = result.data<float>();
  std::vector<float> values(data, data + residual_values.size());
  if (timing != nullptr) {
    timing->readback_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - readback_start).count();
  }
  return values;
#else
  (void)record;
  (void)input_values;
  (void)residual_values;
  (void)timing;
  throw std::runtime_error("MLX native link is unavailable");
#endif
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> quantized_linear_triple_values_mlx(
    const TensorGroupRecord& first_record,
    const TensorGroupRecord& second_record,
    const TensorGroupRecord& third_record,
    const std::vector<float>& input_values) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  struct LocalQuantizedMlxArrays {
    std::vector<std::uint32_t> weight_words;
    std::vector<mlx::core::bfloat16_t> scale_values;
    std::vector<mlx::core::bfloat16_t> bias_values;
    std::optional<mlx::core::array> weight_array;
    std::optional<mlx::core::array> scales_array;
    std::optional<mlx::core::array> biases_array;
  };
  LocalQuantizedMlxArrays first_local;
  LocalQuantizedMlxArrays second_local;
  LocalQuantizedMlxArrays third_local;
  const bool use_cached_arrays =
      use_resident_mlx_projection_arrays() || enable_cached_mlx_quantized_arrays();
  auto make_quantized_matmul = [&](const TensorGroupRecord& record, LocalQuantizedMlxArrays& local) {
    if (!record.quantized_layout_cached) {
      throw std::runtime_error("MLX quantized triple requires cached quantized layout metadata");
    }
    if (input_values.size() != record.quantized_logical_width) {
      throw std::runtime_error("MLX quantized triple input length mismatch");
    }
    const std::size_t rows = record.quantized_rows;
    const std::size_t packed_cols = record.quantized_packed_cols;
    const std::size_t scale_cols = record.quantized_scale_cols;
    if (rows == 0 || packed_cols == 0 || scale_cols == 0) {
      throw std::runtime_error("MLX quantized triple has invalid layout metadata");
    }
    if (use_cached_arrays) {
      ensure_mlx_quantized_arrays_cached(record);
    } else {
      const auto weight_it = record.tensors.find("weight");
      const auto scales_it = record.tensors.find("scales");
      const auto biases_it = record.tensors.find("biases");
      if (weight_it == record.tensors.end() ||
          scales_it == record.tensors.end() ||
          biases_it == record.tensors.end()) {
        throw std::runtime_error("MLX quantized triple requires weight/scales/biases");
      }
      local.weight_words.resize(rows * packed_cols);
      std::memcpy(
          local.weight_words.data(),
          tensor_payload_data(weight_it->second),
          local.weight_words.size() * sizeof(std::uint32_t));
      local.scale_values.resize(rows * scale_cols);
      local.bias_values.resize(rows * scale_cols);
      std::memcpy(
          local.scale_values.data(),
          tensor_payload_data(scales_it->second),
          local.scale_values.size() * sizeof(mlx::core::bfloat16_t));
      std::memcpy(
          local.bias_values.data(),
          tensor_payload_data(biases_it->second),
          local.bias_values.size() * sizeof(mlx::core::bfloat16_t));
      local.weight_array.emplace(
          local.weight_words.data(),
          mlx::core::Shape{static_cast<int>(rows), static_cast<int>(packed_cols)},
          mlx::core::uint32);
      local.scales_array.emplace(
          local.scale_values.data(),
          mlx::core::Shape{static_cast<int>(rows), static_cast<int>(scale_cols)},
          mlx::core::bfloat16);
      local.biases_array.emplace(
          local.bias_values.data(),
          mlx::core::Shape{static_cast<int>(rows), static_cast<int>(scale_cols)},
          mlx::core::bfloat16);
    }
    const mlx::core::array& weight_ref = use_cached_arrays ? *record.mlx_weight_array : *local.weight_array;
    const mlx::core::array& scales_ref = use_cached_arrays ? *record.mlx_scales_array : *local.scales_array;
    const mlx::core::array& biases_ref = use_cached_arrays ? *record.mlx_biases_array : *local.biases_array;
    mlx::core::array input_array(
        input_values.data(),
        mlx::core::Shape{1, static_cast<int>(input_values.size())},
        mlx::core::float32);
    return mlx::core::quantized_matmul(
        input_array,
        weight_ref,
        scales_ref,
        std::optional<mlx::core::array>(biases_ref),
        true,
        64,
        4,
        "affine");
  };

  auto first_result = make_quantized_matmul(first_record, first_local);
  auto second_result = make_quantized_matmul(second_record, second_local);
  auto third_result = make_quantized_matmul(third_record, third_local);
  mlx::core::eval(first_result, second_result, third_result);
  mlx::core::synchronize();
  if (first_result.size() != static_cast<int>(first_record.quantized_rows) ||
      second_result.size() != static_cast<int>(second_record.quantized_rows) ||
      third_result.size() != static_cast<int>(third_record.quantized_rows)) {
    throw std::runtime_error("MLX quantized triple output length mismatch");
  }
  record_mlx_decode_readback("qkv_for_cpu_attention");
  const float* first_data = first_result.data<float>();
  const float* second_data = second_result.data<float>();
  const float* third_data = third_result.data<float>();
  return {
      std::vector<float>(first_data, first_data + first_record.quantized_rows),
      std::vector<float>(second_data, second_data + second_record.quantized_rows),
      std::vector<float>(third_data, third_data + third_record.quantized_rows)};
#else
  (void)first_record;
  (void)second_record;
  (void)third_record;
  (void)input_values;
  throw std::runtime_error("MLX native link is unavailable");
#endif
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> quantized_linear_triple_values_from_mlx_input(
    const TensorGroupRecord& first_record,
    const TensorGroupRecord& second_record,
    const TensorGroupRecord& third_record,
    const mlx::core::array& input_array,
    const LoraProjectionRecord* first_lora = nullptr,
    const LoraProjectionRecord* second_lora = nullptr,
    const LoraProjectionRecord* third_lora = nullptr) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  LocalQuantizedMlxArrays first_local;
  LocalQuantizedMlxArrays second_local;
  LocalQuantizedMlxArrays third_local;
  const std::size_t input_width = static_cast<std::size_t>(input_array.size());
  auto first_result = apply_lora_delta_if_present(
      make_quantized_matmul_from_mlx_input(first_record, input_array, first_local, input_width),
      input_array,
      first_lora);
  auto second_result = apply_lora_delta_if_present(
      make_quantized_matmul_from_mlx_input(second_record, input_array, second_local, input_width),
      input_array,
      second_lora);
  auto third_result = apply_lora_delta_if_present(
      make_quantized_matmul_from_mlx_input(third_record, input_array, third_local, input_width),
      input_array,
      third_lora);
  mlx::core::eval(first_result, second_result, third_result);
  mlx::core::synchronize();
  if (first_result.size() != static_cast<int>(first_record.quantized_rows) ||
      second_result.size() != static_cast<int>(second_record.quantized_rows) ||
      third_result.size() != static_cast<int>(third_record.quantized_rows)) {
    throw std::runtime_error("MLX quantized triple output length mismatch");
  }
  const float* first_data = first_result.data<float>();
  const float* second_data = second_result.data<float>();
  const float* third_data = third_result.data<float>();
  return {
      std::vector<float>(first_data, first_data + first_record.quantized_rows),
      std::vector<float>(second_data, second_data + second_record.quantized_rows),
      std::vector<float>(third_data, third_data + third_record.quantized_rows)};
#else
  (void)first_record;
  (void)second_record;
  (void)third_record;
  (void)input_array;
  (void)first_lora;
  (void)second_lora;
  (void)third_lora;
  throw std::runtime_error("MLX native link is unavailable");
#endif
}

#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
std::tuple<mlx::core::array, mlx::core::array, mlx::core::array> quantized_linear_triple_arrays_from_mlx_input(
    const TensorGroupRecord& first_record,
    const TensorGroupRecord& second_record,
    const TensorGroupRecord& third_record,
    const mlx::core::array& input_array,
    const LoraProjectionRecord* first_lora = nullptr,
    const LoraProjectionRecord* second_lora = nullptr,
    const LoraProjectionRecord* third_lora = nullptr) {
  LocalQuantizedMlxArrays first_local;
  LocalQuantizedMlxArrays second_local;
  LocalQuantizedMlxArrays third_local;
  const std::size_t input_width = static_cast<std::size_t>(input_array.size());
  auto first_result = apply_lora_delta_if_present(
      make_quantized_matmul_from_mlx_input(first_record, input_array, first_local, input_width),
      input_array,
      first_lora);
  auto second_result = apply_lora_delta_if_present(
      make_quantized_matmul_from_mlx_input(second_record, input_array, second_local, input_width),
      input_array,
      second_lora);
  auto third_result = apply_lora_delta_if_present(
      make_quantized_matmul_from_mlx_input(third_record, input_array, third_local, input_width),
      input_array,
      third_lora);
  return {first_result, second_result, third_result};
}

mlx::core::array rmsnorm_heads_mlx_array(
    const TensorGroupRecord& record,
    const mlx::core::array& input,
    std::size_t heads,
    std::size_t head_dim,
    double eps) {
  const auto weight_it = record.tensors.find("weight");
  if (weight_it == record.tensors.end()) {
    throw std::runtime_error("per-head RMSNorm MLX group is missing weight");
  }
  if (heads == 0 || head_dim == 0 ||
      static_cast<std::size_t>(input.size()) != heads * head_dim) {
    throw std::runtime_error("per-head RMSNorm MLX input shape mismatch");
  }
  const auto& weight_tensor = weight_it->second;
  if (weight_tensor.shape.empty() ||
      static_cast<std::size_t>(weight_tensor.shape[0]) != head_dim) {
    throw std::runtime_error("per-head RMSNorm MLX weight size mismatch");
  }
  mlx::core::array weight_bf16(
      reinterpret_cast<const mlx::core::bfloat16_t*>(tensor_payload_data(weight_tensor)),
      mlx::core::Shape{1, static_cast<int>(head_dim)},
      mlx::core::bfloat16);
  auto weight = mlx::core::astype(weight_bf16, mlx::core::float32);
  auto shaped = mlx::core::reshape(
      input,
      mlx::core::Shape{static_cast<int>(heads), static_cast<int>(head_dim)});
  auto mean_square = mlx::core::mean(mlx::core::square(shaped), 1, true);
  auto normalized = shaped * mlx::core::rsqrt(mean_square + static_cast<float>(eps)) * weight;
  return mlx::core::reshape(
      normalized,
      mlx::core::Shape{1, static_cast<int>(heads * head_dim)});
}

mlx::core::array rope_mlx_array(
    const mlx::core::array& input,
    std::size_t heads,
    std::size_t head_dim,
    std::uint64_t position,
    double theta) {
  if (heads == 0 || head_dim == 0 || head_dim % 2 != 0 || theta <= 0.0 ||
      static_cast<std::size_t>(input.size()) != heads * head_dim) {
    throw std::runtime_error("RoPE MLX input shape mismatch");
  }
  const std::size_t half_dim = head_dim / 2;
  std::vector<float> cos_values(heads * half_dim, 0.0f);
  std::vector<float> sin_values(heads * half_dim, 0.0f);
  for (std::size_t head = 0; head < heads; ++head) {
    const std::size_t offset = head * half_dim;
    for (std::size_t i = 0; i < half_dim; ++i) {
      const double inv_freq =
          1.0 / std::pow(theta, static_cast<double>(i * 2) / static_cast<double>(head_dim));
      const double angle = static_cast<double>(position) * inv_freq;
      cos_values[offset + i] = static_cast<float>(std::cos(angle));
      sin_values[offset + i] = static_cast<float>(std::sin(angle));
    }
  }
  mlx::core::array cos_array(
      cos_values.data(),
      mlx::core::Shape{static_cast<int>(heads), static_cast<int>(half_dim)},
      mlx::core::float32);
  mlx::core::array sin_array(
      sin_values.data(),
      mlx::core::Shape{static_cast<int>(heads), static_cast<int>(half_dim)},
      mlx::core::float32);
  auto shaped = mlx::core::reshape(
      input,
      mlx::core::Shape{static_cast<int>(heads), static_cast<int>(head_dim)});
  auto x1 = mlx::core::slice(
      shaped,
      mlx::core::Shape{0, 0},
      mlx::core::Shape{static_cast<int>(heads), static_cast<int>(half_dim)});
  auto x2 = mlx::core::slice(
      shaped,
      mlx::core::Shape{0, static_cast<int>(half_dim)},
      mlx::core::Shape{static_cast<int>(heads), static_cast<int>(head_dim)});
  auto first = x1 * cos_array - x2 * sin_array;
  auto second = x2 * cos_array + x1 * sin_array;
  auto rotated = mlx::core::concatenate({first, second}, 1);
  return mlx::core::reshape(
      rotated,
      mlx::core::Shape{1, static_cast<int>(heads * head_dim)});
}

std::pair<mlx::core::array, mlx::core::array> qk_norm_rope_mlx_arrays(
    const TensorGroupRecord& q_norm,
    const TensorGroupRecord& k_norm,
    const mlx::core::array& q_array,
    const mlx::core::array& k_array,
    double eps,
    std::uint64_t position) {
  const RopeRuntimeConfig& config = active_rope_config();
  if (!config.enabled) {
    return {q_array, k_array};
  }
  auto q_normed = rmsnorm_heads_mlx_array(
      q_norm,
      q_array,
      config.num_attention_heads,
      config.head_dim,
      eps);
  auto k_normed = rmsnorm_heads_mlx_array(
      k_norm,
      k_array,
      config.num_key_value_heads,
      config.head_dim,
      eps);
  return {
      rope_mlx_array(q_normed, config.num_attention_heads, config.head_dim, position, config.theta),
      rope_mlx_array(k_normed, config.num_key_value_heads, config.head_dim, position, config.theta)};
}
#endif

std::vector<std::pair<std::uint64_t, float>> top_logits(
    const std::vector<float>& logits,
    std::size_t count) {
  std::vector<std::pair<std::uint64_t, float>> top;
  top.reserve(count);
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const float score = logits[i];
    if (top.size() < count) {
      top.emplace_back(static_cast<std::uint64_t>(i), score);
      std::sort(top.begin(), top.end(), [](const auto& left, const auto& right) {
        return left.second > right.second;
      });
      continue;
    }
    if (score > top.back().second) {
      top.back() = {static_cast<std::uint64_t>(i), score};
      std::sort(top.begin(), top.end(), [](const auto& left, const auto& right) {
        return left.second > right.second;
      });
    }
  }
  return top;
}

struct SamplingConfig {
  bool enabled = false;
  double temperature = 0.0;
  std::uint64_t top_k = 0;
  double top_p = 1.0;
  std::uint64_t seed = 1234;
  std::mt19937_64 rng{1234};
};

thread_local SamplingConfig generation_sampling_config;

struct GenerationStopConfig {
  bool stop_on_eos = false;
  std::uint64_t eos_token_id = 0;
  std::vector<std::uint64_t> stop_token_ids;
};

thread_local GenerationStopConfig generation_stop_config;

struct RepetitionDiagnosticEntry {
  std::uint64_t token_position = 0;
  std::uint64_t generated_token_id = 0;
  std::string generated_token_text;
  std::uint64_t eos_token_id = 0;
  float eos_raw_logit = 0.0f;
  std::uint64_t eos_rank = 0;
  double eos_probability = 0.0;
  float selected_token_raw_logit = 0.0f;
  std::uint64_t selected_token_rank = 0;
  double selected_token_probability = 0.0;
  std::vector<std::pair<std::uint64_t, float>> top10;
  std::vector<double> top10_probabilities;
  double entropy = 0.0;
  double top1_minus_top2_logit_gap = 0.0;
  std::uint64_t repeated_token_streak_current = 0;
  std::uint64_t repeated_token_streak_max = 0;
  std::uint64_t low_id_token_count_so_far = 0;
  std::uint64_t numeric_token_count_so_far = 0;
  bool adapter_zeroed_comparison_available = false;
  std::string adapter_zeroed_comparison_reason =
      "not_available_without_parallel_no_adapter_decode_state";
};

bool repetition_diagnostic_enabled() {
  return env_truthy("RUSTY_REPETITION_DIAGNOSTIC");
}

bool token_is_numeric_or_low_id(std::uint64_t token_id) {
  return token_id <= 1000;
}

bool should_capture_repetition_diagnostic(
    std::uint64_t token_position,
    std::uint64_t requested_tokens) {
  if (!repetition_diagnostic_enabled()) {
    return false;
  }
  if (token_position == 128 || token_position == 256 || token_position == 512) {
    return true;
  }
  if (token_position > 500 && token_position % 100 == 0) {
    return true;
  }
  return requested_tokens >= 32 && token_position > requested_tokens - 32;
}

std::uint64_t logit_rank_desc(const std::vector<float>& logits, std::uint64_t token_id) {
  if (token_id >= logits.size()) {
    return 0;
  }
  const float target = logits[static_cast<std::size_t>(token_id)];
  std::uint64_t rank = 1;
  for (float value : logits) {
    if (value > target) {
      rank += 1;
    }
  }
  return rank;
}

std::vector<double> sampling_distribution_probabilities(
    const std::vector<float>& logits,
    const std::vector<std::pair<std::uint64_t, float>>& top) {
  std::vector<double> probabilities(top.size(), 0.0);
  if (top.empty()) {
    return probabilities;
  }
  const double temperature = generation_sampling_config.enabled
      ? std::max(generation_sampling_config.temperature, 1.0e-6)
      : 1.0;
  double max_scaled = -std::numeric_limits<double>::infinity();
  for (const auto& item : top) {
    max_scaled = std::max(max_scaled, static_cast<double>(item.second) / temperature);
  }
  double total = 0.0;
  for (std::size_t i = 0; i < top.size(); ++i) {
    probabilities[i] = std::exp((static_cast<double>(top[i].second) / temperature) - max_scaled);
    total += probabilities[i];
  }
  if (total > 0.0 && std::isfinite(total)) {
    for (double& value : probabilities) {
      value /= total;
    }
  }
  return probabilities;
}

double distribution_entropy(const std::vector<double>& probabilities) {
  double entropy = 0.0;
  for (double probability : probabilities) {
    if (probability > 0.0) {
      entropy -= probability * std::log(probability);
    }
  }
  return entropy;
}

RepetitionDiagnosticEntry make_repetition_diagnostic_entry(
    const std::vector<float>& logits,
    std::uint64_t selected_token_id,
    std::uint64_t token_position,
    std::uint64_t repeated_token_streak_current,
    std::uint64_t repeated_token_streak_max,
    std::uint64_t low_id_token_count_so_far,
    std::uint64_t numeric_token_count_so_far) {
  RepetitionDiagnosticEntry entry;
  entry.token_position = token_position;
  entry.generated_token_id = selected_token_id;
  entry.generated_token_text = std::to_string(selected_token_id);
  entry.eos_token_id = generation_stop_config.eos_token_id;
  if (generation_stop_config.eos_token_id < logits.size()) {
    entry.eos_raw_logit = logits[static_cast<std::size_t>(generation_stop_config.eos_token_id)];
    entry.eos_rank = logit_rank_desc(logits, generation_stop_config.eos_token_id);
  }
  if (selected_token_id < logits.size()) {
    entry.selected_token_raw_logit = logits[static_cast<std::size_t>(selected_token_id)];
    entry.selected_token_rank = logit_rank_desc(logits, selected_token_id);
  }
  const std::size_t top_count = std::min<std::size_t>(
      std::max<std::size_t>(10, static_cast<std::size_t>(generation_sampling_config.top_k)),
      logits.size());
  std::vector<std::pair<std::uint64_t, float>> distribution_top = top_logits(logits, top_count);
  std::vector<double> distribution_probabilities =
      sampling_distribution_probabilities(logits, distribution_top);
  entry.entropy = distribution_entropy(distribution_probabilities);
  entry.top10 = top_logits(logits, 10);
  entry.top10_probabilities.assign(entry.top10.size(), 0.0);
  for (std::size_t i = 0; i < entry.top10.size(); ++i) {
    const std::uint64_t token_id = entry.top10[i].first;
    for (std::size_t j = 0; j < distribution_top.size(); ++j) {
      if (distribution_top[j].first == token_id) {
        entry.top10_probabilities[i] = distribution_probabilities[j];
        break;
      }
    }
  }
  for (std::size_t i = 0; i < distribution_top.size(); ++i) {
    if (distribution_top[i].first == selected_token_id) {
      entry.selected_token_probability = distribution_probabilities[i];
    }
    if (distribution_top[i].first == generation_stop_config.eos_token_id) {
      entry.eos_probability = distribution_probabilities[i];
    }
  }
  if (entry.top10.size() >= 2) {
    entry.top1_minus_top2_logit_gap =
        static_cast<double>(entry.top10[0].second) -
        static_cast<double>(entry.top10[1].second);
  }
  entry.repeated_token_streak_current = repeated_token_streak_current;
  entry.repeated_token_streak_max = repeated_token_streak_max;
  entry.low_id_token_count_so_far = low_id_token_count_so_far;
  entry.numeric_token_count_so_far = numeric_token_count_so_far;
  return entry;
}

std::string repetition_diagnostics_json(const std::vector<RepetitionDiagnosticEntry>& entries) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < entries.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    const auto& entry = entries[i];
    out << "{"
        << "\"token_position\":" << entry.token_position << ","
        << "\"generated_token_id\":" << entry.generated_token_id << ","
        << "\"decoded_token_text\":\"" << json_escape(entry.generated_token_text) << "\","
        << "\"eos_token_id\":" << entry.eos_token_id << ","
        << "\"eos_raw_logit\":" << entry.eos_raw_logit << ","
        << "\"eos_rank\":" << entry.eos_rank << ","
        << "\"eos_softmax_probability\":" << entry.eos_probability << ","
        << "\"selected_token_raw_logit\":" << entry.selected_token_raw_logit << ","
        << "\"selected_token_rank\":" << entry.selected_token_rank << ","
        << "\"selected_token_probability\":" << entry.selected_token_probability << ","
        << "\"top10\":[";
    for (std::size_t j = 0; j < entry.top10.size(); ++j) {
      if (j > 0) {
        out << ",";
      }
      out << "{"
          << "\"token_id\":" << entry.top10[j].first << ","
          << "\"decoded_text\":\"" << entry.top10[j].first << "\","
          << "\"logit\":" << entry.top10[j].second << ","
          << "\"probability\":" << (j < entry.top10_probabilities.size() ? entry.top10_probabilities[j] : 0.0)
          << "}";
    }
    out << "],"
        << "\"entropy\":" << entry.entropy << ","
        << "\"top1_minus_top2_logit_gap\":" << entry.top1_minus_top2_logit_gap << ","
        << "\"repeated_token_streak_current\":" << entry.repeated_token_streak_current << ","
        << "\"repeated_token_streak_max\":" << entry.repeated_token_streak_max << ","
        << "\"low_id_token_count_so_far\":" << entry.low_id_token_count_so_far << ","
        << "\"numeric_token_count_so_far\":" << entry.numeric_token_count_so_far << ","
        << "\"base_no_adapter_comparison_available\":false,"
        << "\"base_no_adapter_comparison_reason\":\""
        << json_escape(entry.adapter_zeroed_comparison_reason) << "\","
        << "\"base_no_adapter_eos_rank\":null,"
        << "\"base_no_adapter_eos_logit\":null,"
        << "\"base_no_adapter_eos_probability\":null,"
        << "\"base_no_adapter_top10\":[],"
        << "\"adapter_delta_for_selected_token\":null,"
        << "\"adapter_delta_for_eos_token\":null,"
        << "\"eos_rank_improves_when_adapter_zeroed\":null"
        << "}";
  }
  out << "]";
  return out.str();
}

std::vector<std::uint64_t> parse_u64_csv(const char* raw) {
  std::vector<std::uint64_t> values;
  if (raw == nullptr) {
    return values;
  }
  std::stringstream stream(raw);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (item.empty()) {
      continue;
    }
    values.push_back(static_cast<std::uint64_t>(std::stoull(item)));
  }
  return values;
}

bool token_matches_stop(std::uint64_t token_id, std::string& reason) {
  if (generation_stop_config.stop_on_eos &&
      token_id == generation_stop_config.eos_token_id) {
    reason = "eos";
    return true;
  }
  if (std::find(
          generation_stop_config.stop_token_ids.begin(),
          generation_stop_config.stop_token_ids.end(),
          token_id) != generation_stop_config.stop_token_ids.end()) {
    reason = "stop_token";
    return true;
  }
  return false;
}

std::pair<std::uint64_t, float> select_next_token_from_logits(
    const std::vector<float>& logits) {
  if (!generation_sampling_config.enabled ||
      generation_sampling_config.temperature <= 0.0 ||
      generation_sampling_config.top_k == 0) {
    auto top = top_logits(logits, 1);
    if (top.empty()) {
      return {0, 0.0f};
    }
    return top[0];
  }

  auto top = top_logits(
      logits,
      std::min<std::size_t>(
          static_cast<std::size_t>(generation_sampling_config.top_k),
          logits.size()));
  if (top.empty()) {
    return {0, 0.0f};
  }

  const double temperature = std::max(generation_sampling_config.temperature, 1.0e-6);
  double max_scaled = -std::numeric_limits<double>::infinity();
  for (const auto& item : top) {
    max_scaled = std::max(max_scaled, static_cast<double>(item.second) / temperature);
  }

  std::vector<double> weights;
  weights.reserve(top.size());
  double total_weight = 0.0;
  for (const auto& item : top) {
    const double weight = std::exp((static_cast<double>(item.second) / temperature) - max_scaled);
    weights.push_back(weight);
    total_weight += weight;
  }
  if (!(total_weight > 0.0) || !std::isfinite(total_weight)) {
    return top[0];
  }

  std::uniform_real_distribution<double> dist(0.0, total_weight);
  double draw = dist(generation_sampling_config.rng);
  for (std::size_t i = 0; i < top.size(); ++i) {
    if (draw <= weights[i]) {
      return top[i];
    }
    draw -= weights[i];
  }
  return top.back();
}

#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
std::vector<std::pair<std::uint64_t, float>> top_logits_from_mlx_array(
    const mlx::core::array& logits,
    std::size_t count) {
  if (count == 0 || logits.size() == 0) {
    return {};
  }
  const int vocab = logits.size();
  const int k = static_cast<int>(std::min<std::size_t>(count, static_cast<std::size_t>(vocab)));
  auto sorted_indices = mlx::core::argsort(logits);
  auto top_indices = mlx::core::slice(
      sorted_indices,
      mlx::core::Shape{vocab - k},
      mlx::core::Shape{vocab});
  top_indices = mlx::core::astype(top_indices, mlx::core::uint32);
  auto top_scores = mlx::core::take(logits, top_indices);
  mlx::core::eval(top_indices, top_scores);
  mlx::core::synchronize();
  const std::uint32_t* id_data = top_indices.data<std::uint32_t>();
  const float* score_data = top_scores.data<float>();
  std::vector<std::pair<std::uint64_t, float>> out;
  out.reserve(static_cast<std::size_t>(k));
  for (int i = 0; i < k; ++i) {
    out.emplace_back(static_cast<std::uint64_t>(id_data[i]), score_data[i]);
  }
  std::sort(
      out.begin(),
      out.end(),
      [](const auto& a, const auto& b) {
        if (a.second == b.second) {
          return a.first < b.first;
        }
        return a.second > b.second;
      });
  return out;
}

std::pair<std::uint64_t, float> select_next_token_from_mlx_logits(
    const mlx::core::array& logits) {
  if (!generation_sampling_config.enabled ||
      generation_sampling_config.temperature <= 0.0 ||
      generation_sampling_config.top_k == 0) {
    auto top_index = mlx::core::argmax(logits);
    top_index = mlx::core::astype(top_index, mlx::core::uint32);
    auto top_score = mlx::core::take(logits, top_index);
    mlx::core::eval(top_index, top_score);
    mlx::core::synchronize();
    return {
        static_cast<std::uint64_t>(*top_index.data<std::uint32_t>()),
        *top_score.data<float>()};
  }

  auto top = top_logits_from_mlx_array(
      logits,
      std::min<std::size_t>(
          static_cast<std::size_t>(generation_sampling_config.top_k),
          static_cast<std::size_t>(logits.size())));
  if (top.empty()) {
    return {0, 0.0f};
  }

  const double temperature = std::max(generation_sampling_config.temperature, 1.0e-6);
  double max_scaled = -std::numeric_limits<double>::infinity();
  for (const auto& item : top) {
    max_scaled = std::max(max_scaled, static_cast<double>(item.second) / temperature);
  }

  std::vector<double> weights;
  weights.reserve(top.size());
  double total_weight = 0.0;
  for (const auto& item : top) {
    const double weight = std::exp((static_cast<double>(item.second) / temperature) - max_scaled);
    weights.push_back(weight);
    total_weight += weight;
  }
  if (!(total_weight > 0.0) || !std::isfinite(total_weight)) {
    return top[0];
  }

  std::uniform_real_distribution<double> dist(0.0, total_weight);
  double draw = dist(generation_sampling_config.rng);
  for (std::size_t i = 0; i < top.size(); ++i) {
    if (draw <= weights[i]) {
      return top[i];
    }
    draw -= weights[i];
  }
  return top.back();
}

mlx::core::array logits_array_from_mlx_final_norm(
    const TensorGroupRecord& embedding,
    const mlx::core::array& final_norm_mlx) {
  LocalQuantizedMlxArrays logits_local;
  return make_quantized_matmul_from_mlx_input(
      embedding,
      final_norm_mlx,
      logits_local,
      final_norm_mlx.size());
}
#endif

std::vector<float> rmsnorm_values(
    const TensorGroupRecord& record,
    const std::vector<float>& input_values,
    double eps) {
  const auto weight_it = record.tensors.find("weight");
  if (weight_it == record.tensors.end()) {
    throw std::runtime_error("RMSNorm group is missing weight");
  }
  const auto weight_values = load_bf16_vector_from_tensor(weight_it->second);
  if (weight_values.size() != input_values.size()) {
    throw std::runtime_error("RMSNorm weight size does not match input");
  }

  double mean_sq = 0.0;
  for (float value : input_values) {
    mean_sq += static_cast<double>(value) * static_cast<double>(value);
  }
  mean_sq /= static_cast<double>(input_values.size());
  const double scale = 1.0 / std::sqrt(mean_sq + eps);

  std::vector<float> output_values(input_values.size(), 0.0f);
  for (std::size_t i = 0; i < input_values.size(); ++i) {
    output_values[i] = input_values[i] * static_cast<float>(scale) * weight_values[i];
  }
  return output_values;
}

std::vector<float> rmsnorm_heads_values(
    const TensorGroupRecord& record,
    const std::vector<float>& input_values,
    std::size_t heads,
    std::size_t head_dim,
    double eps) {
  const auto weight_it = record.tensors.find("weight");
  if (weight_it == record.tensors.end()) {
    throw std::runtime_error("per-head RMSNorm group is missing weight");
  }
  const auto weight_values = load_bf16_vector_from_tensor(weight_it->second);
  if (heads == 0 || head_dim == 0 || input_values.size() != heads * head_dim) {
    throw std::runtime_error("per-head RMSNorm input shape mismatch");
  }
  if (weight_values.size() != head_dim) {
    throw std::runtime_error("per-head RMSNorm weight size does not match head_dim");
  }

  std::vector<float> output_values(input_values.size(), 0.0f);
  for (std::size_t head = 0; head < heads; ++head) {
    const std::size_t head_offset = head * head_dim;
    double mean_sq = 0.0;
    for (std::size_t i = 0; i < head_dim; ++i) {
      const float value = input_values[head_offset + i];
      mean_sq += static_cast<double>(value) * static_cast<double>(value);
    }
    mean_sq /= static_cast<double>(head_dim);
    const double scale = 1.0 / std::sqrt(mean_sq + eps);
    for (std::size_t i = 0; i < head_dim; ++i) {
      output_values[head_offset + i] =
          input_values[head_offset + i] * static_cast<float>(scale) * weight_values[i];
    }
  }
  return output_values;
}

void apply_qk_norm_in_place(
    const TensorGroupRecord& q_norm,
    const TensorGroupRecord& k_norm,
    std::vector<float>& q_values,
    std::vector<float>& k_values,
    double eps) {
  const RopeRuntimeConfig& config = active_rope_config();
  if (!config.enabled) {
    return;
  }
  q_values = rmsnorm_heads_values(
      q_norm,
      q_values,
      config.num_attention_heads,
      config.head_dim,
      eps);
  k_values = rmsnorm_heads_values(
      k_norm,
      k_values,
      config.num_key_value_heads,
      config.head_dim,
      eps);
}

RopeRuntimeConfig rope_config_from_json(const std::string& config_json) {
  RopeRuntimeConfig config;
  std::uint64_t value = 0;
  if (extract_json_u64_value(config_json, "head_dim", value) && value > 0) {
    config.head_dim = static_cast<std::size_t>(value);
  }
  if (extract_json_u64_value(config_json, "num_attention_heads", value) && value > 0) {
    config.num_attention_heads = static_cast<std::size_t>(value);
  }
  if (extract_json_u64_value(config_json, "num_key_value_heads", value) && value > 0) {
    config.num_key_value_heads = static_cast<std::size_t>(value);
  }
  extract_json_u64_value(config_json, "max_position_embeddings", config.max_position_embeddings);
  extract_json_double_value(config_json, "rope_theta", config.theta);
  config.enabled =
      config.head_dim > 0 &&
      config.head_dim % 2 == 0 &&
      config.num_attention_heads > 0 &&
      config.num_key_value_heads > 0 &&
      config.theta > 0.0;
  return config;
}

void apply_rope_in_place(
    std::vector<float>& values,
    std::size_t heads,
    std::size_t head_dim,
    std::uint64_t position,
    double theta) {
  if (heads == 0 || head_dim == 0 || head_dim % 2 != 0 || theta <= 0.0) {
    return;
  }
  if (values.size() != heads * head_dim) {
    return;
  }
  const std::size_t half_dim = head_dim / 2;
  for (std::size_t head = 0; head < heads; ++head) {
    const std::size_t head_offset = head * head_dim;
    for (std::size_t i = 0; i < half_dim; ++i) {
      const double inv_freq = 1.0 / std::pow(theta, static_cast<double>(i * 2) / static_cast<double>(head_dim));
      const double angle = static_cast<double>(position) * inv_freq;
      const float cos_value = static_cast<float>(std::cos(angle));
      const float sin_value = static_cast<float>(std::sin(angle));
      const float x1 = values[head_offset + i];
      const float x2 = values[head_offset + half_dim + i];
      values[head_offset + i] = x1 * cos_value - x2 * sin_value;
      values[head_offset + half_dim + i] = x2 * cos_value + x1 * sin_value;
    }
  }
}

void apply_rope_interleaved_in_place(
    std::vector<float>& values,
    std::size_t heads,
    std::size_t head_dim,
    std::uint64_t position,
    double theta) {
  if (heads == 0 || head_dim == 0 || head_dim % 2 != 0 || theta <= 0.0) {
    return;
  }
  if (values.size() != heads * head_dim) {
    return;
  }
  for (std::size_t head = 0; head < heads; ++head) {
    const std::size_t head_offset = head * head_dim;
    for (std::size_t i = 0; i < head_dim; i += 2) {
      const double inv_freq = 1.0 / std::pow(theta, static_cast<double>(i) / static_cast<double>(head_dim));
      const double angle = static_cast<double>(position) * inv_freq;
      const float cos_value = static_cast<float>(std::cos(angle));
      const float sin_value = static_cast<float>(std::sin(angle));
      const float x1 = values[head_offset + i];
      const float x2 = values[head_offset + i + 1];
      values[head_offset + i] = x1 * cos_value - x2 * sin_value;
      values[head_offset + i + 1] = x2 * cos_value + x1 * sin_value;
    }
  }
}

void apply_active_rope_to_qk(
    std::vector<float>& q_values,
    std::vector<float>& k_values,
    std::uint64_t position) {
  const RopeRuntimeConfig& config = active_rope_config();
  if (!config.enabled) {
    return;
  }
  apply_rope_in_place(
      q_values,
      config.num_attention_heads,
      config.head_dim,
      position,
      config.theta);
  apply_rope_in_place(
      k_values,
      config.num_key_value_heads,
      config.head_dim,
      position,
      config.theta);
}

#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
mlx::core::array rmsnorm_mlx_array(
    const TensorGroupRecord& record,
    const mlx::core::array& input,
    double eps) {
  const auto weight_it = record.tensors.find("weight");
  if (weight_it == record.tensors.end()) {
    throw std::runtime_error("RMSNorm group is missing weight");
  }
  const auto& weight_tensor = weight_it->second;
  const std::size_t hidden_size = weight_tensor.shape.empty()
      ? 0
      : static_cast<std::size_t>(weight_tensor.shape[0]);
  if (hidden_size == 0 || static_cast<std::size_t>(input.size()) != hidden_size) {
    throw std::runtime_error("RMSNorm MLX input length mismatch");
  }
  mlx::core::array weight_bf16(
      reinterpret_cast<const mlx::core::bfloat16_t*>(tensor_payload_data(weight_tensor)),
      mlx::core::Shape{static_cast<int>(hidden_size)},
      mlx::core::bfloat16);
  auto weight = mlx::core::astype(weight_bf16, mlx::core::float32);
  auto mean_square = mlx::core::mean(mlx::core::square(input), 1, true);
  return input * mlx::core::rsqrt(mean_square + static_cast<float>(eps)) * weight;
}
#endif

std::vector<float> single_token_attention_values(
    const std::vector<float>& q_values,
    const std::vector<float>& k_values,
    const std::vector<float>& v_values,
    double& qk_score_checksum) {
  const std::size_t head_dim = 128;
  if (q_values.empty() || k_values.empty() || v_values.empty() ||
      q_values.size() % head_dim != 0 ||
      k_values.size() % head_dim != 0 ||
      v_values.size() % head_dim != 0 ||
      k_values.size() != v_values.size()) {
    throw std::runtime_error("single-token attention probe received invalid q/k/v shapes");
  }
  const std::size_t q_heads = q_values.size() / head_dim;
  const std::size_t kv_heads = k_values.size() / head_dim;
  if (kv_heads == 0 || q_heads % kv_heads != 0) {
    throw std::runtime_error("single-token attention probe requires grouped query heads");
  }
  const std::size_t repeat = q_heads / kv_heads;
  std::vector<float> attention(q_values.size(), 0.0f);
  qk_score_checksum = 0.0;

  for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
    const std::size_t kv_head = q_head / repeat;
    double score = 0.0;
    for (std::size_t i = 0; i < head_dim; ++i) {
      const float q = q_values[q_head * head_dim + i];
      const float k = k_values[kv_head * head_dim + i];
      const float v = v_values[kv_head * head_dim + i];
      score += static_cast<double>(q) * static_cast<double>(k);
      attention[q_head * head_dim + i] = v;
    }
    qk_score_checksum += score / std::sqrt(static_cast<double>(head_dim));
  }

  return attention;
}

void append_native_kv(
    NativeSessionKvCache* cache,
    std::size_t layer,
    const std::vector<float>& k_values,
    const std::vector<float>& v_values) {
  if (cache == nullptr) {
    return;
  }
  if (layer >= cache->layers.size()) {
    cache->layers.resize(layer + 1);
  }
  if (keep_cpu_kv_mirror()) {
    cache->layers[layer].keys.push_back(k_values);
    cache->layers[layer].values.push_back(v_values);
  }
  cache->layers_allocated = std::max<std::uint64_t>(
      cache->layers_allocated,
      static_cast<std::uint64_t>(cache->layers.size()));
  ChunkedExpandedKvCache layer_cache(cache->layers[layer]);
  const std::uint64_t layer_len =
      keep_cpu_kv_mirror()
          ? static_cast<std::uint64_t>(cache->layers[layer].keys.size())
          : layer_cache.len();
  cache->positions_stored = std::max<std::uint64_t>(cache->positions_stored, layer_len);
}

std::vector<float> cached_single_token_attention_values(
    const std::vector<float>& q_values,
    const NativeLayerKvCache& layer_cache,
    const std::vector<float>& current_k_values,
    const std::vector<float>& current_v_values,
    CachedAttentionTiming* timing = nullptr) {
  const std::size_t head_dim = 128;
  if (q_values.empty() || current_k_values.empty() || current_v_values.empty() ||
      q_values.size() % head_dim != 0 ||
      current_k_values.size() % head_dim != 0 ||
      current_v_values.size() % head_dim != 0 ||
      current_k_values.size() != current_v_values.size()) {
    throw std::runtime_error("incremental attention received invalid q/k/v shapes");
  }
  const std::size_t q_heads = q_values.size() / head_dim;
  const std::size_t kv_heads = current_k_values.size() / head_dim;
  if (kv_heads == 0 || q_heads % kv_heads != 0) {
    throw std::runtime_error("incremental attention requires grouped query heads");
  }
  if (layer_cache.keys.size() != layer_cache.values.size()) {
    throw std::runtime_error("incremental attention cache key/value position counts differ");
  }
  for (const auto& key : layer_cache.keys) {
    if (key.size() != current_k_values.size()) {
      throw std::runtime_error("incremental attention cached key shape mismatch");
    }
  }
  for (const auto& value : layer_cache.values) {
    if (value.size() != current_v_values.size()) {
      throw std::runtime_error("incremental attention cached value shape mismatch");
    }
  }

  const std::size_t prior_positions = layer_cache.keys.size();
  const std::size_t total_positions = prior_positions + 1;
  const std::size_t repeat = q_heads / kv_heads;
  const double inv_sqrt_dim = 1.0 / std::sqrt(static_cast<double>(head_dim));
  std::vector<float> attention(q_values.size(), 0.0f);

  for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
    const std::size_t kv_head = q_head / repeat;
    std::vector<double> scores(total_positions, 0.0);
    double max_score = -std::numeric_limits<double>::infinity();

    auto score_start = std::chrono::steady_clock::now();
    for (std::size_t position = 0; position < total_positions; ++position) {
      const std::vector<float>& key =
          position < prior_positions ? layer_cache.keys[position] : current_k_values;
      double score = 0.0;
      for (std::size_t i = 0; i < head_dim; ++i) {
        score += static_cast<double>(q_values[q_head * head_dim + i]) *
                 static_cast<double>(key[kv_head * head_dim + i]);
      }
      score *= inv_sqrt_dim;
      scores[position] = score;
      max_score = std::max(max_score, score);
    }
    if (timing != nullptr) {
      add_elapsed_ms(timing->score_ms, score_start);
    }

    double denom = 0.0;
    auto softmax_start = std::chrono::steady_clock::now();
    for (double& score : scores) {
      score = std::exp(score - max_score);
      denom += score;
    }
    if (denom <= 0.0 || !std::isfinite(denom)) {
      throw std::runtime_error("incremental attention softmax denominator invalid");
    }
    if (timing != nullptr) {
      add_elapsed_ms(timing->softmax_ms, softmax_start);
    }

    auto value_mix_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < head_dim; ++i) {
      double value_sum = 0.0;
      for (std::size_t position = 0; position < total_positions; ++position) {
        const std::vector<float>& value =
            position < prior_positions ? layer_cache.values[position] : current_v_values;
        value_sum += (scores[position] / denom) *
                     static_cast<double>(value[kv_head * head_dim + i]);
      }
      attention[q_head * head_dim + i] = static_cast<float>(value_sum);
    }
    if (timing != nullptr) {
      add_elapsed_ms(timing->value_mix_ms, value_mix_start);
    }
  }

  return attention;
}

#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
struct MlxSingleHeadAttentionProbeResult {
  std::vector<float> output;
  std::vector<float> scores;
  std::vector<float> probabilities;
};

MlxSingleHeadAttentionProbeResult mlx_single_head_attention_values_probe(
    const std::vector<float>& q_values,
    const NativeLayerKvCache& layer_cache,
    const std::vector<float>& current_k_values,
    const std::vector<float>& current_v_values,
    std::size_t q_head,
    std::size_t kv_head,
    std::size_t head_dim) {
  const std::size_t prior_positions = layer_cache.keys.size();
  const std::size_t total_positions = prior_positions + 1;
  if ((q_head + 1) * head_dim > q_values.size() ||
      (kv_head + 1) * head_dim > current_k_values.size() ||
      current_k_values.size() != current_v_values.size()) {
    throw std::runtime_error("MLX single-head attention probe received invalid shape");
  }
  std::vector<float> q_head_values(head_dim, 0.0f);
  std::vector<float> k_values(total_positions * head_dim, 0.0f);
  std::vector<float> v_values(total_positions * head_dim, 0.0f);
  std::copy(
      q_values.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim),
      q_values.begin() + static_cast<std::ptrdiff_t>((q_head + 1) * head_dim),
      q_head_values.begin());
  for (std::size_t position = 0; position < total_positions; ++position) {
    const std::vector<float>& key =
        position < prior_positions ? layer_cache.keys[position] : current_k_values;
    const std::vector<float>& value =
        position < prior_positions ? layer_cache.values[position] : current_v_values;
    std::copy(
        key.begin() + static_cast<std::ptrdiff_t>(kv_head * head_dim),
        key.begin() + static_cast<std::ptrdiff_t>((kv_head + 1) * head_dim),
        k_values.begin() + static_cast<std::ptrdiff_t>(position * head_dim));
    std::copy(
        value.begin() + static_cast<std::ptrdiff_t>(kv_head * head_dim),
        value.begin() + static_cast<std::ptrdiff_t>((kv_head + 1) * head_dim),
        v_values.begin() + static_cast<std::ptrdiff_t>(position * head_dim));
  }

  const float inv_sqrt_dim =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  mlx::core::array q_array(
      q_head_values.data(),
      mlx::core::Shape{1, static_cast<int>(head_dim)},
      mlx::core::float32);
  mlx::core::array k_array(
      k_values.data(),
      mlx::core::Shape{static_cast<int>(total_positions), static_cast<int>(head_dim)},
      mlx::core::float32);
  mlx::core::array v_array(
      v_values.data(),
      mlx::core::Shape{static_cast<int>(total_positions), static_cast<int>(head_dim)},
      mlx::core::float32);
  auto scores_array =
      mlx::core::matmul(q_array, mlx::core::transpose(k_array)) * inv_sqrt_dim;
  auto probabilities_array = mlx::core::softmax(scores_array, -1);
  auto output_array = mlx::core::matmul(probabilities_array, v_array);
  mlx::core::eval(scores_array, probabilities_array, output_array);
  mlx::core::synchronize();

  const float* scores_data = scores_array.data<float>();
  const float* probabilities_data = probabilities_array.data<float>();
  const float* output_data = output_array.data<float>();
  MlxSingleHeadAttentionProbeResult result;
  result.scores.assign(scores_data, scores_data + total_positions);
  result.probabilities.assign(probabilities_data, probabilities_data + total_positions);
  result.output.assign(output_data, output_data + head_dim);
  return result;
}

std::vector<float> mlx_single_token_attention_values_probe(
    const std::vector<float>& q_values,
    const NativeLayerKvCache& layer_cache,
    const std::vector<float>& current_k_values,
    const std::vector<float>& current_v_values,
    double* timing_ms = nullptr) {
  const std::size_t head_dim = 128;
  if (q_values.empty() || current_k_values.empty() || current_v_values.empty() ||
      q_values.size() % head_dim != 0 ||
      current_k_values.size() % head_dim != 0 ||
      current_v_values.size() % head_dim != 0 ||
      current_k_values.size() != current_v_values.size()) {
    throw std::runtime_error("MLX attention probe received invalid q/k/v shapes");
  }
  const std::size_t q_heads = q_values.size() / head_dim;
  const std::size_t kv_heads = current_k_values.size() / head_dim;
  if (kv_heads == 0 || q_heads % kv_heads != 0) {
    throw std::runtime_error("MLX attention probe requires grouped query heads");
  }
  const std::size_t prior_positions = layer_cache.keys.size();
  const std::size_t total_positions = prior_positions + 1;
  const std::size_t repeat = q_heads / kv_heads;
  const float inv_sqrt_dim = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  std::vector<float> attention(q_values.size(), 0.0f);
  auto start = std::chrono::steady_clock::now();

  for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
    const std::size_t kv_head = q_head / repeat;
    auto head_result = mlx_single_head_attention_values_probe(
        q_values,
        layer_cache,
        current_k_values,
        current_v_values,
        q_head,
        kv_head,
        head_dim);
    std::copy(
        head_result.output.begin(),
        head_result.output.end(),
        attention.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim));
  }

  if (timing_ms != nullptr) {
    *timing_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
  return attention;
}

std::vector<float> mlx_grouped_attention_values_probe(
    const std::vector<float>& q_values,
    const NativeLayerKvCache& layer_cache,
    const std::vector<float>& current_k_values,
    const std::vector<float>& current_v_values,
    double* timing_ms = nullptr) {
  const std::size_t head_dim = 128;
  if (q_values.empty() || current_k_values.empty() || current_v_values.empty() ||
      q_values.size() % head_dim != 0 ||
      current_k_values.size() % head_dim != 0 ||
      current_v_values.size() % head_dim != 0 ||
      current_k_values.size() != current_v_values.size()) {
    throw std::runtime_error("MLX grouped attention probe received invalid q/k/v shapes");
  }
  const std::size_t q_heads = q_values.size() / head_dim;
  const std::size_t kv_heads = current_k_values.size() / head_dim;
  if (kv_heads == 0 || q_heads % kv_heads != 0) {
    throw std::runtime_error("MLX grouped attention probe requires grouped query heads");
  }
  const std::size_t prior_positions = layer_cache.keys.size();
  const std::size_t total_positions = prior_positions + 1;
  const std::size_t q_heads_per_kv = q_heads / kv_heads;
  const float inv_sqrt_dim = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  std::vector<float> attention(q_values.size(), 0.0f);
  auto start = std::chrono::steady_clock::now();

  for (std::size_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
    std::vector<float> q_group_values(q_heads_per_kv * head_dim, 0.0f);
    std::vector<float> k_values(total_positions * head_dim, 0.0f);
    std::vector<float> v_values(total_positions * head_dim, 0.0f);
    for (std::size_t local_q = 0; local_q < q_heads_per_kv; ++local_q) {
      const std::size_t q_head = kv_head * q_heads_per_kv + local_q;
      std::copy(
          q_values.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim),
          q_values.begin() + static_cast<std::ptrdiff_t>((q_head + 1) * head_dim),
          q_group_values.begin() + static_cast<std::ptrdiff_t>(local_q * head_dim));
    }
    for (std::size_t position = 0; position < total_positions; ++position) {
      const std::vector<float>& key =
          position < prior_positions ? layer_cache.keys[position] : current_k_values;
      const std::vector<float>& value =
          position < prior_positions ? layer_cache.values[position] : current_v_values;
      std::copy(
          key.begin() + static_cast<std::ptrdiff_t>(kv_head * head_dim),
          key.begin() + static_cast<std::ptrdiff_t>((kv_head + 1) * head_dim),
          k_values.begin() + static_cast<std::ptrdiff_t>(position * head_dim));
      std::copy(
          value.begin() + static_cast<std::ptrdiff_t>(kv_head * head_dim),
          value.begin() + static_cast<std::ptrdiff_t>((kv_head + 1) * head_dim),
          v_values.begin() + static_cast<std::ptrdiff_t>(position * head_dim));
    }

    mlx::core::array q_array(
        q_group_values.data(),
        mlx::core::Shape{static_cast<int>(q_heads_per_kv), static_cast<int>(head_dim)},
        mlx::core::float32);
    mlx::core::array k_array(
        k_values.data(),
        mlx::core::Shape{static_cast<int>(total_positions), static_cast<int>(head_dim)},
        mlx::core::float32);
    mlx::core::array v_array(
        v_values.data(),
        mlx::core::Shape{static_cast<int>(total_positions), static_cast<int>(head_dim)},
        mlx::core::float32);
    auto scores = mlx::core::matmul(q_array, mlx::core::transpose(k_array)) * inv_sqrt_dim;
    auto probs = mlx::core::softmax(scores, -1);
    auto mixed = mlx::core::matmul(probs, v_array);
    mlx::core::eval(scores, probs, mixed);
    mlx::core::synchronize();
    const float* mixed_data = mixed.data<float>();
    for (std::size_t local_q = 0; local_q < q_heads_per_kv; ++local_q) {
      const std::size_t q_head = kv_head * q_heads_per_kv + local_q;
      std::copy(
          mixed_data + static_cast<std::ptrdiff_t>(local_q * head_dim),
          mixed_data + static_cast<std::ptrdiff_t>((local_q + 1) * head_dim),
          attention.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim));
    }
  }

  if (timing_ms != nullptr) {
    *timing_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
  return attention;
}

std::vector<float> mlx_all_head_batched_attention_values_probe(
    const std::vector<float>& q_values,
    const NativeLayerKvCache& layer_cache,
    const std::vector<float>& current_k_values,
    const std::vector<float>& current_v_values,
    double* timing_ms = nullptr) {
  const std::size_t head_dim = 128;
  if (q_values.empty() || current_k_values.empty() || current_v_values.empty() ||
      q_values.size() % head_dim != 0 ||
      current_k_values.size() % head_dim != 0 ||
      current_v_values.size() % head_dim != 0 ||
      current_k_values.size() != current_v_values.size()) {
    throw std::runtime_error("MLX all-head attention probe received invalid q/k/v shapes");
  }
  const std::size_t q_heads = q_values.size() / head_dim;
  const std::size_t kv_heads = current_k_values.size() / head_dim;
  if (kv_heads == 0 || q_heads % kv_heads != 0) {
    throw std::runtime_error("MLX all-head attention probe requires grouped query heads");
  }
  const std::size_t prior_positions = layer_cache.keys.size();
  const std::size_t total_positions = prior_positions + 1;
  const std::size_t q_heads_per_kv = q_heads / kv_heads;
  const float inv_sqrt_dim = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  std::vector<float> kq_values(q_heads * total_positions * head_dim, 0.0f);
  std::vector<float> vq_values(q_heads * total_positions * head_dim, 0.0f);
  for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
    const std::size_t kv_head = q_head / q_heads_per_kv;
    for (std::size_t position = 0; position < total_positions; ++position) {
      const std::vector<float>& key =
          position < prior_positions ? layer_cache.keys[position] : current_k_values;
      const std::vector<float>& value =
          position < prior_positions ? layer_cache.values[position] : current_v_values;
      const std::size_t dst = (q_head * total_positions + position) * head_dim;
      std::copy(
          key.begin() + static_cast<std::ptrdiff_t>(kv_head * head_dim),
          key.begin() + static_cast<std::ptrdiff_t>((kv_head + 1) * head_dim),
          kq_values.begin() + static_cast<std::ptrdiff_t>(dst));
      std::copy(
          value.begin() + static_cast<std::ptrdiff_t>(kv_head * head_dim),
          value.begin() + static_cast<std::ptrdiff_t>((kv_head + 1) * head_dim),
          vq_values.begin() + static_cast<std::ptrdiff_t>(dst));
    }
  }

  auto start = std::chrono::steady_clock::now();
  mlx::core::array q_array(
      q_values.data(),
      mlx::core::Shape{static_cast<int>(q_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  mlx::core::array k_array(
      kq_values.data(),
      mlx::core::Shape{
          static_cast<int>(q_heads),
          static_cast<int>(total_positions),
          static_cast<int>(head_dim)},
      mlx::core::float32);
  mlx::core::array v_array(
      vq_values.data(),
      mlx::core::Shape{
          static_cast<int>(q_heads),
          static_cast<int>(total_positions),
          static_cast<int>(head_dim)},
      mlx::core::float32);
  auto scores = mlx::core::matmul(q_array, mlx::core::transpose(k_array, {0, 2, 1})) * inv_sqrt_dim;
  auto probs = mlx::core::softmax(scores, -1);
  auto mixed = mlx::core::matmul(probs, v_array);
  mixed.eval();
  mlx::core::synchronize();
  if (timing_ms != nullptr) {
    *timing_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
  const float* mixed_data = mixed.data<float>();
  std::vector<float> attention(q_values.size(), 0.0f);
  for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
    std::copy(
        mixed_data + static_cast<std::ptrdiff_t>(q_head * head_dim),
        mixed_data + static_cast<std::ptrdiff_t>((q_head + 1) * head_dim),
        attention.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim));
  }
  return attention;
}

std::vector<float> mlx_fast_attention_values_probe(
    const std::vector<float>& q_values,
    const NativeLayerKvCache& layer_cache,
    const std::vector<float>& current_k_values,
    const std::vector<float>& current_v_values,
    double* timing_ms = nullptr) {
  const std::size_t head_dim = 128;
  if (q_values.empty() || current_k_values.empty() || current_v_values.empty() ||
      q_values.size() % head_dim != 0 ||
      current_k_values.size() % head_dim != 0 ||
      current_v_values.size() % head_dim != 0 ||
      current_k_values.size() != current_v_values.size()) {
    throw std::runtime_error("MLX fast attention probe received invalid q/k/v shapes");
  }
  const std::size_t q_heads = q_values.size() / head_dim;
  const std::size_t kv_heads = current_k_values.size() / head_dim;
  if (kv_heads == 0 || q_heads % kv_heads != 0) {
    throw std::runtime_error("MLX fast attention probe requires grouped query heads");
  }
  const std::size_t prior_positions = layer_cache.keys.size();
  const std::size_t total_positions = prior_positions + 1;
  const std::size_t q_heads_per_kv = q_heads / kv_heads;
  std::vector<float> kq_values(q_heads * total_positions * head_dim, 0.0f);
  std::vector<float> vq_values(q_heads * total_positions * head_dim, 0.0f);
  for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
    const std::size_t kv_head = q_head / q_heads_per_kv;
    for (std::size_t position = 0; position < total_positions; ++position) {
      const std::vector<float>& key =
          position < prior_positions ? layer_cache.keys[position] : current_k_values;
      const std::vector<float>& value =
          position < prior_positions ? layer_cache.values[position] : current_v_values;
      const std::size_t dst = (q_head * total_positions + position) * head_dim;
      std::copy(
          key.begin() + static_cast<std::ptrdiff_t>(kv_head * head_dim),
          key.begin() + static_cast<std::ptrdiff_t>((kv_head + 1) * head_dim),
          kq_values.begin() + static_cast<std::ptrdiff_t>(dst));
      std::copy(
          value.begin() + static_cast<std::ptrdiff_t>(kv_head * head_dim),
          value.begin() + static_cast<std::ptrdiff_t>((kv_head + 1) * head_dim),
          vq_values.begin() + static_cast<std::ptrdiff_t>(dst));
    }
  }

  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  auto start = std::chrono::steady_clock::now();
  mlx::core::array q_array(
      q_values.data(),
      mlx::core::Shape{1, static_cast<int>(q_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  mlx::core::array k_array(
      kq_values.data(),
      mlx::core::Shape{
          1,
          static_cast<int>(q_heads),
          static_cast<int>(total_positions),
          static_cast<int>(head_dim)},
      mlx::core::float32);
  mlx::core::array v_array(
      vq_values.data(),
      mlx::core::Shape{
          1,
          static_cast<int>(q_heads),
          static_cast<int>(total_positions),
          static_cast<int>(head_dim)},
      mlx::core::float32);
  auto output = mlx::core::fast::scaled_dot_product_attention(
      q_array,
      k_array,
      v_array,
      scale,
      "");
  output.eval();
  mlx::core::synchronize();
  if (timing_ms != nullptr) {
    *timing_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
  const float* output_data = output.data<float>();
  return std::vector<float>(output_data, output_data + q_values.size());
}

struct MlxResidentKvCacheProbe {
  std::optional<mlx::core::array> keys;
  std::optional<mlx::core::array> values;
  std::size_t current_len = 0;
  double append_ms = 0.0;
  std::string append_mode = "concat_probe_only";
};

struct MlxExpandedResidentKvCacheProbe {
  std::optional<mlx::core::array> keys;
  std::optional<mlx::core::array> values;
  std::size_t current_len = 0;
  double append_ms = 0.0;
  std::string append_mode = "expanded_q_heads_concat_probe_only";
};

void mlx_resident_kv_cache_append_probe(
    MlxResidentKvCacheProbe& cache,
    const std::vector<float>& current_k_values,
    const std::vector<float>& current_v_values,
    std::size_t kv_heads,
    std::size_t head_dim) {
  if (current_k_values.size() != kv_heads * head_dim ||
      current_v_values.size() != kv_heads * head_dim) {
    throw std::runtime_error("MLX resident KV append probe received invalid k/v shapes");
  }
  auto start = std::chrono::steady_clock::now();
  mlx::core::array current_k(
      current_k_values.data(),
      mlx::core::Shape{static_cast<int>(kv_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  mlx::core::array current_v(
      current_v_values.data(),
      mlx::core::Shape{static_cast<int>(kv_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  if (cache.keys.has_value() && cache.values.has_value()) {
    cache.keys = mlx::core::concatenate({*cache.keys, current_k}, 1);
    cache.values = mlx::core::concatenate({*cache.values, current_v}, 1);
  } else {
    cache.keys = current_k;
    cache.values = current_v;
  }
  cache.keys->eval();
  cache.values->eval();
  mlx::core::synchronize();
  cache.current_len += 1;
  cache.append_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
}

std::vector<float> mlx_resident_kv_attention_values_probe(
    const std::vector<float>& q_values,
    const MlxResidentKvCacheProbe& cache,
    std::size_t q_heads,
    std::size_t kv_heads,
    std::size_t head_dim,
    double* timing_ms = nullptr) {
  if (!cache.keys.has_value() || !cache.values.has_value() ||
      q_values.size() != q_heads * head_dim ||
      q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0) {
    throw std::runtime_error("MLX resident KV attention probe received invalid state");
  }
  const std::size_t q_heads_per_kv = q_heads / kv_heads;
  std::vector<std::uint32_t> kv_indices(q_heads, 0);
  for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
    kv_indices[q_head] = static_cast<std::uint32_t>(q_head / q_heads_per_kv);
  }
  const float inv_sqrt_dim =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  auto start = std::chrono::steady_clock::now();
  mlx::core::array q_array(
      q_values.data(),
      mlx::core::Shape{static_cast<int>(q_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  mlx::core::array indices_array(
      kv_indices.data(),
      mlx::core::Shape{static_cast<int>(q_heads)},
      mlx::core::uint32);
  auto kq = mlx::core::take(*cache.keys, indices_array, 0);
  auto vq = mlx::core::take(*cache.values, indices_array, 0);
  auto scores = mlx::core::matmul(q_array, mlx::core::transpose(kq, {0, 2, 1})) * inv_sqrt_dim;
  auto probs = mlx::core::softmax(scores, -1);
  auto mixed = mlx::core::matmul(probs, vq);
  mixed.eval();
  mlx::core::synchronize();
  if (timing_ms != nullptr) {
    *timing_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
  const float* mixed_data = mixed.data<float>();
  std::vector<float> attention(q_values.size(), 0.0f);
  for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
    std::copy(
        mixed_data + static_cast<std::ptrdiff_t>(q_head * head_dim),
        mixed_data + static_cast<std::ptrdiff_t>((q_head + 1) * head_dim),
        attention.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim));
  }
  return attention;
}

void mlx_expanded_resident_kv_cache_append_probe(
    MlxExpandedResidentKvCacheProbe& cache,
    const std::vector<float>& current_k_values,
    const std::vector<float>& current_v_values,
    std::size_t q_heads,
    std::size_t kv_heads,
    std::size_t head_dim) {
  if (kv_heads == 0 || q_heads == 0 || q_heads % kv_heads != 0 ||
      current_k_values.size() != kv_heads * head_dim ||
      current_v_values.size() != kv_heads * head_dim) {
    throw std::runtime_error("MLX expanded resident KV append probe received invalid k/v shapes");
  }
  const std::size_t q_heads_per_kv = q_heads / kv_heads;
  std::vector<float> expanded_k(q_heads * head_dim, 0.0f);
  std::vector<float> expanded_v(q_heads * head_dim, 0.0f);
  for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
    const std::size_t kv_head = q_head / q_heads_per_kv;
    std::copy(
        current_k_values.begin() + static_cast<std::ptrdiff_t>(kv_head * head_dim),
        current_k_values.begin() + static_cast<std::ptrdiff_t>((kv_head + 1) * head_dim),
        expanded_k.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim));
    std::copy(
        current_v_values.begin() + static_cast<std::ptrdiff_t>(kv_head * head_dim),
        current_v_values.begin() + static_cast<std::ptrdiff_t>((kv_head + 1) * head_dim),
        expanded_v.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim));
  }
  auto start = std::chrono::steady_clock::now();
  mlx::core::array current_k(
      expanded_k.data(),
      mlx::core::Shape{static_cast<int>(q_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  mlx::core::array current_v(
      expanded_v.data(),
      mlx::core::Shape{static_cast<int>(q_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  if (cache.keys.has_value() && cache.values.has_value()) {
    cache.keys = mlx::core::concatenate({*cache.keys, current_k}, 1);
    cache.values = mlx::core::concatenate({*cache.values, current_v}, 1);
  } else {
    cache.keys = current_k;
    cache.values = current_v;
  }
  cache.keys->eval();
  cache.values->eval();
  mlx::core::synchronize();
  cache.current_len += 1;
  cache.append_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
}

std::vector<float> mlx_expanded_resident_kv_attention_values_probe(
    const std::vector<float>& q_values,
    const MlxExpandedResidentKvCacheProbe& cache,
    std::size_t q_heads,
    std::size_t head_dim,
    double* timing_ms = nullptr) {
  if (!cache.keys.has_value() || !cache.values.has_value() ||
      q_values.size() != q_heads * head_dim) {
    throw std::runtime_error("MLX expanded resident KV attention probe received invalid state");
  }
  const float inv_sqrt_dim =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  auto start = std::chrono::steady_clock::now();
  mlx::core::array q_array(
      q_values.data(),
      mlx::core::Shape{static_cast<int>(q_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  auto scores =
      mlx::core::matmul(q_array, mlx::core::transpose(*cache.keys, {0, 2, 1})) *
      inv_sqrt_dim;
  auto probs = mlx::core::softmax(scores, -1);
  auto mixed = mlx::core::matmul(probs, *cache.values);
  mixed.eval();
  mlx::core::synchronize();
  if (timing_ms != nullptr) {
    *timing_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
  const float* mixed_data = mixed.data<float>();
  std::vector<float> attention(q_values.size(), 0.0f);
  for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
    std::copy(
        mixed_data + static_cast<std::ptrdiff_t>(q_head * head_dim),
        mixed_data + static_cast<std::ptrdiff_t>((q_head + 1) * head_dim),
        attention.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim));
  }
  return attention;
}

void mlx_expanded_kv_cache_append_preallocated(
    NativeLayerKvCache& layer_cache,
    const std::vector<float>& current_k_values,
    const std::vector<float>& current_v_values,
    std::size_t q_heads,
    std::size_t kv_heads,
    std::size_t head_dim,
    std::size_t max_seq,
    double* append_ms = nullptr) {
  if (max_seq == 0) {
    throw std::runtime_error("expanded KV cache requires max_seq");
  }
  if (layer_cache.expanded_current_len >= max_seq) {
    throw std::runtime_error("expanded KV cache exceeded max_seq");
  }
  if (kv_heads == 0 || q_heads == 0 || q_heads % kv_heads != 0 ||
      current_k_values.size() != kv_heads * head_dim ||
      current_v_values.size() != kv_heads * head_dim) {
    throw std::runtime_error("expanded KV cache append received invalid k/v shapes");
  }
  const std::size_t q_heads_per_kv = q_heads / kv_heads;
  std::vector<float> expanded_k(q_heads * head_dim, 0.0f);
  std::vector<float> expanded_v(q_heads * head_dim, 0.0f);
  for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
    const std::size_t kv_head = q_head / q_heads_per_kv;
    std::copy(
        current_k_values.begin() + static_cast<std::ptrdiff_t>(kv_head * head_dim),
        current_k_values.begin() + static_cast<std::ptrdiff_t>((kv_head + 1) * head_dim),
        expanded_k.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim));
    std::copy(
        current_v_values.begin() + static_cast<std::ptrdiff_t>(kv_head * head_dim),
        current_v_values.begin() + static_cast<std::ptrdiff_t>((kv_head + 1) * head_dim),
        expanded_v.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim));
  }
  auto start = std::chrono::steady_clock::now();
  if (!layer_cache.expanded_keys.has_value() || !layer_cache.expanded_values.has_value()) {
    layer_cache.expanded_keys = mlx::core::zeros(
        mlx::core::Shape{
            static_cast<int>(q_heads),
            static_cast<int>(max_seq),
            static_cast<int>(head_dim)},
        mlx::core::float32);
    layer_cache.expanded_values = mlx::core::zeros(
        mlx::core::Shape{
            static_cast<int>(q_heads),
            static_cast<int>(max_seq),
            static_cast<int>(head_dim)},
        mlx::core::float32);
  }
  mlx::core::array current_k(
      expanded_k.data(),
      mlx::core::Shape{static_cast<int>(q_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  mlx::core::array current_v(
      expanded_v.data(),
      mlx::core::Shape{static_cast<int>(q_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  const int pos = static_cast<int>(layer_cache.expanded_current_len);
  layer_cache.expanded_keys = mlx::core::slice_update(
      *layer_cache.expanded_keys,
      current_k,
      mlx::core::Shape{0, pos, 0},
      mlx::core::Shape{static_cast<int>(q_heads), pos + 1, static_cast<int>(head_dim)});
  layer_cache.expanded_values = mlx::core::slice_update(
      *layer_cache.expanded_values,
      current_v,
      mlx::core::Shape{0, pos, 0},
      mlx::core::Shape{static_cast<int>(q_heads), pos + 1, static_cast<int>(head_dim)});
  layer_cache.expanded_keys->eval();
  layer_cache.expanded_values->eval();
  mlx::core::synchronize();
  layer_cache.expanded_current_len += 1;
  if (append_ms != nullptr) {
    *append_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
}

std::vector<float> mlx_expanded_kv_cache_attention_preallocated(
    const std::vector<float>& q_values,
    const NativeLayerKvCache& layer_cache,
    std::size_t q_heads,
    std::size_t head_dim,
    double* attention_ms = nullptr) {
  if (!layer_cache.expanded_keys.has_value() || !layer_cache.expanded_values.has_value() ||
      layer_cache.expanded_current_len == 0 ||
      q_values.size() != q_heads * head_dim) {
    throw std::runtime_error("expanded KV cache attention received invalid state");
  }
  const float inv_sqrt_dim =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  auto start = std::chrono::steady_clock::now();
  mlx::core::array q_array(
      q_values.data(),
      mlx::core::Shape{static_cast<int>(q_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  auto key_slice = mlx::core::slice(
      *layer_cache.expanded_keys,
      mlx::core::Shape{0, 0, 0},
      mlx::core::Shape{
          static_cast<int>(q_heads),
          static_cast<int>(layer_cache.expanded_current_len),
          static_cast<int>(head_dim)});
  auto value_slice = mlx::core::slice(
      *layer_cache.expanded_values,
      mlx::core::Shape{0, 0, 0},
      mlx::core::Shape{
          static_cast<int>(q_heads),
          static_cast<int>(layer_cache.expanded_current_len),
          static_cast<int>(head_dim)});
  auto scores =
      mlx::core::matmul(q_array, mlx::core::transpose(key_slice, {0, 2, 1})) *
      inv_sqrt_dim;
  auto probs = mlx::core::softmax(scores, -1);
  auto mixed = mlx::core::matmul(probs, value_slice);
  mixed.eval();
  mlx::core::synchronize();
  if (attention_ms != nullptr) {
    *attention_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
  const float* mixed_data = mixed.data<float>();
  std::vector<float> attention(q_values.size(), 0.0f);
  for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
    std::copy(
        mixed_data + static_cast<std::ptrdiff_t>(q_head * head_dim),
        mixed_data + static_cast<std::ptrdiff_t>((q_head + 1) * head_dim),
        attention.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim));
  }
  return attention;
}

void mlx_expanded_kv_cache_append_chunked(
    NativeLayerKvCache& layer_cache,
    const std::vector<float>& current_k_values,
    const std::vector<float>& current_v_values,
    std::size_t q_heads,
    std::size_t kv_heads,
    std::size_t head_dim,
    std::size_t chunk_size,
    double* append_ms = nullptr) {
  if (chunk_size == 0) {
    throw std::runtime_error("chunked expanded KV cache requires chunk_size");
  }
  if (kv_heads == 0 || q_heads == 0 || q_heads % kv_heads != 0 ||
      current_k_values.size() != kv_heads * head_dim ||
      current_v_values.size() != kv_heads * head_dim) {
    throw std::runtime_error("chunked expanded KV cache append received invalid k/v shapes");
  }
  const std::size_t q_heads_per_kv = q_heads / kv_heads;
  std::vector<float> expanded_k(q_heads * head_dim, 0.0f);
  std::vector<float> expanded_v(q_heads * head_dim, 0.0f);
  for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
    const std::size_t kv_head = q_head / q_heads_per_kv;
    std::copy(
        current_k_values.begin() + static_cast<std::ptrdiff_t>(kv_head * head_dim),
        current_k_values.begin() + static_cast<std::ptrdiff_t>((kv_head + 1) * head_dim),
        expanded_k.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim));
    std::copy(
        current_v_values.begin() + static_cast<std::ptrdiff_t>(kv_head * head_dim),
        current_v_values.begin() + static_cast<std::ptrdiff_t>((kv_head + 1) * head_dim),
        expanded_v.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim));
  }

  auto start = std::chrono::steady_clock::now();
  layer_cache.expanded_chunk_size = chunk_size;
  const std::size_t pos = layer_cache.expanded_chunk_current_len;
  const std::size_t chunk_index = pos / chunk_size;
  const std::size_t pos_in_chunk = pos % chunk_size;
  if (layer_cache.expanded_key_chunks.size() <= chunk_index) {
    layer_cache.expanded_key_chunks.push_back(mlx::core::zeros(
        mlx::core::Shape{
            static_cast<int>(q_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(head_dim)},
        mlx::core::float32));
    layer_cache.expanded_value_chunks.push_back(mlx::core::zeros(
        mlx::core::Shape{
            static_cast<int>(q_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(head_dim)},
        mlx::core::float32));
  }

  mlx::core::array current_k(
      expanded_k.data(),
      mlx::core::Shape{static_cast<int>(q_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  mlx::core::array current_v(
      expanded_v.data(),
      mlx::core::Shape{static_cast<int>(q_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  const int pos_i = static_cast<int>(pos_in_chunk);
  layer_cache.expanded_key_chunks[chunk_index] = mlx::core::slice_update(
      layer_cache.expanded_key_chunks[chunk_index],
      current_k,
      mlx::core::Shape{0, pos_i, 0},
      mlx::core::Shape{static_cast<int>(q_heads), pos_i + 1, static_cast<int>(head_dim)});
  layer_cache.expanded_value_chunks[chunk_index] = mlx::core::slice_update(
      layer_cache.expanded_value_chunks[chunk_index],
      current_v,
      mlx::core::Shape{0, pos_i, 0},
      mlx::core::Shape{static_cast<int>(q_heads), pos_i + 1, static_cast<int>(head_dim)});
  if (!defer_chunked_kv_append_sync()) {
    layer_cache.expanded_key_chunks[chunk_index].eval();
    layer_cache.expanded_value_chunks[chunk_index].eval();
    mlx::core::synchronize();
  }
  layer_cache.expanded_chunk_current_len += 1;
  if (append_ms != nullptr) {
    *append_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
}

std::vector<float> mlx_expanded_kv_cache_attention_chunked(
    const std::vector<float>& q_values,
    const NativeLayerKvCache& layer_cache,
    std::size_t q_heads,
    std::size_t head_dim,
    double* attention_ms = nullptr,
    MlxAttentionDetailTiming* detail_timing = nullptr) {
  if (layer_cache.expanded_key_chunks.empty() ||
      layer_cache.expanded_value_chunks.empty() ||
      layer_cache.expanded_chunk_current_len == 0 ||
      layer_cache.expanded_chunk_size == 0 ||
      q_values.size() != q_heads * head_dim) {
    throw std::runtime_error("chunked expanded KV cache attention received invalid state");
  }
  const float inv_sqrt_dim =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  auto start = std::chrono::steady_clock::now();
  auto segment_start = std::chrono::steady_clock::now();
  std::vector<mlx::core::array> key_slices;
  std::vector<mlx::core::array> value_slices;
  const std::size_t chunk_size = layer_cache.expanded_chunk_size;
  std::size_t remaining = layer_cache.expanded_chunk_current_len;
  for (std::size_t chunk = 0; chunk < layer_cache.expanded_key_chunks.size() && remaining > 0; ++chunk) {
    const std::size_t take = std::min(chunk_size, remaining);
    key_slices.push_back(mlx::core::slice(
        layer_cache.expanded_key_chunks[chunk],
        mlx::core::Shape{0, 0, 0},
        mlx::core::Shape{
            static_cast<int>(q_heads),
            static_cast<int>(take),
            static_cast<int>(head_dim)}));
    value_slices.push_back(mlx::core::slice(
        layer_cache.expanded_value_chunks[chunk],
        mlx::core::Shape{0, 0, 0},
        mlx::core::Shape{
            static_cast<int>(q_heads),
            static_cast<int>(take),
            static_cast<int>(head_dim)}));
    remaining -= take;
  }
  mlx::core::array key_view = key_slices.size() == 1
      ? key_slices[0]
      : mlx::core::concatenate(key_slices, 1);
  mlx::core::array value_view = value_slices.size() == 1
      ? value_slices[0]
      : mlx::core::concatenate(value_slices, 1);
  mlx::core::array q_array(
      q_values.data(),
      mlx::core::Shape{static_cast<int>(q_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->kv_view_assembly_ms, segment_start);
  }
  segment_start = std::chrono::steady_clock::now();
  auto scores =
      mlx::core::matmul(q_array, mlx::core::transpose(key_view, {0, 2, 1})) *
      inv_sqrt_dim;
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->score_matmul_ms, segment_start);
  }
  segment_start = std::chrono::steady_clock::now();
  auto probs = mlx::core::softmax(scores, -1);
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->softmax_ms, segment_start);
  }
  segment_start = std::chrono::steady_clock::now();
  auto mixed = mlx::core::matmul(probs, value_view);
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->value_mix_matmul_ms, segment_start);
  }
  segment_start = std::chrono::steady_clock::now();
  mixed.eval();
  mlx::core::synchronize();
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->eval_sync_ms, segment_start);
  }
  if (attention_ms != nullptr) {
    *attention_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
  segment_start = std::chrono::steady_clock::now();
  const float* mixed_data = mixed.data<float>();
  std::vector<float> attention(q_values.size(), 0.0f);
  for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
    std::copy(
        mixed_data + static_cast<std::ptrdiff_t>(q_head * head_dim),
        mixed_data + static_cast<std::ptrdiff_t>((q_head + 1) * head_dim),
        attention.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim));
  }
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->reshape_flatten_ms, segment_start);
  }
  return attention;
}

void mlx_compact_kv_cache_append_chunked(
    NativeLayerKvCache& layer_cache,
    const std::vector<float>& current_k_values,
    const std::vector<float>& current_v_values,
    std::size_t kv_heads,
    std::size_t head_dim,
    std::size_t chunk_size,
    double* append_ms = nullptr) {
  if (chunk_size == 0) {
    throw std::runtime_error("chunked compact MLX KV cache requires chunk_size");
  }
  if (kv_heads == 0 ||
      current_k_values.size() != kv_heads * head_dim ||
      current_v_values.size() != kv_heads * head_dim) {
    throw std::runtime_error("chunked compact MLX KV cache append received invalid k/v shapes");
  }

  auto start = std::chrono::steady_clock::now();
  layer_cache.compact_chunk_size = chunk_size;
  const std::size_t pos = layer_cache.compact_chunk_current_len;
  const std::size_t chunk_index = pos / chunk_size;
  const std::size_t pos_in_chunk = pos % chunk_size;
  if (layer_cache.compact_key_chunks.size() <= chunk_index) {
    layer_cache.compact_key_chunks.push_back(mlx::core::zeros(
        mlx::core::Shape{
            static_cast<int>(kv_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(head_dim)},
        mlx::core::float32));
    layer_cache.compact_value_chunks.push_back(mlx::core::zeros(
        mlx::core::Shape{
            static_cast<int>(kv_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(head_dim)},
        mlx::core::float32));
  }

  mlx::core::array current_k(
      current_k_values.data(),
      mlx::core::Shape{static_cast<int>(kv_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  mlx::core::array current_v(
      current_v_values.data(),
      mlx::core::Shape{static_cast<int>(kv_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  const int pos_i = static_cast<int>(pos_in_chunk);
  layer_cache.compact_key_chunks[chunk_index] = mlx::core::slice_update(
      layer_cache.compact_key_chunks[chunk_index],
      current_k,
      mlx::core::Shape{0, pos_i, 0},
      mlx::core::Shape{static_cast<int>(kv_heads), pos_i + 1, static_cast<int>(head_dim)});
  layer_cache.compact_value_chunks[chunk_index] = mlx::core::slice_update(
      layer_cache.compact_value_chunks[chunk_index],
      current_v,
      mlx::core::Shape{0, pos_i, 0},
      mlx::core::Shape{static_cast<int>(kv_heads), pos_i + 1, static_cast<int>(head_dim)});
  if (!defer_chunked_kv_append_sync()) {
    layer_cache.compact_key_chunks[chunk_index].eval();
    layer_cache.compact_value_chunks[chunk_index].eval();
    mlx::core::synchronize();
  }
  layer_cache.compact_chunk_current_len += 1;
  if (append_ms != nullptr) {
    *append_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
}

void mlx_compact_kv_cache_append_chunked_arrays(
    NativeLayerKvCache& layer_cache,
    const mlx::core::array& current_k_array,
    const mlx::core::array& current_v_array,
    std::size_t kv_heads,
    std::size_t head_dim,
    std::size_t chunk_size,
    double* append_ms = nullptr) {
  if (chunk_size == 0) {
    throw std::runtime_error("chunked compact MLX KV cache requires chunk_size");
  }
  if (kv_heads == 0 ||
      static_cast<std::size_t>(current_k_array.size()) != kv_heads * head_dim ||
      static_cast<std::size_t>(current_v_array.size()) != kv_heads * head_dim) {
    throw std::runtime_error("chunked compact MLX KV cache array append received invalid k/v shapes");
  }

  auto start = std::chrono::steady_clock::now();
  layer_cache.compact_chunk_size = chunk_size;
  const std::size_t pos = layer_cache.compact_chunk_current_len;
  const std::size_t chunk_index = pos / chunk_size;
  const std::size_t pos_in_chunk = pos % chunk_size;
  if (layer_cache.compact_key_chunks.size() <= chunk_index) {
    layer_cache.compact_key_chunks.push_back(mlx::core::zeros(
        mlx::core::Shape{
            static_cast<int>(kv_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(head_dim)},
        mlx::core::float32));
    layer_cache.compact_value_chunks.push_back(mlx::core::zeros(
        mlx::core::Shape{
            static_cast<int>(kv_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(head_dim)},
        mlx::core::float32));
  }

  auto current_k = mlx::core::reshape(
      current_k_array,
      mlx::core::Shape{static_cast<int>(kv_heads), 1, static_cast<int>(head_dim)});
  auto current_v = mlx::core::reshape(
      current_v_array,
      mlx::core::Shape{static_cast<int>(kv_heads), 1, static_cast<int>(head_dim)});
  const int pos_i = static_cast<int>(pos_in_chunk);
  layer_cache.compact_key_chunks[chunk_index] = mlx::core::slice_update(
      layer_cache.compact_key_chunks[chunk_index],
      current_k,
      mlx::core::Shape{0, pos_i, 0},
      mlx::core::Shape{static_cast<int>(kv_heads), pos_i + 1, static_cast<int>(head_dim)});
  layer_cache.compact_value_chunks[chunk_index] = mlx::core::slice_update(
      layer_cache.compact_value_chunks[chunk_index],
      current_v,
      mlx::core::Shape{0, pos_i, 0},
      mlx::core::Shape{static_cast<int>(kv_heads), pos_i + 1, static_cast<int>(head_dim)});
  if (!defer_chunked_kv_append_sync()) {
    layer_cache.compact_key_chunks[chunk_index].eval();
    layer_cache.compact_value_chunks[chunk_index].eval();
    mlx::core::synchronize();
  }
  layer_cache.compact_chunk_current_len += 1;
  if (append_ms != nullptr) {
    *append_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
}

std::vector<float> mlx_compact_kv_cache_attention_chunked(
    const std::vector<float>& q_values,
    const NativeLayerKvCache& layer_cache,
    std::size_t q_heads,
    std::size_t kv_heads,
    std::size_t head_dim,
    double* attention_ms = nullptr,
    MlxAttentionDetailTiming* detail_timing = nullptr) {
  if (layer_cache.compact_key_chunks.empty() ||
      layer_cache.compact_value_chunks.empty() ||
      layer_cache.compact_chunk_current_len == 0 ||
      layer_cache.compact_chunk_size == 0 ||
      kv_heads == 0 ||
      q_heads == 0 ||
      q_heads % kv_heads != 0 ||
      q_values.size() != q_heads * head_dim) {
    throw std::runtime_error("chunked compact MLX KV cache attention received invalid state");
  }
  const float inv_sqrt_dim =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  auto start = std::chrono::steady_clock::now();
  auto segment_start = std::chrono::steady_clock::now();
  std::vector<mlx::core::array> key_slices;
  std::vector<mlx::core::array> value_slices;
  const std::size_t chunk_size = layer_cache.compact_chunk_size;
  std::size_t remaining = layer_cache.compact_chunk_current_len;
  for (std::size_t chunk = 0; chunk < layer_cache.compact_key_chunks.size() && remaining > 0; ++chunk) {
    const std::size_t take = std::min(chunk_size, remaining);
    key_slices.push_back(mlx::core::slice(
        layer_cache.compact_key_chunks[chunk],
        mlx::core::Shape{0, 0, 0},
        mlx::core::Shape{
            static_cast<int>(kv_heads),
            static_cast<int>(take),
            static_cast<int>(head_dim)}));
    value_slices.push_back(mlx::core::slice(
        layer_cache.compact_value_chunks[chunk],
        mlx::core::Shape{0, 0, 0},
        mlx::core::Shape{
            static_cast<int>(kv_heads),
            static_cast<int>(take),
            static_cast<int>(head_dim)}));
    remaining -= take;
  }
  mlx::core::array key_compact_view = key_slices.size() == 1
      ? key_slices[0]
      : mlx::core::concatenate(key_slices, 1);
  mlx::core::array value_compact_view = value_slices.size() == 1
      ? value_slices[0]
      : mlx::core::concatenate(value_slices, 1);
  const int q_heads_per_kv = static_cast<int>(q_heads / kv_heads);
  mlx::core::array key_view = mlx::core::repeat(key_compact_view, q_heads_per_kv, 0);
  mlx::core::array value_view = mlx::core::repeat(value_compact_view, q_heads_per_kv, 0);
  mlx::core::array q_array(
      q_values.data(),
      mlx::core::Shape{static_cast<int>(q_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->kv_view_assembly_ms, segment_start);
  }
  segment_start = std::chrono::steady_clock::now();
  auto scores =
      mlx::core::matmul(q_array, mlx::core::transpose(key_view, {0, 2, 1})) *
      inv_sqrt_dim;
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->score_matmul_ms, segment_start);
  }
  segment_start = std::chrono::steady_clock::now();
  auto probs = mlx::core::softmax(scores, -1);
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->softmax_ms, segment_start);
  }
  segment_start = std::chrono::steady_clock::now();
  auto mixed = mlx::core::matmul(probs, value_view);
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->value_mix_matmul_ms, segment_start);
  }
  segment_start = std::chrono::steady_clock::now();
  mixed.eval();
  mlx::core::synchronize();
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->eval_sync_ms, segment_start);
  }
  if (attention_ms != nullptr) {
    *attention_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
  segment_start = std::chrono::steady_clock::now();
  const float* mixed_data = mixed.data<float>();
  std::vector<float> attention(q_values.size(), 0.0f);
  for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
    std::copy(
        mixed_data + static_cast<std::ptrdiff_t>(q_head * head_dim),
        mixed_data + static_cast<std::ptrdiff_t>((q_head + 1) * head_dim),
        attention.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim));
  }
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->reshape_flatten_ms, segment_start);
  }
  return attention;
}

std::vector<float> mlx_compact_kv_cache_attention_chunk_aware(
    const std::vector<float>& q_values,
    const NativeLayerKvCache& layer_cache,
    std::size_t q_heads,
    std::size_t kv_heads,
    std::size_t head_dim,
    double* attention_ms = nullptr,
    MlxAttentionDetailTiming* detail_timing = nullptr) {
  if (layer_cache.compact_key_chunks.empty() ||
      layer_cache.compact_value_chunks.empty() ||
      layer_cache.compact_chunk_current_len == 0 ||
      layer_cache.compact_chunk_size == 0 ||
      kv_heads == 0 ||
      q_heads == 0 ||
      q_heads % kv_heads != 0 ||
      q_values.size() != q_heads * head_dim) {
    throw std::runtime_error("chunk-aware compact MLX KV attention received invalid state");
  }
  const float inv_sqrt_dim =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  const int q_heads_i = static_cast<int>(q_heads);
  const int head_dim_i = static_cast<int>(head_dim);
  const int kv_heads_i = static_cast<int>(kv_heads);
  const int q_heads_per_kv = static_cast<int>(q_heads / kv_heads);
  auto start = std::chrono::steady_clock::now();
  auto segment_start = std::chrono::steady_clock::now();
  mlx::core::array q_array(
      q_values.data(),
      mlx::core::Shape{q_heads_i, 1, head_dim_i},
      mlx::core::float32);

  std::optional<mlx::core::array> global_max;
  std::optional<mlx::core::array> denom;
  std::optional<mlx::core::array> numerator;
  const std::size_t chunk_size = layer_cache.compact_chunk_size;
  std::size_t remaining = layer_cache.compact_chunk_current_len;
  for (std::size_t chunk = 0; chunk < layer_cache.compact_key_chunks.size() && remaining > 0; ++chunk) {
    const std::size_t take = std::min(chunk_size, remaining);
    mlx::core::array key_compact = mlx::core::slice(
        layer_cache.compact_key_chunks[chunk],
        mlx::core::Shape{0, 0, 0},
        mlx::core::Shape{kv_heads_i, static_cast<int>(take), head_dim_i});
    mlx::core::array value_compact = mlx::core::slice(
        layer_cache.compact_value_chunks[chunk],
        mlx::core::Shape{0, 0, 0},
        mlx::core::Shape{kv_heads_i, static_cast<int>(take), head_dim_i});
    mlx::core::array key_view = mlx::core::repeat(key_compact, q_heads_per_kv, 0);
    mlx::core::array value_view = mlx::core::repeat(value_compact, q_heads_per_kv, 0);
    if (detail_timing != nullptr) {
      add_elapsed_ms(detail_timing->kv_view_assembly_ms, segment_start);
    }

    segment_start = std::chrono::steady_clock::now();
    auto scores =
        mlx::core::matmul(q_array, mlx::core::transpose(key_view, {0, 2, 1})) *
        inv_sqrt_dim;
    auto local_max = mlx::core::max(scores, -1, true);
    if (detail_timing != nullptr) {
      add_elapsed_ms(detail_timing->score_matmul_ms, segment_start);
    }

    segment_start = std::chrono::steady_clock::now();
    auto local_exp = mlx::core::exp(scores - local_max);
    auto local_denom = mlx::core::sum(local_exp, -1, true);
    if (detail_timing != nullptr) {
      add_elapsed_ms(detail_timing->softmax_ms, segment_start);
    }

    segment_start = std::chrono::steady_clock::now();
    auto local_numerator = mlx::core::matmul(local_exp, value_view);
    if (detail_timing != nullptr) {
      add_elapsed_ms(detail_timing->value_mix_matmul_ms, segment_start);
    }

    if (!global_max.has_value()) {
      global_max = local_max;
      denom = local_denom;
      numerator = local_numerator;
    } else {
      auto next_max = mlx::core::maximum(*global_max, local_max);
      auto old_scale = mlx::core::exp(*global_max - next_max);
      auto new_scale = mlx::core::exp(local_max - next_max);
      denom = (*denom * old_scale) + (local_denom * new_scale);
      numerator = (*numerator * old_scale) + (local_numerator * new_scale);
      global_max = next_max;
    }
    remaining -= take;
    segment_start = std::chrono::steady_clock::now();
  }
  if (!denom.has_value() || !numerator.has_value()) {
    throw std::runtime_error("chunk-aware compact MLX KV attention produced no chunks");
  }
  auto mixed = *numerator / *denom;
  segment_start = std::chrono::steady_clock::now();
  mixed.eval();
  mlx::core::synchronize();
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->eval_sync_ms, segment_start);
  }
  if (attention_ms != nullptr) {
    *attention_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
  segment_start = std::chrono::steady_clock::now();
  const float* mixed_data = mixed.data<float>();
  std::vector<float> attention(q_values.size(), 0.0f);
  for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
    std::copy(
        mixed_data + static_cast<std::ptrdiff_t>(q_head * head_dim),
        mixed_data + static_cast<std::ptrdiff_t>((q_head + 1) * head_dim),
        attention.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim));
  }
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->reshape_flatten_ms, segment_start);
  }
  return attention;
}

mlx::core::array mlx_compact_kv_cache_attention_chunk_aware_array(
    const mlx::core::array& q_array_flat,
    const NativeLayerKvCache& layer_cache,
    std::size_t q_heads,
    std::size_t kv_heads,
    std::size_t head_dim,
    double* attention_ms = nullptr,
    MlxAttentionDetailTiming* detail_timing = nullptr) {
  if (layer_cache.compact_key_chunks.empty() ||
      layer_cache.compact_value_chunks.empty() ||
      layer_cache.compact_chunk_current_len == 0 ||
      layer_cache.compact_chunk_size == 0 ||
      kv_heads == 0 ||
      q_heads == 0 ||
      q_heads % kv_heads != 0 ||
      static_cast<std::size_t>(q_array_flat.size()) != q_heads * head_dim) {
    throw std::runtime_error("chunk-aware compact MLX KV attention array received invalid state");
  }
  const float inv_sqrt_dim =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  const int q_heads_i = static_cast<int>(q_heads);
  const int head_dim_i = static_cast<int>(head_dim);
  const int kv_heads_i = static_cast<int>(kv_heads);
  const int q_heads_per_kv = static_cast<int>(q_heads / kv_heads);
  auto start = std::chrono::steady_clock::now();
  auto segment_start = std::chrono::steady_clock::now();
  auto q_array = mlx::core::reshape(
      q_array_flat,
      mlx::core::Shape{q_heads_i, 1, head_dim_i});

  std::optional<mlx::core::array> global_max;
  std::optional<mlx::core::array> denom;
  std::optional<mlx::core::array> numerator;
  const std::size_t chunk_size = layer_cache.compact_chunk_size;
  std::size_t remaining = layer_cache.compact_chunk_current_len;
  for (std::size_t chunk = 0; chunk < layer_cache.compact_key_chunks.size() && remaining > 0; ++chunk) {
    const std::size_t take = std::min(chunk_size, remaining);
    mlx::core::array key_compact = mlx::core::slice(
        layer_cache.compact_key_chunks[chunk],
        mlx::core::Shape{0, 0, 0},
        mlx::core::Shape{kv_heads_i, static_cast<int>(take), head_dim_i});
    mlx::core::array value_compact = mlx::core::slice(
        layer_cache.compact_value_chunks[chunk],
        mlx::core::Shape{0, 0, 0},
        mlx::core::Shape{kv_heads_i, static_cast<int>(take), head_dim_i});
    mlx::core::array key_view = mlx::core::repeat(key_compact, q_heads_per_kv, 0);
    mlx::core::array value_view = mlx::core::repeat(value_compact, q_heads_per_kv, 0);
    if (detail_timing != nullptr) {
      add_elapsed_ms(detail_timing->kv_view_assembly_ms, segment_start);
    }

    segment_start = std::chrono::steady_clock::now();
    auto scores =
        mlx::core::matmul(q_array, mlx::core::transpose(key_view, {0, 2, 1})) *
        inv_sqrt_dim;
    auto local_max = mlx::core::max(scores, -1, true);
    if (detail_timing != nullptr) {
      add_elapsed_ms(detail_timing->score_matmul_ms, segment_start);
    }

    segment_start = std::chrono::steady_clock::now();
    auto local_exp = mlx::core::exp(scores - local_max);
    auto local_denom = mlx::core::sum(local_exp, -1, true);
    if (detail_timing != nullptr) {
      add_elapsed_ms(detail_timing->softmax_ms, segment_start);
    }

    segment_start = std::chrono::steady_clock::now();
    auto local_numerator = mlx::core::matmul(local_exp, value_view);
    if (detail_timing != nullptr) {
      add_elapsed_ms(detail_timing->value_mix_matmul_ms, segment_start);
    }

    if (!global_max.has_value()) {
      global_max = local_max;
      denom = local_denom;
      numerator = local_numerator;
    } else {
      auto next_max = mlx::core::maximum(*global_max, local_max);
      auto old_scale = mlx::core::exp(*global_max - next_max);
      auto new_scale = mlx::core::exp(local_max - next_max);
      denom = (*denom * old_scale) + (local_denom * new_scale);
      numerator = (*numerator * old_scale) + (local_numerator * new_scale);
      global_max = next_max;
    }
    remaining -= take;
    segment_start = std::chrono::steady_clock::now();
  }
  if (!denom.has_value() || !numerator.has_value()) {
    throw std::runtime_error("chunk-aware compact MLX KV attention array produced no chunks");
  }
  auto mixed = *numerator / *denom;
  segment_start = std::chrono::steady_clock::now();
  auto flattened = mlx::core::reshape(
      mixed,
      mlx::core::Shape{1, static_cast<int>(q_heads * head_dim)});
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->reshape_flatten_ms, segment_start);
  }
  if (attention_ms != nullptr) {
    *attention_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
  return flattened;
}

mlx::core::array mlx_expanded_kv_cache_attention_chunked_array(
    const std::vector<float>& q_values,
    const NativeLayerKvCache& layer_cache,
    std::size_t q_heads,
    std::size_t head_dim,
    MlxAttentionDetailTiming* detail_timing = nullptr) {
  if (layer_cache.expanded_key_chunks.empty() ||
      layer_cache.expanded_value_chunks.empty() ||
      layer_cache.expanded_chunk_current_len == 0 ||
      layer_cache.expanded_chunk_size == 0 ||
      q_values.size() != q_heads * head_dim) {
    throw std::runtime_error("chunked expanded KV cache attention array received invalid state");
  }
  const float inv_sqrt_dim =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  auto segment_start = std::chrono::steady_clock::now();
  std::vector<mlx::core::array> key_slices;
  std::vector<mlx::core::array> value_slices;
  const std::size_t chunk_size = layer_cache.expanded_chunk_size;
  std::size_t remaining = layer_cache.expanded_chunk_current_len;
  for (std::size_t chunk = 0; chunk < layer_cache.expanded_key_chunks.size() && remaining > 0; ++chunk) {
    const std::size_t take = std::min(chunk_size, remaining);
    key_slices.push_back(mlx::core::slice(
        layer_cache.expanded_key_chunks[chunk],
        mlx::core::Shape{0, 0, 0},
        mlx::core::Shape{
            static_cast<int>(q_heads),
            static_cast<int>(take),
            static_cast<int>(head_dim)}));
    value_slices.push_back(mlx::core::slice(
        layer_cache.expanded_value_chunks[chunk],
        mlx::core::Shape{0, 0, 0},
        mlx::core::Shape{
            static_cast<int>(q_heads),
            static_cast<int>(take),
            static_cast<int>(head_dim)}));
    remaining -= take;
  }
  mlx::core::array key_view = key_slices.size() == 1
      ? key_slices[0]
      : mlx::core::concatenate(key_slices, 1);
  mlx::core::array value_view = value_slices.size() == 1
      ? value_slices[0]
      : mlx::core::concatenate(value_slices, 1);
  mlx::core::array q_array(
      q_values.data(),
      mlx::core::Shape{static_cast<int>(q_heads), 1, static_cast<int>(head_dim)},
      mlx::core::float32);
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->kv_view_assembly_ms, segment_start);
  }
  segment_start = std::chrono::steady_clock::now();
  auto scores =
      mlx::core::matmul(q_array, mlx::core::transpose(key_view, {0, 2, 1})) *
      inv_sqrt_dim;
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->score_matmul_ms, segment_start);
  }
  segment_start = std::chrono::steady_clock::now();
  auto probs = mlx::core::softmax(scores, -1);
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->softmax_ms, segment_start);
  }
  segment_start = std::chrono::steady_clock::now();
  auto mixed = mlx::core::matmul(probs, value_view);
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->value_mix_matmul_ms, segment_start);
  }
  segment_start = std::chrono::steady_clock::now();
  auto flattened = mlx::core::reshape(
      mixed,
      mlx::core::Shape{1, static_cast<int>(q_heads * head_dim)});
  if (detail_timing != nullptr) {
    add_elapsed_ms(detail_timing->reshape_flatten_ms, segment_start);
  }
  return flattened;
}

ResidentDecodeValue layer_decode_value_resident_incremental_from_input(
    const ResidentLayerGroups& groups,
    std::size_t layer,
    const ResidentDecodeValue& input_value,
    double eps,
    NativeSessionKvCache& kv_cache,
    ResidentDecodeResult* timing);

std::string layer0_mlx_resident_full_block_probe_json(
    const ResidentLayerGroups& layer0,
    const std::vector<float>& input_values,
    double eps) {
  try {
    const auto total_start = std::chrono::steady_clock::now();
    NativeSessionKvCache cpu_boundary_cache;
    cpu_boundary_cache.owner_session = "layer0_mlx_resident_full_block_probe_cpu_boundary";
    cpu_boundary_cache.layers.resize(1);
    cpu_boundary_cache.layers_allocated = 1;
    cpu_boundary_cache.max_seq = 1;
    ResidentDecodeValue input_value(mlx_array_from_vector(input_values));
    ResidentDecodeResult cpu_boundary_timing;
    const auto cpu_boundary_start = std::chrono::steady_clock::now();
    ResidentDecodeValue cpu_boundary_output = layer_decode_value_resident_incremental_from_input(
        layer0,
        0,
        input_value,
        eps,
        cpu_boundary_cache,
        &cpu_boundary_timing);
    const std::vector<float> cpu_boundary_values =
        cpu_boundary_output.cpu("layer0_probe_cpu_boundary_output");
    const double cpu_boundary_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - cpu_boundary_start).count();

    NativeSessionKvCache resident_cache;
    resident_cache.owner_session = "layer0_mlx_resident_full_block_probe";
    resident_cache.layers.resize(1);
    resident_cache.layers_allocated = 1;
    resident_cache.max_seq = 1;
    const auto resident_start = std::chrono::steady_clock::now();
    ResidentDecodeValue resident_input(mlx_array_from_vector(input_values));
    auto input_norm = rmsnorm_mlx_array(layer0.input_norm, resident_input.mlx, eps);
    std::vector<float> q_values;
    std::vector<float> k_values;
    std::vector<float> v_values;
    const auto qkv_start = std::chrono::steady_clock::now();
    {
      MlxQuantizedLinearStepScope qkv_scope("layer0_probe_qkv");
      auto qkv = quantized_linear_triple_values_from_mlx_input(
          layer0.q_proj,
          layer0.k_proj,
          layer0.v_proj,
          input_norm);
      q_values = std::move(std::get<0>(qkv));
      k_values = std::move(std::get<1>(qkv));
      v_values = std::move(std::get<2>(qkv));
    }
    const double qkv_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - qkv_start).count();
    const auto qk_norm_rope_start = std::chrono::steady_clock::now();
    apply_qk_norm_in_place(layer0.q_norm, layer0.k_norm, q_values, k_values, eps);
    apply_active_rope_to_qk(q_values, k_values, 0);
    const double qk_norm_rope_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - qk_norm_rope_start).count();

    const auto append_start = std::chrono::steady_clock::now();
    double append_ms = 0.0;
    mlx_expanded_kv_cache_append_chunked(
        resident_cache.layers[0],
        k_values,
        v_values,
        active_rope_config().num_attention_heads,
        active_rope_config().num_key_value_heads,
        active_rope_config().head_dim,
        expanded_kv_chunk_size(),
        &append_ms);
    append_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - append_start).count() - append_ms;

    MlxAttentionDetailTiming attention_detail;
    const auto attention_start = std::chrono::steady_clock::now();
    auto attention_array = mlx_expanded_kv_cache_attention_chunked_array(
        q_values,
        resident_cache.layers[0],
        active_rope_config().num_attention_heads,
        active_rope_config().head_dim,
        &attention_detail);
    const double attention_expression_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - attention_start).count();

    const auto o_start = std::chrono::steady_clock::now();
    LocalQuantizedMlxArrays o_local;
    auto o_result = make_quantized_matmul_from_mlx_input(
        layer0.o_proj,
        attention_array,
        o_local,
        attention_array.size());
    auto attention_residual = o_result + resident_input.mlx;
    const double o_expression_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - o_start).count();

    const auto post_norm_start = std::chrono::steady_clock::now();
    auto post_norm = rmsnorm_mlx_array(layer0.post_attention_norm, attention_residual, eps);
    const double post_norm_expression_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - post_norm_start).count();

    MlxResidentMlpChainTiming mlp_timing;
    const auto mlp_start = std::chrono::steady_clock::now();
    auto resident_output = quantized_linear_mlp_chain_residual_mlx(
        layer0.gate_proj,
        layer0.up_proj,
        layer0.down_proj,
        post_norm,
        attention_residual,
        &mlp_timing);
    const double mlp_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - mlp_start).count();

    const auto readback_start = std::chrono::steady_clock::now();
    ResidentDecodeValue resident_output_value(resident_output);
    const std::vector<float> resident_values =
        resident_output_value.cpu("layer0_probe_final_compare");
    const double final_readback_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - readback_start).count();
    const double resident_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - resident_start).count();
    const double total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - total_start).count();

    const double max_abs_diff = vector_max_abs_diff(cpu_boundary_values, resident_values);
    const std::uint64_t cpu_boundary_sync_count = 6;
    const std::uint64_t resident_sync_count = 4;
    std::ostringstream out;
    out << "{"
        << "\"available\":true,"
        << "\"promoted_to_generation\":false,"
        << "\"append_sync_deferred_default\":"
        << (defer_chunked_kv_append_sync() ? "true" : "false") << ","
        << "\"q_norm_mlx_applied\":false,"
        << "\"k_norm_mlx_applied\":false,"
        << "\"rope_mlx_applied\":false,"
        << "\"minimal_cpu_boundary\":\"q_norm/k_norm/RoPE only\","
        << "\"output_len\":" << resident_values.size() << ","
        << "\"max_abs_diff\":" << max_abs_diff << ","
        << "\"checksum_cpu_boundary\":" << vector_checksum(cpu_boundary_values) << ","
        << "\"checksum_mlx_resident\":" << vector_checksum(resident_values) << ","
        << "\"first_values_cpu_boundary\":"
        << json_array_float_to_string(first_values_of(cpu_boundary_values)) << ","
        << "\"first_values_mlx_resident\":"
        << json_array_float_to_string(first_values_of(resident_values)) << ","
        << "\"sync_count_cpu_boundary_estimate\":" << cpu_boundary_sync_count << ","
        << "\"sync_count_mlx_resident_estimate\":" << resident_sync_count << ","
        << "\"timing_ms\":{"
        << "\"cpu_boundary_total\":" << cpu_boundary_ms << ","
        << "\"mlx_resident_total\":" << resident_ms << ","
        << "\"qkv_readback_for_q_norm_rope\":" << qkv_ms << ","
        << "\"q_norm_k_norm_rope_cpu\":" << qk_norm_rope_ms << ","
        << "\"kv_append_update\":" << append_ms << ","
        << "\"attention_expression_no_readback\":" << attention_expression_ms << ","
        << "\"attention_kv_chunk_view_assembly\":" << attention_detail.kv_view_assembly_ms << ","
        << "\"attention_score_matmul\":" << attention_detail.score_matmul_ms << ","
        << "\"attention_softmax\":" << attention_detail.softmax_ms << ","
        << "\"attention_value_mix_matmul\":" << attention_detail.value_mix_matmul_ms << ","
        << "\"attention_reshape_flatten\":" << attention_detail.reshape_flatten_ms << ","
        << "\"o_proj_residual_expression\":" << o_expression_ms << ","
        << "\"post_attention_rmsnorm_expression\":" << post_norm_expression_ms << ","
        << "\"mlp_chain_residual\":" << mlp_ms << ","
        << "\"mlp_chain_setup\":" << mlp_timing.setup_ms << ","
        << "\"mlp_gate_up_eval\":" << mlp_timing.gate_up_eval_ms << ","
        << "\"mlp_activation_eval\":" << mlp_timing.activation_eval_ms << ","
        << "\"mlp_down_eval_sync\":" << mlp_timing.down_eval_ms << ","
        << "\"final_compare_readback\":" << final_readback_ms << ","
        << "\"probe_total\":" << total_ms
        << "}"
        << "}";
    return out.str();
  } catch (const std::exception& e) {
    return std::string("{\"available\":false,\"error\":\"") + json_escape(e.what()) + "\"}";
  } catch (...) {
    return "{\"available\":false,\"error\":\"unknown_exception\"}";
  }
}

std::string mlx_attention_worst_head_diagnostic_json(
    const std::vector<float>& q_values,
    const NativeLayerKvCache& layer_cache,
    const std::vector<float>& current_k_values,
    const std::vector<float>& current_v_values,
    std::size_t q_head,
    std::size_t kv_head) {
  const std::size_t head_dim = 128;
  const std::size_t prior_positions = layer_cache.keys.size();
  const std::size_t total_positions = prior_positions + 1;
  if ((q_head + 1) * head_dim > q_values.size() ||
      (kv_head + 1) * head_dim > current_k_values.size() ||
      current_k_values.size() != current_v_values.size()) {
    return "{\"available\":false,\"error\":\"bad_head_or_shape\"}";
  }
  auto first_n = [](const std::vector<float>& values, std::size_t n) {
    std::vector<float> out;
    out.reserve(std::min(n, values.size()));
    for (std::size_t i = 0; i < values.size() && i < n; ++i) {
      out.push_back(values[i]);
    }
    return out;
  };
  auto last_n = [](const std::vector<float>& values, std::size_t n) {
    std::vector<float> out;
    const std::size_t start = values.size() > n ? values.size() - n : 0;
    out.reserve(values.size() - start);
    for (std::size_t i = start; i < values.size(); ++i) {
      out.push_back(values[i]);
    }
    return out;
  };
  auto top_positions_json = [](const std::vector<float>& values, std::size_t n) {
    std::vector<std::pair<std::size_t, float>> indexed;
    indexed.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
      indexed.emplace_back(i, values[i]);
    }
    std::sort(indexed.begin(), indexed.end(), [](const auto& a, const auto& b) {
      return a.second > b.second;
    });
    std::ostringstream out;
    out << "[";
    const std::size_t count = std::min(n, indexed.size());
    for (std::size_t i = 0; i < count; ++i) {
      if (i > 0) {
        out << ",";
      }
      out << "{\"position\":" << indexed[i].first << ",\"value\":" << indexed[i].second << "}";
    }
    out << "]";
    return out.str();
  };
  auto vector_window_json = [&](const std::vector<float>& values) {
    std::ostringstream out;
    out << "{"
        << "\"len\":" << values.size() << ","
        << "\"checksum\":" << vector_checksum(values) << ","
        << "\"first16\":" << json_array_float_to_string(first_n(values, 16)) << ","
        << "\"last16\":" << json_array_float_to_string(last_n(values, 16))
        << "}";
    return out.str();
  };
  auto cache_positions_json = [&](bool keys) {
    std::ostringstream out;
    out << "[";
    const std::size_t count = std::min<std::size_t>(3, total_positions);
    for (std::size_t position = 0; position < count; ++position) {
      if (position > 0) {
        out << ",";
      }
      const std::vector<float>& source =
          position < prior_positions
              ? (keys ? layer_cache.keys[position] : layer_cache.values[position])
              : (keys ? current_k_values : current_v_values);
      std::vector<float> head_values(head_dim, 0.0f);
      std::copy(
          source.begin() + static_cast<std::ptrdiff_t>(kv_head * head_dim),
          source.begin() + static_cast<std::ptrdiff_t>((kv_head + 1) * head_dim),
          head_values.begin());
      out << "{"
          << "\"position\":" << position << ","
          << "\"checksum\":" << vector_checksum(head_values) << ","
          << "\"first16\":" << json_array_float_to_string(first_n(head_values, 16))
          << "}";
    }
    out << "]";
    return out.str();
  };

  std::vector<float> q_head_values(head_dim, 0.0f);
  std::vector<float> k_values(total_positions * head_dim, 0.0f);
  std::vector<float> v_values(total_positions * head_dim, 0.0f);
  std::copy(
      q_values.begin() + static_cast<std::ptrdiff_t>(q_head * head_dim),
      q_values.begin() + static_cast<std::ptrdiff_t>((q_head + 1) * head_dim),
      q_head_values.begin());
  for (std::size_t position = 0; position < total_positions; ++position) {
    const std::vector<float>& key =
        position < prior_positions ? layer_cache.keys[position] : current_k_values;
    const std::vector<float>& value =
        position < prior_positions ? layer_cache.values[position] : current_v_values;
    std::copy(
        key.begin() + static_cast<std::ptrdiff_t>(kv_head * head_dim),
        key.begin() + static_cast<std::ptrdiff_t>((kv_head + 1) * head_dim),
        k_values.begin() + static_cast<std::ptrdiff_t>(position * head_dim));
    std::copy(
        value.begin() + static_cast<std::ptrdiff_t>(kv_head * head_dim),
        value.begin() + static_cast<std::ptrdiff_t>((kv_head + 1) * head_dim),
        v_values.begin() + static_cast<std::ptrdiff_t>(position * head_dim));
  }

  const double inv_sqrt_dim = 1.0 / std::sqrt(static_cast<double>(head_dim));
  std::vector<float> cpu_scores(total_positions, 0.0f);
  double max_score = -std::numeric_limits<double>::infinity();
  for (std::size_t position = 0; position < total_positions; ++position) {
    double score = 0.0;
    for (std::size_t i = 0; i < head_dim; ++i) {
      score += static_cast<double>(q_head_values[i]) *
               static_cast<double>(k_values[position * head_dim + i]);
    }
    score *= inv_sqrt_dim;
    cpu_scores[position] = static_cast<float>(score);
    max_score = std::max(max_score, score);
  }
  std::vector<float> cpu_probs(total_positions, 0.0f);
  double cpu_prob_sum = 0.0;
  for (std::size_t i = 0; i < total_positions; ++i) {
    const double value = std::exp(static_cast<double>(cpu_scores[i]) - max_score);
    cpu_probs[i] = static_cast<float>(value);
    cpu_prob_sum += value;
  }
  if (cpu_prob_sum > 0.0) {
    for (float& value : cpu_probs) {
      value = static_cast<float>(static_cast<double>(value) / cpu_prob_sum);
    }
  }
  std::vector<float> cpu_output(head_dim, 0.0f);
  for (std::size_t i = 0; i < head_dim; ++i) {
    double sum = 0.0;
    for (std::size_t position = 0; position < total_positions; ++position) {
      sum += static_cast<double>(cpu_probs[position]) *
             static_cast<double>(v_values[position * head_dim + i]);
    }
    cpu_output[i] = static_cast<float>(sum);
  }

  auto mlx_head = mlx_single_head_attention_values_probe(
      q_values,
      layer_cache,
      current_k_values,
      current_v_values,
      q_head,
      kv_head,
      head_dim);
  const std::vector<float>& mlx_scores = mlx_head.scores;
  const std::vector<float>& mlx_probs = mlx_head.probabilities;
  const std::vector<float>& mlx_output = mlx_head.output;
  double mlx_prob_sum = 0.0;
  for (float value : mlx_probs) {
    mlx_prob_sum += static_cast<double>(value);
  }
  const auto cpu_argmax = static_cast<std::size_t>(
      std::distance(cpu_scores.begin(), std::max_element(cpu_scores.begin(), cpu_scores.end())));
  const auto mlx_argmax = static_cast<std::size_t>(
      std::distance(mlx_scores.begin(), std::max_element(mlx_scores.begin(), mlx_scores.end())));

  std::ostringstream out;
  out << "{"
      << "\"available\":true,"
      << "\"q_head\":" << q_head << ","
      << "\"kv_head\":" << kv_head << ","
      << "\"sequence_length\":" << total_positions << ","
      << "\"q_vector\":" << vector_window_json(q_head_values) << ","
      << "\"k_cache_first3\":" << cache_positions_json(true) << ","
      << "\"v_cache_first3\":" << cache_positions_json(false) << ","
      << "\"scores\":{"
      << "\"cpu\":" << vector_window_json(cpu_scores) << ","
      << "\"mlx\":" << vector_window_json(mlx_scores) << ","
      << "\"max_abs_diff\":" << vector_max_abs_diff(cpu_scores, mlx_scores) << ","
      << "\"argmax_cpu\":" << cpu_argmax << ","
      << "\"argmax_mlx\":" << mlx_argmax << ","
      << "\"top5_cpu\":" << top_positions_json(cpu_scores, 5) << ","
      << "\"top5_mlx\":" << top_positions_json(mlx_scores, 5)
      << "},"
      << "\"probabilities\":{"
      << "\"cpu\":" << vector_window_json(cpu_probs) << ","
      << "\"mlx\":" << vector_window_json(mlx_probs) << ","
      << "\"max_abs_diff\":" << vector_max_abs_diff(cpu_probs, mlx_probs) << ","
      << "\"sum_cpu\":" << vector_checksum(cpu_probs) << ","
      << "\"sum_mlx\":" << mlx_prob_sum << ","
      << "\"top5_cpu\":" << top_positions_json(cpu_probs, 5) << ","
      << "\"top5_mlx\":" << top_positions_json(mlx_probs, 5)
      << "},"
      << "\"value_mix_output\":{"
      << "\"cpu\":" << vector_window_json(cpu_output) << ","
      << "\"mlx\":" << vector_window_json(mlx_output) << ","
      << "\"max_abs_diff\":" << vector_max_abs_diff(cpu_output, mlx_output)
      << "}"
      << "}";
  return out.str();
}
#endif

mlx::core::array make_dequantized_slice_array(
    const TensorGroupRecord& record,
    std::size_t row,
    std::size_t cols) {
  const auto values = provisional_dequant_slice_values(record, row, cols);
  auto buffer = mlx::core::allocator::malloc(values.size() * sizeof(float));
  auto* out = static_cast<float*>(buffer.ptr());
  std::copy(values.begin(), values.end(), out);

  return mlx::core::array(
      std::move(buffer),
      mlx::core::Shape{1, static_cast<int>(values.size())},
      mlx::core::float32);
}

struct GroupLoadSpec {
  std::string group;
  bool quantized_group = false;
  std::vector<std::string> tensor_names;
};

std::optional<GroupLoadSpec> group_load_spec_for(const std::string& group) {
  const std::string quantized_suffixes[] = {
      "self_attn.q_proj",
      "self_attn.k_proj",
      "self_attn.v_proj",
      "self_attn.o_proj",
      "mlp.gate_proj",
      "mlp.up_proj",
      "mlp.down_proj",
  };
  for (const auto& suffix : quantized_suffixes) {
    if (string_ends_with(group, suffix)) {
      return GroupLoadSpec{
          group,
          true,
          {group + ".weight", group + ".scales", group + ".biases"},
      };
    }
  }
  const std::string norm_suffixes[] = {
      "input_layernorm",
      "post_attention_layernorm",
      "self_attn.q_norm",
      "self_attn.k_norm",
  };
  for (const auto& suffix : norm_suffixes) {
    if (string_ends_with(group, suffix)) {
      return GroupLoadSpec{
          group,
          false,
          {group + ".weight"},
      };
    }
  }
  if (group == "model.norm") {
    return GroupLoadSpec{
        group,
        false,
        {group + ".weight"},
    };
  }
  return std::nullopt;
}

std::optional<GroupLoadSpec> embedding_group_spec_for() {
  return GroupLoadSpec{
      "model.embed_tokens",
      true,
      {
          "model.embed_tokens.weight",
          "model.embed_tokens.scales",
          "model.embed_tokens.biases",
      },
  };
}

extern "C" bool load_tensor_group_record(
    const std::string& model_dir_str,
    const GroupLoadSpec& spec,
    TensorGroupRecord& record,
    ResidentLoadTiming* timing = nullptr);

std::vector<float> layer0_embedding_values(
    const std::string& model_dir_str,
    std::size_t token_id) {
  TensorGroupRecord embedding;
  auto embedding_spec = embedding_group_spec_for();
  if (!load_tensor_group_record(model_dir_str, *embedding_spec, embedding)) {
    throw std::runtime_error("failed to load embedding group");
  }
  const auto& embedding_weight = embedding.tensors.at("weight");
  const std::size_t vocab_size = embedding_weight.shape.size() > 0
      ? static_cast<std::size_t>(embedding_weight.shape[0])
      : 0;
  const std::size_t embedding_packed_width = embedding_weight.shape.size() > 1
      ? static_cast<std::size_t>(embedding_weight.shape[1])
      : 0;
  const std::size_t hidden_size = embedding_packed_width * 8;
  if (vocab_size == 0 || hidden_size == 0 || token_id >= vocab_size) {
    throw std::runtime_error("token id out of range");
  }
  return provisional_dequant_slice_values(embedding, token_id, hidden_size);
}

std::vector<float> embedding_values_from_record(
    const TensorGroupRecord& embedding,
    std::size_t token_id) {
  const auto weight_it = embedding.tensors.find("weight");
  if (weight_it == embedding.tensors.end()) {
    throw std::runtime_error("embedding group is missing weight");
  }
  const auto& embedding_weight = weight_it->second;
  const std::size_t vocab_size = embedding_weight.shape.size() > 0
      ? static_cast<std::size_t>(embedding_weight.shape[0])
      : 0;
  const std::size_t embedding_packed_width = embedding_weight.shape.size() > 1
      ? static_cast<std::size_t>(embedding_weight.shape[1])
      : 0;
  const std::size_t hidden_size = embedding_packed_width * 8;
  if (vocab_size == 0 || hidden_size == 0 || token_id >= vocab_size) {
    throw std::runtime_error("token id out of range");
  }
  return provisional_dequant_slice_values(embedding, token_id, hidden_size);
}

std::vector<float> layer0_attention_residual_values(
    const std::string& model_dir_str,
    std::size_t token_id,
    double eps) {
  TensorGroupRecord input_norm;
  TensorGroupRecord q_proj;
  TensorGroupRecord k_proj;
  TensorGroupRecord v_proj;
  TensorGroupRecord o_proj;
  TensorGroupRecord q_norm;
  TensorGroupRecord k_norm;
  const std::string groups[] = {
      "model.layers.0.input_layernorm",
      "model.layers.0.self_attn.q_proj",
      "model.layers.0.self_attn.k_proj",
      "model.layers.0.self_attn.v_proj",
      "model.layers.0.self_attn.o_proj",
      "model.layers.0.self_attn.q_norm",
      "model.layers.0.self_attn.k_norm",
  };
  TensorGroupRecord* records[] = {
      &input_norm,
      &q_proj,
      &k_proj,
      &v_proj,
      &o_proj,
      &q_norm,
      &k_norm,
  };
  for (std::size_t i = 0; i < 7; ++i) {
    auto spec = group_load_spec_for(groups[i]);
    if (!spec || !load_tensor_group_record(model_dir_str, *spec, *records[i])) {
      throw std::runtime_error("failed to load layer 0 attention tensor group");
    }
  }

  std::vector<float> embedding_values = layer0_embedding_values(model_dir_str, token_id);
  std::vector<float> norm_values = rmsnorm_values(input_norm, embedding_values, eps);
  std::vector<float> q_values = quantized_linear_vector_values(q_proj, norm_values);
  std::vector<float> k_values = quantized_linear_vector_values(k_proj, norm_values);
  std::vector<float> v_values = quantized_linear_vector_values(v_proj, norm_values);
  apply_qk_norm_in_place(q_norm, k_norm, q_values, k_values, eps);
  double qk_score_checksum = 0.0;
  std::vector<float> attention_values =
      single_token_attention_values(q_values, k_values, v_values, qk_score_checksum);
  std::vector<float> o_values = quantized_linear_vector_values(o_proj, attention_values);
  if (o_values.size() != embedding_values.size()) {
    throw std::runtime_error("o_proj output does not match residual width");
  }

  std::vector<float> residual_values(o_values.size(), 0.0f);
  for (std::size_t i = 0; i < residual_values.size(); ++i) {
    residual_values[i] = embedding_values[i] + o_values[i];
  }
  return residual_values;
}

std::vector<float> layer_attention_residual_from_input(
    const std::string& model_dir_str,
    int layer,
    const std::vector<float>& input_values,
    double eps,
    NativeSessionKvCache* kv_cache = nullptr) {
  TensorGroupRecord input_norm;
  TensorGroupRecord q_proj;
  TensorGroupRecord k_proj;
  TensorGroupRecord v_proj;
  TensorGroupRecord o_proj;
  TensorGroupRecord q_norm;
  TensorGroupRecord k_norm;
  const std::string prefix = "model.layers." + std::to_string(layer) + ".";
  const std::string groups[] = {
      prefix + "input_layernorm",
      prefix + "self_attn.q_proj",
      prefix + "self_attn.k_proj",
      prefix + "self_attn.v_proj",
      prefix + "self_attn.o_proj",
      prefix + "self_attn.q_norm",
      prefix + "self_attn.k_norm",
  };
  TensorGroupRecord* records[] = {
      &input_norm,
      &q_proj,
      &k_proj,
      &v_proj,
      &o_proj,
      &q_norm,
      &k_norm,
  };
  for (std::size_t i = 0; i < 7; ++i) {
    auto spec = group_load_spec_for(groups[i]);
    if (!spec || !load_tensor_group_record(model_dir_str, *spec, *records[i])) {
      throw std::runtime_error("failed to load attention tensor group");
    }
  }

  std::vector<float> norm_values = rmsnorm_values(input_norm, input_values, eps);
  std::vector<float> q_values = quantized_linear_vector_values(q_proj, norm_values);
  std::vector<float> k_values = quantized_linear_vector_values(k_proj, norm_values);
  std::vector<float> v_values = quantized_linear_vector_values(v_proj, norm_values);
  apply_qk_norm_in_place(q_norm, k_norm, q_values, k_values, eps);
  append_native_kv(kv_cache, static_cast<std::size_t>(layer), k_values, v_values);
  double qk_score_checksum = 0.0;
  std::vector<float> attention_values =
      single_token_attention_values(q_values, k_values, v_values, qk_score_checksum);
  std::vector<float> o_values = quantized_linear_vector_values(o_proj, attention_values);
  if (o_values.size() != input_values.size()) {
    throw std::runtime_error("attention o_proj output does not match residual width");
  }

  std::vector<float> residual_values(o_values.size(), 0.0f);
  for (std::size_t i = 0; i < residual_values.size(); ++i) {
    residual_values[i] = input_values[i] + o_values[i];
  }
  return residual_values;
}

std::vector<float> layer_attention_residual_incremental_from_input(
    const std::string& model_dir_str,
    int layer,
    const std::vector<float>& input_values,
    double eps,
    NativeSessionKvCache& kv_cache) {
  TensorGroupRecord input_norm;
  TensorGroupRecord q_proj;
  TensorGroupRecord k_proj;
  TensorGroupRecord v_proj;
  TensorGroupRecord o_proj;
  TensorGroupRecord q_norm;
  TensorGroupRecord k_norm;
  const std::string prefix = "model.layers." + std::to_string(layer) + ".";
  const std::string groups[] = {
      prefix + "input_layernorm",
      prefix + "self_attn.q_proj",
      prefix + "self_attn.k_proj",
      prefix + "self_attn.v_proj",
      prefix + "self_attn.o_proj",
      prefix + "self_attn.q_norm",
      prefix + "self_attn.k_norm",
  };
  TensorGroupRecord* records[] = {
      &input_norm,
      &q_proj,
      &k_proj,
      &v_proj,
      &o_proj,
      &q_norm,
      &k_norm,
  };
  for (std::size_t i = 0; i < 7; ++i) {
    auto spec = group_load_spec_for(groups[i]);
    if (!spec || !load_tensor_group_record(model_dir_str, *spec, *records[i])) {
      throw std::runtime_error("failed to load incremental attention tensor group");
    }
  }
  if (layer < 0 || static_cast<std::size_t>(layer) >= kv_cache.layers.size()) {
    throw std::runtime_error("incremental attention layer missing from KV cache");
  }

  std::vector<float> norm_values = rmsnorm_values(input_norm, input_values, eps);
  std::vector<float> q_values = quantized_linear_vector_values(q_proj, norm_values);
  std::vector<float> k_values = quantized_linear_vector_values(k_proj, norm_values);
  std::vector<float> v_values = quantized_linear_vector_values(v_proj, norm_values);
  apply_qk_norm_in_place(q_norm, k_norm, q_values, k_values, eps);
  std::vector<float> attention_values = cached_single_token_attention_values(
      q_values,
      kv_cache.layers[static_cast<std::size_t>(layer)],
      k_values,
      v_values);
  append_native_kv(&kv_cache, static_cast<std::size_t>(layer), k_values, v_values);

  std::vector<float> o_values = quantized_linear_vector_values(o_proj, attention_values);
  if (o_values.size() != input_values.size()) {
    throw std::runtime_error("incremental attention o_proj output does not match residual width");
  }

  std::vector<float> residual_values(o_values.size(), 0.0f);
  for (std::size_t i = 0; i < residual_values.size(); ++i) {
    residual_values[i] = input_values[i] + o_values[i];
  }
  return residual_values;
}

std::vector<float> layer_attention_residual_resident_from_input(
    const ResidentLayerGroups& groups,
    std::size_t layer,
    const std::vector<float>& input_values,
    double eps,
    NativeSessionKvCache* kv_cache = nullptr,
    bool use_optimized_linear = false,
    bool use_layout_cached_linear = false) {
  std::vector<float> norm_values = rmsnorm_values(groups.input_norm, input_values, eps);
  std::vector<float> q_values;
  std::vector<float> k_values;
  std::vector<float> v_values;
  if (use_optimized_linear &&
      use_layout_cached_linear &&
      mlx_quantized_linear_available_flag() &&
      !disable_mlx_quantized_linear()) {
    MlxQuantizedLinearStepScope qkv_scope("qkv");
    try {
      auto qkv = quantized_linear_triple_values_mlx(
          groups.q_proj,
          groups.k_proj,
          groups.v_proj,
          norm_values);
      q_values = std::move(std::get<0>(qkv));
      k_values = std::move(std::get<1>(qkv));
      v_values = std::move(std::get<2>(qkv));
    } catch (...) {
      record_mlx_quantized_linear_fallback_step();
      q_values = quantized_linear_vector_values_layout_cached(groups.q_proj, norm_values);
      k_values = quantized_linear_vector_values_layout_cached(groups.k_proj, norm_values);
      v_values = quantized_linear_vector_values_layout_cached(groups.v_proj, norm_values);
    }
  } else {
    q_values =
        quantized_linear_vector_values_selected(
            groups.q_proj, norm_values, use_optimized_linear, use_layout_cached_linear);
    k_values =
        quantized_linear_vector_values_selected(
            groups.k_proj, norm_values, use_optimized_linear, use_layout_cached_linear);
    v_values =
        quantized_linear_vector_values_selected(
            groups.v_proj, norm_values, use_optimized_linear, use_layout_cached_linear);
  }
  apply_qk_norm_in_place(groups.q_norm, groups.k_norm, q_values, k_values, eps);
  append_native_kv(kv_cache, layer, k_values, v_values);
  double qk_score_checksum = 0.0;
  std::vector<float> attention_values =
      single_token_attention_values(q_values, k_values, v_values, qk_score_checksum);
  std::vector<float> o_values =
      quantized_linear_vector_values_selected(
          groups.o_proj, attention_values, use_optimized_linear, use_layout_cached_linear);
  if (o_values.size() != input_values.size()) {
    throw std::runtime_error("resident attention o_proj output does not match residual width");
  }

  std::vector<float> residual_values(o_values.size(), 0.0f);
  for (std::size_t i = 0; i < residual_values.size(); ++i) {
    residual_values[i] = input_values[i] + o_values[i];
  }
  return residual_values;
}

std::vector<float> layer_attention_residual_resident_incremental_from_input(
    const ResidentLayerGroups& groups,
    std::size_t layer,
    const std::vector<float>& input_values,
    double eps,
    NativeSessionKvCache& kv_cache,
    bool use_optimized_linear = false,
    bool use_layout_cached_linear = false,
    bool use_mlx_resident_layer_block = false,
    ResidentDecodeResult* timing = nullptr) {
  if (layer >= kv_cache.layers.size()) {
    throw std::runtime_error("resident incremental attention layer missing from KV cache");
  }
  std::vector<float> norm_values = rmsnorm_values(groups.input_norm, input_values, eps);
  auto qkv_start = std::chrono::steady_clock::now();
  std::vector<float> q_values;
  std::vector<float> k_values;
  std::vector<float> v_values;
  std::optional<mlx::core::array> q_mlx_after_rope;
  std::optional<mlx::core::array> k_mlx_after_rope;
  std::optional<mlx::core::array> v_mlx;
  MlxQuantizedLinearStepScope qkv_scope("qkv");
  if (use_optimized_linear &&
      use_layout_cached_linear &&
      mlx_quantized_linear_available_flag() &&
      !disable_mlx_quantized_linear()) {
    try {
      auto qkv = quantized_linear_triple_values_mlx(
          groups.q_proj,
          groups.k_proj,
          groups.v_proj,
          norm_values);
      q_values = std::move(std::get<0>(qkv));
      k_values = std::move(std::get<1>(qkv));
      v_values = std::move(std::get<2>(qkv));
    } catch (...) {
      record_mlx_quantized_linear_fallback_step();
      q_values = quantized_linear_vector_values_layout_cached(groups.q_proj, norm_values);
      k_values = quantized_linear_vector_values_layout_cached(groups.k_proj, norm_values);
      v_values = quantized_linear_vector_values_layout_cached(groups.v_proj, norm_values);
    }
  } else {
    q_values =
        quantized_linear_vector_values_selected(
            groups.q_proj, norm_values, use_optimized_linear, use_layout_cached_linear);
    k_values =
        quantized_linear_vector_values_selected(
            groups.k_proj, norm_values, use_optimized_linear, use_layout_cached_linear);
    v_values =
        quantized_linear_vector_values_selected(
            groups.v_proj, norm_values, use_optimized_linear, use_layout_cached_linear);
  }
  apply_qk_norm_in_place(groups.q_norm, groups.k_norm, q_values, k_values, eps);
  if (timing != nullptr) {
    add_elapsed_ms(timing->qkv_projection_ms, qkv_start);
  }
  MlxQuantizedLinearStepScope attention_scope("attention");
  CachedAttentionTiming attention_timing;
  std::vector<float> attention_values = cached_single_token_attention_values(
      q_values,
      kv_cache.layers[layer],
      k_values,
      v_values,
      &attention_timing);
  auto kv_append_start = std::chrono::steady_clock::now();
  append_native_kv(&kv_cache, layer, k_values, v_values);
  if (timing != nullptr) {
    add_elapsed_ms(attention_timing.kv_append_ms, kv_append_start);
    timing->kv_append_ms += attention_timing.kv_append_ms;
    timing->attention_score_ms += attention_timing.score_ms;
    timing->attention_softmax_ms += attention_timing.softmax_ms;
    timing->attention_value_mix_ms += attention_timing.value_mix_ms;
    timing->per_layer_attention_ms.push_back(attention_timing.total_attention_ms());
  }
  auto o_start = std::chrono::steady_clock::now();
  MlxQuantizedLinearStepScope o_scope("o_proj");
  std::vector<float> residual_values;
  const bool metal_resident_o_residual_available =
      use_mlx_resident_layer_block &&
      use_optimized_linear &&
      use_layout_cached_linear &&
      mlx_quantized_linear_available_flag() &&
      !disable_mlx_quantized_linear() &&
      !disable_mlx_resident_layer_block();
  if (metal_resident_o_residual_available) {
    try {
      MlxProjectionTiming o_mlx_timing;
      residual_values = quantized_linear_residual_values_mlx(
          groups.o_proj,
          attention_values,
          input_values,
          &o_mlx_timing);
      if (timing != nullptr) {
        timing->mlx_resident_layer_block_applied = true;
      }
    } catch (...) {
      record_mlx_quantized_linear_fallback_step();
      if (timing != nullptr) {
        timing->mlx_resident_layer_block_applied = false;
        timing->mlx_resident_layer_block_fallback_used = true;
      }
    }
  }
  if (residual_values.empty()) {
    std::vector<float> o_values =
        quantized_linear_vector_values_selected(
            groups.o_proj, attention_values, use_optimized_linear, use_layout_cached_linear);
    if (o_values.size() != input_values.size()) {
      throw std::runtime_error("resident incremental o_proj output does not match residual width");
    }
    residual_values.assign(o_values.size(), 0.0f);
    for (std::size_t i = 0; i < residual_values.size(); ++i) {
      residual_values[i] = input_values[i] + o_values[i];
    }
  }
  if (timing != nullptr) {
    add_elapsed_ms(timing->o_projection_ms, o_start);
  }
  return residual_values;
}

#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
ResidentDecodeValue layer_decode_value_resident_incremental_from_input(
    const ResidentLayerGroups& groups,
    std::size_t layer,
    const ResidentDecodeValue& input_value,
    double eps,
    NativeSessionKvCache& kv_cache,
    ResidentDecodeResult* timing = nullptr) {
  if (layer >= kv_cache.layers.size()) {
    throw std::runtime_error("resident decode value attention layer missing from KV cache");
  }
  auto input_norm = rmsnorm_mlx_array(groups.input_norm, input_value.mlx, eps);
  const LoraLayerRecord* lora_layer = adapter_layer_for(layer);
  auto qkv_start = std::chrono::steady_clock::now();
  std::vector<float> q_values;
  std::vector<float> k_values;
  std::vector<float> v_values;
  std::optional<mlx::core::array> q_mlx_after_rope;
  std::optional<mlx::core::array> k_mlx_after_rope;
  std::optional<mlx::core::array> v_mlx;
  MlxQuantizedLinearStepScope qkv_scope("qkv");
  auto qk_norm_rope_start = std::chrono::steady_clock::now();
  const std::uint64_t rope_position = layer_kv_cache_len(kv_cache.layers[layer]);
  bool qk_norm_rope_done_on_mlx = false;
  try {
    if (experimental_mlx_qk_norm_rope_enabled()) {
      auto qkv_arrays = quantized_linear_triple_arrays_from_mlx_input(
          groups.q_proj,
          groups.k_proj,
          groups.v_proj,
          input_norm,
          lora_layer == nullptr ? nullptr : &lora_layer->q_proj,
          lora_layer == nullptr ? nullptr : &lora_layer->k_proj,
          lora_layer == nullptr ? nullptr : &lora_layer->v_proj);
      auto qk_arrays = qk_norm_rope_mlx_arrays(
          groups.q_norm,
          groups.k_norm,
          std::get<0>(qkv_arrays),
          std::get<1>(qkv_arrays),
          eps,
          rope_position);
      auto q_array = qk_arrays.first;
      auto k_array = qk_arrays.second;
      auto v_array = std::get<2>(qkv_arrays);
      if (verify_mlx_qk_norm_rope_enabled()) {
        mlx::core::eval(q_array, k_array, v_array);
        mlx::core::synchronize();
      }
      if (q_array.size() != static_cast<int>(groups.q_proj.quantized_rows) ||
          k_array.size() != static_cast<int>(groups.k_proj.quantized_rows) ||
          v_array.size() != static_cast<int>(groups.v_proj.quantized_rows)) {
        throw std::runtime_error("MLX q/k norm RoPE qkv output length mismatch");
      }
      mlx_qk_norm_rope_applied_to_generation = true;
      if (verify_mlx_qk_norm_rope_enabled()) {
        const float* q_data = q_array.data<float>();
        const float* k_data = k_array.data<float>();
        const float* v_data = v_array.data<float>();
        q_values.assign(q_data, q_data + groups.q_proj.quantized_rows);
        k_values.assign(k_data, k_data + groups.k_proj.quantized_rows);
        v_values.assign(v_data, v_data + groups.v_proj.quantized_rows);
        record_mlx_decode_readback("qkv_after_mlx_qk_norm_rope_verify");
        auto cpu_qkv = quantized_linear_triple_values_from_mlx_input(
            groups.q_proj,
            groups.k_proj,
            groups.v_proj,
            input_norm,
            lora_layer == nullptr ? nullptr : &lora_layer->q_proj,
            lora_layer == nullptr ? nullptr : &lora_layer->k_proj,
            lora_layer == nullptr ? nullptr : &lora_layer->v_proj);
        std::vector<float> cpu_q = std::move(std::get<0>(cpu_qkv));
        std::vector<float> cpu_k = std::move(std::get<1>(cpu_qkv));
        apply_qk_norm_in_place(groups.q_norm, groups.k_norm, cpu_q, cpu_k, eps);
        apply_active_rope_to_qk(cpu_q, cpu_k, rope_position);
        if (cpu_q.size() != q_values.size() || cpu_k.size() != k_values.size()) {
          throw std::runtime_error("MLX q/k norm RoPE verification shape mismatch");
        }
        double max_diff = 0.0;
        for (std::size_t i = 0; i < q_values.size(); ++i) {
          max_diff = std::max(
              max_diff,
              std::abs(static_cast<double>(q_values[i]) - static_cast<double>(cpu_q[i])));
        }
        for (std::size_t i = 0; i < k_values.size(); ++i) {
          max_diff = std::max(
              max_diff,
              std::abs(static_cast<double>(k_values[i]) - static_cast<double>(cpu_k[i])));
        }
        mlx_qk_norm_rope_verify_ran = true;
        mlx_qk_norm_rope_verify_max_abs_diff =
            std::max(mlx_qk_norm_rope_verify_max_abs_diff, max_diff);
      }
      q_mlx_after_rope.emplace(std::move(q_array));
      k_mlx_after_rope.emplace(std::move(k_array));
      v_mlx.emplace(std::move(v_array));
      qk_norm_rope_done_on_mlx = true;
    }
  } catch (...) {
    mlx_qk_norm_rope_fallback_used = true;
    record_mlx_quantized_linear_fallback_step();
    q_values.clear();
    k_values.clear();
    v_values.clear();
    qk_norm_rope_done_on_mlx = false;
  }
  if (!qk_norm_rope_done_on_mlx &&
      (q_values.empty() || k_values.empty() || v_values.empty())) {
    try {
      auto qkv = quantized_linear_triple_values_from_mlx_input(
          groups.q_proj,
          groups.k_proj,
          groups.v_proj,
          input_norm,
          lora_layer == nullptr ? nullptr : &lora_layer->q_proj,
          lora_layer == nullptr ? nullptr : &lora_layer->k_proj,
          lora_layer == nullptr ? nullptr : &lora_layer->v_proj);
      q_values = std::move(std::get<0>(qkv));
      k_values = std::move(std::get<1>(qkv));
      v_values = std::move(std::get<2>(qkv));
    } catch (...) {
      record_mlx_quantized_linear_fallback_step();
      std::vector<float> input_norm_values =
          ResidentDecodeValue(input_norm).cpu("qkv_cpu_fallback_input_norm");
      q_values = quantized_linear_vector_values_layout_cached(groups.q_proj, input_norm_values);
      k_values = quantized_linear_vector_values_layout_cached(groups.k_proj, input_norm_values);
      v_values = quantized_linear_vector_values_layout_cached(groups.v_proj, input_norm_values);
    }
  }
  if (timing != nullptr) {
    add_elapsed_ms(timing->qkv_projection_ms, qkv_start);
  }
  if (!qk_norm_rope_done_on_mlx) {
    apply_qk_norm_in_place(groups.q_norm, groups.k_norm, q_values, k_values, eps);
    apply_active_rope_to_qk(q_values, k_values, rope_position);
  }
  if (timing != nullptr) {
    add_elapsed_ms(timing->qk_norm_rope_ms, qk_norm_rope_start);
  }

  if (experimental_mlx_resident_block()) {
    const auto resident_block_start = std::chrono::steady_clock::now();
    try {
      const std::string attention_mode = experimental_mlx_attention_mode();
      if (attention_mode != "chunked_expanded_kv") {
        throw std::runtime_error("resident block requires chunked_expanded_kv attention");
      }
      NativeLayerKvCache resident_layer_cache = kv_cache.layers[layer];
      double append_ms = 0.0;
      mlx_expanded_kv_cache_append_chunked(
          resident_layer_cache,
          k_values,
          v_values,
          active_rope_config().num_attention_heads,
          active_rope_config().num_key_value_heads,
          active_rope_config().head_dim,
          expanded_kv_chunk_size(),
          &append_ms);
      MlxAttentionDetailTiming attention_detail_timing;
      auto attention_array = mlx_expanded_kv_cache_attention_chunked_array(
          q_values,
          resident_layer_cache,
          active_rope_config().num_attention_heads,
          active_rope_config().head_dim,
          &attention_detail_timing);

      auto o_start = std::chrono::steady_clock::now();
      LocalQuantizedMlxArrays o_local;
      auto o_result = make_quantized_matmul_from_mlx_input(
          groups.o_proj,
          attention_array,
          o_local,
          attention_array.size());
      o_result = apply_lora_delta_if_present(
          o_result,
          attention_array,
          lora_layer == nullptr ? nullptr : &lora_layer->o_proj);
      auto attention_residual = o_result + input_value.mlx;
      double o_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - o_start).count();

      auto post_norm = rmsnorm_mlx_array(groups.post_attention_norm, attention_residual, eps);
      MlxResidentMlpChainTiming mlp_timing;
      auto mlp_residual = quantized_linear_mlp_chain_residual_mlx(
          groups.gate_proj,
          groups.up_proj,
          groups.down_proj,
          post_norm,
          attention_residual,
          &mlp_timing,
          lora_layer == nullptr ? nullptr : &lora_layer->gate_proj,
          lora_layer == nullptr ? nullptr : &lora_layer->up_proj,
          lora_layer == nullptr ? nullptr : &lora_layer->down_proj);

      kv_cache.layers[layer] = std::move(resident_layer_cache);
      append_native_kv(&kv_cache, layer, k_values, v_values);
      if (timing != nullptr) {
        timing->kv_append_ms += append_ms;
        timing->attention_value_mix_ms +=
            attention_detail_timing.kv_view_assembly_ms +
            attention_detail_timing.score_matmul_ms +
            attention_detail_timing.softmax_ms +
            attention_detail_timing.value_mix_matmul_ms +
            attention_detail_timing.reshape_flatten_ms;
        timing->attention_kv_view_assembly_ms += attention_detail_timing.kv_view_assembly_ms;
        timing->attention_score_matmul_ms += attention_detail_timing.score_matmul_ms;
        timing->attention_softmax_detail_ms += attention_detail_timing.softmax_ms;
        timing->attention_value_mix_matmul_ms += attention_detail_timing.value_mix_matmul_ms;
        timing->attention_reshape_flatten_ms += attention_detail_timing.reshape_flatten_ms;
        timing->attention_eval_sync_ms += mlp_timing.down_eval_ms;
        timing->o_projection_ms += o_ms;
        timing->gate_up_projection_ms += mlp_timing.gate_up_eval_ms + mlp_timing.setup_ms;
        timing->gate_up_activation_ms += mlp_timing.activation_eval_ms;
        timing->down_projection_ms += mlp_timing.down_eval_ms;
        timing->per_layer_attention_ms.push_back(
            attention_detail_timing.kv_view_assembly_ms +
            attention_detail_timing.score_matmul_ms +
            attention_detail_timing.softmax_ms +
            attention_detail_timing.value_mix_matmul_ms +
            attention_detail_timing.reshape_flatten_ms);
        timing->per_layer_attention_backends.push_back("mlx_resident_block_chunked_expanded_kv");
        timing->per_layer_attention_fallback_reasons.push_back("");
        timing->mlx_resident_layer_block_applied = true;
        timing->gate_backend = "metal";
        timing->up_backend = "metal";
        timing->activation_backend = "metal";
        add_elapsed_ms(timing->timing_ms, resident_block_start);
      }
      return ResidentDecodeValue(mlp_residual);
    } catch (const std::exception& e) {
      if (timing != nullptr) {
        timing->mlx_resident_layer_block_applied = false;
        timing->mlx_resident_layer_block_fallback_used = true;
        timing->per_layer_attention_fallback_reasons.push_back(e.what());
      }
      record_mlx_quantized_linear_fallback_step();
    } catch (...) {
      if (timing != nullptr) {
        timing->mlx_resident_layer_block_applied = false;
        timing->mlx_resident_layer_block_fallback_used = true;
        timing->per_layer_attention_fallback_reasons.push_back("unknown_exception");
      }
      record_mlx_quantized_linear_fallback_step();
    }
  }

  if (qk_norm_rope_done_on_mlx &&
      q_mlx_after_rope.has_value() &&
      k_mlx_after_rope.has_value() &&
      v_mlx.has_value() &&
      experimental_mlx_attention_mode() == "chunked_compact_mlx") {
    try {
      double append_ms = 0.0;
      MlxAttentionDetailTiming attention_detail_timing;
      double attention_ms = 0.0;
      mlx_compact_kv_cache_append_chunked_arrays(
          kv_cache.layers[layer],
          *k_mlx_after_rope,
          *v_mlx,
          active_rope_config().num_key_value_heads,
          active_rope_config().head_dim,
          expanded_kv_chunk_size(),
          &append_ms);
      auto attention_array = mlx_compact_kv_cache_attention_chunk_aware_array(
          *q_mlx_after_rope,
          kv_cache.layers[layer],
          active_rope_config().num_attention_heads,
          active_rope_config().num_key_value_heads,
          active_rope_config().head_dim,
          &attention_ms,
          &attention_detail_timing);

      auto o_start = std::chrono::steady_clock::now();
      LocalQuantizedMlxArrays o_local;
      auto o_result = make_quantized_matmul_from_mlx_input(
          groups.o_proj,
          attention_array,
          o_local,
          attention_array.size());
      o_result = apply_lora_delta_if_present(
          o_result,
          attention_array,
          lora_layer == nullptr ? nullptr : &lora_layer->o_proj);
      auto attention_residual = o_result + input_value.mlx;
      double o_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - o_start).count();

      auto post_norm = rmsnorm_mlx_array(groups.post_attention_norm, attention_residual, eps);
      MlxResidentMlpChainTiming mlp_timing;
      auto mlp_residual = quantized_linear_mlp_chain_residual_mlx(
          groups.gate_proj,
          groups.up_proj,
          groups.down_proj,
          post_norm,
          attention_residual,
          &mlp_timing,
          lora_layer == nullptr ? nullptr : &lora_layer->gate_proj,
          lora_layer == nullptr ? nullptr : &lora_layer->up_proj,
          lora_layer == nullptr ? nullptr : &lora_layer->down_proj);

      kv_cache.layers_allocated = std::max<std::uint64_t>(
          kv_cache.layers_allocated,
          static_cast<std::uint64_t>(kv_cache.layers.size()));
      ChunkedExpandedKvCache layer_cache(kv_cache.layers[layer]);
      kv_cache.positions_stored = std::max<std::uint64_t>(
          kv_cache.positions_stored,
          layer_cache.len());

      if (timing != nullptr) {
        timing->kv_append_ms += append_ms;
        timing->attention_value_mix_ms += attention_ms;
        timing->attention_kv_view_assembly_ms += attention_detail_timing.kv_view_assembly_ms;
        timing->attention_score_matmul_ms += attention_detail_timing.score_matmul_ms;
        timing->attention_softmax_detail_ms += attention_detail_timing.softmax_ms;
        timing->attention_value_mix_matmul_ms += attention_detail_timing.value_mix_matmul_ms;
        timing->attention_reshape_flatten_ms += attention_detail_timing.reshape_flatten_ms;
        timing->attention_eval_sync_ms += attention_detail_timing.eval_sync_ms + mlp_timing.down_eval_ms;
        timing->o_projection_ms += o_ms;
        timing->gate_up_projection_ms += mlp_timing.gate_up_eval_ms + mlp_timing.setup_ms;
        timing->gate_up_activation_ms += mlp_timing.activation_eval_ms;
        timing->down_projection_ms += mlp_timing.down_eval_ms;
        timing->per_layer_attention_ms.push_back(attention_ms);
        timing->per_layer_attention_backends.push_back("mlx_qk_norm_rope_chunked_compact_mlx");
        timing->per_layer_attention_fallback_reasons.push_back("");
        timing->gate_backend = "metal";
        timing->up_backend = "metal";
        timing->activation_backend = "metal";
      }
      return ResidentDecodeValue(mlp_residual);
    } catch (const std::exception& e) {
      mlx_qk_norm_rope_fallback_used = true;
      if (timing != nullptr) {
        timing->per_layer_attention_fallback_reasons.push_back(e.what());
      }
      record_mlx_quantized_linear_fallback_step();
    } catch (...) {
      mlx_qk_norm_rope_fallback_used = true;
      if (timing != nullptr) {
        timing->per_layer_attention_fallback_reasons.push_back("unknown_exception");
      }
      record_mlx_quantized_linear_fallback_step();
    }
  }

  if (q_values.empty() || k_values.empty() || v_values.empty()) {
    try {
      auto qkv = quantized_linear_triple_values_from_mlx_input(
          groups.q_proj,
          groups.k_proj,
          groups.v_proj,
          input_norm,
          lora_layer == nullptr ? nullptr : &lora_layer->q_proj,
          lora_layer == nullptr ? nullptr : &lora_layer->k_proj,
          lora_layer == nullptr ? nullptr : &lora_layer->v_proj);
      q_values = std::move(std::get<0>(qkv));
      k_values = std::move(std::get<1>(qkv));
      v_values = std::move(std::get<2>(qkv));
      apply_qk_norm_in_place(groups.q_norm, groups.k_norm, q_values, k_values, eps);
      apply_active_rope_to_qk(q_values, k_values, rope_position);
    } catch (...) {
      record_mlx_quantized_linear_fallback_step();
      std::vector<float> input_norm_values =
          ResidentDecodeValue(input_norm).cpu("qkv_cpu_fallback_input_norm");
      q_values = quantized_linear_vector_values_layout_cached(groups.q_proj, input_norm_values);
      k_values = quantized_linear_vector_values_layout_cached(groups.k_proj, input_norm_values);
      v_values = quantized_linear_vector_values_layout_cached(groups.v_proj, input_norm_values);
      apply_qk_norm_in_place(groups.q_norm, groups.k_norm, q_values, k_values, eps);
      apply_active_rope_to_qk(q_values, k_values, rope_position);
    }
  }

  if (resident_attention_to_o_enabled() || resident_o_residual_enabled()) {
    try {
      const bool keep_residual_mlx = resident_o_residual_enabled();
      if (experimental_mlx_attention_mode() != "chunked_expanded_kv") {
        throw std::runtime_error("resident attention/o requires chunked_expanded_kv attention");
      }
      NativeLayerKvCache resident_layer_cache = kv_cache.layers[layer];
      double append_ms = 0.0;
      mlx_expanded_kv_cache_append_chunked(
          resident_layer_cache,
          k_values,
          v_values,
          active_rope_config().num_attention_heads,
          active_rope_config().num_key_value_heads,
          active_rope_config().head_dim,
          expanded_kv_chunk_size(),
          &append_ms);
      MlxAttentionDetailTiming attention_detail_timing;
      auto attention_array = mlx_expanded_kv_cache_attention_chunked_array(
          q_values,
          resident_layer_cache,
          active_rope_config().num_attention_heads,
          active_rope_config().head_dim,
          &attention_detail_timing);
      auto o_start = std::chrono::steady_clock::now();
      LocalQuantizedMlxArrays o_local;
      auto o_result = make_quantized_matmul_from_mlx_input(
          groups.o_proj,
          attention_array,
          o_local,
          attention_array.size());
      o_result = apply_lora_delta_if_present(
          o_result,
          attention_array,
          lora_layer == nullptr ? nullptr : &lora_layer->o_proj);
      kv_cache.layers[layer] = std::move(resident_layer_cache);
      append_native_kv(&kv_cache, layer, k_values, v_values);
      if (timing != nullptr) {
        timing->kv_append_ms += append_ms;
        timing->attention_value_mix_ms +=
            attention_detail_timing.kv_view_assembly_ms +
            attention_detail_timing.score_matmul_ms +
            attention_detail_timing.softmax_ms +
            attention_detail_timing.value_mix_matmul_ms +
            attention_detail_timing.reshape_flatten_ms;
        timing->attention_kv_view_assembly_ms += attention_detail_timing.kv_view_assembly_ms;
        timing->attention_score_matmul_ms += attention_detail_timing.score_matmul_ms;
        timing->attention_softmax_detail_ms += attention_detail_timing.softmax_ms;
        timing->attention_value_mix_matmul_ms += attention_detail_timing.value_mix_matmul_ms;
        timing->attention_reshape_flatten_ms += attention_detail_timing.reshape_flatten_ms;
        timing->per_layer_attention_ms.push_back(
            attention_detail_timing.kv_view_assembly_ms +
            attention_detail_timing.score_matmul_ms +
            attention_detail_timing.softmax_ms +
            attention_detail_timing.value_mix_matmul_ms +
            attention_detail_timing.reshape_flatten_ms);
        timing->per_layer_attention_backends.push_back(
            keep_residual_mlx ? "mlx_resident_o_residual" : "mlx_resident_attention_to_o");
        timing->per_layer_attention_fallback_reasons.push_back("");
        add_elapsed_ms(timing->o_projection_ms, o_start);
      }
      mlx::core::array attention_residual = o_result + input_value.mlx;
      if (!keep_residual_mlx) {
        o_result.eval();
        mlx::core::synchronize();
        const float* o_data = o_result.data<float>();
        std::vector<float> o_values(o_data, o_data + groups.o_proj.quantized_rows);
        std::vector<float> input_values = input_value.cpu("resident_attention_to_o_residual_input");
        if (o_values.size() != input_values.size()) {
          throw std::runtime_error("resident attention/o output length mismatch");
        }
        std::vector<float> residual_values(o_values.size(), 0.0f);
        for (std::size_t i = 0; i < o_values.size(); ++i) {
          residual_values[i] = input_values[i] + o_values[i];
        }
        attention_residual = mlx_array_from_vector(residual_values);
      }
      auto post_norm = rmsnorm_mlx_array(groups.post_attention_norm, attention_residual, eps);
      MlxResidentMlpChainTiming mlp_timing;
      auto mlp_residual = quantized_linear_mlp_chain_residual_mlx(
          groups.gate_proj,
          groups.up_proj,
          groups.down_proj,
          post_norm,
          attention_residual,
          &mlp_timing,
          lora_layer == nullptr ? nullptr : &lora_layer->gate_proj,
          lora_layer == nullptr ? nullptr : &lora_layer->up_proj,
          lora_layer == nullptr ? nullptr : &lora_layer->down_proj);
      if (timing != nullptr) {
        timing->gate_up_projection_ms += mlp_timing.gate_up_eval_ms + mlp_timing.setup_ms;
        timing->gate_up_activation_ms += mlp_timing.activation_eval_ms;
        timing->down_projection_ms += mlp_timing.down_eval_ms;
        timing->attention_eval_sync_ms += mlp_timing.down_eval_ms;
        timing->gate_backend = "metal";
        timing->up_backend = "metal";
        timing->activation_backend = "metal";
      }
      return ResidentDecodeValue(mlp_residual);
    } catch (const std::exception& e) {
      if (timing != nullptr) {
        timing->per_layer_attention_fallback_reasons.push_back(e.what());
      }
      record_mlx_quantized_linear_fallback_step();
    } catch (...) {
      if (timing != nullptr) {
        timing->per_layer_attention_fallback_reasons.push_back("unknown_exception");
      }
      record_mlx_quantized_linear_fallback_step();
    }
  }

  MlxQuantizedLinearStepScope attention_scope("attention");
  CachedAttentionTiming attention_timing;
  MlxAttentionDetailTiming attention_detail_timing;
  std::vector<float> attention_values;
  std::string attention_backend = "cpu";
  std::string attention_fallback_reason;
  bool attention_timing_populated_by_backend = false;
  if (experimental_mlx_attention()) {
    auto attention_start = std::chrono::steady_clock::now();
    try {
      const std::string attention_mode = experimental_mlx_attention_mode();
      if (attention_mode == "chunked_expanded_kv") {
        double append_ms = 0.0;
        double attention_ms = 0.0;
        mlx_expanded_kv_cache_append_chunked(
            kv_cache.layers[layer],
            k_values,
            v_values,
            active_rope_config().num_attention_heads,
            active_rope_config().num_key_value_heads,
            active_rope_config().head_dim,
            expanded_kv_chunk_size(),
            &append_ms);
        attention_values = mlx_expanded_kv_cache_attention_chunked(
            q_values,
            kv_cache.layers[layer],
            active_rope_config().num_attention_heads,
            active_rope_config().head_dim,
            &attention_ms,
            &attention_detail_timing);
        attention_timing.kv_append_ms += append_ms;
        attention_timing.value_mix_ms += attention_ms;
        attention_timing_populated_by_backend = true;
        attention_backend = "mlx_chunked_expanded_kv";
      } else if (attention_mode == "chunked_compact_mlx") {
        double append_ms = 0.0;
        double attention_ms = 0.0;
        const bool chunk_aware_attention = chunk_aware_attention_enabled();
        mlx_compact_kv_cache_append_chunked(
            kv_cache.layers[layer],
            k_values,
            v_values,
            active_rope_config().num_key_value_heads,
            active_rope_config().head_dim,
            expanded_kv_chunk_size(),
            &append_ms);
        if (chunk_aware_attention) {
          attention_values = mlx_compact_kv_cache_attention_chunk_aware(
              q_values,
              kv_cache.layers[layer],
              active_rope_config().num_attention_heads,
              active_rope_config().num_key_value_heads,
              active_rope_config().head_dim,
              &attention_ms,
              &attention_detail_timing);
        } else {
          attention_values = mlx_compact_kv_cache_attention_chunked(
              q_values,
              kv_cache.layers[layer],
              active_rope_config().num_attention_heads,
              active_rope_config().num_key_value_heads,
              active_rope_config().head_dim,
              &attention_ms,
              &attention_detail_timing);
        }
        attention_timing.kv_append_ms += append_ms;
        attention_timing.value_mix_ms += attention_ms;
        attention_timing_populated_by_backend = true;
        attention_backend = chunk_aware_attention
            ? "mlx_chunked_compact_mlx_chunk_aware"
            : "mlx_chunked_compact_mlx";
      } else if (attention_mode == "expanded_kv") {
        double append_ms = 0.0;
        double attention_ms = 0.0;
        mlx_expanded_kv_cache_append_preallocated(
            kv_cache.layers[layer],
            k_values,
            v_values,
            active_rope_config().num_attention_heads,
            active_rope_config().num_key_value_heads,
            active_rope_config().head_dim,
            static_cast<std::size_t>(kv_cache.max_seq),
            &append_ms);
        attention_values = mlx_expanded_kv_cache_attention_preallocated(
            q_values,
            kv_cache.layers[layer],
            active_rope_config().num_attention_heads,
            active_rope_config().head_dim,
            &attention_ms);
        attention_timing.kv_append_ms += append_ms;
        attention_timing.value_mix_ms += attention_ms;
        attention_timing_populated_by_backend = true;
        attention_backend = "mlx_expanded_kv";
      } else if (attention_mode == "batched") {
        attention_values = mlx_all_head_batched_attention_values_probe(
            q_values,
            kv_cache.layers[layer],
            k_values,
            v_values);
        attention_backend = "mlx_batched";
      } else {
        attention_values = mlx_single_token_attention_values_probe(
            q_values,
            kv_cache.layers[layer],
            k_values,
            v_values);
        attention_backend = "mlx";
      }
      if (!attention_timing_populated_by_backend) {
        attention_timing.value_mix_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - attention_start).count();
      }
    } catch (const std::exception& e) {
      attention_fallback_reason = e.what();
    } catch (...) {
      attention_fallback_reason = "unknown_exception";
    }
  }
  if (attention_values.empty()) {
    if (!keep_cpu_kv_mirror() &&
        (experimental_mlx_attention_mode() == "chunked_expanded_kv" ||
         experimental_mlx_attention_mode() == "chunked_compact_mlx" ||
         experimental_mlx_attention_mode() == "expanded_kv")) {
      throw std::runtime_error(
          "CPU attention fallback requires RUSTY_KEEP_CPU_KV_MIRROR=1");
    }
    attention_values = cached_single_token_attention_values(
        q_values,
        kv_cache.layers[layer],
        k_values,
        v_values,
        &attention_timing);
    if (attention_backend == "mlx") {
      attention_backend = "cpu";
    }
  }
  auto kv_append_start = std::chrono::steady_clock::now();
  append_native_kv(&kv_cache, layer, k_values, v_values);
  if (timing != nullptr) {
    add_elapsed_ms(attention_timing.kv_append_ms, kv_append_start);
    timing->kv_append_ms += attention_timing.kv_append_ms;
    timing->attention_score_ms += attention_timing.score_ms;
    timing->attention_softmax_ms += attention_timing.softmax_ms;
    timing->attention_value_mix_ms += attention_timing.value_mix_ms;
    timing->attention_kv_view_assembly_ms += attention_detail_timing.kv_view_assembly_ms;
    timing->attention_score_matmul_ms += attention_detail_timing.score_matmul_ms;
    timing->attention_softmax_detail_ms += attention_detail_timing.softmax_ms;
    timing->attention_value_mix_matmul_ms += attention_detail_timing.value_mix_matmul_ms;
    timing->attention_reshape_flatten_ms += attention_detail_timing.reshape_flatten_ms;
    timing->attention_eval_sync_ms += attention_detail_timing.eval_sync_ms;
    timing->per_layer_attention_ms.push_back(attention_timing.total_attention_ms());
    timing->per_layer_attention_backends.push_back(attention_backend);
    timing->per_layer_attention_fallback_reasons.push_back(attention_fallback_reason);
  }

  auto o_start = std::chrono::steady_clock::now();
  MlxProjectionTiming o_timing;
  MlxQuantizedLinearStepScope o_scope("o_proj");
  LocalQuantizedMlxArrays o_local;
  mlx::core::array attention_array = mlx_array_from_vector(attention_values);
  auto o_result = make_quantized_matmul_from_mlx_input(
      groups.o_proj,
      attention_array,
      o_local,
      attention_values.size());
  o_result = apply_lora_delta_if_present(
      o_result,
      attention_array,
      lora_layer == nullptr ? nullptr : &lora_layer->o_proj);
  auto attention_residual = o_result + input_value.mlx;
  attention_residual.eval();
  mlx::core::synchronize();
  if (timing != nullptr) {
    add_elapsed_ms(timing->o_projection_ms, o_start);
    timing->mlx_resident_layer_block_applied = true;
  }

  auto post_norm = rmsnorm_mlx_array(groups.post_attention_norm, attention_residual, eps);
  auto mlp_start = std::chrono::steady_clock::now();
  MlxResidentMlpChainTiming mlp_timing;
  mlx::core::array mlp_residual = quantized_linear_mlp_chain_residual_mlx(
      groups.gate_proj,
      groups.up_proj,
      groups.down_proj,
      post_norm,
      attention_residual,
      &mlp_timing,
      lora_layer == nullptr ? nullptr : &lora_layer->gate_proj,
      lora_layer == nullptr ? nullptr : &lora_layer->up_proj,
      lora_layer == nullptr ? nullptr : &lora_layer->down_proj);
  if (timing != nullptr) {
    timing->gate_up_projection_ms += mlp_timing.setup_ms + mlp_timing.gate_up_eval_ms;
    timing->gate_projection_ms += (mlp_timing.setup_ms + mlp_timing.gate_up_eval_ms) / 2.0;
    timing->up_projection_ms += (mlp_timing.setup_ms + mlp_timing.gate_up_eval_ms) / 2.0;
    timing->gate_projection_setup_ms += mlp_timing.setup_ms / 2.0;
    timing->up_projection_setup_ms += mlp_timing.setup_ms / 2.0;
    timing->gate_projection_eval_ms += mlp_timing.gate_up_eval_ms / 2.0;
    timing->up_projection_eval_ms += mlp_timing.gate_up_eval_ms / 2.0;
    timing->gate_up_activation_ms += mlp_timing.activation_eval_ms;
    timing->down_projection_ms += mlp_timing.down_eval_ms + mlp_timing.readback_ms;
    timing->gate_backend = "metal";
    timing->up_backend = "metal";
    timing->activation_backend = "metal";
    timing->gate_up_fallback_used = false;
    timing->mlx_resident_mlp_chain_applied = true;
    timing->mlx_resident_mlp_chain_fallback_used = false;
    (void)mlp_start;
  }
  return ResidentDecodeValue(mlp_residual);
}
#endif

std::vector<float> layer_mlp_residual_from_input(
    const std::string& model_dir_str,
    int layer,
    const std::vector<float>& input_values,
    double eps) {
  TensorGroupRecord post_attention_norm;
  TensorGroupRecord gate_proj;
  TensorGroupRecord up_proj;
  TensorGroupRecord down_proj;
  const std::string prefix = "model.layers." + std::to_string(layer) + ".";
  const std::string groups[] = {
      prefix + "post_attention_layernorm",
      prefix + "mlp.gate_proj",
      prefix + "mlp.up_proj",
      prefix + "mlp.down_proj",
  };
  TensorGroupRecord* records[] = {
      &post_attention_norm,
      &gate_proj,
      &up_proj,
      &down_proj,
  };
  for (std::size_t i = 0; i < 4; ++i) {
    auto spec = group_load_spec_for(groups[i]);
    if (!spec || !load_tensor_group_record(model_dir_str, *spec, *records[i])) {
      throw std::runtime_error("failed to load MLP tensor group");
    }
  }

  std::vector<float> norm_values = rmsnorm_values(post_attention_norm, input_values, eps);
  std::vector<float> gate_values = quantized_linear_vector_values(gate_proj, norm_values);
  std::vector<float> up_values = quantized_linear_vector_values(up_proj, norm_values);
  if (gate_values.size() != up_values.size()) {
    throw std::runtime_error("gate/up projection sizes differ");
  }
  std::vector<float> activated_values(gate_values.size(), 0.0f);
  for (std::size_t i = 0; i < activated_values.size(); ++i) {
    const float gate = gate_values[i];
    const float silu = gate / (1.0f + std::exp(-gate));
    activated_values[i] = silu * up_values[i];
  }
  std::vector<float> down_values = quantized_linear_vector_values(down_proj, activated_values);
  if (down_values.size() != input_values.size()) {
    throw std::runtime_error("MLP down_proj output does not match residual width");
  }
  std::vector<float> residual_values(down_values.size(), 0.0f);
  for (std::size_t i = 0; i < residual_values.size(); ++i) {
    residual_values[i] = input_values[i] + down_values[i];
  }
  return residual_values;
}

std::vector<float> layer_mlp_residual_resident_from_input(
    const ResidentLayerGroups& groups,
    const std::vector<float>& input_values,
    double eps,
    bool use_optimized_linear = false,
    bool use_layout_cached_linear = false,
    bool use_mlp_pair_optimized = false,
    bool use_gate_up_full_block_optimized = false,
    bool use_down_full_block_optimized = false,
    bool use_mlx_resident_mlp_chain = false,
    ResidentDecodeResult* timing = nullptr) {
  std::vector<float> norm_values = rmsnorm_values(groups.post_attention_norm, input_values, eps);
  const bool metal_mlp_chain_available =
      use_mlx_resident_mlp_chain &&
      use_optimized_linear &&
      use_layout_cached_linear &&
      mlx_quantized_linear_available_flag() &&
      !disable_mlx_quantized_linear() &&
      !disable_mlx_resident_mlp_chain();
  if (metal_mlp_chain_available) {
    auto chain_start = std::chrono::steady_clock::now();
    MlxResidentMlpChainTiming chain_timing;
    MlxQuantizedLinearStepScope mlp_chain_scope("resident_mlp_chain");
    try {
      std::vector<float> down_values = quantized_linear_mlp_chain_down_values_mlx(
          groups.gate_proj,
          groups.up_proj,
          groups.down_proj,
          norm_values,
          &chain_timing);
      if (down_values.size() != input_values.size()) {
        throw std::runtime_error("resident MLP chain down_proj output does not match residual width");
      }
      if (timing != nullptr) {
        const double chain_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - chain_start).count();
        timing->gate_up_projection_ms += chain_timing.setup_ms + chain_timing.gate_up_eval_ms;
        timing->gate_projection_ms += (chain_timing.setup_ms + chain_timing.gate_up_eval_ms) / 2.0;
        timing->up_projection_ms += (chain_timing.setup_ms + chain_timing.gate_up_eval_ms) / 2.0;
        timing->gate_projection_setup_ms += chain_timing.setup_ms / 2.0;
        timing->up_projection_setup_ms += chain_timing.setup_ms / 2.0;
        timing->gate_projection_eval_ms += chain_timing.gate_up_eval_ms / 2.0;
        timing->up_projection_eval_ms += chain_timing.gate_up_eval_ms / 2.0;
        timing->gate_projection_readback_ms += 0.0;
        timing->up_projection_readback_ms += 0.0;
        timing->gate_up_activation_ms += chain_timing.activation_eval_ms;
        timing->down_projection_ms += chain_timing.down_eval_ms + chain_timing.readback_ms;
        timing->gate_backend = "metal";
        timing->up_backend = "metal";
        timing->activation_backend = "metal";
        timing->gate_up_fallback_used = false;
        timing->mlx_resident_mlp_chain_applied = true;
        timing->mlx_resident_mlp_chain_fallback_used = false;
        (void)chain_ms;
      }
      std::vector<float> residual_values(down_values.size(), 0.0f);
      for (std::size_t i = 0; i < residual_values.size(); ++i) {
        residual_values[i] = input_values[i] + down_values[i];
      }
      return residual_values;
    } catch (...) {
      record_mlx_quantized_linear_fallback_step();
      if (timing != nullptr) {
        timing->mlx_resident_mlp_chain_applied = false;
        timing->mlx_resident_mlp_chain_fallback_used = true;
        timing->gate_up_fallback_used = true;
      }
    }
  }
  std::vector<float> gate_values;
  std::vector<float> up_values;
  auto gate_up_start = std::chrono::steady_clock::now();
  const bool metal_gate_up_available =
      use_optimized_linear &&
      use_layout_cached_linear &&
      mlx_quantized_linear_available_flag() &&
      !disable_mlx_quantized_linear();
  if (metal_gate_up_available) {
    auto metal_pair_start = std::chrono::steady_clock::now();
    MlxProjectionTiming gate_mlx_timing;
    MlxProjectionTiming up_mlx_timing;
    MlxQuantizedLinearStepScope gate_up_metal_scope("gate_up");
    try {
      auto paired = quantized_linear_pair_values_mlx(
          groups.gate_proj,
          groups.up_proj,
          norm_values,
          &gate_mlx_timing,
          &up_mlx_timing);
      gate_values = std::move(paired.first);
      up_values = std::move(paired.second);
    } catch (...) {
      record_mlx_quantized_linear_fallback_step();
      {
        MlxQuantizedLinearStepScope gate_scope("gate_proj");
        gate_values = quantized_linear_vector_values_layout_cached(groups.gate_proj, norm_values);
      }
      {
        MlxQuantizedLinearStepScope up_scope("up_proj");
        up_values = quantized_linear_vector_values_layout_cached(groups.up_proj, norm_values);
      }
    }
    if (timing != nullptr) {
      const double combined_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - metal_pair_start).count();
      timing->gate_projection_ms += combined_ms / 2.0;
      timing->up_projection_ms += combined_ms / 2.0;
      timing->gate_projection_setup_ms += gate_mlx_timing.setup_ms;
      timing->gate_projection_eval_ms += gate_mlx_timing.eval_ms;
      timing->gate_projection_readback_ms += gate_mlx_timing.readback_ms;
      timing->up_projection_setup_ms += up_mlx_timing.setup_ms;
      timing->up_projection_eval_ms += up_mlx_timing.eval_ms;
      timing->up_projection_readback_ms += up_mlx_timing.readback_ms;
    }
  } else if (use_optimized_linear && use_layout_cached_linear && use_mlp_pair_optimized) {
    MlxQuantizedLinearStepScope gate_up_scope("gate_up");
    auto paired = use_gate_up_full_block_optimized
        ? quantized_linear_pair_values_layout_cached_full_blocks(
              groups.gate_proj,
              groups.up_proj,
              norm_values)
        : quantized_linear_pair_values_layout_cached(
              groups.gate_proj,
              groups.up_proj,
              norm_values);
    gate_values = std::move(paired.first);
    up_values = std::move(paired.second);
  } else {
    gate_values =
        quantized_linear_vector_values_selected(
            groups.gate_proj, norm_values, use_optimized_linear, use_layout_cached_linear);
    up_values =
        quantized_linear_vector_values_selected(
            groups.up_proj, norm_values, use_optimized_linear, use_layout_cached_linear);
  }
  if (timing != nullptr) {
    add_elapsed_ms(timing->gate_up_projection_ms, gate_up_start);
    const bool gate_fell_back = std::find(
        mlx_quantized_linear_fallback_steps.begin(),
        mlx_quantized_linear_fallback_steps.end(),
        "gate_proj") != mlx_quantized_linear_fallback_steps.end();
    const bool up_fell_back = std::find(
        mlx_quantized_linear_fallback_steps.begin(),
        mlx_quantized_linear_fallback_steps.end(),
        "up_proj") != mlx_quantized_linear_fallback_steps.end();
    const bool gate_up_pair_fell_back = std::find(
        mlx_quantized_linear_fallback_steps.begin(),
        mlx_quantized_linear_fallback_steps.end(),
        "gate_up") != mlx_quantized_linear_fallback_steps.end();
    timing->gate_backend = metal_gate_up_available && !gate_fell_back && !gate_up_pair_fell_back ? "metal" : "cpu";
    timing->up_backend = metal_gate_up_available && !up_fell_back && !gate_up_pair_fell_back ? "metal" : "cpu";
    timing->activation_backend = "cpu";
    timing->gate_up_fallback_used = gate_fell_back || up_fell_back || gate_up_pair_fell_back;
  }
  if (gate_values.size() != up_values.size()) {
    throw std::runtime_error("resident gate/up projection sizes differ");
  }
  std::vector<float> activated_values(gate_values.size(), 0.0f);
  auto activation_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < activated_values.size(); ++i) {
    const float gate = gate_values[i];
    const float silu = gate / (1.0f + std::exp(-gate));
    activated_values[i] = silu * up_values[i];
  }
  if (timing != nullptr) {
    add_elapsed_ms(timing->gate_up_activation_ms, activation_start);
  }
  auto down_start = std::chrono::steady_clock::now();
  MlxQuantizedLinearStepScope down_scope("down");
  std::vector<float> down_values =
      (use_optimized_linear && use_layout_cached_linear && use_down_full_block_optimized)
          ? quantized_linear_vector_values_layout_cached_full_blocks(groups.down_proj, activated_values)
          : quantized_linear_vector_values_selected(
                groups.down_proj, activated_values, use_optimized_linear, use_layout_cached_linear);
  if (timing != nullptr) {
    add_elapsed_ms(timing->down_projection_ms, down_start);
  }
  if (down_values.size() != input_values.size()) {
    throw std::runtime_error("resident MLP down_proj output does not match residual width");
  }
  std::vector<float> residual_values(down_values.size(), 0.0f);
  for (std::size_t i = 0; i < residual_values.size(); ++i) {
    residual_values[i] = input_values[i] + down_values[i];
  }
  return residual_values;
}

ResidentDecodeResult resident_incremental_decode_once(
    const std::string& model_dir_str,
    std::uint64_t layers,
    double eps,
    const ResidentModelLayers& resident,
    NativeSessionKvCache kv_cache,
    std::uint64_t decode_token_id,
    const TensorGroupRecord& final_norm,
    const TensorGroupRecord& embedding,
    bool use_optimized_linear,
    bool use_layout_cached_linear = false,
    bool use_mlp_pair_optimized = false,
    bool use_logits_top1_optimized = false,
    bool use_gate_up_full_block_optimized = false,
    bool use_down_full_block_optimized = false) {
  ResidentDecodeResult result;
  result.positions_before = kv_cache.positions_stored;
  auto start = std::chrono::steady_clock::now();
  std::vector<float> current_values =
      layer0_embedding_values(model_dir_str, static_cast<std::size_t>(decode_token_id));
  for (std::uint64_t layer = 0; layer < layers; ++layer) {
    const ResidentLayerGroups& groups = resident.layers[static_cast<std::size_t>(layer)];
    std::vector<float> attention_values =
        layer_attention_residual_resident_incremental_from_input(
            groups,
            static_cast<std::size_t>(layer),
            current_values,
            eps,
            kv_cache,
            use_optimized_linear,
            use_layout_cached_linear,
            false,
            &result);
    current_values =
        layer_mlp_residual_resident_from_input(
            groups,
            attention_values,
            eps,
            use_optimized_linear,
            use_layout_cached_linear,
            use_mlp_pair_optimized,
            use_gate_up_full_block_optimized,
            use_down_full_block_optimized,
            false,
            &result);
  }
  auto final_norm_start = std::chrono::steady_clock::now();
  MlxQuantizedLinearStepScope final_norm_scope("final_norm");
  std::vector<float> final_norm_values = rmsnorm_values(final_norm, current_values, eps);
  add_elapsed_ms(result.final_norm_ms, final_norm_start);
  result.final_norm_checksum = vector_checksum(final_norm_values);
  auto logits_start = std::chrono::steady_clock::now();
  MlxQuantizedLinearStepScope logits_scope("logits/top1");
  if (use_optimized_linear && use_layout_cached_linear && use_logits_top1_optimized) {
    result.logits_len = embedding.quantized_layout_cached
        ? embedding.quantized_output_len
        : 0;
    auto top = quantized_linear_top1_layout_cached(embedding, final_norm_values);
    result.top_token_id = top.first;
    result.top_token_score = top.second;
  } else {
    result.logits = quantized_linear_vector_values_selected(
        embedding,
        final_norm_values,
        use_optimized_linear,
        use_layout_cached_linear);
    result.logits_len = result.logits.size();
    std::vector<std::pair<std::uint64_t, float>> top = top_logits(result.logits, 1);
    if (!top.empty()) {
      result.top_token_id = top[0].first;
      result.top_token_score = top[0].second;
    }
  }
  add_elapsed_ms(result.logits_projection_ms, logits_start);
  auto end = std::chrono::steady_clock::now();
  result.timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
  result.positions_after = kv_cache.positions_stored;
  update_largest_arithmetic_bucket(result);
  return result;
}

std::pair<std::uint64_t, float> greedy_next_token_for_input(
    const std::string& model_dir_str,
    std::uint64_t token_id,
    std::uint64_t layers,
    double eps,
    const TensorGroupRecord& final_norm,
    const TensorGroupRecord& embedding,
    double& final_norm_checksum,
    std::size_t& logits_len) {
  std::vector<float> current_values =
      layer0_embedding_values(model_dir_str, static_cast<std::size_t>(token_id));
  for (std::uint64_t layer = 0; layer < layers; ++layer) {
    std::vector<float> attention_values =
        layer_attention_residual_from_input(model_dir_str, static_cast<int>(layer), current_values, eps);
    current_values =
        layer_mlp_residual_from_input(model_dir_str, static_cast<int>(layer), attention_values, eps);
  }

  std::vector<float> final_norm_values = rmsnorm_values(final_norm, current_values, eps);
  final_norm_checksum = vector_checksum(final_norm_values);
  std::vector<float> logits = quantized_linear_vector_values(embedding, final_norm_values);
  logits_len = logits.size();
  std::vector<std::pair<std::uint64_t, float>> top = top_logits(logits, 1);
  if (top.empty()) {
    throw std::runtime_error("no logits produced");
  }
  return top[0];
}

std::vector<GroupLoadSpec> layer_group_specs_for(int layer) {
  const std::string prefix = "model.layers." + std::to_string(layer) + ".";
  return {
      GroupLoadSpec{prefix + "self_attn.q_proj", true, {prefix + "self_attn.q_proj.weight", prefix + "self_attn.q_proj.scales", prefix + "self_attn.q_proj.biases"}},
      GroupLoadSpec{prefix + "self_attn.k_proj", true, {prefix + "self_attn.k_proj.weight", prefix + "self_attn.k_proj.scales", prefix + "self_attn.k_proj.biases"}},
      GroupLoadSpec{prefix + "self_attn.v_proj", true, {prefix + "self_attn.v_proj.weight", prefix + "self_attn.v_proj.scales", prefix + "self_attn.v_proj.biases"}},
      GroupLoadSpec{prefix + "self_attn.o_proj", true, {prefix + "self_attn.o_proj.weight", prefix + "self_attn.o_proj.scales", prefix + "self_attn.o_proj.biases"}},
      GroupLoadSpec{prefix + "self_attn.q_norm", false, {prefix + "self_attn.q_norm.weight"}},
      GroupLoadSpec{prefix + "self_attn.k_norm", false, {prefix + "self_attn.k_norm.weight"}},
      GroupLoadSpec{prefix + "mlp.gate_proj", true, {prefix + "mlp.gate_proj.weight", prefix + "mlp.gate_proj.scales", prefix + "mlp.gate_proj.biases"}},
      GroupLoadSpec{prefix + "mlp.up_proj", true, {prefix + "mlp.up_proj.weight", prefix + "mlp.up_proj.scales", prefix + "mlp.up_proj.biases"}},
      GroupLoadSpec{prefix + "mlp.down_proj", true, {prefix + "mlp.down_proj.weight", prefix + "mlp.down_proj.scales", prefix + "mlp.down_proj.biases"}},
      GroupLoadSpec{prefix + "input_layernorm", false, {prefix + "input_layernorm.weight"}},
      GroupLoadSpec{prefix + "post_attention_layernorm", false, {prefix + "post_attention_layernorm.weight"}},
  };
}

std::string resident_model_key(const std::string& model_dir_str) {
  return model_dir_str;
}

bool load_resident_model_layers(
    const std::string& model_dir_str,
    std::uint64_t layers,
    ResidentModelLayers& resident);

bool load_resident_model_layers(
    const std::string& model_dir_str,
    std::uint64_t layers,
    ResidentModelLayers& resident) {
  resident.model_dir = model_dir_str;
  resident.layers.clear();
  resident.layers.resize(static_cast<std::size_t>(layers));
  resident.total_groups = 0;
  resident.total_byte_size = 0;
  resident.tensor_group_load_count = 0;
  resident.load_timing = ResidentLoadTiming{};

  for (std::uint64_t layer = 0; layer < layers; ++layer) {
    ResidentLayerGroups& out = resident.layers[static_cast<std::size_t>(layer)];
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    const std::string groups[] = {
        prefix + "input_layernorm",
        prefix + "self_attn.q_proj",
        prefix + "self_attn.k_proj",
        prefix + "self_attn.v_proj",
        prefix + "self_attn.o_proj",
        prefix + "self_attn.q_norm",
        prefix + "self_attn.k_norm",
        prefix + "post_attention_layernorm",
        prefix + "mlp.gate_proj",
        prefix + "mlp.up_proj",
        prefix + "mlp.down_proj",
    };
    TensorGroupRecord* records[] = {
        &out.input_norm,
        &out.q_proj,
        &out.k_proj,
        &out.v_proj,
        &out.o_proj,
        &out.q_norm,
        &out.k_norm,
        &out.post_attention_norm,
        &out.gate_proj,
        &out.up_proj,
        &out.down_proj,
    };
    for (std::size_t i = 0; i < 11; ++i) {
      auto spec = group_load_spec_for(groups[i]);
      if (!spec || !load_tensor_group_record(model_dir_str, *spec, *records[i], &resident.load_timing)) {
        return false;
      }
      out.total_byte_size += records[i]->total_byte_size;
      resident.total_byte_size += records[i]->total_byte_size;
      resident.total_groups += 1;
      resident.tensor_group_load_count += 1;
    }
  }
  return true;
}

ResidentModelLayers& resident_layers_for(
    const std::string& model_dir_str,
    std::uint64_t layers,
    bool& loaded_now) {
  const std::string key = resident_model_key(model_dir_str);
  std::lock_guard<std::mutex> lock(resident_model_mutex());
  auto existing = resident_model_table().find(key);
  if (existing != resident_model_table().end() &&
      existing->second.layers.size() == static_cast<std::size_t>(layers)) {
    loaded_now = false;
    return existing->second;
  }
  ResidentModelLayers resident;
  if (!load_resident_model_layers(model_dir_str, layers, resident)) {
    throw std::runtime_error("failed to load resident model layer groups");
  }
  loaded_now = true;
  auto inserted = resident_model_table().insert_or_assign(key, std::move(resident));
  return inserted.first->second;
}

std::uint64_t read_u64_le(const std::uint8_t* data) {
  return static_cast<std::uint64_t>(data[0]) |
         (static_cast<std::uint64_t>(data[1]) << 8) |
         (static_cast<std::uint64_t>(data[2]) << 16) |
         (static_cast<std::uint64_t>(data[3]) << 24) |
         (static_cast<std::uint64_t>(data[4]) << 32) |
         (static_cast<std::uint64_t>(data[5]) << 40) |
         (static_cast<std::uint64_t>(data[6]) << 48) |
         (static_cast<std::uint64_t>(data[7]) << 56);
}

bool run_tiny_array_probe_once(bool use_cpu, RuntimeProbeResult& result) {
  std::string stage = "array construction";
  try {
    if (use_cpu) {
      mlx::core::set_default_device(
          mlx::core::Device(mlx::core::Device::cpu, 0));
    }

    auto handle = next_test_array_handle();
    {
      std::lock_guard<std::mutex> lock(test_array_mutex());
      test_array_table().emplace(handle, TestArrayRecord{make_test_array()});
    }

    stage = "eval";
    {
      std::lock_guard<std::mutex> lock(test_array_mutex());
      auto it = test_array_table().find(handle);
      if (it == test_array_table().end()) {
        throw std::runtime_error("test array handle disappeared before eval");
      }
      it->second.value.eval();
    }

    stage = "scalar extraction";
    {
      std::lock_guard<std::mutex> lock(test_array_mutex());
      auto it = test_array_table().find(handle);
      if (it == test_array_table().end()) {
        throw std::runtime_error(
            "test array handle disappeared before scalar extraction");
      }
      (void)it->second.value.item<float>();
    }

    stage = "device sync";
    mlx::core::synchronize();

    stage = "free handle";
    {
      std::lock_guard<std::mutex> lock(test_array_mutex());
      auto erased = test_array_table().erase(handle);
      if (erased != 1) {
        throw std::runtime_error("test array handle was not freed");
      }
    }

    return true;
  } catch (const std::exception& err) {
    if (use_cpu) {
      result.cpu_failure_stage = stage;
      result.cpu_exception = err.what();
    } else {
      result.primary_failure_stage = stage;
      result.primary_exception = err.what();
    }
    return false;
  } catch (...) {
    if (use_cpu) {
      result.cpu_failure_stage = stage;
      result.cpu_exception = "unknown exception";
    } else {
      result.primary_failure_stage = stage;
      result.primary_exception = "unknown exception";
    }
    return false;
  }
}

RuntimeProbeResult run_runtime_probe() {
  RuntimeProbeResult result;
  result.primary_success = run_tiny_array_probe_once(false, result);
  if (!result.primary_success) {
    result.cpu_attempted = true;
    result.cpu_success = run_tiny_array_probe_once(true, result);
  }
  return result;
}
#endif

}  // namespace

extern "C" {

const char* rusty_mlx_shim_version() {
  return "rusty-mlx-shim/0.1";
}

int rusty_mlx_shim_probe() {
  return 1;
}

int rusty_mlx_link_probe() {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    const auto& device = mlx::core::default_device();
    return mlx::core::is_available(device) ? 1 : 1;
  } catch (...) {
    return 0;
  }
#else
  return 0;
#endif
}

unsigned long long rusty_mlx_create_test_array() {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    std::lock_guard<std::mutex> lock(test_array_mutex());
    auto handle = next_test_array_handle();
    test_array_table().emplace(handle, TestArrayRecord{make_test_array()});
    return handle;
  } catch (...) {
    return 0;
  }
#else
  return 0;
#endif
}

double rusty_mlx_test_array_sum(unsigned long long handle) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    std::lock_guard<std::mutex> lock(test_array_mutex());
    auto it = test_array_table().find(handle);
    if (it == test_array_table().end()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(it->second.value.item<float>());
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
#else
  (void)handle;
  return std::numeric_limits<double>::quiet_NaN();
#endif
}

int rusty_mlx_free_test_array(unsigned long long handle) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    std::lock_guard<std::mutex> lock(test_array_mutex());
    auto erased = test_array_table().erase(handle);
    return erased == 1 ? 1 : 0;
  } catch (...) {
    return 0;
  }
#else
  (void)handle;
  return 0;
#endif
}

unsigned long long rusty_mlx_create_token_array(
    const unsigned long long* tokens,
    unsigned long long length) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    auto handle = next_test_array_handle();
    auto value = make_token_array(tokens, static_cast<std::size_t>(length));
    std::lock_guard<std::mutex> lock(token_array_mutex());
    token_array_table().emplace(handle, TokenArrayRecord{std::move(value)});
    return handle;
  } catch (...) {
    return 0;
  }
#else
  (void)tokens;
  (void)length;
  return 0;
#endif
}

const char* rusty_mlx_token_array_info_json(unsigned long long handle) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    std::lock_guard<std::mutex> lock(token_array_mutex());
    auto it = token_array_table().find(handle);
    if (it == token_array_table().end()) {
      output = "{\"ok\":false,\"error\":\"unknown_handle\"}";
      return output.c_str();
    }
    const auto& value = it->second.value;
    std::ostringstream json;
    json << "{"
         << "\"ok\":true,"
         << "\"dtype\":\"" << json_escape(mlx::core::dtype_to_string(value.dtype())) << "\","
         << "\"ndim\":" << value.ndim() << ","
         << "\"size\":" << value.size() << ","
         << "\"shape\":" << shape_to_json(value.shape()) << ","
         << "\"source_group\":";
    auto source_group = it->second.source_group;
    if (source_group.empty()) {
      json << "null";
    } else {
      json << "\"" << json_escape(source_group) << "\"";
    }
    json << "}";
    output = json.str();
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)handle;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

int rusty_mlx_free_token_array(unsigned long long handle) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    std::lock_guard<std::mutex> lock(token_array_mutex());
    auto erased = token_array_table().erase(handle);
    return erased == 1 ? 1 : 0;
  } catch (...) {
    return 0;
  }
#else
  (void)handle;
  return 0;
#endif
}

unsigned long long rusty_mlx_mock_forward(
    unsigned long long session_handle,
    unsigned long long token_array_handle) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    (void)session_handle;
    std::lock_guard<std::mutex> lock(token_array_mutex());
    auto it = token_array_table().find(token_array_handle);
    if (it == token_array_table().end()) {
      return 0;
    }
    auto logits_handle = next_array_handle();
    auto logits = make_logits_array(it->second.value.size());
    {
      std::lock_guard<std::mutex> array_lock(array_mutex());
      array_table().emplace(logits_handle, ArrayRecord{std::move(logits)});
    }
    return logits_handle;
  } catch (...) {
    return 0;
  }
#else
  (void)session_handle;
  (void)token_array_handle;
  return 0;
#endif
}

unsigned long long rusty_mlx_mock_sample(unsigned long long logits_handle) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    std::lock_guard<std::mutex> lock(array_mutex());
    auto it = array_table().find(logits_handle);
    if (it == array_table().end()) {
      return 0;
    }
    auto& value = it->second.value;
    if (value.size() == 0) {
      return 0;
    }
    value.eval();
    const auto* data = value.data<float>();
    std::size_t best_index = 0;
    float best_value = data[0];
    for (std::size_t i = 1; i < value.size(); ++i) {
      if (data[i] > best_value) {
        best_value = data[i];
        best_index = i;
      }
    }
    return static_cast<unsigned long long>(best_index + 1);
  } catch (...) {
    return 0;
  }
#else
  (void)logits_handle;
  return 0;
#endif
}

const char* rusty_mlx_array_info_json(unsigned long long handle) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    std::lock_guard<std::mutex> lock(array_mutex());
    auto it = array_table().find(handle);
    if (it == array_table().end()) {
      output = "{\"ok\":false,\"error\":\"unknown_handle\"}";
      return output.c_str();
    }
    const auto& value = it->second.value;
    std::ostringstream json;
    json << "{"
         << "\"ok\":true,"
         << "\"dtype\":\"" << json_escape(mlx::core::dtype_to_string(value.dtype())) << "\","
         << "\"ndim\":" << value.ndim() << ","
         << "\"size\":" << value.size() << ","
         << "\"shape\":" << shape_to_json(value.shape()) << ","
         << "\"source_group\":";
    auto source_group = it->second.source_group;
    if (source_group.empty()) {
      json << "null";
    } else {
      json << "\"" << json_escape(source_group) << "\"";
    }
    json << "}";
    output = json.str();
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)handle;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

int rusty_mlx_free_array(unsigned long long handle) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    std::lock_guard<std::mutex> lock(array_mutex());
    auto it = array_table().find(handle);
    if (it == array_table().end()) {
      return 0;
    }
    // Remove from the live handle table synchronously, but keep the MLX array
    // object alive until bridge shutdown. Some Metal-backed MLX destructors can
    // block in runtime probe contexts; live handle counts must still reflect
    // that JavaScript/Rust can no longer access the array.
    retired_array_records().push_back(std::move(it->second));
    array_table().erase(it);
    return 1;
  } catch (...) {
    return 0;
  }
#else
  (void)handle;
  return 0;
#endif
}

const char* rusty_mlx_handle_counts_json() {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    std::lock_guard<std::mutex> token_lock(token_array_mutex());
    std::lock_guard<std::mutex> array_lock(array_mutex());
    std::ostringstream json;
    json << "{"
         << "\"ok\":true,"
         << "\"token_arrays\":" << token_array_table().size() << ","
         << "\"arrays\":" << array_table().size() << ","
         << "\"embedding_groups\":" << embedding_group_table().size() << ","
         << "\"tensor_groups\":" << tensor_group_table().size() << ","
         << "\"layer_groups\":" << layer_group_table().size()
         << "}";
    output = json.str();
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

bool load_tensor_group_record(
    const std::string& model_dir_str,
    const GroupLoadSpec& spec,
    TensorGroupRecord& record,
    ResidentLoadTiming* timing) {
  auto add_timing = [](double& bucket, const std::chrono::steady_clock::time_point& start) {
    bucket += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  };
  auto index_start = std::chrono::steady_clock::now();
  std::string index_path = file_exists(model_dir_str)
      ? (model_dir_str.find(".index.json") != std::string::npos ? model_dir_str
                                                                : join_path(model_dir_str, "model.safetensors.index.json"))
      : join_path(model_dir_str, "model.safetensors.index.json");
  if (!file_exists(index_path)) {
    return false;
  }

  std::string index_json;
  if (!read_file_to_string_cached(index_path, index_json)) {
    return false;
  }
  std::string shard_name = index_shard_for_tensor(index_json, spec.tensor_names.front());
  if (shard_name.empty()) {
    return false;
  }
  std::string shard_path = join_path(parent_dir(index_path), shard_name);
  if (!file_exists(shard_path)) {
    return false;
  }
  if (timing != nullptr) {
    add_timing(timing->safetensor_index_lookup_ms, index_start);
  }

  auto metadata_start = std::chrono::steady_clock::now();
  std::uint64_t header_len = 0;
  std::string header_json;
  if (!read_safetensor_header_cached(shard_path, header_len, header_json)) {
    return false;
  }

  record.group = spec.group;
  record.source_dir = model_dir_str;
  record.index_path = index_path;
  record.loaded = true;
  record.quantized_group = spec.quantized_group;
  record.total_byte_size = 0;
  record.tensors.clear();
  record.quantized_layout_cached = false;
  record.quantized_rows = 0;
  record.quantized_packed_cols = 0;
  record.quantized_scale_cols = 0;
  record.quantized_logical_width = 0;
  record.quantized_block_size = 8;
  record.quantized_output_len = 0;
  if (timing != nullptr) {
    add_timing(timing->tensor_group_metadata_construction_ms, metadata_start);
  }

  for (const auto& tensor_name : spec.tensor_names) {
    auto tensor_metadata_start = std::chrono::steady_clock::now();
    auto object = tensor_object_for_key(header_json, tensor_name);
    if (object.empty()) {
      return false;
    }
    GroupTensorRecord tensor_record;
    if (string_ends_with(tensor_name, ".weight")) {
      tensor_record.kind = "weight";
    } else if (string_ends_with(tensor_name, ".scales")) {
      tensor_record.kind = "scales";
    } else if (string_ends_with(tensor_name, ".biases")) {
      tensor_record.kind = "biases";
    }
    if (!extract_json_string_value(object, "dtype", tensor_record.dtype)) {
      return false;
    }
    if (!extract_json_array_u64(object, "shape", tensor_record.shape)) {
      return false;
    }
    std::vector<std::uint64_t> offsets;
    if (!extract_json_array_u64(object, "data_offsets", offsets) || offsets.size() != 2) {
      return false;
    }
    tensor_record.byte_start = offsets[0];
    tensor_record.byte_end = offsets[1];
    tensor_record.source_file = shard_name;
    std::uint64_t byte_count = tensor_record.byte_end - tensor_record.byte_start;
    if (byte_count == 0) {
      return false;
    }
    if (timing != nullptr) {
      add_timing(timing->tensor_group_metadata_construction_ms, tensor_metadata_start);
    }
    auto read_start = std::chrono::steady_clock::now();
    if (timing != nullptr) {
      auto mapped = mapped_file_for_path(shard_path);
      const std::uint64_t payload_offset = 8 + header_len + tensor_record.byte_start;
      if (payload_offset + byte_count > mapped->size) {
        return false;
      }
      tensor_record.mapped_file = std::move(mapped);
      tensor_record.payload_view = tensor_record.mapped_file->data + payload_offset;
      tensor_record.payload_view_size = static_cast<std::size_t>(byte_count);
    } else if (!read_file_range_to_bytes(
                   shard_path,
                   8 + header_len + tensor_record.byte_start,
                   byte_count,
                   tensor_record.payload)) {
      return false;
    }
    if (timing != nullptr) {
      add_timing(timing->mmap_file_read_ms, read_start);
    }
    record.total_byte_size += byte_count;
    record.tensors.emplace(tensor_record.kind, std::move(tensor_record));
  }

  auto prep_start = std::chrono::steady_clock::now();
  cache_quantized_linear_layout(record);
  if (timing != nullptr) {
    add_timing(timing->quantized_group_preparation_ms, prep_start);
  }
  return true;
}

int rusty_mlx_load_tensor_group(
    const char* handle,
    const char* model_dir,
    const char* group) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (handle == nullptr || model_dir == nullptr || group == nullptr) {
      return 0;
    }
    std::string handle_str(handle);
    std::string model_dir_str(model_dir);
    std::string group_str(group);
    auto spec = group_load_spec_for(group_str);
    if (!spec.has_value()) {
      return 0;
    }
    TensorGroupRecord record;
    if (!load_tensor_group_record(model_dir_str, *spec, record)) {
      return 0;
    }

    std::lock_guard<std::mutex> lock(tensor_group_mutex());
    tensor_group_table().emplace(handle_str, std::move(record));
    return 1;
  } catch (...) {
    return 0;
  }
#else
  (void)handle;
  (void)model_dir;
  (void)group;
  return 0;
#endif
}

const char* rusty_mlx_tensor_group_info_json(const char* handle) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (handle == nullptr) {
      output = "{\"ok\":false,\"error\":\"unknown_handle\"}";
      return output.c_str();
    }
    std::lock_guard<std::mutex> lock(tensor_group_mutex());
    auto it = tensor_group_table().find(handle);
    if (it == tensor_group_table().end()) {
      output = "{\"ok\":false,\"error\":\"unknown_handle\"}";
      return output.c_str();
    }
    const auto& record = it->second;
    auto tensor_json = [&](const char* kind) -> std::string {
      auto tensor_it = record.tensors.find(kind);
      if (tensor_it == record.tensors.end()) {
        return "null";
      }
      const auto& tensor = tensor_it->second;
      std::ostringstream out;
      out << "{"
          << "\"dtype\":\"" << json_escape(tensor.dtype) << "\","
          << "\"shape\":" << json_array_u64_to_string(tensor.shape) << ","
          << "\"byte_size\":" << tensor.payload.size() << ","
          << "\"source_file\":\"" << json_escape(tensor.source_file) << "\","
          << "\"byte_offsets\":" << json_array_u64_to_string({tensor.byte_start, tensor.byte_end})
          << "}";
      return out.str();
    };
    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"handle\":\"" << json_escape(record.group) << "\","
             << "\"group\":\"" << json_escape(record.group) << "\","
             << "\"loaded\":true,"
             << "\"source_dir\":\"" << json_escape(record.source_dir) << "\","
             << "\"index_path\":\"" << json_escape(record.index_path) << "\","
             << "\"quantized_group\":"
             << (record.quantized_group ? "true" : "false")
             << ","
             << "\"byte_size\":" << record.total_byte_size
             << ","
             << "\"weight\":" << tensor_json("weight") << ","
             << "\"scales\":" << tensor_json("scales") << ","
             << "\"biases\":" << tensor_json("biases")
             << "}";
    output = json_out.str();
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)handle;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_quantization_layout_probe_json(const char* handle) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (handle == nullptr) {
      output = "{\"ok\":false,\"error\":\"unknown_handle\"}";
      return output.c_str();
    }
    std::lock_guard<std::mutex> lock(tensor_group_mutex());
    auto it = tensor_group_table().find(handle);
    if (it == tensor_group_table().end()) {
      output = "{\"ok\":false,\"error\":\"unknown_handle\"}";
      return output.c_str();
    }
    const auto& record = it->second;
    auto weight_it = record.tensors.find("weight");
    auto scales_it = record.tensors.find("scales");
    auto biases_it = record.tensors.find("biases");
    if (weight_it == record.tensors.end() ||
        scales_it == record.tensors.end() ||
        biases_it == record.tensors.end()) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const auto& weight = weight_it->second;
    const auto& scales = scales_it->second;
    const auto& biases = biases_it->second;
    std::uint64_t weight_rows = weight.shape.size() > 0 ? weight.shape[0] : 0;
    std::uint64_t weight_cols = weight.shape.size() > 1 ? weight.shape[1] : 0;
    std::uint64_t scale_rows = scales.shape.size() > 0 ? scales.shape[0] : 0;
    std::uint64_t scale_cols = scales.shape.size() > 1 ? scales.shape[1] : 0;
    std::uint64_t bias_rows = biases.shape.size() > 0 ? biases.shape[0] : 0;
    std::uint64_t bias_cols = biases.shape.size() > 1 ? biases.shape[1] : 0;
    std::uint64_t inferred_block_size = scale_cols > 0 ? (weight_cols / scale_cols) : 0;
    std::uint64_t values_per_u32_guess = inferred_block_size;

    auto sample_raw_u32 = [](const std::vector<std::uint8_t>& payload) {
      std::vector<std::uint64_t> values;
      std::size_t limit = std::min<std::size_t>(8, payload.size() / 4);
      values.reserve(limit);
      for (std::size_t i = 0; i < limit; ++i) {
        values.push_back(read_u32_le_from_bytes(payload, i));
      }
      return values;
    };
    auto sample_raw_u16 = [](const std::vector<std::uint8_t>& payload) {
      std::vector<std::uint64_t> values;
      std::size_t limit = std::min<std::size_t>(8, payload.size() / 2);
      values.reserve(limit);
      for (std::size_t i = 0; i < limit; ++i) {
        values.push_back(read_u16_le_from_bytes(payload, i));
      }
      return values;
    };

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"handle\":\"" << json_escape(handle) << "\","
             << "\"weight\":{"
             << "\"dtype\":\"" << json_escape(weight.dtype) << "\","
             << "\"shape\":" << json_array_u64_to_string(weight.shape) << ","
             << "\"byte_size\":" << weight.payload.size() << ","
             << "\"source_file\":\"" << json_escape(weight.source_file) << "\","
             << "\"raw_words\":" << json_array_u64_to_string(sample_raw_u32(weight.payload))
             << "},"
             << "\"scales\":{"
             << "\"dtype\":\"" << json_escape(scales.dtype) << "\","
             << "\"shape\":" << json_array_u64_to_string(scales.shape) << ","
             << "\"byte_size\":" << scales.payload.size() << ","
             << "\"source_file\":\"" << json_escape(scales.source_file) << "\","
             << "\"raw_words\":" << json_array_u64_to_string(sample_raw_u16(scales.payload))
             << "},"
             << "\"biases\":{"
             << "\"dtype\":\"" << json_escape(biases.dtype) << "\","
             << "\"shape\":" << json_array_u64_to_string(biases.shape) << ","
             << "\"byte_size\":" << biases.payload.size() << ","
             << "\"source_file\":\"" << json_escape(biases.source_file) << "\","
             << "\"raw_words\":" << json_array_u64_to_string(sample_raw_u16(biases.payload))
             << "},"
             << "\"inferred_block_size\":" << inferred_block_size << ","
             << "\"values_per_u32_guess\":" << values_per_u32_guess << ","
             << "\"rows_cols_relation\":{"
             << "\"weight_rows\":" << weight_rows << ","
             << "\"weight_cols\":" << weight_cols << ","
             << "\"scale_rows\":" << scale_rows << ","
             << "\"scale_cols\":" << scale_cols << ","
             << "\"bias_rows\":" << bias_rows << ","
             << "\"bias_cols\":" << bias_cols << "},"
             << "\"notes\":["
             << "\"weight payload is stored as raw U32 words\","
             << "\"scales and biases are stored as BF16 words\","
             << "\"the first columns appear to be packed into blocks rather than materialized as full float tensors\","
             << "\"this is a layout probe only; real dequantization must match MLX quantized kernels before any meaningful matmul or generate work\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)handle;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

int rusty_mlx_free_tensor_group(const char* handle) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (handle == nullptr) {
      return 0;
    }
    std::lock_guard<std::mutex> lock(tensor_group_mutex());
    auto erased = tensor_group_table().erase(handle);
    return erased == 1 ? 1 : 0;
  } catch (...) {
    return 0;
  }
#else
  (void)handle;
  return 0;
#endif
}

int rusty_mlx_load_embedding_group(const char* handle, const char* model_dir) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (handle == nullptr || model_dir == nullptr) {
      return 0;
    }
    auto spec = embedding_group_spec_for();
    if (!spec.has_value()) {
      return 0;
    }
    std::string handle_str(handle);
    std::string model_dir_str(model_dir);
    TensorGroupRecord temp_record;
    if (!load_tensor_group_record(model_dir_str, *spec, temp_record)) {
      return 0;
    }
    EmbeddingGroupRecord record;
    record.group = temp_record.group;
    record.source_dir = temp_record.source_dir;
    record.index_path = temp_record.index_path;
    record.loaded = temp_record.loaded;
    record.quantized_group = temp_record.quantized_group;
    record.total_byte_size = temp_record.total_byte_size;
    record.tensors = std::move(temp_record.tensors);
    std::lock_guard<std::mutex> lock(embedding_group_mutex());
    embedding_group_table().emplace(handle_str, std::move(record));
    return 1;
  } catch (...) {
    return 0;
  }
#else
  (void)handle;
  (void)model_dir;
  return 0;
#endif
}

const char* rusty_mlx_embedding_group_info_json(const char* handle) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (handle == nullptr) {
      output = "{\"ok\":false,\"error\":\"unknown_handle\"}";
      return output.c_str();
    }
    std::lock_guard<std::mutex> lock(embedding_group_mutex());
    auto it = embedding_group_table().find(handle);
    if (it == embedding_group_table().end()) {
      output = "{\"ok\":false,\"error\":\"unknown_handle\"}";
      return output.c_str();
    }
    const auto& record = it->second;
    auto tensor_json = [&](const char* kind) -> std::string {
      auto tensor_it = record.tensors.find(kind);
      if (tensor_it == record.tensors.end()) {
        return "null";
      }
      const auto& tensor = tensor_it->second;
      std::ostringstream out;
      out << "{"
          << "\"dtype\":\"" << json_escape(tensor.dtype) << "\","
          << "\"shape\":" << json_array_u64_to_string(tensor.shape) << ","
          << "\"byte_size\":" << tensor.payload.size() << ","
          << "\"source_file\":\"" << json_escape(tensor.source_file) << "\","
          << "\"byte_offsets\":" << json_array_u64_to_string({tensor.byte_start, tensor.byte_end})
          << "}";
      return out.str();
    };
    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"handle\":\"" << json_escape(handle) << "\","
             << "\"group\":\"" << json_escape(record.group) << "\","
             << "\"loaded\":true,"
             << "\"source_dir\":\"" << json_escape(record.source_dir) << "\","
             << "\"index_path\":\"" << json_escape(record.index_path) << "\","
             << "\"quantized_group\":"
             << (record.quantized_group ? "true" : "false")
             << ","
             << "\"byte_size\":" << record.total_byte_size
             << ","
             << "\"weight\":" << tensor_json("weight") << ","
             << "\"scales\":" << tensor_json("scales") << ","
             << "\"biases\":" << tensor_json("biases")
             << "}";
    output = json_out.str();
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)handle;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

int rusty_mlx_free_embedding_group(const char* handle) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (handle == nullptr) {
      return 0;
    }
    std::lock_guard<std::mutex> lock(embedding_group_mutex());
    auto erased = embedding_group_table().erase(handle);
    return erased == 1 ? 1 : 0;
  } catch (...) {
    return 0;
  }
#else
  (void)handle;
  return 0;
#endif
}

const char* rusty_mlx_compare_dequant_slice_json(
    const char* handle,
    unsigned long long row,
    unsigned long long cols) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (handle == nullptr || cols == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    std::string handle_str(handle);
    std::lock_guard<std::mutex> lock(tensor_group_mutex());
    auto it = tensor_group_table().find(handle_str);
    if (it == tensor_group_table().end()) {
      output = "{\"ok\":false,\"error\":\"unknown_handle\"}";
      return output.c_str();
    }
    const auto& source = it->second;
    if (!source.quantized_group) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }

    const std::size_t comparison_cols = static_cast<std::size_t>(std::min<unsigned long long>(cols, 8));
    std::vector<std::string> notes = {
        "comparison uses a tiny 8-word packed block from the loaded quantized group",
        "provisional values use nibble * scale + bias and should match MLX only after the layout is confirmed",
        "real matmul/generate remains blocked until this dequantization is verified against MLX or the native quantized linear path is used directly",
    };
    if (comparison_cols < static_cast<std::size_t>(cols)) {
      notes.push_back("comparison is limited to the first 8 values of the requested slice");
    }
    if (comparison_cols == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }

    std::vector<float> provisional_values;
    bool comparison_available = false;
    std::vector<float> mlx_values;
    std::optional<double> max_abs_diff;

    provisional_values = provisional_dequant_slice_values(
        source, static_cast<std::size_t>(row), comparison_cols);

    try {
      const auto weight_it = source.tensors.find("weight");
      const auto scales_it = source.tensors.find("scales");
      const auto biases_it = source.tensors.find("biases");
      if (weight_it == source.tensors.end() ||
          scales_it == source.tensors.end() ||
          biases_it == source.tensors.end()) {
        throw std::runtime_error("quantized tensor group is missing weight/scales/biases");
      }

      const std::size_t weight_rows = weight_it->second.shape.size() > 0
          ? static_cast<std::size_t>(weight_it->second.shape[0])
          : 0;
      const std::size_t weight_cols = weight_it->second.shape.size() > 1
          ? static_cast<std::size_t>(weight_it->second.shape[1])
          : 0;
      const std::size_t scale_cols = scales_it->second.shape.size() > 1
          ? static_cast<std::size_t>(scales_it->second.shape[1])
          : 0;
      if (weight_rows == 0 || weight_cols < 8 || scale_cols == 0) {
        throw std::runtime_error("quantized tensor group has invalid shape metadata");
      }
      if (static_cast<std::size_t>(row) >= weight_rows) {
        throw std::runtime_error("requested row is out of range for quantized tensor group");
      }

      std::vector<std::uint32_t> tiny_weight_words(8, 0);
      for (std::size_t i = 0; i < tiny_weight_words.size(); ++i) {
        tiny_weight_words[i] = read_u32_le_from_bytes(
            weight_it->second.payload,
            static_cast<std::size_t>(row) * weight_cols + i);
      }

      std::vector<mlx::core::bfloat16_t> tiny_scales(1);
      std::vector<mlx::core::bfloat16_t> tiny_biases(1);
      std::memcpy(
          tiny_scales.data(),
          tensor_payload_data(scales_it->second) + static_cast<std::size_t>(row) * scale_cols * sizeof(std::uint16_t),
          sizeof(mlx::core::bfloat16_t));
      std::memcpy(
          tiny_biases.data(),
          tensor_payload_data(biases_it->second) + static_cast<std::size_t>(row) * scale_cols * sizeof(std::uint16_t),
          sizeof(mlx::core::bfloat16_t));

      mlx::core::array weight_array(
          tiny_weight_words.data(), mlx::core::Shape{1, 8}, mlx::core::uint32);
      mlx::core::array scales_array(
          tiny_scales.data(), mlx::core::Shape{1, 1}, mlx::core::bfloat16);
      mlx::core::array biases_array(
          tiny_biases.data(), mlx::core::Shape{1, 1}, mlx::core::bfloat16);
      auto native = mlx::core::dequantize(
          weight_array,
          scales_array,
          std::optional<mlx::core::array>(biases_array),
          64,
          4,
          "affine",
          std::optional<mlx::core::Dtype>(mlx::core::float32));
      native.eval();
      const auto* data = native.data<float>();
      std::size_t native_count = native.size();
      std::size_t compare_count = std::min<std::size_t>(comparison_cols, native_count);
      mlx_values.assign(data, data + compare_count);
      comparison_available = true;
      double diff = 0.0;
      for (std::size_t i = 0; i < compare_count; ++i) {
        diff = std::max(diff, static_cast<double>(std::fabs(provisional_values[i] - mlx_values[i])));
      }
      max_abs_diff = diff;
    } catch (const std::exception& err) {
      notes.push_back(std::string("MLX comparison unavailable: ") + err.what());
      notes.push_back("searched MLX public dequantization entry points: mlx::core::dequantize, mlx::core::quantized_matmul");
      notes.push_back("searched headers: mlx/ops.h and mlx/fast.h");
      comparison_available = false;
    } catch (...) {
      notes.push_back("MLX comparison unavailable: unknown exception");
      notes.push_back("searched MLX public dequantization entry points: mlx::core::dequantize, mlx::core::quantized_matmul");
      notes.push_back("searched headers: mlx/ops.h and mlx/fast.h");
      comparison_available = false;
    }

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"handle\":\"" << json_escape(handle) << "\","
             << "\"comparison_available\":" << (comparison_available ? "true" : "false") << ","
             << "\"provisional_values\":" << json_array_float_to_string(provisional_values) << ","
             << "\"mlx_values\":";
    if (comparison_available) {
      json_out << "[";
      for (std::size_t i = 0; i < mlx_values.size(); ++i) {
        if (i > 0) {
          json_out << ",";
        }
        json_out << mlx_values[i];
      }
      json_out << "]";
    } else {
      json_out << "null";
    }
    json_out << ","
             << "\"max_abs_diff\":";
    if (max_abs_diff.has_value()) {
      json_out << *max_abs_diff;
    } else {
      json_out << "null";
    }
    json_out << ","
             << "\"notes\":[";
    for (std::size_t i = 0; i < notes.size(); ++i) {
      if (i > 0) {
        json_out << ",";
      }
      json_out << "\"" << json_escape(notes[i]) << "\"";
    }
    json_out << "]}";
    output = json_out.str();
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)handle;
  (void)row;
  (void)cols;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_quantized_linear_slice_probe_json(
    const char* handle,
    const double* input,
    unsigned long long input_length,
    unsigned long long row,
    unsigned long long cols) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (handle == nullptr || input == nullptr || cols == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    std::string handle_str(handle);
    std::lock_guard<std::mutex> lock(tensor_group_mutex());
    auto it = tensor_group_table().find(handle_str);
    if (it == tensor_group_table().end()) {
      output = "{\"ok\":false,\"error\":\"unknown_handle\"}";
      return output.c_str();
    }
    const auto& source = it->second;
    if (!source.quantized_group) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }

    const std::size_t probe_cols = static_cast<std::size_t>(std::min<unsigned long long>(cols, 8));
    if (probe_cols == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }

    std::vector<float> dequantized_slice = provisional_dequant_slice_values(
        source, static_cast<std::size_t>(row), probe_cols);
    std::vector<float> input_values(probe_cols, 0.0f);
    std::size_t available = std::min<std::size_t>(probe_cols, static_cast<std::size_t>(input_length));
    for (std::size_t i = 0; i < available; ++i) {
      input_values[i] = static_cast<float>(input[i]);
    }

    double dot = 0.0;
    for (std::size_t i = 0; i < probe_cols; ++i) {
      dot += static_cast<double>(input_values[i]) * static_cast<double>(dequantized_slice[i]);
    }

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"handle\":\"" << json_escape(handle) << "\","
             << "\"row\":" << row << ","
             << "\"cols\":" << probe_cols << ","
             << "\"input\":" << json_array_float_to_string(input_values) << ","
             << "\"dequantized_slice\":" << json_array_float_to_string(dequantized_slice) << ","
             << "\"dot\":" << dot << ","
             << "\"provisional\":true,"
             << "\"notes\":["
             << "\"this probes one verified quantized slice, not full matrix multiplication\","
             << "\"real linear work still requires full vector/block traversal or MLX native quantized linear\","
             << "\"the slice uses the provisional nibble * scale + bias layout until matched against MLX\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)handle;
  (void)input;
  (void)input_length;
  (void)row;
  (void)cols;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_quantized_linear_rows_probe_json(
    const char* handle,
    const double* input,
    unsigned long long input_length,
    unsigned long long rows,
    unsigned long long cols) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (handle == nullptr || input == nullptr || rows == 0 || cols == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    std::string handle_str(handle);
    std::lock_guard<std::mutex> lock(tensor_group_mutex());
    auto it = tensor_group_table().find(handle_str);
    if (it == tensor_group_table().end()) {
      output = "{\"ok\":false,\"error\":\"unknown_handle\"}";
      return output.c_str();
    }
    const auto& source = it->second;
    if (!source.quantized_group) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }

    const std::size_t probe_rows = static_cast<std::size_t>(rows);
    const std::size_t probe_cols = static_cast<std::size_t>(std::min<unsigned long long>(cols, 8));
    if (probe_cols == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }

    std::vector<float> input_values(probe_cols, 0.0f);
    const std::size_t available = std::min<std::size_t>(probe_cols, static_cast<std::size_t>(input_length));
    for (std::size_t i = 0; i < available; ++i) {
      input_values[i] = static_cast<float>(input[i]);
    }

    std::vector<float> output_values;
    output_values.reserve(probe_rows);
    std::vector<std::string> notes = {
        "this proves row traversal for tiny quantized linear",
        "full projection still requires all input cols and optimized or native MLX quantized linear",
    };

    for (std::size_t row = 0; row < probe_rows; ++row) {
      std::vector<float> slice = provisional_dequant_slice_values(source, row, probe_cols);
      double dot = 0.0;
      for (std::size_t i = 0; i < probe_cols; ++i) {
        dot += static_cast<double>(input_values[i]) * static_cast<double>(slice[i]);
      }
      output_values.push_back(static_cast<float>(dot));
    }

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"handle\":\"" << json_escape(handle) << "\","
             << "\"rows\":" << probe_rows << ","
             << "\"cols\":" << probe_cols << ","
             << "\"input\":" << json_array_float_to_string(input_values) << ","
             << "\"output\":" << json_array_float_to_string(output_values) << ","
             << "\"provisional\":true,"
             << "\"notes\":[";
    for (std::size_t i = 0; i < notes.size(); ++i) {
      if (i > 0) {
        json_out << ",";
      }
      json_out << "\"" << json_escape(notes[i]) << "\"";
    }
    json_out << "]}";
    output = json_out.str();
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)handle;
  (void)input;
  (void)input_length;
  (void)rows;
  (void)cols;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_quantized_linear_fullrow_probe_json(
    const char* handle,
    const double* input,
    unsigned long long input_length,
    unsigned long long rows) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (handle == nullptr || input == nullptr || rows == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    std::string handle_str(handle);
    std::lock_guard<std::mutex> lock(tensor_group_mutex());
    auto it = tensor_group_table().find(handle_str);
    if (it == tensor_group_table().end()) {
      output = "{\"ok\":false,\"error\":\"unknown_handle\"}";
      return output.c_str();
    }
    const auto& source = it->second;
    if (!source.quantized_group) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }

    const auto& weight = source.tensors.at("weight");
    const std::size_t weight_rows = weight.shape.size() > 0
        ? static_cast<std::size_t>(weight.shape[0])
        : 0;
    const std::size_t weight_cols = weight.shape.size() > 1
        ? static_cast<std::size_t>(weight.shape[1])
        : 0;
    if (weight_rows == 0 || weight_cols == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }

    const std::size_t probe_rows = static_cast<std::size_t>(rows);
    const std::size_t logical_width = weight_cols * 8;
    if (probe_rows == 0 || static_cast<std::size_t>(input_length) != logical_width) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    if (probe_rows > weight_rows) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }

    std::vector<float> input_values(logical_width, 0.0f);
    for (std::size_t i = 0; i < logical_width; ++i) {
      input_values[i] = static_cast<float>(input[i]);
    }

    auto start = std::chrono::steady_clock::now();
    std::vector<float> output_values;
    output_values.reserve(probe_rows);
    for (std::size_t row = 0; row < probe_rows; ++row) {
      output_values.push_back(static_cast<float>(
          provisional_fullrow_dot_value(source, row, input_values)));
    }
    auto end = std::chrono::steady_clock::now();
    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"handle\":\"" << json_escape(handle) << "\","
             << "\"rows\":" << probe_rows << ","
             << "\"input_len\":" << logical_width << ","
             << "\"output\":" << json_array_float_to_string(output_values) << ","
             << "\"timing_ms\":" << timing_ms << ","
             << "\"provisional\":true,"
             << "\"notes\":["
             << "\"this traverses the full packed input width for one quantized projection\","
             << "\"the math remains provisional until matched against MLX quantized matmul\","
             << "\"future work may replace this with MLX native quantized linear\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)handle;
  (void)input;
  (void)input_length;
  (void)rows;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_quantized_linear_vector_probe_json(
    const char* handle,
    const double* input,
    unsigned long long input_length) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (handle == nullptr || input == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    std::string handle_str(handle);
    std::lock_guard<std::mutex> lock(tensor_group_mutex());
    auto it = tensor_group_table().find(handle_str);
    if (it == tensor_group_table().end()) {
      output = "{\"ok\":false,\"error\":\"unknown_handle\"}";
      return output.c_str();
    }
    const auto& source = it->second;
    if (!source.quantized_group) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }

    const auto& weight = source.tensors.at("weight");
    const std::size_t weight_rows = weight.shape.size() > 0
        ? static_cast<std::size_t>(weight.shape[0])
        : 0;
    const std::size_t weight_cols = weight.shape.size() > 1
        ? static_cast<std::size_t>(weight.shape[1])
        : 0;
    if (weight_rows == 0 || weight_cols == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }

    const std::size_t logical_width = weight_cols * 8;
    if (static_cast<std::size_t>(input_length) != logical_width) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }

    std::vector<float> input_values(logical_width, 0.0f);
    for (std::size_t i = 0; i < logical_width; ++i) {
      input_values[i] = static_cast<float>(input[i]);
    }

    auto start = std::chrono::steady_clock::now();
    std::vector<float> output_values;
    output_values.reserve(weight_rows);
    double checksum = 0.0;
    for (std::size_t row = 0; row < weight_rows; ++row) {
      const float value = static_cast<float>(provisional_fullrow_dot_value(source, row, input_values));
      output_values.push_back(value);
      checksum += static_cast<double>(value);
    }
    auto end = std::chrono::steady_clock::now();
    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();

    const std::size_t first_count = std::min<std::size_t>(8, output_values.size());
    const std::size_t last_count = std::min<std::size_t>(8, output_values.size());
    std::vector<float> first_values(output_values.begin(), output_values.begin() + first_count);
    std::vector<float> last_values(output_values.end() - last_count, output_values.end());

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"handle\":\"" << json_escape(handle) << "\","
             << "\"input_len\":" << logical_width << ","
             << "\"output_len\":" << output_values.size() << ","
             << "\"output\":" << json_array_float_to_string(output_values) << ","
             << "\"first_values\":" << json_array_float_to_string(first_values) << ","
             << "\"last_values\":" << json_array_float_to_string(last_values) << ","
             << "\"checksum\":" << checksum << ","
             << "\"timing_ms\":" << timing_ms << ","
             << "\"provisional\":true,"
             << "\"notes\":["
             << "\"this traverses the full output vector for one quantized projection\","
             << "\"the math remains provisional until matched against MLX quantized matmul\","
             << "\"future work may replace this with MLX native quantized linear or matmul\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)handle;
  (void)input;
  (void)input_length;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_rmsnorm_probe_json(
    const char* handle,
    const double* input,
    unsigned long long input_length) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (handle == nullptr || input == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    std::string handle_str(handle);
    std::lock_guard<std::mutex> lock(tensor_group_mutex());
    auto it = tensor_group_table().find(handle_str);
    if (it == tensor_group_table().end()) {
      output = "{\"ok\":false,\"error\":\"unknown_handle\"}";
      return output.c_str();
    }
    const auto& source = it->second;
    if (source.quantized_group) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }

    const auto& weight_tensor = source.tensors.at("weight");
    const std::size_t hidden_size = weight_tensor.shape.size() > 0
        ? static_cast<std::size_t>(weight_tensor.shape[0])
        : 0;
    if (hidden_size == 0 || static_cast<std::size_t>(input_length) != hidden_size) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }

    std::string config_json;
    const std::string config_path = join_path(source.source_dir, "config.json");
    if (!read_file_to_string(config_path, config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }

    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }

    std::vector<float> input_values(hidden_size, 0.0f);
    for (std::size_t i = 0; i < hidden_size; ++i) {
      input_values[i] = static_cast<float>(input[i]);
    }

    std::vector<float> weight_values = load_bf16_vector_from_tensor(weight_tensor);
    if (weight_values.size() != hidden_size) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }

    auto start = std::chrono::steady_clock::now();
    double mean_sq = 0.0;
    for (float value : input_values) {
      mean_sq += static_cast<double>(value) * static_cast<double>(value);
    }
    mean_sq /= static_cast<double>(hidden_size);
    const double scale = 1.0 / std::sqrt(mean_sq + eps);

    std::vector<float> output_values(hidden_size, 0.0f);
    double checksum = 0.0;
    for (std::size_t i = 0; i < hidden_size; ++i) {
      const float value = input_values[i] * static_cast<float>(scale) * weight_values[i];
      output_values[i] = value;
      checksum += static_cast<double>(value);
    }
    auto end = std::chrono::steady_clock::now();
    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();

    const std::size_t first_count = std::min<std::size_t>(8, output_values.size());
    const std::size_t last_count = std::min<std::size_t>(8, output_values.size());
    std::vector<float> first_values(output_values.begin(), output_values.begin() + first_count);
    std::vector<float> last_values(output_values.end() - last_count, output_values.end());

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"handle\":\"" << json_escape(handle) << "\","
             << "\"input_len\":" << hidden_size << ","
             << "\"output_len\":" << output_values.size() << ","
             << "\"eps\":" << eps << ","
             << "\"output\":" << json_array_float_to_string(output_values) << ","
             << "\"first_values\":" << json_array_float_to_string(first_values) << ","
             << "\"last_values\":" << json_array_float_to_string(last_values) << ","
             << "\"checksum\":" << checksum << ","
             << "\"timing_ms\":" << timing_ms << ","
             << "\"provisional\":true,"
             << "\"notes\":["
             << "\"this is a dense RMSNorm arithmetic probe for one loaded layernorm weight\","
             << "\"real attention is not started\","
             << "\"the next step after this is to compare against MLX's native norm path\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)handle;
  (void)input;
  (void)input_length;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_layer0_single_token_probe_json(
    const char* model_dir,
    unsigned long long token_id) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);

    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }

    TensorGroupRecord embedding;
    auto embedding_spec = embedding_group_spec_for();
    if (!load_tensor_group_record(model_dir_str, *embedding_spec, embedding)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load embedding group\"}";
      return output.c_str();
    }
    const auto& embedding_weight = embedding.tensors.at("weight");
    const std::size_t vocab_size = embedding_weight.shape.size() > 0
        ? static_cast<std::size_t>(embedding_weight.shape[0])
        : 0;
    const std::size_t embedding_packed_width = embedding_weight.shape.size() > 1
        ? static_cast<std::size_t>(embedding_weight.shape[1])
        : 0;
    const std::size_t hidden_size = embedding_packed_width * 8;
    if (vocab_size == 0 || hidden_size == 0 ||
        static_cast<std::size_t>(token_id) >= vocab_size) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"token id out of range\"}";
      return output.c_str();
    }

    TensorGroupRecord input_norm;
    TensorGroupRecord q_proj;
    TensorGroupRecord k_proj;
    TensorGroupRecord v_proj;
    TensorGroupRecord o_proj;
    const std::string groups[] = {
        "model.layers.0.input_layernorm",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
    };
    TensorGroupRecord* records[] = {
        &input_norm,
        &q_proj,
        &k_proj,
        &v_proj,
        &o_proj,
    };
    for (std::size_t i = 0; i < 5; ++i) {
      auto spec = group_load_spec_for(groups[i]);
      if (!spec || !load_tensor_group_record(model_dir_str, *spec, *records[i])) {
        output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load layer 0 tensor group\"}";
        return output.c_str();
      }
    }

    auto start = std::chrono::steady_clock::now();
    std::vector<float> embedding_values =
        provisional_dequant_slice_values(embedding, static_cast<std::size_t>(token_id), hidden_size);
    std::vector<float> norm_values = rmsnorm_values(input_norm, embedding_values, eps);
    std::vector<float> q_values = quantized_linear_vector_values(q_proj, norm_values);
    std::vector<float> k_values = quantized_linear_vector_values(k_proj, norm_values);
    std::vector<float> v_values = quantized_linear_vector_values(v_proj, norm_values);
    double qk_score_checksum = 0.0;
    std::vector<float> attention_values =
        single_token_attention_values(q_values, k_values, v_values, qk_score_checksum);
    std::vector<float> o_values = quantized_linear_vector_values(o_proj, attention_values);
    if (o_values.size() != embedding_values.size()) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"o_proj output does not match residual width\"}";
      return output.c_str();
    }
    std::vector<float> residual_values(o_values.size(), 0.0f);
    for (std::size_t i = 0; i < residual_values.size(); ++i) {
      residual_values[i] = embedding_values[i] + o_values[i];
    }
    auto end = std::chrono::steady_clock::now();
    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"token_id\":" << token_id << ","
             << "\"embedding_len\":" << embedding_values.size() << ","
             << "\"norm_len\":" << norm_values.size() << ","
             << "\"q_len\":" << q_values.size() << ","
             << "\"k_len\":" << k_values.size() << ","
             << "\"v_len\":" << v_values.size() << ","
             << "\"attention_len\":" << attention_values.size() << ","
             << "\"o_len\":" << o_values.size() << ","
             << "\"residual_len\":" << residual_values.size() << ","
             << "\"checksums\":{"
             << "\"embedding\":" << vector_checksum(embedding_values) << ","
             << "\"norm\":" << vector_checksum(norm_values) << ","
             << "\"q\":" << vector_checksum(q_values) << ","
             << "\"k\":" << vector_checksum(k_values) << ","
             << "\"v\":" << vector_checksum(v_values) << ","
             << "\"qk_score\":" << qk_score_checksum << ","
             << "\"attention\":" << vector_checksum(attention_values) << ","
             << "\"o\":" << vector_checksum(o_values) << ","
             << "\"residual\":" << vector_checksum(residual_values)
             << "},"
             << "\"first_values\":{"
             << "\"embedding\":" << json_array_float_to_string(first_values_of(embedding_values)) << ","
             << "\"norm\":" << json_array_float_to_string(first_values_of(norm_values)) << ","
             << "\"q\":" << json_array_float_to_string(first_values_of(q_values)) << ","
             << "\"k\":" << json_array_float_to_string(first_values_of(k_values)) << ","
             << "\"v\":" << json_array_float_to_string(first_values_of(v_values)) << ","
             << "\"attention\":" << json_array_float_to_string(first_values_of(attention_values)) << ","
             << "\"o\":" << json_array_float_to_string(first_values_of(o_values)) << ","
             << "\"residual\":" << json_array_float_to_string(first_values_of(residual_values))
             << "},"
             << "\"timing_ms\":" << timing_ms << ","
             << "\"provisional\":true,"
             << "\"notes\":["
             << "\"single-token attention has softmax weight 1, so attention output is grouped-query V expansion\","
             << "\"this probes layer 0 only and does not create a KV cache\","
             << "\"projection math uses the existing provisional quantized layout\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)token_id;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_layer0_mlp_probe_json(
    const char* model_dir,
    unsigned long long token_id) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);

    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }

    TensorGroupRecord post_attention_norm;
    TensorGroupRecord gate_proj;
    TensorGroupRecord up_proj;
    TensorGroupRecord down_proj;
    const std::string groups[] = {
        "model.layers.0.post_attention_layernorm",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    };
    TensorGroupRecord* records[] = {
        &post_attention_norm,
        &gate_proj,
        &up_proj,
        &down_proj,
    };
    for (std::size_t i = 0; i < 4; ++i) {
      auto spec = group_load_spec_for(groups[i]);
      if (!spec || !load_tensor_group_record(model_dir_str, *spec, *records[i])) {
        output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load layer 0 MLP tensor group\"}";
        return output.c_str();
      }
    }

    auto start = std::chrono::steady_clock::now();
    std::vector<float> input_values =
        layer0_attention_residual_values(model_dir_str, static_cast<std::size_t>(token_id), eps);
    std::vector<float> norm_values = rmsnorm_values(post_attention_norm, input_values, eps);
    std::vector<float> gate_values = quantized_linear_vector_values(gate_proj, norm_values);
    std::vector<float> up_values = quantized_linear_vector_values(up_proj, norm_values);
    if (gate_values.size() != up_values.size()) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"gate/up projection sizes differ\"}";
      return output.c_str();
    }

    std::vector<float> activated_values(gate_values.size(), 0.0f);
    for (std::size_t i = 0; i < activated_values.size(); ++i) {
      const float gate = gate_values[i];
      const float silu = gate / (1.0f + std::exp(-gate));
      activated_values[i] = silu * up_values[i];
    }

    std::vector<float> down_values = quantized_linear_vector_values(down_proj, activated_values);
    if (down_values.size() != input_values.size()) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"down_proj output does not match residual width\"}";
      return output.c_str();
    }
    std::vector<float> residual_values(down_values.size(), 0.0f);
    for (std::size_t i = 0; i < residual_values.size(); ++i) {
      residual_values[i] = input_values[i] + down_values[i];
    }
    auto end = std::chrono::steady_clock::now();
    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"token_id\":" << token_id << ","
             << "\"input_len\":" << input_values.size() << ","
             << "\"norm_len\":" << norm_values.size() << ","
             << "\"gate_len\":" << gate_values.size() << ","
             << "\"up_len\":" << up_values.size() << ","
             << "\"activated_len\":" << activated_values.size() << ","
             << "\"down_len\":" << down_values.size() << ","
             << "\"residual_len\":" << residual_values.size() << ","
             << "\"checksums\":{"
             << "\"input\":" << vector_checksum(input_values) << ","
             << "\"norm\":" << vector_checksum(norm_values) << ","
             << "\"gate\":" << vector_checksum(gate_values) << ","
             << "\"up\":" << vector_checksum(up_values) << ","
             << "\"activated\":" << vector_checksum(activated_values) << ","
             << "\"down\":" << vector_checksum(down_values) << ","
             << "\"residual\":" << vector_checksum(residual_values)
             << "},"
             << "\"first_values\":{"
             << "\"input\":" << json_array_float_to_string(first_values_of(input_values)) << ","
             << "\"norm\":" << json_array_float_to_string(first_values_of(norm_values)) << ","
             << "\"gate\":" << json_array_float_to_string(first_values_of(gate_values)) << ","
             << "\"up\":" << json_array_float_to_string(first_values_of(up_values)) << ","
             << "\"activated\":" << json_array_float_to_string(first_values_of(activated_values)) << ","
             << "\"down\":" << json_array_float_to_string(first_values_of(down_values)) << ","
             << "\"residual\":" << json_array_float_to_string(first_values_of(residual_values))
             << "},"
             << "\"timing_ms\":" << timing_ms << ","
             << "\"provisional\":true,"
             << "\"notes\":["
             << "\"this is a layer 0 MLP verifier probe only\","
             << "\"input is the residual output of the layer0 single-token attention path\","
             << "\"projection math uses the existing provisional quantized layout\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)token_id;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_layer0_block_probe_json(
    const char* model_dir,
    unsigned long long token_id) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);

    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }

    TensorGroupRecord post_attention_norm;
    TensorGroupRecord gate_proj;
    TensorGroupRecord up_proj;
    TensorGroupRecord down_proj;
    const std::string groups[] = {
        "model.layers.0.post_attention_layernorm",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    };
    TensorGroupRecord* records[] = {
        &post_attention_norm,
        &gate_proj,
        &up_proj,
        &down_proj,
    };
    for (std::size_t i = 0; i < 4; ++i) {
      auto spec = group_load_spec_for(groups[i]);
      if (!spec || !load_tensor_group_record(model_dir_str, *spec, *records[i])) {
        output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load layer 0 MLP tensor group\"}";
        return output.c_str();
      }
    }

    auto start = std::chrono::steady_clock::now();
    std::vector<float> embedding_values =
        layer0_embedding_values(model_dir_str, static_cast<std::size_t>(token_id));
    std::vector<float> attention_residual_values =
        layer0_attention_residual_values(model_dir_str, static_cast<std::size_t>(token_id), eps);
    std::vector<float> norm_values =
        rmsnorm_values(post_attention_norm, attention_residual_values, eps);
    std::vector<float> gate_values = quantized_linear_vector_values(gate_proj, norm_values);
    std::vector<float> up_values = quantized_linear_vector_values(up_proj, norm_values);
    if (gate_values.size() != up_values.size()) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"gate/up projection sizes differ\"}";
      return output.c_str();
    }

    std::vector<float> activated_values(gate_values.size(), 0.0f);
    for (std::size_t i = 0; i < activated_values.size(); ++i) {
      const float gate = gate_values[i];
      const float silu = gate / (1.0f + std::exp(-gate));
      activated_values[i] = silu * up_values[i];
    }
    std::vector<float> down_values = quantized_linear_vector_values(down_proj, activated_values);
    if (down_values.size() != attention_residual_values.size()) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"down_proj output does not match residual width\"}";
      return output.c_str();
    }
    std::vector<float> mlp_residual_values(down_values.size(), 0.0f);
    for (std::size_t i = 0; i < mlp_residual_values.size(); ++i) {
      mlp_residual_values[i] = attention_residual_values[i] + down_values[i];
    }
    auto end = std::chrono::steady_clock::now();
    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"token_id\":" << token_id << ","
             << "\"input_embedding_checksum\":" << vector_checksum(embedding_values) << ","
             << "\"attention_residual_checksum\":" << vector_checksum(attention_residual_values) << ","
             << "\"mlp_residual_checksum\":" << vector_checksum(mlp_residual_values) << ","
             << "\"output_len\":" << mlp_residual_values.size() << ","
             << "\"first_values\":" << json_array_float_to_string(first_values_of(mlp_residual_values)) << ","
             << "\"timing_ms\":" << timing_ms << ","
             << "\"provisional\":true,"
             << "\"notes\":["
             << "\"this is a consolidated layer 0 block verifier probe only\","
             << "\"it reuses the layer0 attention and MLP probe arithmetic\","
             << "\"no KV cache, generation loop, or production wiring is added\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)token_id;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_layer_stack_probe_json(
    const char* model_dir,
    unsigned long long token_id,
    unsigned long long layers) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    if (layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"layers must be greater than zero\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);

    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }

    auto start = std::chrono::steady_clock::now();
    std::vector<float> embedding_values =
        layer0_embedding_values(model_dir_str, static_cast<std::size_t>(token_id));
    std::vector<float> current_values = embedding_values;
    std::vector<double> per_layer_output_checksums;
    per_layer_output_checksums.reserve(static_cast<std::size_t>(layers));
    for (unsigned long long layer = 0; layer < layers; ++layer) {
      std::vector<float> attention_values =
          layer_attention_residual_from_input(model_dir_str, static_cast<int>(layer), current_values, eps);
      current_values =
          layer_mlp_residual_from_input(model_dir_str, static_cast<int>(layer), attention_values, eps);
      per_layer_output_checksums.push_back(vector_checksum(current_values));
    }
    auto end = std::chrono::steady_clock::now();
    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"token_id\":" << token_id << ","
             << "\"layers_run\":" << layers << ","
             << "\"per_layer_output_checksums\":" << json_array_double_to_string(per_layer_output_checksums) << ","
             << "\"layer0_output_checksum\":" << (per_layer_output_checksums.size() > 0 ? per_layer_output_checksums[0] : 0.0) << ","
             << "\"layer1_output_checksum\":" << (per_layer_output_checksums.size() > 1 ? per_layer_output_checksums[1] : 0.0) << ","
             << "\"final_output_len\":" << current_values.size() << ","
             << "\"output_len\":" << current_values.size() << ","
             << "\"first_values\":" << json_array_float_to_string(first_values_of(current_values)) << ","
             << "\"timing_ms\":" << timing_ms << ","
             << "\"provisional\":true,"
             << "\"notes\":["
             << "\"this is a verifier-only N-layer stack probe\","
             << "\"token id 1 is embedded, then passed through the requested first layers\","
             << "\"no KV cache, generation loop, or production wiring is added\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)token_id;
  (void)layers;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_full_stack_single_token_probe_json(
    const char* model_dir,
    unsigned long long token_id) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);

    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }
    std::uint64_t layers = 0;
    if (!extract_json_u64_value(config_json, "num_hidden_layers", layers) || layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"num_hidden_layers missing\"}";
      return output.c_str();
    }
    TensorGroupRecord final_norm;
    auto final_norm_spec = group_load_spec_for("model.norm");
    if (!final_norm_spec || !load_tensor_group_record(model_dir_str, *final_norm_spec, final_norm)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load final norm\"}";
      return output.c_str();
    }

    TensorGroupRecord embedding;
    auto embedding_spec = embedding_group_spec_for();
    if (!embedding_spec || !load_tensor_group_record(model_dir_str, *embedding_spec, embedding)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load tied embedding projection\"}";
      return output.c_str();
    }

    auto start = std::chrono::steady_clock::now();
    NativeSessionKvCache structural_cache;
    structural_cache.owner_session = "full_stack_single_token_probe";
    structural_cache.layers.resize(static_cast<std::size_t>(layers));
    structural_cache.layers_allocated = layers;

    std::vector<float> current_values =
        layer0_embedding_values(model_dir_str, static_cast<std::size_t>(token_id));
    std::vector<double> per_layer_output_checksums;
    per_layer_output_checksums.reserve(static_cast<std::size_t>(layers));
    for (std::uint64_t layer = 0; layer < layers; ++layer) {
      std::vector<float> attention_values =
          layer_attention_residual_from_input(
              model_dir_str,
              static_cast<int>(layer),
              current_values,
              eps,
              &structural_cache);
      current_values =
          layer_mlp_residual_from_input(model_dir_str, static_cast<int>(layer), attention_values, eps);
      per_layer_output_checksums.push_back(vector_checksum(current_values));
    }

    const double final_hidden_checksum = vector_checksum(current_values);
    std::vector<float> final_norm_values = rmsnorm_values(final_norm, current_values, eps);
    const double final_norm_checksum = vector_checksum(final_norm_values);
    std::vector<float> logits = quantized_linear_vector_values(embedding, final_norm_values);
    std::vector<std::pair<std::uint64_t, float>> top = top_logits(logits, 10);
    auto end = std::chrono::steady_clock::now();
    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();

    const double layer0_checksum = per_layer_output_checksums.size() > 0 ? per_layer_output_checksums[0] : 0.0;
    const double layer1_checksum = per_layer_output_checksums.size() > 1 ? per_layer_output_checksums[1] : 0.0;
    const double layer2_checksum = per_layer_output_checksums.size() > 2 ? per_layer_output_checksums[2] : 0.0;
    const double layer3_checksum = per_layer_output_checksums.size() > 3 ? per_layer_output_checksums[3] : 0.0;
    const double last_checksum = per_layer_output_checksums.empty() ? 0.0 : per_layer_output_checksums.back();

    std::size_t first_k_length = 0;
    std::size_t first_v_length = 0;
    if (!structural_cache.layers.empty() &&
        !structural_cache.layers[0].keys.empty() &&
        !structural_cache.layers[0].values.empty()) {
      first_k_length = structural_cache.layers[0].keys[0].size();
      first_v_length = structural_cache.layers[0].values[0].size();
    }
    {
      std::lock_guard<std::mutex> lock(native_kv_cache_mutex());
      native_kv_cache_table()["full_stack_single_token_probe"] = structural_cache;
    }

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"token_id\":" << token_id << ","
             << "\"layers_run\":" << layers << ","
             << "\"selected_per_layer_checksums\":{"
             << "\"0\":" << layer0_checksum << ","
             << "\"1\":" << layer1_checksum << ","
             << "\"2\":" << layer2_checksum << ","
             << "\"3\":" << layer3_checksum << ","
             << "\"last\":" << last_checksum
             << "},"
             << "\"final_hidden_checksum\":" << final_hidden_checksum << ","
             << "\"final_norm_checksum\":" << final_norm_checksum << ","
             << "\"logits_len\":" << logits.size() << ","
             << "\"top_logits\":" << json_top_logits_to_string(top) << ","
             << "\"kv_cache_handle\":\"full_stack_single_token_probe\","
             << "\"kv_cache_structural\":true,"
             << "\"kv_cache_layers_allocated\":" << structural_cache.layers_allocated << ","
             << "\"kv_cache_positions_stored\":" << structural_cache.positions_stored << ","
             << "\"kv_cache_k_length\":" << first_k_length << ","
             << "\"kv_cache_v_length\":" << first_v_length << ","
             << "\"timing_ms\":" << timing_ms << ","
             << "\"provisional\":true,"
             << "\"projection_source\":\"model.embed_tokens\","
             << "\"notes\":["
             << "\"this is a verifier-only full-stack single-token probe\","
             << "\"lm_head is absent for model4, so tied embedding projection is used\","
             << "\"no KV cache, generation loop, or production wiring is added\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)token_id;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_kv_cache_storage_probe_json() {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    std::lock_guard<std::mutex> lock(native_kv_cache_mutex());
    auto it = native_kv_cache_table().find("full_stack_single_token_probe");
    if (it == native_kv_cache_table().end()) {
      output = "{\"ok\":false,\"error\":\"missing_cache\",\"message\":\"full-stack probe has not populated the structural KV cache\"}";
      return output.c_str();
    }
    const NativeSessionKvCache& cache = it->second;
    std::uint64_t layers_with_positions = 0;
    std::uint64_t first_k_length = 0;
    std::uint64_t first_v_length = 0;
    std::uint64_t last_k_length = 0;
    std::uint64_t last_v_length = 0;
    for (std::size_t i = 0; i < cache.layers.size(); ++i) {
      const auto& layer = cache.layers[i];
      if (!layer.keys.empty() && !layer.values.empty()) {
        layers_with_positions += 1;
        if (first_k_length == 0 && first_v_length == 0) {
          first_k_length = static_cast<std::uint64_t>(layer.keys[0].size());
          first_v_length = static_cast<std::uint64_t>(layer.values[0].size());
        }
        last_k_length = static_cast<std::uint64_t>(layer.keys.back().size());
        last_v_length = static_cast<std::uint64_t>(layer.values.back().size());
      }
    }

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"cache_handle\":\"full_stack_single_token_probe\","
             << "\"owner_session\":\"" << json_escape(cache.owner_session) << "\","
             << "\"structural_cache\":true,"
             << "\"optimized_incremental_attention\":false,"
             << "\"layers_allocated\":" << cache.layers_allocated << ","
             << "\"layers_with_positions\":" << layers_with_positions << ","
             << "\"positions_stored\":" << cache.positions_stored << ","
             << "\"k_length\":" << first_k_length << ","
             << "\"v_length\":" << first_v_length << ","
             << "\"first_layer\":{"
             << "\"positions\":" << (cache.layers.empty() ? 0 : cache.layers[0].keys.size()) << ","
             << "\"k_length\":" << first_k_length << ","
             << "\"v_length\":" << first_v_length
             << "},"
             << "\"last_layer\":{"
             << "\"positions\":" << (cache.layers.empty() ? 0 : cache.layers.back().keys.size()) << ","
             << "\"k_length\":" << last_k_length << ","
             << "\"v_length\":" << last_v_length
             << "},"
             << "\"notes\":["
             << "\"native structural KV vectors are stored per layer and token position\","
             << "\"append-only storage is present for decode positions\","
             << "\"this probe does not implement optimized incremental attention or generation\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_incremental_attention_probe_json(
    const char* model_dir,
    unsigned long long prompt_token_id,
    unsigned long long decode_token_id) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);

    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }
    std::uint64_t layers = 0;
    if (!extract_json_u64_value(config_json, "num_hidden_layers", layers) || layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"num_hidden_layers missing\"}";
      return output.c_str();
    }

    NativeSessionKvCache working_cache;
    {
      std::lock_guard<std::mutex> lock(native_kv_cache_mutex());
      auto it = native_kv_cache_table().find("full_stack_single_token_probe");
      if (it == native_kv_cache_table().end()) {
        output = "{\"ok\":false,\"error\":\"missing_cache\",\"message\":\"full-stack probe has not populated the structural KV cache\"}";
        return output.c_str();
      }
      working_cache = it->second;
    }
    const std::uint64_t positions_before = working_cache.positions_stored;
    if (working_cache.layers_allocated != layers || positions_before == 0) {
      output = "{\"ok\":false,\"error\":\"bad_cache\",\"message\":\"KV cache does not match model layer count or has no positions\"}";
      return output.c_str();
    }

    TensorGroupRecord final_norm;
    auto final_norm_spec = group_load_spec_for("model.norm");
    if (!final_norm_spec || !load_tensor_group_record(model_dir_str, *final_norm_spec, final_norm)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load final norm\"}";
      return output.c_str();
    }

    TensorGroupRecord embedding;
    auto embedding_spec = embedding_group_spec_for();
    if (!embedding_spec || !load_tensor_group_record(model_dir_str, *embedding_spec, embedding)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load tied embedding projection\"}";
      return output.c_str();
    }

    std::size_t k_length = 0;
    std::size_t v_length = 0;
    if (!working_cache.layers.empty() &&
        !working_cache.layers[0].keys.empty() &&
        !working_cache.layers[0].values.empty()) {
      k_length = working_cache.layers[0].keys[0].size();
      v_length = working_cache.layers[0].values[0].size();
    }

    auto start = std::chrono::steady_clock::now();
    std::vector<float> current_values =
        layer0_embedding_values(model_dir_str, static_cast<std::size_t>(decode_token_id));
    for (std::uint64_t layer = 0; layer < layers; ++layer) {
      std::vector<float> attention_values =
          layer_attention_residual_incremental_from_input(
              model_dir_str,
              static_cast<int>(layer),
              current_values,
              eps,
              working_cache);
      current_values =
          layer_mlp_residual_from_input(model_dir_str, static_cast<int>(layer), attention_values, eps);
    }

    std::vector<float> final_norm_values = rmsnorm_values(final_norm, current_values, eps);
    const double final_norm_checksum = vector_checksum(final_norm_values);
    std::vector<float> logits = quantized_linear_vector_values(embedding, final_norm_values);
    std::vector<std::pair<std::uint64_t, float>> top = top_logits(logits, 1);
    if (top.empty()) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"no logits produced\"}";
      return output.c_str();
    }
    auto end = std::chrono::steady_clock::now();
    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const std::uint64_t positions_after = working_cache.positions_stored;

    {
      std::lock_guard<std::mutex> lock(native_kv_cache_mutex());
      native_kv_cache_table()["full_stack_single_token_probe"] = working_cache;
    }

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"prompt_token_id\":" << prompt_token_id << ","
             << "\"decode_token_id\":" << decode_token_id << ","
             << "\"layers_run\":" << layers << ","
             << "\"positions_before\":" << positions_before << ","
             << "\"positions_after\":" << positions_after << ","
             << "\"k_length\":" << k_length << ","
             << "\"v_length\":" << v_length << ","
             << "\"logits_len\":" << logits.size() << ","
             << "\"top_token_id\":" << top[0].first << ","
             << "\"top_token_score\":" << top[0].second << ","
             << "\"final_norm_checksum\":" << final_norm_checksum << ","
             << "\"timing_ms\":" << timing_ms << ","
             << "\"kv_cache_reused\":" << (positions_before > 0 ? "true" : "false") << ","
             << "\"optimized_incremental_attention\":" << (positions_before > 0 ? "true" : "false") << ","
             << "\"structural_cache\":false,"
             << "\"provisional\":true,"
             << "\"notes\":["
             << "\"decode token Q/K/V are computed once\","
             << "\"attention reads cached prior K/V plus current K/V\","
             << "\"current K/V are appended to the native session cache\","
             << "\"this is CPU/provisional and not yet an optimized MLX kernel\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)prompt_token_id;
  (void)decode_token_id;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_session_layer_residency_probe_json(
    const char* model_dir,
    unsigned long long prompt_token_id,
    unsigned long long decode_token_id) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    const auto total_probe_start = std::chrono::steady_clock::now();
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);
    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }
    std::uint64_t layers = 0;
    if (!extract_json_u64_value(config_json, "num_hidden_layers", layers) || layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"num_hidden_layers missing\"}";
      return output.c_str();
    }
    const bool fallback_used = force_scalar_quantized_linear();
    const bool use_optimized_linear = !fallback_used;
    const bool use_layout_cached_linear = use_optimized_linear;
    const bool use_mlp_pair_optimized = use_layout_cached_linear;
    const bool use_logits_top1_optimized = use_mlp_pair_optimized;
    const bool use_mlx_resident_mlp_chain =
        use_mlp_pair_optimized && !disable_mlx_resident_mlp_chain();
    const bool use_mlx_resident_layer_block =
        use_mlx_resident_mlp_chain && !disable_mlx_resident_layer_block();
    const bool use_gate_up_full_block_optimized = false;
    const bool use_down_full_block_optimized = false;
    if (use_optimized_linear && !disable_mlx_quantized_linear()) {
      // Metal-first decode must not depend on a prior comparison probe to enable
      // the native quantized path. Local failures are still recorded as fallback
      // steps by quantized_linear_vector_values_selected.
      mlx_quantized_linear_available_flag() = true;
    }
    const bool mlx_quantized_linear_available =
        mlx_quantized_linear_available_flag() && !disable_mlx_quantized_linear();
    mlx_quantized_linear_runtime_fallback_used = false;
    mlx_quantized_linear_fallback_steps.clear();
    bool loaded_now = false;
    const auto resident_group_load_start = std::chrono::steady_clock::now();
    ResidentModelLayers& resident = resident_layers_for(model_dir_str, layers, loaded_now);
    const auto resident_group_load_end = std::chrono::steady_clock::now();
    const double resident_group_load_ms =
        std::chrono::duration<double, std::milli>(
            resident_group_load_end - resident_group_load_start).count();
    const std::uint64_t tensor_group_load_count = loaded_now ? resident.tensor_group_load_count : 0;

    TensorGroupRecord final_norm;
    auto final_norm_spec = group_load_spec_for("model.norm");
    if (!final_norm_spec || !load_tensor_group_record(model_dir_str, *final_norm_spec, final_norm)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load final norm\"}";
      return output.c_str();
    }
    TensorGroupRecord embedding;
    auto embedding_spec = embedding_group_spec_for();
    if (!embedding_spec || !load_tensor_group_record(model_dir_str, *embedding_spec, embedding)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load tied embedding projection\"}";
      return output.c_str();
    }
    const auto projection_warmup_start = std::chrono::steady_clock::now();
    ResidentProjectionWarmupTiming projection_warmup_timing;
    if (use_resident_mlx_projection_arrays()) {
      warm_resident_mlx_projection_arrays(resident, embedding, &projection_warmup_timing);
    }
    const double resident_projection_array_warmup_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - projection_warmup_start).count();

    bool mlp_chain_compare_ok = false;
    bool mlp_chain_compare_available = false;
    std::size_t mlp_chain_compare_output_len = 0;
    double mlp_chain_compare_checksum_current = 0.0;
    double mlp_chain_compare_checksum_resident = 0.0;
    double mlp_chain_compare_max_abs_diff = 0.0;
    double mlp_chain_compare_timing_current_ms = 0.0;
    double mlp_chain_compare_timing_resident_ms = 0.0;
    const bool run_mlp_chain_compare =
        enable_mlx_resident_mlp_chain_compare() && use_mlx_resident_mlp_chain && !resident.layers.empty();
    if (run_mlp_chain_compare) {
      try {
        const ResidentLayerGroups& layer0 = resident.layers[0];
        std::vector<float> compare_input(layer0.down_proj.quantized_rows, 0.0f);
        for (std::size_t i = 0; i < compare_input.size(); ++i) {
          compare_input[i] = static_cast<float>((static_cast<int>(i % 17) - 8)) / 8.0f;
        }
        ResidentDecodeResult current_timing;
        auto compare_current_start = std::chrono::steady_clock::now();
        std::vector<float> current_residual = layer_mlp_residual_resident_from_input(
            layer0,
            compare_input,
            eps,
            use_optimized_linear,
            use_layout_cached_linear,
            use_mlp_pair_optimized,
            false,
            false,
            false,
            &current_timing);
        mlp_chain_compare_timing_current_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - compare_current_start).count();
        ResidentDecodeResult chain_timing;
        auto compare_chain_start = std::chrono::steady_clock::now();
        std::vector<float> chain_residual = layer_mlp_residual_resident_from_input(
            layer0,
            compare_input,
            eps,
            use_optimized_linear,
            use_layout_cached_linear,
            use_mlp_pair_optimized,
            false,
            false,
            true,
            &chain_timing);
        mlp_chain_compare_timing_resident_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - compare_chain_start).count();
        if (current_residual.size() == chain_residual.size()) {
          mlp_chain_compare_output_len = chain_residual.size();
          for (std::size_t i = 0; i < current_residual.size(); ++i) {
            mlp_chain_compare_max_abs_diff = std::max(
                mlp_chain_compare_max_abs_diff,
                std::abs(static_cast<double>(current_residual[i]) - static_cast<double>(chain_residual[i])));
          }
          mlp_chain_compare_checksum_current = vector_checksum(current_residual);
          mlp_chain_compare_checksum_resident = vector_checksum(chain_residual);
          mlp_chain_compare_available = chain_timing.mlx_resident_mlp_chain_applied;
          mlp_chain_compare_ok = mlp_chain_compare_available && mlp_chain_compare_max_abs_diff <= 0.05;
        }
      } catch (...) {
        mlp_chain_compare_ok = false;
        mlp_chain_compare_available = false;
      }
      mlx_quantized_linear_runtime_fallback_used = false;
      mlx_quantized_linear_fallback_steps.clear();
    }
    const bool apply_mlx_resident_mlp_chain =
        use_mlx_resident_mlp_chain && (!run_mlp_chain_compare || mlp_chain_compare_ok);

    NativeSessionKvCache kv_cache;
    kv_cache.owner_session = "session_layer_residency_probe";
    kv_cache.layers.resize(static_cast<std::size_t>(layers));
    kv_cache.layers_allocated = layers;

    auto prompt_start = std::chrono::steady_clock::now();
    std::vector<float> current_values =
        layer0_embedding_values(model_dir_str, static_cast<std::size_t>(prompt_token_id));
    for (std::uint64_t layer = 0; layer < layers; ++layer) {
      const ResidentLayerGroups& groups = resident.layers[static_cast<std::size_t>(layer)];
      std::vector<float> attention_values =
          layer_attention_residual_resident_from_input(
              groups,
              static_cast<std::size_t>(layer),
              current_values,
              eps,
              &kv_cache,
              use_optimized_linear,
              use_layout_cached_linear);
      current_values = layer_mlp_residual_resident_from_input(
          groups,
          attention_values,
          eps,
          use_optimized_linear,
          use_layout_cached_linear,
          use_mlp_pair_optimized,
          false,
          false,
          apply_mlx_resident_mlp_chain);
    }
    auto prompt_end = std::chrono::steady_clock::now();
    const double prompt_pass_timing_ms =
        std::chrono::duration<double, std::milli>(prompt_end - prompt_start).count();
    const std::uint64_t positions_before = kv_cache.positions_stored;

    ResidentDecodeResult incremental_result;
    incremental_result.positions_before = positions_before;
    auto incremental_start = std::chrono::steady_clock::now();
    current_values =
        layer0_embedding_values(model_dir_str, static_cast<std::size_t>(decode_token_id));
    for (std::uint64_t layer = 0; layer < layers; ++layer) {
      const ResidentLayerGroups& groups = resident.layers[static_cast<std::size_t>(layer)];
      std::vector<float> attention_values =
          layer_attention_residual_resident_incremental_from_input(
          groups,
          static_cast<std::size_t>(layer),
          current_values,
          eps,
            kv_cache,
            use_optimized_linear,
            use_layout_cached_linear,
            use_mlx_resident_layer_block,
            &incremental_result);
      current_values = layer_mlp_residual_resident_from_input(
          groups,
          attention_values,
          eps,
          use_optimized_linear,
          use_layout_cached_linear,
          use_mlp_pair_optimized,
          use_gate_up_full_block_optimized,
          use_down_full_block_optimized,
          false,
          &incremental_result);
    }
    auto final_norm_start = std::chrono::steady_clock::now();
    std::vector<float> final_norm_values = rmsnorm_values(final_norm, current_values, eps);
    add_elapsed_ms(incremental_result.final_norm_ms, final_norm_start);
    const double final_norm_checksum = vector_checksum(final_norm_values);
    std::uint64_t top_token_id = 0;
    float top_token_score = 0.0f;
    std::size_t logits_len = 0;
    auto logits_start = std::chrono::steady_clock::now();
    if (use_logits_top1_optimized && mlx_quantized_linear_available) {
      try {
        MlxQuantizedLinearStepScope logits_scope("logits/top1");
        std::vector<float> logits = quantized_linear_vector_values_mlx(embedding, final_norm_values);
        std::vector<std::pair<std::uint64_t, float>> top = top_logits(logits, 1);
        if (top.empty()) {
          output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"no logits produced\"}";
          return output.c_str();
        }
        top_token_id = top[0].first;
        top_token_score = top[0].second;
        logits_len = logits.size();
      } catch (...) {
        mlx_quantized_linear_runtime_fallback_used = true;
        auto top = quantized_linear_top1_layout_cached(embedding, final_norm_values);
        top_token_id = top.first;
        top_token_score = top.second;
        logits_len = embedding.quantized_output_len;
      }
    } else if (use_logits_top1_optimized) {
      auto top = quantized_linear_top1_layout_cached(embedding, final_norm_values);
      top_token_id = top.first;
      top_token_score = top.second;
      logits_len = embedding.quantized_output_len;
    } else {
      std::vector<float> logits = quantized_linear_vector_values_selected(
          embedding,
          final_norm_values,
          use_optimized_linear,
          use_layout_cached_linear);
      std::vector<std::pair<std::uint64_t, float>> top = top_logits(logits, 1);
      if (top.empty()) {
        output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"no logits produced\"}";
        return output.c_str();
      }
      top_token_id = top[0].first;
      top_token_score = top[0].second;
      logits_len = logits.size();
    }
    add_elapsed_ms(incremental_result.logits_projection_ms, logits_start);
    auto incremental_end = std::chrono::steady_clock::now();
    const double incremental_pass_timing_ms =
        std::chrono::duration<double, std::milli>(incremental_end - incremental_start).count();
    const auto decode_cleanup_start = std::chrono::steady_clock::now();
    const std::uint64_t positions_after = kv_cache.positions_stored;
    incremental_result.timing_ms = incremental_pass_timing_ms;
    incremental_result.positions_after = positions_after;
    incremental_result.final_norm_checksum = final_norm_checksum;
    incremental_result.top_token_id = top_token_id;
    incremental_result.top_token_score = top_token_score;
    incremental_result.logits_len = logits_len;
    update_largest_arithmetic_bucket(incremental_result);

    std::uint64_t k_length = 0;
    std::uint64_t v_length = 0;
    if (!kv_cache.layers.empty() &&
        !kv_cache.layers[0].keys.empty() &&
        !kv_cache.layers[0].values.empty()) {
      k_length = static_cast<std::uint64_t>(kv_cache.layers[0].keys[0].size());
      v_length = static_cast<std::uint64_t>(kv_cache.layers[0].values[0].size());
    }
    {
      std::lock_guard<std::mutex> lock(native_kv_cache_mutex());
      native_kv_cache_table()["session_layer_residency_probe"] = kv_cache;
    }
    const auto decode_cleanup_end = std::chrono::steady_clock::now();
    const double decode_cleanup_ms =
        std::chrono::duration<double, std::milli>(
            decode_cleanup_end - decode_cleanup_start).count();
    const double total_probe_ms =
        std::chrono::duration<double, std::milli>(
            decode_cleanup_end - total_probe_start).count();

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"prompt_token_id\":" << prompt_token_id << ","
             << "\"decode_token_id\":" << decode_token_id << ","
             << "\"layers_resident\":" << resident.layers.size() << ","
             << "\"total_groups_resident\":" << resident.total_groups << ","
             << "\"resident_total_byte_size\":" << resident.total_byte_size << ","
             << "\"resident_group_load_ms\":" << resident_group_load_ms << ","
             << "\"prompt_pass_ms\":" << prompt_pass_timing_ms << ","
             << "\"incremental_pass_ms\":" << incremental_pass_timing_ms << ","
             << "\"decode_cleanup_ms\":" << decode_cleanup_ms << ","
             << "\"total_probe_ms\":" << total_probe_ms << ","
             << "\"resident_groups_persistent_across_probe\":false,"
             << "\"prompt_pass_timing_ms\":" << prompt_pass_timing_ms << ","
             << "\"incremental_pass_timing_ms\":" << incremental_pass_timing_ms << ","
             << "\"tensor_group_load_count\":" << tensor_group_load_count << ","
             << "\"positions_before\":" << positions_before << ","
             << "\"positions_after\":" << positions_after << ","
             << "\"k_length\":" << k_length << ","
             << "\"v_length\":" << v_length << ","
             << "\"logits_len\":" << logits_len << ","
             << "\"top_token_id\":" << top_token_id << ","
             << "\"top_token_score\":" << top_token_score << ","
             << "\"final_norm_checksum\":" << final_norm_checksum << ","
             << "\"timing_buckets_ms\":" << resident_decode_timing_buckets_json(incremental_result) << ","
             << "\"projection_timing_breakdown_ms\":" << projection_timing_breakdown_json(incremental_result) << ","
             << "\"largest_arithmetic_bucket\":\"" << json_escape(incremental_result.largest_arithmetic_bucket) << "\","
             << "\"largest_arithmetic_bucket_ms\":" << incremental_result.largest_arithmetic_bucket_ms << ","
             << "\"backend_report\":" << backend_report_json(
                    incremental_result,
                    mlx_quantized_linear_available,
                    mlx_quantized_linear_fallback_steps) << ","
             << "\"fallback_steps\":" << fallback_steps_json() << ","
             << "\"resident_groups_reused\":true,"
             << "\"optimized_incremental_attention\":true,"
             << "\"optimized_path_applied_to_generation\":" << (use_optimized_linear ? "true" : "false") << ","
             << "\"layout_cached_path_applied_to_generation\":" << (use_layout_cached_linear ? "true" : "false") << ","
             << "\"mlp_pair_optimized_path_applied_to_generation\":" << (use_mlp_pair_optimized ? "true" : "false") << ","
             << "\"logits_top1_optimized_path_applied_to_generation\":" << (use_logits_top1_optimized ? "true" : "false") << ","
             << "\"cached_mlx_arrays_path_applied_to_generation\":" << (enable_cached_mlx_quantized_arrays() ? "true" : "false") << ","
             << "\"mlx_resident_mlp_chain_available\":" << (use_mlx_resident_mlp_chain ? "true" : "false") << ","
             << "\"mlx_resident_mlp_chain_applied_to_generation\":" << (apply_mlx_resident_mlp_chain ? "true" : "false") << ","
             << "\"mlx_resident_mlp_chain_layer0_compare\":{"
             << "\"ran\":" << (run_mlp_chain_compare ? "true" : "false") << ","
             << "\"ok\":" << (mlp_chain_compare_ok ? "true" : "false") << ","
             << "\"output_len\":" << mlp_chain_compare_output_len << ","
             << "\"checksum_current\":" << mlp_chain_compare_checksum_current << ","
             << "\"checksum_resident\":" << mlp_chain_compare_checksum_resident << ","
             << "\"max_abs_diff\":" << mlp_chain_compare_max_abs_diff << ","
             << "\"timing_current_ms\":" << mlp_chain_compare_timing_current_ms << ","
             << "\"timing_resident_ms\":" << mlp_chain_compare_timing_resident_ms
             << "},"
             << "\"mlx_quantized_linear_available\":" << (mlx_quantized_linear_available ? "true" : "false") << ","
             << "\"mlx_quantized_linear_applied_to_generation\":"
             << ((mlx_quantized_linear_available && !mlx_quantized_linear_runtime_fallback_used && use_optimized_linear) ? "true" : "false") << ","
             << "\"gate_up_full_block_optimized_path_applied_to_generation\":" << (use_gate_up_full_block_optimized ? "true" : "false") << ","
             << "\"down_full_block_optimized_path_applied_to_generation\":" << (use_down_full_block_optimized ? "true" : "false") << ","
             << "\"fallback_used\":" << ((fallback_used || mlx_quantized_linear_runtime_fallback_used) ? "true" : "false") << ","
             << "\"fallback_env\":\"RUSTY_FORCE_SCALAR_QUANTIZED_LINEAR/RUSTY_DISABLE_MLX_QUANTIZED_LINEAR\","
             << "\"provisional\":true,"
             << "\"notes\":["
             << "\"all layer tensor groups are resident for this verifier session\","
             << "\"prompt and incremental passes reuse resident layer groups\","
             << "\"layout-cached row-block quantized linear is the default resident decode path unless scalar fallback is forced\","
             << "\"resident MLP gate/up projections use a paired layout-cached path when enabled\","
             << "\"final tied embedding projection uses layout-cached top-1 selection when enabled\","
             << "\"down_proj uses a full-block layout-cached path when cached metadata proves no tail block is present\","
             << "\"this removes repeated tensor-group loads but remains CPU/provisional\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)prompt_token_id;
  (void)decode_token_id;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_fastsmoke_generation_probe_json(
    const char* model_dir,
    unsigned long long prompt_token_id,
    const char* prompt_token_ids_csv,
    unsigned long long first_decode_token_id,
    unsigned long long generated_tokens) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    const auto total_start = std::chrono::steady_clock::now();
    const std::uint64_t thread_count_start = current_process_thread_count();
    if (model_dir == nullptr || generated_tokens == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);
    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }
    std::uint64_t layers = 0;
    if (!extract_json_u64_value(config_json, "num_hidden_layers", layers) || layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"num_hidden_layers missing\"}";
      return output.c_str();
    }
    active_rope_config() = rope_config_from_json(config_json);
    std::vector<std::uint64_t> prompt_token_ids = parse_u64_csv(prompt_token_ids_csv);
    if (prompt_token_ids.empty()) {
      prompt_token_ids.push_back(static_cast<std::uint64_t>(prompt_token_id));
    }

    const bool fallback_used = force_scalar_quantized_linear();
    const bool use_optimized_linear = !fallback_used;
    const bool use_layout_cached_linear = use_optimized_linear;
    const bool use_mlp_pair_optimized = use_layout_cached_linear;
    const bool use_logits_top1_optimized = use_mlp_pair_optimized;
    const bool use_mlx_resident_mlp_chain =
        use_mlp_pair_optimized && !disable_mlx_resident_mlp_chain();
    const bool use_mlx_resident_layer_block =
        use_mlx_resident_mlp_chain && !disable_mlx_resident_layer_block();
    if (use_optimized_linear && !disable_mlx_quantized_linear()) {
      mlx_quantized_linear_available_flag() = true;
    }
    const bool mlx_quantized_linear_available =
        mlx_quantized_linear_available_flag() && !disable_mlx_quantized_linear();
    mlx_quantized_linear_runtime_fallback_used = false;
    mlx_quantized_linear_fallback_steps.clear();

    bool loaded_now = false;
    const auto resident_group_load_start = std::chrono::steady_clock::now();
    ResidentModelLayers& resident = resident_layers_for(model_dir_str, layers, loaded_now);
    const auto resident_group_load_end = std::chrono::steady_clock::now();
    const double resident_group_load_ms =
        std::chrono::duration<double, std::milli>(
            resident_group_load_end - resident_group_load_start).count();
    const std::uint64_t tensor_group_load_count = loaded_now ? resident.tensor_group_load_count : 0;

    TensorGroupRecord final_norm;
    auto final_norm_spec = group_load_spec_for("model.norm");
    if (!final_norm_spec || !load_tensor_group_record(model_dir_str, *final_norm_spec, final_norm)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load final norm\"}";
      return output.c_str();
    }
    TensorGroupRecord embedding;
    auto embedding_spec = embedding_group_spec_for();
    if (!embedding_spec || !load_tensor_group_record(model_dir_str, *embedding_spec, embedding)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load tied embedding projection\"}";
      return output.c_str();
    }
    const auto projection_warmup_start = std::chrono::steady_clock::now();
    ResidentProjectionWarmupTiming projection_warmup_timing;
    if (use_resident_mlx_projection_arrays()) {
      warm_resident_mlx_projection_arrays(resident, embedding, &projection_warmup_timing);
    }
    const double resident_projection_array_warmup_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - projection_warmup_start).count();
    const std::uint64_t thread_count_after_warmup = current_process_thread_count();

    bool mlp_chain_compare_ok = false;
    bool mlp_chain_compare_available = false;
    std::size_t mlp_chain_compare_output_len = 0;
    double mlp_chain_compare_checksum_current = 0.0;
    double mlp_chain_compare_checksum_resident = 0.0;
    double mlp_chain_compare_max_abs_diff = 0.0;
    double mlp_chain_compare_timing_current_ms = 0.0;
    double mlp_chain_compare_timing_resident_ms = 0.0;
    const bool run_mlp_chain_compare =
        enable_mlx_resident_mlp_chain_compare() && use_mlx_resident_mlp_chain && !resident.layers.empty();
    if (run_mlp_chain_compare) {
      try {
        const ResidentLayerGroups& layer0 = resident.layers[0];
        std::vector<float> compare_input(layer0.down_proj.quantized_rows, 0.0f);
        for (std::size_t i = 0; i < compare_input.size(); ++i) {
          compare_input[i] = static_cast<float>((static_cast<int>(i % 17) - 8)) / 8.0f;
        }
        ResidentDecodeResult current_timing;
        auto compare_current_start = std::chrono::steady_clock::now();
        std::vector<float> current_residual = layer_mlp_residual_resident_from_input(
            layer0,
            compare_input,
            eps,
            use_optimized_linear,
            use_layout_cached_linear,
            use_mlp_pair_optimized,
            false,
            false,
            false,
            &current_timing);
        mlp_chain_compare_timing_current_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - compare_current_start).count();
        ResidentDecodeResult chain_timing;
        auto compare_chain_start = std::chrono::steady_clock::now();
        std::vector<float> chain_residual = layer_mlp_residual_resident_from_input(
            layer0,
            compare_input,
            eps,
            use_optimized_linear,
            use_layout_cached_linear,
            use_mlp_pair_optimized,
            false,
            false,
            true,
            &chain_timing);
        mlp_chain_compare_timing_resident_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - compare_chain_start).count();
        if (current_residual.size() == chain_residual.size()) {
          mlp_chain_compare_output_len = chain_residual.size();
          for (std::size_t i = 0; i < current_residual.size(); ++i) {
            mlp_chain_compare_max_abs_diff = std::max(
                mlp_chain_compare_max_abs_diff,
                std::abs(static_cast<double>(current_residual[i]) - static_cast<double>(chain_residual[i])));
          }
          mlp_chain_compare_checksum_current = vector_checksum(current_residual);
          mlp_chain_compare_checksum_resident = vector_checksum(chain_residual);
          mlp_chain_compare_available = chain_timing.mlx_resident_mlp_chain_applied;
          mlp_chain_compare_ok = mlp_chain_compare_available && mlp_chain_compare_max_abs_diff <= 0.05;
        }
      } catch (...) {
        mlp_chain_compare_ok = false;
        mlp_chain_compare_available = false;
      }
      mlx_quantized_linear_runtime_fallback_used = false;
      mlx_quantized_linear_fallback_steps.clear();
    }
    const bool apply_mlx_resident_mlp_chain =
        use_mlx_resident_mlp_chain && (!run_mlp_chain_compare || mlp_chain_compare_ok);

    NativeSessionKvCache kv_cache;
    kv_cache.owner_session = "fastsmoke_generation_probe";
    kv_cache.layers.resize(static_cast<std::size_t>(layers));
    kv_cache.layers_allocated = layers;
    kv_cache.max_seq = static_cast<std::uint64_t>(
        prompt_token_ids.size() + static_cast<std::size_t>(generated_tokens) + 1);

    mlx_quantized_linear_runtime_fallback_used = false;
    mlx_quantized_linear_fallback_steps.clear();
    mlx_qk_norm_rope_applied_to_generation = false;
    mlx_qk_norm_rope_fallback_used = false;
    mlx_qk_norm_rope_verify_ran = false;
    mlx_qk_norm_rope_verify_max_abs_diff = 0.0;
    const auto prompt_start = std::chrono::steady_clock::now();
    ResidentDecodeResult prompt_result;
    ResidentDecodeValue current_value(mlx_array_from_vector(
        embedding_values_from_record(embedding, static_cast<std::size_t>(prompt_token_ids[0]))));
    for (std::size_t prompt_index = 0; prompt_index < prompt_token_ids.size(); ++prompt_index) {
      auto prompt_embedding_start = std::chrono::steady_clock::now();
      std::vector<float> prompt_embedding_values =
          embedding_values_from_record(embedding, static_cast<std::size_t>(prompt_token_ids[prompt_index]));
      add_elapsed_ms(prompt_result.embedding_ms, prompt_embedding_start);
      current_value = ResidentDecodeValue(mlx_array_from_vector(prompt_embedding_values));
      for (std::uint64_t layer = 0; layer < layers; ++layer) {
        const ResidentLayerGroups& groups = resident.layers[static_cast<std::size_t>(layer)];
        current_value = layer_decode_value_resident_incremental_from_input(
            groups,
            static_cast<std::size_t>(layer),
            current_value,
            eps,
            kv_cache,
            &prompt_result);
      }
    }
    const auto prompt_end = std::chrono::steady_clock::now();
    const double prompt_pass_ms =
        std::chrono::duration<double, std::milli>(prompt_end - prompt_start).count();
    prompt_result.timing_ms = prompt_pass_ms;
    prompt_result.positions_after = kv_cache.positions_stored;
    update_largest_arithmetic_bucket(prompt_result);
    std::vector<std::string> prompt_fallback_steps = mlx_quantized_linear_fallback_steps;
    const std::uint64_t prompt_readback_count =
        layers * static_cast<std::uint64_t>(prompt_token_ids.size());

    std::vector<std::uint64_t> generated_ids;
    std::vector<float> generated_scores;
    std::vector<double> per_token_incremental_ms;
    std::vector<std::vector<std::string>> fallback_steps_per_token;
    generated_ids.reserve(static_cast<std::size_t>(generated_tokens));
    generated_scores.reserve(static_cast<std::size_t>(generated_tokens));
    per_token_incremental_ms.reserve(static_cast<std::size_t>(generated_tokens));
    fallback_steps_per_token.reserve(static_cast<std::size_t>(generated_tokens));

    std::uint64_t decode_token = 0;
    const bool collect_generation_checksums = generation_checksum_diagnostics_enabled();
    double last_final_norm_checksum = 0.0;
    double first_generated_final_norm_checksum = 0.0;
    bool final_norm_checksum_available = false;
    bool first_generated_final_norm_checksum_available = false;
    std::size_t last_logits_len = 0;
    bool mlx_logits_topk_applied_to_generation = false;
    bool mlx_logits_topk_fallback_used = false;
    std::vector<std::pair<std::uint64_t, float>> first_generated_top_logits;
    bool stopped = false;
    std::string stop_reason = "max_tokens";
    std::uint64_t stop_token_id = 0;
    ResidentDecodeResult last_incremental_result;
    GenerationTimingBuckets generation_timing_buckets;
    std::vector<DecodeTimingSnapshot> diagnostic_snapshots;
    std::vector<RepetitionDiagnosticEntry> repetition_diagnostics;
    std::uint64_t repeated_token_streak_current = 0;
    std::uint64_t repeated_token_streak_max = 0;
    std::uint64_t low_id_token_count_so_far = 0;
    std::uint64_t numeric_token_count_so_far = 0;
    std::uint64_t previous_generated_token = std::numeric_limits<std::uint64_t>::max();
    auto should_capture_decode_snapshot = [](std::uint64_t token_position) {
      return token_position == 1 ||
             token_position == 8 ||
             token_position == 16 ||
             token_position == 32 ||
             token_position == 64 ||
             token_position == 128 ||
             token_position == 192 ||
             token_position == 256;
    };
    auto first_kv_lengths = [](const NativeSessionKvCache& cache) {
      std::pair<std::size_t, std::size_t> lengths{0, 0};
      if (!cache.layers.empty()) {
        const auto& layer = cache.layers[0];
        if (!layer.keys.empty()) {
          lengths.first = layer.keys.back().size();
        }
        if (!layer.values.empty()) {
          lengths.second = layer.values.back().size();
        }
      }
      return lengths;
    };
    const auto generation_start = std::chrono::steady_clock::now();
    const std::uint64_t thread_count_generation_start = current_process_thread_count();
    for (std::uint64_t step = 0; step < generated_tokens; ++step) {
      mlx_quantized_linear_runtime_fallback_used = false;
      mlx_quantized_linear_fallback_steps.clear();
      mlx_decode_readback_reasons.clear();
      ResidentDecodeResult incremental_result;
      incremental_result.positions_before = kv_cache.positions_stored;
      const auto incremental_start = std::chrono::steady_clock::now();
      if (step > 0) {
        auto decode_embedding_start = std::chrono::steady_clock::now();
        std::vector<float> decode_embedding_values =
            embedding_values_from_record(embedding, static_cast<std::size_t>(decode_token));
        add_elapsed_ms(incremental_result.embedding_ms, decode_embedding_start);
        current_value = ResidentDecodeValue(mlx_array_from_vector(decode_embedding_values));
        for (std::uint64_t layer = 0; layer < layers; ++layer) {
          const ResidentLayerGroups& groups = resident.layers[static_cast<std::size_t>(layer)];
          current_value = layer_decode_value_resident_incremental_from_input(
              groups,
              static_cast<std::size_t>(layer),
              current_value,
              eps,
              kv_cache,
              &incremental_result);
        }
      }
      auto final_norm_start = std::chrono::steady_clock::now();
      auto final_norm_mlx = rmsnorm_mlx_array(final_norm, current_value.mlx, eps);
      add_elapsed_ms(incremental_result.final_norm_ms, final_norm_start);
      std::vector<float> final_norm_values;
      bool final_norm_values_loaded = false;
      auto load_final_norm_values = [&](const std::string& reason) -> const std::vector<float>& {
        if (!final_norm_values_loaded) {
          final_norm_values = ResidentDecodeValue(final_norm_mlx).cpu(reason);
          final_norm_values_loaded = true;
        }
        return final_norm_values;
      };
      if (collect_generation_checksums) {
        const auto& checksum_values = load_final_norm_values("final_checksum_debug");
        last_final_norm_checksum = vector_checksum(checksum_values);
        final_norm_checksum_available = true;
        if (step == 0) {
          first_generated_final_norm_checksum = last_final_norm_checksum;
          first_generated_final_norm_checksum_available = true;
        }
      }

      std::uint64_t top_token_id = 0;
      float top_token_score = 0.0f;
      std::size_t logits_len = 0;
      bool token_selected = false;
      auto logits_start = std::chrono::steady_clock::now();
      if (experimental_mlx_logits_topk_enabled() &&
          use_logits_top1_optimized &&
          mlx_quantized_linear_available) {
        try {
          MlxQuantizedLinearStepScope logits_scope("logits/top1");
          auto logits_array = logits_array_from_mlx_final_norm(embedding, final_norm_mlx);
          auto selected = select_next_token_from_mlx_logits(logits_array);
          if (step == 0) {
            first_generated_top_logits = top_logits_from_mlx_array(logits_array, 10);
          }
          top_token_id = selected.first;
          top_token_score = selected.second;
          logits_len = static_cast<std::size_t>(logits_array.size());
          token_selected = true;
          mlx_logits_topk_applied_to_generation = true;
          record_mlx_decode_readback("logits/topk_ids_scores");
          if (top_token_id == previous_generated_token) {
            repeated_token_streak_current += 1;
          } else {
            repeated_token_streak_current = 1;
          }
          repeated_token_streak_max =
              std::max(repeated_token_streak_max, repeated_token_streak_current);
          previous_generated_token = top_token_id;
          if (top_token_id < 100) {
            low_id_token_count_so_far += 1;
          }
          if (token_is_numeric_or_low_id(top_token_id)) {
            numeric_token_count_so_far += 1;
          }
        } catch (...) {
          mlx_logits_topk_fallback_used = true;
          record_mlx_quantized_linear_fallback_step();
        }
      }
      if (!token_selected &&
          use_logits_top1_optimized && mlx_quantized_linear_available) {
        try {
          MlxQuantizedLinearStepScope logits_scope("logits/top1");
          const auto& logits_input_values =
              load_final_norm_values("final_logits_input_cpu_path");
          std::vector<float> logits = quantized_linear_vector_values_mlx(embedding, logits_input_values);
          auto selected = select_next_token_from_logits(logits);
          if (logits.empty()) {
            output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"no logits produced\"}";
            return output.c_str();
          }
          if (step == 0) {
            first_generated_top_logits = top_logits(logits, 10);
          }
          top_token_id = selected.first;
          top_token_score = selected.second;
          logits_len = logits.size();
          if (top_token_id == previous_generated_token) {
            repeated_token_streak_current += 1;
          } else {
            repeated_token_streak_current = 1;
          }
          repeated_token_streak_max =
              std::max(repeated_token_streak_max, repeated_token_streak_current);
          previous_generated_token = top_token_id;
          if (top_token_id < 100) {
            low_id_token_count_so_far += 1;
          }
          if (token_is_numeric_or_low_id(top_token_id)) {
            numeric_token_count_so_far += 1;
          }
          const std::uint64_t token_position_for_diag = step + 1;
          if (should_capture_repetition_diagnostic(token_position_for_diag, generated_tokens)) {
            repetition_diagnostics.push_back(make_repetition_diagnostic_entry(
                logits,
                top_token_id,
                token_position_for_diag,
                repeated_token_streak_current,
                repeated_token_streak_max,
                low_id_token_count_so_far,
                numeric_token_count_so_far));
          }
        } catch (...) {
          record_mlx_quantized_linear_fallback_step();
          const auto& logits_input_values =
              load_final_norm_values("final_logits_input_cpu_fallback");
          auto top = quantized_linear_top1_layout_cached(embedding, logits_input_values);
          top_token_id = top.first;
          top_token_score = top.second;
          logits_len = embedding.quantized_output_len;
          if (top_token_id == previous_generated_token) {
            repeated_token_streak_current += 1;
          } else {
            repeated_token_streak_current = 1;
          }
          repeated_token_streak_max =
              std::max(repeated_token_streak_max, repeated_token_streak_current);
          previous_generated_token = top_token_id;
          if (top_token_id < 100) {
            low_id_token_count_so_far += 1;
          }
          if (token_is_numeric_or_low_id(top_token_id)) {
            numeric_token_count_so_far += 1;
          }
        }
      } else if (!token_selected) {
        const auto& logits_input_values =
            load_final_norm_values("final_logits_input_cpu_fallback");
        auto top = quantized_linear_top1_layout_cached(embedding, logits_input_values);
        top_token_id = top.first;
        top_token_score = top.second;
        logits_len = embedding.quantized_output_len;
        if (top_token_id == previous_generated_token) {
          repeated_token_streak_current += 1;
        } else {
          repeated_token_streak_current = 1;
        }
        repeated_token_streak_max =
            std::max(repeated_token_streak_max, repeated_token_streak_current);
        previous_generated_token = top_token_id;
        if (top_token_id < 100) {
          low_id_token_count_so_far += 1;
        }
        if (token_is_numeric_or_low_id(top_token_id)) {
          numeric_token_count_so_far += 1;
        }
      }
      add_elapsed_ms(incremental_result.logits_projection_ms, logits_start);
      const auto incremental_end = std::chrono::steady_clock::now();
      const double incremental_ms =
          std::chrono::duration<double, std::milli>(
              incremental_end - incremental_start).count();
      incremental_result.timing_ms = incremental_ms;
      incremental_result.positions_after = kv_cache.positions_stored;
      incremental_result.final_norm_checksum = last_final_norm_checksum;
      incremental_result.top_token_id = top_token_id;
      incremental_result.top_token_score = top_token_score;
      incremental_result.logits_len = logits_len;
      update_largest_arithmetic_bucket(incremental_result);
      generation_timing_buckets.add(incremental_result);

      generated_ids.push_back(top_token_id);
      generated_scores.push_back(top_token_score);
      per_token_incremental_ms.push_back(incremental_ms);
      fallback_steps_per_token.push_back(mlx_quantized_linear_fallback_steps);
      const std::uint64_t token_position = step + 1;
      if (should_capture_decode_snapshot(token_position)) {
        const auto kv_lengths = first_kv_lengths(kv_cache);
        DecodeTimingSnapshot snapshot;
        snapshot.token_position = token_position;
        snapshot.sequence_length_at_token = kv_cache.positions_stored;
        snapshot.kv_layers_allocated = kv_cache.layers_allocated;
        snapshot.kv_positions_stored = kv_cache.positions_stored;
        snapshot.k_length = kv_lengths.first;
        snapshot.v_length = kv_lengths.second;
        snapshot.result = incremental_result;
        snapshot.fallback_steps = mlx_quantized_linear_fallback_steps;
        snapshot.readback_reasons = mlx_decode_readback_reasons;
        diagnostic_snapshots.push_back(std::move(snapshot));
      }
      last_logits_len = logits_len;
      last_incremental_result = incremental_result;
      std::string matched_stop_reason;
      if (token_matches_stop(top_token_id, matched_stop_reason)) {
        stopped = true;
        stop_reason = matched_stop_reason;
        stop_token_id = top_token_id;
        break;
      }
      decode_token = top_token_id;
    }
    const auto generation_end = std::chrono::steady_clock::now();
    const std::uint64_t thread_count_generation_end = current_process_thread_count();
    const double total_generation_ms =
        std::chrono::duration<double, std::milli>(generation_end - generation_start).count();

    const bool run_second_generation_proof = env_truthy("RUSTY_VERIFY_SECOND_GENERATION");
    ResidentProjectionWarmupTiming second_projection_warmup_timing;
    double second_resident_projection_array_warmup_ms = 0.0;
    double second_prompt_pass_ms = 0.0;
    double second_total_generation_ms = 0.0;
    std::vector<std::uint64_t> second_generated_ids;
    std::vector<float> second_generated_scores;
    std::vector<double> second_per_token_incremental_ms;
    std::vector<std::vector<std::string>> second_fallback_steps_per_token;
    double second_first_generated_final_norm_checksum = 0.0;
    std::size_t second_last_logits_len = 0;
    bool second_stopped = false;
    std::string second_stop_reason = "max_tokens";
    std::uint64_t second_stop_token_id = 0;

    if (run_second_generation_proof) {
      const auto second_warmup_start = std::chrono::steady_clock::now();
      if (use_resident_mlx_projection_arrays()) {
        warm_resident_mlx_projection_arrays(resident, embedding, &second_projection_warmup_timing);
      }
      second_resident_projection_array_warmup_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - second_warmup_start).count();

      NativeSessionKvCache second_kv_cache;
      second_kv_cache.owner_session = "fastsmoke_generation_probe_second";
      second_kv_cache.layers.resize(static_cast<std::size_t>(layers));
      second_kv_cache.layers_allocated = layers;
      second_kv_cache.max_seq = kv_cache.max_seq;
      auto second_prompt_start = std::chrono::steady_clock::now();
      ResidentDecodeResult second_prompt_result;
      ResidentDecodeValue second_current_value(mlx_array_from_vector(
          embedding_values_from_record(embedding, static_cast<std::size_t>(prompt_token_ids[0]))));
      for (std::size_t prompt_index = 0; prompt_index < prompt_token_ids.size(); ++prompt_index) {
        std::vector<float> second_prompt_embedding_values =
            embedding_values_from_record(embedding, static_cast<std::size_t>(prompt_token_ids[prompt_index]));
        second_current_value = ResidentDecodeValue(mlx_array_from_vector(second_prompt_embedding_values));
        for (std::uint64_t layer = 0; layer < layers; ++layer) {
          const ResidentLayerGroups& groups = resident.layers[static_cast<std::size_t>(layer)];
          second_current_value = layer_decode_value_resident_incremental_from_input(
              groups,
              static_cast<std::size_t>(layer),
              second_current_value,
              eps,
              second_kv_cache,
              &second_prompt_result);
        }
      }
      second_prompt_pass_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - second_prompt_start).count();

      second_generated_ids.reserve(static_cast<std::size_t>(generated_tokens));
      second_generated_scores.reserve(static_cast<std::size_t>(generated_tokens));
      second_per_token_incremental_ms.reserve(static_cast<std::size_t>(generated_tokens));
      second_fallback_steps_per_token.reserve(static_cast<std::size_t>(generated_tokens));
      std::uint64_t second_decode_token = 0;
      const auto second_generation_start = std::chrono::steady_clock::now();
      for (std::uint64_t step = 0; step < generated_tokens; ++step) {
        mlx_quantized_linear_runtime_fallback_used = false;
        mlx_quantized_linear_fallback_steps.clear();
        mlx_decode_readback_reasons.clear();
        ResidentDecodeResult second_incremental_result;
        second_incremental_result.positions_before = second_kv_cache.positions_stored;
        const auto second_incremental_start = std::chrono::steady_clock::now();
        if (step > 0) {
          std::vector<float> second_decode_embedding_values =
              embedding_values_from_record(embedding, static_cast<std::size_t>(second_decode_token));
          second_current_value = ResidentDecodeValue(mlx_array_from_vector(second_decode_embedding_values));
          for (std::uint64_t layer = 0; layer < layers; ++layer) {
            const ResidentLayerGroups& groups = resident.layers[static_cast<std::size_t>(layer)];
            second_current_value = layer_decode_value_resident_incremental_from_input(
                groups,
                static_cast<std::size_t>(layer),
                second_current_value,
                eps,
                second_kv_cache,
                &second_incremental_result);
          }
        }
        auto second_final_norm_mlx = rmsnorm_mlx_array(final_norm, second_current_value.mlx, eps);
        std::vector<float> second_final_norm_values =
            ResidentDecodeValue(second_final_norm_mlx).cpu("final_checksum_and_logits");
        const double second_final_norm_checksum = vector_checksum(second_final_norm_values);
        if (step == 0) {
          second_first_generated_final_norm_checksum = second_final_norm_checksum;
        }
        std::uint64_t second_top_token_id = 0;
        float second_top_token_score = 0.0f;
        std::size_t second_logits_len = 0;
        if (use_logits_top1_optimized && mlx_quantized_linear_available) {
          try {
            MlxQuantizedLinearStepScope logits_scope("logits/top1");
            std::vector<float> second_logits =
                quantized_linear_vector_values_mlx(embedding, second_final_norm_values);
            auto second_selected = select_next_token_from_logits(second_logits);
            if (second_logits.empty()) {
              output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"no logits produced\"}";
              return output.c_str();
            }
            second_top_token_id = second_selected.first;
            second_top_token_score = second_selected.second;
            second_logits_len = second_logits.size();
          } catch (...) {
            record_mlx_quantized_linear_fallback_step();
            auto second_top = quantized_linear_top1_layout_cached(embedding, second_final_norm_values);
            second_top_token_id = second_top.first;
            second_top_token_score = second_top.second;
            second_logits_len = embedding.quantized_output_len;
          }
        } else {
          auto second_top = quantized_linear_top1_layout_cached(embedding, second_final_norm_values);
          second_top_token_id = second_top.first;
          second_top_token_score = second_top.second;
          second_logits_len = embedding.quantized_output_len;
        }
        const double second_incremental_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - second_incremental_start).count();
        second_generated_ids.push_back(second_top_token_id);
        second_generated_scores.push_back(second_top_token_score);
        second_per_token_incremental_ms.push_back(second_incremental_ms);
        second_fallback_steps_per_token.push_back(mlx_quantized_linear_fallback_steps);
        std::string second_matched_stop_reason;
        if (token_matches_stop(second_top_token_id, second_matched_stop_reason)) {
          second_stopped = true;
          second_stop_reason = second_matched_stop_reason;
          second_stop_token_id = second_top_token_id;
          break;
        }
        second_decode_token = second_top_token_id;
        second_last_logits_len = second_logits_len;
      }
      const auto second_generation_end = std::chrono::steady_clock::now();
      second_total_generation_ms =
          std::chrono::duration<double, std::milli>(
              second_generation_end - second_generation_start).count();
    }
    const double total_probe_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start).count();
    const double tokens_per_second_after_resident_load =
        total_generation_ms > 0.0
            ? (static_cast<double>(generated_tokens) * 1000.0) / total_generation_ms
            : 0.0;
    const double tokens_per_second_including_resident_load =
        total_probe_ms > 0.0
            ? (static_cast<double>(generated_tokens) * 1000.0) / total_probe_ms
            : 0.0;
    const std::uint64_t expanded_kv_bytes_estimate =
        layers *
        active_rope_config().num_attention_heads *
        kv_cache.max_seq *
        active_rope_config().head_dim * 2ULL * sizeof(float);
    const std::uint64_t compact_kv_bytes_estimate =
        layers *
        active_rope_config().num_key_value_heads *
        kv_cache.max_seq *
        active_rope_config().head_dim * 2ULL * sizeof(float);
    const std::size_t chunk_size = expanded_kv_chunk_size();
    std::uint64_t chunked_allocated_chunks_total = 0;
    std::uint64_t chunked_layers_with_chunks = 0;
    std::uint64_t chunked_max_chunks_per_layer = 0;
    std::uint64_t compact_chunked_allocated_chunks_total = 0;
    std::uint64_t compact_chunked_layers_with_chunks = 0;
    std::uint64_t compact_chunked_max_chunks_per_layer = 0;
    std::uint64_t cache_object_len_max = 0;
    std::uint64_t cache_object_nbytes_total = 0;
    std::uint64_t cpu_compact_kv_mirror_bytes = 0;
    for (const auto& layer_cache : kv_cache.layers) {
      ChunkedExpandedKvCache cache_object(layer_cache);
      cache_object_len_max = std::max(cache_object_len_max, cache_object.len());
      cache_object_nbytes_total += cache_object.nbytes();
      for (const auto& key : layer_cache.keys) {
        cpu_compact_kv_mirror_bytes +=
            static_cast<std::uint64_t>(key.size() * sizeof(float));
      }
      for (const auto& value : layer_cache.values) {
        cpu_compact_kv_mirror_bytes +=
            static_cast<std::uint64_t>(value.size() * sizeof(float));
      }
      const std::uint64_t chunks =
          static_cast<std::uint64_t>(layer_cache.expanded_key_chunks.size());
      chunked_allocated_chunks_total += chunks;
      if (chunks > 0) {
        chunked_layers_with_chunks += 1;
      }
      chunked_max_chunks_per_layer = std::max(chunked_max_chunks_per_layer, chunks);
      const std::uint64_t compact_chunks =
          static_cast<std::uint64_t>(layer_cache.compact_key_chunks.size());
      compact_chunked_allocated_chunks_total += compact_chunks;
      if (compact_chunks > 0) {
        compact_chunked_layers_with_chunks += 1;
      }
      compact_chunked_max_chunks_per_layer =
          std::max(compact_chunked_max_chunks_per_layer, compact_chunks);
    }
    const std::uint64_t expanded_kv_chunk_bytes =
        active_rope_config().num_attention_heads *
        chunk_size *
        active_rope_config().head_dim * 2ULL * sizeof(float);
    const std::uint64_t compact_kv_chunk_bytes =
        active_rope_config().num_key_value_heads *
        chunk_size *
        active_rope_config().head_dim * 2ULL * sizeof(float);
    const std::uint64_t chunked_expanded_kv_active_bytes =
        chunked_allocated_chunks_total * expanded_kv_chunk_bytes;
    const std::uint64_t chunked_compact_mlx_reserved_bytes =
        compact_chunked_allocated_chunks_total * compact_kv_chunk_bytes;
    const std::uint64_t resident_projection_bytes_estimate =
        resident.total_byte_size + embedding.total_byte_size + final_norm.total_byte_size;
    const std::string active_attention_mode = experimental_mlx_attention_mode();
    const std::uint64_t active_kv_bytes_estimate =
        active_attention_mode == "expanded_kv"
            ? expanded_kv_bytes_estimate
            : (active_attention_mode == "chunked_expanded_kv"
                ? chunked_expanded_kv_active_bytes
                : (active_attention_mode == "chunked_compact_mlx"
                    ? chunked_compact_mlx_reserved_bytes
                    : compact_kv_bytes_estimate));
    const CachePolicy active_cache_policy{
        CachePolicyKind::Full,
        static_cast<std::uint64_t>(chunk_size),
        0,
        0,
        0,
        0,
        0,
        0,
        ""};
    const std::uint64_t logical_expanded_kv_bytes_active =
        layers *
        active_rope_config().num_attention_heads *
        kv_cache.positions_stored *
        active_rope_config().head_dim * 2ULL * sizeof(float);
    const std::uint64_t logical_compact_kv_bytes_active =
        layers *
        active_rope_config().num_key_value_heads *
        kv_cache.positions_stored *
        active_rope_config().head_dim * 2ULL * sizeof(float);
    const std::uint64_t kv_capacity_tokens =
        active_attention_mode == "chunked_expanded_kv"
            ? chunked_max_chunks_per_layer * static_cast<std::uint64_t>(chunk_size)
            : (active_attention_mode == "chunked_compact_mlx"
                ? compact_chunked_max_chunks_per_layer * static_cast<std::uint64_t>(chunk_size)
                : kv_cache.max_seq);
    const std::uint64_t kv_bytes_active =
        active_attention_mode == "chunked_expanded_kv"
            ? logical_expanded_kv_bytes_active
            : (active_attention_mode == "chunked_compact_mlx"
                ? logical_compact_kv_bytes_active
                : active_kv_bytes_estimate);
    const std::uint64_t kv_bytes_reserved =
        active_attention_mode == "chunked_expanded_kv"
            ? chunked_expanded_kv_active_bytes
            : (active_attention_mode == "chunked_compact_mlx"
                ? chunked_compact_mlx_reserved_bytes
                : active_kv_bytes_estimate);
    const bool cpu_compact_kv_mirror_enabled = cpu_compact_kv_mirror_bytes > 0;
    const std::uint64_t cache_total_bytes =
        active_attention_mode == "chunked_expanded_kv"
            ? chunked_expanded_kv_active_bytes + cpu_compact_kv_mirror_bytes
            : (active_attention_mode == "chunked_compact_mlx"
                ? chunked_compact_mlx_reserved_bytes + cpu_compact_kv_mirror_bytes
                : cache_object_nbytes_total);
    const std::uint64_t total_estimated_model_runtime_bytes =
        resident.total_byte_size +
        resident_projection_bytes_estimate +
        active_kv_bytes_estimate;

    auto known_token_text = [](std::uint64_t token) {
      return token == 24 ? std::string("9") : std::to_string(token);
    };
    std::ostringstream decoded_text;
    for (std::size_t i = 0; i < generated_ids.size(); ++i) {
      if (i > 0) {
        decoded_text << " ";
      }
      decoded_text << known_token_text(generated_ids[i]);
    }
    std::ostringstream second_decoded_text;
    for (std::size_t i = 0; i < second_generated_ids.size(); ++i) {
      if (i > 0) {
        second_decoded_text << " ";
      }
      second_decoded_text << known_token_text(second_generated_ids[i]);
    }

    auto fallback_steps_json_for = [](const std::vector<std::vector<std::string>>& steps_by_token) {
      std::ostringstream out;
      out << "[";
      for (std::size_t i = 0; i < steps_by_token.size(); ++i) {
        if (i > 0) {
          out << ",";
        }
        out << "[";
        for (std::size_t j = 0; j < steps_by_token[i].size(); ++j) {
          if (j > 0) {
            out << ",";
          }
          out << "\"" << json_escape(steps_by_token[i][j]) << "\"";
        }
        out << "]";
      }
      out << "]";
      return out.str();
    };
    auto fallback_steps_per_token_json = [&]() {
      return fallback_steps_json_for(fallback_steps_per_token);
    };
    auto diagnostic_snapshots_json = [&]() {
      std::ostringstream out;
      out << "[";
      for (std::size_t i = 0; i < diagnostic_snapshots.size(); ++i) {
        if (i > 0) {
          out << ",";
        }
        out << decode_timing_snapshot_json(
            diagnostic_snapshots[i],
            mlx_quantized_linear_available);
      }
      out << "]";
      return out.str();
    };
    auto second_fallback_steps_per_token_json = [&]() {
      return fallback_steps_json_for(second_fallback_steps_per_token);
    };

    auto readback_reasons_json = [&]() {
      std::ostringstream out;
      out << "[";
      for (std::size_t i = 0; i < mlx_decode_readback_reasons.size(); ++i) {
        if (i > 0) {
          out << ",";
        }
        out << "\"" << json_escape(mlx_decode_readback_reasons[i]) << "\"";
      }
      out << "]";
      return out.str();
    };
    auto cache_stats_json = [&]() {
      std::ostringstream out;
      out << "{"
          << "\"cache_policy\":" << cache_policy_to_json(active_cache_policy) << ","
          << "\"cache_object\":\"ChunkedExpandedKvCache\","
          << "\"cache_backend\":\""
          << (active_attention_mode == "0" ? "compact_cpu" : json_escape(active_attention_mode))
          << "\","
          << "\"kv_len\":" << kv_cache.positions_stored << ","
          << "\"kv_len_from_layer_cache_interface\":" << cache_object_len_max << ","
          << "\"kv_capacity\":" << kv_capacity_tokens << ","
          << "\"kv_bytes_active\":" << kv_bytes_active << ","
          << "\"kv_bytes_reserved\":" << kv_bytes_reserved << ","
          << "\"kv_bytes_from_layer_cache_interface\":" << cache_object_nbytes_total << ","
          << "\"mlx_expanded_kv_active_bytes\":" << logical_expanded_kv_bytes_active << ","
          << "\"mlx_expanded_kv_reserved_bytes\":" << chunked_expanded_kv_active_bytes << ","
          << "\"compact_mlx_kv_active_bytes\":" << logical_compact_kv_bytes_active << ","
          << "\"compact_mlx_kv_reserved_bytes\":" << chunked_compact_mlx_reserved_bytes << ","
          << "\"compact_mlx_allocated_chunks_total\":" << compact_chunked_allocated_chunks_total << ","
          << "\"compact_mlx_layers_with_chunks\":" << compact_chunked_layers_with_chunks << ","
          << "\"compact_mlx_max_chunks_per_layer\":" << compact_chunked_max_chunks_per_layer << ","
          << "\"compact_mlx_chunk_bytes\":" << compact_kv_chunk_bytes << ","
          << "\"cpu_compact_kv_mirror_bytes\":" << cpu_compact_kv_mirror_bytes << ","
          << "\"cpu_compact_kv_mirror_enabled\":"
          << (cpu_compact_kv_mirror_enabled ? "true" : "false") << ","
          << "\"cpu_compact_kv_mirror_policy\":\""
          << (keep_cpu_kv_mirror()
                ? "enabled_by_backend_or_RUSTY_KEEP_CPU_KV_MIRROR"
                : "disabled_for_mlx_expanded_kv_backends")
          << "\","
          << "\"cache_total_bytes\":" << cache_total_bytes << ","
          << "\"chunks_allocated\":" << chunked_allocated_chunks_total << ","
          << "\"chunk_size\":" << chunk_size << ","
          << "\"layer_count\":" << layers << ","
          << "\"q_heads\":" << active_rope_config().num_attention_heads << ","
          << "\"kv_heads\":" << active_rope_config().num_key_value_heads << ","
          << "\"head_dim\":" << active_rope_config().head_dim << ","
          << "\"arena_active_bytes\":0,"
          << "\"arena_cached_bytes\":0,"
          << "\"arena_peak_bytes\":0,"
          << "\"arena_accounting\":\"not_implemented\""
          << "}";
      return out.str();
    };

    std::vector<std::string> cpu_fallback_steps;
    for (const auto& step : mlx_quantized_linear_fallback_steps) {
      cpu_fallback_steps.push_back(step);
    }
    auto cpu_fallback_steps_json = [&]() {
      std::ostringstream out;
      out << "[";
      for (std::size_t i = 0; i < cpu_fallback_steps.size(); ++i) {
        if (i > 0) {
          out << ",";
        }
        out << "\"" << json_escape(cpu_fallback_steps[i]) << "\"";
      }
      out << "]";
      return out.str();
    };

    const bool any_fallback = std::any_of(
        fallback_steps_per_token.begin(),
        fallback_steps_per_token.end(),
        [](const auto& steps) { return !steps.empty(); });
    std::uint64_t dominant_tail_token_id = 0;
    std::uint64_t dominant_tail_token_count = 0;
    std::uint64_t eos_best_rank_seen = 0;
    std::uint64_t eos_final_rank = 0;
    double entropy_min = 0.0;
    double entropy_final = 0.0;
    bool collapse_detected = false;
    std::uint64_t first_collapse_token_index = 0;
    std::string likely_cause = "unknown";
    if (!repetition_diagnostics.empty()) {
      entropy_min = repetition_diagnostics.front().entropy;
      entropy_final = repetition_diagnostics.back().entropy;
      eos_final_rank = repetition_diagnostics.back().eos_rank;
      for (const auto& entry : repetition_diagnostics) {
        entropy_min = std::min(entropy_min, entry.entropy);
        if (entry.eos_rank > 0 &&
            (eos_best_rank_seen == 0 || entry.eos_rank < eos_best_rank_seen)) {
          eos_best_rank_seen = entry.eos_rank;
        }
        if (!collapse_detected &&
            (entry.repeated_token_streak_current >= 16 ||
             entry.low_id_token_count_so_far > entry.token_position / 2)) {
          collapse_detected = true;
          first_collapse_token_index = entry.token_position;
        }
      }
      const std::size_t tail_start =
          generated_ids.size() > 128 ? generated_ids.size() - 128 : 0;
      std::unordered_map<std::uint64_t, std::uint64_t> tail_counts;
      for (std::size_t i = tail_start; i < generated_ids.size(); ++i) {
        const std::uint64_t count = ++tail_counts[generated_ids[i]];
        if (count > dominant_tail_token_count) {
          dominant_tail_token_count = count;
          dominant_tail_token_id = generated_ids[i];
        }
      }
      if (collapse_detected && repeated_token_streak_max >= 16) {
        likely_cause = "repetition_penalty";
      } else if (collapse_detected && eos_final_rank > 1000) {
        likely_cause = "eos_buried_base_or_adapter";
      } else if (collapse_detected && generation_sampling_config.enabled) {
        likely_cause = "sampling_temperature";
      }
    }
    auto collapse_summary_json = [&]() {
      std::ostringstream out;
      out << "{"
          << "\"collapse_detected\":" << (collapse_detected ? "true" : "false") << ","
          << "\"likely_cause\":\"" << json_escape(likely_cause) << "\","
          << "\"first_collapse_token_index\":" << first_collapse_token_index << ","
          << "\"dominant_tail_token_id\":" << dominant_tail_token_id << ","
          << "\"dominant_tail_token_count\":" << dominant_tail_token_count << ","
          << "\"eos_best_rank_seen\":" << eos_best_rank_seen << ","
          << "\"eos_final_rank\":" << eos_final_rank << ","
          << "\"entropy_min\":" << entropy_min << ","
          << "\"entropy_final\":" << entropy_final << ","
          << "\"max_repeated_streak\":" << repeated_token_streak_max
          << "}";
      return out.str();
    };
    std::vector<std::string> resident_block_fallback_reasons;
    for (const auto& snapshot : diagnostic_snapshots) {
      for (const auto& reason : snapshot.result.per_layer_attention_fallback_reasons) {
        if (!reason.empty()) {
          resident_block_fallback_reasons.push_back(reason);
        }
      }
    }
    const std::uint64_t resident_block_fallback_count =
        static_cast<std::uint64_t>(resident_block_fallback_reasons.size());
    const bool full_resident_block_gate_enabled = experimental_mlx_resident_block();
    const bool full_resident_block_applied_to_generation =
        full_resident_block_gate_enabled &&
        last_incremental_result.mlx_resident_layer_block_applied &&
        resident_block_fallback_count == 0;
    const bool partial_resident_layer_block_applied =
        !full_resident_block_gate_enabled &&
        last_incremental_result.mlx_resident_layer_block_applied;
    const ResidentLoadTiming& load_timing = resident.load_timing;
    auto resident_load_timing_json = [&]() {
      std::ostringstream out;
      out << "{"
          << "\"safetensor_index_lookup_ms\":" << load_timing.safetensor_index_lookup_ms << ","
          << "\"tensor_group_metadata_construction_ms\":"
          << load_timing.tensor_group_metadata_construction_ms << ","
          << "\"mmap_file_read_ms\":" << load_timing.mmap_file_read_ms << ","
          << "\"mlx_array_creation_ms\":" << load_timing.mlx_array_creation_ms << ","
          << "\"quantized_group_preparation_ms\":"
          << load_timing.quantized_group_preparation_ms << ","
          << "\"synchronization_eval_ms\":" << load_timing.synchronization_eval_ms
          << "}";
      return out.str();
    };
    auto projection_warmup_timing_json = [](const ResidentProjectionWarmupTiming& timing) {
      std::ostringstream out;
      out << "{"
          << "\"enumerate_groups_ms\":" << timing.enumerate_groups_ms << ","
          << "\"mmap_setup_ms\":" << timing.mmap_setup_ms << ","
          << "\"mlx_array_construction_ms\":" << timing.mlx_array_construction_ms << ","
          << "\"first_eval_compile_warmup_ms\":" << timing.first_eval_compile_warmup_ms << ","
          << "\"metadata_cache_storage_ms\":" << timing.metadata_cache_storage_ms
          << "}";
      return out.str();
    };
    auto kv_positions_per_layer_json = [](const NativeSessionKvCache& cache) {
      std::vector<std::uint64_t> positions;
      positions.reserve(cache.layers.size());
      for (const auto& layer_cache : cache.layers) {
        positions.push_back(static_cast<std::uint64_t>(layer_cache.keys.size()));
      }
      return json_array_u64_to_string(positions);
    };
    auto attention_scores_probs_head0_json = [](const std::vector<float>& q_values,
                                                const NativeLayerKvCache& cache,
                                                const std::vector<float>& current_k_values) {
      const std::size_t head_dim = 128;
      std::vector<float> scores;
      std::vector<float> probs;
      if (q_values.size() < head_dim || current_k_values.size() < head_dim) {
        return std::string("{\"scores\":{\"shape\":[0],\"checksum\":0,\"first_values\":[]},\"probabilities\":{\"shape\":[0],\"checksum\":0,\"first_values\":[]}}");
      }
      const std::size_t total_positions = cache.keys.size() + 1;
      scores.reserve(total_positions);
      double max_score = -std::numeric_limits<double>::infinity();
      const double inv_sqrt_dim = 1.0 / std::sqrt(static_cast<double>(head_dim));
      for (std::size_t position = 0; position < total_positions; ++position) {
        const std::vector<float>& key = position < cache.keys.size() ? cache.keys[position] : current_k_values;
        double score = 0.0;
        for (std::size_t i = 0; i < head_dim; ++i) {
          score += static_cast<double>(q_values[i]) * static_cast<double>(key[i]);
        }
        score *= inv_sqrt_dim;
        scores.push_back(static_cast<float>(score));
        max_score = std::max(max_score, score);
      }
      double denom = 0.0;
      probs.resize(scores.size(), 0.0f);
      for (std::size_t i = 0; i < scores.size(); ++i) {
        const double value = std::exp(static_cast<double>(scores[i]) - max_score);
        probs[i] = static_cast<float>(value);
        denom += value;
      }
      if (denom > 0.0) {
        for (float& value : probs) {
          value = static_cast<float>(static_cast<double>(value) / denom);
        }
      }
      std::ostringstream out;
      out << "{"
          << "\"scores\":" << parity_vector_summary_json(scores) << ","
          << "\"probabilities\":" << parity_vector_summary_json(probs)
          << "}";
      return out.str();
    };
    auto attention_head_comparison_json = [](const std::vector<float>& cpu_values,
                                             const std::vector<float>& mlx_values,
                                             std::size_t q_heads,
                                             std::size_t kv_heads,
                                             std::size_t head_dim) {
      std::ostringstream out;
      out << "[";
      if (q_heads == 0 || kv_heads == 0 || head_dim == 0 || q_heads % kv_heads != 0 ||
          cpu_values.size() < q_heads * head_dim || mlx_values.size() < q_heads * head_dim) {
        out << "]";
        return out.str();
      }
      const std::size_t q_heads_per_kv = q_heads / kv_heads;
      for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
        if (q_head > 0) {
          out << ",";
        }
        const std::size_t kv_head = q_head / q_heads_per_kv;
        double max_diff = 0.0;
        double cpu_checksum = 0.0;
        double mlx_checksum = 0.0;
        std::vector<float> cpu_first;
        std::vector<float> mlx_first;
        for (std::size_t i = 0; i < head_dim; ++i) {
          const std::size_t offset = q_head * head_dim + i;
          const float cpu = cpu_values[offset];
          const float mlx = mlx_values[offset];
          cpu_checksum += static_cast<double>(cpu);
          mlx_checksum += static_cast<double>(mlx);
          max_diff = std::max(max_diff, std::abs(static_cast<double>(cpu) - static_cast<double>(mlx)));
          if (i < 8) {
            cpu_first.push_back(cpu);
            mlx_first.push_back(mlx);
          }
        }
        out << "{"
            << "\"head_index\":" << q_head << ","
            << "\"kv_head_index\":" << kv_head << ","
            << "\"cpu_output_checksum\":" << cpu_checksum << ","
            << "\"mlx_output_checksum\":" << mlx_checksum << ","
            << "\"max_abs_diff\":" << max_diff << ","
            << "\"first8_cpu\":" << json_array_float_to_string(cpu_first) << ","
            << "\"first8_mlx\":" << json_array_float_to_string(mlx_first)
            << "}";
      }
      out << "]";
      return out.str();
    };
    auto attention_head_summary_json = [](const std::vector<float>& cpu_values,
                                          const std::vector<float>& mlx_values,
                                          std::size_t q_heads,
                                          std::size_t kv_heads,
                                          std::size_t head_dim) {
      std::size_t worst_head = 0;
      std::size_t worst_kv_head = 0;
      std::size_t first_mismatch_index = 0;
      double worst_diff = 0.0;
      double overall_diff = vector_max_abs_diff(cpu_values, mlx_values);
      if (q_heads > 0 && kv_heads > 0 && head_dim > 0 && q_heads % kv_heads == 0 &&
          cpu_values.size() >= q_heads * head_dim && mlx_values.size() >= q_heads * head_dim) {
        const std::size_t q_heads_per_kv = q_heads / kv_heads;
        bool mismatch_recorded = false;
        for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
          double head_diff = 0.0;
          for (std::size_t i = 0; i < head_dim; ++i) {
            const std::size_t offset = q_head * head_dim + i;
            const double diff = std::abs(
                static_cast<double>(cpu_values[offset]) - static_cast<double>(mlx_values[offset]));
            if (!mismatch_recorded && diff > 1e-6) {
              first_mismatch_index = offset;
              mismatch_recorded = true;
            }
            head_diff = std::max(head_diff, diff);
          }
          if (head_diff > worst_diff) {
            worst_diff = head_diff;
            worst_head = q_head;
            worst_kv_head = q_head / q_heads_per_kv;
          }
        }
      }
      std::ostringstream out;
      out << "{"
          << "\"overall_max_abs_diff\":" << overall_diff << ","
          << "\"worst_head\":" << worst_head << ","
          << "\"worst_kv_head\":" << worst_kv_head << ","
          << "\"worst_diff\":" << worst_diff << ","
          << "\"first_mismatch_index\":" << first_mismatch_index << ","
          << "\"first_mismatch_head\":" << (head_dim == 0 ? 0 : first_mismatch_index / head_dim)
          << ","
          << "\"first_mismatch_head_offset\":" << (head_dim == 0 ? 0 : first_mismatch_index % head_dim)
          << "}";
      return out.str();
    };
    std::string layer0_parity_checkpoints_json = "{\"available\":false}";
    try {
      const ResidentLayerGroups& layer0 = resident.layers[0];
      NativeSessionKvCache debug_cache;
      debug_cache.owner_session = "parity_layer0_checkpoint";
      debug_cache.layers.resize(1);
      debug_cache.layers_allocated = 1;
      std::vector<float> token0_embedding;
      std::vector<float> final_embedding;
      std::vector<float> final_norm;
      std::vector<float> final_q;
      std::vector<float> final_k;
      std::vector<float> final_q_after_q_norm;
      std::vector<float> final_k_after_k_norm;
      std::vector<float> final_q_after_rope;
      std::vector<float> final_k_after_rope;
      std::vector<float> final_q_after_rope_interleaved;
      std::vector<float> final_k_after_rope_interleaved;
      std::vector<float> final_v;
      std::vector<float> final_attention;
      std::vector<float> final_attention_mlx;
      std::vector<float> final_attention_mlx_grouped;
      std::vector<float> final_attention_mlx_all_head_batched;
      std::vector<float> final_attention_mlx_resident_kv;
      std::vector<float> final_attention_mlx_expanded_resident_kv;
      std::vector<float> final_attention_mlx_fast;
      std::vector<float> final_o;
      std::vector<float> final_attention_residual;
      std::vector<float> final_post_norm;
      std::vector<float> final_mlp_down;
      std::vector<float> final_residual;
      std::string final_attention_scores_probs = "{}";
      bool mlx_attention_probe_available = false;
      double mlx_attention_probe_timing_ms = 0.0;
      double mlx_attention_probe_max_abs_diff = 0.0;
      std::string mlx_attention_probe_error;
      std::size_t mlx_attention_probe_seq_len = 0;
      std::size_t cpu_attention_seq_len = 0;
      std::size_t mlx_attention_seq_len = 0;
      bool cache_appended_before_cpu_attention = false;
      bool cache_appended_before_mlx_attention = false;
      double cpu_attention_probe_timing_ms = 0.0;
      double mlx_grouped_attention_probe_timing_ms = 0.0;
      double mlx_grouped_attention_probe_max_abs_diff = 0.0;
      std::string mlx_grouped_attention_probe_error;
      double mlx_all_head_attention_probe_timing_ms = 0.0;
      double mlx_all_head_attention_probe_max_abs_diff = 0.0;
      std::string mlx_all_head_attention_probe_error;
      double mlx_resident_kv_attention_probe_timing_ms = 0.0;
      double mlx_resident_kv_attention_probe_max_abs_diff = 0.0;
      double mlx_resident_kv_final_append_ms = 0.0;
      std::string mlx_resident_kv_attention_probe_error;
      double mlx_expanded_resident_kv_attention_probe_timing_ms = 0.0;
      double mlx_expanded_resident_kv_attention_probe_max_abs_diff = 0.0;
      double mlx_expanded_resident_kv_final_append_ms = 0.0;
      std::string mlx_expanded_resident_kv_attention_probe_error;
      double mlx_fast_attention_probe_timing_ms = 0.0;
      double mlx_fast_attention_probe_max_abs_diff = 0.0;
      std::string mlx_fast_attention_probe_error;
      double mlx_compact_concat_attention_probe_timing_ms = 0.0;
      double mlx_compact_concat_attention_probe_max_abs_diff = 0.0;
      double mlx_compact_chunk_aware_attention_probe_timing_ms = 0.0;
      double mlx_compact_chunk_aware_attention_probe_max_abs_diff = 0.0;
      std::string mlx_compact_attention_probe_error;
      std::vector<float> final_attention_mlx_compact_concat;
      std::vector<float> final_attention_mlx_compact_chunk_aware;
      MlxAttentionDetailTiming mlx_compact_concat_attention_detail;
      MlxAttentionDetailTiming mlx_compact_chunk_aware_attention_detail;
      std::string mlx_attention_worst_head_diagnostic = "{\"available\":false}";
      MlxResidentKvCacheProbe mlx_layer0_kv_cache_probe;
      MlxExpandedResidentKvCacheProbe mlx_layer0_expanded_kv_cache_probe;
      NativeLayerKvCache mlx_layer0_compact_chunk_cache_probe;
      for (std::size_t prompt_index = 0; prompt_index < prompt_token_ids.size(); ++prompt_index) {
        std::vector<float> input_values =
            embedding_values_from_record(embedding, static_cast<std::size_t>(prompt_token_ids[prompt_index]));
        if (prompt_index == 0) {
          token0_embedding = input_values;
        }
        std::vector<float> norm_values = rmsnorm_values(layer0.input_norm, input_values, eps);
        std::vector<float> q_values = quantized_linear_vector_values_layout_cached(layer0.q_proj, norm_values);
        std::vector<float> k_values = quantized_linear_vector_values_layout_cached(layer0.k_proj, norm_values);
        std::vector<float> v_values = quantized_linear_vector_values_layout_cached(layer0.v_proj, norm_values);
        std::vector<float> q_raw_values = q_values;
        std::vector<float> k_raw_values = k_values;
        apply_qk_norm_in_place(layer0.q_norm, layer0.k_norm, q_values, k_values, eps);
        std::vector<float> q_after_q_norm_values = q_values;
        std::vector<float> k_after_k_norm_values = k_values;
        apply_active_rope_to_qk(q_values, k_values, static_cast<std::uint64_t>(prompt_index));
        std::string attention_scores_probs =
            attention_scores_probs_head0_json(q_values, debug_cache.layers[0], k_values);
        CachedAttentionTiming cpu_attention_timing;
        auto cpu_attention_start = std::chrono::steady_clock::now();
        std::vector<float> attention_values =
            cached_single_token_attention_values(
                q_values,
                debug_cache.layers[0],
                k_values,
                v_values,
                &cpu_attention_timing);
        const double current_cpu_attention_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cpu_attention_start).count();
        if (prompt_index + 1 == prompt_token_ids.size()) {
          mlx_attention_probe_seq_len = debug_cache.layers[0].keys.size() + 1;
          cpu_attention_seq_len = debug_cache.layers[0].keys.size() + 1;
          mlx_attention_seq_len = mlx_attention_probe_seq_len;
          cpu_attention_probe_timing_ms = current_cpu_attention_ms;
          try {
            final_attention_mlx = mlx_single_token_attention_values_probe(
                q_values,
                debug_cache.layers[0],
                k_values,
                v_values,
                &mlx_attention_probe_timing_ms);
            mlx_attention_probe_available = true;
            mlx_attention_probe_max_abs_diff =
                vector_max_abs_diff(final_attention_mlx, attention_values);
            mlx_attention_worst_head_diagnostic =
                mlx_attention_worst_head_diagnostic_json(
                    q_values,
                    debug_cache.layers[0],
                    k_values,
                    v_values,
                    8,
                    2);
          } catch (const std::exception& e) {
            mlx_attention_probe_error = e.what();
          } catch (...) {
            mlx_attention_probe_error = "unknown_exception";
          }
          try {
            final_attention_mlx_grouped = mlx_grouped_attention_values_probe(
                q_values,
                debug_cache.layers[0],
                k_values,
                v_values,
                &mlx_grouped_attention_probe_timing_ms);
            mlx_grouped_attention_probe_max_abs_diff =
                vector_max_abs_diff(final_attention_mlx_grouped, attention_values);
          } catch (const std::exception& e) {
            mlx_grouped_attention_probe_error = e.what();
          } catch (...) {
            mlx_grouped_attention_probe_error = "unknown_exception";
          }
          try {
            final_attention_mlx_all_head_batched = mlx_all_head_batched_attention_values_probe(
                q_values,
                debug_cache.layers[0],
                k_values,
                v_values,
                &mlx_all_head_attention_probe_timing_ms);
            mlx_all_head_attention_probe_max_abs_diff =
                vector_max_abs_diff(final_attention_mlx_all_head_batched, attention_values);
          } catch (const std::exception& e) {
            mlx_all_head_attention_probe_error = e.what();
          } catch (...) {
            mlx_all_head_attention_probe_error = "unknown_exception";
          }
          try {
            const double append_before = mlx_layer0_kv_cache_probe.append_ms;
            mlx_resident_kv_cache_append_probe(
                mlx_layer0_kv_cache_probe,
                k_values,
                v_values,
                active_rope_config().num_key_value_heads,
                active_rope_config().head_dim);
            mlx_resident_kv_final_append_ms = mlx_layer0_kv_cache_probe.append_ms - append_before;
            final_attention_mlx_resident_kv = mlx_resident_kv_attention_values_probe(
                q_values,
                mlx_layer0_kv_cache_probe,
                active_rope_config().num_attention_heads,
                active_rope_config().num_key_value_heads,
                active_rope_config().head_dim,
                &mlx_resident_kv_attention_probe_timing_ms);
            mlx_resident_kv_attention_probe_max_abs_diff =
                vector_max_abs_diff(final_attention_mlx_resident_kv, attention_values);
          } catch (const std::exception& e) {
            mlx_resident_kv_attention_probe_error = e.what();
          } catch (...) {
            mlx_resident_kv_attention_probe_error = "unknown_exception";
          }
          try {
            const double append_before = mlx_layer0_expanded_kv_cache_probe.append_ms;
            mlx_expanded_resident_kv_cache_append_probe(
                mlx_layer0_expanded_kv_cache_probe,
                k_values,
                v_values,
                active_rope_config().num_attention_heads,
                active_rope_config().num_key_value_heads,
                active_rope_config().head_dim);
            mlx_expanded_resident_kv_final_append_ms =
                mlx_layer0_expanded_kv_cache_probe.append_ms - append_before;
            final_attention_mlx_expanded_resident_kv =
                mlx_expanded_resident_kv_attention_values_probe(
                    q_values,
                    mlx_layer0_expanded_kv_cache_probe,
                    active_rope_config().num_attention_heads,
                    active_rope_config().head_dim,
                    &mlx_expanded_resident_kv_attention_probe_timing_ms);
            mlx_expanded_resident_kv_attention_probe_max_abs_diff =
                vector_max_abs_diff(final_attention_mlx_expanded_resident_kv, attention_values);
          } catch (const std::exception& e) {
            mlx_expanded_resident_kv_attention_probe_error = e.what();
          } catch (...) {
            mlx_expanded_resident_kv_attention_probe_error = "unknown_exception";
          }
          try {
            final_attention_mlx_fast = mlx_fast_attention_values_probe(
                q_values,
                debug_cache.layers[0],
                k_values,
                v_values,
                &mlx_fast_attention_probe_timing_ms);
            mlx_fast_attention_probe_max_abs_diff =
                vector_max_abs_diff(final_attention_mlx_fast, attention_values);
          } catch (const std::exception& e) {
            mlx_fast_attention_probe_error = e.what();
          } catch (...) {
            mlx_fast_attention_probe_error = "unknown_exception";
          }
          try {
            double compact_append_ms = 0.0;
            mlx_compact_kv_cache_append_chunked(
                mlx_layer0_compact_chunk_cache_probe,
                k_values,
                v_values,
                active_rope_config().num_key_value_heads,
                active_rope_config().head_dim,
                expanded_kv_chunk_size(),
                &compact_append_ms);
            final_attention_mlx_compact_concat = mlx_compact_kv_cache_attention_chunked(
                q_values,
                mlx_layer0_compact_chunk_cache_probe,
                active_rope_config().num_attention_heads,
                active_rope_config().num_key_value_heads,
                active_rope_config().head_dim,
                &mlx_compact_concat_attention_probe_timing_ms,
                &mlx_compact_concat_attention_detail);
            mlx_compact_concat_attention_probe_max_abs_diff =
                vector_max_abs_diff(final_attention_mlx_compact_concat, attention_values);
            final_attention_mlx_compact_chunk_aware = mlx_compact_kv_cache_attention_chunk_aware(
                q_values,
                mlx_layer0_compact_chunk_cache_probe,
                active_rope_config().num_attention_heads,
                active_rope_config().num_key_value_heads,
                active_rope_config().head_dim,
                &mlx_compact_chunk_aware_attention_probe_timing_ms,
                &mlx_compact_chunk_aware_attention_detail);
            mlx_compact_chunk_aware_attention_probe_max_abs_diff =
                vector_max_abs_diff(final_attention_mlx_compact_chunk_aware, final_attention_mlx_compact_concat);
          } catch (const std::exception& e) {
            mlx_compact_attention_probe_error = e.what();
          } catch (...) {
            mlx_compact_attention_probe_error = "unknown_exception";
          }
        } else {
          try {
            mlx_resident_kv_cache_append_probe(
                mlx_layer0_kv_cache_probe,
                k_values,
                v_values,
                active_rope_config().num_key_value_heads,
                active_rope_config().head_dim);
          } catch (...) {
            // The resident KV probe is diagnostic only. The CPU cache below
            // remains the source for parity checkpoints.
          }
          try {
            mlx_expanded_resident_kv_cache_append_probe(
                mlx_layer0_expanded_kv_cache_probe,
                k_values,
                v_values,
                active_rope_config().num_attention_heads,
                active_rope_config().num_key_value_heads,
                active_rope_config().head_dim);
          } catch (...) {
            // Diagnostic only.
          }
          try {
            mlx_compact_kv_cache_append_chunked(
                mlx_layer0_compact_chunk_cache_probe,
                k_values,
                v_values,
                active_rope_config().num_key_value_heads,
                active_rope_config().head_dim,
                expanded_kv_chunk_size());
          } catch (...) {
            // Diagnostic only.
          }
        }
        append_native_kv(&debug_cache, 0, k_values, v_values);
        std::vector<float> o_values = quantized_linear_vector_values_layout_cached(layer0.o_proj, attention_values);
        std::vector<float> attention_residual(input_values.size(), 0.0f);
        for (std::size_t i = 0; i < attention_residual.size(); ++i) {
          attention_residual[i] = input_values[i] + o_values[i];
        }
        std::vector<float> post_norm_values = rmsnorm_values(layer0.post_attention_norm, attention_residual, eps);
        std::vector<float> gate_values = quantized_linear_vector_values_layout_cached(layer0.gate_proj, post_norm_values);
        std::vector<float> up_values = quantized_linear_vector_values_layout_cached(layer0.up_proj, post_norm_values);
        std::vector<float> activated_values(gate_values.size(), 0.0f);
        for (std::size_t i = 0; i < activated_values.size(); ++i) {
          const float gate = gate_values[i];
          const float silu = gate / (1.0f + std::exp(-gate));
          activated_values[i] = silu * up_values[i];
        }
        std::vector<float> down_values = quantized_linear_vector_values_layout_cached(layer0.down_proj, activated_values);
        std::vector<float> residual_values(attention_residual.size(), 0.0f);
        for (std::size_t i = 0; i < residual_values.size(); ++i) {
          residual_values[i] = attention_residual[i] + down_values[i];
        }
        if (prompt_index + 1 == prompt_token_ids.size()) {
          final_embedding = input_values;
          final_norm = norm_values;
          final_q = q_raw_values;
          final_k = k_raw_values;
          final_q_after_q_norm = q_after_q_norm_values;
          final_k_after_k_norm = k_after_k_norm_values;
          final_v = v_values;
          final_q_after_rope = q_values;
          final_k_after_rope = k_values;
          final_q_after_rope_interleaved = q_after_q_norm_values;
          final_k_after_rope_interleaved = k_after_k_norm_values;
          apply_rope_interleaved_in_place(
              final_q_after_rope_interleaved,
              active_rope_config().num_attention_heads,
              active_rope_config().head_dim,
              static_cast<std::uint64_t>(prompt_index),
              active_rope_config().theta);
          apply_rope_interleaved_in_place(
              final_k_after_rope_interleaved,
              active_rope_config().num_key_value_heads,
              active_rope_config().head_dim,
              static_cast<std::uint64_t>(prompt_index),
              active_rope_config().theta);
          final_attention_scores_probs = attention_scores_probs;
          final_attention = attention_values;
          final_o = o_values;
          final_attention_residual = attention_residual;
          final_post_norm = post_norm_values;
          final_mlp_down = down_values;
          final_residual = residual_values;
        }
      }
      std::ostringstream out;
      out << "{"
          << "\"available\":true,"
          << "\"prompt_token_ids\":" << json_array_u64_to_string(prompt_token_ids) << ","
          << "\"target_layer\":0,"
          << "\"target_position\":" << (prompt_token_ids.empty() ? 0 : prompt_token_ids.size() - 1) << ","
          << "\"rope_applied\":" << (active_rope_config().enabled ? "true" : "false") << ","
          << "\"rope_position\":" << (prompt_token_ids.empty() ? 0 : prompt_token_ids.size() - 1) << ","
          << "\"rope_theta\":" << active_rope_config().theta << ","
          << "\"q_norm_applied\":true,"
          << "\"k_norm_applied\":true,"
          << "\"head_dim\":" << active_rope_config().head_dim << ","
          << "\"num_attention_heads\":" << active_rope_config().num_attention_heads << ","
          << "\"num_key_value_heads\":" << active_rope_config().num_key_value_heads << ","
          << "\"embedding_token0\":" << parity_vector_summary_json(token0_embedding) << ","
          << "\"embedding_final_prompt_token\":" << parity_vector_summary_json(final_embedding) << ","
          << "\"input_rmsnorm\":" << parity_vector_summary_json(final_norm) << ","
          << "\"q_before_rope\":" << parity_vector_summary_json(final_q) << ","
          << "\"k_before_rope\":" << parity_vector_summary_json(final_k) << ","
          << "\"q_after_q_norm\":" << parity_vector_summary_json(final_q_after_q_norm) << ","
          << "\"k_after_k_norm\":" << parity_vector_summary_json(final_k_after_k_norm) << ","
          << "\"v\":" << parity_vector_summary_json(final_v) << ","
          << "\"q_after_rope\":" << parity_vector_summary_json(final_q_after_rope) << ","
          << "\"k_after_rope\":" << parity_vector_summary_json(final_k_after_rope) << ","
          << "\"rope_layout_experiments\":{"
          << "\"active_layout\":\"half_split\","
          << "\"half_split\":{"
          << "\"q_after_rope\":" << parity_vector_summary_json(final_q_after_rope) << ","
          << "\"k_after_rope\":" << parity_vector_summary_json(final_k_after_rope)
          << "},"
          << "\"interleaved_pair\":{"
          << "\"q_after_rope\":" << parity_vector_summary_json(final_q_after_rope_interleaved) << ","
          << "\"k_after_rope\":" << parity_vector_summary_json(final_k_after_rope_interleaved)
          << "}"
          << "},"
          << "\"attention_head0\":" << final_attention_scores_probs << ","
          << "\"attention_output\":" << parity_vector_summary_json(final_attention) << ","
          << "\"mlx_attention_probe\":{"
          << "\"available\":" << (mlx_attention_probe_available ? "true" : "false") << ","
          << "\"target_layer\":0,"
          << "\"backend\":\"metal\","
          << "\"cpu_attention_seq_len\":" << cpu_attention_seq_len << ","
          << "\"mlx_attention_seq_len\":" << mlx_attention_seq_len << ","
          << "\"cache_appended_before_cpu_attention\":"
          << (cache_appended_before_cpu_attention ? "true" : "false") << ","
          << "\"cache_appended_before_mlx_attention\":"
          << (cache_appended_before_mlx_attention ? "true" : "false") << ","
          << "\"current_position_included_once\":"
          << ((cpu_attention_seq_len == mlx_attention_seq_len &&
               !cache_appended_before_cpu_attention &&
               !cache_appended_before_mlx_attention)
              ? "true"
              : "false")
          << ","
          << "\"gqa\":{"
          << "\"num_attention_heads\":" << active_rope_config().num_attention_heads << ","
          << "\"num_key_value_heads\":" << active_rope_config().num_key_value_heads << ","
          << "\"head_dim\":" << active_rope_config().head_dim << ","
          << "\"q_heads_per_kv\":"
          << (active_rope_config().num_key_value_heads == 0
              ? 0
              : active_rope_config().num_attention_heads / active_rope_config().num_key_value_heads)
          << ","
          << "\"mapping\":\"kv_head = q_head / q_heads_per_kv\""
          << "},"
          << "\"layout\":{"
          << "\"cpu_k_cache_shape\":["
          << debug_cache.layers[0].keys.size() << ","
          << active_rope_config().num_key_value_heads << ","
          << active_rope_config().head_dim << "],"
          << "\"cpu_k_cache_layout\":\"positions_outer_then_flat_kv_heads_head_dim\","
          << "\"mlx_k_array_shape\":["
          << mlx_attention_probe_seq_len << "," << active_rope_config().head_dim << "],"
          << "\"cpu_v_cache_shape\":["
          << debug_cache.layers[0].values.size() << ","
          << active_rope_config().num_key_value_heads << ","
          << active_rope_config().head_dim << "],"
          << "\"cpu_v_cache_layout\":\"positions_outer_then_flat_kv_heads_head_dim\","
          << "\"mlx_v_array_shape\":["
          << mlx_attention_probe_seq_len << "," << active_rope_config().head_dim << "],"
          << "\"attention_score_shape\":[1," << mlx_attention_probe_seq_len << "],"
          << "\"softmax_axis\":-1,"
          << "\"value_mix_shape\":[1," << active_rope_config().head_dim << "]"
          << "},"
          << "\"variants\":[{"
          << "\"name\":\"per_head_seq_head_dim_softmax_axis_neg1\","
          << "\"k_shape\":\"[seq, head_dim]\","
          << "\"v_shape\":\"[seq, head_dim]\","
          << "\"score_matmul\":\"[1, head_dim] @ [head_dim, seq]\","
          << "\"softmax_axis\":-1,"
          << "\"max_abs_diff\":" << mlx_attention_probe_max_abs_diff << ","
          << "\"timing_ms\":" << mlx_attention_probe_timing_ms << ","
          << "\"ran\":true"
          << "},{"
          << "\"name\":\"grouped_by_kv_head_q4_seq_head_dim_softmax_axis_neg1\","
          << "\"q_shape\":\"[q_heads_per_kv, head_dim]\","
          << "\"k_shape\":\"[seq, head_dim]\","
          << "\"v_shape\":\"[seq, head_dim]\","
          << "\"score_shape\":\"[q_heads_per_kv, seq]\","
          << "\"softmax_axis\":-1,"
          << "\"max_abs_diff\":" << mlx_grouped_attention_probe_max_abs_diff << ","
          << "\"timing_ms\":" << mlx_grouped_attention_probe_timing_ms << ","
          << "\"ran\":" << (final_attention_mlx_grouped.empty() ? "false" : "true") << ","
          << "\"error\":\"" << json_escape(mlx_grouped_attention_probe_error) << "\""
          << "},{"
          << "\"name\":\"all_head_batched_q_heads_seq_head_dim\","
          << "\"q_shape\":\"[q_heads, 1, head_dim]\","
          << "\"k_shape\":\"[q_heads, seq, head_dim]\","
          << "\"v_shape\":\"[q_heads, seq, head_dim]\","
          << "\"score_shape\":\"[q_heads, 1, seq]\","
          << "\"softmax_axis\":-1,"
          << "\"max_abs_diff\":" << mlx_all_head_attention_probe_max_abs_diff << ","
          << "\"checksum\":" << vector_checksum(final_attention_mlx_all_head_batched) << ","
          << "\"timing_ms\":" << mlx_all_head_attention_probe_timing_ms << ","
          << "\"ran\":" << (final_attention_mlx_all_head_batched.empty() ? "false" : "true") << ","
          << "\"error\":\"" << json_escape(mlx_all_head_attention_probe_error) << "\""
          << "},{"
          << "\"name\":\"resident_mlx_kv_all_head_batched\","
          << "\"available\":" << (final_attention_mlx_resident_kv.empty() ? "false" : "true") << ","
          << "\"kv_cache_backend\":\"mlx\","
          << "\"kv_cache_shape\":["
          << active_rope_config().num_key_value_heads << ","
          << mlx_layer0_kv_cache_probe.current_len << ","
          << active_rope_config().head_dim << "],"
          << "\"append_mode\":\"" << json_escape(mlx_layer0_kv_cache_probe.append_mode) << "\","
          << "\"q_shape\":\"[q_heads, 1, head_dim]\","
          << "\"k_take_shape\":\"[q_heads, current_len, head_dim]\","
          << "\"v_take_shape\":\"[q_heads, current_len, head_dim]\","
          << "\"max_abs_diff\":" << mlx_resident_kv_attention_probe_max_abs_diff << ","
          << "\"checksum\":" << vector_checksum(final_attention_mlx_resident_kv) << ","
          << "\"append_ms\":" << mlx_resident_kv_final_append_ms << ","
          << "\"attention_ms\":" << mlx_resident_kv_attention_probe_timing_ms << ","
          << "\"total_ms\":"
          << (mlx_resident_kv_final_append_ms + mlx_resident_kv_attention_probe_timing_ms) << ","
          << "\"current_len\":" << mlx_layer0_kv_cache_probe.current_len << ","
          << "\"ran\":" << (final_attention_mlx_resident_kv.empty() ? "false" : "true") << ","
          << "\"error\":\"" << json_escape(mlx_resident_kv_attention_probe_error) << "\""
          << "},{"
          << "\"name\":\"expanded_q_heads_resident_mlx_kv_all_head_batched\","
          << "\"available\":" << (final_attention_mlx_expanded_resident_kv.empty() ? "false" : "true") << ","
          << "\"kv_cache_backend\":\"mlx\","
          << "\"kv_cache_shape\":["
          << active_rope_config().num_attention_heads << ","
          << mlx_layer0_expanded_kv_cache_probe.current_len << ","
          << active_rope_config().head_dim << "],"
          << "\"append_mode\":\"" << json_escape(mlx_layer0_expanded_kv_cache_probe.append_mode) << "\","
          << "\"q_shape\":\"[q_heads, 1, head_dim]\","
          << "\"k_shape\":\"[q_heads, current_len, head_dim]\","
          << "\"v_shape\":\"[q_heads, current_len, head_dim]\","
          << "\"max_abs_diff\":" << mlx_expanded_resident_kv_attention_probe_max_abs_diff << ","
          << "\"checksum\":" << vector_checksum(final_attention_mlx_expanded_resident_kv) << ","
          << "\"append_ms\":" << mlx_expanded_resident_kv_final_append_ms << ","
          << "\"attention_ms\":" << mlx_expanded_resident_kv_attention_probe_timing_ms << ","
          << "\"total_ms\":"
          << (mlx_expanded_resident_kv_final_append_ms +
              mlx_expanded_resident_kv_attention_probe_timing_ms) << ","
          << "\"current_len\":" << mlx_layer0_expanded_kv_cache_probe.current_len << ","
          << "\"ran\":" << (final_attention_mlx_expanded_resident_kv.empty() ? "false" : "true") << ","
          << "\"error\":\"" << json_escape(mlx_expanded_resident_kv_attention_probe_error) << "\","
          << "\"memory_estimate\":{"
          << "\"kv_heads_cache_bytes\":"
          << (active_rope_config().num_key_value_heads *
              mlx_layer0_expanded_kv_cache_probe.current_len *
              active_rope_config().head_dim * 2ULL * sizeof(float)) << ","
          << "\"expanded_q_heads_cache_bytes\":"
          << (active_rope_config().num_attention_heads *
              mlx_layer0_expanded_kv_cache_probe.current_len *
              active_rope_config().head_dim * 2ULL * sizeof(float)) << ","
          << "\"expansion_factor\":"
          << (active_rope_config().num_key_value_heads == 0
              ? 0
              : static_cast<double>(active_rope_config().num_attention_heads) /
                    static_cast<double>(active_rope_config().num_key_value_heads))
          << "}"
          << "},{"
          << "\"name\":\"compact_chunked_concat_attention\","
          << "\"available\":" << (final_attention_mlx_compact_concat.empty() ? "false" : "true") << ","
          << "\"promoted_to_generation\":true,"
          << "\"kv_cache_backend\":\"mlx_compact_kv_heads_chunked\","
          << "\"layout\":\"[kv_heads, chunks, chunk_size, head_dim] logical\","
          << "\"concat_used_for_attention\":true,"
          << "\"chunk_aware\":false,"
          << "\"max_abs_diff_vs_cpu\":" << mlx_compact_concat_attention_probe_max_abs_diff << ","
          << "\"checksum\":" << vector_checksum(final_attention_mlx_compact_concat) << ","
          << "\"timing_ms\":" << mlx_compact_concat_attention_probe_timing_ms << ","
          << "\"detail_ms\":{"
          << "\"kv_view_assembly\":" << mlx_compact_concat_attention_detail.kv_view_assembly_ms << ","
          << "\"score_matmul\":" << mlx_compact_concat_attention_detail.score_matmul_ms << ","
          << "\"softmax\":" << mlx_compact_concat_attention_detail.softmax_ms << ","
          << "\"value_mix_matmul\":" << mlx_compact_concat_attention_detail.value_mix_matmul_ms << ","
          << "\"reshape_flatten\":" << mlx_compact_concat_attention_detail.reshape_flatten_ms << ","
          << "\"eval_sync\":" << mlx_compact_concat_attention_detail.eval_sync_ms
          << "},"
          << "\"ran\":" << (final_attention_mlx_compact_concat.empty() ? "false" : "true") << ","
          << "\"error\":\"" << json_escape(mlx_compact_attention_probe_error) << "\""
          << "},{"
          << "\"name\":\"compact_chunked_chunk_aware_streaming_softmax\","
          << "\"available\":" << (final_attention_mlx_compact_chunk_aware.empty() ? "false" : "true") << ","
          << "\"promoted_to_generation\":false,"
          << "\"kv_cache_backend\":\"mlx_compact_kv_heads_chunked\","
          << "\"layout\":\"[kv_heads, chunks, chunk_size, head_dim] logical\","
          << "\"concat_used_for_attention\":false,"
          << "\"chunk_aware\":true,"
          << "\"algorithm\":\"streaming_softmax_global_max_denominator_numerator_accumulation\","
          << "\"max_abs_diff_vs_concat_compact\":" << mlx_compact_chunk_aware_attention_probe_max_abs_diff << ","
          << "\"max_abs_diff_vs_cpu\":"
          << (final_attention_mlx_compact_chunk_aware.empty()
                ? 0.0
                : vector_max_abs_diff(final_attention_mlx_compact_chunk_aware, final_attention))
          << ","
          << "\"checksum\":" << vector_checksum(final_attention_mlx_compact_chunk_aware) << ","
          << "\"timing_ms\":" << mlx_compact_chunk_aware_attention_probe_timing_ms << ","
          << "\"detail_ms\":{"
          << "\"kv_view_assembly\":" << mlx_compact_chunk_aware_attention_detail.kv_view_assembly_ms << ","
          << "\"score_matmul\":" << mlx_compact_chunk_aware_attention_detail.score_matmul_ms << ","
          << "\"softmax\":" << mlx_compact_chunk_aware_attention_detail.softmax_ms << ","
          << "\"value_mix_matmul\":" << mlx_compact_chunk_aware_attention_detail.value_mix_matmul_ms << ","
          << "\"reshape_flatten\":" << mlx_compact_chunk_aware_attention_detail.reshape_flatten_ms << ","
          << "\"eval_sync\":" << mlx_compact_chunk_aware_attention_detail.eval_sync_ms
          << "},"
          << "\"ran\":" << (final_attention_mlx_compact_chunk_aware.empty() ? "false" : "true") << ","
          << "\"error\":\"" << json_escape(mlx_compact_attention_probe_error) << "\""
          << "},{"
          << "\"name\":\"mlx_fast_scaled_dot_product_attention\","
          << "\"available\":" << (final_attention_mlx_fast.empty() ? "false" : "true") << ","
          << "\"q_shape\":\"[1, q_heads, 1, head_dim]\","
          << "\"k_shape\":\"[1, q_heads, seq, head_dim]\","
          << "\"v_shape\":\"[1, q_heads, seq, head_dim]\","
          << "\"max_abs_diff\":" << mlx_fast_attention_probe_max_abs_diff << ","
          << "\"checksum\":" << vector_checksum(final_attention_mlx_fast) << ","
          << "\"timing_ms\":" << mlx_fast_attention_probe_timing_ms << ","
          << "\"ran\":" << (final_attention_mlx_fast.empty() ? "false" : "true") << ","
          << "\"error\":\"" << json_escape(mlx_fast_attention_probe_error) << "\""
          << "}],"
          << "\"best_variant\":\""
          << (mlx_grouped_attention_probe_max_abs_diff <= mlx_attention_probe_max_abs_diff
              ? "grouped_by_kv_head_q4_seq_head_dim_softmax_axis_neg1"
              : "per_head_seq_head_dim_softmax_axis_neg1")
          << "\","
          << "\"timing_comparison_ms\":{"
          << "\"cpu_layer0\":" << cpu_attention_probe_timing_ms << ","
          << "\"old_per_head_mlx_layer0\":" << mlx_attention_probe_timing_ms << ","
          << "\"grouped_by_kv_head_mlx_layer0\":" << mlx_grouped_attention_probe_timing_ms << ","
          << "\"all_head_batched_mlx_layer0\":" << mlx_all_head_attention_probe_timing_ms << ","
          << "\"resident_mlx_kv_append_layer0\":" << mlx_resident_kv_final_append_ms << ","
          << "\"resident_mlx_kv_attention_layer0\":" << mlx_resident_kv_attention_probe_timing_ms << ","
          << "\"resident_mlx_kv_total_layer0\":"
          << (mlx_resident_kv_final_append_ms + mlx_resident_kv_attention_probe_timing_ms) << ","
          << "\"expanded_q_heads_resident_mlx_kv_append_layer0\":"
          << mlx_expanded_resident_kv_final_append_ms << ","
          << "\"expanded_q_heads_resident_mlx_kv_attention_layer0\":"
          << mlx_expanded_resident_kv_attention_probe_timing_ms << ","
          << "\"expanded_q_heads_resident_mlx_kv_total_layer0\":"
          << (mlx_expanded_resident_kv_final_append_ms +
              mlx_expanded_resident_kv_attention_probe_timing_ms) << ","
          << "\"compact_chunked_concat_attention_layer0\":"
          << mlx_compact_concat_attention_probe_timing_ms << ","
          << "\"compact_chunked_chunk_aware_attention_layer0\":"
          << mlx_compact_chunk_aware_attention_probe_timing_ms << ","
          << "\"native_fast_attention_mlx_layer0\":" << mlx_fast_attention_probe_timing_ms
          << "},"
          << "\"grouped_probe\":{"
          << "\"available\":" << (final_attention_mlx_grouped.empty() ? "false" : "true") << ","
          << "\"checksum_mlx_grouped\":" << vector_checksum(final_attention_mlx_grouped) << ","
          << "\"max_abs_diff\":" << mlx_grouped_attention_probe_max_abs_diff << ","
          << "\"worst_summary\":"
          << attention_head_summary_json(
                 final_attention,
                 final_attention_mlx_grouped,
                 active_rope_config().num_attention_heads,
                 active_rope_config().num_key_value_heads,
                 active_rope_config().head_dim)
          << "},"
          << "\"per_head_comparison\":"
          << attention_head_comparison_json(
                 final_attention,
                 final_attention_mlx,
                 active_rope_config().num_attention_heads,
                 active_rope_config().num_key_value_heads,
                 active_rope_config().head_dim)
          << ","
          << "\"per_head_max_abs_diff_summary\":"
          << attention_head_summary_json(
                 final_attention,
                 final_attention_mlx,
                 active_rope_config().num_attention_heads,
                 active_rope_config().num_key_value_heads,
                 active_rope_config().head_dim)
          << ","
          << "\"worst_head_diagnostic\":" << mlx_attention_worst_head_diagnostic << ","
          << "\"shape\":[1," << final_attention.size() << "],"
          << "\"checksum_cpu\":" << vector_checksum(final_attention) << ","
          << "\"checksum_mlx\":" << vector_checksum(final_attention_mlx) << ","
          << "\"max_abs_diff\":" << mlx_attention_probe_max_abs_diff << ","
          << "\"timing_ms\":" << mlx_attention_probe_timing_ms << ","
          << "\"first_values_cpu\":" << json_array_float_to_string(first_values_of(final_attention)) << ","
          << "\"first_values_mlx\":" << json_array_float_to_string(first_values_of(final_attention_mlx)) << ","
          << "\"error\":\"" << json_escape(mlx_attention_probe_error) << "\""
          << "},"
          << "\"o_proj_output\":" << parity_vector_summary_json(final_o) << ","
          << "\"post_attention_residual\":" << parity_vector_summary_json(final_attention_residual) << ","
          << "\"post_attention_rmsnorm\":" << parity_vector_summary_json(final_post_norm) << ","
          << "\"mlp_output\":" << parity_vector_summary_json(final_mlp_down) << ","
          << "\"layer0_final_residual\":" << parity_vector_summary_json(final_residual)
          << "}";
      layer0_parity_checkpoints_json = out.str();
    } catch (const std::exception& e) {
      layer0_parity_checkpoints_json =
          std::string("{\"available\":false,\"error\":\"") + json_escape(e.what()) + "\"}";
    } catch (...) {
      layer0_parity_checkpoints_json = "{\"available\":false,\"error\":\"unknown_exception\"}";
    }

    std::string layer0_mlx_resident_full_block_probe = "{\"available\":false,\"error\":\"not_run\"}";
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
    try {
      if (enable_layer0_diagnostic_probe() && !resident.layers.empty()) {
        const std::vector<float> layer0_probe_input =
            embedding_values_from_record(embedding, static_cast<std::size_t>(prompt_token_ids[0]));
        layer0_mlx_resident_full_block_probe =
            layer0_mlx_resident_full_block_probe_json(
                resident.layers[0],
                layer0_probe_input,
                eps);
      }
    } catch (const std::exception& e) {
      layer0_mlx_resident_full_block_probe =
          std::string("{\"available\":false,\"error\":\"") + json_escape(e.what()) + "\"}";
    } catch (...) {
      layer0_mlx_resident_full_block_probe =
          "{\"available\":false,\"error\":\"unknown_exception\"}";
    }
#endif

    const auto adapter = active_generation_adapter;
    const bool adapter_active = adapter != nullptr;
    const std::uint64_t adapter_layer_count = adapter_active
        ? static_cast<std::uint64_t>(adapter->layers.size())
        : 0ULL;
    const std::uint64_t adapter_sites_per_token = adapter_active
        ? adapter_layer_count * 7ULL
        : 0ULL;
    const std::uint64_t adapter_decode_token_count = generated_ids.size() > 1
        ? static_cast<std::uint64_t>(generated_ids.size() - 1)
        : 0ULL;
    const std::uint64_t adapter_applied_projection_count =
        adapter_sites_per_token *
        (static_cast<std::uint64_t>(prompt_token_ids.size()) + adapter_decode_token_count);

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"prompt_token_id\":" << prompt_token_id << ","
             << "\"prefill_token_ids\":" << json_array_u64_to_string(prompt_token_ids) << ","
             << "\"prompt_tokens_processed\":" << prompt_token_ids.size() << ","
             << "\"positions_after_prefill\":" << prompt_result.positions_after << ","
             << "\"first_decode_position\":" << prompt_result.positions_after << ","
             << "\"kv_positions_per_layer\":" << kv_positions_per_layer_json(kv_cache) << ","
             << "\"first_decode_token_id\":" << first_decode_token_id << ","
             << "\"generated_token_count\":" << generated_tokens << ","
             << "\"generated_token_ids\":" << json_array_u64_to_string(generated_ids) << ","
             << "\"generated_token_scores\":" << json_array_float_to_string(generated_scores) << ","
             << "\"first_generated_top_logits\":" << json_top_logits_to_string(first_generated_top_logits) << ","
             << "\"parity_checkpoints\":{"
             << "\"layer0\":" << layer0_parity_checkpoints_json
             << "},"
             << "\"layer0_mlx_resident_full_block_probe\":"
             << layer0_mlx_resident_full_block_probe << ","
             << "\"decoded_generated_text\":\"" << json_escape(decoded_text.str()) << "\","
             << "\"resident_group_load_ms\":" << resident_group_load_ms << ","
             << "\"resident_group_load_timing_ms\":" << resident_load_timing_json() << ","
             << "\"resident_projection_array_warmup_ms\":" << resident_projection_array_warmup_ms << ","
             << "\"resident_projection_array_warmup_timing_ms\":"
             << projection_warmup_timing_json(projection_warmup_timing) << ","
             << "\"second_resident_projection_array_warmup_ms\":"
             << second_resident_projection_array_warmup_ms << ","
             << "\"second_resident_projection_array_warmup_timing_ms\":"
             << projection_warmup_timing_json(second_projection_warmup_timing) << ","
             << "\"prompt_pass_ms\":" << prompt_pass_ms << ","
             << "\"prompt_pass_timing_buckets_ms\":" << resident_decode_timing_buckets_json(prompt_result) << ","
             << "\"prompt_largest_arithmetic_bucket\":\"" << json_escape(prompt_result.largest_arithmetic_bucket) << "\","
             << "\"prompt_largest_arithmetic_bucket_ms\":" << prompt_result.largest_arithmetic_bucket_ms << ","
             << "\"prompt_uses_mlx_decode_value\":true,"
             << "\"prompt_uses_mmap_projection_arrays\":" << (use_resident_mlx_projection_arrays() ? "true" : "false") << ","
             << "\"prompt_uses_fast_layer_kernel\":true,"
             << "\"prompt_readback_count\":" << prompt_readback_count << ","
             << "\"prompt_readback_reasons\":[\"qkv_for_cpu_attention\"],"
             << "\"prompt_fallback_steps\":" << json_array_string_to_string(prompt_fallback_steps) << ","
             << "\"per_token_incremental_ms\":" << json_array_double_to_string(per_token_incremental_ms) << ","
             << "\"generation_timing_bucket_summary\":"
             << generation_timing_bucket_summary_json(generation_timing_buckets, total_generation_ms) << ","
             << "\"long_decode_diagnostic_snapshots\":" << diagnostic_snapshots_json() << ","
             << "\"repetition_penalty_supported\":false,"
             << "\"repetition_penalty_active\":false,"
             << "\"repetition_penalty_value\":1.0,"
             << "\"repetition_penalty_applied_with_adapter\":false,"
             << "\"repetition_penalty_generated_token_history_considered\":0,"
             << "\"repetition_penalty_prompt_tokens_considered\":false,"
             << "\"repetition_diagnostic_enabled\":"
             << (repetition_diagnostic_enabled() ? "true" : "false") << ","
             << "\"repetition_diagnostics\":" << repetition_diagnostics_json(repetition_diagnostics) << ","
             << "\"repetition_collapse_summary\":" << collapse_summary_json() << ","
             << "\"total_generation_ms\":" << total_generation_ms << ","
             << "\"generation_1_total_ms\":" << total_generation_ms << ","
             << "\"generation_2_prompt_pass_ms\":" << second_prompt_pass_ms << ","
             << "\"generation_2_per_token_incremental_ms\":"
             << json_array_double_to_string(second_per_token_incremental_ms) << ","
             << "\"generation_2_total_ms\":" << second_total_generation_ms << ","
             << "\"generation_2_generated_token_ids\":"
             << json_array_u64_to_string(second_generated_ids) << ","
             << "\"generation_2_generated_token_scores\":"
             << json_array_float_to_string(second_generated_scores) << ","
             << "\"generation_2_decoded_generated_text\":\""
             << json_escape(second_decoded_text.str()) << "\","
             << "\"generation_2_stopped\":" << (second_stopped ? "true" : "false") << ","
             << "\"generation_2_stop_reason\":\"" << json_escape(second_stop_reason) << "\","
             << "\"generation_2_stop_token_id\":";
    if (second_stopped) {
      json_out << second_stop_token_id;
    } else {
      json_out << "null";
    }
    json_out << ","
             << "\"generation_2_first_generated_final_norm_checksum\":"
             << second_first_generated_final_norm_checksum << ","
             << "\"generation_2_logits_len\":" << second_last_logits_len << ","
             << "\"generation_2_fallback_steps_per_token\":"
             << second_fallback_steps_per_token_json() << ","
             << "\"second_generation_proof_enabled\":"
             << (run_second_generation_proof ? "true" : "false") << ","
             << "\"total_probe_ms\":" << total_probe_ms << ","
             << "\"tokens_per_second_after_resident_load\":" << tokens_per_second_after_resident_load << ","
             << "\"tokens_per_second_including_resident_load\":" << tokens_per_second_including_resident_load << ","
             << "\"stopped\":" << (stopped ? "true" : "false") << ","
             << "\"stop_reason\":\"" << json_escape(stop_reason) << "\","
             << "\"stop_token_id\":";
    if (stopped) {
      json_out << stop_token_id;
    } else {
      json_out << "null";
    }
    json_out << ","
             << "\"fallback_steps_per_token\":" << fallback_steps_per_token_json() << ","
             << "\"fallback_steps\":[],"
             << "\"readback_count\":" << mlx_decode_readback_reasons.size() << ","
             << "\"readback_reasons\":" << readback_reasons_json() << ","
             << "\"cpu_fallback_steps\":" << cpu_fallback_steps_json() << ","
             << "\"fallback_used\":" << (any_fallback || fallback_used ? "true" : "false") << ","
             << "\"adapter_requested\":" << (adapter_active ? "true" : "false") << ","
             << "\"adapter_active\":" << (adapter_active ? "true" : "false") << ","
             << "\"adapter_path\":\"" << json_escape(adapter_active ? adapter->adapter_dir : "") << "\","
             << "\"adapter_tensor_count\":" << (adapter_active ? adapter->tensor_count : 0) << ","
             << "\"adapter_dtype\":\"" << json_escape(adapter_active ? adapter->dtype : "") << "\","
             << "\"adapter_rank\":" << (adapter_active ? adapter->rank : 0) << ","
             << "\"adapter_scale\":" << (adapter_active ? adapter->scale : 0.0) << ","
             << "\"adapter_layers\":" << (adapter_active ? json_array_int_to_string(adapter->layers) : "[]") << ","
             << "\"adapter_layer_count\":" << adapter_layer_count << ","
             << "\"adapter_targets\":" << (adapter_active ? json_array_string_to_string(adapter->targets) : "[]") << ","
             << "\"adapter_tensors_loaded\":" << (adapter_active ? adapter->tensor_count : 0) << ","
             << "\"adapter_missing_expected_tensors\":"
             << (adapter_active ? json_array_string_to_string(adapter->missing_expected_tensors) : "[]") << ","
             << "\"adapter_unexpected_tensors\":"
             << (adapter_active ? json_array_string_to_string(adapter->unexpected_tensors) : "[]") << ","
             << "\"adapter_applied_to_prefill\":"
             << ((adapter_active && !prompt_token_ids.empty()) ? "true" : "false") << ","
             << "\"adapter_applied_to_decode\":"
             << ((adapter_active && adapter_decode_token_count > 0) ? "true" : "false") << ","
             << "\"adapter_applied_projection_count\":" << adapter_applied_projection_count << ","
             << "\"adapter_fallback_used\":false,"
             << "\"adapter_error\":null,"
             << "\"cached_mlx_arrays_path_applied_to_generation\":" << (enable_cached_mlx_quantized_arrays() ? "true" : "false") << ","
             << "\"resident_mlx_projection_arrays_applied_to_generation\":"
             << (use_resident_mlx_projection_arrays() ? "true" : "false") << ","
             << "\"mlx_decode_value_available\":true,"
             << "\"mlx_decode_value_applied_to_generation\":true,"
             << "\"mlx_rmsnorm_applied\":true,"
             << "\"mlx_residual_applied\":true,"
             << "\"q_norm_applied\":true,"
             << "\"k_norm_applied\":true,"
             << "\"mlx_resident_layer_block_available\":" << (use_mlx_resident_layer_block ? "true" : "false") << ","
             << "\"full_resident_block_gate_enabled\":"
             << (full_resident_block_gate_enabled ? "true" : "false") << ","
             << "\"full_resident_block_applied_to_generation\":"
             << (full_resident_block_applied_to_generation ? "true" : "false") << ","
             << "\"partial_resident_layer_block_applied\":"
             << (partial_resident_layer_block_applied ? "true" : "false") << ","
             << "\"resident_attention_to_o_enabled\":"
             << (resident_attention_to_o_enabled() ? "true" : "false") << ","
             << "\"resident_o_residual_enabled\":"
             << (resident_o_residual_enabled() ? "true" : "false") << ","
             << "\"resident_mlp_only_requested\":"
             << (resident_mlp_only_requested() ? "true" : "false") << ","
             << "\"resident_mlp_only_enabled\":"
             << "false,"
             << "\"resident_mlp_only_semantics\":\"deprecated_noop_already_default\","
             << "\"resident_block_fallback_count\":"
             << resident_block_fallback_count << ","
             << "\"resident_block_fallback_reasons\":"
             << json_array_string_to_string(resident_block_fallback_reasons) << ","
             << "\"mlx_resident_layer_block_applied_to_generation\":"
             << (last_incremental_result.mlx_resident_layer_block_applied ? "true" : "false") << ","
             << "\"mlx_resident_layer_block_fallback_used\":"
             << (last_incremental_result.mlx_resident_layer_block_fallback_used ? "true" : "false") << ","
             << "\"mlx_resident_mlp_chain_available\":" << (use_mlx_resident_mlp_chain ? "true" : "false") << ","
             << "\"mlx_resident_mlp_chain_applied_to_generation\":" << (apply_mlx_resident_mlp_chain ? "true" : "false") << ","
             << "\"mlx_resident_mlp_chain_default_path\":"
             << (apply_mlx_resident_mlp_chain ? "true" : "false") << ","
             << "\"mlx_resident_mlp_chain_layer0_compare\":{"
             << "\"ran\":" << (run_mlp_chain_compare ? "true" : "false") << ","
             << "\"ok\":" << (mlp_chain_compare_ok ? "true" : "false") << ","
             << "\"output_len\":" << mlp_chain_compare_output_len << ","
             << "\"checksum_current\":" << mlp_chain_compare_checksum_current << ","
             << "\"checksum_resident\":" << mlp_chain_compare_checksum_resident << ","
             << "\"max_abs_diff\":" << mlp_chain_compare_max_abs_diff << ","
             << "\"timing_current_ms\":" << mlp_chain_compare_timing_current_ms << ","
             << "\"timing_resident_ms\":" << mlp_chain_compare_timing_resident_ms
             << "},"
             << "\"mlx_quantized_linear_available\":" << (mlx_quantized_linear_available ? "true" : "false") << ","
             << "\"mlx_quantized_linear_applied_to_generation\":"
             << ((mlx_quantized_linear_available && !any_fallback && use_optimized_linear) ? "true" : "false") << ","
             << "\"experimental_mlx_logits_topk_enabled\":"
             << (experimental_mlx_logits_topk_enabled() ? "true" : "false") << ","
             << "\"mlx_logits_topk_applied_to_generation\":"
             << (mlx_logits_topk_applied_to_generation ? "true" : "false") << ","
             << "\"mlx_logits_topk_fallback_used\":"
             << (mlx_logits_topk_fallback_used ? "true" : "false") << ","
             << "\"experimental_mlx_qk_norm_rope_enabled\":"
             << (experimental_mlx_qk_norm_rope_enabled() ? "true" : "false") << ","
             << "\"mlx_qk_norm_rope_applied_to_generation\":"
             << (mlx_qk_norm_rope_applied_to_generation ? "true" : "false") << ","
             << "\"mlx_qk_norm_rope_fallback_used\":"
             << (mlx_qk_norm_rope_fallback_used ? "true" : "false") << ","
             << "\"mlx_qk_norm_rope_verify_ran\":"
             << (mlx_qk_norm_rope_verify_ran ? "true" : "false") << ","
             << "\"mlx_qk_norm_rope_verify_max_abs_diff\":"
             << mlx_qk_norm_rope_verify_max_abs_diff << ","
             << "\"tensor_group_load_count\":" << tensor_group_load_count << ","
             << "\"resident_total_byte_size\":" << resident.total_byte_size << ","
             << "\"memory_diagnostics\":{"
             << "\"attention_backend_active\":\""
             << (active_attention_mode == "0" ? "compact_cpu" : json_escape(active_attention_mode))
             << "\","
             << "\"chunk_aware_attention_enabled\":"
             << (chunk_aware_attention_enabled() ? "true" : "false") << ","
             << "\"max_seq_allocated\":" << kv_cache.max_seq << ","
             << "\"q_heads\":" << active_rope_config().num_attention_heads << ","
             << "\"kv_heads\":" << active_rope_config().num_key_value_heads << ","
             << "\"head_dim\":" << active_rope_config().head_dim << ","
             << "\"layer_count\":" << layers << ","
             << "\"expanded_kv_bytes_estimate\":" << expanded_kv_bytes_estimate << ","
             << "\"chunked_expanded_kv_active_bytes\":" << chunked_expanded_kv_active_bytes << ","
             << "\"chunked_expanded_kv_chunk_size\":" << chunk_size << ","
             << "\"chunked_expanded_kv_chunk_bytes\":" << expanded_kv_chunk_bytes << ","
             << "\"chunked_expanded_kv_allocated_chunks_total\":"
             << chunked_allocated_chunks_total << ","
             << "\"chunked_expanded_kv_layers_with_chunks\":"
             << chunked_layers_with_chunks << ","
             << "\"chunked_expanded_kv_max_chunks_per_layer\":"
             << chunked_max_chunks_per_layer << ","
             << "\"chunked_compact_mlx_reserved_bytes\":" << chunked_compact_mlx_reserved_bytes << ","
             << "\"chunked_compact_mlx_active_bytes\":" << logical_compact_kv_bytes_active << ","
             << "\"chunked_compact_mlx_chunk_bytes\":" << compact_kv_chunk_bytes << ","
             << "\"chunked_compact_mlx_allocated_chunks_total\":"
             << compact_chunked_allocated_chunks_total << ","
             << "\"chunked_compact_mlx_layers_with_chunks\":"
             << compact_chunked_layers_with_chunks << ","
             << "\"chunked_compact_mlx_max_chunks_per_layer\":"
             << compact_chunked_max_chunks_per_layer << ","
             << "\"compact_kv_bytes_estimate\":" << compact_kv_bytes_estimate << ","
             << "\"resident_projection_bytes_estimate\":" << resident_projection_bytes_estimate << ","
             << "\"resident_layer_group_bytes\":" << resident.total_byte_size << ","
             << "\"active_kv_bytes_estimate\":" << active_kv_bytes_estimate << ","
             << "\"total_estimated_model_runtime_bytes\":" << total_estimated_model_runtime_bytes
             << "},"
             << "\"cache_stats\":" << cache_stats_json() << ","
             << "\"runtime_thread_diagnostics\":{"
             << "\"process_thread_count_start\":" << thread_count_start << ","
             << "\"process_thread_count_after_warmup\":" << thread_count_after_warmup << ","
             << "\"process_thread_count_generation_start\":" << thread_count_generation_start << ","
             << "\"process_thread_count_generation_end\":" << thread_count_generation_end << ","
             << "\"hardware_concurrency\":" << std::thread::hardware_concurrency() << ","
             << "\"mlx_backend_selection\":\"homebrew_mlx_native\","
             << "\"thread_env\":" << runtime_thread_env_json()
             << "},"
             << "\"positions_after\":" << kv_cache.positions_stored << ","
             << "\"logits_len\":" << last_logits_len << ","
             << "\"final_norm_checksum_diagnostics_enabled\":"
             << (collect_generation_checksums ? "true" : "false") << ","
             << "\"first_generated_final_norm_checksum_available\":"
             << (first_generated_final_norm_checksum_available ? "true" : "false") << ","
             << "\"first_generated_final_norm_checksum\":" << first_generated_final_norm_checksum << ","
             << "\"final_norm_checksum_available\":"
             << (final_norm_checksum_available ? "true" : "false") << ","
             << "\"final_norm_checksum\":" << last_final_norm_checksum << ","
             << "\"last_token_backend_report\":" << backend_report_json(
                    last_incremental_result,
                    mlx_quantized_linear_available,
                    fallback_steps_per_token.empty() ? std::vector<std::string>{} : fallback_steps_per_token.back()) << ","
             << "\"last_token_timing_buckets_ms\":" << resident_decode_timing_buckets_json(last_incremental_result) << ","
             << "\"last_token_attention_backends_per_layer\":"
             << attention_backends_json(last_incremental_result) << ","
             << "\"experimental_mlx_attention_enabled\":"
             << (experimental_mlx_attention() ? "true" : "false") << ","
             << "\"experimental_mlx_attention_mode\":\""
             << json_escape(experimental_mlx_attention_mode()) << "\","
             << "\"attention_backend_default\":\"chunked_compact_mlx\","
             << "\"attention_backend_active\":\""
             << (active_attention_mode == "0" ? "compact_cpu" : json_escape(active_attention_mode))
             << "\","
             << "\"chunk_aware_attention_enabled\":"
             << (chunk_aware_attention_enabled() ? "true" : "false") << ","
             << "\"expanded_kv_cache\":{"
             << "\"enabled\":"
             << (active_attention_mode == "expanded_kv" ? "true" : "false") << ","
             << "\"backend\":\""
             << (active_attention_mode == "expanded_kv" ? "mlx_expanded_q_heads" : "disabled")
             << "\","
             << "\"layout\":\"[layers,q_heads,max_seq,head_dim]\","
             << "\"append_strategy\":\"preallocated_slice_update\","
             << "\"concat_used\":false,"
             << "\"layers\":" << layers << ","
             << "\"q_heads\":" << active_rope_config().num_attention_heads << ","
             << "\"kv_heads\":" << active_rope_config().num_key_value_heads << ","
             << "\"head_dim\":" << active_rope_config().head_dim << ","
             << "\"max_seq\":" << kv_cache.max_seq << ","
             << "\"bytes_per_layer\":"
             << (active_rope_config().num_attention_heads *
                 kv_cache.max_seq *
                 active_rope_config().head_dim * 2ULL * sizeof(float)) << ","
             << "\"total_bytes\":" << expanded_kv_bytes_estimate << ","
             << "\"compact_total_bytes\":" << compact_kv_bytes_estimate << ","
             << "\"expansion_factor\":"
             << (active_rope_config().num_key_value_heads == 0
                    ? 0.0
                    : static_cast<double>(active_rope_config().num_attention_heads) /
                          static_cast<double>(active_rope_config().num_key_value_heads))
             << "},"
             << "\"chunked_expanded_kv_cache\":{"
             << "\"enabled\":"
             << (active_attention_mode == "chunked_expanded_kv" ? "true" : "false") << ","
             << "\"backend\":\""
             << (active_attention_mode == "chunked_expanded_kv" ? "mlx_expanded_q_heads_chunked" : "disabled")
             << "\","
             << "\"layout\":\"[layers,chunks,q_heads,chunk_size,head_dim]\","
             << "\"append_strategy\":\"on_demand_preallocated_chunk_slice_update\","
             << "\"concat_used_for_attention\":true,"
             << "\"future_spill_ready\":true,"
             << "\"layers\":" << layers << ","
             << "\"q_heads\":" << active_rope_config().num_attention_heads << ","
             << "\"kv_heads\":" << active_rope_config().num_key_value_heads << ","
             << "\"head_dim\":" << active_rope_config().head_dim << ","
             << "\"chunk_size\":" << chunk_size << ","
             << "\"chunk_bytes\":" << expanded_kv_chunk_bytes << ","
             << "\"allocated_chunks_total\":" << chunked_allocated_chunks_total << ","
             << "\"layers_with_chunks\":" << chunked_layers_with_chunks << ","
             << "\"max_chunks_per_layer\":" << chunked_max_chunks_per_layer << ","
             << "\"active_bytes\":" << chunked_expanded_kv_active_bytes << ","
             << "\"full_preallocated_equivalent_bytes\":" << expanded_kv_bytes_estimate << ","
             << "\"compact_total_bytes\":" << compact_kv_bytes_estimate << ","
             << "\"expansion_factor\":"
             << (active_rope_config().num_key_value_heads == 0
                    ? 0.0
                    : static_cast<double>(active_rope_config().num_attention_heads) /
                          static_cast<double>(active_rope_config().num_key_value_heads))
             << "},"
             << "\"chunked_compact_mlx_cache\":{"
             << "\"enabled\":"
             << (active_attention_mode == "chunked_compact_mlx" ? "true" : "false") << ","
             << "\"backend\":\""
             << (active_attention_mode == "chunked_compact_mlx" ? "mlx_compact_kv_heads_chunked" : "disabled")
             << "\","
             << "\"layout\":\"[layers,chunks,kv_heads,chunk_size,head_dim]\","
             << "\"attention_expansion\":\"repeat_kv_heads_to_q_heads_at_attention_time\","
             << "\"append_strategy\":\"on_demand_preallocated_chunk_slice_update\","
             << "\"concat_used_for_attention\":true,"
             << "\"layers\":" << layers << ","
             << "\"q_heads\":" << active_rope_config().num_attention_heads << ","
             << "\"kv_heads\":" << active_rope_config().num_key_value_heads << ","
             << "\"head_dim\":" << active_rope_config().head_dim << ","
             << "\"chunk_size\":" << chunk_size << ","
             << "\"chunk_bytes\":" << compact_kv_chunk_bytes << ","
             << "\"allocated_chunks_total\":" << compact_chunked_allocated_chunks_total << ","
             << "\"layers_with_chunks\":" << compact_chunked_layers_with_chunks << ","
             << "\"max_chunks_per_layer\":" << compact_chunked_max_chunks_per_layer << ","
             << "\"active_bytes\":" << logical_compact_kv_bytes_active << ","
             << "\"reserved_bytes\":" << chunked_compact_mlx_reserved_bytes << ","
             << "\"expanded_equivalent_reserved_bytes\":"
             << (compact_chunked_allocated_chunks_total *
                 active_rope_config().num_attention_heads *
                 chunk_size *
                 active_rope_config().head_dim * 2ULL * sizeof(float)) << ","
             << "\"expansion_factor\":"
             << (active_rope_config().num_key_value_heads == 0
                    ? 0.0
                    : static_cast<double>(active_rope_config().num_attention_heads) /
                          static_cast<double>(active_rope_config().num_key_value_heads))
             << "},"
             << "\"last_token_projection_timing_breakdown_ms\":" << projection_timing_breakdown_json(last_incremental_result) << ","
             << "\"resident_groups_persistent_across_probe\":false,"
             << "\"provisional\":true"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)prompt_token_id;
  (void)first_decode_token_id;
  (void)generated_tokens;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_warm_resident_session_json(
    const char* session,
    const char* model_dir,
    const char* adapter_dir) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (session == nullptr || model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string session_str(session);
    const std::string model_dir_str(model_dir);
    const std::string adapter_dir_str(adapter_dir == nullptr ? "" : adapter_dir);
    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    std::uint64_t layers = 0;
    if (!extract_json_u64_value(config_json, "num_hidden_layers", layers) || layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"num_hidden_layers missing\"}";
      return output.c_str();
    }

    std::lock_guard<std::mutex> lock(native_resident_session_mutex());
    auto existing = native_resident_session_table().find(session_str);
    if (existing != native_resident_session_table().end() && existing->second.warmed) {
      std::ostringstream out;
      out << "{"
          << "\"ok\":true,"
          << "\"session\":\"" << json_escape(session_str) << "\","
          << "\"warmed\":true,"
          << "\"reused\":true,"
          << "\"warmup_ms\":0,"
          << "\"first_warmup_ms\":" << existing->second.first_warmup_ms << ","
          << "\"adapter_path\":\"" << json_escape(existing->second.adapter_dir) << "\""
          << "}";
      output = out.str();
      return output.c_str();
    }

    bool loaded_now = false;
    ResidentModelLayers& resident = resident_layers_for(model_dir_str, layers, loaded_now);
    NativeResidentSessionRecord record;
    record.session = session_str;
    record.model_dir = model_dir_str;
    record.adapter_dir = adapter_dir_str;
    auto final_norm_spec = group_load_spec_for("model.norm");
    if (!final_norm_spec || !load_tensor_group_record(model_dir_str, *final_norm_spec, record.final_norm)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load final norm\"}";
      return output.c_str();
    }
    auto embedding_spec = embedding_group_spec_for();
    if (!embedding_spec || !load_tensor_group_record(model_dir_str, *embedding_spec, record.embedding)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load tied embedding projection\"}";
      return output.c_str();
    }
    const auto warmup_start = std::chrono::steady_clock::now();
    warm_resident_mlx_projection_arrays(resident, record.embedding, &record.first_warmup_timing);
    if (!adapter_dir_str.empty()) {
      (void)load_native_adapter_record(adapter_dir_str);
    }
    record.first_warmup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - warmup_start).count();
    record.warmed = true;
    native_resident_session_table()[session_str] = std::move(record);
    const auto& stored = native_resident_session_table().at(session_str);
    std::ostringstream out;
    out << "{"
        << "\"ok\":true,"
        << "\"session\":\"" << json_escape(session_str) << "\","
        << "\"warmed\":true,"
        << "\"reused\":false,"
        << "\"adapter_path\":\"" << json_escape(stored.adapter_dir) << "\","
        << "\"warmup_ms\":" << stored.first_warmup_ms << ","
        << "\"timing_ms\":{"
        << "\"enumerate_groups_ms\":" << stored.first_warmup_timing.enumerate_groups_ms << ","
        << "\"mmap_setup_ms\":" << stored.first_warmup_timing.mmap_setup_ms << ","
        << "\"mlx_array_construction_ms\":" << stored.first_warmup_timing.mlx_array_construction_ms << ","
        << "\"first_eval_compile_warmup_ms\":" << stored.first_warmup_timing.first_eval_compile_warmup_ms << ","
        << "\"metadata_cache_storage_ms\":" << stored.first_warmup_timing.metadata_cache_storage_ms
        << "}"
        << "}";
    output = out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)session;
  (void)model_dir;
  (void)adapter_dir;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_generate_tokens_for_session_json(
    const char* session,
    const char* model_dir,
    const char* adapter_dir,
    unsigned long long prompt_token_id,
    const char* prompt_token_ids_csv,
    unsigned long long first_decode_token_id,
    unsigned long long generated_tokens,
    double temperature,
    unsigned long long top_k,
    double top_p,
    unsigned long long seed,
    int stop_on_eos,
    unsigned long long eos_token_id,
    const char* stop_token_ids_csv) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  std::shared_ptr<NativeAdapterRecord> previous_adapter = active_generation_adapter;
  try {
    if (session == nullptr || model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    {
      std::lock_guard<std::mutex> lock(native_resident_session_mutex());
      auto it = native_resident_session_table().find(session);
      if (it == native_resident_session_table().end() || !it->second.warmed) {
        output = "{\"ok\":false,\"error\":\"session_not_warmed\"}";
        return output.c_str();
      }
      const std::string session_adapter_dir = it->second.adapter_dir;
      const std::string requested_adapter_dir(adapter_dir == nullptr ? "" : adapter_dir);
      const std::string active_adapter_dir =
          requested_adapter_dir.empty() ? session_adapter_dir : requested_adapter_dir;
      active_generation_adapter = active_adapter_dir.empty()
          ? nullptr
          : load_native_adapter_record(active_adapter_dir);
    }
    if (top_p != 1.0) {
      output = "{\"ok\":false,\"error\":\"unsupported_sampling\",\"message\":\"top_p sampling is not implemented\"}";
      active_generation_adapter = previous_adapter;
      return output.c_str();
    }
    SamplingConfig previous_sampling_config = generation_sampling_config;
    GenerationStopConfig previous_stop_config = generation_stop_config;
    generation_sampling_config.enabled = temperature > 0.0 && top_k > 0;
    generation_sampling_config.temperature = temperature;
    generation_sampling_config.top_k = static_cast<std::uint64_t>(top_k);
    generation_sampling_config.top_p = top_p;
    generation_sampling_config.seed = static_cast<std::uint64_t>(seed);
    generation_sampling_config.rng.seed(generation_sampling_config.seed);
    generation_stop_config.stop_on_eos = stop_on_eos != 0;
    generation_stop_config.eos_token_id = static_cast<std::uint64_t>(eos_token_id);
    generation_stop_config.stop_token_ids = parse_u64_csv(stop_token_ids_csv);
    // Reuse the resident global model/projection cache. The fastsmoke helper now
    // proves the second warmup is idempotent and near-zero in this process.
    output = rusty_mlx_fastsmoke_generation_probe_json(
        model_dir,
        prompt_token_id,
        prompt_token_ids_csv,
        first_decode_token_id,
        generated_tokens);
    generation_sampling_config = previous_sampling_config;
    generation_stop_config = previous_stop_config;
    active_generation_adapter = previous_adapter;
  } catch (...) {
    active_generation_adapter = previous_adapter;
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)session;
  (void)model_dir;
  (void)adapter_dir;
  (void)prompt_token_id;
  (void)prompt_token_ids_csv;
  (void)first_decode_token_id;
  (void)generated_tokens;
  (void)temperature;
  (void)top_k;
  (void)top_p;
  (void)seed;
  (void)stop_on_eos;
  (void)eos_token_id;
  (void)stop_token_ids_csv;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

void rusty_mlx_free_resident_session(const char* session) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  if (session == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(native_resident_session_mutex());
  native_resident_session_table().erase(session);
#else
  (void)session;
#endif
}

const char* rusty_mlx_resident_incremental_timing_breakdown_probe_json(
    const char* model_dir,
    unsigned long long decode_token_id) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);
    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }
    std::uint64_t layers = 0;
    if (!extract_json_u64_value(config_json, "num_hidden_layers", layers) || layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"num_hidden_layers missing\"}";
      return output.c_str();
    }

    bool loaded_now = false;
    ResidentModelLayers& resident = resident_layers_for(model_dir_str, layers, loaded_now);
    if (loaded_now) {
      output = "{\"ok\":false,\"error\":\"bad_state\",\"message\":\"resident layer groups were not already loaded\"}";
      return output.c_str();
    }

    NativeSessionKvCache kv_cache;
    {
      std::lock_guard<std::mutex> lock(native_kv_cache_mutex());
      auto it = native_kv_cache_table().find("session_layer_residency_probe");
      if (it == native_kv_cache_table().end()) {
        it = native_kv_cache_table().find("full_stack_single_token_probe");
      }
      if (it == native_kv_cache_table().end()) {
        output = "{\"ok\":false,\"error\":\"missing_cache\",\"message\":\"no resident KV cache available\"}";
        return output.c_str();
      }
      kv_cache = it->second;
    }
    const std::uint64_t positions_before = kv_cache.positions_stored;

    TensorGroupRecord final_norm;
    auto final_norm_spec = group_load_spec_for("model.norm");
    if (!final_norm_spec || !load_tensor_group_record(model_dir_str, *final_norm_spec, final_norm)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load final norm\"}";
      return output.c_str();
    }
    TensorGroupRecord embedding;
    auto embedding_spec = embedding_group_spec_for();
    if (!embedding_spec || !load_tensor_group_record(model_dir_str, *embedding_spec, embedding)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load tied embedding projection\"}";
      return output.c_str();
    }

    double input_rmsnorm_ms = 0.0;
    double qkv_projection_ms = 0.0;
    double attention_math_ms = 0.0;
    double o_projection_ms = 0.0;
    double post_attention_rmsnorm_ms = 0.0;
    double gate_up_projection_ms = 0.0;
    double silu_multiply_ms = 0.0;
    double down_projection_ms = 0.0;
    double final_rmsnorm_ms = 0.0;
    double logits_projection_ms = 0.0;
    std::vector<double> per_layer_ms;
    per_layer_ms.reserve(static_cast<std::size_t>(layers));

    auto total_start = std::chrono::steady_clock::now();
    std::vector<float> current_values =
        layer0_embedding_values(model_dir_str, static_cast<std::size_t>(decode_token_id));

    for (std::uint64_t layer = 0; layer < layers; ++layer) {
      const auto layer_start = std::chrono::steady_clock::now();
      const ResidentLayerGroups& groups = resident.layers[static_cast<std::size_t>(layer)];
      if (static_cast<std::size_t>(layer) >= kv_cache.layers.size()) {
        throw std::runtime_error("resident timing KV cache missing layer");
      }

      auto segment_start = std::chrono::steady_clock::now();
      std::vector<float> input_norm_values = rmsnorm_values(groups.input_norm, current_values, eps);
      auto segment_end = std::chrono::steady_clock::now();
      input_rmsnorm_ms += std::chrono::duration<double, std::milli>(segment_end - segment_start).count();

      segment_start = std::chrono::steady_clock::now();
      std::vector<float> q_values = quantized_linear_vector_values(groups.q_proj, input_norm_values);
      std::vector<float> k_values = quantized_linear_vector_values(groups.k_proj, input_norm_values);
      std::vector<float> v_values = quantized_linear_vector_values(groups.v_proj, input_norm_values);
      segment_end = std::chrono::steady_clock::now();
      qkv_projection_ms += std::chrono::duration<double, std::milli>(segment_end - segment_start).count();

      segment_start = std::chrono::steady_clock::now();
      std::vector<float> attention_values = cached_single_token_attention_values(
          q_values,
          kv_cache.layers[static_cast<std::size_t>(layer)],
          k_values,
          v_values);
      append_native_kv(&kv_cache, static_cast<std::size_t>(layer), k_values, v_values);
      segment_end = std::chrono::steady_clock::now();
      attention_math_ms += std::chrono::duration<double, std::milli>(segment_end - segment_start).count();

      segment_start = std::chrono::steady_clock::now();
      std::vector<float> o_values = quantized_linear_vector_values(groups.o_proj, attention_values);
      segment_end = std::chrono::steady_clock::now();
      o_projection_ms += std::chrono::duration<double, std::milli>(segment_end - segment_start).count();
      if (o_values.size() != current_values.size()) {
        throw std::runtime_error("resident timing o_proj output does not match residual width");
      }
      std::vector<float> attention_residual(o_values.size(), 0.0f);
      for (std::size_t i = 0; i < o_values.size(); ++i) {
        attention_residual[i] = current_values[i] + o_values[i];
      }

      segment_start = std::chrono::steady_clock::now();
      std::vector<float> post_norm_values = rmsnorm_values(groups.post_attention_norm, attention_residual, eps);
      segment_end = std::chrono::steady_clock::now();
      post_attention_rmsnorm_ms += std::chrono::duration<double, std::milli>(segment_end - segment_start).count();

      segment_start = std::chrono::steady_clock::now();
      std::vector<float> gate_values = quantized_linear_vector_values(groups.gate_proj, post_norm_values);
      std::vector<float> up_values = quantized_linear_vector_values(groups.up_proj, post_norm_values);
      segment_end = std::chrono::steady_clock::now();
      gate_up_projection_ms += std::chrono::duration<double, std::milli>(segment_end - segment_start).count();
      if (gate_values.size() != up_values.size()) {
        throw std::runtime_error("resident timing gate/up projection sizes differ");
      }

      segment_start = std::chrono::steady_clock::now();
      std::vector<float> activated_values(gate_values.size(), 0.0f);
      for (std::size_t i = 0; i < activated_values.size(); ++i) {
        const float gate = gate_values[i];
        const float silu = gate / (1.0f + std::exp(-gate));
        activated_values[i] = silu * up_values[i];
      }
      segment_end = std::chrono::steady_clock::now();
      silu_multiply_ms += std::chrono::duration<double, std::milli>(segment_end - segment_start).count();

      segment_start = std::chrono::steady_clock::now();
      std::vector<float> down_values = quantized_linear_vector_values(groups.down_proj, activated_values);
      segment_end = std::chrono::steady_clock::now();
      down_projection_ms += std::chrono::duration<double, std::milli>(segment_end - segment_start).count();
      if (down_values.size() != attention_residual.size()) {
        throw std::runtime_error("resident timing down_proj output does not match residual width");
      }
      current_values.assign(down_values.size(), 0.0f);
      for (std::size_t i = 0; i < down_values.size(); ++i) {
        current_values[i] = attention_residual[i] + down_values[i];
      }

      const auto layer_end = std::chrono::steady_clock::now();
      per_layer_ms.push_back(std::chrono::duration<double, std::milli>(layer_end - layer_start).count());
    }

    auto segment_start = std::chrono::steady_clock::now();
    std::vector<float> final_norm_values = rmsnorm_values(final_norm, current_values, eps);
    auto segment_end = std::chrono::steady_clock::now();
    final_rmsnorm_ms = std::chrono::duration<double, std::milli>(segment_end - segment_start).count();

    segment_start = std::chrono::steady_clock::now();
    std::vector<float> logits = quantized_linear_vector_values(embedding, final_norm_values);
    segment_end = std::chrono::steady_clock::now();
    logits_projection_ms = std::chrono::duration<double, std::milli>(segment_end - segment_start).count();
    std::vector<std::pair<std::uint64_t, float>> top = top_logits(logits, 1);
    const auto total_end = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    double min_layer = per_layer_ms.empty() ? 0.0 : per_layer_ms[0];
    double max_layer = per_layer_ms.empty() ? 0.0 : per_layer_ms[0];
    double sum_layer = 0.0;
    std::size_t slowest_layer = 0;
    for (std::size_t i = 0; i < per_layer_ms.size(); ++i) {
      const double value = per_layer_ms[i];
      min_layer = std::min(min_layer, value);
      if (value > max_layer) {
        max_layer = value;
        slowest_layer = i;
      }
      sum_layer += value;
    }
    const double avg_layer = per_layer_ms.empty() ? 0.0 : sum_layer / static_cast<double>(per_layer_ms.size());

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"decode_token_id\":" << decode_token_id << ","
             << "\"layers_run\":" << layers << ","
             << "\"positions_before\":" << positions_before << ","
             << "\"positions_after\":" << kv_cache.positions_stored << ","
             << "\"logits_len\":" << logits.size() << ","
             << "\"top_token_id\":" << (top.empty() ? 0 : top[0].first) << ","
             << "\"top_token_score\":" << (top.empty() ? 0.0f : top[0].second) << ","
             << "\"resident_groups_reused\":true,"
             << "\"optimized_incremental_attention\":true,"
             << "\"timings_ms\":{"
             << "\"input_rmsnorm\":" << input_rmsnorm_ms << ","
             << "\"qkv_projections\":" << qkv_projection_ms << ","
             << "\"attention_math\":" << attention_math_ms << ","
             << "\"o_projection\":" << o_projection_ms << ","
             << "\"post_attention_rmsnorm\":" << post_attention_rmsnorm_ms << ","
             << "\"gate_up_projections\":" << gate_up_projection_ms << ","
             << "\"silu_multiply\":" << silu_multiply_ms << ","
             << "\"down_projection\":" << down_projection_ms << ","
             << "\"final_rmsnorm\":" << final_rmsnorm_ms << ","
             << "\"logits_projection\":" << logits_projection_ms << ","
             << "\"total\":" << total_ms
             << "},"
             << "\"per_layer_timing_ms\":{"
             << "\"min\":" << min_layer << ","
             << "\"max\":" << max_layer << ","
             << "\"average\":" << avg_layer << ","
             << "\"slowest_layer_index\":" << slowest_layer << ","
             << "\"slowest_layer_timing\":" << max_layer
             << "},"
             << "\"provisional\":true,"
             << "\"notes\":["
             << "\"measurement-only resident incremental timing breakdown\","
             << "\"resident layer groups are reused and no layer tensor groups are reloaded\","
             << "\"no optimization is applied by this probe\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)decode_token_id;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_quantized_linear_kernel_probe_json(const char* model_dir) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);
    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    std::uint64_t layers = 0;
    if (!extract_json_u64_value(config_json, "num_hidden_layers", layers) || layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"num_hidden_layers missing\"}";
      return output.c_str();
    }
    bool loaded_now = false;
    ResidentModelLayers& resident = resident_layers_for(model_dir_str, layers, loaded_now);
    if (resident.layers.empty()) {
      output = "{\"ok\":false,\"error\":\"bad_state\",\"message\":\"resident layer 0 missing\"}";
      return output.c_str();
    }
    const ResidentLayerGroups& layer0 = resident.layers[0];

    auto deterministic_input = [](std::size_t length) {
      std::vector<float> input(length, 0.0f);
      for (std::size_t i = 0; i < length; ++i) {
        input[i] = static_cast<float>((static_cast<int>(i % 17) - 8) / 8.0);
      }
      return input;
    };
    auto logical_width = [](const TensorGroupRecord& record) {
      const auto weight_it = record.tensors.find("weight");
      if (weight_it == record.tensors.end() || weight_it->second.shape.size() < 2) {
        throw std::runtime_error("kernel probe group missing weight shape");
      }
      return static_cast<std::size_t>(weight_it->second.shape[1]) * 8;
    };
    auto run_one = [&](const char* name, const TensorGroupRecord& group) {
      const std::vector<float> input = deterministic_input(logical_width(group));
      const auto existing_start = std::chrono::steady_clock::now();
      std::vector<float> existing = quantized_linear_vector_values(group, input);
      const auto existing_end = std::chrono::steady_clock::now();
      const auto optimized_start = std::chrono::steady_clock::now();
      std::vector<float> optimized = quantized_linear_vector_values_row_optimized(group, input);
      const auto optimized_end = std::chrono::steady_clock::now();
      if (existing.size() != optimized.size()) {
        throw std::runtime_error("kernel probe output length mismatch");
      }
      double max_abs_diff = 0.0;
      for (std::size_t i = 0; i < existing.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, std::abs(static_cast<double>(existing[i]) - static_cast<double>(optimized[i])));
      }
      const double timing_existing_ms =
          std::chrono::duration<double, std::milli>(existing_end - existing_start).count();
      const double timing_optimized_ms =
          std::chrono::duration<double, std::milli>(optimized_end - optimized_start).count();
      const double speedup = timing_optimized_ms > 0.0
          ? timing_existing_ms / timing_optimized_ms
          : 0.0;
      std::ostringstream item;
      item << "{"
           << "\"group\":\"" << name << "\","
           << "\"output_len\":" << existing.size() << ","
           << "\"max_abs_diff\":" << max_abs_diff << ","
           << "\"checksum_existing\":" << vector_checksum(existing) << ","
           << "\"checksum_optimized\":" << vector_checksum(optimized) << ","
           << "\"timing_existing_ms\":" << timing_existing_ms << ","
           << "\"timing_optimized_ms\":" << timing_optimized_ms << ","
           << "\"speedup\":" << speedup
           << "}";
      return item.str();
    };

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"loaded_resident_layers_now\":" << (loaded_now ? "true" : "false") << ","
             << "\"measurement_only\":true,"
             << "\"optimized_path_applied_to_generation\":false,"
             << "\"groups\":["
             << run_one("layer0.gate_proj", layer0.gate_proj) << ","
             << run_one("layer0.up_proj", layer0.up_proj) << ","
             << run_one("layer0.down_proj", layer0.down_proj) << ","
             << run_one("layer0.q_proj", layer0.q_proj)
             << "],"
             << "\"notes\":["
             << "\"compares current scalar row path against row-block optimized CPU path\","
             << "\"optimized path is measurement-only until output differences are accepted\","
             << "\"generation behavior is unchanged\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_quantized_linear_mlx_probe_json(const char* model_dir) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);
    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    std::uint64_t layers = 0;
    if (!extract_json_u64_value(config_json, "num_hidden_layers", layers) || layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"num_hidden_layers missing\"}";
      return output.c_str();
    }

    bool loaded_now = false;
    ResidentModelLayers& resident = resident_layers_for(model_dir_str, layers, loaded_now);
    if (resident.layers.empty()) {
      output = "{\"ok\":false,\"error\":\"bad_state\",\"message\":\"resident layer 0 missing\"}";
      return output.c_str();
    }
    const TensorGroupRecord& group = resident.layers[0].q_proj;
    if (!group.quantized_layout_cached) {
      output = "{\"ok\":false,\"error\":\"bad_state\",\"message\":\"q_proj layout metadata not cached\"}";
      return output.c_str();
    }

    std::vector<float> input(group.quantized_logical_width, 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) {
      input[i] = static_cast<float>((static_cast<int>(i % 17) - 8) / 8.0);
    }

    const auto cpu_start = std::chrono::steady_clock::now();
    std::vector<float> cpu = quantized_linear_vector_values_layout_cached(group, input);
    const auto cpu_end = std::chrono::steady_clock::now();

    bool mlx_available = false;
    bool applied = false;
    std::vector<float> mlx_values;
    std::string mlx_error;
    double timing_mlx_ms = 0.0;
    try {
      const auto mlx_start = std::chrono::steady_clock::now();
      mlx_values = quantized_linear_vector_values_mlx(group, input);
      const auto mlx_end = std::chrono::steady_clock::now();
      timing_mlx_ms = std::chrono::duration<double, std::milli>(mlx_end - mlx_start).count();
      mlx_available = true;
    } catch (const std::exception& e) {
      mlx_error = e.what();
    } catch (...) {
      mlx_error = "unknown_exception";
    }

    double max_abs_diff = 0.0;
    if (mlx_available && cpu.size() == mlx_values.size()) {
      for (std::size_t i = 0; i < cpu.size(); ++i) {
        max_abs_diff = std::max(
            max_abs_diff,
            std::abs(static_cast<double>(cpu[i]) - static_cast<double>(mlx_values[i])));
      }
      applied = max_abs_diff <= 0.05;
    }
    mlx_quantized_linear_available_flag() = applied;

    const double timing_cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    const double speedup = (mlx_available && timing_mlx_ms > 0.0)
        ? timing_cpu_ms / timing_mlx_ms
        : 0.0;

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"group\":\"layer0.q_proj\","
             << "\"loaded_resident_layers_now\":" << (loaded_now ? "true" : "false") << ","
             << "\"mlx_quantized_linear_available\":" << (mlx_available ? "true" : "false") << ","
             << "\"mlx_quantized_linear_applied_to_probe\":" << (applied ? "true" : "false") << ","
             << "\"mlx_quantized_linear_applied_to_generation\":" << (applied ? "true" : "false") << ","
             << "\"fallback_used\":" << (!applied ? "true" : "false") << ","
             << "\"shape\":{"
             << "\"cpu\":[" << cpu.size() << "],"
             << "\"mlx\":[" << (mlx_available ? mlx_values.size() : 0) << "]"
             << "},"
             << "\"first_values\":{"
             << "\"cpu\":" << json_array_float_to_string(first_values_of(cpu)) << ","
             << "\"mlx\":" << json_array_float_to_string(first_values_of(mlx_values))
             << "},"
             << "\"checksum_cpu\":" << vector_checksum(cpu) << ","
             << "\"checksum_mlx\":" << (mlx_available ? vector_checksum(mlx_values) : 0.0) << ","
             << "\"max_abs_diff\":" << max_abs_diff << ","
             << "\"timing_cpu_ms\":" << timing_cpu_ms << ","
             << "\"timing_mlx_ms\":" << timing_mlx_ms << ","
             << "\"speedup\":" << speedup << ","
             << "\"comparison_tolerance\":0.05,"
             << "\"mlx_error\":\"" << json_escape(mlx_error) << "\","
             << "\"notes\":["
             << "\"single-projection MLX/Metal quantized_matmul probe only\","
             << "\"CPU layout-cached quantized linear remains the oracle and fallback\","
             << "\"generation may use MLX quantized linear only after this probe marks it available\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  output = "{\"ok\":true,\"group\":\"layer0.q_proj\",\"mlx_quantized_linear_available\":false,"
           "\"mlx_quantized_linear_applied_to_probe\":false,"
           "\"mlx_quantized_linear_applied_to_generation\":false,"
           "\"fallback_used\":true,\"shape\":{\"cpu\":[0],\"mlx\":[0]},"
           "\"first_values\":{\"cpu\":[],\"mlx\":[]},\"checksum_cpu\":0,"
           "\"checksum_mlx\":0,\"max_abs_diff\":0,\"timing_cpu_ms\":0,"
           "\"timing_mlx_ms\":0,\"speedup\":0,"
           "\"notes\":[\"MLX native link unavailable; CPU path remains fallback\"]}";
#endif
  return output.c_str();
}

const char* rusty_mlx_metal_first_resident_decode_probe_json(
    const char* model_dir,
    unsigned long long prompt_token_id,
    unsigned long long decode_token_id) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);
    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }
    std::uint64_t layers = 0;
    if (!extract_json_u64_value(config_json, "num_hidden_layers", layers) || layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"num_hidden_layers missing\"}";
      return output.c_str();
    }

    bool loaded_now = false;
    ResidentModelLayers& resident = resident_layers_for(model_dir_str, layers, loaded_now);
    TensorGroupRecord final_norm;
    auto final_norm_spec = group_load_spec_for("model.norm");
    if (!final_norm_spec || !load_tensor_group_record(model_dir_str, *final_norm_spec, final_norm)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load final norm\"}";
      return output.c_str();
    }
    TensorGroupRecord embedding;
    auto embedding_spec = embedding_group_spec_for();
    if (!embedding_spec || !load_tensor_group_record(model_dir_str, *embedding_spec, embedding)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load tied embedding projection\"}";
      return output.c_str();
    }

    const bool saved_mlx_available = mlx_quantized_linear_available_flag();
    auto run_one = [&](bool metal_first) {
      mlx_quantized_linear_available_flag() =
          metal_first && saved_mlx_available && !disable_mlx_quantized_linear();
      mlx_quantized_linear_runtime_fallback_used = false;
      mlx_quantized_linear_fallback_steps.clear();

      NativeSessionKvCache kv_cache;
      kv_cache.owner_session = metal_first ? "metal_first_probe" : "cpu_oracle_probe";
      kv_cache.layers.resize(static_cast<std::size_t>(layers));
      kv_cache.layers_allocated = layers;

      std::vector<float> current_values =
          layer0_embedding_values(model_dir_str, static_cast<std::size_t>(prompt_token_id));
      for (std::uint64_t layer = 0; layer < layers; ++layer) {
        const ResidentLayerGroups& groups = resident.layers[static_cast<std::size_t>(layer)];
        std::vector<float> attention_values =
            layer_attention_residual_resident_from_input(
                groups,
                static_cast<std::size_t>(layer),
                current_values,
                eps,
                &kv_cache,
                true,
                true);
        current_values = layer_mlp_residual_resident_from_input(
            groups,
            attention_values,
            eps,
            true,
            true,
            true);
      }

      ResidentDecodeResult result = resident_incremental_decode_once(
          model_dir_str,
          layers,
          eps,
          resident,
          kv_cache,
          decode_token_id,
          final_norm,
          embedding,
          true,
          true,
          true,
          false,
          false,
          false);
      std::vector<std::string> fallbacks = mlx_quantized_linear_fallback_steps;
      return std::make_pair(result, fallbacks);
    };

    auto cpu_pair = run_one(false);
    auto metal_pair = run_one(true);
    mlx_quantized_linear_available_flag() = saved_mlx_available;

    const ResidentDecodeResult& cpu = cpu_pair.first;
    const ResidentDecodeResult& metal = metal_pair.first;
    const std::vector<std::string>& metal_fallbacks = metal_pair.second;
    double max_abs_diff = 0.0;
    if (cpu.logits.size() == metal.logits.size()) {
      for (std::size_t i = 0; i < cpu.logits.size(); ++i) {
        max_abs_diff = std::max(
            max_abs_diff,
            std::abs(static_cast<double>(cpu.logits[i]) - static_cast<double>(metal.logits[i])));
      }
    }
    const double checksum_diff =
        std::abs(cpu.final_norm_checksum - metal.final_norm_checksum);
    const bool same_token = cpu.top_token_id == metal.top_token_id;

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"comparison_only\":true,"
             << "\"prompt_token_id\":" << prompt_token_id << ","
             << "\"decode_token_id\":" << decode_token_id << ","
             << "\"layers_run\":" << layers << ","
             << "\"metal_first_requested\":true,"
             << "\"mlx_quantized_linear_available\":" << (saved_mlx_available ? "true" : "false") << ","
             << "\"cpu_oracle\":{"
             << "\"top_token_id\":" << cpu.top_token_id << ","
             << "\"top_token_score\":" << cpu.top_token_score << ","
             << "\"final_norm_checksum\":" << cpu.final_norm_checksum << ","
             << "\"logits_len\":" << cpu.logits_len << ","
             << "\"timing_ms\":" << cpu.timing_ms << ","
             << "\"backend_report\":" << backend_report_json(cpu, false, cpu_pair.second)
             << "},"
             << "\"metal_first\":{"
             << "\"top_token_id\":" << metal.top_token_id << ","
             << "\"top_token_score\":" << metal.top_token_score << ","
             << "\"final_norm_checksum\":" << metal.final_norm_checksum << ","
             << "\"logits_len\":" << metal.logits_len << ","
             << "\"timing_ms\":" << metal.timing_ms << ","
             << "\"fallback_used\":" << (!metal_fallbacks.empty() ? "true" : "false") << ","
             << "\"fallback_steps\":" << fallback_steps_json() << ","
             << "\"backend_report\":" << backend_report_json(metal, saved_mlx_available, metal_fallbacks)
             << "},"
             << "\"same_token\":" << (same_token ? "true" : "false") << ","
             << "\"final_norm_checksum_diff\":" << checksum_diff << ","
             << "\"max_abs_diff\":" << max_abs_diff << ","
             << "\"notes\":["
             << "\"CPU resident decode remains the oracle\","
             << "\"Metal-first attempts MLX quantized linear locally and falls back per step\","
             << "\"MLX probe speed does not gate this correctness/fallback report\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)prompt_token_id;
  (void)decode_token_id;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_resident_incremental_optimized_compare_probe_json(
    const char* model_dir,
    unsigned long long decode_token_id) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);
    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }
    std::uint64_t layers = 0;
    if (!extract_json_u64_value(config_json, "num_hidden_layers", layers) || layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"num_hidden_layers missing\"}";
      return output.c_str();
    }

    bool loaded_now = false;
    ResidentModelLayers& resident = resident_layers_for(model_dir_str, layers, loaded_now);
    if (loaded_now) {
      output = "{\"ok\":false,\"error\":\"bad_state\",\"message\":\"resident layer groups were not already loaded\"}";
      return output.c_str();
    }

    NativeSessionKvCache base_cache;
    {
      std::lock_guard<std::mutex> lock(native_kv_cache_mutex());
      auto it = native_kv_cache_table().find("session_layer_residency_probe");
      if (it == native_kv_cache_table().end()) {
        it = native_kv_cache_table().find("full_stack_single_token_probe");
      }
      if (it == native_kv_cache_table().end()) {
        output = "{\"ok\":false,\"error\":\"missing_cache\",\"message\":\"no resident KV cache available\"}";
        return output.c_str();
      }
      base_cache = it->second;
    }

    TensorGroupRecord final_norm;
    auto final_norm_spec = group_load_spec_for("model.norm");
    if (!final_norm_spec || !load_tensor_group_record(model_dir_str, *final_norm_spec, final_norm)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load final norm\"}";
      return output.c_str();
    }
    TensorGroupRecord embedding;
    auto embedding_spec = embedding_group_spec_for();
    if (!embedding_spec || !load_tensor_group_record(model_dir_str, *embedding_spec, embedding)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load tied embedding projection\"}";
      return output.c_str();
    }

    ResidentDecodeResult scalar = resident_incremental_decode_once(
        model_dir_str,
        layers,
        eps,
        resident,
        base_cache,
        decode_token_id,
        final_norm,
        embedding,
        false);
    ResidentDecodeResult optimized = resident_incremental_decode_once(
        model_dir_str,
        layers,
        eps,
        resident,
        base_cache,
        decode_token_id,
        final_norm,
        embedding,
        true,
        false);
    if (scalar.logits.size() != optimized.logits.size()) {
      output = "{\"ok\":false,\"error\":\"bad_state\",\"message\":\"scalar/optimized logits length mismatch\"}";
      return output.c_str();
    }
    double max_abs_diff = 0.0;
    for (std::size_t i = 0; i < scalar.logits.size(); ++i) {
      max_abs_diff = std::max(
          max_abs_diff,
          std::abs(static_cast<double>(scalar.logits[i]) - static_cast<double>(optimized.logits[i])));
    }
    const double speedup = optimized.timing_ms > 0.0
        ? scalar.timing_ms / optimized.timing_ms
        : 0.0;

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"decode_token_id\":" << decode_token_id << ","
             << "\"layers_run\":" << layers << ","
             << "\"positions_before\":" << scalar.positions_before << ","
             << "\"positions_after_scalar\":" << scalar.positions_after << ","
             << "\"positions_after_optimized\":" << optimized.positions_after << ","
             << "\"logits_len\":" << scalar.logits.size() << ","
             << "\"max_abs_diff\":" << max_abs_diff << ","
             << "\"top_token_scalar\":{"
             << "\"token_id\":" << scalar.top_token_id << ","
             << "\"score\":" << scalar.top_token_score
             << "},"
             << "\"top_token_optimized\":{"
             << "\"token_id\":" << optimized.top_token_id << ","
             << "\"score\":" << optimized.top_token_score
             << "},"
             << "\"final_norm_checksum_scalar\":" << scalar.final_norm_checksum << ","
             << "\"final_norm_checksum_optimized\":" << optimized.final_norm_checksum << ","
             << "\"timing_scalar_ms\":" << scalar.timing_ms << ","
             << "\"timing_optimized_ms\":" << optimized.timing_ms << ","
             << "\"speedup\":" << speedup << ","
             << "\"optimized_path_applied_to_generation\":false,"
             << "\"comparison_only\":true,"
             << "\"notes\":["
             << "\"resident incremental decode is run both scalar and optimized from the same KV cache snapshot\","
             << "\"optimized row-block linear is applied to q/k/v/o/gate/up/down and tied embedding logits in this comparison\","
             << "\"generation behavior is unchanged until the optimized path is promoted\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)decode_token_id;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_resident_incremental_layout_cached_compare_probe_json(
    const char* model_dir,
    unsigned long long decode_token_id) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);
    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }
    std::uint64_t layers = 0;
    if (!extract_json_u64_value(config_json, "num_hidden_layers", layers) || layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"num_hidden_layers missing\"}";
      return output.c_str();
    }

    bool loaded_now = false;
    ResidentModelLayers& resident = resident_layers_for(model_dir_str, layers, loaded_now);
    if (loaded_now) {
      output = "{\"ok\":false,\"error\":\"bad_state\",\"message\":\"resident layer groups were not already loaded\"}";
      return output.c_str();
    }

    NativeSessionKvCache base_cache;
    {
      std::lock_guard<std::mutex> lock(native_kv_cache_mutex());
      auto it = native_kv_cache_table().find("session_layer_residency_probe");
      if (it == native_kv_cache_table().end()) {
        output = "{\"ok\":false,\"error\":\"missing_cache\",\"message\":\"no resident KV cache available\"}";
        return output.c_str();
      }
      base_cache = it->second;
    }

    TensorGroupRecord final_norm;
    auto final_norm_spec = group_load_spec_for("model.norm");
    if (!final_norm_spec || !load_tensor_group_record(model_dir_str, *final_norm_spec, final_norm)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load final norm\"}";
      return output.c_str();
    }
    TensorGroupRecord embedding;
    auto embedding_spec = embedding_group_spec_for();
    if (!embedding_spec || !load_tensor_group_record(model_dir_str, *embedding_spec, embedding)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load tied embedding projection\"}";
      return output.c_str();
    }

    ResidentDecodeResult old_optimized = resident_incremental_decode_once(
        model_dir_str,
        layers,
        eps,
        resident,
        base_cache,
        decode_token_id,
        final_norm,
        embedding,
        true,
        false);
    ResidentDecodeResult layout_cached = resident_incremental_decode_once(
        model_dir_str,
        layers,
        eps,
        resident,
        base_cache,
        decode_token_id,
        final_norm,
        embedding,
        true,
        true);
    if (old_optimized.logits.size() != layout_cached.logits.size()) {
      output = "{\"ok\":false,\"error\":\"bad_state\",\"message\":\"old/layout-cached logits length mismatch\"}";
      return output.c_str();
    }
    double max_abs_diff = 0.0;
    for (std::size_t i = 0; i < old_optimized.logits.size(); ++i) {
      max_abs_diff = std::max(
          max_abs_diff,
          std::abs(static_cast<double>(old_optimized.logits[i]) -
                   static_cast<double>(layout_cached.logits[i])));
    }
    const double speedup = layout_cached.timing_ms > 0.0
        ? old_optimized.timing_ms / layout_cached.timing_ms
        : 0.0;

    std::ostringstream layout_json;
    if (!resident.layers.empty()) {
      const TensorGroupRecord& q = resident.layers[0].q_proj;
      layout_json << "{"
                  << "\"rows\":" << q.quantized_rows << ","
                  << "\"packed_cols\":" << q.quantized_packed_cols << ","
                  << "\"scale_bias_stride\":" << q.quantized_scale_cols << ","
                  << "\"block_size\":" << q.quantized_block_size << ","
                  << "\"logical_width\":" << q.quantized_logical_width << ","
                  << "\"output_len\":" << q.quantized_output_len << ","
                  << "\"cached\":" << (q.quantized_layout_cached ? "true" : "false")
                  << "}";
    } else {
      layout_json << "{}";
    }

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"decode_token_id\":" << decode_token_id << ","
             << "\"layers_run\":" << layers << ","
             << "\"positions_before\":" << old_optimized.positions_before << ","
             << "\"positions_after_old_optimized\":" << old_optimized.positions_after << ","
             << "\"positions_after_layout_cached\":" << layout_cached.positions_after << ","
             << "\"logits_len\":" << old_optimized.logits.size() << ","
             << "\"max_abs_diff\":" << max_abs_diff << ","
             << "\"top_token_old_optimized\":{"
             << "\"token_id\":" << old_optimized.top_token_id << ","
             << "\"score\":" << old_optimized.top_token_score
             << "},"
             << "\"top_token_layout_cached\":{"
             << "\"token_id\":" << layout_cached.top_token_id << ","
             << "\"score\":" << layout_cached.top_token_score
             << "},"
             << "\"final_norm_checksum_old_optimized\":" << old_optimized.final_norm_checksum << ","
             << "\"final_norm_checksum_layout_cached\":" << layout_cached.final_norm_checksum << ","
             << "\"timing_old_optimized_ms\":" << old_optimized.timing_ms << ","
             << "\"timing_layout_cached_ms\":" << layout_cached.timing_ms << ","
             << "\"speedup\":" << speedup << ","
             << "\"layout_metadata_example\":" << layout_json.str() << ","
             << "\"comparison_only\":true,"
             << "\"optimized_path_applied_to_generation\":true,"
             << "\"layout_cached_path_applied_to_generation\":true,"
             << "\"notes\":["
             << "\"compares previous row-optimized resident decode against layout-cached resident decode\","
             << "\"both paths use the same resident layer groups and KV cache snapshot\","
             << "\"math order is preserved inside each row/block; cached metadata removes repeated shape derivation and checked byte loads\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)decode_token_id;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_resident_incremental_mlp_optimized_compare_probe_json(
    const char* model_dir,
    unsigned long long decode_token_id) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);
    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }
    std::uint64_t layers = 0;
    if (!extract_json_u64_value(config_json, "num_hidden_layers", layers) || layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"num_hidden_layers missing\"}";
      return output.c_str();
    }

    bool loaded_now = false;
    ResidentModelLayers& resident = resident_layers_for(model_dir_str, layers, loaded_now);
    if (loaded_now) {
      output = "{\"ok\":false,\"error\":\"bad_state\",\"message\":\"resident layer groups were not already loaded\"}";
      return output.c_str();
    }

    NativeSessionKvCache base_cache;
    {
      std::lock_guard<std::mutex> lock(native_kv_cache_mutex());
      auto it = native_kv_cache_table().find("session_layer_residency_probe");
      if (it == native_kv_cache_table().end()) {
        output = "{\"ok\":false,\"error\":\"missing_cache\",\"message\":\"no resident KV cache available\"}";
        return output.c_str();
      }
      base_cache = it->second;
    }

    TensorGroupRecord final_norm;
    auto final_norm_spec = group_load_spec_for("model.norm");
    if (!final_norm_spec || !load_tensor_group_record(model_dir_str, *final_norm_spec, final_norm)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load final norm\"}";
      return output.c_str();
    }
    TensorGroupRecord embedding;
    auto embedding_spec = embedding_group_spec_for();
    if (!embedding_spec || !load_tensor_group_record(model_dir_str, *embedding_spec, embedding)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load tied embedding projection\"}";
      return output.c_str();
    }

    ResidentDecodeResult old_layout_cached = resident_incremental_decode_once(
        model_dir_str,
        layers,
        eps,
        resident,
        base_cache,
        decode_token_id,
        final_norm,
        embedding,
        true,
        true,
        false);
    ResidentDecodeResult mlp_optimized = resident_incremental_decode_once(
        model_dir_str,
        layers,
        eps,
        resident,
        base_cache,
        decode_token_id,
        final_norm,
        embedding,
        true,
        true,
        true);
    if (old_layout_cached.logits.size() != mlp_optimized.logits.size()) {
      output = "{\"ok\":false,\"error\":\"bad_state\",\"message\":\"old/mlp-optimized logits length mismatch\"}";
      return output.c_str();
    }
    double max_abs_diff = 0.0;
    for (std::size_t i = 0; i < old_layout_cached.logits.size(); ++i) {
      max_abs_diff = std::max(
          max_abs_diff,
          std::abs(static_cast<double>(old_layout_cached.logits[i]) -
                   static_cast<double>(mlp_optimized.logits[i])));
    }
    const double speedup = mlp_optimized.timing_ms > 0.0
        ? old_layout_cached.timing_ms / mlp_optimized.timing_ms
        : 0.0;

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"decode_token_id\":" << decode_token_id << ","
             << "\"layers_run\":" << layers << ","
             << "\"positions_before\":" << old_layout_cached.positions_before << ","
             << "\"positions_after_old_layout_cached\":" << old_layout_cached.positions_after << ","
             << "\"positions_after_mlp_optimized\":" << mlp_optimized.positions_after << ","
             << "\"logits_len\":" << old_layout_cached.logits.size() << ","
             << "\"max_abs_diff\":" << max_abs_diff << ","
             << "\"top_token_old_layout_cached\":{"
             << "\"token_id\":" << old_layout_cached.top_token_id << ","
             << "\"score\":" << old_layout_cached.top_token_score
             << "},"
             << "\"top_token_mlp_optimized\":{"
             << "\"token_id\":" << mlp_optimized.top_token_id << ","
             << "\"score\":" << mlp_optimized.top_token_score
             << "},"
             << "\"final_norm_checksum_old_layout_cached\":" << old_layout_cached.final_norm_checksum << ","
             << "\"final_norm_checksum_mlp_optimized\":" << mlp_optimized.final_norm_checksum << ","
             << "\"timing_old_layout_cached_ms\":" << old_layout_cached.timing_ms << ","
             << "\"timing_mlp_optimized_ms\":" << mlp_optimized.timing_ms << ","
             << "\"speedup\":" << speedup << ","
             << "\"comparison_only\":true,"
             << "\"optimized_path_applied_to_generation\":true,"
             << "\"layout_cached_path_applied_to_generation\":true,"
             << "\"mlp_pair_optimized_path_applied_to_generation\":true,"
             << "\"notes\":["
             << "\"compares old layout-cached resident decode against paired MLP gate/up resident decode\","
             << "\"both paths use the same resident layer groups and KV cache snapshot\","
             << "\"gate and up outputs keep their per-row math order; only the outer traversal is paired\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)decode_token_id;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_resident_incremental_logits_optimized_compare_probe_json(
    const char* model_dir,
    unsigned long long decode_token_id) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);
    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }
    std::uint64_t layers = 0;
    if (!extract_json_u64_value(config_json, "num_hidden_layers", layers) || layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"num_hidden_layers missing\"}";
      return output.c_str();
    }

    bool loaded_now = false;
    ResidentModelLayers& resident = resident_layers_for(model_dir_str, layers, loaded_now);
    if (loaded_now) {
      output = "{\"ok\":false,\"error\":\"bad_state\",\"message\":\"resident layer groups were not already loaded\"}";
      return output.c_str();
    }

    NativeSessionKvCache base_cache;
    {
      std::lock_guard<std::mutex> lock(native_kv_cache_mutex());
      auto it = native_kv_cache_table().find("session_layer_residency_probe");
      if (it == native_kv_cache_table().end()) {
        output = "{\"ok\":false,\"error\":\"missing_cache\",\"message\":\"no resident KV cache available\"}";
        return output.c_str();
      }
      base_cache = it->second;
    }

    TensorGroupRecord final_norm;
    auto final_norm_spec = group_load_spec_for("model.norm");
    if (!final_norm_spec || !load_tensor_group_record(model_dir_str, *final_norm_spec, final_norm)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load final norm\"}";
      return output.c_str();
    }
    TensorGroupRecord embedding;
    auto embedding_spec = embedding_group_spec_for();
    if (!embedding_spec || !load_tensor_group_record(model_dir_str, *embedding_spec, embedding)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load tied embedding projection\"}";
      return output.c_str();
    }

    ResidentDecodeResult old_mlp_optimized = resident_incremental_decode_once(
        model_dir_str,
        layers,
        eps,
        resident,
        base_cache,
        decode_token_id,
        final_norm,
        embedding,
        true,
        true,
        true,
        false);
    ResidentDecodeResult logits_optimized = resident_incremental_decode_once(
        model_dir_str,
        layers,
        eps,
        resident,
        base_cache,
        decode_token_id,
        final_norm,
        embedding,
        true,
        true,
        true,
        true);
    if (old_mlp_optimized.logits.empty()) {
      output = "{\"ok\":false,\"error\":\"bad_state\",\"message\":\"old MLP-optimized logits missing\"}";
      return output.c_str();
    }
    if (logits_optimized.top_token_id >= old_mlp_optimized.logits.size()) {
      output = "{\"ok\":false,\"error\":\"bad_state\",\"message\":\"top token outside old logits\"}";
      return output.c_str();
    }
    const double max_abs_diff = std::abs(
        static_cast<double>(old_mlp_optimized.logits[static_cast<std::size_t>(logits_optimized.top_token_id)]) -
        static_cast<double>(logits_optimized.top_token_score));
    const double speedup = logits_optimized.timing_ms > 0.0
        ? old_mlp_optimized.timing_ms / logits_optimized.timing_ms
        : 0.0;

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"decode_token_id\":" << decode_token_id << ","
             << "\"layers_run\":" << layers << ","
             << "\"positions_before\":" << old_mlp_optimized.positions_before << ","
             << "\"positions_after_old_mlp_optimized\":" << old_mlp_optimized.positions_after << ","
             << "\"positions_after_logits_optimized\":" << logits_optimized.positions_after << ","
             << "\"logits_len\":" << old_mlp_optimized.logits_len << ","
             << "\"logits_materialized_old\":true,"
             << "\"logits_materialized_optimized\":false,"
             << "\"max_abs_diff\":" << max_abs_diff << ","
             << "\"top_token_old_mlp_optimized\":{"
             << "\"token_id\":" << old_mlp_optimized.top_token_id << ","
             << "\"score\":" << old_mlp_optimized.top_token_score
             << "},"
             << "\"top_token_logits_optimized\":{"
             << "\"token_id\":" << logits_optimized.top_token_id << ","
             << "\"score\":" << logits_optimized.top_token_score
             << "},"
             << "\"final_norm_checksum_old_mlp_optimized\":" << old_mlp_optimized.final_norm_checksum << ","
             << "\"final_norm_checksum_logits_optimized\":" << logits_optimized.final_norm_checksum << ","
             << "\"timing_old_mlp_optimized_ms\":" << old_mlp_optimized.timing_ms << ","
             << "\"timing_logits_optimized_ms\":" << logits_optimized.timing_ms << ","
             << "\"speedup\":" << speedup << ","
             << "\"comparison_only\":true,"
             << "\"optimized_path_applied_to_generation\":true,"
             << "\"layout_cached_path_applied_to_generation\":true,"
             << "\"mlp_pair_optimized_path_applied_to_generation\":true,"
             << "\"logits_top1_optimized_path_applied_to_generation\":true,"
             << "\"notes\":["
             << "\"compares full-logits MLP-optimized resident decode against top-1 logits resident decode\","
             << "\"optimized logits path scans tied embedding rows once and does not materialize the logits vector\","
             << "\"max_abs_diff compares the optimized top score against the same token in the full old logits vector\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)decode_token_id;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_resident_incremental_down_full_block_compare_probe_json(
    const char* model_dir,
    unsigned long long decode_token_id) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);
    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }
    std::uint64_t layers = 0;
    if (!extract_json_u64_value(config_json, "num_hidden_layers", layers) || layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"num_hidden_layers missing\"}";
      return output.c_str();
    }

    bool loaded_now = false;
    ResidentModelLayers& resident = resident_layers_for(model_dir_str, layers, loaded_now);
    if (loaded_now) {
      output = "{\"ok\":false,\"error\":\"bad_state\",\"message\":\"resident layer groups were not already loaded\"}";
      return output.c_str();
    }

    NativeSessionKvCache base_cache;
    {
      std::lock_guard<std::mutex> lock(native_kv_cache_mutex());
      auto it = native_kv_cache_table().find("session_layer_residency_probe");
      if (it == native_kv_cache_table().end()) {
        output = "{\"ok\":false,\"error\":\"missing_cache\",\"message\":\"no resident KV cache available\"}";
        return output.c_str();
      }
      base_cache = it->second;
    }

    TensorGroupRecord final_norm;
    auto final_norm_spec = group_load_spec_for("model.norm");
    if (!final_norm_spec || !load_tensor_group_record(model_dir_str, *final_norm_spec, final_norm)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load final norm\"}";
      return output.c_str();
    }
    TensorGroupRecord embedding;
    auto embedding_spec = embedding_group_spec_for();
    if (!embedding_spec || !load_tensor_group_record(model_dir_str, *embedding_spec, embedding)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load tied embedding projection\"}";
      return output.c_str();
    }

    ResidentDecodeResult current_promoted = resident_incremental_decode_once(
        model_dir_str,
        layers,
        eps,
        resident,
        base_cache,
        decode_token_id,
        final_norm,
        embedding,
        true,
        true,
        true,
        true,
        false,
        false);
    ResidentDecodeResult down_full_block = resident_incremental_decode_once(
        model_dir_str,
        layers,
        eps,
        resident,
        base_cache,
        decode_token_id,
        final_norm,
        embedding,
        true,
        true,
        true,
        true,
        false,
        true);
    const double max_abs_diff = std::abs(
        static_cast<double>(current_promoted.top_token_score) -
        static_cast<double>(down_full_block.top_token_score));
    const double checksum_diff = std::abs(
        current_promoted.final_norm_checksum - down_full_block.final_norm_checksum);
    const double speedup = down_full_block.timing_ms > 0.0
        ? current_promoted.timing_ms / down_full_block.timing_ms
        : 0.0;

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"decode_token_id\":" << decode_token_id << ","
             << "\"layers_run\":" << layers << ","
             << "\"positions_before\":" << current_promoted.positions_before << ","
             << "\"positions_after_current_promoted\":" << current_promoted.positions_after << ","
             << "\"positions_after_down_full_block\":" << down_full_block.positions_after << ","
             << "\"logits_len\":" << current_promoted.logits_len << ","
             << "\"max_abs_diff\":" << max_abs_diff << ","
             << "\"final_norm_checksum_diff\":" << checksum_diff << ","
             << "\"top_token_current_promoted\":{"
             << "\"token_id\":" << current_promoted.top_token_id << ","
             << "\"score\":" << current_promoted.top_token_score
             << "},"
             << "\"top_token_down_full_block\":{"
             << "\"token_id\":" << down_full_block.top_token_id << ","
             << "\"score\":" << down_full_block.top_token_score
             << "},"
             << "\"final_norm_checksum_current_promoted\":" << current_promoted.final_norm_checksum << ","
             << "\"final_norm_checksum_down_full_block\":" << down_full_block.final_norm_checksum << ","
             << "\"timing_current_promoted_ms\":" << current_promoted.timing_ms << ","
             << "\"timing_down_full_block_ms\":" << down_full_block.timing_ms << ","
             << "\"speedup\":" << speedup << ","
             << "\"timing_buckets_current_promoted_ms\":" << resident_decode_timing_buckets_json(current_promoted) << ","
             << "\"timing_buckets_down_full_block_ms\":" << resident_decode_timing_buckets_json(down_full_block) << ","
             << "\"largest_arithmetic_bucket_current_promoted\":\"" << json_escape(current_promoted.largest_arithmetic_bucket) << "\","
             << "\"largest_arithmetic_bucket_down_full_block\":\"" << json_escape(down_full_block.largest_arithmetic_bucket) << "\","
             << "\"comparison_only\":true,"
             << "\"optimized_path_applied_to_generation\":true,"
             << "\"layout_cached_path_applied_to_generation\":true,"
             << "\"mlp_pair_optimized_path_applied_to_generation\":true,"
             << "\"logits_top1_optimized_path_applied_to_generation\":true,"
             << "\"down_full_block_optimized_path_applied_to_generation\":true,"
             << "\"notes\":["
             << "\"compares current promoted resident decode against the down_proj full-block layout-cached path\","
             << "\"full-block path is limited to down_proj and only applies when cached layout metadata proves scale_cols * 8 == packed_cols\","
             << "\"max_abs_diff compares top-token scores because logits are intentionally not materialized in smoke\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)decode_token_id;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_resident_incremental_gate_up_full_block_compare_probe_json(
    const char* model_dir,
    unsigned long long decode_token_id) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);
    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }
    std::uint64_t layers = 0;
    if (!extract_json_u64_value(config_json, "num_hidden_layers", layers) || layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"num_hidden_layers missing\"}";
      return output.c_str();
    }

    bool loaded_now = false;
    ResidentModelLayers& resident = resident_layers_for(model_dir_str, layers, loaded_now);
    if (loaded_now) {
      output = "{\"ok\":false,\"error\":\"bad_state\",\"message\":\"resident layer groups were not already loaded\"}";
      return output.c_str();
    }

    NativeSessionKvCache base_cache;
    {
      std::lock_guard<std::mutex> lock(native_kv_cache_mutex());
      auto it = native_kv_cache_table().find("session_layer_residency_probe");
      if (it == native_kv_cache_table().end()) {
        output = "{\"ok\":false,\"error\":\"missing_cache\",\"message\":\"no resident KV cache available\"}";
        return output.c_str();
      }
      base_cache = it->second;
    }

    TensorGroupRecord final_norm;
    auto final_norm_spec = group_load_spec_for("model.norm");
    if (!final_norm_spec || !load_tensor_group_record(model_dir_str, *final_norm_spec, final_norm)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load final norm\"}";
      return output.c_str();
    }
    TensorGroupRecord embedding;
    auto embedding_spec = embedding_group_spec_for();
    if (!embedding_spec || !load_tensor_group_record(model_dir_str, *embedding_spec, embedding)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load tied embedding projection\"}";
      return output.c_str();
    }

    ResidentDecodeResult current_promoted = resident_incremental_decode_once(
        model_dir_str,
        layers,
        eps,
        resident,
        base_cache,
        decode_token_id,
        final_norm,
        embedding,
        true,
        true,
        true,
        true,
        false,
        false);
    ResidentDecodeResult gate_up_full_block = resident_incremental_decode_once(
        model_dir_str,
        layers,
        eps,
        resident,
        base_cache,
        decode_token_id,
        final_norm,
        embedding,
        true,
        true,
        true,
        true,
        true,
        false);
    const double max_abs_diff = std::abs(
        static_cast<double>(current_promoted.top_token_score) -
        static_cast<double>(gate_up_full_block.top_token_score));
    const double checksum_diff = std::abs(
        current_promoted.final_norm_checksum - gate_up_full_block.final_norm_checksum);
    const double speedup = gate_up_full_block.timing_ms > 0.0
        ? current_promoted.timing_ms / gate_up_full_block.timing_ms
        : 0.0;

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"decode_token_id\":" << decode_token_id << ","
             << "\"layers_run\":" << layers << ","
             << "\"positions_before\":" << current_promoted.positions_before << ","
             << "\"positions_after_current_promoted\":" << current_promoted.positions_after << ","
             << "\"positions_after_gate_up_full_block\":" << gate_up_full_block.positions_after << ","
             << "\"logits_len\":" << current_promoted.logits_len << ","
             << "\"max_abs_diff\":" << max_abs_diff << ","
             << "\"final_norm_checksum_diff\":" << checksum_diff << ","
             << "\"top_token_current_promoted\":{"
             << "\"token_id\":" << current_promoted.top_token_id << ","
             << "\"score\":" << current_promoted.top_token_score
             << "},"
             << "\"top_token_gate_up_full_block\":{"
             << "\"token_id\":" << gate_up_full_block.top_token_id << ","
             << "\"score\":" << gate_up_full_block.top_token_score
             << "},"
             << "\"final_norm_checksum_current_promoted\":" << current_promoted.final_norm_checksum << ","
             << "\"final_norm_checksum_gate_up_full_block\":" << gate_up_full_block.final_norm_checksum << ","
             << "\"timing_current_promoted_ms\":" << current_promoted.timing_ms << ","
             << "\"timing_gate_up_full_block_ms\":" << gate_up_full_block.timing_ms << ","
             << "\"speedup\":" << speedup << ","
             << "\"timing_buckets_current_promoted_ms\":" << resident_decode_timing_buckets_json(current_promoted) << ","
             << "\"timing_buckets_gate_up_full_block_ms\":" << resident_decode_timing_buckets_json(gate_up_full_block) << ","
             << "\"largest_arithmetic_bucket_current_promoted\":\"" << json_escape(current_promoted.largest_arithmetic_bucket) << "\","
             << "\"largest_arithmetic_bucket_gate_up_full_block\":\"" << json_escape(gate_up_full_block.largest_arithmetic_bucket) << "\","
             << "\"comparison_only\":true,"
             << "\"optimized_path_applied_to_generation\":true,"
             << "\"layout_cached_path_applied_to_generation\":true,"
             << "\"mlp_pair_optimized_path_applied_to_generation\":true,"
             << "\"logits_top1_optimized_path_applied_to_generation\":true,"
             << "\"gate_up_full_block_optimized_path_applied_to_generation\":true,"
             << "\"down_full_block_optimized_path_applied_to_generation\":false,"
             << "\"notes\":["
             << "\"compares current promoted resident decode against a gate/up paired full-block layout-cached path\","
             << "\"full-block gate/up path applies only when cached layout metadata proves scale_cols * 8 == packed_cols\","
             << "\"max_abs_diff compares top-token scores because logits are intentionally not materialized in smoke\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)decode_token_id;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_greedy_next_token_probe_json(
    const char* model_dir,
    unsigned long long token_id) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);

    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }
    std::uint64_t layers = 0;
    if (!extract_json_u64_value(config_json, "num_hidden_layers", layers) || layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"num_hidden_layers missing\"}";
      return output.c_str();
    }

    TensorGroupRecord final_norm;
    auto final_norm_spec = group_load_spec_for("model.norm");
    if (!final_norm_spec || !load_tensor_group_record(model_dir_str, *final_norm_spec, final_norm)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load final norm\"}";
      return output.c_str();
    }

    TensorGroupRecord embedding;
    auto embedding_spec = embedding_group_spec_for();
    if (!embedding_spec || !load_tensor_group_record(model_dir_str, *embedding_spec, embedding)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load tied embedding projection\"}";
      return output.c_str();
    }

    auto start = std::chrono::steady_clock::now();
    std::vector<float> current_values =
        layer0_embedding_values(model_dir_str, static_cast<std::size_t>(token_id));
    for (std::uint64_t layer = 0; layer < layers; ++layer) {
      std::vector<float> attention_values =
          layer_attention_residual_from_input(model_dir_str, static_cast<int>(layer), current_values, eps);
      current_values =
          layer_mlp_residual_from_input(model_dir_str, static_cast<int>(layer), attention_values, eps);
    }

    std::vector<float> final_norm_values = rmsnorm_values(final_norm, current_values, eps);
    const double final_norm_checksum = vector_checksum(final_norm_values);
    std::vector<float> logits = quantized_linear_vector_values(embedding, final_norm_values);
    std::vector<std::pair<std::uint64_t, float>> top = top_logits(logits, 1);
    if (top.empty()) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"no logits produced\"}";
      return output.c_str();
    }
    auto end = std::chrono::steady_clock::now();
    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"input_token_id\":" << token_id << ","
             << "\"next_token_id\":" << top[0].first << ","
             << "\"next_token_score\":" << top[0].second << ","
             << "\"layers_run\":" << layers << ","
             << "\"final_norm_checksum\":" << final_norm_checksum << ","
             << "\"logits_len\":" << logits.size() << ","
             << "\"timing_ms\":" << timing_ms << ","
             << "\"provisional\":true,"
             << "\"projection_source\":\"model.embed_tokens\","
             << "\"notes\":["
             << "\"this is a verifier-only greedy next-token probe\","
             << "\"lm_head is absent for model4, so tied embedding projection is used\","
             << "\"no KV cache, generation loop, or production wiring is added\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)token_id;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

const char* rusty_mlx_greedy_two_token_probe_json(
    const char* model_dir,
    unsigned long long token_id) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (model_dir == nullptr) {
      output = "{\"ok\":false,\"error\":\"bad_args\"}";
      return output.c_str();
    }
    const std::string model_dir_str(model_dir);

    std::string config_json;
    if (!read_file_to_string(join_path(model_dir_str, "config.json"), config_json)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"config.json not readable\"}";
      return output.c_str();
    }
    double eps = 1e-6;
    if (!extract_json_double_value(config_json, "rms_norm_eps", eps)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"rms_norm_eps missing\"}";
      return output.c_str();
    }
    std::uint64_t layers = 0;
    if (!extract_json_u64_value(config_json, "num_hidden_layers", layers) || layers == 0) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"num_hidden_layers missing\"}";
      return output.c_str();
    }

    TensorGroupRecord final_norm;
    auto final_norm_spec = group_load_spec_for("model.norm");
    if (!final_norm_spec || !load_tensor_group_record(model_dir_str, *final_norm_spec, final_norm)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load final norm\"}";
      return output.c_str();
    }

    TensorGroupRecord embedding;
    auto embedding_spec = embedding_group_spec_for();
    if (!embedding_spec || !load_tensor_group_record(model_dir_str, *embedding_spec, embedding)) {
      output = "{\"ok\":false,\"error\":\"bad_args\",\"message\":\"failed to load tied embedding projection\"}";
      return output.c_str();
    }

    auto start = std::chrono::steady_clock::now();
    double first_final_norm_checksum = 0.0;
    std::size_t first_logits_len = 0;
    auto first = greedy_next_token_for_input(
        model_dir_str,
        static_cast<std::uint64_t>(token_id),
        layers,
        eps,
        final_norm,
        embedding,
        first_final_norm_checksum,
        first_logits_len);

    double second_final_norm_checksum = 0.0;
    std::size_t second_logits_len = 0;
    auto second = greedy_next_token_for_input(
        model_dir_str,
        first.first,
        layers,
        eps,
        final_norm,
        embedding,
        second_final_norm_checksum,
        second_logits_len);
    auto end = std::chrono::steady_clock::now();
    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"input_token_id\":" << token_id << ","
             << "\"first_next_token_id\":" << first.first << ","
             << "\"first_next_token_score\":" << first.second << ","
             << "\"second_next_token_id\":" << second.first << ","
             << "\"second_next_token_score\":" << second.second << ","
             << "\"layers_run_each_pass\":" << layers << ","
             << "\"logits_len\":" << second_logits_len << ","
             << "\"first_pass_logits_len\":" << first_logits_len << ","
             << "\"first_pass_final_norm_checksum\":" << first_final_norm_checksum << ","
             << "\"second_pass_final_norm_checksum\":" << second_final_norm_checksum << ","
             << "\"timing_total_ms\":" << timing_ms << ","
             << "\"provisional\":true,"
             << "\"projection_source\":\"model.embed_tokens\","
             << "\"notes\":["
             << "\"this is a verifier-only greedy two-token probe\","
             << "\"each token runs an independent full-stack pass without KV cache\","
             << "\"no generation loop or production wiring is added\""
             << "]"
             << "}";
    output = json_out.str();
  } catch (const std::exception& e) {
    output = std::string("{\"ok\":false,\"error\":\"exception\",\"message\":\"") +
             json_escape(e.what()) + "\"}";
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)model_dir;
  (void)token_id;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

unsigned long long rusty_mlx_dequantize_group_slice(
    const char* handle,
    unsigned long long row,
    unsigned long long cols) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (handle == nullptr || cols == 0) {
      return 0;
    }
    std::string handle_str(handle);
    std::lock_guard<std::mutex> lock(tensor_group_mutex());
    auto it = tensor_group_table().find(handle_str);
    if (it == tensor_group_table().end()) {
      return 0;
    }
    const auto& source = it->second;
    if (!source.quantized_group) {
      return 0;
    }

    auto slice = make_dequantized_slice_array(source, static_cast<std::size_t>(row), static_cast<std::size_t>(cols));
    auto array_handle = next_array_handle();
    {
      std::lock_guard<std::mutex> array_lock(array_mutex());
      array_table().emplace(array_handle, ArrayRecord{std::move(slice), source.group});
    }
    return array_handle;
  } catch (...) {
    return 0;
  }
#else
  (void)handle;
  (void)row;
  (void)cols;
  return 0;
#endif
}

int rusty_mlx_load_layer_groups(
    const char* handle,
    const char* model_dir,
    int layer) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (handle == nullptr || model_dir == nullptr || layer < 0) {
      return 0;
    }
    std::string layer_handle(handle);
    std::string model_dir_str(model_dir);
    auto specs = layer_group_specs_for(layer);
    if (specs.empty()) {
      return 0;
    }

    std::unordered_map<std::string, std::string> group_handles;
    std::vector<std::string> created_children;
    std::uint64_t total_byte_size = 0;
    std::uint64_t quantized_group_count = 0;
    std::uint64_t norm_group_count = 0;
    std::vector<std::pair<std::string, TensorGroupRecord>> pending_children;
    pending_children.reserve(specs.size());

    for (const auto& spec : specs) {
      std::string child_handle = next_tensor_group_handle();
      TensorGroupRecord child_record;
      if (!load_tensor_group_record(model_dir_str, spec, child_record)) {
        for (const auto& pending : created_children) {
          std::lock_guard<std::mutex> lock(tensor_group_mutex());
          tensor_group_table().erase(pending);
        }
        return 0;
      }
      total_byte_size += child_record.total_byte_size;
      if (child_record.quantized_group) {
        quantized_group_count += 1;
      } else {
        norm_group_count += 1;
      }
      group_handles.emplace(spec.group.substr(spec.group.find_last_of('.') + 1), child_handle);
      pending_children.emplace_back(child_handle, std::move(child_record));
      created_children.push_back(child_handle);
    }

    {
      std::lock_guard<std::mutex> lock(tensor_group_mutex());
      for (auto& pending : pending_children) {
        tensor_group_table().emplace(pending.first, std::move(pending.second));
      }
    }

    LayerGroupsRecord record;
    record.layer = layer;
    record.source_dir = model_dir_str;
    record.model_dir = model_dir_str;
    record.index_path = join_path(model_dir_str, "model.safetensors.index.json");
    record.loaded = true;
    record.group_handles = group_handles;
    record.total_byte_size = total_byte_size;
    record.quantized_group_count = quantized_group_count;
    record.norm_group_count = norm_group_count;

    {
      std::lock_guard<std::mutex> lock(layer_group_mutex());
      layer_group_table().emplace(layer_handle, std::move(record));
    }

    return 1;
  } catch (...) {
    return 0;
  }
#else
  (void)handle;
  (void)model_dir;
  (void)layer;
  return 0;
#endif
}

const char* rusty_mlx_layer_groups_info_json(const char* handle) {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (handle == nullptr) {
      output = "{\"ok\":false,\"error\":\"unknown_handle\"}";
      return output.c_str();
    }
    std::lock_guard<std::mutex> layer_lock(layer_group_mutex());
    auto layer_it = layer_group_table().find(handle);
    if (layer_it == layer_group_table().end()) {
      output = "{\"ok\":false,\"error\":\"unknown_handle\"}";
      return output.c_str();
    }
    const auto& layer_record = layer_it->second;

    std::lock_guard<std::mutex> tensor_lock(tensor_group_mutex());
    std::ostringstream json_out;
    json_out << "{"
             << "\"ok\":true,"
             << "\"handle\":\"" << json_escape(handle) << "\","
             << "\"layer\":" << layer_record.layer << ","
             << "\"loaded\":true,"
             << "\"model_dir\":\"" << json_escape(layer_record.model_dir) << "\","
             << "\"group_handles\":{";

    bool first = true;
    for (const auto& entry : layer_record.group_handles) {
      if (!first) {
        json_out << ",";
      }
      first = false;
      json_out << "\"" << json_escape(entry.first) << "\":\"" << json_escape(entry.second) << "\"";
    }
    json_out << "},"
             << "\"summary\":{"
             << "\"total_groups\":" << layer_record.group_handles.size() << ","
             << "\"quantized_groups\":" << layer_record.quantized_group_count << ","
             << "\"norm_groups\":" << layer_record.norm_group_count << "},"
             << "\"total_byte_size\":" << layer_record.total_byte_size << ","
             << "\"groups\":{";

    first = true;
    for (const auto& entry : layer_record.group_handles) {
      auto child_it = tensor_group_table().find(entry.second);
      if (child_it == tensor_group_table().end()) {
        output = "{\"ok\":false,\"error\":\"unknown_handle\"}";
        return output.c_str();
      }
      if (!first) {
        json_out << ",";
      }
      first = false;
      const auto& child = child_it->second;
      auto tensor_json = [&](const char* kind) -> std::string {
        auto tensor_it = child.tensors.find(kind);
        if (tensor_it == child.tensors.end()) {
          return "null";
        }
        const auto& tensor = tensor_it->second;
        std::ostringstream out;
        out << "{"
            << "\"dtype\":\"" << json_escape(tensor.dtype) << "\","
            << "\"shape\":" << json_array_u64_to_string(tensor.shape) << ","
            << "\"byte_size\":" << tensor.payload.size() << ","
            << "\"source_file\":\"" << json_escape(tensor.source_file) << "\","
            << "\"byte_offsets\":" << json_array_u64_to_string({tensor.byte_start, tensor.byte_end})
            << "}";
        return out.str();
      };
      json_out << "\"" << json_escape(entry.first) << "\":{"
               << "\"handle\":\"" << json_escape(entry.second) << "\","
               << "\"group\":\"" << json_escape(child.group) << "\","
               << "\"loaded\":true,"
               << "\"quantized_group\":" << (child.quantized_group ? "true" : "false") << ","
               << "\"byte_size\":" << child.total_byte_size << ","
               << "\"weight\":" << tensor_json("weight") << ","
               << "\"scales\":" << tensor_json("scales") << ","
               << "\"biases\":" << tensor_json("biases")
               << "}";
    }
    json_out << "}";
    json_out << "}";
    output = json_out.str();
  } catch (...) {
    output = "{\"ok\":false,\"error\":\"unknown_exception\"}";
  }
#else
  (void)handle;
  output = "{\"ok\":false,\"error\":\"mlx_link_unavailable\"}";
#endif
  return output.c_str();
}

int rusty_mlx_free_layer_groups(const char* handle) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    if (handle == nullptr) {
      return 0;
    }
    std::string layer_handle(handle);
    std::unordered_map<std::string, std::string> children;
    {
      std::lock_guard<std::mutex> lock(layer_group_mutex());
      auto it = layer_group_table().find(layer_handle);
      if (it == layer_group_table().end()) {
        return 0;
      }
      children = it->second.group_handles;
      layer_group_table().erase(it);
    }

    {
      std::lock_guard<std::mutex> lock(tensor_group_mutex());
      for (const auto& entry : children) {
        tensor_group_table().erase(entry.second);
      }
    }
    return 1;
  } catch (...) {
    return 0;
  }
#else
  (void)handle;
  return 0;
#endif
}

const char* rusty_mlx_runtime_diagnose_json() {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  RuntimeProbeResult result = run_runtime_probe();
  std::ostringstream json;
  json << "{"
       << "\"primary_target\":\"" << json_escape(result.primary_target) << "\","
       << "\"primary_success\":" << (result.primary_success ? "true" : "false")
       << ","
       << "\"primary_failure_stage\":\""
       << json_escape(result.primary_failure_stage) << "\","
       << "\"primary_exception\":\"" << json_escape(result.primary_exception)
       << "\","
       << "\"cpu_attempted\":" << (result.cpu_attempted ? "true" : "false")
       << ","
       << "\"cpu_success\":" << (result.cpu_success ? "true" : "false") << ","
       << "\"cpu_failure_stage\":\"" << json_escape(result.cpu_failure_stage)
       << "\","
       << "\"cpu_exception\":\"" << json_escape(result.cpu_exception) << "\""
       << "}";
  output = json.str();
#else
  output =
      "{\"primary_target\":\"none\",\"primary_success\":false,"
      "\"primary_failure_stage\":\"array construction\","
      "\"primary_exception\":\"MLX native link unavailable\","
      "\"cpu_attempted\":false,\"cpu_success\":false,"
      "\"cpu_failure_stage\":\"\",\"cpu_exception\":\"\"}";
#endif
  return output.c_str();
}

}
