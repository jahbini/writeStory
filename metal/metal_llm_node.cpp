// metal_llm_node.cpp — Node.js NAPI wrapper for metal_llm
// --------------------------------------------------------

#include <napi.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <memory>
#include <optional>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <mlx/fast.h>
#include <mlx/transforms.h>
#include <mlx/backend/metal/metal.h>
#include <mlx/memory.h>
#include "metal_llm.h"
#include "direct_metal_probe.h"

namespace fs = std::filesystem;

static std::string ReadTextFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string JsonStringValue(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return "";
    ++pos;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos >= json.size() || json[pos] != '"') return "";
    size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

static bool ParseJsonStringAt(const std::string& text, size_t& pos, std::string& out) {
    if (pos >= text.size() || text[pos] != '"') return false;
    ++pos;
    out.clear();
    while (pos < text.size()) {
        const char c = text[pos++];
        if (c == '"') return true;
        if (c == '\\') {
            if (pos >= text.size()) return false;
            const char e = text[pos++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default: out.push_back(e); break;
            }
        } else {
            out.push_back(c);
        }
    }
    return false;
}

static double JsonNumberValue(const std::string& json, const std::string& key, double fallback) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return fallback;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return fallback;
    ++pos;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos >= json.size() ||
        json[pos] == '"' ||
        json[pos] == '[' ||
        json[pos] == '{' ||
        json[pos] == 'n' ||
        json[pos] == 't' ||
        json[pos] == 'f') {
        return fallback;
    }
    size_t end = pos;
    while (end < json.size()) {
        const char c = json[end];
        if (!(std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')) break;
        ++end;
    }
    if (end == pos) return fallback;
    try {
        return std::stod(json.substr(pos, end - pos));
    } catch (...) {
        return fallback;
    }
}

static Napi::Value JsonScalarToNapi(Napi::Env env, const std::string& json, const std::string& key) {
    const std::string s = JsonStringValue(json, key);
    if (!s.empty()) return Napi::String::New(env, s);
    const double n = JsonNumberValue(json, key, std::numeric_limits<double>::quiet_NaN());
    if (!std::isnan(n)) return Napi::Number::New(env, n);
    return env.Null();
}

static std::uint64_t ReadSafetensorsHeaderLength(std::ifstream& f) {
    unsigned char bytes[8] = {0};
    f.read(reinterpret_cast<char*>(bytes), 8);
    if (f.gcount() != 8) return 0;
    std::uint64_t len = 0;
    for (int i = 0; i < 8; ++i) {
        len |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
    }
    return len;
}

static size_t CountOccurrences(const std::string& text, const std::string& needle) {
    size_t count = 0;
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

static size_t FindMatchingBrace(const std::string& text, size_t open_pos) {
    if (open_pos >= text.size() || text[open_pos] != '{') return std::string::npos;
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = open_pos; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == '{') {
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0) return i;
        }
    }
    return std::string::npos;
}

static bool ContainsName(const std::unordered_set<std::string>& names, const std::string& name) {
    return names.find(name) != names.end();
}

static std::string JsonObjectStringValue(const std::string& object, const std::string& key) {
    return JsonStringValue(object, key);
}

static std::vector<std::uint64_t> JsonArrayU64Value(const std::string& object, const std::string& key) {
    std::vector<std::uint64_t> values;
    const std::string needle = "\"" + key + "\"";
    size_t pos = object.find(needle);
    if (pos == std::string::npos) return values;
    pos = object.find('[', pos + needle.size());
    if (pos == std::string::npos) return values;
    size_t end = object.find(']', pos + 1);
    if (end == std::string::npos) return values;
    std::string body = object.substr(pos + 1, end - pos - 1);
    std::stringstream ss(body);
    std::string part;
    while (std::getline(ss, part, ',')) {
        size_t start = 0;
        while (start < part.size() && std::isspace(static_cast<unsigned char>(part[start]))) ++start;
        size_t stop = part.size();
        while (stop > start && std::isspace(static_cast<unsigned char>(part[stop - 1]))) --stop;
        if (stop <= start) continue;
        try {
            values.push_back(static_cast<std::uint64_t>(std::stoull(part.substr(start, stop - start))));
        } catch (...) {
            values.clear();
            return values;
        }
    }
    return values;
}

struct TensorDescriptor {
    std::string name;
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t offset_begin = 0;
    std::uint64_t offset_end = 0;
    std::uint64_t payload_data_offset = 0;
    std::uint64_t absolute_offset_begin = 0;
    std::uint64_t absolute_offset_end = 0;
    std::uint64_t byte_size = 0;
    std::string source_file;
    std::string source_path;
};

static std::vector<TensorDescriptor> ExtractSafetensorDescriptors(
    const std::string& header,
    const std::string& source_file,
    const std::string& source_path = "",
    std::uint64_t payload_data_offset = 0) {
    std::vector<TensorDescriptor> descriptors;
    size_t pos = 0;
    while ((pos = header.find('"', pos)) != std::string::npos) {
        const size_t key_start = pos + 1;
        const size_t key_end = header.find('"', key_start);
        if (key_end == std::string::npos) break;
        std::string key = header.substr(key_start, key_end - key_start);
        pos = key_end + 1;

        size_t colon = header.find(':', pos);
        if (colon == std::string::npos) break;
        size_t value = colon + 1;
        while (value < header.size() && std::isspace(static_cast<unsigned char>(header[value]))) ++value;
        if (value >= header.size() || header[value] != '{') continue;

        size_t object_end = FindMatchingBrace(header, value);
        if (object_end == std::string::npos) break;
        const std::string object = header.substr(value, object_end - value + 1);
        if (object.find("\"dtype\"") != std::string::npos &&
            object.find("\"shape\"") != std::string::npos &&
            object.find("\"data_offsets\"") != std::string::npos) {
            TensorDescriptor d;
            d.name = key;
            d.dtype = JsonObjectStringValue(object, "dtype");
            d.shape = JsonArrayU64Value(object, "shape");
            const auto offsets = JsonArrayU64Value(object, "data_offsets");
            if (offsets.size() == 2) {
                d.offset_begin = offsets[0];
                d.offset_end = offsets[1];
                d.payload_data_offset = payload_data_offset;
                d.absolute_offset_begin = payload_data_offset + d.offset_begin;
                d.absolute_offset_end = payload_data_offset + d.offset_end;
                d.byte_size = offsets[1] >= offsets[0] ? offsets[1] - offsets[0] : 0;
            }
            d.source_file = source_file;
            d.source_path = source_path;
            descriptors.push_back(std::move(d));
        }
        pos = object_end + 1;
    }
    return descriptors;
}

static Napi::Array U64VectorToNapiArray(Napi::Env env, const std::vector<std::uint64_t>& values) {
    Napi::Array out = Napi::Array::New(env, values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out.Set(static_cast<uint32_t>(i), Napi::Number::New(env, static_cast<double>(values[i])));
    }
    return out;
}

static std::string TensorRoleForName(const std::string& name) {
    if (name == "model.embed_tokens.weight") return "embedding.weight";
    if (name == "model.embed_tokens.scales") return "embedding.scales";
    if (name == "model.embed_tokens.biases") return "embedding.biases";
    if (name == "model.norm.weight") return "final_norm.weight";
    if (name == "lm_head.weight") return "lm_head.weight";
    if (name.find(".self_attn.q_proj.") != std::string::npos) return "attention.q_proj";
    if (name.find(".self_attn.k_proj.") != std::string::npos) return "attention.k_proj";
    if (name.find(".self_attn.v_proj.") != std::string::npos) return "attention.v_proj";
    if (name.find(".self_attn.o_proj.") != std::string::npos) return "attention.o_proj";
    if (name.find(".mlp.gate_proj.") != std::string::npos) return "mlp.gate_proj";
    if (name.find(".mlp.up_proj.") != std::string::npos) return "mlp.up_proj";
    if (name.find(".mlp.down_proj.") != std::string::npos) return "mlp.down_proj";
    if (name.find(".input_layernorm.weight") != std::string::npos) return "layernorm.input";
    if (name.find(".post_attention_layernorm.weight") != std::string::npos) return "layernorm.post_attention";
    if (name.find(".self_attn.q_norm.weight") != std::string::npos) return "layernorm.q_norm";
    if (name.find(".self_attn.k_norm.weight") != std::string::npos) return "layernorm.k_norm";
    return "unknown";
}

static std::string TensorArrayKindForDescriptor(const TensorDescriptor& d) {
    if (d.name.find(".lora_a") != std::string::npos || d.name.find(".lora_b") != std::string::npos) {
        return "lora_dense_matrix";
    }
    if (d.name.find(".weight") != std::string::npos && d.dtype == "U32") {
        return "quantized_packed_weight";
    }
    if (d.name.find(".scales") != std::string::npos) {
        return "quantized_scale";
    }
    if (d.name.find(".biases") != std::string::npos) {
        return "quantized_bias";
    }
    if (d.name.find(".weight") != std::string::npos &&
        (d.name.find("layernorm") != std::string::npos ||
         d.name.find(".q_norm.") != std::string::npos ||
         d.name.find(".k_norm.") != std::string::npos ||
         d.name == "model.norm.weight")) {
        return "norm_weight";
    }
    if (d.name == "lm_head.weight") {
        return "lm_head_weight";
    }
    return "dense_or_unknown";
}

static Napi::Object TensorDescriptorToNapi(Napi::Env env, const TensorDescriptor& d) {
    Napi::Object item = Napi::Object::New(env);
    item.Set("name", Napi::String::New(env, d.name));
    item.Set("role", Napi::String::New(env, TensorRoleForName(d.name)));
    item.Set("dtype", Napi::String::New(env, d.dtype));
    item.Set("shape", U64VectorToNapiArray(env, d.shape));
    item.Set("byte_offsets", U64VectorToNapiArray(env, {d.offset_begin, d.offset_end}));
    item.Set("payload_data_offset", Napi::Number::New(env, static_cast<double>(d.payload_data_offset)));
    item.Set("absolute_byte_offsets", U64VectorToNapiArray(env, {d.absolute_offset_begin, d.absolute_offset_end}));
    item.Set("byte_size", Napi::Number::New(env, static_cast<double>(d.byte_size)));
    item.Set("source_file", Napi::String::New(env, d.source_file));
    item.Set("source_path", Napi::String::New(env, d.source_path));
    return item;
}

static int LayerIndexForTensorName(const std::string& name) {
    const std::string prefix = "model.layers.";
    if (name.rfind(prefix, 0) != 0) return -1;
    const size_t layer_start = prefix.size();
    const size_t layer_end = name.find('.', layer_start);
    if (layer_end == std::string::npos) return -1;
    try {
        return std::stoi(name.substr(layer_start, layer_end - layer_start));
    } catch (...) {
        return -1;
    }
}

static bool ShapeEquals(const std::vector<std::uint64_t>& shape, std::uint64_t a, std::uint64_t b) {
    return shape.size() == 2 && shape[0] == a && shape[1] == b;
}

static std::vector<fs::path> FindSafetensorFiles(const std::string& dir) {
    std::vector<fs::path> safetensors;
    for (const auto& ent : fs::directory_iterator(dir)) {
        if (!ent.is_regular_file()) continue;
        const fs::path p = ent.path();
        if (p.extension() == ".safetensors") safetensors.push_back(p);
    }
    std::sort(safetensors.begin(), safetensors.end());
    return safetensors;
}

static std::vector<TensorDescriptor> LoadTensorDescriptorsFromSafetensors(const std::string& dir) {
    std::vector<TensorDescriptor> descriptors;
    const auto safetensors = FindSafetensorFiles(dir);
    for (const fs::path& p : safetensors) {
        std::ifstream f(p, std::ios::binary);
        if (!f.good()) continue;
        const std::uint64_t header_len = ReadSafetensorsHeaderLength(f);
        if (header_len == 0 || header_len >= (1ULL << 30)) continue;
        std::string header;
        header.resize(static_cast<size_t>(header_len));
        f.read(header.data(), static_cast<std::streamsize>(header.size()));
        const std::uint64_t payload_data_offset = 8 + header_len;
        auto file_descriptors = ExtractSafetensorDescriptors(header, p.filename().string(), p.string(), payload_data_offset);
        descriptors.insert(
            descriptors.end(),
            std::make_move_iterator(file_descriptors.begin()),
            std::make_move_iterator(file_descriptors.end()));
    }
    std::sort(descriptors.begin(), descriptors.end(), [](const auto& a, const auto& b) {
        return a.name < b.name;
    });
    return descriptors;
}

static double TokenIdForAddedTokenContent(const std::string& tokenizer_json, const std::string& content) {
    const std::string needle = "\"content\"";
    size_t content_pos = tokenizer_json.find(needle);
    while (content_pos != std::string::npos) {
        size_t object_start = tokenizer_json.rfind('{', content_pos);
        if (object_start == std::string::npos) return std::numeric_limits<double>::quiet_NaN();
        size_t object_end = FindMatchingBrace(tokenizer_json, object_start);
        if (object_end == std::string::npos) return std::numeric_limits<double>::quiet_NaN();
        const std::string object = tokenizer_json.substr(object_start, object_end - object_start + 1);
        if (JsonStringValue(object, "content") == content) {
            return JsonNumberValue(object, "id", std::numeric_limits<double>::quiet_NaN());
        }
        content_pos = tokenizer_json.find(needle, object_end + 1);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

static std::unordered_map<std::string, std::uint32_t> ParseTokenizerVocab(const std::string& tokenizer_json) {
    std::unordered_map<std::string, std::uint32_t> vocab;
    const size_t model_pos = tokenizer_json.find("\"model\"");
    const size_t vocab_pos = tokenizer_json.find("\"vocab\"", model_pos == std::string::npos ? 0 : model_pos);
    if (vocab_pos == std::string::npos) return vocab;
    const size_t object_start = tokenizer_json.find('{', vocab_pos);
    if (object_start == std::string::npos) return vocab;
    const size_t object_end = FindMatchingBrace(tokenizer_json, object_start);
    if (object_end == std::string::npos || object_end <= object_start) return vocab;

    size_t pos = object_start + 1;
    while (pos < object_end) {
        while (pos < object_end && (std::isspace(static_cast<unsigned char>(tokenizer_json[pos])) || tokenizer_json[pos] == ',')) ++pos;
        if (pos >= object_end || tokenizer_json[pos] != '"') break;
        std::string key;
        if (!ParseJsonStringAt(tokenizer_json, pos, key)) break;
        while (pos < object_end && std::isspace(static_cast<unsigned char>(tokenizer_json[pos]))) ++pos;
        if (pos >= object_end || tokenizer_json[pos] != ':') break;
        ++pos;
        while (pos < object_end && std::isspace(static_cast<unsigned char>(tokenizer_json[pos]))) ++pos;
        size_t number_start = pos;
        while (pos < object_end && std::isdigit(static_cast<unsigned char>(tokenizer_json[pos]))) ++pos;
        if (number_start == pos) break;
        try {
            const auto id = static_cast<std::uint32_t>(std::stoul(tokenizer_json.substr(number_start, pos - number_start)));
            vocab[key] = id;
        } catch (...) {
            break;
        }
    }
    return vocab;
}

static std::string QwenPromptTextToVocabText(const std::string& text) {
    std::string out;
    for (unsigned char c : text) {
        if (c == ' ') out += "Ġ";
        else if (c == '\n') out += "Ċ";
        else out.push_back(static_cast<char>(c));
    }
    return out;
}

static std::string QwenVocabTextToPromptText(const std::string& piece) {
    static const std::unordered_map<std::uint32_t, unsigned char> byte_decoder = [] {
        std::vector<int> bytes;
        for (int b = 33; b <= 126; ++b) bytes.push_back(b);
        for (int b = 161; b <= 172; ++b) bytes.push_back(b);
        for (int b = 174; b <= 255; ++b) bytes.push_back(b);

        std::vector<int> chars = bytes;
        int extra = 0;
        for (int b = 0; b <= 255; ++b) {
            if (std::find(bytes.begin(), bytes.end(), b) == bytes.end()) {
                bytes.push_back(b);
                chars.push_back(256 + extra);
                ++extra;
            }
        }

        std::unordered_map<std::uint32_t, unsigned char> map;
        for (size_t i = 0; i < bytes.size(); ++i) {
            map[static_cast<std::uint32_t>(chars[i])] = static_cast<unsigned char>(bytes[i]);
        }
        return map;
    }();

    std::string out_bytes;
    for (size_t i = 0; i < piece.size();) {
        const unsigned char c = static_cast<unsigned char>(piece[i]);
        std::uint32_t codepoint = 0;
        size_t width = 1;
        if ((c & 0x80U) == 0) {
            codepoint = c;
        } else if ((c & 0xE0U) == 0xC0U && i + 1 < piece.size()) {
            codepoint = ((c & 0x1FU) << 6) |
                (static_cast<unsigned char>(piece[i + 1]) & 0x3FU);
            width = 2;
        } else if ((c & 0xF0U) == 0xE0U && i + 2 < piece.size()) {
            codepoint = ((c & 0x0FU) << 12) |
                ((static_cast<unsigned char>(piece[i + 1]) & 0x3FU) << 6) |
                (static_cast<unsigned char>(piece[i + 2]) & 0x3FU);
            width = 3;
        } else if ((c & 0xF8U) == 0xF0U && i + 3 < piece.size()) {
            codepoint = ((c & 0x07U) << 18) |
                ((static_cast<unsigned char>(piece[i + 1]) & 0x3FU) << 12) |
                ((static_cast<unsigned char>(piece[i + 2]) & 0x3FU) << 6) |
                (static_cast<unsigned char>(piece[i + 3]) & 0x3FU);
            width = 4;
        } else {
            out_bytes.push_back(piece[i++]);
            continue;
        }

        auto it = byte_decoder.find(codepoint);
        if (it == byte_decoder.end()) {
            out_bytes.append(piece.substr(i, width));
        } else {
            out_bytes.push_back(static_cast<char>(it->second));
        }
        i += width;
    }
    return out_bytes;
}

static std::vector<std::uint32_t> GreedyEncodeWithVocab(
    const std::string& text,
    const std::unordered_map<std::string, std::uint32_t>& token_to_id) {
    std::vector<std::uint32_t> tokens;
    const std::string encoded = QwenPromptTextToVocabText(text);
    size_t pos = 0;
    while (pos < encoded.size()) {
        size_t best_len = 0;
        std::uint32_t best_id = 0;
        const size_t max_len = std::min<size_t>(encoded.size() - pos, 64);
        for (size_t len = max_len; len > 0; --len) {
            auto it = token_to_id.find(encoded.substr(pos, len));
            if (it != token_to_id.end()) {
                best_len = len;
                best_id = it->second;
                break;
            }
        }
        if (best_len == 0) {
            auto it = token_to_id.find(encoded.substr(pos, 1));
            if (it == token_to_id.end()) {
                ++pos;
                continue;
            }
            best_len = 1;
            best_id = it->second;
        }
        tokens.push_back(best_id);
        pos += best_len;
    }
    return tokens;
}

static std::string DecodeKnownQwenToken(std::uint32_t token) {
    switch (token) {
        case 0: return "!";
        case 2: return "\"";
        case 3: return "#";
        case 4: return "$";
        case 31: return "@";
        case 368: return " no";
        case 498: return " you";
        case 646: return " can";
        case 785: return "The";
        case 2585: return " help";
        case 3351: return " today";
        case 4340: return "How";
        case 4814: return "can";
        case 9707: return "Hello";
        case 9754: return " world";
        case 14990: return "hello";
        case 151643: return "<|endoftext|>";
        case 151644: return "<|im_start|>";
        case 151645: return "<|im_end|>";
        default:
            return "<token:" + std::to_string(token) + ">";
    }
}

struct GypsyModelRecord {
    std::string handle;
    std::string model_dir;
    Napi::ObjectReference metadata;
    std::vector<TensorDescriptor> tensor_descriptors;
};

struct GypsyTokenizerRecord {
    std::string handle;
    std::string tokenizer_dir;
    Napi::ObjectReference metadata;
    std::unordered_map<std::string, std::uint32_t> token_to_id;
    std::unordered_map<std::uint32_t, std::string> id_to_token;
};

struct GypsyAdapterRecord {
    std::string handle;
    std::string adapter_dir;
    Napi::ObjectReference metadata;
    std::vector<TensorDescriptor> tensor_descriptors;
};

struct MappedFileRecord {
    std::string source_path;
    std::uint64_t file_bytes = 0;
    std::uint64_t descriptor_count = 0;
    std::uint64_t payload_bytes = 0;
    void* mapping = nullptr;
    int fd = -1;

    MappedFileRecord() = default;
    MappedFileRecord(const MappedFileRecord&) = delete;
    MappedFileRecord& operator=(const MappedFileRecord&) = delete;

    MappedFileRecord(MappedFileRecord&& other) noexcept {
        source_path = std::move(other.source_path);
        file_bytes = other.file_bytes;
        descriptor_count = other.descriptor_count;
        payload_bytes = other.payload_bytes;
        mapping = other.mapping;
        fd = other.fd;
        other.mapping = nullptr;
        other.fd = -1;
        other.file_bytes = 0;
        other.descriptor_count = 0;
        other.payload_bytes = 0;
    }

    MappedFileRecord& operator=(MappedFileRecord&& other) noexcept {
        if (this != &other) {
            Close();
            source_path = std::move(other.source_path);
            file_bytes = other.file_bytes;
            descriptor_count = other.descriptor_count;
            payload_bytes = other.payload_bytes;
            mapping = other.mapping;
            fd = other.fd;
            other.mapping = nullptr;
            other.fd = -1;
            other.file_bytes = 0;
            other.descriptor_count = 0;
            other.payload_bytes = 0;
        }
        return *this;
    }

    ~MappedFileRecord() {
        Close();
    }

    void Close() {
        if (mapping != nullptr && mapping != MAP_FAILED && file_bytes > 0) {
            munmap(mapping, static_cast<size_t>(file_bytes));
        }
        mapping = nullptr;
        if (fd >= 0) {
            close(fd);
        }
        fd = -1;
    }
};

struct ResidentArrayRecord {
    std::string name;
    std::string role;
    std::string array_kind;
    std::string dtype;
    std::uint64_t byte_size = 0;
    const void* raw_data = nullptr;
    mlx::core::array array;

    ResidentArrayRecord(
        std::string name_,
        std::string role_,
        std::string array_kind_,
        std::string dtype_,
        std::uint64_t byte_size_,
        const void* raw_data_,
        mlx::core::array array_)
        : name(std::move(name_)),
          role(std::move(role_)),
          array_kind(std::move(array_kind_)),
          dtype(std::move(dtype_)),
          byte_size(byte_size_),
          raw_data(raw_data_),
          array(std::move(array_)) {}
};

struct GypsySessionRecord {
    std::string handle;
    std::string model_handle;
    std::string tokenizer_handle;
    std::string adapter_handle;
    bool warmed = false;
    std::vector<MappedFileRecord> model_mapped_files;
    std::vector<MappedFileRecord> adapter_mapped_files;
    std::vector<ResidentArrayRecord> selected_resident_arrays;
};

static std::unordered_map<std::string, GypsyModelRecord> gypsy_models;
static std::unordered_map<std::string, GypsyTokenizerRecord> gypsy_tokenizers;
static std::unordered_map<std::string, GypsyAdapterRecord> gypsy_adapters;
static std::unordered_map<std::string, GypsySessionRecord> gypsy_sessions;
static std::uint64_t gypsy_next_model_id = 1;
static std::uint64_t gypsy_next_tokenizer_id = 1;
static std::uint64_t gypsy_next_adapter_id = 1;
static std::uint64_t gypsy_next_session_id = 1;

static std::string MakeHandle(const char* prefix, std::uint64_t id) {
    return std::string(prefix) + ":" + std::to_string(id);
}

static Napi::Object GypsyHandleCounts(Napi::Env env) {
    Napi::Object counts = Napi::Object::New(env);
    counts.Set("models", Napi::Number::New(env, static_cast<double>(gypsy_models.size())));
    counts.Set("tokenizers", Napi::Number::New(env, static_cast<double>(gypsy_tokenizers.size())));
    counts.Set("adapters", Napi::Number::New(env, static_cast<double>(gypsy_adapters.size())));
    counts.Set("sessions", Napi::Number::New(env, static_cast<double>(gypsy_sessions.size())));
    return counts;
}

static std::vector<std::string> SessionsReferencingHandle(const std::string& kind, const std::string& handle) {
    std::vector<std::string> sessions;
    for (const auto& kv : gypsy_sessions) {
        const auto& session = kv.second;
        if ((kind == "model" && session.model_handle == handle) ||
            (kind == "tokenizer" && session.tokenizer_handle == handle) ||
            (kind == "adapter" && session.adapter_handle == handle)) {
            sessions.push_back(session.handle);
        }
    }
    std::sort(sessions.begin(), sessions.end());
    return sessions;
}

static bool EnvFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return false;
    std::string s(value);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

static Napi::Array StringVectorToNapiArray(Napi::Env env, const std::vector<std::string>& values) {
    Napi::Array out = Napi::Array::New(env, values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out.Set(static_cast<uint32_t>(i), Napi::String::New(env, values[i]));
    }
    return out;
}

static std::uint64_t MappedFilesByteTotal(const std::vector<MappedFileRecord>& files) {
    std::uint64_t total = 0;
    for (const auto& f : files) {
        total += f.file_bytes;
    }
    return total;
}

static std::uint64_t MappedFilesPayloadByteTotal(const std::vector<MappedFileRecord>& files) {
    std::uint64_t total = 0;
    for (const auto& f : files) {
        total += f.payload_bytes;
    }
    return total;
}

static std::uint64_t ResidentArraysByteTotal(const std::vector<ResidentArrayRecord>& arrays) {
    std::uint64_t total = 0;
    for (const auto& a : arrays) {
        total += a.byte_size;
    }
    return total;
}

static const MappedFileRecord* FindMappedFile(
    const std::vector<MappedFileRecord>* files,
    const std::string& source_path) {
    if (files == nullptr) return nullptr;
    for (const auto& f : *files) {
        if (f.source_path == source_path) return &f;
    }
    return nullptr;
}

static const TensorDescriptor* FindTensorDescriptor(
    const std::vector<TensorDescriptor>& descriptors,
    const std::string& name) {
    for (const auto& d : descriptors) {
        if (d.name == name) return &d;
    }
    return nullptr;
}

static const ResidentArrayRecord* FindResidentArray(
    const std::vector<ResidentArrayRecord>& arrays,
    const std::string& name) {
    for (const auto& a : arrays) {
        if (a.name == name) return &a;
    }
    return nullptr;
}

static bool DtypeFromSafetensors(const std::string& dtype, mlx::core::Dtype& out) {
    if (dtype == "U32") {
        out = mlx::core::uint32;
        return true;
    }
    if (dtype == "BF16") {
        out = mlx::core::bfloat16;
        return true;
    }
    if (dtype == "F32") {
        out = mlx::core::float32;
        return true;
    }
    return false;
}

static bool ShapeFromDescriptor(const TensorDescriptor& d, mlx::core::Shape& out, std::string& error) {
    out.clear();
    for (std::uint64_t dim : d.shape) {
        if (dim > static_cast<std::uint64_t>(std::numeric_limits<mlx::core::ShapeElem>::max())) {
            error = "shape dimension too large for MLX array: " + d.name;
            return false;
        }
        out.push_back(static_cast<mlx::core::ShapeElem>(dim));
    }
    return true;
}

static std::unique_ptr<ResidentArrayRecord> ConstructMappedMlxArray(
    const TensorDescriptor& d,
    const std::vector<MappedFileRecord>& mapped_files,
    std::string& error) {
    const MappedFileRecord* mapped = FindMappedFile(&mapped_files, d.source_path);
    if (mapped == nullptr || mapped->mapping == nullptr || mapped->mapping == MAP_FAILED) {
        error = "missing mapped file for tensor: " + d.name;
        return nullptr;
    }
    if (d.absolute_offset_end > mapped->file_bytes || d.absolute_offset_begin > d.absolute_offset_end) {
        error = "tensor view out of mapped file range: " + d.name;
        return nullptr;
    }

    mlx::core::Dtype dtype = mlx::core::float32;
    if (!DtypeFromSafetensors(d.dtype, dtype)) {
        error = "unsupported safetensors dtype for MLX array construction: " + d.dtype + " tensor: " + d.name;
        return nullptr;
    }

    mlx::core::Shape shape;
    if (!ShapeFromDescriptor(d, shape, error)) {
        return nullptr;
    }

    char* base = static_cast<char*>(mapped->mapping);
    const void* ptr = static_cast<const void*>(base + d.absolute_offset_begin);
    mlx::core::array arr = [&]() {
        if (d.dtype == "U32") {
            const auto* data = static_cast<const std::uint32_t*>(ptr);
            return mlx::core::array(data, shape, mlx::core::uint32);
        }
        if (d.dtype == "BF16") {
            const auto* data = static_cast<const mlx::core::bfloat16_t*>(ptr);
            return mlx::core::array(data, shape, mlx::core::bfloat16);
        }
        const auto* data = static_cast<const float*>(ptr);
        return mlx::core::array(data, shape, mlx::core::float32);
    }();

    return std::make_unique<ResidentArrayRecord>(
        d.name,
        TensorRoleForName(d.name),
        TensorArrayKindForDescriptor(d),
        d.dtype,
        d.byte_size,
        ptr,
        std::move(arr));
}

static bool MapSafetensorFiles(
    const std::vector<TensorDescriptor>& descriptors,
    std::vector<MappedFileRecord>& out,
    std::string& error) {
    struct FileInfo {
        std::string source_path;
        std::uint64_t descriptor_count = 0;
        std::uint64_t payload_bytes = 0;
    };

    std::map<std::string, FileInfo> by_path;
    for (const auto& d : descriptors) {
        if (d.source_path.empty()) continue;
        auto& info = by_path[d.source_path];
        info.source_path = d.source_path;
        info.descriptor_count += 1;
        info.payload_bytes += d.byte_size;
    }

    std::vector<MappedFileRecord> mapped;
    for (const auto& kv : by_path) {
        const auto& info = kv.second;
        if (!fs::exists(info.source_path)) {
            error = "safetensors file does not exist: " + info.source_path;
            return false;
        }
        const auto file_bytes = static_cast<std::uint64_t>(fs::file_size(info.source_path));
        if (file_bytes == 0) {
            error = "safetensors file is empty: " + info.source_path;
            return false;
        }

        int fd = open(info.source_path.c_str(), O_RDONLY);
        if (fd < 0) {
            error = "failed to open safetensors file: " + info.source_path;
            return false;
        }

        void* mapping = mmap(nullptr, static_cast<size_t>(file_bytes), PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapping == MAP_FAILED) {
            close(fd);
            error = "failed to mmap safetensors file: " + info.source_path;
            return false;
        }

        MappedFileRecord record;
        record.source_path = info.source_path;
        record.file_bytes = file_bytes;
        record.descriptor_count = info.descriptor_count;
        record.payload_bytes = info.payload_bytes;
        record.mapping = mapping;
        record.fd = fd;
        mapped.push_back(std::move(record));
    }

    out = std::move(mapped);
    return true;
}

static Napi::Object BuildTensorViewPlan(
    Napi::Env env,
    const std::string& owner,
    const std::vector<TensorDescriptor>& descriptors,
    const std::vector<MappedFileRecord>& mapped_files) {
    std::uint64_t mapped_descriptor_count = 0;
    std::uint64_t missing_mapping_count = 0;
    std::uint64_t out_of_range_count = 0;
    std::uint64_t resolved_payload_bytes = 0;
    std::vector<std::string> errors;
    Napi::Array samples = Napi::Array::New(env);
    uint32_t sample_index = 0;

    const std::vector<std::string> preferred_samples = {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.35.mlp.down_proj.weight",
        "model.norm.weight",
        "model.layers.20.self_attn.q_proj.lora_a",
        "model.layers.20.self_attn.q_proj.lora_b",
        "model.layers.35.mlp.down_proj.lora_a",
        "model.layers.35.mlp.down_proj.lora_b"
    };

    for (const auto& d : descriptors) {
        const MappedFileRecord* mapped = FindMappedFile(&mapped_files, d.source_path);
        if (mapped == nullptr) {
            ++missing_mapping_count;
            if (errors.size() < 16) errors.push_back("missing mapping: " + d.name);
            continue;
        }
        if (d.absolute_offset_end > mapped->file_bytes || d.absolute_offset_begin > d.absolute_offset_end) {
            ++out_of_range_count;
            if (errors.size() < 16) errors.push_back("view out of range: " + d.name);
            continue;
        }
        ++mapped_descriptor_count;
        resolved_payload_bytes += d.byte_size;

        const bool preferred = std::find(preferred_samples.begin(), preferred_samples.end(), d.name) != preferred_samples.end();
        if (preferred || sample_index < 3) {
            Napi::Object sample = Napi::Object::New(env);
            sample.Set("name", Napi::String::New(env, d.name));
            sample.Set("dtype", Napi::String::New(env, d.dtype));
            sample.Set("shape", U64VectorToNapiArray(env, d.shape));
            sample.Set("source_path", Napi::String::New(env, d.source_path));
            sample.Set("absolute_byte_offsets", U64VectorToNapiArray(env, {d.absolute_offset_begin, d.absolute_offset_end}));
            sample.Set("mapped_file_bytes", Napi::Number::New(env, static_cast<double>(mapped->file_bytes)));
            sample.Set("byte_size", Napi::Number::New(env, static_cast<double>(d.byte_size)));
            sample.Set("view_resolved", Napi::Boolean::New(env, true));
            sample.Set("payload_loaded", Napi::Boolean::New(env, false));
            sample.Set("payload_touched", Napi::Boolean::New(env, false));
            samples.Set(sample_index++, sample);
        }
    }

    Napi::Array errors_out = Napi::Array::New(env, errors.size());
    for (size_t i = 0; i < errors.size(); ++i) {
        errors_out.Set(static_cast<uint32_t>(i), Napi::String::New(env, errors[i]));
    }

    Napi::Object out = Napi::Object::New(env);
    out.Set("owner", Napi::String::New(env, owner));
    out.Set("descriptor_count", Napi::Number::New(env, static_cast<double>(descriptors.size())));
    out.Set("mapped_file_count", Napi::Number::New(env, static_cast<double>(mapped_files.size())));
    out.Set("mapped_descriptor_count", Napi::Number::New(env, static_cast<double>(mapped_descriptor_count)));
    out.Set("missing_mapping_count", Napi::Number::New(env, static_cast<double>(missing_mapping_count)));
    out.Set("out_of_range_count", Napi::Number::New(env, static_cast<double>(out_of_range_count)));
    out.Set("resolved_payload_bytes", Napi::Number::New(env, static_cast<double>(resolved_payload_bytes)));
    out.Set("all_views_resolved", Napi::Boolean::New(env, mapped_descriptor_count == descriptors.size() && missing_mapping_count == 0 && out_of_range_count == 0));
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("payload_touched", Napi::Boolean::New(env, false));
    out.Set("mlx_arrays_constructed", Napi::Boolean::New(env, false));
    out.Set("coffeescript_tensor_payload_allowed", Napi::Boolean::New(env, false));
    out.Set("sample_views", samples);
    out.Set("errors", errors_out);
    return out;
}

static Napi::Object BuildTypedTensorPlan(
    Napi::Env env,
    const std::string& owner,
    const std::vector<TensorDescriptor>& descriptors,
    const std::vector<MappedFileRecord>& mapped_files) {
    std::uint64_t planned_array_count = 0;
    std::uint64_t missing_mapping_count = 0;
    std::uint64_t out_of_range_count = 0;
    std::uint64_t planned_payload_bytes = 0;
    std::map<std::string, std::uint64_t> dtype_counts;
    std::map<std::string, std::uint64_t> role_counts;
    std::map<std::string, std::uint64_t> array_kind_counts;
    std::vector<std::string> errors;
    Napi::Array samples = Napi::Array::New(env);
    uint32_t sample_index = 0;

    const std::vector<std::string> preferred_samples = {
        "model.embed_tokens.weight",
        "model.embed_tokens.scales",
        "model.embed_tokens.biases",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.scales",
        "model.layers.0.self_attn.q_proj.biases",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.20.self_attn.q_proj.lora_a",
        "model.layers.20.self_attn.q_proj.lora_b"
    };

    for (const auto& d : descriptors) {
        const MappedFileRecord* mapped = FindMappedFile(&mapped_files, d.source_path);
        if (mapped == nullptr) {
            ++missing_mapping_count;
            if (errors.size() < 16) errors.push_back("missing mapping: " + d.name);
            continue;
        }
        if (d.absolute_offset_end > mapped->file_bytes || d.absolute_offset_begin > d.absolute_offset_end) {
            ++out_of_range_count;
            if (errors.size() < 16) errors.push_back("typed view out of range: " + d.name);
            continue;
        }

        const std::string role = TensorRoleForName(d.name);
        const std::string array_kind = TensorArrayKindForDescriptor(d);
        ++planned_array_count;
        planned_payload_bytes += d.byte_size;
        dtype_counts[d.dtype] += 1;
        role_counts[role] += 1;
        array_kind_counts[array_kind] += 1;

        const bool preferred = std::find(preferred_samples.begin(), preferred_samples.end(), d.name) != preferred_samples.end();
        if (preferred || sample_index < 4) {
            Napi::Object sample = Napi::Object::New(env);
            sample.Set("name", Napi::String::New(env, d.name));
            sample.Set("role", Napi::String::New(env, role));
            sample.Set("array_kind", Napi::String::New(env, array_kind));
            sample.Set("dtype", Napi::String::New(env, d.dtype));
            sample.Set("shape", U64VectorToNapiArray(env, d.shape));
            sample.Set("byte_size", Napi::Number::New(env, static_cast<double>(d.byte_size)));
            sample.Set("source_path", Napi::String::New(env, d.source_path));
            sample.Set("absolute_byte_offsets", U64VectorToNapiArray(env, {d.absolute_offset_begin, d.absolute_offset_end}));
            sample.Set("mapped_file_bytes", Napi::Number::New(env, static_cast<double>(mapped->file_bytes)));
            sample.Set("array_spec_resolved", Napi::Boolean::New(env, true));
            sample.Set("payload_loaded", Napi::Boolean::New(env, false));
            sample.Set("payload_touched", Napi::Boolean::New(env, false));
            sample.Set("mlx_array_constructed", Napi::Boolean::New(env, false));
            samples.Set(sample_index++, sample);
        }
    }

    Napi::Object dtype_counts_out = Napi::Object::New(env);
    for (const auto& kv : dtype_counts) {
        dtype_counts_out.Set(kv.first, Napi::Number::New(env, static_cast<double>(kv.second)));
    }

    Napi::Object role_counts_out = Napi::Object::New(env);
    for (const auto& kv : role_counts) {
        role_counts_out.Set(kv.first, Napi::Number::New(env, static_cast<double>(kv.second)));
    }

    Napi::Object array_kind_counts_out = Napi::Object::New(env);
    for (const auto& kv : array_kind_counts) {
        array_kind_counts_out.Set(kv.first, Napi::Number::New(env, static_cast<double>(kv.second)));
    }

    Napi::Array errors_out = Napi::Array::New(env, errors.size());
    for (size_t i = 0; i < errors.size(); ++i) {
        errors_out.Set(static_cast<uint32_t>(i), Napi::String::New(env, errors[i]));
    }

    Napi::Object out = Napi::Object::New(env);
    out.Set("owner", Napi::String::New(env, owner));
    out.Set("descriptor_count", Napi::Number::New(env, static_cast<double>(descriptors.size())));
    out.Set("mapped_file_count", Napi::Number::New(env, static_cast<double>(mapped_files.size())));
    out.Set("planned_array_count", Napi::Number::New(env, static_cast<double>(planned_array_count)));
    out.Set("missing_mapping_count", Napi::Number::New(env, static_cast<double>(missing_mapping_count)));
    out.Set("out_of_range_count", Napi::Number::New(env, static_cast<double>(out_of_range_count)));
    out.Set("planned_payload_bytes", Napi::Number::New(env, static_cast<double>(planned_payload_bytes)));
    out.Set("all_array_specs_resolved", Napi::Boolean::New(env, planned_array_count == descriptors.size() && missing_mapping_count == 0 && out_of_range_count == 0));
    out.Set("dtype_counts", dtype_counts_out);
    out.Set("role_counts", role_counts_out);
    out.Set("array_kind_counts", array_kind_counts_out);
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("payload_touched", Napi::Boolean::New(env, false));
    out.Set("mlx_arrays_constructed", Napi::Boolean::New(env, false));
    out.Set("coffeescript_tensor_payload_allowed", Napi::Boolean::New(env, false));
    out.Set("sample_array_specs", samples);
    out.Set("errors", errors_out);
    return out;
}

// ---------------------------------------------------------
// Small helper: unwrap external pointer
// ---------------------------------------------------------
static metal_llm_model* getModel(const Napi::CallbackInfo& info) {
    if (info.Length() < 1 || !info[0].IsExternal()) {
        Napi::TypeError::New(info.Env(), "Expected External<metal_llm_model>")
            .ThrowAsJavaScriptException();
        return nullptr;
    }
    return info[0].As<Napi::External<metal_llm_model>>().Data();
}

// ---------------------------------------------------------
// LoadModel(modelDirectory : string)
// Returns External<metal_llm_model>
// ---------------------------------------------------------
Napi::Value LoadModel(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!EnvFlagEnabled("GYPSY_ALLOW_LEGACY_LOAD_MODEL")) {
        Napi::Error::New(
            env,
            "legacy loadModel is disabled by default because it eagerly loads/evals tensors and may upcast weights; use loadModelResident or set GYPSY_ALLOW_LEGACY_LOAD_MODEL=1 for explicit diagnostics"
        ).ThrowAsJavaScriptException();
        return env.Null();
    }

    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected model directory path").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string dir = info[0].As<Napi::String>();

    // Allocate struct
    metal_llm_model* m = new metal_llm_model();

    // Load weights
    llm_status st = llm_load_model(m, dir.c_str());
    if (st != LLM_OK) {
        delete m;
        Napi::Error::New(env, "Failed to load model").ThrowAsJavaScriptException();
        return env.Null();
    }

    // Wrap into External
    return Napi::External<metal_llm_model>::New(env, m);
}

// ---------------------------------------------------------
// inspectModel(modelDirectory : string)
// Metadata-only inspection. Does not load/eval tensor payloads.
// ---------------------------------------------------------
Napi::Value InspectModel(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected model directory path").ThrowAsJavaScriptException();
        return env.Null();
    }

    const std::string dir = info[0].As<Napi::String>();
    const std::string config_path = (fs::path(dir) / "config.json").string();
    const std::string config_json = ReadTextFile(config_path);
    if (config_json.empty()) {
        Napi::Error::New(env, "config.json not found or empty").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::vector<fs::path> safetensors = FindSafetensorFiles(dir);
    std::uint64_t total_file_bytes = 0;
    for (const auto& p : safetensors) {
        total_file_bytes += static_cast<std::uint64_t>(fs::file_size(p));
    }

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("inspection_type", Napi::String::New(env, "metadata_only"));
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("model_dir", Napi::String::New(env, dir));
    out.Set("config_path", Napi::String::New(env, config_path));

    Napi::Object config = Napi::Object::New(env);
    const std::vector<std::string> config_keys = {
        "model_type",
        "architectures",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rope_theta",
        "max_position_embeddings",
        "rms_norm_eps",
        "torch_dtype"
    };
    for (const auto& key : config_keys) {
        config.Set(key, JsonScalarToNapi(env, config_json, key));
    }

    const double hidden_size = JsonNumberValue(config_json, "hidden_size", -1);
    const double num_heads = JsonNumberValue(config_json, "num_attention_heads", -1);
    const double explicit_head_dim = JsonNumberValue(config_json, "head_dim", -1);
    if (explicit_head_dim > 0) {
        config.Set("derived_head_dim", Napi::Number::New(env, explicit_head_dim));
    } else if (hidden_size > 0 && num_heads > 0) {
        config.Set("derived_head_dim", Napi::Number::New(env, hidden_size / num_heads));
    } else {
        config.Set("derived_head_dim", env.Null());
    }
    out.Set("config", config);

    Napi::Array files = Napi::Array::New(env, safetensors.size());
    std::uint64_t total_header_bytes = 0;
    std::uint64_t total_tensor_entries = 0;
    std::map<std::string, std::uint64_t> dtype_counts;
    std::vector<std::string> tensor_names;
    std::vector<TensorDescriptor> tensor_descriptors;

    for (size_t i = 0; i < safetensors.size(); ++i) {
        const fs::path& p = safetensors[i];
        std::ifstream f(p, std::ios::binary);
        const std::uint64_t header_len = f.good() ? ReadSafetensorsHeaderLength(f) : 0;
        std::string header;
        if (header_len > 0 && header_len < (1ULL << 30)) {
            header.resize(static_cast<size_t>(header_len));
            f.read(header.data(), static_cast<std::streamsize>(header.size()));
        }
        const std::uint64_t tensor_entries = static_cast<std::uint64_t>(CountOccurrences(header, "\"dtype\""));
        total_header_bytes += header_len;
        total_tensor_entries += tensor_entries;
        const std::uint64_t payload_data_offset = header_len > 0 ? 8 + header_len : 0;
        std::vector<TensorDescriptor> file_descriptors = ExtractSafetensorDescriptors(header, p.filename().string(), p.string(), payload_data_offset);
        for (const auto& d : file_descriptors) tensor_names.push_back(d.name);
        tensor_descriptors.insert(
            tensor_descriptors.end(),
            std::make_move_iterator(file_descriptors.begin()),
            std::make_move_iterator(file_descriptors.end()));
        const std::vector<std::string> dtypes = {"F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U64", "U32", "U16", "U8", "BOOL"};
        for (const auto& dtype : dtypes) {
            const size_t n = CountOccurrences(header, "\"dtype\":\"" + dtype + "\"");
            if (n > 0) dtype_counts[dtype] += static_cast<std::uint64_t>(n);
        }

        Napi::Object file = Napi::Object::New(env);
        file.Set("path", Napi::String::New(env, p.string()));
        file.Set("name", Napi::String::New(env, p.filename().string()));
        file.Set("file_bytes", Napi::Number::New(env, static_cast<double>(fs::file_size(p))));
        file.Set("header_bytes", Napi::Number::New(env, static_cast<double>(header_len)));
        file.Set("payload_data_offset", Napi::Number::New(env, static_cast<double>(payload_data_offset)));
        file.Set("tensor_entries", Napi::Number::New(env, static_cast<double>(tensor_entries)));
        files.Set(static_cast<uint32_t>(i), file);
    }

    Napi::Object dtype_counts_out = Napi::Object::New(env);
    for (const auto& kv : dtype_counts) {
        dtype_counts_out.Set(kv.first, Napi::Number::New(env, static_cast<double>(kv.second)));
    }

    out.Set("safetensor_file_count", Napi::Number::New(env, static_cast<double>(safetensors.size())));
    out.Set("safetensor_total_file_bytes", Napi::Number::New(env, static_cast<double>(total_file_bytes)));
    out.Set("safetensor_total_header_bytes", Napi::Number::New(env, static_cast<double>(total_header_bytes)));
    out.Set("safetensor_tensor_entries", Napi::Number::New(env, static_cast<double>(total_tensor_entries)));
    out.Set("safetensor_dtype_counts", dtype_counts_out);
    out.Set("safetensor_files", files);

    std::sort(tensor_names.begin(), tensor_names.end());
    tensor_names.erase(std::unique(tensor_names.begin(), tensor_names.end()), tensor_names.end());
    std::sort(tensor_descriptors.begin(), tensor_descriptors.end(), [](const auto& a, const auto& b) {
        return a.name < b.name;
    });

    std::uint64_t descriptor_total_payload_bytes = 0;
    std::map<std::string, std::uint64_t> role_counts;
    for (const auto& d : tensor_descriptors) {
        descriptor_total_payload_bytes += d.byte_size;
        role_counts[TensorRoleForName(d.name)] += 1;
    }

    Napi::Array tensor_names_out = Napi::Array::New(env, tensor_names.size());
    for (size_t i = 0; i < tensor_names.size(); ++i) {
        tensor_names_out.Set(static_cast<uint32_t>(i), Napi::String::New(env, tensor_names[i]));
    }
    out.Set("tensor_name_count", Napi::Number::New(env, static_cast<double>(tensor_names.size())));
    out.Set("tensor_names", tensor_names_out);
    out.Set("tensor_descriptor_count", Napi::Number::New(env, static_cast<double>(tensor_descriptors.size())));
    out.Set("tensor_descriptor_total_payload_bytes", Napi::Number::New(env, static_cast<double>(descriptor_total_payload_bytes)));

    Napi::Object role_counts_out = Napi::Object::New(env);
    for (const auto& kv : role_counts) {
        role_counts_out.Set(kv.first, Napi::Number::New(env, static_cast<double>(kv.second)));
    }
    out.Set("tensor_role_counts", role_counts_out);

    const size_t descriptor_sample_count = std::min<size_t>(tensor_descriptors.size(), 12);
    Napi::Array descriptor_sample = Napi::Array::New(env, descriptor_sample_count);
    for (size_t i = 0; i < descriptor_sample_count; ++i) {
        const auto& d = tensor_descriptors[i];
        descriptor_sample.Set(static_cast<uint32_t>(i), TensorDescriptorToNapi(env, d));
    }
    out.Set("tensor_descriptor_sample", descriptor_sample);

    std::unordered_set<std::string> tensor_name_set(tensor_names.begin(), tensor_names.end());
    const int layer_count = static_cast<int>(JsonNumberValue(config_json, "num_hidden_layers", 0));
    const std::vector<std::string> quantized_projection_suffixes = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
    };
    const std::vector<std::string> norm_suffixes = {
        "input_layernorm",
        "post_attention_layernorm",
        "self_attn.q_norm",
        "self_attn.k_norm"
    };
    std::vector<std::string> missing_expected;
    std::uint64_t expected_layer_tensor_count = 0;
    std::uint64_t present_layer_tensor_count = 0;

    for (int layer = 0; layer < layer_count; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        for (const auto& suffix : quantized_projection_suffixes) {
            for (const auto& part : {"weight", "scales", "biases"}) {
                ++expected_layer_tensor_count;
                const std::string name = prefix + suffix + "." + part;
                if (ContainsName(tensor_name_set, name)) {
                    ++present_layer_tensor_count;
                } else {
                    missing_expected.push_back(name);
                }
            }
        }
        for (const auto& suffix : norm_suffixes) {
            ++expected_layer_tensor_count;
            const std::string name = prefix + suffix + ".weight";
            if (ContainsName(tensor_name_set, name)) {
                ++present_layer_tensor_count;
            } else {
                missing_expected.push_back(name);
            }
        }
    }

    Napi::Array missing_expected_out = Napi::Array::New(env, missing_expected.size());
    for (size_t i = 0; i < missing_expected.size(); ++i) {
        missing_expected_out.Set(static_cast<uint32_t>(i), Napi::String::New(env, missing_expected[i]));
    }

    Napi::Object qwen_layout = Napi::Object::New(env);
    qwen_layout.Set("checked", Napi::Boolean::New(env, layer_count > 0));
    qwen_layout.Set("layer_count", Napi::Number::New(env, layer_count));
    qwen_layout.Set("quantized_projection_groups_per_layer", Napi::Number::New(env, static_cast<double>(quantized_projection_suffixes.size())));
    qwen_layout.Set("norm_groups_per_layer", Napi::Number::New(env, static_cast<double>(norm_suffixes.size())));
    qwen_layout.Set("expected_layer_tensor_count", Napi::Number::New(env, static_cast<double>(expected_layer_tensor_count)));
    qwen_layout.Set("present_layer_tensor_count", Napi::Number::New(env, static_cast<double>(present_layer_tensor_count)));
    qwen_layout.Set("missing_expected_tensor_count", Napi::Number::New(env, static_cast<double>(missing_expected.size())));
    qwen_layout.Set("missing_expected_tensors", missing_expected_out);
    qwen_layout.Set("embed_tokens_weight", Napi::Boolean::New(env, ContainsName(tensor_name_set, "model.embed_tokens.weight")));
    qwen_layout.Set("embed_tokens_scales", Napi::Boolean::New(env, ContainsName(tensor_name_set, "model.embed_tokens.scales")));
    qwen_layout.Set("embed_tokens_biases", Napi::Boolean::New(env, ContainsName(tensor_name_set, "model.embed_tokens.biases")));
    qwen_layout.Set("final_norm_weight", Napi::Boolean::New(env, ContainsName(tensor_name_set, "model.norm.weight")));
    qwen_layout.Set("lm_head_weight", Napi::Boolean::New(env, ContainsName(tensor_name_set, "lm_head.weight")));
    qwen_layout.Set("tied_embeddings_likely", Napi::Boolean::New(env,
        ContainsName(tensor_name_set, "model.embed_tokens.weight") &&
        !ContainsName(tensor_name_set, "lm_head.weight")));
    out.Set("qwen_layout", qwen_layout);

    return out;
}

// ---------------------------------------------------------
// inspectAdapter(adapterDirectory : string)
// Metadata-only LoRA adapter inspection.
// ---------------------------------------------------------
Napi::Value InspectAdapter(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected adapter directory path").ThrowAsJavaScriptException();
        return env.Null();
    }

    const std::string dir = info[0].As<Napi::String>();
    const std::string config_path = (fs::path(dir) / "adapter_config.json").string();
    const std::string config_json = ReadTextFile(config_path);
    if (config_json.empty()) {
        Napi::Error::New(env, "adapter_config.json not found or empty").ThrowAsJavaScriptException();
        return env.Null();
    }

    const std::vector<fs::path> safetensors = FindSafetensorFiles(dir);
    std::uint64_t total_file_bytes = 0;
    std::uint64_t total_header_bytes = 0;
    std::vector<TensorDescriptor> descriptors;
    std::map<std::string, std::uint64_t> dtype_counts;
    for (const auto& p : safetensors) {
        total_file_bytes += static_cast<std::uint64_t>(fs::file_size(p));
        std::ifstream f(p, std::ios::binary);
        const std::uint64_t header_len = f.good() ? ReadSafetensorsHeaderLength(f) : 0;
        total_header_bytes += header_len;
        std::string header;
        if (header_len > 0 && header_len < (1ULL << 30)) {
            header.resize(static_cast<size_t>(header_len));
            f.read(header.data(), static_cast<std::streamsize>(header.size()));
        }
        const std::uint64_t payload_data_offset = header_len > 0 ? 8 + header_len : 0;
        auto file_descriptors = ExtractSafetensorDescriptors(header, p.filename().string(), p.string(), payload_data_offset);
        descriptors.insert(
            descriptors.end(),
            std::make_move_iterator(file_descriptors.begin()),
            std::make_move_iterator(file_descriptors.end()));

        const std::vector<std::string> dtypes = {"F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U64", "U32", "U16", "U8", "BOOL"};
        for (const auto& dtype : dtypes) {
            const size_t n = CountOccurrences(header, "\"dtype\":\"" + dtype + "\"");
            if (n > 0) dtype_counts[dtype] += static_cast<std::uint64_t>(n);
        }
    }
    std::sort(descriptors.begin(), descriptors.end(), [](const auto& a, const auto& b) {
        return a.name < b.name;
    });

    std::unordered_set<std::string> names;
    std::uint64_t total_payload_bytes = 0;
    std::map<std::string, std::uint64_t> target_counts;
    std::map<std::string, std::uint64_t> layer_counts;
    std::vector<int> layers;
    for (const auto& d : descriptors) {
        names.insert(d.name);
        total_payload_bytes += d.byte_size;

        const std::string prefix = "model.layers.";
        if (d.name.rfind(prefix, 0) == 0) {
            size_t layer_start = prefix.size();
            size_t layer_end = d.name.find('.', layer_start);
            if (layer_end != std::string::npos) {
                const std::string layer_text = d.name.substr(layer_start, layer_end - layer_start);
                layer_counts[layer_text] += 1;
                try {
                    int layer = std::stoi(layer_text);
                    if (std::find(layers.begin(), layers.end(), layer) == layers.end()) {
                        layers.push_back(layer);
                    }
                } catch (...) {}
            }
        }

        const std::vector<std::string> targets = {
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj"
        };
        for (const auto& target : targets) {
            if (d.name.find("." + target + ".") != std::string::npos) {
                target_counts[target] += 1;
            }
        }
    }
    std::sort(layers.begin(), layers.end());

    const std::vector<std::string> expected_targets = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
    };
    std::vector<std::string> missing_expected;
    for (int layer : layers) {
        const std::string base = "model.layers." + std::to_string(layer) + ".";
        for (const auto& target : expected_targets) {
            for (const auto& part : {"lora_a", "lora_b"}) {
                const std::string name = base + target + "." + part;
                if (!ContainsName(names, name)) missing_expected.push_back(name);
            }
        }
    }

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("inspection_type", Napi::String::New(env, "adapter_metadata_only"));
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("adapter_dir", Napi::String::New(env, dir));
    out.Set("config_path", Napi::String::New(env, config_path));
    out.Set("rank", Napi::Number::New(env, JsonNumberValue(config_json, "rank", 0)));
    out.Set("scale", Napi::Number::New(env, JsonNumberValue(config_json, "scale", 0)));
    out.Set("safetensor_file_count", Napi::Number::New(env, static_cast<double>(safetensors.size())));
    out.Set("safetensor_total_file_bytes", Napi::Number::New(env, static_cast<double>(total_file_bytes)));
    out.Set("safetensor_total_header_bytes", Napi::Number::New(env, static_cast<double>(total_header_bytes)));
    out.Set("tensor_descriptor_count", Napi::Number::New(env, static_cast<double>(descriptors.size())));
    out.Set("tensor_descriptor_total_payload_bytes", Napi::Number::New(env, static_cast<double>(total_payload_bytes)));

    Napi::Object dtype_counts_out = Napi::Object::New(env);
    for (const auto& kv : dtype_counts) {
        dtype_counts_out.Set(kv.first, Napi::Number::New(env, static_cast<double>(kv.second)));
    }
    out.Set("dtype_counts", dtype_counts_out);

    Napi::Array layers_out = Napi::Array::New(env, layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        layers_out.Set(static_cast<uint32_t>(i), Napi::Number::New(env, layers[i]));
    }
    out.Set("layers", layers_out);
    out.Set("layer_count", Napi::Number::New(env, static_cast<double>(layers.size())));

    Napi::Object target_counts_out = Napi::Object::New(env);
    for (const auto& target : expected_targets) {
        target_counts_out.Set(target, Napi::Number::New(env, static_cast<double>(target_counts[target])));
    }
    out.Set("target_tensor_counts", target_counts_out);

    Napi::Array missing_out = Napi::Array::New(env, missing_expected.size());
    for (size_t i = 0; i < missing_expected.size(); ++i) {
        missing_out.Set(static_cast<uint32_t>(i), Napi::String::New(env, missing_expected[i]));
    }
    out.Set("missing_expected_tensors", missing_out);
    out.Set("missing_expected_tensor_count", Napi::Number::New(env, static_cast<double>(missing_expected.size())));

    const size_t sample_count = std::min<size_t>(descriptors.size(), 12);
    Napi::Array sample = Napi::Array::New(env, sample_count);
    for (size_t i = 0; i < sample_count; ++i) {
        const auto& d = descriptors[i];
        sample.Set(static_cast<uint32_t>(i), TensorDescriptorToNapi(env, d));
    }
    out.Set("tensor_descriptor_sample", sample);

    return out;
}

// ---------------------------------------------------------
// inspectTokenizer(tokenizerDirectory : string)
// Metadata-only tokenizer inspection.
// ---------------------------------------------------------
Napi::Value InspectTokenizer(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected tokenizer directory path").ThrowAsJavaScriptException();
        return env.Null();
    }

    const std::string dir = info[0].As<Napi::String>();
    const fs::path tokenizer_json_path = fs::path(dir) / "tokenizer.json";
    const fs::path tokenizer_config_path = fs::path(dir) / "tokenizer_config.json";
    const fs::path chat_template_path = fs::path(dir) / "chat_template.jinja";

    const bool tokenizer_json_present = fs::exists(tokenizer_json_path);
    const bool tokenizer_config_present = fs::exists(tokenizer_config_path);
    const bool chat_template_present = fs::exists(chat_template_path);

    std::string tokenizer_json = tokenizer_json_present
        ? ReadTextFile(tokenizer_json_path.string())
        : "";
    std::string tokenizer_config = tokenizer_config_present
        ? ReadTextFile(tokenizer_config_path.string())
        : "";
    std::string chat_template = chat_template_present
        ? ReadTextFile(chat_template_path.string())
        : "";

    std::string tokenizer_model_type;
    const size_t model_pos = tokenizer_json.find("\"model\"");
    if (model_pos != std::string::npos) {
        tokenizer_model_type = JsonStringValue(tokenizer_json.substr(model_pos), "type");
    }

    const double im_start_id = TokenIdForAddedTokenContent(tokenizer_json, "<|im_start|>");
    const double im_end_id = TokenIdForAddedTokenContent(tokenizer_json, "<|im_end|>");
    const double pad_id = TokenIdForAddedTokenContent(tokenizer_json, "<|endoftext|>");

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, tokenizer_json_present && tokenizer_config_present));
    out.Set("inspection_type", Napi::String::New(env, "tokenizer_metadata_only"));
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("tokenizer_dir", Napi::String::New(env, dir));
    out.Set("tokenizer_json_present", Napi::Boolean::New(env, tokenizer_json_present));
    out.Set("tokenizer_config_present", Napi::Boolean::New(env, tokenizer_config_present));
    out.Set("chat_template_present", Napi::Boolean::New(env, chat_template_present));
    out.Set("tokenizer_json_bytes", Napi::Number::New(env, tokenizer_json_present ? static_cast<double>(fs::file_size(tokenizer_json_path)) : 0.0));
    out.Set("tokenizer_config_bytes", Napi::Number::New(env, tokenizer_config_present ? static_cast<double>(fs::file_size(tokenizer_config_path)) : 0.0));
    out.Set("chat_template_bytes", Napi::Number::New(env, chat_template_present ? static_cast<double>(fs::file_size(chat_template_path)) : 0.0));
    out.Set("tokenizer_class", JsonScalarToNapi(env, tokenizer_config, "tokenizer_class"));
    out.Set("model_max_length", JsonScalarToNapi(env, tokenizer_config, "model_max_length"));
    out.Set("eos_token", JsonScalarToNapi(env, tokenizer_config, "eos_token"));
    out.Set("pad_token", JsonScalarToNapi(env, tokenizer_config, "pad_token"));
    out.Set("split_special_tokens", JsonScalarToNapi(env, tokenizer_config, "split_special_tokens"));
    out.Set("tokenizer_model_type", tokenizer_model_type.empty() ? env.Null() : Napi::String::New(env, tokenizer_model_type).As<Napi::Value>());
    out.Set("added_tokens_count", Napi::Number::New(env, static_cast<double>(CountOccurrences(tokenizer_json, "\"content\""))));
    out.Set("im_start_token_id", std::isnan(im_start_id) ? env.Null() : Napi::Number::New(env, im_start_id).As<Napi::Value>());
    out.Set("im_end_token_id", std::isnan(im_end_id) ? env.Null() : Napi::Number::New(env, im_end_id).As<Napi::Value>());
    out.Set("pad_token_id", std::isnan(pad_id) ? env.Null() : Napi::Number::New(env, pad_id).As<Napi::Value>());
    out.Set("chat_template_has_generation_prompt", Napi::Boolean::New(env, chat_template.find("add_generation_prompt") != std::string::npos));
    out.Set("chat_template_has_im_start", Napi::Boolean::New(env, chat_template.find("<|im_start|>") != std::string::npos));
    out.Set("chat_template_has_im_end", Napi::Boolean::New(env, chat_template.find("<|im_end|>") != std::string::npos));

    return out;
}

// ---------------------------------------------------------
// loadModelResident(modelDirectory : string)
// New protocol skeleton. Metadata-only, no tensor payload load.
// ---------------------------------------------------------
Napi::Value LoadModelResident(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::EscapableHandleScope scope(env);

    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected model directory path").ThrowAsJavaScriptException();
        return env.Null();
    }

    const std::string dir = info[0].As<Napi::String>();
    const std::string handle = MakeHandle("gmodel", gypsy_next_model_id++);

    // Build metadata by reusing the same safe inspection logic inline through
    // the public helper. This deliberately reads only config/header metadata.
    Napi::Value metadata_value = InspectModel(info);
    if (env.IsExceptionPending()) return env.Null();
    Napi::Object metadata = metadata_value.As<Napi::Object>();

    GypsyModelRecord record;
    record.handle = handle;
    record.model_dir = dir;
    record.metadata = Napi::Persistent(metadata);
    record.tensor_descriptors = LoadTensorDescriptorsFromSafetensors(dir);
    gypsy_models.emplace(handle, std::move(record));

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("handle", Napi::String::New(env, handle));
    out.Set("model_dir", Napi::String::New(env, dir));
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("resident_loaded", Napi::Boolean::New(env, false));
    out.Set("descriptor_index_cached", Napi::Boolean::New(env, true));
    out.Set("cached_descriptor_count", Napi::Number::New(env, static_cast<double>(gypsy_models[handle].tensor_descriptors.size())));
    out.Set("metadata", metadata);
    out.Set("handle_counts", GypsyHandleCounts(env));
    return scope.Escape(out);
}

Napi::Value DescribeModelGroups(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::EscapableHandleScope scope(env);
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected model handle").ThrowAsJavaScriptException();
        return env.Null();
    }

    const std::string handle = info[0].As<Napi::String>();
    auto it = gypsy_models.find(handle);
    if (it == gypsy_models.end()) {
        Napi::Error::New(env, "Unknown model handle").ThrowAsJavaScriptException();
        return env.Null();
    }

    const std::string& model_dir = it->second.model_dir;
    const std::string config_json = ReadTextFile((fs::path(model_dir) / "config.json").string());
    const int layer_count = static_cast<int>(JsonNumberValue(config_json, "num_hidden_layers", 0));
    const auto& descriptors = it->second.tensor_descriptors;

    std::unordered_map<std::string, TensorDescriptor> by_name;
    std::uint64_t total_payload_bytes = 0;
    for (const auto& d : descriptors) {
        by_name[d.name] = d;
        total_payload_bytes += d.byte_size;
    }

    const std::vector<std::string> projection_suffixes = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
    };
    const std::vector<std::string> norm_suffixes = {
        "input_layernorm",
        "post_attention_layernorm",
        "self_attn.q_norm",
        "self_attn.k_norm"
    };

    Napi::Array layers = Napi::Array::New(env, static_cast<size_t>(std::max(layer_count, 0)));
    std::vector<std::string> missing_groups;
    std::uint64_t total_groups = 0;
    std::uint64_t complete_groups = 0;
    std::uint64_t resident_plan_bytes = 0;

    for (int layer = 0; layer < layer_count; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        Napi::Object layer_out = Napi::Object::New(env);
        layer_out.Set("layer", Napi::Number::New(env, layer));
        Napi::Object projections = Napi::Object::New(env);
        Napi::Object norms = Napi::Object::New(env);

        std::uint64_t layer_group_count = 0;
        std::uint64_t layer_complete_group_count = 0;
        std::uint64_t layer_bytes = 0;

        for (const auto& suffix : projection_suffixes) {
            ++total_groups;
            ++layer_group_count;
            const std::string group_name = prefix + suffix;
            Napi::Object group = Napi::Object::New(env);
            group.Set("kind", Napi::String::New(env, "quantized_projection"));
            group.Set("group", Napi::String::New(env, group_name));
            bool complete = true;
            std::uint64_t group_bytes = 0;
            for (const auto& part : {"weight", "scales", "biases"}) {
                const std::string tensor_name = group_name + "." + part;
                auto found = by_name.find(tensor_name);
                if (found == by_name.end()) {
                    complete = false;
                    missing_groups.push_back(tensor_name);
                    group.Set(part, env.Null());
                } else {
                    group_bytes += found->second.byte_size;
                    group.Set(part, TensorDescriptorToNapi(env, found->second));
                }
            }
            group.Set("complete", Napi::Boolean::New(env, complete));
            group.Set("byte_size", Napi::Number::New(env, static_cast<double>(group_bytes)));
            layer_bytes += group_bytes;
            if (complete) {
                ++complete_groups;
                ++layer_complete_group_count;
            }
            projections.Set(suffix, group);
        }

        for (const auto& suffix : norm_suffixes) {
            ++total_groups;
            ++layer_group_count;
            const std::string group_name = prefix + suffix;
            const std::string tensor_name = group_name + ".weight";
            Napi::Object group = Napi::Object::New(env);
            group.Set("kind", Napi::String::New(env, "norm"));
            group.Set("group", Napi::String::New(env, group_name));
            auto found = by_name.find(tensor_name);
            if (found == by_name.end()) {
                missing_groups.push_back(tensor_name);
                group.Set("complete", Napi::Boolean::New(env, false));
                group.Set("weight", env.Null());
                group.Set("byte_size", Napi::Number::New(env, 0));
            } else {
                group.Set("complete", Napi::Boolean::New(env, true));
                group.Set("weight", TensorDescriptorToNapi(env, found->second));
                group.Set("byte_size", Napi::Number::New(env, static_cast<double>(found->second.byte_size)));
                layer_bytes += found->second.byte_size;
                ++complete_groups;
                ++layer_complete_group_count;
            }
            norms.Set(suffix, group);
        }

        resident_plan_bytes += layer_bytes;
        layer_out.Set("projection_groups", projections);
        layer_out.Set("norm_groups", norms);
        layer_out.Set("group_count", Napi::Number::New(env, static_cast<double>(layer_group_count)));
        layer_out.Set("complete_group_count", Napi::Number::New(env, static_cast<double>(layer_complete_group_count)));
        layer_out.Set("byte_size", Napi::Number::New(env, static_cast<double>(layer_bytes)));
        layers.Set(static_cast<uint32_t>(layer), layer_out);
    }

    Napi::Array missing = Napi::Array::New(env, missing_groups.size());
    for (size_t i = 0; i < missing_groups.size(); ++i) {
        missing.Set(static_cast<uint32_t>(i), Napi::String::New(env, missing_groups[i]));
    }

    Napi::Object globals = Napi::Object::New(env);
    for (const auto& tensor_name : {
             "model.embed_tokens.weight",
             "model.embed_tokens.scales",
             "model.embed_tokens.biases",
             "model.norm.weight",
             "lm_head.weight"}) {
        auto found = by_name.find(tensor_name);
        if (found == by_name.end()) {
            globals.Set(tensor_name, env.Null());
        } else {
            globals.Set(tensor_name, TensorDescriptorToNapi(env, found->second));
        }
    }

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("handle", Napi::String::New(env, handle));
    out.Set("model_dir", Napi::String::New(env, model_dir));
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("descriptor_source", Napi::String::New(env, "safetensors_headers_only"));
    out.Set("descriptor_index_cached", Napi::Boolean::New(env, true));
    out.Set("descriptor_index_owner", Napi::String::New(env, "model_handle"));
    out.Set("layer_count", Napi::Number::New(env, layer_count));
    out.Set("descriptor_count", Napi::Number::New(env, static_cast<double>(descriptors.size())));
    out.Set("descriptor_total_payload_bytes", Napi::Number::New(env, static_cast<double>(total_payload_bytes)));
    out.Set("expected_groups", Napi::Number::New(env, static_cast<double>(total_groups)));
    out.Set("complete_groups", Napi::Number::New(env, static_cast<double>(complete_groups)));
    out.Set("missing_group_tensor_count", Napi::Number::New(env, static_cast<double>(missing_groups.size())));
    out.Set("missing_group_tensors", missing);
    out.Set("resident_layer_plan_bytes", Napi::Number::New(env, static_cast<double>(resident_plan_bytes)));
    out.Set("global_tensors", globals);
    out.Set("layers", layers);
    out.Set("handle_counts", GypsyHandleCounts(env));
    return scope.Escape(out);
}

Napi::Value LoadTokenizer(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::EscapableHandleScope scope(env);
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected tokenizer directory path").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string dir = info[0].As<Napi::String>();
    const std::string handle = MakeHandle("gtok", gypsy_next_tokenizer_id++);

    Napi::Value metadata_value = InspectTokenizer(info);
    if (env.IsExceptionPending()) return env.Null();
    Napi::Object metadata = metadata_value.As<Napi::Object>();
    if (!metadata.Get("ok").As<Napi::Boolean>().Value()) {
        Napi::Error::New(env, "Tokenizer metadata inspection failed").ThrowAsJavaScriptException();
        return env.Null();
    }

    GypsyTokenizerRecord record;
    record.handle = handle;
    record.tokenizer_dir = dir;
    record.metadata = Napi::Persistent(metadata);
    record.token_to_id = ParseTokenizerVocab(ReadTextFile((fs::path(dir) / "tokenizer.json").string()));
    for (const auto& kv : record.token_to_id) {
        record.id_to_token[kv.second] = kv.first;
    }
    gypsy_tokenizers.emplace(handle, std::move(record));

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("handle", Napi::String::New(env, handle));
    out.Set("tokenizer_dir", Napi::String::New(env, dir));
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("vocab_index_cached", Napi::Boolean::New(env, !gypsy_tokenizers[handle].token_to_id.empty()));
    out.Set("cached_vocab_count", Napi::Number::New(env, static_cast<double>(gypsy_tokenizers[handle].token_to_id.size())));
    out.Set("metadata", metadata);
    out.Set("handle_counts", GypsyHandleCounts(env));
    return scope.Escape(out);
}

Napi::Value LoadAdapter(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::EscapableHandleScope scope(env);
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected adapter directory path").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string dir = info[0].As<Napi::String>();
    const std::string handle = MakeHandle("gadapter", gypsy_next_adapter_id++);

    Napi::Value metadata_value = InspectAdapter(info);
    if (env.IsExceptionPending()) return env.Null();
    Napi::Object metadata = metadata_value.As<Napi::Object>();
    if (!metadata.Get("ok").As<Napi::Boolean>().Value()) {
        Napi::Error::New(env, "Adapter metadata inspection failed").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (metadata.Get("missing_expected_tensor_count").As<Napi::Number>().Int64Value() != 0) {
        Napi::Error::New(env, "Adapter metadata is missing expected tensors").ThrowAsJavaScriptException();
        return env.Null();
    }

    GypsyAdapterRecord record;
    record.handle = handle;
    record.adapter_dir = dir;
    record.metadata = Napi::Persistent(metadata);
    record.tensor_descriptors = LoadTensorDescriptorsFromSafetensors(dir);
    gypsy_adapters.emplace(handle, std::move(record));

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("handle", Napi::String::New(env, handle));
    out.Set("adapter_dir", Napi::String::New(env, dir));
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("descriptor_index_cached", Napi::Boolean::New(env, true));
    out.Set("cached_descriptor_count", Napi::Number::New(env, static_cast<double>(gypsy_adapters[handle].tensor_descriptors.size())));
    out.Set("metadata", metadata);
    out.Set("handle_counts", GypsyHandleCounts(env));
    return scope.Escape(out);
}

Napi::Value DescribeAdapterGroups(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::EscapableHandleScope scope(env);
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected adapter handle").ThrowAsJavaScriptException();
        return env.Null();
    }

    const std::string handle = info[0].As<Napi::String>();
    auto it = gypsy_adapters.find(handle);
    if (it == gypsy_adapters.end()) {
        Napi::Error::New(env, "Unknown adapter handle").ThrowAsJavaScriptException();
        return env.Null();
    }

    const std::string& adapter_dir = it->second.adapter_dir;
    const std::string config_json = ReadTextFile((fs::path(adapter_dir) / "adapter_config.json").string());
    const double rank = JsonNumberValue(config_json, "rank", 0);
    const double scale = JsonNumberValue(config_json, "scale", 0);
    const auto& descriptors = it->second.tensor_descriptors;

    std::unordered_map<std::string, TensorDescriptor> by_name;
    std::unordered_set<std::string> names;
    std::vector<int> layers;
    std::uint64_t total_payload_bytes = 0;
    for (const auto& d : descriptors) {
        by_name[d.name] = d;
        names.insert(d.name);
        total_payload_bytes += d.byte_size;
        const std::string prefix = "model.layers.";
        if (d.name.rfind(prefix, 0) == 0) {
            const size_t layer_start = prefix.size();
            const size_t layer_end = d.name.find('.', layer_start);
            if (layer_end != std::string::npos) {
                try {
                    const int layer = std::stoi(d.name.substr(layer_start, layer_end - layer_start));
                    if (std::find(layers.begin(), layers.end(), layer) == layers.end()) {
                        layers.push_back(layer);
                    }
                } catch (...) {}
            }
        }
    }
    std::sort(layers.begin(), layers.end());

    const std::vector<std::string> targets = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
    };

    Napi::Array layers_out = Napi::Array::New(env, layers.size());
    std::vector<std::string> missing;
    std::uint64_t expected_groups = 0;
    std::uint64_t complete_groups = 0;
    std::uint64_t resident_plan_bytes = 0;

    for (size_t i = 0; i < layers.size(); ++i) {
        const int layer = layers[i];
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        Napi::Object layer_out = Napi::Object::New(env);
        Napi::Object targets_out = Napi::Object::New(env);
        std::uint64_t layer_bytes = 0;
        std::uint64_t layer_complete = 0;

        layer_out.Set("layer", Napi::Number::New(env, layer));
        for (const auto& target : targets) {
            ++expected_groups;
            const std::string group_name = prefix + target;
            Napi::Object group = Napi::Object::New(env);
            group.Set("kind", Napi::String::New(env, "lora_projection"));
            group.Set("group", Napi::String::New(env, group_name));
            bool complete = true;
            std::uint64_t group_bytes = 0;
            for (const auto& part : {"lora_a", "lora_b"}) {
                const std::string tensor_name = group_name + "." + part;
                auto found = by_name.find(tensor_name);
                if (found == by_name.end()) {
                    complete = false;
                    missing.push_back(tensor_name);
                    group.Set(part, env.Null());
                } else {
                    group_bytes += found->second.byte_size;
                    group.Set(part, TensorDescriptorToNapi(env, found->second));
                }
            }
            group.Set("complete", Napi::Boolean::New(env, complete));
            group.Set("rank", Napi::Number::New(env, rank));
            group.Set("scale", Napi::Number::New(env, scale));
            group.Set("byte_size", Napi::Number::New(env, static_cast<double>(group_bytes)));
            if (complete) {
                ++complete_groups;
                ++layer_complete;
            }
            layer_bytes += group_bytes;
            targets_out.Set(target, group);
        }

        resident_plan_bytes += layer_bytes;
        layer_out.Set("target_groups", targets_out);
        layer_out.Set("target_count", Napi::Number::New(env, static_cast<double>(targets.size())));
        layer_out.Set("complete_target_count", Napi::Number::New(env, static_cast<double>(layer_complete)));
        layer_out.Set("byte_size", Napi::Number::New(env, static_cast<double>(layer_bytes)));
        layers_out.Set(static_cast<uint32_t>(i), layer_out);
    }

    Napi::Array missing_out = Napi::Array::New(env, missing.size());
    for (size_t i = 0; i < missing.size(); ++i) {
        missing_out.Set(static_cast<uint32_t>(i), Napi::String::New(env, missing[i]));
    }

    Napi::Array targets_summary = Napi::Array::New(env, targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        targets_summary.Set(static_cast<uint32_t>(i), Napi::String::New(env, targets[i]));
    }

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("handle", Napi::String::New(env, handle));
    out.Set("adapter_dir", Napi::String::New(env, adapter_dir));
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("descriptor_source", Napi::String::New(env, "safetensors_headers_only"));
    out.Set("descriptor_index_cached", Napi::Boolean::New(env, true));
    out.Set("descriptor_index_owner", Napi::String::New(env, "adapter_handle"));
    out.Set("descriptor_count", Napi::Number::New(env, static_cast<double>(descriptors.size())));
    out.Set("descriptor_total_payload_bytes", Napi::Number::New(env, static_cast<double>(total_payload_bytes)));
    out.Set("rank", Napi::Number::New(env, rank));
    out.Set("scale", Napi::Number::New(env, scale));
    out.Set("layers", layers_out);
    out.Set("layer_count", Napi::Number::New(env, static_cast<double>(layers.size())));
    out.Set("targets", targets_summary);
    out.Set("target_count", Napi::Number::New(env, static_cast<double>(targets.size())));
    out.Set("expected_groups", Napi::Number::New(env, static_cast<double>(expected_groups)));
    out.Set("complete_groups", Napi::Number::New(env, static_cast<double>(complete_groups)));
    out.Set("missing_group_tensor_count", Napi::Number::New(env, static_cast<double>(missing.size())));
    out.Set("missing_group_tensors", missing_out);
    out.Set("resident_adapter_plan_bytes", Napi::Number::New(env, static_cast<double>(resident_plan_bytes)));
    out.Set("strict_layer_range", Napi::String::New(env, layers.empty() ? "" : std::to_string(layers.front()) + ".." + std::to_string(layers.back())));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return scope.Escape(out);
}

static Napi::Object BuildSessionPlan(
    Napi::Env env,
    const GypsyModelRecord& model,
    const GypsyTokenizerRecord& tokenizer,
    const GypsyAdapterRecord* adapter,
    bool& compatible) {
    compatible = true;
    const std::string config_json = ReadTextFile((fs::path(model.model_dir) / "config.json").string());
    const auto layer_count = static_cast<std::uint64_t>(JsonNumberValue(config_json, "num_hidden_layers", 0));
    const auto hidden_size = static_cast<std::uint64_t>(JsonNumberValue(config_json, "hidden_size", 0));
    const auto intermediate_size = static_cast<std::uint64_t>(JsonNumberValue(config_json, "intermediate_size", 0));
    const auto q_heads = static_cast<std::uint64_t>(JsonNumberValue(config_json, "num_attention_heads", 0));
    const auto kv_heads = static_cast<std::uint64_t>(JsonNumberValue(config_json, "num_key_value_heads", 0));
    std::uint64_t head_dim = static_cast<std::uint64_t>(JsonNumberValue(config_json, "head_dim", 0));
    if (head_dim == 0 && hidden_size > 0 && q_heads > 0) head_dim = hidden_size / q_heads;
    const std::uint64_t q_output = q_heads * head_dim;
    const std::uint64_t kv_output = kv_heads * head_dim;

    Napi::Object model_shape = Napi::Object::New(env);
    model_shape.Set("layer_count", Napi::Number::New(env, static_cast<double>(layer_count)));
    model_shape.Set("hidden_size", Napi::Number::New(env, static_cast<double>(hidden_size)));
    model_shape.Set("intermediate_size", Napi::Number::New(env, static_cast<double>(intermediate_size)));
    model_shape.Set("num_attention_heads", Napi::Number::New(env, static_cast<double>(q_heads)));
    model_shape.Set("num_key_value_heads", Napi::Number::New(env, static_cast<double>(kv_heads)));
    model_shape.Set("head_dim", Napi::Number::New(env, static_cast<double>(head_dim)));
    model_shape.Set("q_projection_output_size", Napi::Number::New(env, static_cast<double>(q_output)));
    model_shape.Set("kv_projection_output_size", Napi::Number::New(env, static_cast<double>(kv_output)));

    Napi::Object out = Napi::Object::New(env);
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("model_handle", Napi::String::New(env, model.handle));
    out.Set("tokenizer_handle", Napi::String::New(env, tokenizer.handle));
    out.Set("model_descriptor_index_cached", Napi::Boolean::New(env, !model.tensor_descriptors.empty()));
    out.Set("model_descriptor_count", Napi::Number::New(env, static_cast<double>(model.tensor_descriptors.size())));
    out.Set("tokenizer_payload_loaded", Napi::Boolean::New(env, false));
    out.Set("model_shape", model_shape);

    if (adapter == nullptr) {
        out.Set("adapter_attached", Napi::Boolean::New(env, false));
        out.Set("adapter_compatible", Napi::Boolean::New(env, true));
        out.Set("adapter_error_count", Napi::Number::New(env, 0));
        out.Set("adapter_errors", Napi::Array::New(env, 0));
        return out;
    }

    const std::string adapter_config_json = ReadTextFile((fs::path(adapter->adapter_dir) / "adapter_config.json").string());
    const auto rank = static_cast<std::uint64_t>(JsonNumberValue(adapter_config_json, "rank", 0));
    const double scale = JsonNumberValue(adapter_config_json, "scale", 0);
    std::unordered_map<std::string, TensorDescriptor> adapter_by_name;
    std::vector<int> adapter_layers;
    for (const auto& d : adapter->tensor_descriptors) {
        adapter_by_name[d.name] = d;
        const int layer = LayerIndexForTensorName(d.name);
        if (layer >= 0 && std::find(adapter_layers.begin(), adapter_layers.end(), layer) == adapter_layers.end()) {
            adapter_layers.push_back(layer);
        }
    }
    std::sort(adapter_layers.begin(), adapter_layers.end());

    const struct TargetShape {
        const char* target;
        std::uint64_t input;
        std::uint64_t output;
    } target_shapes[] = {
        {"self_attn.q_proj", hidden_size, q_output},
        {"self_attn.k_proj", hidden_size, kv_output},
        {"self_attn.v_proj", hidden_size, kv_output},
        {"self_attn.o_proj", q_output, hidden_size},
        {"mlp.gate_proj", hidden_size, intermediate_size},
        {"mlp.up_proj", hidden_size, intermediate_size},
        {"mlp.down_proj", intermediate_size, hidden_size}
    };

    std::vector<std::string> errors;
    std::uint64_t checked_groups = 0;
    std::uint64_t compatible_groups = 0;
    for (int layer : adapter_layers) {
        if (layer < 0 || static_cast<std::uint64_t>(layer) >= layer_count) {
            errors.push_back("adapter layer out of model range: " + std::to_string(layer));
            continue;
        }
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        for (const auto& target : target_shapes) {
            ++checked_groups;
            const std::string a_name = prefix + target.target + ".lora_a";
            const std::string b_name = prefix + target.target + ".lora_b";
            auto a = adapter_by_name.find(a_name);
            auto b = adapter_by_name.find(b_name);
            bool ok = true;
            if (a == adapter_by_name.end()) {
                errors.push_back("missing adapter tensor: " + a_name);
                ok = false;
            } else if (!ShapeEquals(a->second.shape, target.input, rank)) {
                errors.push_back("bad adapter A shape: " + a_name);
                ok = false;
            }
            if (b == adapter_by_name.end()) {
                errors.push_back("missing adapter tensor: " + b_name);
                ok = false;
            } else if (!ShapeEquals(b->second.shape, rank, target.output)) {
                errors.push_back("bad adapter B shape: " + b_name);
                ok = false;
            }
            if (ok) ++compatible_groups;
        }
    }
    compatible = errors.empty();

    Napi::Array layers_out = Napi::Array::New(env, adapter_layers.size());
    for (size_t i = 0; i < adapter_layers.size(); ++i) {
        layers_out.Set(static_cast<uint32_t>(i), Napi::Number::New(env, adapter_layers[i]));
    }
    Napi::Array errors_out = Napi::Array::New(env, errors.size());
    for (size_t i = 0; i < errors.size(); ++i) {
        errors_out.Set(static_cast<uint32_t>(i), Napi::String::New(env, errors[i]));
    }

    out.Set("adapter_attached", Napi::Boolean::New(env, true));
    out.Set("adapter_handle", Napi::String::New(env, adapter->handle));
    out.Set("adapter_descriptor_index_cached", Napi::Boolean::New(env, !adapter->tensor_descriptors.empty()));
    out.Set("adapter_descriptor_count", Napi::Number::New(env, static_cast<double>(adapter->tensor_descriptors.size())));
    out.Set("adapter_rank", Napi::Number::New(env, static_cast<double>(rank)));
    out.Set("adapter_scale", Napi::Number::New(env, scale));
    out.Set("adapter_layers", layers_out);
    out.Set("adapter_layer_count", Napi::Number::New(env, static_cast<double>(adapter_layers.size())));
    out.Set("adapter_layer_range", Napi::String::New(env, adapter_layers.empty() ? "" : std::to_string(adapter_layers.front()) + ".." + std::to_string(adapter_layers.back())));
    out.Set("adapter_checked_projection_groups", Napi::Number::New(env, static_cast<double>(checked_groups)));
    out.Set("adapter_compatible_projection_groups", Napi::Number::New(env, static_cast<double>(compatible_groups)));
    out.Set("adapter_compatible", Napi::Boolean::New(env, compatible));
    out.Set("adapter_error_count", Napi::Number::New(env, static_cast<double>(errors.size())));
    out.Set("adapter_errors", errors_out);
    return out;
}

static Napi::Array BuildFileResidencyPlan(
    Napi::Env env,
    const std::vector<TensorDescriptor>& descriptors,
    const std::vector<MappedFileRecord>* mapped_files = nullptr) {
    struct FilePlan {
        std::string source_path;
        std::string source_file;
        std::uint64_t descriptor_count = 0;
        std::uint64_t payload_bytes = 0;
        std::uint64_t min_absolute_offset = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t max_absolute_offset = 0;
        bool all_ranges_valid = true;
        bool all_ranges_non_empty = true;
    };

    std::map<std::string, FilePlan> by_path;
    for (const auto& d : descriptors) {
        const std::string key = d.source_path.empty() ? d.source_file : d.source_path;
        auto& plan = by_path[key];
        plan.source_path = d.source_path;
        plan.source_file = d.source_file;
        plan.descriptor_count += 1;
        plan.payload_bytes += d.byte_size;
        if (d.byte_size == 0 || d.absolute_offset_end <= d.absolute_offset_begin) {
            plan.all_ranges_non_empty = false;
            plan.all_ranges_valid = false;
        }
        if (d.absolute_offset_begin < plan.min_absolute_offset) {
            plan.min_absolute_offset = d.absolute_offset_begin;
        }
        if (d.absolute_offset_end > plan.max_absolute_offset) {
            plan.max_absolute_offset = d.absolute_offset_end;
        }
    }

    Napi::Array out = Napi::Array::New(env, by_path.size());
    size_t i = 0;
    for (const auto& kv : by_path) {
        const auto& plan = kv.second;
        const std::uint64_t file_bytes = !plan.source_path.empty() && fs::exists(plan.source_path)
            ? static_cast<std::uint64_t>(fs::file_size(plan.source_path))
            : 0;
        const std::uint64_t span_begin = plan.min_absolute_offset == std::numeric_limits<std::uint64_t>::max()
            ? 0
            : plan.min_absolute_offset;
        const bool source_exists = !plan.source_path.empty() && fs::exists(plan.source_path);
        const bool span_within_file = source_exists && plan.max_absolute_offset <= file_bytes && span_begin <= plan.max_absolute_offset;
        const bool residency_plan_valid = source_exists && span_within_file && plan.all_ranges_valid && plan.all_ranges_non_empty;
        const MappedFileRecord* mapped = FindMappedFile(mapped_files, plan.source_path);
        const bool mapped_now = mapped != nullptr && mapped->mapping != nullptr && mapped->mapping != MAP_FAILED;

        Napi::Object item = Napi::Object::New(env);
        item.Set("source_file", Napi::String::New(env, plan.source_file));
        item.Set("source_path", Napi::String::New(env, plan.source_path));
        item.Set("source_exists", Napi::Boolean::New(env, source_exists));
        item.Set("file_bytes", Napi::Number::New(env, static_cast<double>(file_bytes)));
        item.Set("descriptor_count", Napi::Number::New(env, static_cast<double>(plan.descriptor_count)));
        item.Set("payload_bytes", Napi::Number::New(env, static_cast<double>(plan.payload_bytes)));
        item.Set("absolute_payload_span", U64VectorToNapiArray(env, {span_begin, plan.max_absolute_offset}));
        item.Set("span_bytes", Napi::Number::New(env, static_cast<double>(plan.max_absolute_offset >= span_begin ? plan.max_absolute_offset - span_begin : 0)));
        item.Set("span_within_file", Napi::Boolean::New(env, span_within_file));
        item.Set("all_ranges_non_empty", Napi::Boolean::New(env, plan.all_ranges_non_empty));
        item.Set("all_ranges_valid", Napi::Boolean::New(env, plan.all_ranges_valid));
        item.Set("residency_plan_valid", Napi::Boolean::New(env, residency_plan_valid));
        item.Set("mmap_planned", Napi::Boolean::New(env, true));
        item.Set("mapped_now", Napi::Boolean::New(env, mapped_now));
        item.Set("mapping_owner", mapped_now ? Napi::String::New(env, "native_session").As<Napi::Value>() : env.Null());
        item.Set("mapped_file_bytes", Napi::Number::New(env, mapped == nullptr ? 0.0 : static_cast<double>(mapped->file_bytes)));
        item.Set("mapped_payload_bytes", Napi::Number::New(env, mapped == nullptr ? 0.0 : static_cast<double>(mapped->payload_bytes)));
        item.Set("payload_loaded", Napi::Boolean::New(env, false));
        out.Set(static_cast<uint32_t>(i++), item);
    }
    return out;
}

static Napi::Object BuildWarmPlan(
    Napi::Env env,
    const GypsySessionRecord& session,
    const GypsyModelRecord& model,
    const GypsyTokenizerRecord& tokenizer,
    const GypsyAdapterRecord* adapter) {
    std::uint64_t model_layer_bytes = 0;
    std::uint64_t model_global_bytes = 0;
    std::uint64_t model_total_bytes = 0;
    for (const auto& d : model.tensor_descriptors) {
        model_total_bytes += d.byte_size;
        if (d.name.rfind("model.layers.", 0) == 0) {
            model_layer_bytes += d.byte_size;
        } else {
            model_global_bytes += d.byte_size;
        }
    }

    std::uint64_t adapter_bytes = 0;
    if (adapter != nullptr) {
        for (const auto& d : adapter->tensor_descriptors) {
            adapter_bytes += d.byte_size;
        }
    }

    Napi::Array execution_units = Napi::Array::New(env, 8);
    const std::vector<std::string> units = {
        "tokenizer_prompt_plan",
        "resident_model_projection_arrays",
        "resident_norm_weights",
        "resident_adapter_lora_arrays",
        "native_prompt_prefill",
        "native_decode_loop",
        "native_kv_cache",
        "native_sampling_stop_handling"
    };
    for (size_t i = 0; i < units.size(); ++i) {
        execution_units.Set(static_cast<uint32_t>(i), Napi::String::New(env, units[i]));
    }

    Napi::Object out = Napi::Object::New(env);
    out.Set("warm_plan_version", Napi::String::New(env, "gypsy-metadata-warm-plan/1"));
    out.Set("session", Napi::String::New(env, session.handle));
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("resident_arrays_constructed", Napi::Boolean::New(env, false));
    out.Set("legacy_loader_used", Napi::Boolean::New(env, false));
    out.Set("execution_owner", Napi::String::New(env, "native"));
    out.Set("generation_loop_owner", Napi::String::New(env, "native"));
    out.Set("coffeescript_tensor_payload_allowed", Napi::Boolean::New(env, false));
    out.Set("model_handle", Napi::String::New(env, model.handle));
    out.Set("model_descriptor_index_cached", Napi::Boolean::New(env, !model.tensor_descriptors.empty()));
    out.Set("model_descriptor_count", Napi::Number::New(env, static_cast<double>(model.tensor_descriptors.size())));
    out.Set("model_file_residency_plan", BuildFileResidencyPlan(env, model.tensor_descriptors, &session.model_mapped_files));
    out.Set("model_mapped_file_count", Napi::Number::New(env, static_cast<double>(session.model_mapped_files.size())));
    out.Set("model_mapped_file_bytes", Napi::Number::New(env, static_cast<double>(MappedFilesByteTotal(session.model_mapped_files))));
    out.Set("model_mapped_payload_bytes", Napi::Number::New(env, static_cast<double>(MappedFilesPayloadByteTotal(session.model_mapped_files))));
    out.Set("model_layer_resident_plan_bytes", Napi::Number::New(env, static_cast<double>(model_layer_bytes)));
    out.Set("model_global_resident_plan_bytes", Napi::Number::New(env, static_cast<double>(model_global_bytes)));
    out.Set("model_total_payload_bytes", Napi::Number::New(env, static_cast<double>(model_total_bytes)));
    out.Set("tokenizer_handle", Napi::String::New(env, tokenizer.handle));
    out.Set("tokenizer_payload_loaded", Napi::Boolean::New(env, false));
    out.Set("adapter_attached", Napi::Boolean::New(env, adapter != nullptr));
    out.Set("adapter_handle", adapter == nullptr ? env.Null() : Napi::String::New(env, adapter->handle).As<Napi::Value>());
    out.Set("adapter_descriptor_index_cached", Napi::Boolean::New(env, adapter != nullptr && !adapter->tensor_descriptors.empty()));
    out.Set("adapter_descriptor_count", Napi::Number::New(env, adapter == nullptr ? 0 : static_cast<double>(adapter->tensor_descriptors.size())));
    out.Set("adapter_file_residency_plan", adapter == nullptr ? Napi::Array::New(env, 0) : BuildFileResidencyPlan(env, adapter->tensor_descriptors, &session.adapter_mapped_files));
    out.Set("adapter_mapped_file_count", Napi::Number::New(env, static_cast<double>(session.adapter_mapped_files.size())));
    out.Set("adapter_mapped_file_bytes", Napi::Number::New(env, static_cast<double>(MappedFilesByteTotal(session.adapter_mapped_files))));
    out.Set("adapter_mapped_payload_bytes", Napi::Number::New(env, static_cast<double>(MappedFilesPayloadByteTotal(session.adapter_mapped_files))));
    out.Set("adapter_resident_plan_bytes", Napi::Number::New(env, static_cast<double>(adapter_bytes)));
    out.Set("total_resident_plan_bytes", Napi::Number::New(env, static_cast<double>(model_total_bytes + adapter_bytes)));
    out.Set("mapped_file_count", Napi::Number::New(env, static_cast<double>(session.model_mapped_files.size() + session.adapter_mapped_files.size())));
    out.Set("mapped_file_bytes", Napi::Number::New(env, static_cast<double>(MappedFilesByteTotal(session.model_mapped_files) + MappedFilesByteTotal(session.adapter_mapped_files))));
    out.Set("mapped_payload_bytes", Napi::Number::New(env, static_cast<double>(MappedFilesPayloadByteTotal(session.model_mapped_files) + MappedFilesPayloadByteTotal(session.adapter_mapped_files))));
    out.Set("execution_units", execution_units);
    return out;
}

Napi::Value CreateSession(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsString() || !info[1].IsString()) {
        Napi::TypeError::New(env, "Expected model handle and tokenizer handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string model_handle = info[0].As<Napi::String>();
    const std::string tokenizer_handle = info[1].As<Napi::String>();
    std::string adapter_handle;
    if (info.Length() >= 3 && info[2].IsObject()) {
        Napi::Object opts = info[2].As<Napi::Object>();
        if (opts.Has("adapter") && opts.Get("adapter").IsString()) {
            adapter_handle = opts.Get("adapter").As<Napi::String>();
        }
    }
    auto model_it = gypsy_models.find(model_handle);
    if (model_it == gypsy_models.end()) {
        Napi::Error::New(env, "Unknown model handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    auto tokenizer_it = gypsy_tokenizers.find(tokenizer_handle);
    if (tokenizer_it == gypsy_tokenizers.end()) {
        Napi::Error::New(env, "Unknown tokenizer handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    GypsyAdapterRecord* adapter_record = nullptr;
    if (!adapter_handle.empty()) {
        auto adapter_it = gypsy_adapters.find(adapter_handle);
        if (adapter_it == gypsy_adapters.end()) {
            Napi::Error::New(env, "Unknown adapter handle").ThrowAsJavaScriptException();
            return env.Null();
        }
        adapter_record = &adapter_it->second;
    }

    bool session_compatible = true;
    Napi::Object session_plan = BuildSessionPlan(env, model_it->second, tokenizer_it->second, adapter_record, session_compatible);
    if (!session_compatible) {
        Napi::Error::New(env, "Adapter is incompatible with model session plan").ThrowAsJavaScriptException();
        return env.Null();
    }

    const std::string handle = MakeHandle("gsess", gypsy_next_session_id++);
    GypsySessionRecord session_record;
    session_record.handle = handle;
    session_record.model_handle = model_handle;
    session_record.tokenizer_handle = tokenizer_handle;
    session_record.adapter_handle = adapter_handle;
    session_record.warmed = false;
    gypsy_sessions.emplace(handle, std::move(session_record));

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("handle", Napi::String::New(env, handle));
    out.Set("model", Napi::String::New(env, model_handle));
    out.Set("tokenizer", Napi::String::New(env, tokenizer_handle));
    out.Set("adapter", adapter_handle.empty() ? env.Null() : Napi::String::New(env, adapter_handle).As<Napi::Value>());
    out.Set("model_metadata", model_it->second.metadata.Value());
    out.Set("tokenizer_metadata", tokenizer_it->second.metadata.Value());
    if (!adapter_handle.empty()) {
        out.Set("adapter_metadata", adapter_record->metadata.Value());
    } else {
        out.Set("adapter_metadata", env.Null());
    }
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("session_plan", session_plan);
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value WarmSession(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto it = gypsy_sessions.find(handle);
    if (it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const bool already_warmed = it->second.warmed;
    GypsySessionRecord& session = it->second;
    auto model_it = gypsy_models.find(session.model_handle);
    auto tokenizer_it = gypsy_tokenizers.find(session.tokenizer_handle);
    if (model_it == gypsy_models.end() || tokenizer_it == gypsy_tokenizers.end()) {
        Napi::Error::New(env, "Session references unloaded model or tokenizer").ThrowAsJavaScriptException();
        return env.Null();
    }
    GypsyAdapterRecord* adapter = nullptr;
    if (!session.adapter_handle.empty()) {
        auto adapter_it = gypsy_adapters.find(session.adapter_handle);
        if (adapter_it == gypsy_adapters.end()) {
            Napi::Error::New(env, "Session references unloaded adapter").ThrowAsJavaScriptException();
            return env.Null();
        }
        adapter = &adapter_it->second;
    }
    if (!already_warmed) {
        std::string map_error;
        if (!MapSafetensorFiles(model_it->second.tensor_descriptors, session.model_mapped_files, map_error)) {
            Napi::Error::New(env, "Failed to map model safetensors: " + map_error).ThrowAsJavaScriptException();
            return env.Null();
        }
        if (adapter != nullptr && !MapSafetensorFiles(adapter->tensor_descriptors, session.adapter_mapped_files, map_error)) {
            session.model_mapped_files.clear();
            Napi::Error::New(env, "Failed to map adapter safetensors: " + map_error).ThrowAsJavaScriptException();
            return env.Null();
        }
    }
    session.warmed = true;
    Napi::Object warm_plan = BuildWarmPlan(env, session, model_it->second, tokenizer_it->second, adapter);

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("warmed", Napi::Boolean::New(env, true));
    out.Set("already_warmed", Napi::Boolean::New(env, already_warmed));
    out.Set("reused_existing_plan", Napi::Boolean::New(env, already_warmed));
    out.Set("status", Napi::String::New(env, "metadata_only_no_payload_loaded"));
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("warm_plan", warm_plan);
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value DescribeSession(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto it = gypsy_sessions.find(handle);
    if (it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }

    const GypsySessionRecord& session = it->second;
    auto model_it = gypsy_models.find(session.model_handle);
    auto tokenizer_it = gypsy_tokenizers.find(session.tokenizer_handle);
    const bool model_live = model_it != gypsy_models.end();
    const bool tokenizer_live = tokenizer_it != gypsy_tokenizers.end();
    bool adapter_live = session.adapter_handle.empty();
    GypsyAdapterRecord* adapter = nullptr;
    if (!session.adapter_handle.empty()) {
        auto adapter_it = gypsy_adapters.find(session.adapter_handle);
        adapter_live = adapter_it != gypsy_adapters.end();
        if (adapter_live) adapter = &adapter_it->second;
    }

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("warmed", Napi::Boolean::New(env, session.warmed));
    out.Set("model", Napi::String::New(env, session.model_handle));
    out.Set("tokenizer", Napi::String::New(env, session.tokenizer_handle));
    out.Set("adapter", session.adapter_handle.empty() ? env.Null() : Napi::String::New(env, session.adapter_handle).As<Napi::Value>());
    out.Set("model_live", Napi::Boolean::New(env, model_live));
    out.Set("tokenizer_live", Napi::Boolean::New(env, tokenizer_live));
    out.Set("adapter_live", Napi::Boolean::New(env, adapter_live));
    out.Set("references_live", Napi::Boolean::New(env, model_live && tokenizer_live && adapter_live));
    out.Set("execution_owner", Napi::String::New(env, "native"));
    out.Set("generation_loop_owner", Napi::String::New(env, "native"));
    out.Set("coffeescript_tensor_payload_allowed", Napi::Boolean::New(env, false));

    if (model_live && tokenizer_live && adapter_live) {
        bool compatible = true;
        Napi::Object session_plan = BuildSessionPlan(env, model_it->second, tokenizer_it->second, adapter, compatible);
        out.Set("session_plan", session_plan);
        out.Set("session_compatible", Napi::Boolean::New(env, compatible));
        if (session.warmed) {
            out.Set("warm_plan", BuildWarmPlan(env, session, model_it->second, tokenizer_it->second, adapter));
        } else {
            out.Set("warm_plan", env.Null());
        }
    } else {
        out.Set("session_plan", env.Null());
        out.Set("session_compatible", Napi::Boolean::New(env, false));
        out.Set("warm_plan", env.Null());
    }
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value DescribeTensorViews(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before tensor views can be described").ThrowAsJavaScriptException();
        return env.Null();
    }
    auto model_it = gypsy_models.find(session.model_handle);
    auto tokenizer_it = gypsy_tokenizers.find(session.tokenizer_handle);
    if (model_it == gypsy_models.end() || tokenizer_it == gypsy_tokenizers.end()) {
        Napi::Error::New(env, "Session references unloaded model or tokenizer").ThrowAsJavaScriptException();
        return env.Null();
    }
    GypsyAdapterRecord* adapter = nullptr;
    if (!session.adapter_handle.empty()) {
        auto adapter_it = gypsy_adapters.find(session.adapter_handle);
        if (adapter_it == gypsy_adapters.end()) {
            Napi::Error::New(env, "Session references unloaded adapter").ThrowAsJavaScriptException();
            return env.Null();
        }
        adapter = &adapter_it->second;
    }

    Napi::Object model_views = BuildTensorViewPlan(
        env,
        "model",
        model_it->second.tensor_descriptors,
        session.model_mapped_files);
    Napi::Object adapter_views = adapter == nullptr
        ? BuildTensorViewPlan(env, "adapter", std::vector<TensorDescriptor>{}, session.adapter_mapped_files)
        : BuildTensorViewPlan(env, "adapter", adapter->tensor_descriptors, session.adapter_mapped_files);

    const bool model_ok = model_views.Get("all_views_resolved").As<Napi::Boolean>().Value();
    const bool adapter_ok = adapter_views.Get("all_views_resolved").As<Napi::Boolean>().Value();

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, model_ok && adapter_ok));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("view_plan_version", Napi::String::New(env, "gypsy-tensor-view-plan/1"));
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("payload_touched", Napi::Boolean::New(env, false));
    out.Set("mlx_arrays_constructed", Napi::Boolean::New(env, false));
    out.Set("coffeescript_tensor_payload_allowed", Napi::Boolean::New(env, false));
    out.Set("model_views", model_views);
    out.Set("adapter_attached", Napi::Boolean::New(env, adapter != nullptr));
    out.Set("adapter_views", adapter_views);
    out.Set("mapped_file_count", Napi::Number::New(env, static_cast<double>(session.model_mapped_files.size() + session.adapter_mapped_files.size())));
    out.Set("mapped_file_bytes", Napi::Number::New(env, static_cast<double>(MappedFilesByteTotal(session.model_mapped_files) + MappedFilesByteTotal(session.adapter_mapped_files))));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value DescribeTypedTensorPlan(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before typed tensor plan can be described").ThrowAsJavaScriptException();
        return env.Null();
    }
    auto model_it = gypsy_models.find(session.model_handle);
    auto tokenizer_it = gypsy_tokenizers.find(session.tokenizer_handle);
    if (model_it == gypsy_models.end() || tokenizer_it == gypsy_tokenizers.end()) {
        Napi::Error::New(env, "Session references unloaded model or tokenizer").ThrowAsJavaScriptException();
        return env.Null();
    }
    GypsyAdapterRecord* adapter = nullptr;
    if (!session.adapter_handle.empty()) {
        auto adapter_it = gypsy_adapters.find(session.adapter_handle);
        if (adapter_it == gypsy_adapters.end()) {
            Napi::Error::New(env, "Session references unloaded adapter").ThrowAsJavaScriptException();
            return env.Null();
        }
        adapter = &adapter_it->second;
    }

    Napi::Object model_plan = BuildTypedTensorPlan(
        env,
        "model",
        model_it->second.tensor_descriptors,
        session.model_mapped_files);
    Napi::Object adapter_plan = adapter == nullptr
        ? BuildTypedTensorPlan(env, "adapter", std::vector<TensorDescriptor>{}, session.adapter_mapped_files)
        : BuildTypedTensorPlan(env, "adapter", adapter->tensor_descriptors, session.adapter_mapped_files);

    const bool model_ok = model_plan.Get("all_array_specs_resolved").As<Napi::Boolean>().Value();
    const bool adapter_ok = adapter_plan.Get("all_array_specs_resolved").As<Napi::Boolean>().Value();

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, model_ok && adapter_ok));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("typed_plan_version", Napi::String::New(env, "gypsy-typed-tensor-plan/1"));
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("payload_touched", Napi::Boolean::New(env, false));
    out.Set("mlx_arrays_constructed", Napi::Boolean::New(env, false));
    out.Set("coffeescript_tensor_payload_allowed", Napi::Boolean::New(env, false));
    out.Set("model_plan", model_plan);
    out.Set("adapter_attached", Napi::Boolean::New(env, adapter != nullptr));
    out.Set("adapter_plan", adapter_plan);
    out.Set("mapped_file_count", Napi::Number::New(env, static_cast<double>(session.model_mapped_files.size() + session.adapter_mapped_files.size())));
    out.Set("mapped_file_bytes", Napi::Number::New(env, static_cast<double>(MappedFilesByteTotal(session.model_mapped_files) + MappedFilesByteTotal(session.adapter_mapped_files))));
    out.Set("next_loader_step", Napi::String::New(env, "construct_selected_mlx_arrays_from_typed_specs"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value ConstructSelectedResidentArrays(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before resident arrays can be constructed").ThrowAsJavaScriptException();
        return env.Null();
    }
    auto model_it = gypsy_models.find(session.model_handle);
    auto tokenizer_it = gypsy_tokenizers.find(session.tokenizer_handle);
    if (model_it == gypsy_models.end() || tokenizer_it == gypsy_tokenizers.end()) {
        Napi::Error::New(env, "Session references unloaded model or tokenizer").ThrowAsJavaScriptException();
        return env.Null();
    }
    GypsyAdapterRecord* adapter = nullptr;
    if (!session.adapter_handle.empty()) {
        auto adapter_it = gypsy_adapters.find(session.adapter_handle);
        if (adapter_it == gypsy_adapters.end()) {
            Napi::Error::New(env, "Session references unloaded adapter").ThrowAsJavaScriptException();
            return env.Null();
        }
        adapter = &adapter_it->second;
    }

    const bool already_constructed = !session.selected_resident_arrays.empty();
    if (!already_constructed) {
        const std::vector<std::string> model_tensor_names = {
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.biases",
            "model.layers.20.self_attn.q_proj.weight",
            "model.layers.20.self_attn.q_proj.scales",
            "model.layers.20.self_attn.q_proj.biases",
            "model.layers.20.self_attn.k_proj.weight",
            "model.layers.20.self_attn.k_proj.scales",
            "model.layers.20.self_attn.k_proj.biases",
            "model.layers.20.self_attn.v_proj.weight",
            "model.layers.20.self_attn.v_proj.scales",
            "model.layers.20.self_attn.v_proj.biases",
            "model.layers.20.self_attn.o_proj.weight",
            "model.layers.20.self_attn.o_proj.scales",
            "model.layers.20.self_attn.o_proj.biases",
            "model.layers.20.mlp.gate_proj.weight",
            "model.layers.20.mlp.gate_proj.scales",
            "model.layers.20.mlp.gate_proj.biases",
            "model.layers.20.mlp.up_proj.weight",
            "model.layers.20.mlp.up_proj.scales",
            "model.layers.20.mlp.up_proj.biases",
            "model.layers.20.mlp.down_proj.weight",
            "model.layers.20.mlp.down_proj.scales",
            "model.layers.20.mlp.down_proj.biases",
            "model.layers.20.self_attn.q_norm.weight",
            "model.layers.20.self_attn.k_norm.weight",
            "model.layers.20.post_attention_layernorm.weight",
            "model.norm.weight"
        };
        const std::vector<std::string> adapter_tensor_names = {
            "model.layers.20.self_attn.q_proj.lora_a",
            "model.layers.20.self_attn.q_proj.lora_b",
            "model.layers.20.self_attn.k_proj.lora_a",
            "model.layers.20.self_attn.k_proj.lora_b",
            "model.layers.20.self_attn.v_proj.lora_a",
            "model.layers.20.self_attn.v_proj.lora_b",
            "model.layers.20.self_attn.o_proj.lora_a",
            "model.layers.20.self_attn.o_proj.lora_b",
            "model.layers.20.mlp.gate_proj.lora_a",
            "model.layers.20.mlp.gate_proj.lora_b",
            "model.layers.20.mlp.up_proj.lora_a",
            "model.layers.20.mlp.up_proj.lora_b",
            "model.layers.20.mlp.down_proj.lora_a",
            "model.layers.20.mlp.down_proj.lora_b"
        };

        std::vector<ResidentArrayRecord> constructed;
        std::string error;
        for (const auto& name : model_tensor_names) {
            const TensorDescriptor* d = FindTensorDescriptor(model_it->second.tensor_descriptors, name);
            if (d == nullptr) {
                Napi::Error::New(env, "Missing selected model tensor: " + name).ThrowAsJavaScriptException();
                return env.Null();
            }
            auto record = ConstructMappedMlxArray(*d, session.model_mapped_files, error);
            if (!record) {
                Napi::Error::New(env, "Failed to construct selected model MLX array: " + error).ThrowAsJavaScriptException();
                return env.Null();
            }
            constructed.push_back(std::move(*record));
        }

        if (adapter != nullptr) {
            for (const auto& name : adapter_tensor_names) {
                const TensorDescriptor* d = FindTensorDescriptor(adapter->tensor_descriptors, name);
                if (d == nullptr) {
                    Napi::Error::New(env, "Missing selected adapter tensor: " + name).ThrowAsJavaScriptException();
                    return env.Null();
                }
                auto record = ConstructMappedMlxArray(*d, session.adapter_mapped_files, error);
                if (!record) {
                    Napi::Error::New(env, "Failed to construct selected adapter MLX array: " + error).ThrowAsJavaScriptException();
                    return env.Null();
                }
                constructed.push_back(std::move(*record));
            }
        }

        session.selected_resident_arrays = std::move(constructed);
    }

    Napi::Array arrays = Napi::Array::New(env, session.selected_resident_arrays.size());
    for (size_t i = 0; i < session.selected_resident_arrays.size(); ++i) {
        const auto& r = session.selected_resident_arrays[i];
        Napi::Object item = Napi::Object::New(env);
        item.Set("name", Napi::String::New(env, r.name));
        item.Set("role", Napi::String::New(env, r.role));
        item.Set("array_kind", Napi::String::New(env, r.array_kind));
        item.Set("dtype", Napi::String::New(env, r.dtype));
        item.Set("byte_size", Napi::Number::New(env, static_cast<double>(r.byte_size)));
        item.Set("mlx_array_constructed", Napi::Boolean::New(env, true));
        item.Set("native_payload_copied_to_mlx_allocator", Napi::Boolean::New(env, true));
        item.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
        item.Set("payload_loaded_by_coffeescript", Napi::Boolean::New(env, false));
        item.Set("shape_rank", Napi::Number::New(env, static_cast<double>(r.array.ndim())));
        item.Set("element_count", Napi::Number::New(env, static_cast<double>(r.array.size())));
        item.Set("nbytes", Napi::Number::New(env, static_cast<double>(r.array.nbytes())));
        arrays.Set(static_cast<uint32_t>(i), item);
    }

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_resident_array_plan_version", Napi::String::New(env, "gypsy-selected-resident-arrays/1"));
    out.Set("already_constructed", Napi::Boolean::New(env, already_constructed));
    out.Set("reused_existing_arrays", Napi::Boolean::New(env, already_constructed));
    out.Set("array_count", Napi::Number::New(env, static_cast<double>(session.selected_resident_arrays.size())));
    out.Set("array_bytes", Napi::Number::New(env, static_cast<double>(ResidentArraysByteTotal(session.selected_resident_arrays))));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("payload_loaded_by_coffeescript", Napi::Boolean::New(env, false));
    out.Set("arrays", arrays);
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value DescribeSelectedResidentGroups(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before resident groups can be described").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before resident groups can be described").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto add_array_ref = [&](Napi::Env env, Napi::Object group, const std::string& field, const std::string& name, bool& complete, std::uint64_t& byte_total) {
        const ResidentArrayRecord* record = FindResidentArray(session.selected_resident_arrays, name);
        if (record == nullptr) {
            complete = false;
            group.Set(field, env.Null());
            return;
        }
        Napi::Object item = Napi::Object::New(env);
        item.Set("name", Napi::String::New(env, record->name));
        item.Set("role", Napi::String::New(env, record->role));
        item.Set("array_kind", Napi::String::New(env, record->array_kind));
        item.Set("dtype", Napi::String::New(env, record->dtype));
        item.Set("shape_rank", Napi::Number::New(env, static_cast<double>(record->array.ndim())));
        item.Set("element_count", Napi::Number::New(env, static_cast<double>(record->array.size())));
        item.Set("nbytes", Napi::Number::New(env, static_cast<double>(record->array.nbytes())));
        item.Set("mlx_array_constructed", Napi::Boolean::New(env, true));
        item.Set("native_payload_copied_to_mlx_allocator", Napi::Boolean::New(env, true));
        item.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
        group.Set(field, item);
        byte_total += record->byte_size;
    };

    Napi::Object q_proj = Napi::Object::New(env);
    bool q_proj_complete = true;
    std::uint64_t q_proj_bytes = 0;
    q_proj.Set("group", Napi::String::New(env, "model.layers.0.self_attn.q_proj"));
    q_proj.Set("group_kind", Napi::String::New(env, "quantized_projection"));
    add_array_ref(env, q_proj, "weight", "model.layers.0.self_attn.q_proj.weight", q_proj_complete, q_proj_bytes);
    add_array_ref(env, q_proj, "scales", "model.layers.0.self_attn.q_proj.scales", q_proj_complete, q_proj_bytes);
    add_array_ref(env, q_proj, "biases", "model.layers.0.self_attn.q_proj.biases", q_proj_complete, q_proj_bytes);
    q_proj.Set("complete", Napi::Boolean::New(env, q_proj_complete));
    q_proj.Set("byte_size", Napi::Number::New(env, static_cast<double>(q_proj_bytes)));
    q_proj.Set("execution_ready", Napi::Boolean::New(env, q_proj_complete));

    Napi::Object layer20_q_proj = Napi::Object::New(env);
    bool layer20_q_proj_complete = true;
    std::uint64_t layer20_q_proj_bytes = 0;
    layer20_q_proj.Set("group", Napi::String::New(env, "model.layers.20.self_attn.q_proj"));
    layer20_q_proj.Set("group_kind", Napi::String::New(env, "quantized_projection"));
    add_array_ref(env, layer20_q_proj, "weight", "model.layers.20.self_attn.q_proj.weight", layer20_q_proj_complete, layer20_q_proj_bytes);
    add_array_ref(env, layer20_q_proj, "scales", "model.layers.20.self_attn.q_proj.scales", layer20_q_proj_complete, layer20_q_proj_bytes);
    add_array_ref(env, layer20_q_proj, "biases", "model.layers.20.self_attn.q_proj.biases", layer20_q_proj_complete, layer20_q_proj_bytes);
    layer20_q_proj.Set("complete", Napi::Boolean::New(env, layer20_q_proj_complete));
    layer20_q_proj.Set("byte_size", Napi::Number::New(env, static_cast<double>(layer20_q_proj_bytes)));
    layer20_q_proj.Set("execution_ready", Napi::Boolean::New(env, layer20_q_proj_complete));

    Napi::Object layer20_k_proj = Napi::Object::New(env);
    bool layer20_k_proj_complete = true;
    std::uint64_t layer20_k_proj_bytes = 0;
    layer20_k_proj.Set("group", Napi::String::New(env, "model.layers.20.self_attn.k_proj"));
    layer20_k_proj.Set("group_kind", Napi::String::New(env, "quantized_projection"));
    add_array_ref(env, layer20_k_proj, "weight", "model.layers.20.self_attn.k_proj.weight", layer20_k_proj_complete, layer20_k_proj_bytes);
    add_array_ref(env, layer20_k_proj, "scales", "model.layers.20.self_attn.k_proj.scales", layer20_k_proj_complete, layer20_k_proj_bytes);
    add_array_ref(env, layer20_k_proj, "biases", "model.layers.20.self_attn.k_proj.biases", layer20_k_proj_complete, layer20_k_proj_bytes);
    layer20_k_proj.Set("complete", Napi::Boolean::New(env, layer20_k_proj_complete));
    layer20_k_proj.Set("byte_size", Napi::Number::New(env, static_cast<double>(layer20_k_proj_bytes)));
    layer20_k_proj.Set("execution_ready", Napi::Boolean::New(env, layer20_k_proj_complete));

    Napi::Object layer20_v_proj = Napi::Object::New(env);
    bool layer20_v_proj_complete = true;
    std::uint64_t layer20_v_proj_bytes = 0;
    layer20_v_proj.Set("group", Napi::String::New(env, "model.layers.20.self_attn.v_proj"));
    layer20_v_proj.Set("group_kind", Napi::String::New(env, "quantized_projection"));
    add_array_ref(env, layer20_v_proj, "weight", "model.layers.20.self_attn.v_proj.weight", layer20_v_proj_complete, layer20_v_proj_bytes);
    add_array_ref(env, layer20_v_proj, "scales", "model.layers.20.self_attn.v_proj.scales", layer20_v_proj_complete, layer20_v_proj_bytes);
    add_array_ref(env, layer20_v_proj, "biases", "model.layers.20.self_attn.v_proj.biases", layer20_v_proj_complete, layer20_v_proj_bytes);
    layer20_v_proj.Set("complete", Napi::Boolean::New(env, layer20_v_proj_complete));
    layer20_v_proj.Set("byte_size", Napi::Number::New(env, static_cast<double>(layer20_v_proj_bytes)));
    layer20_v_proj.Set("execution_ready", Napi::Boolean::New(env, layer20_v_proj_complete));

    Napi::Object layer20_o_proj = Napi::Object::New(env);
    bool layer20_o_proj_complete = true;
    std::uint64_t layer20_o_proj_bytes = 0;
    layer20_o_proj.Set("group", Napi::String::New(env, "model.layers.20.self_attn.o_proj"));
    layer20_o_proj.Set("group_kind", Napi::String::New(env, "quantized_projection"));
    add_array_ref(env, layer20_o_proj, "weight", "model.layers.20.self_attn.o_proj.weight", layer20_o_proj_complete, layer20_o_proj_bytes);
    add_array_ref(env, layer20_o_proj, "scales", "model.layers.20.self_attn.o_proj.scales", layer20_o_proj_complete, layer20_o_proj_bytes);
    add_array_ref(env, layer20_o_proj, "biases", "model.layers.20.self_attn.o_proj.biases", layer20_o_proj_complete, layer20_o_proj_bytes);
    layer20_o_proj.Set("complete", Napi::Boolean::New(env, layer20_o_proj_complete));
    layer20_o_proj.Set("byte_size", Napi::Number::New(env, static_cast<double>(layer20_o_proj_bytes)));
    layer20_o_proj.Set("execution_ready", Napi::Boolean::New(env, layer20_o_proj_complete));

    auto make_quant_group = [&](const std::string& name) {
        Napi::Object group = Napi::Object::New(env);
        bool complete = true;
        std::uint64_t bytes = 0;
        group.Set("group", Napi::String::New(env, name));
        group.Set("group_kind", Napi::String::New(env, "quantized_projection"));
        add_array_ref(env, group, "weight", name + ".weight", complete, bytes);
        add_array_ref(env, group, "scales", name + ".scales", complete, bytes);
        add_array_ref(env, group, "biases", name + ".biases", complete, bytes);
        group.Set("complete", Napi::Boolean::New(env, complete));
        group.Set("byte_size", Napi::Number::New(env, static_cast<double>(bytes)));
        group.Set("execution_ready", Napi::Boolean::New(env, complete));
        return std::make_pair(group, std::make_pair(complete, bytes));
    };
    auto layer20_gate_proj_pair = make_quant_group("model.layers.20.mlp.gate_proj");
    auto layer20_up_proj_pair = make_quant_group("model.layers.20.mlp.up_proj");
    auto layer20_down_proj_pair = make_quant_group("model.layers.20.mlp.down_proj");
    Napi::Object layer20_gate_proj = layer20_gate_proj_pair.first;
    Napi::Object layer20_up_proj = layer20_up_proj_pair.first;
    Napi::Object layer20_down_proj = layer20_down_proj_pair.first;
    const bool layer20_gate_proj_complete = layer20_gate_proj_pair.second.first;
    const bool layer20_up_proj_complete = layer20_up_proj_pair.second.first;
    const bool layer20_down_proj_complete = layer20_down_proj_pair.second.first;
    const std::uint64_t layer20_gate_proj_bytes = layer20_gate_proj_pair.second.second;
    const std::uint64_t layer20_up_proj_bytes = layer20_up_proj_pair.second.second;
    const std::uint64_t layer20_down_proj_bytes = layer20_down_proj_pair.second.second;

    Napi::Object norms = Napi::Object::New(env);
    bool norms_complete = true;
    std::uint64_t norms_bytes = 0;
    add_array_ref(env, norms, "input_layernorm", "model.layers.0.input_layernorm.weight", norms_complete, norms_bytes);
    add_array_ref(env, norms, "q_norm", "model.layers.0.self_attn.q_norm.weight", norms_complete, norms_bytes);
    add_array_ref(env, norms, "layer20_q_norm", "model.layers.20.self_attn.q_norm.weight", norms_complete, norms_bytes);
    add_array_ref(env, norms, "layer20_k_norm", "model.layers.20.self_attn.k_norm.weight", norms_complete, norms_bytes);
    add_array_ref(env, norms, "layer20_post_attention_layernorm", "model.layers.20.post_attention_layernorm.weight", norms_complete, norms_bytes);
    add_array_ref(env, norms, "final_norm", "model.norm.weight", norms_complete, norms_bytes);
    norms.Set("group_kind", Napi::String::New(env, "norm_weights"));
    norms.Set("complete", Napi::Boolean::New(env, norms_complete));
    norms.Set("byte_size", Napi::Number::New(env, static_cast<double>(norms_bytes)));
    norms.Set("execution_ready", Napi::Boolean::New(env, norms_complete));

    Napi::Object lora = Napi::Object::New(env);
    bool lora_complete = true;
    std::uint64_t lora_bytes = 0;
    lora.Set("group", Napi::String::New(env, "model.layers.20.self_attn.q_proj"));
    lora.Set("group_kind", Napi::String::New(env, "lora_projection_delta"));
    lora.Set("scale", Napi::Number::New(env, 20.0));
    add_array_ref(env, lora, "lora_a", "model.layers.20.self_attn.q_proj.lora_a", lora_complete, lora_bytes);
    add_array_ref(env, lora, "lora_b", "model.layers.20.self_attn.q_proj.lora_b", lora_complete, lora_bytes);
    lora.Set("complete", Napi::Boolean::New(env, lora_complete));
    lora.Set("byte_size", Napi::Number::New(env, static_cast<double>(lora_bytes)));
    lora.Set("execution_ready", Napi::Boolean::New(env, lora_complete));

    Napi::Object lora_k = Napi::Object::New(env);
    bool lora_k_complete = true;
    std::uint64_t lora_k_bytes = 0;
    lora_k.Set("group", Napi::String::New(env, "model.layers.20.self_attn.k_proj"));
    lora_k.Set("group_kind", Napi::String::New(env, "lora_projection_delta"));
    lora_k.Set("scale", Napi::Number::New(env, 20.0));
    add_array_ref(env, lora_k, "lora_a", "model.layers.20.self_attn.k_proj.lora_a", lora_k_complete, lora_k_bytes);
    add_array_ref(env, lora_k, "lora_b", "model.layers.20.self_attn.k_proj.lora_b", lora_k_complete, lora_k_bytes);
    lora_k.Set("complete", Napi::Boolean::New(env, lora_k_complete));
    lora_k.Set("byte_size", Napi::Number::New(env, static_cast<double>(lora_k_bytes)));
    lora_k.Set("execution_ready", Napi::Boolean::New(env, lora_k_complete));

    Napi::Object lora_v = Napi::Object::New(env);
    bool lora_v_complete = true;
    std::uint64_t lora_v_bytes = 0;
    lora_v.Set("group", Napi::String::New(env, "model.layers.20.self_attn.v_proj"));
    lora_v.Set("group_kind", Napi::String::New(env, "lora_projection_delta"));
    lora_v.Set("scale", Napi::Number::New(env, 20.0));
    add_array_ref(env, lora_v, "lora_a", "model.layers.20.self_attn.v_proj.lora_a", lora_v_complete, lora_v_bytes);
    add_array_ref(env, lora_v, "lora_b", "model.layers.20.self_attn.v_proj.lora_b", lora_v_complete, lora_v_bytes);
    lora_v.Set("complete", Napi::Boolean::New(env, lora_v_complete));
    lora_v.Set("byte_size", Napi::Number::New(env, static_cast<double>(lora_v_bytes)));
    lora_v.Set("execution_ready", Napi::Boolean::New(env, lora_v_complete));

    Napi::Object lora_o = Napi::Object::New(env);
    bool lora_o_complete = true;
    std::uint64_t lora_o_bytes = 0;
    lora_o.Set("group", Napi::String::New(env, "model.layers.20.self_attn.o_proj"));
    lora_o.Set("group_kind", Napi::String::New(env, "lora_projection_delta"));
    lora_o.Set("scale", Napi::Number::New(env, 20.0));
    add_array_ref(env, lora_o, "lora_a", "model.layers.20.self_attn.o_proj.lora_a", lora_o_complete, lora_o_bytes);
    add_array_ref(env, lora_o, "lora_b", "model.layers.20.self_attn.o_proj.lora_b", lora_o_complete, lora_o_bytes);
    lora_o.Set("complete", Napi::Boolean::New(env, lora_o_complete));
    lora_o.Set("byte_size", Napi::Number::New(env, static_cast<double>(lora_o_bytes)));
    lora_o.Set("execution_ready", Napi::Boolean::New(env, lora_o_complete));

    auto make_lora_group = [&](const std::string& name) {
        Napi::Object group = Napi::Object::New(env);
        bool complete = true;
        std::uint64_t bytes = 0;
        group.Set("group", Napi::String::New(env, name));
        group.Set("group_kind", Napi::String::New(env, "lora_projection_delta"));
        group.Set("scale", Napi::Number::New(env, 20.0));
        add_array_ref(env, group, "lora_a", name + ".lora_a", complete, bytes);
        add_array_ref(env, group, "lora_b", name + ".lora_b", complete, bytes);
        group.Set("complete", Napi::Boolean::New(env, complete));
        group.Set("byte_size", Napi::Number::New(env, static_cast<double>(bytes)));
        group.Set("execution_ready", Napi::Boolean::New(env, complete));
        return std::make_pair(group, std::make_pair(complete, bytes));
    };
    auto lora_gate_pair = make_lora_group("model.layers.20.mlp.gate_proj");
    auto lora_up_pair = make_lora_group("model.layers.20.mlp.up_proj");
    auto lora_down_pair = make_lora_group("model.layers.20.mlp.down_proj");
    Napi::Object lora_gate = lora_gate_pair.first;
    Napi::Object lora_up = lora_up_pair.first;
    Napi::Object lora_down = lora_down_pair.first;
    const bool lora_gate_complete = lora_gate_pair.second.first;
    const bool lora_up_complete = lora_up_pair.second.first;
    const bool lora_down_complete = lora_down_pair.second.first;
    const std::uint64_t lora_gate_bytes = lora_gate_pair.second.second;
    const std::uint64_t lora_up_bytes = lora_up_pair.second.second;
    const std::uint64_t lora_down_bytes = lora_down_pair.second.second;

    const bool ok = q_proj_complete && layer20_q_proj_complete && layer20_k_proj_complete && layer20_v_proj_complete && layer20_o_proj_complete && layer20_gate_proj_complete && layer20_up_proj_complete && layer20_down_proj_complete && norms_complete && lora_complete && lora_k_complete && lora_v_complete && lora_o_complete && lora_gate_complete && lora_up_complete && lora_down_complete;
    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, ok));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_resident_group_plan_version", Napi::String::New(env, "gypsy-selected-resident-groups/1"));
    out.Set("selected_resident_array_count", Napi::Number::New(env, static_cast<double>(session.selected_resident_arrays.size())));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("math_executed", Napi::Boolean::New(env, false));
    out.Set("groups_complete", Napi::Boolean::New(env, ok));
    out.Set("quantized_q_proj_group", q_proj);
    out.Set("layer20_quantized_q_proj_group", layer20_q_proj);
    out.Set("layer20_quantized_k_proj_group", layer20_k_proj);
    out.Set("layer20_quantized_v_proj_group", layer20_v_proj);
    out.Set("layer20_quantized_o_proj_group", layer20_o_proj);
    out.Set("layer20_quantized_gate_proj_group", layer20_gate_proj);
    out.Set("layer20_quantized_up_proj_group", layer20_up_proj);
    out.Set("layer20_quantized_down_proj_group", layer20_down_proj);
    out.Set("norm_group", norms);
    out.Set("lora_q_proj_group", lora);
    out.Set("lora_k_proj_group", lora_k);
    out.Set("lora_v_proj_group", lora_v);
    out.Set("lora_o_proj_group", lora_o);
    out.Set("lora_gate_proj_group", lora_gate);
    out.Set("lora_up_proj_group", lora_up);
    out.Set("lora_down_proj_group", lora_down);
    out.Set("group_count", Napi::Number::New(env, 16));
    out.Set("group_bytes", Napi::Number::New(env, static_cast<double>(q_proj_bytes + layer20_q_proj_bytes + layer20_k_proj_bytes + layer20_v_proj_bytes + layer20_o_proj_bytes + layer20_gate_proj_bytes + layer20_up_proj_bytes + layer20_down_proj_bytes + norms_bytes + lora_bytes + lora_k_bytes + lora_v_bytes + lora_o_bytes + lora_gate_bytes + lora_up_bytes + lora_down_bytes)));
    out.Set("next_execution_step", Napi::String::New(env, "run_selected_norm_or_projection_probe"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

static Napi::Object RunRmsNormSubprobe(
    Napi::Env env,
    const std::string& name,
    const ResidentArrayRecord& weight,
    int input_size,
    float eps) {
    std::vector<float> input(static_cast<size_t>(input_size));
    for (int i = 0; i < input_size; ++i) {
        input[static_cast<size_t>(i)] = static_cast<float>((i % 17) - 8) / 17.0f;
    }

    auto start = std::chrono::steady_clock::now();
    mlx::core::array input_array(input.begin(), mlx::core::Shape{input_size}, mlx::core::float32);
    mlx::core::array output = mlx::core::astype(
        mlx::core::fast::rms_norm(input_array, std::make_optional(weight.array), eps),
        mlx::core::float32);
    mlx::core::eval(output);
    auto end = std::chrono::steady_clock::now();

    const float* values = output.data<float>();
    double checksum = 0.0;
    for (int i = 0; i < input_size; ++i) {
        checksum += static_cast<double>(values[i]);
    }

    Napi::Array first_values = Napi::Array::New(env, 8);
    const int sample_count = std::min(input_size, 8);
    for (int i = 0; i < sample_count; ++i) {
        first_values.Set(static_cast<uint32_t>(i), Napi::Number::New(env, values[i]));
    }

    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    Napi::Object out = Napi::Object::New(env);
    out.Set("name", Napi::String::New(env, name));
    out.Set("weight_tensor", Napi::String::New(env, weight.name));
    out.Set("weight_dtype", Napi::String::New(env, weight.dtype));
    out.Set("weight_elements", Napi::Number::New(env, static_cast<double>(weight.array.size())));
    out.Set("input_len", Napi::Number::New(env, input_size));
    out.Set("output_len", Napi::Number::New(env, static_cast<double>(output.size())));
    out.Set("eps", Napi::Number::New(env, eps));
    out.Set("checksum", Napi::Number::New(env, checksum));
    out.Set("first_values", first_values);
    out.Set("timing_ms", Napi::Number::New(env, timing_ms));
    out.Set("backend", Napi::String::New(env, "mlx"));
    out.Set("readback_reason", Napi::String::New(env, "probe_output_summary"));
    return out;
}

Napi::Value RunSelectedNormProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected norm probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected norm probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }

    const ResidentArrayRecord* input_norm = FindResidentArray(session.selected_resident_arrays, "model.layers.0.input_layernorm.weight");
    const ResidentArrayRecord* q_norm = FindResidentArray(session.selected_resident_arrays, "model.layers.0.self_attn.q_norm.weight");
    const ResidentArrayRecord* final_norm = FindResidentArray(session.selected_resident_arrays, "model.norm.weight");
    if (input_norm == nullptr || q_norm == nullptr || final_norm == nullptr) {
        Napi::Error::New(env, "Selected norm arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    const float eps = 1.0e-6f;
    Napi::Object input_probe = RunRmsNormSubprobe(env, "layer0_input_rmsnorm", *input_norm, 2560, eps);
    Napi::Object q_probe = RunRmsNormSubprobe(env, "layer0_q_rmsnorm", *q_norm, 128, eps);
    Napi::Object final_probe = RunRmsNormSubprobe(env, "final_rmsnorm", *final_norm, 2560, eps);

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_norm_probe_version", Napi::String::New(env, "gypsy-selected-norm-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_inputs_are_deterministic", Napi::Boolean::New(env, true));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_norm_arrays_only"));
    out.Set("subprobe_count", Napi::Number::New(env, 3));
    out.Set("readback_count", Napi::Number::New(env, 3));
    Napi::Array reasons = Napi::Array::New(env, 3);
    reasons.Set(static_cast<uint32_t>(0), Napi::String::New(env, "layer0_input_rmsnorm_summary"));
    reasons.Set(static_cast<uint32_t>(1), Napi::String::New(env, "layer0_q_rmsnorm_summary"));
    reasons.Set(static_cast<uint32_t>(2), Napi::String::New(env, "final_rmsnorm_summary"));
    out.Set("readback_reasons", reasons);
    out.Set("layer0_input_rmsnorm", input_probe);
    out.Set("layer0_q_rmsnorm", q_probe);
    out.Set("final_rmsnorm", final_probe);
    out.Set("next_execution_step", Napi::String::New(env, "run_selected_quantized_projection_probe"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedQuantizedProjectionProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected quantized projection probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected quantized projection probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }

    const ResidentArrayRecord* weight = FindResidentArray(session.selected_resident_arrays, "model.layers.0.self_attn.q_proj.weight");
    const ResidentArrayRecord* scales = FindResidentArray(session.selected_resident_arrays, "model.layers.0.self_attn.q_proj.scales");
    const ResidentArrayRecord* biases = FindResidentArray(session.selected_resident_arrays, "model.layers.0.self_attn.q_proj.biases");
    if (weight == nullptr || scales == nullptr || biases == nullptr) {
        Napi::Error::New(env, "Selected q_proj quantized arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    constexpr int input_size = 2560;
    constexpr int output_size = 4096;
    constexpr int group_size = 64;
    constexpr int bits = 4;

    std::vector<float> input(static_cast<size_t>(input_size));
    for (int i = 0; i < input_size; ++i) {
        input[static_cast<size_t>(i)] = static_cast<float>((i % 23) - 11) / 23.0f;
    }

    auto start = std::chrono::steady_clock::now();
    mlx::core::array input_array(input.begin(), mlx::core::Shape{1, input_size}, mlx::core::float32);
    mlx::core::array output = mlx::core::astype(
        mlx::core::quantized_matmul(
            input_array,
            weight->array,
            scales->array,
            std::make_optional(biases->array),
            true,
            group_size,
            bits,
            "affine"),
        mlx::core::float32);
    mlx::core::eval(output);
    auto end = std::chrono::steady_clock::now();

    const float* values = output.data<float>();
    double checksum = 0.0;
    double abs_checksum = 0.0;
    for (int i = 0; i < output_size; ++i) {
        checksum += static_cast<double>(values[i]);
        abs_checksum += std::abs(static_cast<double>(values[i]));
    }

    Napi::Array first_values = Napi::Array::New(env, 8);
    for (int i = 0; i < 8; ++i) {
        first_values.Set(static_cast<uint32_t>(i), Napi::Number::New(env, values[i]));
    }

    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    Napi::Object projection = Napi::Object::New(env);
    projection.Set("group", Napi::String::New(env, "model.layers.0.self_attn.q_proj"));
    projection.Set("input_len", Napi::Number::New(env, input_size));
    projection.Set("output_len", Napi::Number::New(env, static_cast<double>(output.size())));
    projection.Set("expected_output_len", Napi::Number::New(env, output_size));
    projection.Set("group_size", Napi::Number::New(env, group_size));
    projection.Set("bits", Napi::Number::New(env, bits));
    projection.Set("mode", Napi::String::New(env, "affine"));
    projection.Set("transpose", Napi::Boolean::New(env, true));
    projection.Set("checksum", Napi::Number::New(env, checksum));
    projection.Set("abs_checksum", Napi::Number::New(env, abs_checksum));
    projection.Set("first_values", first_values);
    projection.Set("timing_ms", Napi::Number::New(env, timing_ms));
    projection.Set("backend", Napi::String::New(env, "mlx"));
    projection.Set("readback_reason", Napi::String::New(env, "probe_output_summary"));

    std::vector<float> metal_values(static_cast<size_t>(output_size), 0.0f);
    std::vector<float> tiled_metal_values(static_cast<size_t>(output_size), 0.0f);
    GypsyDirectMetalProbeResult metal_projection = GypsyRunDirectMetalQuantizedProjectionCompare(
        input.data(),
        static_cast<const std::uint32_t*>(weight->raw_data),
        static_cast<const std::uint16_t*>(scales->raw_data),
        static_cast<const std::uint16_t*>(biases->raw_data),
        values,
        metal_values.data(),
        static_cast<std::uint32_t>(output_size),
        320,
        static_cast<std::uint32_t>(input_size));
    GypsyDirectMetalProbeResult tiled_metal_projection = GypsyRunDirectMetalQuantizedProjectionTiledCompare(
        input.data(),
        static_cast<const std::uint32_t*>(weight->raw_data),
        static_cast<const std::uint16_t*>(scales->raw_data),
        static_cast<const std::uint16_t*>(biases->raw_data),
        values,
        tiled_metal_values.data(),
        static_cast<std::uint32_t>(output_size),
        320,
        static_cast<std::uint32_t>(input_size));

    Napi::Array metal_first_values = Napi::Array::New(env, 8);
    double metal_abs_checksum = 0.0;
    for (int i = 0; i < output_size; ++i) {
        metal_abs_checksum += std::abs(static_cast<double>(metal_values[static_cast<size_t>(i)]));
        if (i < 8) {
            metal_first_values.Set(static_cast<uint32_t>(i), Napi::Number::New(env, metal_values[static_cast<size_t>(i)]));
        }
    }
    Napi::Object compiled_metal = Napi::Object::New(env);
    compiled_metal.Set("available", Napi::Boolean::New(env, true));
    compiled_metal.Set("backend", Napi::String::New(env, "direct_compiled_metal_quantized_projection"));
    compiled_metal.Set("ok", Napi::Boolean::New(env, metal_projection.ok));
    compiled_metal.Set("device_available", Napi::Boolean::New(env, metal_projection.device_available));
    compiled_metal.Set("library_compiled", Napi::Boolean::New(env, metal_projection.library_compiled));
    compiled_metal.Set("pipeline_created", Napi::Boolean::New(env, metal_projection.pipeline_created));
    compiled_metal.Set("command_completed", Napi::Boolean::New(env, metal_projection.command_completed));
    compiled_metal.Set("checksum", Napi::Number::New(env, metal_projection.checksum));
    compiled_metal.Set("abs_checksum", Napi::Number::New(env, metal_abs_checksum));
    compiled_metal.Set("max_abs_diff_vs_mlx", Napi::Number::New(env, metal_projection.max_abs_diff));
    compiled_metal.Set("timing_ms", Napi::Number::New(env, metal_projection.elapsed_ms));
    compiled_metal.Set("first_values", metal_first_values);
    compiled_metal.Set("kernel_unit", Napi::String::New(env, "one_thread_per_output_row_u32_4bit_bf16_affine"));
    if (metal_projection.error != nullptr) {
        compiled_metal.Set("error", Napi::String::New(env, metal_projection.error));
    } else {
        compiled_metal.Set("error", env.Null());
    }

    Napi::Array tiled_first_values = Napi::Array::New(env, 8);
    double tiled_abs_checksum = 0.0;
    for (int i = 0; i < output_size; ++i) {
        tiled_abs_checksum += std::abs(static_cast<double>(tiled_metal_values[static_cast<size_t>(i)]));
        if (i < 8) {
            tiled_first_values.Set(static_cast<uint32_t>(i), Napi::Number::New(env, tiled_metal_values[static_cast<size_t>(i)]));
        }
    }
    Napi::Object tiled_compiled_metal = Napi::Object::New(env);
    tiled_compiled_metal.Set("available", Napi::Boolean::New(env, true));
    tiled_compiled_metal.Set("backend", Napi::String::New(env, "direct_compiled_metal_quantized_projection_tiled"));
    tiled_compiled_metal.Set("ok", Napi::Boolean::New(env, tiled_metal_projection.ok));
    tiled_compiled_metal.Set("device_available", Napi::Boolean::New(env, tiled_metal_projection.device_available));
    tiled_compiled_metal.Set("library_compiled", Napi::Boolean::New(env, tiled_metal_projection.library_compiled));
    tiled_compiled_metal.Set("pipeline_created", Napi::Boolean::New(env, tiled_metal_projection.pipeline_created));
    tiled_compiled_metal.Set("command_completed", Napi::Boolean::New(env, tiled_metal_projection.command_completed));
    tiled_compiled_metal.Set("checksum", Napi::Number::New(env, tiled_metal_projection.checksum));
    tiled_compiled_metal.Set("abs_checksum", Napi::Number::New(env, tiled_abs_checksum));
    tiled_compiled_metal.Set("max_abs_diff_vs_mlx", Napi::Number::New(env, tiled_metal_projection.max_abs_diff));
    tiled_compiled_metal.Set("timing_ms", Napi::Number::New(env, tiled_metal_projection.elapsed_ms));
    tiled_compiled_metal.Set("first_values", tiled_first_values);
    tiled_compiled_metal.Set("kernel_unit", Napi::String::New(env, "one_threadgroup_per_output_row_64_threads_reduction"));
    tiled_compiled_metal.Set("threads_per_output_row", Napi::Number::New(env, 64));
    tiled_compiled_metal.Set("shape_guard", Napi::String::New(env, "input_len % 64 == 0 && packed_cols == input_len / 8"));
    if (tiled_metal_projection.error != nullptr) {
        tiled_compiled_metal.Set("error", Napi::String::New(env, tiled_metal_projection.error));
    } else {
        tiled_compiled_metal.Set("error", env.Null());
    }

    static const std::string mlx_resident_projection_source = R"METAL(
        uint row = thread_position_in_grid.x;
        if (row >= 4096) {
            return;
        }
        constexpr uint group_size = 64;
        constexpr uint values_per_word = 8;
        constexpr uint words_per_group = group_size / values_per_word;
        constexpr uint groups = 2560 / group_size;
        constexpr uint packed_cols = 2560 / values_per_word;
        float acc = 0.0f;
        for (uint g = 0; g < groups; ++g) {
            const float scale = float(scales[row * groups + g]);
            const float bias = float(biases[row * groups + g]);
            for (uint word = 0; word < words_per_group; ++word) {
                const uint packed = weight[row * packed_cols + g * words_per_group + word];
                for (uint n = 0; n < values_per_word; ++n) {
                    const uint q = (packed >> (n * 4)) & 0xFu;
                    const uint input_index = g * group_size + word * values_per_word + n;
                    acc += input[input_index] * (((float)q * scale) + bias);
                }
            }
        }
        out[row] = acc;
    )METAL";
    static auto mlx_resident_projection_kernel = mlx::core::fast::metal_kernel(
        "gypsy_quantized_projection_mlx_resident_debug",
        {"input", "weight", "scales", "biases"},
        {"out"},
        mlx_resident_projection_source,
        "",
        true,
        false);
    Napi::Object mlx_resident_custom_metal = Napi::Object::New(env);
    try {
        std::vector<mlx::core::array> resident_outputs = mlx_resident_projection_kernel(
            {input_array, weight->array, scales->array, biases->array},
            {mlx::core::Shape{1, output_size}},
            {mlx::core::float32},
            {static_cast<size_t>(output_size), 1, 1},
            {1, 1, 1},
            {},
            std::nullopt,
            false,
            {});
        mlx::core::array resident_output = mlx::core::flatten(mlx::core::astype(resident_outputs[0], mlx::core::float32));
        mlx::core::eval(resident_output);
        const float* resident_values = resident_output.data<float>();
        Napi::Array resident_first_values = Napi::Array::New(env, 8);
        double resident_checksum = 0.0;
        double resident_abs_checksum = 0.0;
        double resident_max_abs_diff = 0.0;
        int resident_worst_index = -1;
        for (int i = 0; i < output_size; ++i) {
            const float rv = resident_values[i];
            const double diff = std::abs(static_cast<double>(rv) - static_cast<double>(values[i]));
            if (diff > resident_max_abs_diff) {
                resident_max_abs_diff = diff;
                resident_worst_index = i;
            }
            resident_checksum += static_cast<double>(rv);
            resident_abs_checksum += std::abs(static_cast<double>(rv));
            if (i < 8) {
                resident_first_values.Set(static_cast<uint32_t>(i), Napi::Number::New(env, rv));
            }
        }
        mlx_resident_custom_metal.Set("available", Napi::Boolean::New(env, true));
        mlx_resident_custom_metal.Set("ok", Napi::Boolean::New(env, resident_max_abs_diff < 1.0e-3));
        mlx_resident_custom_metal.Set("backend", Napi::String::New(env, "mlx_fast_custom_metal_quantized_projection"));
        mlx_resident_custom_metal.Set("checksum", Napi::Number::New(env, resident_checksum));
        mlx_resident_custom_metal.Set("abs_checksum", Napi::Number::New(env, resident_abs_checksum));
        mlx_resident_custom_metal.Set("max_abs_diff_vs_mlx", Napi::Number::New(env, resident_max_abs_diff));
        mlx_resident_custom_metal.Set("worst_index", Napi::Number::New(env, resident_worst_index));
        mlx_resident_custom_metal.Set("first_values", resident_first_values);
        mlx_resident_custom_metal.Set("error", env.Null());
    } catch (const std::exception& e) {
        mlx_resident_custom_metal.Set("available", Napi::Boolean::New(env, false));
        mlx_resident_custom_metal.Set("ok", Napi::Boolean::New(env, false));
        mlx_resident_custom_metal.Set("backend", Napi::String::New(env, "mlx_fast_custom_metal_quantized_projection"));
        mlx_resident_custom_metal.Set("error", Napi::String::New(env, e.what()));
    }

    Napi::Object weight_meta = Napi::Object::New(env);
    weight_meta.Set("weight_tensor", Napi::String::New(env, weight->name));
    weight_meta.Set("weight_dtype", Napi::String::New(env, weight->dtype));
    weight_meta.Set("weight_elements", Napi::Number::New(env, static_cast<double>(weight->array.size())));
    weight_meta.Set("scales_tensor", Napi::String::New(env, scales->name));
    weight_meta.Set("scales_dtype", Napi::String::New(env, scales->dtype));
    weight_meta.Set("scales_elements", Napi::Number::New(env, static_cast<double>(scales->array.size())));
    weight_meta.Set("biases_tensor", Napi::String::New(env, biases->name));
    weight_meta.Set("biases_dtype", Napi::String::New(env, biases->dtype));
    weight_meta.Set("biases_elements", Napi::Number::New(env, static_cast<double>(biases->array.size())));

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_quantized_projection_probe_version", Napi::String::New(env, "gypsy-selected-quantized-projection-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_inputs_are_deterministic", Napi::Boolean::New(env, true));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer0_q_proj_only"));
    out.Set("readback_count", Napi::Number::New(env, 1));
    Napi::Array reasons = Napi::Array::New(env, 1);
    reasons.Set(static_cast<uint32_t>(0), Napi::String::New(env, "layer0_q_proj_quantized_projection_summary"));
    out.Set("readback_reasons", reasons);
    out.Set("projection", projection);
    out.Set("compiled_metal_projection", compiled_metal);
    out.Set("compiled_metal_projection_tiled", tiled_compiled_metal);
    out.Set("mlx_resident_custom_metal_projection", mlx_resident_custom_metal);
    out.Set("resident_group", weight_meta);
    out.Set("next_execution_step", Napi::String::New(env, "run_selected_lora_projection_delta_probe"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedLoraProjectionDeltaProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected LoRA projection delta probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected LoRA projection delta probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }

    const ResidentArrayRecord* lora_a = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_proj.lora_a");
    const ResidentArrayRecord* lora_b = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_proj.lora_b");
    if (lora_a == nullptr || lora_b == nullptr) {
        Napi::Error::New(env, "Selected q_proj LoRA arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    constexpr int input_size = 2560;
    constexpr int rank = 8;
    constexpr int output_size = 4096;
    constexpr float scale = 20.0f;

    std::vector<float> input(static_cast<size_t>(input_size));
    for (int i = 0; i < input_size; ++i) {
        input[static_cast<size_t>(i)] = static_cast<float>((i % 29) - 14) / 29.0f;
    }

    auto start = std::chrono::steady_clock::now();
    mlx::core::array input_array(input.begin(), mlx::core::Shape{1, input_size}, mlx::core::float32);
    mlx::core::array intermediate = mlx::core::matmul(input_array, lora_a->array);
    mlx::core::array delta = mlx::core::astype(scale * mlx::core::matmul(intermediate, lora_b->array), mlx::core::float32);
    mlx::core::eval(delta);
    auto end = std::chrono::steady_clock::now();

    const float* values = delta.data<float>();
    double checksum = 0.0;
    double abs_checksum = 0.0;
    for (int i = 0; i < output_size; ++i) {
        checksum += static_cast<double>(values[i]);
        abs_checksum += std::abs(static_cast<double>(values[i]));
    }

    Napi::Array first_values = Napi::Array::New(env, 8);
    for (int i = 0; i < 8; ++i) {
        first_values.Set(static_cast<uint32_t>(i), Napi::Number::New(env, values[i]));
    }

    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    Napi::Object projection = Napi::Object::New(env);
    projection.Set("group", Napi::String::New(env, "model.layers.20.self_attn.q_proj"));
    projection.Set("input_len", Napi::Number::New(env, input_size));
    projection.Set("rank", Napi::Number::New(env, rank));
    projection.Set("output_len", Napi::Number::New(env, static_cast<double>(delta.size())));
    projection.Set("expected_output_len", Napi::Number::New(env, output_size));
    projection.Set("scale", Napi::Number::New(env, scale));
    projection.Set("formula", Napi::String::New(env, "scale * ((input @ lora_a) @ lora_b)"));
    projection.Set("checksum", Napi::Number::New(env, checksum));
    projection.Set("abs_checksum", Napi::Number::New(env, abs_checksum));
    projection.Set("first_values", first_values);
    projection.Set("timing_ms", Napi::Number::New(env, timing_ms));
    projection.Set("backend", Napi::String::New(env, "mlx"));
    projection.Set("readback_reason", Napi::String::New(env, "probe_output_summary"));

    Napi::Object tensors = Napi::Object::New(env);
    tensors.Set("lora_a_tensor", Napi::String::New(env, lora_a->name));
    tensors.Set("lora_a_dtype", Napi::String::New(env, lora_a->dtype));
    tensors.Set("lora_a_elements", Napi::Number::New(env, static_cast<double>(lora_a->array.size())));
    tensors.Set("lora_b_tensor", Napi::String::New(env, lora_b->name));
    tensors.Set("lora_b_dtype", Napi::String::New(env, lora_b->dtype));
    tensors.Set("lora_b_elements", Napi::Number::New(env, static_cast<double>(lora_b->array.size())));

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_lora_projection_delta_probe_version", Napi::String::New(env, "gypsy-selected-lora-projection-delta-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_inputs_are_deterministic", Napi::Boolean::New(env, true));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer20_q_proj_lora_only"));
    out.Set("adapter_layer", Napi::Number::New(env, 20));
    out.Set("adapter_target", Napi::String::New(env, "self_attn.q_proj"));
    out.Set("readback_count", Napi::Number::New(env, 1));
    Napi::Array reasons = Napi::Array::New(env, 1);
    reasons.Set(static_cast<uint32_t>(0), Napi::String::New(env, "layer20_q_proj_lora_delta_summary"));
    out.Set("readback_reasons", reasons);
    out.Set("projection_delta", projection);
    out.Set("resident_tensors", tensors);
    out.Set("next_execution_step", Napi::String::New(env, "run_selected_base_plus_lora_projection_probe"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedBasePlusLoraProjectionProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected base-plus-LoRA projection probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected base-plus-LoRA projection probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }

    const ResidentArrayRecord* weight = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_proj.weight");
    const ResidentArrayRecord* scales = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_proj.scales");
    const ResidentArrayRecord* biases = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_proj.biases");
    const ResidentArrayRecord* lora_a = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_proj.lora_a");
    const ResidentArrayRecord* lora_b = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_proj.lora_b");
    if (weight == nullptr || scales == nullptr || biases == nullptr || lora_a == nullptr || lora_b == nullptr) {
        Napi::Error::New(env, "Selected layer20 q_proj base-plus-LoRA arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    constexpr int input_size = 2560;
    constexpr int rank = 8;
    constexpr int output_size = 4096;
    constexpr int group_size = 64;
    constexpr int bits = 4;
    constexpr float scale = 20.0f;

    std::vector<float> input(static_cast<size_t>(input_size));
    for (int i = 0; i < input_size; ++i) {
        input[static_cast<size_t>(i)] = static_cast<float>((i % 31) - 15) / 31.0f;
    }

    auto start = std::chrono::steady_clock::now();
    mlx::core::array input_array(input.begin(), mlx::core::Shape{1, input_size}, mlx::core::float32);
    mlx::core::array base = mlx::core::astype(
        mlx::core::quantized_matmul(
            input_array,
            weight->array,
            scales->array,
            std::make_optional(biases->array),
            true,
            group_size,
            bits,
            "affine"),
        mlx::core::float32);
    mlx::core::array delta = mlx::core::astype(
        scale * mlx::core::matmul(mlx::core::matmul(input_array, lora_a->array), lora_b->array),
        mlx::core::float32);
    mlx::core::array combined = mlx::core::astype(base + delta, mlx::core::float32);
    mlx::core::eval(base, delta, combined);
    auto end = std::chrono::steady_clock::now();

    const float* base_values = base.data<float>();
    const float* delta_values = delta.data<float>();
    const float* combined_values = combined.data<float>();
    double base_checksum = 0.0;
    double delta_checksum = 0.0;
    double combined_checksum = 0.0;
    double delta_abs_checksum = 0.0;
    double max_abs_delta = 0.0;
    for (int i = 0; i < output_size; ++i) {
        const double b = static_cast<double>(base_values[i]);
        const double d = static_cast<double>(delta_values[i]);
        const double c = static_cast<double>(combined_values[i]);
        base_checksum += b;
        delta_checksum += d;
        combined_checksum += c;
        delta_abs_checksum += std::abs(d);
        max_abs_delta = std::max(max_abs_delta, std::abs(d));
    }

    Napi::Array first_base = Napi::Array::New(env, 8);
    Napi::Array first_delta = Napi::Array::New(env, 8);
    Napi::Array first_combined = Napi::Array::New(env, 8);
    for (int i = 0; i < 8; ++i) {
        first_base.Set(static_cast<uint32_t>(i), Napi::Number::New(env, base_values[i]));
        first_delta.Set(static_cast<uint32_t>(i), Napi::Number::New(env, delta_values[i]));
        first_combined.Set(static_cast<uint32_t>(i), Napi::Number::New(env, combined_values[i]));
    }

    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    Napi::Object projection = Napi::Object::New(env);
    projection.Set("group", Napi::String::New(env, "model.layers.20.self_attn.q_proj"));
    projection.Set("input_len", Napi::Number::New(env, input_size));
    projection.Set("rank", Napi::Number::New(env, rank));
    projection.Set("output_len", Napi::Number::New(env, static_cast<double>(combined.size())));
    projection.Set("expected_output_len", Napi::Number::New(env, output_size));
    projection.Set("group_size", Napi::Number::New(env, group_size));
    projection.Set("bits", Napi::Number::New(env, bits));
    projection.Set("mode", Napi::String::New(env, "affine"));
    projection.Set("scale", Napi::Number::New(env, scale));
    projection.Set("formula", Napi::String::New(env, "quantized_matmul(input, base) + scale * ((input @ lora_a) @ lora_b)"));
    projection.Set("base_checksum", Napi::Number::New(env, base_checksum));
    projection.Set("delta_checksum", Napi::Number::New(env, delta_checksum));
    projection.Set("delta_abs_checksum", Napi::Number::New(env, delta_abs_checksum));
    projection.Set("combined_checksum", Napi::Number::New(env, combined_checksum));
    projection.Set("max_abs_delta", Napi::Number::New(env, max_abs_delta));
    projection.Set("first_base_values", first_base);
    projection.Set("first_delta_values", first_delta);
    projection.Set("first_combined_values", first_combined);
    projection.Set("timing_ms", Napi::Number::New(env, timing_ms));
    projection.Set("backend", Napi::String::New(env, "mlx"));
    projection.Set("readback_reason", Napi::String::New(env, "probe_output_summary"));

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_base_plus_lora_projection_probe_version", Napi::String::New(env, "gypsy-selected-base-plus-lora-projection-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_inputs_are_deterministic", Napi::Boolean::New(env, true));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer20_q_proj_base_plus_lora"));
    out.Set("adapter_layer", Napi::Number::New(env, 20));
    out.Set("adapter_target", Napi::String::New(env, "self_attn.q_proj"));
    out.Set("adapter_delta_applied", Napi::Boolean::New(env, true));
    out.Set("readback_count", Napi::Number::New(env, 3));
    Napi::Array reasons = Napi::Array::New(env, 3);
    reasons.Set(static_cast<uint32_t>(0), Napi::String::New(env, "layer20_q_proj_base_summary"));
    reasons.Set(static_cast<uint32_t>(1), Napi::String::New(env, "layer20_q_proj_lora_delta_summary"));
    reasons.Set(static_cast<uint32_t>(2), Napi::String::New(env, "layer20_q_proj_combined_summary"));
    out.Set("readback_reasons", reasons);
    out.Set("projection", projection);
    out.Set("next_execution_step", Napi::String::New(env, "run_selected_q_norm_after_lora_probe"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedQNormAfterLoraProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected q_norm-after-LoRA probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected q_norm-after-LoRA probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }

    const ResidentArrayRecord* weight = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_proj.weight");
    const ResidentArrayRecord* scales = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_proj.scales");
    const ResidentArrayRecord* biases = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_proj.biases");
    const ResidentArrayRecord* lora_a = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_proj.lora_a");
    const ResidentArrayRecord* lora_b = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_proj.lora_b");
    const ResidentArrayRecord* q_norm = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_norm.weight");
    if (weight == nullptr || scales == nullptr || biases == nullptr || lora_a == nullptr || lora_b == nullptr || q_norm == nullptr) {
        Napi::Error::New(env, "Selected layer20 q_proj/q_norm arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    constexpr int input_size = 2560;
    constexpr int rank = 8;
    constexpr int output_size = 4096;
    constexpr int q_head_dim = 128;
    constexpr int group_size = 64;
    constexpr int bits = 4;
    constexpr float scale = 20.0f;
    constexpr float eps = 1.0e-6f;

    std::vector<float> input(static_cast<size_t>(input_size));
    for (int i = 0; i < input_size; ++i) {
        input[static_cast<size_t>(i)] = static_cast<float>((i % 37) - 18) / 37.0f;
    }

    auto start = std::chrono::steady_clock::now();
    mlx::core::array input_array(input.begin(), mlx::core::Shape{1, input_size}, mlx::core::float32);
    mlx::core::array base = mlx::core::astype(
        mlx::core::quantized_matmul(
            input_array,
            weight->array,
            scales->array,
            std::make_optional(biases->array),
            true,
            group_size,
            bits,
            "affine"),
        mlx::core::float32);
    mlx::core::array delta = mlx::core::astype(
        scale * mlx::core::matmul(mlx::core::matmul(input_array, lora_a->array), lora_b->array),
        mlx::core::float32);
    mlx::core::array combined = mlx::core::astype(base + delta, mlx::core::float32);
    mlx::core::array combined_heads = mlx::core::reshape(combined, mlx::core::Shape{32, q_head_dim});
    mlx::core::array normalized = mlx::core::astype(
        mlx::core::fast::rms_norm(combined_heads, std::make_optional(q_norm->array), eps),
        mlx::core::float32);
    mlx::core::array normalized_flat = mlx::core::flatten(normalized);
    mlx::core::eval(combined, normalized_flat);
    auto end = std::chrono::steady_clock::now();

    const float* combined_values = combined.data<float>();
    const float* normalized_values = normalized_flat.data<float>();
    double combined_checksum = 0.0;
    double normalized_checksum = 0.0;
    double max_abs_norm_change = 0.0;
    for (int i = 0; i < output_size; ++i) {
        const double c = static_cast<double>(combined_values[i]);
        const double n = static_cast<double>(normalized_values[i]);
        combined_checksum += c;
        normalized_checksum += n;
        max_abs_norm_change = std::max(max_abs_norm_change, std::abs(n - c));
    }

    Napi::Array first_combined = Napi::Array::New(env, 8);
    Napi::Array first_normalized = Napi::Array::New(env, 8);
    for (int i = 0; i < 8; ++i) {
        first_combined.Set(static_cast<uint32_t>(i), Napi::Number::New(env, combined_values[i]));
        first_normalized.Set(static_cast<uint32_t>(i), Napi::Number::New(env, normalized_values[i]));
    }

    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    Napi::Object projection = Napi::Object::New(env);
    projection.Set("group", Napi::String::New(env, "model.layers.20.self_attn.q_proj"));
    projection.Set("input_len", Napi::Number::New(env, input_size));
    projection.Set("rank", Napi::Number::New(env, rank));
    projection.Set("output_len", Napi::Number::New(env, output_size));
    projection.Set("q_heads", Napi::Number::New(env, 32));
    projection.Set("q_head_dim", Napi::Number::New(env, q_head_dim));
    projection.Set("group_size", Napi::Number::New(env, group_size));
    projection.Set("bits", Napi::Number::New(env, bits));
    projection.Set("scale", Napi::Number::New(env, scale));
    projection.Set("eps", Napi::Number::New(env, eps));
    projection.Set("ordering", Napi::String::New(env, "base_plus_lora_then_q_norm"));
    projection.Set("combined_checksum", Napi::Number::New(env, combined_checksum));
    projection.Set("normalized_checksum", Napi::Number::New(env, normalized_checksum));
    projection.Set("max_abs_norm_change", Napi::Number::New(env, max_abs_norm_change));
    projection.Set("first_combined_values", first_combined);
    projection.Set("first_normalized_values", first_normalized);
    projection.Set("timing_ms", Napi::Number::New(env, timing_ms));
    projection.Set("backend", Napi::String::New(env, "mlx"));
    projection.Set("readback_reason", Napi::String::New(env, "probe_output_summary"));

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_q_norm_after_lora_probe_version", Napi::String::New(env, "gypsy-selected-q-norm-after-lora-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_inputs_are_deterministic", Napi::Boolean::New(env, true));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer20_q_proj_base_plus_lora_then_q_norm"));
    out.Set("adapter_layer", Napi::Number::New(env, 20));
    out.Set("adapter_target", Napi::String::New(env, "self_attn.q_proj"));
    out.Set("adapter_delta_applied_before_q_norm", Napi::Boolean::New(env, true));
    out.Set("q_norm_applied", Napi::Boolean::New(env, true));
    out.Set("readback_count", Napi::Number::New(env, 2));
    Napi::Array reasons = Napi::Array::New(env, 2);
    reasons.Set(static_cast<uint32_t>(0), Napi::String::New(env, "layer20_q_proj_combined_before_q_norm_summary"));
    reasons.Set(static_cast<uint32_t>(1), Napi::String::New(env, "layer20_q_proj_after_q_norm_summary"));
    out.Set("readback_reasons", reasons);
    out.Set("projection", projection);
    out.Set("next_execution_step", Napi::String::New(env, "run_selected_rope_after_q_norm_probe"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedRopeAfterQNormProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected RoPE-after-q_norm probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected RoPE-after-q_norm probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }

    const ResidentArrayRecord* weight = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_proj.weight");
    const ResidentArrayRecord* scales = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_proj.scales");
    const ResidentArrayRecord* biases = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_proj.biases");
    const ResidentArrayRecord* lora_a = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_proj.lora_a");
    const ResidentArrayRecord* lora_b = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_proj.lora_b");
    const ResidentArrayRecord* q_norm = FindResidentArray(session.selected_resident_arrays, "model.layers.20.self_attn.q_norm.weight");
    if (weight == nullptr || scales == nullptr || biases == nullptr || lora_a == nullptr || lora_b == nullptr || q_norm == nullptr) {
        Napi::Error::New(env, "Selected layer20 q_proj/q_norm arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    constexpr int input_size = 2560;
    constexpr int rank = 8;
    constexpr int output_size = 4096;
    constexpr int q_heads = 32;
    constexpr int q_head_dim = 128;
    constexpr int group_size = 64;
    constexpr int bits = 4;
    constexpr int rope_position = 7;
    constexpr float adapter_scale = 20.0f;
    constexpr float eps = 1.0e-6f;
    constexpr float rope_theta = 5000000.0f;
    constexpr float rope_scale = 1.0f;
    constexpr bool rope_traditional = false;

    std::vector<float> input(static_cast<size_t>(input_size));
    for (int i = 0; i < input_size; ++i) {
        input[static_cast<size_t>(i)] = static_cast<float>((i % 41) - 20) / 41.0f;
    }

    auto start = std::chrono::steady_clock::now();
    mlx::core::array input_array(input.begin(), mlx::core::Shape{1, input_size}, mlx::core::float32);
    mlx::core::array base = mlx::core::astype(
        mlx::core::quantized_matmul(
            input_array,
            weight->array,
            scales->array,
            std::make_optional(biases->array),
            true,
            group_size,
            bits,
            "affine"),
        mlx::core::float32);
    mlx::core::array delta = mlx::core::astype(
        adapter_scale * mlx::core::matmul(mlx::core::matmul(input_array, lora_a->array), lora_b->array),
        mlx::core::float32);
    mlx::core::array combined = mlx::core::astype(base + delta, mlx::core::float32);
    mlx::core::array combined_heads = mlx::core::reshape(combined, mlx::core::Shape{q_heads, q_head_dim});
    mlx::core::array normalized = mlx::core::astype(
        mlx::core::fast::rms_norm(combined_heads, std::make_optional(q_norm->array), eps),
        mlx::core::float32);
    mlx::core::array normalized_for_rope = mlx::core::reshape(normalized, mlx::core::Shape{1, q_heads, 1, q_head_dim});
    mlx::core::array rope = mlx::core::astype(
        mlx::core::fast::rope(
            normalized_for_rope,
            q_head_dim,
            rope_traditional,
            std::make_optional(rope_theta),
            rope_scale,
            rope_position),
        mlx::core::float32);
    mlx::core::array normalized_flat = mlx::core::flatten(normalized);
    mlx::core::array rope_flat = mlx::core::flatten(rope);
    mlx::core::eval(normalized_flat, rope_flat);
    auto end = std::chrono::steady_clock::now();

    const float* normalized_values = normalized_flat.data<float>();
    const float* rope_values = rope_flat.data<float>();
    double normalized_checksum = 0.0;
    double rope_checksum = 0.0;
    double max_abs_rope_change = 0.0;
    for (int i = 0; i < output_size; ++i) {
        const double n = static_cast<double>(normalized_values[i]);
        const double r = static_cast<double>(rope_values[i]);
        normalized_checksum += n;
        rope_checksum += r;
        max_abs_rope_change = std::max(max_abs_rope_change, std::abs(r - n));
    }

    Napi::Array first_normalized = Napi::Array::New(env, 8);
    Napi::Array first_rope = Napi::Array::New(env, 8);
    for (int i = 0; i < 8; ++i) {
        first_normalized.Set(static_cast<uint32_t>(i), Napi::Number::New(env, normalized_values[i]));
        first_rope.Set(static_cast<uint32_t>(i), Napi::Number::New(env, rope_values[i]));
    }

    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    Napi::Object projection = Napi::Object::New(env);
    projection.Set("group", Napi::String::New(env, "model.layers.20.self_attn.q_proj"));
    projection.Set("input_len", Napi::Number::New(env, input_size));
    projection.Set("rank", Napi::Number::New(env, rank));
    projection.Set("output_len", Napi::Number::New(env, output_size));
    projection.Set("q_heads", Napi::Number::New(env, q_heads));
    projection.Set("q_head_dim", Napi::Number::New(env, q_head_dim));
    projection.Set("group_size", Napi::Number::New(env, group_size));
    projection.Set("bits", Napi::Number::New(env, bits));
    projection.Set("adapter_scale", Napi::Number::New(env, adapter_scale));
    projection.Set("eps", Napi::Number::New(env, eps));
    projection.Set("rope_position", Napi::Number::New(env, rope_position));
    projection.Set("rope_theta", Napi::Number::New(env, rope_theta));
    projection.Set("rope_scale", Napi::Number::New(env, rope_scale));
    projection.Set("rope_traditional", Napi::Boolean::New(env, rope_traditional));
    projection.Set("ordering", Napi::String::New(env, "base_plus_lora_then_q_norm_then_rope"));
    projection.Set("normalized_checksum", Napi::Number::New(env, normalized_checksum));
    projection.Set("rope_checksum", Napi::Number::New(env, rope_checksum));
    projection.Set("max_abs_rope_change", Napi::Number::New(env, max_abs_rope_change));
    projection.Set("first_normalized_values", first_normalized);
    projection.Set("first_rope_values", first_rope);
    projection.Set("timing_ms", Napi::Number::New(env, timing_ms));
    projection.Set("backend", Napi::String::New(env, "mlx"));
    projection.Set("readback_reason", Napi::String::New(env, "probe_output_summary"));

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_rope_after_q_norm_probe_version", Napi::String::New(env, "gypsy-selected-rope-after-q-norm-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_inputs_are_deterministic", Napi::Boolean::New(env, true));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer20_q_proj_base_plus_lora_then_q_norm_then_rope"));
    out.Set("adapter_layer", Napi::Number::New(env, 20));
    out.Set("adapter_target", Napi::String::New(env, "self_attn.q_proj"));
    out.Set("adapter_delta_applied_before_q_norm", Napi::Boolean::New(env, true));
    out.Set("q_norm_applied_before_rope", Napi::Boolean::New(env, true));
    out.Set("rope_applied", Napi::Boolean::New(env, true));
    out.Set("readback_count", Napi::Number::New(env, 2));
    Napi::Array reasons = Napi::Array::New(env, 2);
    reasons.Set(static_cast<uint32_t>(0), Napi::String::New(env, "layer20_q_proj_after_q_norm_summary"));
    reasons.Set(static_cast<uint32_t>(1), Napi::String::New(env, "layer20_q_proj_after_rope_summary"));
    out.Set("readback_reasons", reasons);
    out.Set("projection", projection);
    out.Set("next_execution_step", Napi::String::New(env, "add_selected_kv_projection_path"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedKvProjectionPathProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected K/V projection path probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected K/V projection path probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto require_array = [&](const std::string& name) -> const ResidentArrayRecord* {
        return FindResidentArray(session.selected_resident_arrays, name);
    };
    const ResidentArrayRecord* qw = require_array("model.layers.20.self_attn.q_proj.weight");
    const ResidentArrayRecord* qs = require_array("model.layers.20.self_attn.q_proj.scales");
    const ResidentArrayRecord* qb = require_array("model.layers.20.self_attn.q_proj.biases");
    const ResidentArrayRecord* kw = require_array("model.layers.20.self_attn.k_proj.weight");
    const ResidentArrayRecord* ks = require_array("model.layers.20.self_attn.k_proj.scales");
    const ResidentArrayRecord* kb = require_array("model.layers.20.self_attn.k_proj.biases");
    const ResidentArrayRecord* vw = require_array("model.layers.20.self_attn.v_proj.weight");
    const ResidentArrayRecord* vs = require_array("model.layers.20.self_attn.v_proj.scales");
    const ResidentArrayRecord* vb = require_array("model.layers.20.self_attn.v_proj.biases");
    const ResidentArrayRecord* qa = require_array("model.layers.20.self_attn.q_proj.lora_a");
    const ResidentArrayRecord* qlb = require_array("model.layers.20.self_attn.q_proj.lora_b");
    const ResidentArrayRecord* ka = require_array("model.layers.20.self_attn.k_proj.lora_a");
    const ResidentArrayRecord* klb = require_array("model.layers.20.self_attn.k_proj.lora_b");
    const ResidentArrayRecord* va = require_array("model.layers.20.self_attn.v_proj.lora_a");
    const ResidentArrayRecord* vlb = require_array("model.layers.20.self_attn.v_proj.lora_b");
    const ResidentArrayRecord* q_norm = require_array("model.layers.20.self_attn.q_norm.weight");
    const ResidentArrayRecord* k_norm = require_array("model.layers.20.self_attn.k_norm.weight");
    if (qw == nullptr || qs == nullptr || qb == nullptr || kw == nullptr || ks == nullptr || kb == nullptr ||
        vw == nullptr || vs == nullptr || vb == nullptr || qa == nullptr || qlb == nullptr || ka == nullptr ||
        klb == nullptr || va == nullptr || vlb == nullptr || q_norm == nullptr || k_norm == nullptr) {
        Napi::Error::New(env, "Selected layer20 q/k/v projection arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    constexpr int input_size = 2560;
    constexpr int q_output_size = 4096;
    constexpr int kv_output_size = 1024;
    constexpr int q_heads = 32;
    constexpr int kv_heads = 8;
    constexpr int head_dim = 128;
    constexpr int group_size = 64;
    constexpr int bits = 4;
    constexpr int rope_position = 7;
    constexpr float adapter_scale = 20.0f;
    constexpr float eps = 1.0e-6f;
    constexpr float rope_theta = 5000000.0f;
    constexpr float rope_scale = 1.0f;
    constexpr bool rope_traditional = false;

    std::vector<float> input(static_cast<size_t>(input_size));
    for (int i = 0; i < input_size; ++i) {
        input[static_cast<size_t>(i)] = static_cast<float>((i % 43) - 21) / 43.0f;
    }

    auto start = std::chrono::steady_clock::now();
    mlx::core::array input_array(input.begin(), mlx::core::Shape{1, input_size}, mlx::core::float32);

    auto project_base = [&](const ResidentArrayRecord* w, const ResidentArrayRecord* s, const ResidentArrayRecord* b) {
        return mlx::core::astype(
            mlx::core::quantized_matmul(
                input_array,
                w->array,
                s->array,
                std::make_optional(b->array),
                true,
                group_size,
                bits,
                "affine"),
            mlx::core::float32);
    };
    auto project_delta = [&](const ResidentArrayRecord* la, const ResidentArrayRecord* lb) {
        mlx::core::array delta = mlx::core::astype(
            adapter_scale * mlx::core::matmul(mlx::core::matmul(input_array, la->array), lb->array),
            mlx::core::float32);
        return delta;
    };

    mlx::core::array q_base = project_base(qw, qs, qb);
    mlx::core::array k_base = project_base(kw, ks, kb);
    mlx::core::array v_base = project_base(vw, vs, vb);
    mlx::core::array q_combined = mlx::core::astype(q_base + project_delta(qa, qlb), mlx::core::float32);
    mlx::core::array k_combined = mlx::core::astype(k_base + project_delta(ka, klb), mlx::core::float32);
    mlx::core::array v_combined = mlx::core::astype(v_base + project_delta(va, vlb), mlx::core::float32);

    mlx::core::array q_normed = mlx::core::astype(
        mlx::core::fast::rms_norm(mlx::core::reshape(q_combined, mlx::core::Shape{q_heads, head_dim}), std::make_optional(q_norm->array), eps),
        mlx::core::float32);
    mlx::core::array k_normed = mlx::core::astype(
        mlx::core::fast::rms_norm(mlx::core::reshape(k_combined, mlx::core::Shape{kv_heads, head_dim}), std::make_optional(k_norm->array), eps),
        mlx::core::float32);
    mlx::core::array q_rope = mlx::core::flatten(mlx::core::astype(
        mlx::core::fast::rope(
            mlx::core::reshape(q_normed, mlx::core::Shape{1, q_heads, 1, head_dim}),
            head_dim,
            rope_traditional,
            std::make_optional(rope_theta),
            rope_scale,
            rope_position),
        mlx::core::float32));
    mlx::core::array k_rope = mlx::core::flatten(mlx::core::astype(
        mlx::core::fast::rope(
            mlx::core::reshape(k_normed, mlx::core::Shape{1, kv_heads, 1, head_dim}),
            head_dim,
            rope_traditional,
            std::make_optional(rope_theta),
            rope_scale,
            rope_position),
        mlx::core::float32));
    mlx::core::array v_flat = mlx::core::flatten(mlx::core::astype(v_combined, mlx::core::float32));
    mlx::core::eval(q_base, k_base, v_base, q_rope, k_rope, v_flat);
    auto end = std::chrono::steady_clock::now();

    auto run_compiled_base_projection = [&](const std::string& name,
                                            const ResidentArrayRecord* w,
                                            const ResidentArrayRecord* s,
                                            const ResidentArrayRecord* b,
                                            const mlx::core::array& expected,
                                            int output_len) {
        const float* expected_values = expected.data<float>();
        std::vector<float> metal_values(static_cast<size_t>(output_len), 0.0f);
        GypsyDirectMetalProbeResult metal = GypsyRunDirectMetalQuantizedProjectionTiledCompare(
            input.data(),
            static_cast<const std::uint32_t*>(w->raw_data),
            static_cast<const std::uint16_t*>(s->raw_data),
            static_cast<const std::uint16_t*>(b->raw_data),
            expected_values,
            metal_values.data(),
            static_cast<std::uint32_t>(output_len),
            static_cast<std::uint32_t>(input_size / 8),
            static_cast<std::uint32_t>(input_size));
        Napi::Array first = Napi::Array::New(env, 8);
        double abs_checksum = 0.0;
        for (int i = 0; i < output_len; ++i) {
            abs_checksum += std::abs(static_cast<double>(metal_values[static_cast<size_t>(i)]));
            if (i < 8) {
                first.Set(static_cast<uint32_t>(i), Napi::Number::New(env, metal_values[static_cast<size_t>(i)]));
            }
        }
        Napi::Object o = Napi::Object::New(env);
        o.Set("name", Napi::String::New(env, name));
        o.Set("backend", Napi::String::New(env, "direct_compiled_metal_quantized_projection_tiled"));
        o.Set("ok", Napi::Boolean::New(env, metal.ok));
        o.Set("output_len", Napi::Number::New(env, output_len));
        o.Set("max_abs_diff_vs_mlx_base", Napi::Number::New(env, metal.max_abs_diff));
        o.Set("checksum", Napi::Number::New(env, metal.checksum));
        o.Set("abs_checksum", Napi::Number::New(env, abs_checksum));
        o.Set("timing_ms", Napi::Number::New(env, metal.elapsed_ms));
        o.Set("threads_per_output_row", Napi::Number::New(env, 64));
        o.Set("compile_time_shape", Napi::String::New(env, "rows/input_len/packed_cols baked into Metal source"));
        o.Set("first_values", first);
        if (metal.error != nullptr) {
            o.Set("error", Napi::String::New(env, metal.error));
        } else {
            o.Set("error", env.Null());
        }
        return o;
    };

    auto summarize = [&](const std::string& name, const mlx::core::array& arr, int expected_len) {
        const float* values = arr.data<float>();
        double checksum = 0.0;
        double abs_checksum = 0.0;
        for (int i = 0; i < expected_len; ++i) {
            const double v = static_cast<double>(values[i]);
            checksum += v;
            abs_checksum += std::abs(v);
        }
        Napi::Array first = Napi::Array::New(env, 8);
        for (int i = 0; i < 8; ++i) {
            first.Set(static_cast<uint32_t>(i), Napi::Number::New(env, values[i]));
        }
        Napi::Object o = Napi::Object::New(env);
        o.Set("name", Napi::String::New(env, name));
        o.Set("output_len", Napi::Number::New(env, static_cast<double>(arr.size())));
        o.Set("expected_output_len", Napi::Number::New(env, expected_len));
        o.Set("checksum", Napi::Number::New(env, checksum));
        o.Set("abs_checksum", Napi::Number::New(env, abs_checksum));
        o.Set("first_values", first);
        return o;
    };

    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_kv_projection_path_probe_version", Napi::String::New(env, "gypsy-selected-kv-projection-path-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_inputs_are_deterministic", Napi::Boolean::New(env, true));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer20_qkv_projection_path"));
    out.Set("adapter_layer", Napi::Number::New(env, 20));
    out.Set("adapter_targets", StringVectorToNapiArray(env, {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}));
    out.Set("adapter_delta_applied_before_qk_norm", Napi::Boolean::New(env, true));
    out.Set("q_norm_applied_before_rope", Napi::Boolean::New(env, true));
    out.Set("k_norm_applied_before_rope", Napi::Boolean::New(env, true));
    out.Set("rope_applied_to_q", Napi::Boolean::New(env, true));
    out.Set("rope_applied_to_k", Napi::Boolean::New(env, true));
    out.Set("v_projection_has_no_norm_or_rope", Napi::Boolean::New(env, true));
    out.Set("input_len", Napi::Number::New(env, input_size));
    out.Set("q_heads", Napi::Number::New(env, q_heads));
    out.Set("kv_heads", Napi::Number::New(env, kv_heads));
    out.Set("head_dim", Napi::Number::New(env, head_dim));
    out.Set("rope_position", Napi::Number::New(env, rope_position));
    out.Set("rope_theta", Napi::Number::New(env, rope_theta));
    out.Set("timing_ms", Napi::Number::New(env, timing_ms));
    Napi::Object compiled_base_qkv = Napi::Object::New(env);
    compiled_base_qkv.Set("q_proj", run_compiled_base_projection("q_proj_base", qw, qs, qb, q_base, q_output_size));
    compiled_base_qkv.Set("k_proj", run_compiled_base_projection("k_proj_base", kw, ks, kb, k_base, kv_output_size));
    compiled_base_qkv.Set("v_proj", run_compiled_base_projection("v_proj_base", vw, vs, vb, v_base, kv_output_size));
    compiled_base_qkv.Set("adapter_delta_included", Napi::Boolean::New(env, false));
    compiled_base_qkv.Set("reason", Napi::String::New(env, "compiled Metal base projection validated before LoRA fusion"));
    out.Set("compiled_metal_base_qkv", compiled_base_qkv);
    out.Set("q_after_rope", summarize("q_after_rope", q_rope, q_output_size));
    out.Set("k_after_rope", summarize("k_after_rope", k_rope, kv_output_size));
    out.Set("v_projection", summarize("v_projection", v_flat, kv_output_size));
    out.Set("readback_count", Napi::Number::New(env, 3));
    Napi::Array reasons = Napi::Array::New(env, 3);
    reasons.Set(static_cast<uint32_t>(0), Napi::String::New(env, "layer20_q_after_rope_summary"));
    reasons.Set(static_cast<uint32_t>(1), Napi::String::New(env, "layer20_k_after_rope_summary"));
    reasons.Set(static_cast<uint32_t>(2), Napi::String::New(env, "layer20_v_projection_summary"));
    out.Set("readback_reasons", reasons);
    out.Set("next_execution_step", Napi::String::New(env, "run_selected_single_token_attention_probe"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedSingleTokenAttentionProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected single-token attention probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected single-token attention probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto require_array = [&](const std::string& name) -> const ResidentArrayRecord* {
        return FindResidentArray(session.selected_resident_arrays, name);
    };
    const ResidentArrayRecord* qw = require_array("model.layers.20.self_attn.q_proj.weight");
    const ResidentArrayRecord* qs = require_array("model.layers.20.self_attn.q_proj.scales");
    const ResidentArrayRecord* qb = require_array("model.layers.20.self_attn.q_proj.biases");
    const ResidentArrayRecord* kw = require_array("model.layers.20.self_attn.k_proj.weight");
    const ResidentArrayRecord* ks = require_array("model.layers.20.self_attn.k_proj.scales");
    const ResidentArrayRecord* kb = require_array("model.layers.20.self_attn.k_proj.biases");
    const ResidentArrayRecord* vw = require_array("model.layers.20.self_attn.v_proj.weight");
    const ResidentArrayRecord* vs = require_array("model.layers.20.self_attn.v_proj.scales");
    const ResidentArrayRecord* vb = require_array("model.layers.20.self_attn.v_proj.biases");
    const ResidentArrayRecord* qa = require_array("model.layers.20.self_attn.q_proj.lora_a");
    const ResidentArrayRecord* qlb = require_array("model.layers.20.self_attn.q_proj.lora_b");
    const ResidentArrayRecord* ka = require_array("model.layers.20.self_attn.k_proj.lora_a");
    const ResidentArrayRecord* klb = require_array("model.layers.20.self_attn.k_proj.lora_b");
    const ResidentArrayRecord* va = require_array("model.layers.20.self_attn.v_proj.lora_a");
    const ResidentArrayRecord* vlb = require_array("model.layers.20.self_attn.v_proj.lora_b");
    const ResidentArrayRecord* q_norm = require_array("model.layers.20.self_attn.q_norm.weight");
    const ResidentArrayRecord* k_norm = require_array("model.layers.20.self_attn.k_norm.weight");
    if (qw == nullptr || qs == nullptr || qb == nullptr || kw == nullptr || ks == nullptr || kb == nullptr ||
        vw == nullptr || vs == nullptr || vb == nullptr || qa == nullptr || qlb == nullptr || ka == nullptr ||
        klb == nullptr || va == nullptr || vlb == nullptr || q_norm == nullptr || k_norm == nullptr) {
        Napi::Error::New(env, "Selected layer20 attention arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    constexpr int input_size = 2560;
    constexpr int q_output_size = 4096;
    constexpr int kv_output_size = 1024;
    constexpr int q_heads = 32;
    constexpr int kv_heads = 8;
    constexpr int q_heads_per_kv = 4;
    constexpr int head_dim = 128;
    constexpr int group_size = 64;
    constexpr int bits = 4;
    constexpr int rope_position = 7;
    constexpr float adapter_scale = 20.0f;
    constexpr float eps = 1.0e-6f;
    constexpr float rope_theta = 5000000.0f;
    constexpr float rope_scale = 1.0f;
    constexpr bool rope_traditional = false;

    std::vector<float> input(static_cast<size_t>(input_size));
    for (int i = 0; i < input_size; ++i) {
        input[static_cast<size_t>(i)] = static_cast<float>((i % 47) - 23) / 47.0f;
    }

    auto start = std::chrono::steady_clock::now();
    mlx::core::array input_array(input.begin(), mlx::core::Shape{1, input_size}, mlx::core::float32);
    auto project = [&](const ResidentArrayRecord* w, const ResidentArrayRecord* s, const ResidentArrayRecord* b,
                       const ResidentArrayRecord* la, const ResidentArrayRecord* lb) {
        mlx::core::array base = mlx::core::astype(
            mlx::core::quantized_matmul(
                input_array,
                w->array,
                s->array,
                std::make_optional(b->array),
                true,
                group_size,
                bits,
                "affine"),
            mlx::core::float32);
        mlx::core::array delta = mlx::core::astype(
            adapter_scale * mlx::core::matmul(mlx::core::matmul(input_array, la->array), lb->array),
            mlx::core::float32);
        return mlx::core::astype(base + delta, mlx::core::float32);
    };

    mlx::core::array q_combined = project(qw, qs, qb, qa, qlb);
    mlx::core::array k_combined = project(kw, ks, kb, ka, klb);
    mlx::core::array v_combined = project(vw, vs, vb, va, vlb);
    mlx::core::array q_normed = mlx::core::astype(
        mlx::core::fast::rms_norm(mlx::core::reshape(q_combined, mlx::core::Shape{q_heads, head_dim}), std::make_optional(q_norm->array), eps),
        mlx::core::float32);
    mlx::core::array k_normed = mlx::core::astype(
        mlx::core::fast::rms_norm(mlx::core::reshape(k_combined, mlx::core::Shape{kv_heads, head_dim}), std::make_optional(k_norm->array), eps),
        mlx::core::float32);
    mlx::core::array q_rope = mlx::core::astype(
        mlx::core::fast::rope(
            mlx::core::reshape(q_normed, mlx::core::Shape{1, q_heads, 1, head_dim}),
            head_dim,
            rope_traditional,
            std::make_optional(rope_theta),
            rope_scale,
            rope_position),
        mlx::core::float32);
    mlx::core::array k_rope = mlx::core::astype(
        mlx::core::fast::rope(
            mlx::core::reshape(k_normed, mlx::core::Shape{1, kv_heads, 1, head_dim}),
            head_dim,
            rope_traditional,
            std::make_optional(rope_theta),
            rope_scale,
            rope_position),
        mlx::core::float32);
    mlx::core::array v_heads = mlx::core::reshape(v_combined, mlx::core::Shape{1, kv_heads, 1, head_dim});
    mlx::core::array k_expanded = mlx::core::repeat(k_rope, q_heads_per_kv, 1);
    mlx::core::array v_expanded = mlx::core::repeat(v_heads, q_heads_per_kv, 1);
    mlx::core::array attention = mlx::core::astype(
        mlx::core::fast::scaled_dot_product_attention(
            q_rope,
            k_expanded,
            v_expanded,
            1.0f / std::sqrt(static_cast<float>(head_dim))),
        mlx::core::float32);
    mlx::core::array attention_flat = mlx::core::flatten(attention);
    mlx::core::array q_direct = mlx::core::flatten(mlx::core::astype(q_rope, mlx::core::float32));
    mlx::core::array k_direct = mlx::core::flatten(mlx::core::astype(k_rope, mlx::core::float32));
    mlx::core::array v_direct = mlx::core::flatten(mlx::core::astype(v_combined, mlx::core::float32));
    mlx::core::eval(attention_flat, q_direct, k_direct, v_direct);
    auto end = std::chrono::steady_clock::now();

    const float* values = attention_flat.data<float>();
    const float* q_values = q_direct.data<float>();
    const float* k_values = k_direct.data<float>();
    const float* v_values = v_direct.data<float>();
    GypsyDirectMetalProbeResult direct_attention = GypsyRunDirectMetalAttentionCompare(
        q_values,
        k_values,
        v_values,
        values,
        1);
    double checksum = 0.0;
    double abs_checksum = 0.0;
    for (int i = 0; i < q_output_size; ++i) {
        const double v = static_cast<double>(values[i]);
        checksum += v;
        abs_checksum += std::abs(v);
    }
    Napi::Array first = Napi::Array::New(env, 8);
    for (int i = 0; i < 8; ++i) {
        first.Set(static_cast<uint32_t>(i), Napi::Number::New(env, values[i]));
    }

    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    Napi::Object output = Napi::Object::New(env);
    output.Set("output_len", Napi::Number::New(env, static_cast<double>(attention_flat.size())));
    output.Set("expected_output_len", Napi::Number::New(env, q_output_size));
    output.Set("checksum", Napi::Number::New(env, checksum));
    output.Set("abs_checksum", Napi::Number::New(env, abs_checksum));
    output.Set("first_values", first);
    output.Set("backend", Napi::String::New(env, "mlx"));

    Napi::Object direct = Napi::Object::New(env);
    direct.Set("ok", Napi::Boolean::New(env, direct_attention.ok));
    direct.Set("runtime", Napi::String::New(env, "direct_metal_objcxx"));
    direct.Set("kernel", Napi::String::New(env, "gypsy_attention_probe"));
    direct.Set("seq_len", Napi::Number::New(env, 1));
    direct.Set("checksum", Napi::Number::New(env, direct_attention.checksum));
    direct.Set("max_abs_diff_vs_mlx", Napi::Number::New(env, direct_attention.max_abs_diff));
    direct.Set("elapsed_ms", Napi::Number::New(env, direct_attention.elapsed_ms));
    direct.Set("device_available", Napi::Boolean::New(env, direct_attention.device_available));
    direct.Set("library_compiled", Napi::Boolean::New(env, direct_attention.library_compiled));
    direct.Set("pipeline_created", Napi::Boolean::New(env, direct_attention.pipeline_created));
    direct.Set("command_completed", Napi::Boolean::New(env, direct_attention.command_completed));
    direct.Set("error", direct_attention.error == nullptr ? env.Null().As<Napi::Value>() : Napi::String::New(env, direct_attention.error).As<Napi::Value>());

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_single_token_attention_probe_version", Napi::String::New(env, "gypsy-selected-single-token-attention-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_inputs_are_deterministic", Napi::Boolean::New(env, true));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer20_single_token_attention"));
    out.Set("adapter_layer", Napi::Number::New(env, 20));
    out.Set("q_heads", Napi::Number::New(env, q_heads));
    out.Set("kv_heads", Napi::Number::New(env, kv_heads));
    out.Set("q_heads_per_kv", Napi::Number::New(env, q_heads_per_kv));
    out.Set("head_dim", Napi::Number::New(env, head_dim));
    out.Set("attention_seq_len", Napi::Number::New(env, 1));
    out.Set("rope_position", Napi::Number::New(env, rope_position));
    out.Set("q_path", Napi::String::New(env, "base_plus_lora_then_q_norm_then_rope"));
    out.Set("k_path", Napi::String::New(env, "base_plus_lora_then_k_norm_then_rope"));
    out.Set("v_path", Napi::String::New(env, "base_plus_lora"));
    out.Set("gqa_kv_expanded_to_q_heads", Napi::Boolean::New(env, true));
    out.Set("attention_backend", Napi::String::New(env, "mlx_scaled_dot_product_attention"));
    out.Set("direct_metal_attention_backend", Napi::String::New(env, "direct_metal_objcxx_mtlcompute"));
    out.Set("timing_ms", Napi::Number::New(env, timing_ms));
    out.Set("attention_output", output);
    out.Set("direct_metal_attention", direct);
    out.Set("readback_count", Napi::Number::New(env, 4));
    Napi::Array reasons = Napi::Array::New(env, 4);
    reasons.Set(static_cast<uint32_t>(0), Napi::String::New(env, "layer20_single_token_attention_output_summary"));
    reasons.Set(static_cast<uint32_t>(1), Napi::String::New(env, "layer20_direct_metal_q_input"));
    reasons.Set(static_cast<uint32_t>(2), Napi::String::New(env, "layer20_direct_metal_k_input"));
    reasons.Set(static_cast<uint32_t>(3), Napi::String::New(env, "layer20_direct_metal_v_input"));
    out.Set("readback_reasons", reasons);
    out.Set("next_execution_step", Napi::String::New(env, "add_selected_o_projection_path"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedOProjectionPathProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected o_proj path probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected o_proj path probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto require_array = [&](const std::string& name) -> const ResidentArrayRecord* {
        return FindResidentArray(session.selected_resident_arrays, name);
    };
    const ResidentArrayRecord* qw = require_array("model.layers.20.self_attn.q_proj.weight");
    const ResidentArrayRecord* qs = require_array("model.layers.20.self_attn.q_proj.scales");
    const ResidentArrayRecord* qb = require_array("model.layers.20.self_attn.q_proj.biases");
    const ResidentArrayRecord* kw = require_array("model.layers.20.self_attn.k_proj.weight");
    const ResidentArrayRecord* ks = require_array("model.layers.20.self_attn.k_proj.scales");
    const ResidentArrayRecord* kb = require_array("model.layers.20.self_attn.k_proj.biases");
    const ResidentArrayRecord* vw = require_array("model.layers.20.self_attn.v_proj.weight");
    const ResidentArrayRecord* vs = require_array("model.layers.20.self_attn.v_proj.scales");
    const ResidentArrayRecord* vb = require_array("model.layers.20.self_attn.v_proj.biases");
    const ResidentArrayRecord* ow = require_array("model.layers.20.self_attn.o_proj.weight");
    const ResidentArrayRecord* os = require_array("model.layers.20.self_attn.o_proj.scales");
    const ResidentArrayRecord* ob = require_array("model.layers.20.self_attn.o_proj.biases");
    const ResidentArrayRecord* qa = require_array("model.layers.20.self_attn.q_proj.lora_a");
    const ResidentArrayRecord* qlb = require_array("model.layers.20.self_attn.q_proj.lora_b");
    const ResidentArrayRecord* ka = require_array("model.layers.20.self_attn.k_proj.lora_a");
    const ResidentArrayRecord* klb = require_array("model.layers.20.self_attn.k_proj.lora_b");
    const ResidentArrayRecord* va = require_array("model.layers.20.self_attn.v_proj.lora_a");
    const ResidentArrayRecord* vlb = require_array("model.layers.20.self_attn.v_proj.lora_b");
    const ResidentArrayRecord* oa = require_array("model.layers.20.self_attn.o_proj.lora_a");
    const ResidentArrayRecord* olb = require_array("model.layers.20.self_attn.o_proj.lora_b");
    const ResidentArrayRecord* q_norm = require_array("model.layers.20.self_attn.q_norm.weight");
    const ResidentArrayRecord* k_norm = require_array("model.layers.20.self_attn.k_norm.weight");
    if (qw == nullptr || qs == nullptr || qb == nullptr || kw == nullptr || ks == nullptr || kb == nullptr ||
        vw == nullptr || vs == nullptr || vb == nullptr || ow == nullptr || os == nullptr || ob == nullptr ||
        qa == nullptr || qlb == nullptr || ka == nullptr || klb == nullptr || va == nullptr || vlb == nullptr ||
        oa == nullptr || olb == nullptr || q_norm == nullptr || k_norm == nullptr) {
        Napi::Error::New(env, "Selected layer20 attention/o_proj arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    constexpr int input_size = 2560;
    constexpr int attention_size = 4096;
    constexpr int output_size = 2560;
    constexpr int q_heads = 32;
    constexpr int kv_heads = 8;
    constexpr int q_heads_per_kv = 4;
    constexpr int head_dim = 128;
    constexpr int group_size = 64;
    constexpr int bits = 4;
    constexpr int rope_position = 7;
    constexpr float adapter_scale = 20.0f;
    constexpr float eps = 1.0e-6f;
    constexpr float rope_theta = 5000000.0f;
    constexpr float rope_scale = 1.0f;
    constexpr bool rope_traditional = false;

    std::vector<float> input(static_cast<size_t>(input_size));
    for (int i = 0; i < input_size; ++i) {
        input[static_cast<size_t>(i)] = static_cast<float>((i % 53) - 26) / 53.0f;
    }

    auto start = std::chrono::steady_clock::now();
    mlx::core::array input_array(input.begin(), mlx::core::Shape{1, input_size}, mlx::core::float32);
    auto project = [&](const mlx::core::array& x, const ResidentArrayRecord* w, const ResidentArrayRecord* s, const ResidentArrayRecord* b,
                       const ResidentArrayRecord* la, const ResidentArrayRecord* lb) {
        mlx::core::array base = mlx::core::astype(
            mlx::core::quantized_matmul(
                x,
                w->array,
                s->array,
                std::make_optional(b->array),
                true,
                group_size,
                bits,
                "affine"),
            mlx::core::float32);
        mlx::core::array delta = mlx::core::astype(
            adapter_scale * mlx::core::matmul(mlx::core::matmul(x, la->array), lb->array),
            mlx::core::float32);
        return mlx::core::astype(base + delta, mlx::core::float32);
    };
    auto project_base = [&](const mlx::core::array& x, const ResidentArrayRecord* w, const ResidentArrayRecord* s, const ResidentArrayRecord* b) {
        return mlx::core::astype(
            mlx::core::quantized_matmul(
                x,
                w->array,
                s->array,
                std::make_optional(b->array),
                true,
                group_size,
                bits,
                "affine"),
            mlx::core::float32);
    };

    mlx::core::array q_combined = project(input_array, qw, qs, qb, qa, qlb);
    mlx::core::array k_combined = project(input_array, kw, ks, kb, ka, klb);
    mlx::core::array v_combined = project(input_array, vw, vs, vb, va, vlb);
    mlx::core::array q_normed = mlx::core::astype(
        mlx::core::fast::rms_norm(mlx::core::reshape(q_combined, mlx::core::Shape{q_heads, head_dim}), std::make_optional(q_norm->array), eps),
        mlx::core::float32);
    mlx::core::array k_normed = mlx::core::astype(
        mlx::core::fast::rms_norm(mlx::core::reshape(k_combined, mlx::core::Shape{kv_heads, head_dim}), std::make_optional(k_norm->array), eps),
        mlx::core::float32);
    mlx::core::array q_rope = mlx::core::astype(
        mlx::core::fast::rope(
            mlx::core::reshape(q_normed, mlx::core::Shape{1, q_heads, 1, head_dim}),
            head_dim,
            rope_traditional,
            std::make_optional(rope_theta),
            rope_scale,
            rope_position),
        mlx::core::float32);
    mlx::core::array k_rope = mlx::core::astype(
        mlx::core::fast::rope(
            mlx::core::reshape(k_normed, mlx::core::Shape{1, kv_heads, 1, head_dim}),
            head_dim,
            rope_traditional,
            std::make_optional(rope_theta),
            rope_scale,
            rope_position),
        mlx::core::float32);
    mlx::core::array v_heads = mlx::core::reshape(v_combined, mlx::core::Shape{1, kv_heads, 1, head_dim});
    mlx::core::array attention = mlx::core::astype(
        mlx::core::fast::scaled_dot_product_attention(
            q_rope,
            mlx::core::repeat(k_rope, q_heads_per_kv, 1),
            mlx::core::repeat(v_heads, q_heads_per_kv, 1),
            1.0f / std::sqrt(static_cast<float>(head_dim))),
        mlx::core::float32);
    mlx::core::array attention_input = mlx::core::reshape(attention, mlx::core::Shape{1, attention_size});
    mlx::core::array o_projection_base = project_base(attention_input, ow, os, ob);
    mlx::core::array o_projection = project(attention_input, ow, os, ob, oa, olb);
    mlx::core::array q_flat = mlx::core::flatten(mlx::core::astype(q_rope, mlx::core::float32));
    mlx::core::array k_flat = mlx::core::flatten(mlx::core::astype(k_rope, mlx::core::float32));
    mlx::core::array v_flat = mlx::core::flatten(mlx::core::astype(v_combined, mlx::core::float32));
    mlx::core::eval(o_projection, o_projection_base, q_flat, k_flat, v_flat);
    auto end = std::chrono::steady_clock::now();

    const float* values = o_projection.data<float>();
    const float* base_values = o_projection_base.data<float>();
    const float* q_values = q_flat.data<float>();
    const float* k_values = k_flat.data<float>();
    const float* v_values = v_flat.data<float>();
    double checksum = 0.0;
    double abs_checksum = 0.0;
    for (int i = 0; i < output_size; ++i) {
        const double v = static_cast<double>(values[i]);
        checksum += v;
        abs_checksum += std::abs(v);
    }
    Napi::Array first = Napi::Array::New(env, 8);
    for (int i = 0; i < 8; ++i) {
        first.Set(static_cast<uint32_t>(i), Napi::Number::New(env, values[i]));
    }

    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    Napi::Object output = Napi::Object::New(env);
    output.Set("output_len", Napi::Number::New(env, static_cast<double>(o_projection.size())));
    output.Set("expected_output_len", Napi::Number::New(env, output_size));
    output.Set("checksum", Napi::Number::New(env, checksum));
    output.Set("abs_checksum", Napi::Number::New(env, abs_checksum));
    output.Set("first_values", first);
    output.Set("backend", Napi::String::New(env, "mlx"));

    std::vector<float> fused_values(static_cast<size_t>(output_size), 0.0f);
    GypsyResetDirectMetalAttentionKvCache();
    GypsyDirectMetalProbeResult fused = GypsyRunDirectMetalAttentionOProjectionAppendToHost(
        20,
        q_values,
        k_values,
        v_values,
        static_cast<const std::uint32_t*>(ow->raw_data),
        static_cast<const std::uint16_t*>(os->raw_data),
        static_cast<const std::uint16_t*>(ob->raw_data),
        fused_values.data(),
        1);
    Napi::Array fused_first = Napi::Array::New(env, 8);
    double fused_abs_checksum = 0.0;
    double fused_max_abs_diff = 0.0;
    int fused_worst_index = -1;
    for (int i = 0; i < output_size; ++i) {
        const float fv = fused_values[static_cast<size_t>(i)];
        const double diff = std::abs(static_cast<double>(fv) - static_cast<double>(base_values[i]));
        if (diff > fused_max_abs_diff) {
            fused_max_abs_diff = diff;
            fused_worst_index = i;
        }
        fused_abs_checksum += std::abs(static_cast<double>(fv));
        if (i < 8) {
            fused_first.Set(static_cast<uint32_t>(i), Napi::Number::New(env, fv));
        }
    }
    Napi::Object fused_direct = Napi::Object::New(env);
    fused_direct.Set("available", Napi::Boolean::New(env, true));
    fused_direct.Set("ok", Napi::Boolean::New(env, fused.ok && fused_max_abs_diff < 1.0e-3));
    fused_direct.Set("backend", Napi::String::New(env, "direct_metal_attention_then_base_o_projection"));
    fused_direct.Set("seq_len", Napi::Number::New(env, 1));
    fused_direct.Set("max_abs_diff_vs_mlx_base_o_projection", Napi::Number::New(env, fused_max_abs_diff));
    fused_direct.Set("worst_index", Napi::Number::New(env, fused_worst_index));
    fused_direct.Set("checksum", Napi::Number::New(env, fused.checksum));
    fused_direct.Set("abs_checksum", Napi::Number::New(env, fused_abs_checksum));
    fused_direct.Set("elapsed_ms", Napi::Number::New(env, fused.elapsed_ms));
    fused_direct.Set("first_values", fused_first);
    fused_direct.Set("error", fused.error == nullptr ? env.Null().As<Napi::Value>() : Napi::String::New(env, fused.error).As<Napi::Value>());

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_o_projection_path_probe_version", Napi::String::New(env, "gypsy-selected-o-projection-path-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_inputs_are_deterministic", Napi::Boolean::New(env, true));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer20_attention_then_o_proj"));
    out.Set("adapter_layer", Napi::Number::New(env, 20));
    out.Set("attention_output_consumed_by_o_proj", Napi::Boolean::New(env, true));
    out.Set("adapter_delta_applied_to_o_proj", Napi::Boolean::New(env, true));
    out.Set("o_proj_input_len", Napi::Number::New(env, attention_size));
    out.Set("o_proj_output_len", Napi::Number::New(env, output_size));
    out.Set("timing_ms", Napi::Number::New(env, timing_ms));
    out.Set("o_projection_output", output);
    out.Set("fused_direct_attention_o_projection", fused_direct);
    out.Set("readback_count", Napi::Number::New(env, 4));
    Napi::Array reasons = Napi::Array::New(env, 4);
    reasons.Set(static_cast<uint32_t>(0), Napi::String::New(env, "layer20_o_projection_output_summary"));
    reasons.Set(static_cast<uint32_t>(1), Napi::String::New(env, "layer20_direct_metal_fused_q_input"));
    reasons.Set(static_cast<uint32_t>(2), Napi::String::New(env, "layer20_direct_metal_fused_k_input"));
    reasons.Set(static_cast<uint32_t>(3), Napi::String::New(env, "layer20_direct_metal_fused_v_input"));
    out.Set("readback_reasons", reasons);
    out.Set("next_execution_step", Napi::String::New(env, "add_selected_post_attention_residual_path"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedPostAttentionResidualProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected post-attention residual probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected post-attention residual probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto require_array = [&](const std::string& name) -> const ResidentArrayRecord* {
        return FindResidentArray(session.selected_resident_arrays, name);
    };
    const ResidentArrayRecord* qw = require_array("model.layers.20.self_attn.q_proj.weight");
    const ResidentArrayRecord* qs = require_array("model.layers.20.self_attn.q_proj.scales");
    const ResidentArrayRecord* qb = require_array("model.layers.20.self_attn.q_proj.biases");
    const ResidentArrayRecord* kw = require_array("model.layers.20.self_attn.k_proj.weight");
    const ResidentArrayRecord* ks = require_array("model.layers.20.self_attn.k_proj.scales");
    const ResidentArrayRecord* kb = require_array("model.layers.20.self_attn.k_proj.biases");
    const ResidentArrayRecord* vw = require_array("model.layers.20.self_attn.v_proj.weight");
    const ResidentArrayRecord* vs = require_array("model.layers.20.self_attn.v_proj.scales");
    const ResidentArrayRecord* vb = require_array("model.layers.20.self_attn.v_proj.biases");
    const ResidentArrayRecord* ow = require_array("model.layers.20.self_attn.o_proj.weight");
    const ResidentArrayRecord* os = require_array("model.layers.20.self_attn.o_proj.scales");
    const ResidentArrayRecord* ob = require_array("model.layers.20.self_attn.o_proj.biases");
    const ResidentArrayRecord* qa = require_array("model.layers.20.self_attn.q_proj.lora_a");
    const ResidentArrayRecord* qlb = require_array("model.layers.20.self_attn.q_proj.lora_b");
    const ResidentArrayRecord* ka = require_array("model.layers.20.self_attn.k_proj.lora_a");
    const ResidentArrayRecord* klb = require_array("model.layers.20.self_attn.k_proj.lora_b");
    const ResidentArrayRecord* va = require_array("model.layers.20.self_attn.v_proj.lora_a");
    const ResidentArrayRecord* vlb = require_array("model.layers.20.self_attn.v_proj.lora_b");
    const ResidentArrayRecord* oa = require_array("model.layers.20.self_attn.o_proj.lora_a");
    const ResidentArrayRecord* olb = require_array("model.layers.20.self_attn.o_proj.lora_b");
    const ResidentArrayRecord* q_norm = require_array("model.layers.20.self_attn.q_norm.weight");
    const ResidentArrayRecord* k_norm = require_array("model.layers.20.self_attn.k_norm.weight");
    if (qw == nullptr || qs == nullptr || qb == nullptr || kw == nullptr || ks == nullptr || kb == nullptr ||
        vw == nullptr || vs == nullptr || vb == nullptr || ow == nullptr || os == nullptr || ob == nullptr ||
        qa == nullptr || qlb == nullptr || ka == nullptr || klb == nullptr || va == nullptr || vlb == nullptr ||
        oa == nullptr || olb == nullptr || q_norm == nullptr || k_norm == nullptr) {
        Napi::Error::New(env, "Selected layer20 post-attention residual arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    constexpr int input_size = 2560;
    constexpr int attention_size = 4096;
    constexpr int q_heads = 32;
    constexpr int kv_heads = 8;
    constexpr int q_heads_per_kv = 4;
    constexpr int head_dim = 128;
    constexpr int group_size = 64;
    constexpr int bits = 4;
    constexpr int rope_position = 7;
    constexpr float adapter_scale = 20.0f;
    constexpr float eps = 1.0e-6f;
    constexpr float rope_theta = 5000000.0f;
    constexpr float rope_scale = 1.0f;
    constexpr bool rope_traditional = false;

    std::vector<float> input(static_cast<size_t>(input_size));
    for (int i = 0; i < input_size; ++i) {
        input[static_cast<size_t>(i)] = static_cast<float>((i % 59) - 29) / 59.0f;
    }

    auto start = std::chrono::steady_clock::now();
    mlx::core::array residual_input(input.begin(), mlx::core::Shape{1, input_size}, mlx::core::float32);
    auto project = [&](const mlx::core::array& x, const ResidentArrayRecord* w, const ResidentArrayRecord* s, const ResidentArrayRecord* b,
                       const ResidentArrayRecord* la, const ResidentArrayRecord* lb) {
        mlx::core::array base = mlx::core::astype(
            mlx::core::quantized_matmul(
                x,
                w->array,
                s->array,
                std::make_optional(b->array),
                true,
                group_size,
                bits,
                "affine"),
            mlx::core::float32);
        mlx::core::array delta = mlx::core::astype(
            adapter_scale * mlx::core::matmul(mlx::core::matmul(x, la->array), lb->array),
            mlx::core::float32);
        return mlx::core::astype(base + delta, mlx::core::float32);
    };

    mlx::core::array q_combined = project(residual_input, qw, qs, qb, qa, qlb);
    mlx::core::array k_combined = project(residual_input, kw, ks, kb, ka, klb);
    mlx::core::array v_combined = project(residual_input, vw, vs, vb, va, vlb);
    mlx::core::array q_normed = mlx::core::astype(
        mlx::core::fast::rms_norm(mlx::core::reshape(q_combined, mlx::core::Shape{q_heads, head_dim}), std::make_optional(q_norm->array), eps),
        mlx::core::float32);
    mlx::core::array k_normed = mlx::core::astype(
        mlx::core::fast::rms_norm(mlx::core::reshape(k_combined, mlx::core::Shape{kv_heads, head_dim}), std::make_optional(k_norm->array), eps),
        mlx::core::float32);
    mlx::core::array q_rope = mlx::core::astype(
        mlx::core::fast::rope(
            mlx::core::reshape(q_normed, mlx::core::Shape{1, q_heads, 1, head_dim}),
            head_dim,
            rope_traditional,
            std::make_optional(rope_theta),
            rope_scale,
            rope_position),
        mlx::core::float32);
    mlx::core::array k_rope = mlx::core::astype(
        mlx::core::fast::rope(
            mlx::core::reshape(k_normed, mlx::core::Shape{1, kv_heads, 1, head_dim}),
            head_dim,
            rope_traditional,
            std::make_optional(rope_theta),
            rope_scale,
            rope_position),
        mlx::core::float32);
    mlx::core::array attention = mlx::core::astype(
        mlx::core::fast::scaled_dot_product_attention(
            q_rope,
            mlx::core::repeat(k_rope, q_heads_per_kv, 1),
            mlx::core::repeat(mlx::core::reshape(v_combined, mlx::core::Shape{1, kv_heads, 1, head_dim}), q_heads_per_kv, 1),
            1.0f / std::sqrt(static_cast<float>(head_dim))),
        mlx::core::float32);
    mlx::core::array o_projection = project(mlx::core::reshape(attention, mlx::core::Shape{1, attention_size}), ow, os, ob, oa, olb);
    mlx::core::array residual = mlx::core::astype(residual_input + o_projection, mlx::core::float32);
    mlx::core::eval(o_projection, residual);
    auto end = std::chrono::steady_clock::now();

    auto summarize = [&](const std::string& name, const mlx::core::array& arr) {
        const float* values = arr.data<float>();
        double checksum = 0.0;
        double abs_checksum = 0.0;
        for (int i = 0; i < input_size; ++i) {
            const double v = static_cast<double>(values[i]);
            checksum += v;
            abs_checksum += std::abs(v);
        }
        Napi::Array first = Napi::Array::New(env, 8);
        for (int i = 0; i < 8; ++i) {
            first.Set(static_cast<uint32_t>(i), Napi::Number::New(env, values[i]));
        }
        Napi::Object out = Napi::Object::New(env);
        out.Set("name", Napi::String::New(env, name));
        out.Set("output_len", Napi::Number::New(env, static_cast<double>(arr.size())));
        out.Set("expected_output_len", Napi::Number::New(env, input_size));
        out.Set("checksum", Napi::Number::New(env, checksum));
        out.Set("abs_checksum", Napi::Number::New(env, abs_checksum));
        out.Set("first_values", first);
        out.Set("backend", Napi::String::New(env, "mlx"));
        return out;
    };

    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_post_attention_residual_probe_version", Napi::String::New(env, "gypsy-selected-post-attention-residual-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_inputs_are_deterministic", Napi::Boolean::New(env, true));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer20_post_attention_residual"));
    out.Set("adapter_layer", Napi::Number::New(env, 20));
    out.Set("attention_output_consumed_by_o_proj", Napi::Boolean::New(env, true));
    out.Set("adapter_delta_applied_to_o_proj", Napi::Boolean::New(env, true));
    out.Set("residual_add_applied", Napi::Boolean::New(env, true));
    out.Set("residual_input_len", Napi::Number::New(env, input_size));
    out.Set("o_projection_output_len", Napi::Number::New(env, input_size));
    out.Set("residual_output_len", Napi::Number::New(env, input_size));
    out.Set("timing_ms", Napi::Number::New(env, timing_ms));
    out.Set("o_projection_output", summarize("o_projection_output", o_projection));
    out.Set("post_attention_residual", summarize("post_attention_residual", residual));
    out.Set("readback_count", Napi::Number::New(env, 2));
    Napi::Array reasons = Napi::Array::New(env, 2);
    reasons.Set(static_cast<uint32_t>(0), Napi::String::New(env, "layer20_o_projection_output_summary"));
    reasons.Set(static_cast<uint32_t>(1), Napi::String::New(env, "layer20_post_attention_residual_summary"));
    out.Set("readback_reasons", reasons);
    out.Set("next_execution_step", Napi::String::New(env, "add_selected_post_attention_rmsnorm_path"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedPostAttentionRmsNormProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected post-attention RMSNorm probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected post-attention RMSNorm probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto require_array = [&](const std::string& name) -> const ResidentArrayRecord* {
        return FindResidentArray(session.selected_resident_arrays, name);
    };
    const ResidentArrayRecord* qw = require_array("model.layers.20.self_attn.q_proj.weight");
    const ResidentArrayRecord* qs = require_array("model.layers.20.self_attn.q_proj.scales");
    const ResidentArrayRecord* qb = require_array("model.layers.20.self_attn.q_proj.biases");
    const ResidentArrayRecord* kw = require_array("model.layers.20.self_attn.k_proj.weight");
    const ResidentArrayRecord* ks = require_array("model.layers.20.self_attn.k_proj.scales");
    const ResidentArrayRecord* kb = require_array("model.layers.20.self_attn.k_proj.biases");
    const ResidentArrayRecord* vw = require_array("model.layers.20.self_attn.v_proj.weight");
    const ResidentArrayRecord* vs = require_array("model.layers.20.self_attn.v_proj.scales");
    const ResidentArrayRecord* vb = require_array("model.layers.20.self_attn.v_proj.biases");
    const ResidentArrayRecord* ow = require_array("model.layers.20.self_attn.o_proj.weight");
    const ResidentArrayRecord* os = require_array("model.layers.20.self_attn.o_proj.scales");
    const ResidentArrayRecord* ob = require_array("model.layers.20.self_attn.o_proj.biases");
    const ResidentArrayRecord* qa = require_array("model.layers.20.self_attn.q_proj.lora_a");
    const ResidentArrayRecord* qlb = require_array("model.layers.20.self_attn.q_proj.lora_b");
    const ResidentArrayRecord* ka = require_array("model.layers.20.self_attn.k_proj.lora_a");
    const ResidentArrayRecord* klb = require_array("model.layers.20.self_attn.k_proj.lora_b");
    const ResidentArrayRecord* va = require_array("model.layers.20.self_attn.v_proj.lora_a");
    const ResidentArrayRecord* vlb = require_array("model.layers.20.self_attn.v_proj.lora_b");
    const ResidentArrayRecord* oa = require_array("model.layers.20.self_attn.o_proj.lora_a");
    const ResidentArrayRecord* olb = require_array("model.layers.20.self_attn.o_proj.lora_b");
    const ResidentArrayRecord* q_norm = require_array("model.layers.20.self_attn.q_norm.weight");
    const ResidentArrayRecord* k_norm = require_array("model.layers.20.self_attn.k_norm.weight");
    const ResidentArrayRecord* post_norm = require_array("model.layers.20.post_attention_layernorm.weight");
    if (qw == nullptr || qs == nullptr || qb == nullptr || kw == nullptr || ks == nullptr || kb == nullptr ||
        vw == nullptr || vs == nullptr || vb == nullptr || ow == nullptr || os == nullptr || ob == nullptr ||
        qa == nullptr || qlb == nullptr || ka == nullptr || klb == nullptr || va == nullptr || vlb == nullptr ||
        oa == nullptr || olb == nullptr || q_norm == nullptr || k_norm == nullptr || post_norm == nullptr) {
        Napi::Error::New(env, "Selected layer20 post-attention RMSNorm arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    constexpr int input_size = 2560;
    constexpr int attention_size = 4096;
    constexpr int q_heads = 32;
    constexpr int kv_heads = 8;
    constexpr int q_heads_per_kv = 4;
    constexpr int head_dim = 128;
    constexpr int group_size = 64;
    constexpr int bits = 4;
    constexpr int rope_position = 7;
    constexpr float adapter_scale = 20.0f;
    constexpr float eps = 1.0e-6f;
    constexpr float rope_theta = 5000000.0f;
    constexpr float rope_scale = 1.0f;
    constexpr bool rope_traditional = false;

    std::vector<float> input(static_cast<size_t>(input_size));
    for (int i = 0; i < input_size; ++i) {
        input[static_cast<size_t>(i)] = static_cast<float>((i % 61) - 30) / 61.0f;
    }

    auto start = std::chrono::steady_clock::now();
    mlx::core::array residual_input(input.begin(), mlx::core::Shape{1, input_size}, mlx::core::float32);
    auto project = [&](const mlx::core::array& x, const ResidentArrayRecord* w, const ResidentArrayRecord* s, const ResidentArrayRecord* b,
                       const ResidentArrayRecord* la, const ResidentArrayRecord* lb) {
        mlx::core::array base = mlx::core::astype(
            mlx::core::quantized_matmul(
                x,
                w->array,
                s->array,
                std::make_optional(b->array),
                true,
                group_size,
                bits,
                "affine"),
            mlx::core::float32);
        mlx::core::array delta = mlx::core::astype(
            adapter_scale * mlx::core::matmul(mlx::core::matmul(x, la->array), lb->array),
            mlx::core::float32);
        return mlx::core::astype(base + delta, mlx::core::float32);
    };

    mlx::core::array q_combined = project(residual_input, qw, qs, qb, qa, qlb);
    mlx::core::array k_combined = project(residual_input, kw, ks, kb, ka, klb);
    mlx::core::array v_combined = project(residual_input, vw, vs, vb, va, vlb);
    mlx::core::array q_normed = mlx::core::astype(
        mlx::core::fast::rms_norm(mlx::core::reshape(q_combined, mlx::core::Shape{q_heads, head_dim}), std::make_optional(q_norm->array), eps),
        mlx::core::float32);
    mlx::core::array k_normed = mlx::core::astype(
        mlx::core::fast::rms_norm(mlx::core::reshape(k_combined, mlx::core::Shape{kv_heads, head_dim}), std::make_optional(k_norm->array), eps),
        mlx::core::float32);
    mlx::core::array q_rope = mlx::core::astype(
        mlx::core::fast::rope(
            mlx::core::reshape(q_normed, mlx::core::Shape{1, q_heads, 1, head_dim}),
            head_dim,
            rope_traditional,
            std::make_optional(rope_theta),
            rope_scale,
            rope_position),
        mlx::core::float32);
    mlx::core::array k_rope = mlx::core::astype(
        mlx::core::fast::rope(
            mlx::core::reshape(k_normed, mlx::core::Shape{1, kv_heads, 1, head_dim}),
            head_dim,
            rope_traditional,
            std::make_optional(rope_theta),
            rope_scale,
            rope_position),
        mlx::core::float32);
    mlx::core::array attention = mlx::core::astype(
        mlx::core::fast::scaled_dot_product_attention(
            q_rope,
            mlx::core::repeat(k_rope, q_heads_per_kv, 1),
            mlx::core::repeat(mlx::core::reshape(v_combined, mlx::core::Shape{1, kv_heads, 1, head_dim}), q_heads_per_kv, 1),
            1.0f / std::sqrt(static_cast<float>(head_dim))),
        mlx::core::float32);
    mlx::core::array o_projection = project(mlx::core::reshape(attention, mlx::core::Shape{1, attention_size}), ow, os, ob, oa, olb);
    mlx::core::array residual = mlx::core::astype(residual_input + o_projection, mlx::core::float32);
    mlx::core::array normalized = mlx::core::astype(
        mlx::core::fast::rms_norm(residual, std::make_optional(post_norm->array), eps),
        mlx::core::float32);
    mlx::core::eval(residual, normalized);
    auto end = std::chrono::steady_clock::now();

    auto summarize = [&](const std::string& name, const mlx::core::array& arr) {
        const float* values = arr.data<float>();
        double checksum = 0.0;
        double abs_checksum = 0.0;
        for (int i = 0; i < input_size; ++i) {
            const double v = static_cast<double>(values[i]);
            checksum += v;
            abs_checksum += std::abs(v);
        }
        Napi::Array first = Napi::Array::New(env, 8);
        for (int i = 0; i < 8; ++i) {
            first.Set(static_cast<uint32_t>(i), Napi::Number::New(env, values[i]));
        }
        Napi::Object out = Napi::Object::New(env);
        out.Set("name", Napi::String::New(env, name));
        out.Set("output_len", Napi::Number::New(env, static_cast<double>(arr.size())));
        out.Set("expected_output_len", Napi::Number::New(env, input_size));
        out.Set("checksum", Napi::Number::New(env, checksum));
        out.Set("abs_checksum", Napi::Number::New(env, abs_checksum));
        out.Set("first_values", first);
        out.Set("backend", Napi::String::New(env, "mlx"));
        return out;
    };

    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_post_attention_rmsnorm_probe_version", Napi::String::New(env, "gypsy-selected-post-attention-rmsnorm-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_inputs_are_deterministic", Napi::Boolean::New(env, true));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer20_post_attention_rmsnorm"));
    out.Set("adapter_layer", Napi::Number::New(env, 20));
    out.Set("post_attention_residual_consumed_by_rmsnorm", Napi::Boolean::New(env, true));
    out.Set("post_attention_rmsnorm_applied", Napi::Boolean::New(env, true));
    out.Set("post_attention_norm_weight", Napi::String::New(env, "model.layers.20.post_attention_layernorm.weight"));
    out.Set("rmsnorm_eps", Napi::Number::New(env, eps));
    out.Set("residual_output_len", Napi::Number::New(env, input_size));
    out.Set("normalized_output_len", Napi::Number::New(env, input_size));
    out.Set("timing_ms", Napi::Number::New(env, timing_ms));
    out.Set("post_attention_residual", summarize("post_attention_residual", residual));
    out.Set("post_attention_normalized", summarize("post_attention_normalized", normalized));
    out.Set("readback_count", Napi::Number::New(env, 2));
    Napi::Array reasons = Napi::Array::New(env, 2);
    reasons.Set(static_cast<uint32_t>(0), Napi::String::New(env, "layer20_post_attention_residual_summary"));
    reasons.Set(static_cast<uint32_t>(1), Napi::String::New(env, "layer20_post_attention_rmsnorm_summary"));
    out.Set("readback_reasons", reasons);
    out.Set("next_execution_step", Napi::String::New(env, "add_selected_mlp_gate_up_path"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedMlpGateUpProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected MLP gate/up probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected MLP gate/up probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto require_array = [&](const std::string& name) -> const ResidentArrayRecord* {
        return FindResidentArray(session.selected_resident_arrays, name);
    };
    const ResidentArrayRecord* gw = require_array("model.layers.20.mlp.gate_proj.weight");
    const ResidentArrayRecord* gs = require_array("model.layers.20.mlp.gate_proj.scales");
    const ResidentArrayRecord* gb = require_array("model.layers.20.mlp.gate_proj.biases");
    const ResidentArrayRecord* uw = require_array("model.layers.20.mlp.up_proj.weight");
    const ResidentArrayRecord* us = require_array("model.layers.20.mlp.up_proj.scales");
    const ResidentArrayRecord* ub = require_array("model.layers.20.mlp.up_proj.biases");
    const ResidentArrayRecord* ga = require_array("model.layers.20.mlp.gate_proj.lora_a");
    const ResidentArrayRecord* glb = require_array("model.layers.20.mlp.gate_proj.lora_b");
    const ResidentArrayRecord* ua = require_array("model.layers.20.mlp.up_proj.lora_a");
    const ResidentArrayRecord* ulb = require_array("model.layers.20.mlp.up_proj.lora_b");
    const ResidentArrayRecord* post_norm = require_array("model.layers.20.post_attention_layernorm.weight");
    if (gw == nullptr || gs == nullptr || gb == nullptr || uw == nullptr || us == nullptr || ub == nullptr ||
        ga == nullptr || glb == nullptr || ua == nullptr || ulb == nullptr || post_norm == nullptr) {
        Napi::Error::New(env, "Selected layer20 MLP gate/up arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    constexpr int hidden_size = 2560;
    constexpr int intermediate_size = 9728;
    constexpr int group_size = 64;
    constexpr int bits = 4;
    constexpr float adapter_scale = 20.0f;
    constexpr float eps = 1.0e-6f;

    std::vector<float> residual(static_cast<size_t>(hidden_size));
    for (int i = 0; i < hidden_size; ++i) {
        residual[static_cast<size_t>(i)] = static_cast<float>((i % 67) - 33) / 67.0f;
    }

    auto start = std::chrono::steady_clock::now();
    mlx::core::array residual_array(residual.begin(), mlx::core::Shape{1, hidden_size}, mlx::core::float32);
    mlx::core::array normalized = mlx::core::astype(
        mlx::core::fast::rms_norm(residual_array, std::make_optional(post_norm->array), eps),
        mlx::core::float32);
    auto project = [&](const mlx::core::array& x, const ResidentArrayRecord* w, const ResidentArrayRecord* s, const ResidentArrayRecord* b,
                       const ResidentArrayRecord* la, const ResidentArrayRecord* lb) {
        mlx::core::array base = mlx::core::astype(
            mlx::core::quantized_matmul(
                x,
                w->array,
                s->array,
                std::make_optional(b->array),
                true,
                group_size,
                bits,
                "affine"),
            mlx::core::float32);
        mlx::core::array delta = mlx::core::astype(
            adapter_scale * mlx::core::matmul(mlx::core::matmul(x, la->array), lb->array),
            mlx::core::float32);
        return mlx::core::astype(base + delta, mlx::core::float32);
    };
    mlx::core::array gate = project(normalized, gw, gs, gb, ga, glb);
    mlx::core::array up = project(normalized, uw, us, ub, ua, ulb);
    mlx::core::eval(normalized, gate, up);
    auto end = std::chrono::steady_clock::now();

    auto summarize = [&](const std::string& name, const mlx::core::array& arr, int expected_len) {
        const float* values = arr.data<float>();
        double checksum = 0.0;
        double abs_checksum = 0.0;
        for (int i = 0; i < expected_len; ++i) {
            const double v = static_cast<double>(values[i]);
            checksum += v;
            abs_checksum += std::abs(v);
        }
        Napi::Array first = Napi::Array::New(env, 8);
        for (int i = 0; i < 8; ++i) {
            first.Set(static_cast<uint32_t>(i), Napi::Number::New(env, values[i]));
        }
        Napi::Object out = Napi::Object::New(env);
        out.Set("name", Napi::String::New(env, name));
        out.Set("output_len", Napi::Number::New(env, static_cast<double>(arr.size())));
        out.Set("expected_output_len", Napi::Number::New(env, expected_len));
        out.Set("checksum", Napi::Number::New(env, checksum));
        out.Set("abs_checksum", Napi::Number::New(env, abs_checksum));
        out.Set("first_values", first);
        out.Set("backend", Napi::String::New(env, "mlx"));
        return out;
    };

    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_mlp_gate_up_probe_version", Napi::String::New(env, "gypsy-selected-mlp-gate-up-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_inputs_are_deterministic", Napi::Boolean::New(env, true));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer20_mlp_gate_up"));
    out.Set("adapter_layer", Napi::Number::New(env, 20));
    out.Set("post_attention_rmsnorm_feeds_mlp", Napi::Boolean::New(env, true));
    out.Set("adapter_delta_applied_to_gate_proj", Napi::Boolean::New(env, true));
    out.Set("adapter_delta_applied_to_up_proj", Napi::Boolean::New(env, true));
    out.Set("mlp_input_len", Napi::Number::New(env, hidden_size));
    out.Set("intermediate_len", Napi::Number::New(env, intermediate_size));
    out.Set("timing_ms", Napi::Number::New(env, timing_ms));
    out.Set("post_attention_normalized", summarize("post_attention_normalized", normalized, hidden_size));
    out.Set("gate_projection", summarize("gate_projection", gate, intermediate_size));
    out.Set("up_projection", summarize("up_projection", up, intermediate_size));
    out.Set("readback_count", Napi::Number::New(env, 3));
    Napi::Array reasons = Napi::Array::New(env, 3);
    reasons.Set(static_cast<uint32_t>(0), Napi::String::New(env, "layer20_post_attention_rmsnorm_summary"));
    reasons.Set(static_cast<uint32_t>(1), Napi::String::New(env, "layer20_mlp_gate_projection_summary"));
    reasons.Set(static_cast<uint32_t>(2), Napi::String::New(env, "layer20_mlp_up_projection_summary"));
    out.Set("readback_reasons", reasons);
    out.Set("next_execution_step", Napi::String::New(env, "add_selected_mlp_activation_path"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedMlpActivationProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected MLP activation probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected MLP activation probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto require_array = [&](const std::string& name) -> const ResidentArrayRecord* {
        return FindResidentArray(session.selected_resident_arrays, name);
    };
    const ResidentArrayRecord* gw = require_array("model.layers.20.mlp.gate_proj.weight");
    const ResidentArrayRecord* gs = require_array("model.layers.20.mlp.gate_proj.scales");
    const ResidentArrayRecord* gb = require_array("model.layers.20.mlp.gate_proj.biases");
    const ResidentArrayRecord* uw = require_array("model.layers.20.mlp.up_proj.weight");
    const ResidentArrayRecord* us = require_array("model.layers.20.mlp.up_proj.scales");
    const ResidentArrayRecord* ub = require_array("model.layers.20.mlp.up_proj.biases");
    const ResidentArrayRecord* ga = require_array("model.layers.20.mlp.gate_proj.lora_a");
    const ResidentArrayRecord* glb = require_array("model.layers.20.mlp.gate_proj.lora_b");
    const ResidentArrayRecord* ua = require_array("model.layers.20.mlp.up_proj.lora_a");
    const ResidentArrayRecord* ulb = require_array("model.layers.20.mlp.up_proj.lora_b");
    const ResidentArrayRecord* post_norm = require_array("model.layers.20.post_attention_layernorm.weight");
    if (gw == nullptr || gs == nullptr || gb == nullptr || uw == nullptr || us == nullptr || ub == nullptr ||
        ga == nullptr || glb == nullptr || ua == nullptr || ulb == nullptr || post_norm == nullptr) {
        Napi::Error::New(env, "Selected layer20 MLP activation arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    constexpr int hidden_size = 2560;
    constexpr int intermediate_size = 9728;
    constexpr int group_size = 64;
    constexpr int bits = 4;
    constexpr float adapter_scale = 20.0f;
    constexpr float eps = 1.0e-6f;

    std::vector<float> residual(static_cast<size_t>(hidden_size));
    for (int i = 0; i < hidden_size; ++i) {
        residual[static_cast<size_t>(i)] = static_cast<float>((i % 71) - 35) / 71.0f;
    }

    auto start = std::chrono::steady_clock::now();
    mlx::core::array residual_array(residual.begin(), mlx::core::Shape{1, hidden_size}, mlx::core::float32);
    mlx::core::array normalized = mlx::core::astype(
        mlx::core::fast::rms_norm(residual_array, std::make_optional(post_norm->array), eps),
        mlx::core::float32);
    auto project = [&](const mlx::core::array& x, const ResidentArrayRecord* w, const ResidentArrayRecord* s, const ResidentArrayRecord* b,
                       const ResidentArrayRecord* la, const ResidentArrayRecord* lb) {
        mlx::core::array base = mlx::core::astype(
            mlx::core::quantized_matmul(
                x,
                w->array,
                s->array,
                std::make_optional(b->array),
                true,
                group_size,
                bits,
                "affine"),
            mlx::core::float32);
        mlx::core::array delta = mlx::core::astype(
            adapter_scale * mlx::core::matmul(mlx::core::matmul(x, la->array), lb->array),
            mlx::core::float32);
        return mlx::core::astype(base + delta, mlx::core::float32);
    };
    mlx::core::array gate = project(normalized, gw, gs, gb, ga, glb);
    mlx::core::array up = project(normalized, uw, us, ub, ua, ulb);
    mlx::core::array activated = mlx::core::astype((gate * mlx::core::sigmoid(gate)) * up, mlx::core::float32);
    mlx::core::eval(gate, up, activated);
    auto end = std::chrono::steady_clock::now();

    auto summarize = [&](const std::string& name, const mlx::core::array& arr, int expected_len) {
        const float* values = arr.data<float>();
        double checksum = 0.0;
        double abs_checksum = 0.0;
        for (int i = 0; i < expected_len; ++i) {
            const double v = static_cast<double>(values[i]);
            checksum += v;
            abs_checksum += std::abs(v);
        }
        Napi::Array first = Napi::Array::New(env, 8);
        for (int i = 0; i < 8; ++i) {
            first.Set(static_cast<uint32_t>(i), Napi::Number::New(env, values[i]));
        }
        Napi::Object out = Napi::Object::New(env);
        out.Set("name", Napi::String::New(env, name));
        out.Set("output_len", Napi::Number::New(env, static_cast<double>(arr.size())));
        out.Set("expected_output_len", Napi::Number::New(env, expected_len));
        out.Set("checksum", Napi::Number::New(env, checksum));
        out.Set("abs_checksum", Napi::Number::New(env, abs_checksum));
        out.Set("first_values", first);
        out.Set("backend", Napi::String::New(env, "mlx"));
        return out;
    };

    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_mlp_activation_probe_version", Napi::String::New(env, "gypsy-selected-mlp-activation-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_inputs_are_deterministic", Napi::Boolean::New(env, true));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer20_mlp_activation"));
    out.Set("adapter_layer", Napi::Number::New(env, 20));
    out.Set("activation_formula", Napi::String::New(env, "silu(gate) * up"));
    out.Set("silu_formula", Napi::String::New(env, "gate * sigmoid(gate)"));
    out.Set("adapter_delta_applied_to_gate_proj", Napi::Boolean::New(env, true));
    out.Set("adapter_delta_applied_to_up_proj", Napi::Boolean::New(env, true));
    out.Set("intermediate_len", Napi::Number::New(env, intermediate_size));
    out.Set("timing_ms", Napi::Number::New(env, timing_ms));
    out.Set("gate_projection", summarize("gate_projection", gate, intermediate_size));
    out.Set("up_projection", summarize("up_projection", up, intermediate_size));
    out.Set("activated_mlp", summarize("activated_mlp", activated, intermediate_size));
    out.Set("readback_count", Napi::Number::New(env, 3));
    Napi::Array reasons = Napi::Array::New(env, 3);
    reasons.Set(static_cast<uint32_t>(0), Napi::String::New(env, "layer20_mlp_gate_projection_summary"));
    reasons.Set(static_cast<uint32_t>(1), Napi::String::New(env, "layer20_mlp_up_projection_summary"));
    reasons.Set(static_cast<uint32_t>(2), Napi::String::New(env, "layer20_mlp_activation_summary"));
    out.Set("readback_reasons", reasons);
    out.Set("next_execution_step", Napi::String::New(env, "add_selected_mlp_down_projection_path"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedMlpDownProjectionProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected MLP down projection probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected MLP down projection probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto require_array = [&](const std::string& name) -> const ResidentArrayRecord* {
        return FindResidentArray(session.selected_resident_arrays, name);
    };
    const ResidentArrayRecord* gw = require_array("model.layers.20.mlp.gate_proj.weight");
    const ResidentArrayRecord* gs = require_array("model.layers.20.mlp.gate_proj.scales");
    const ResidentArrayRecord* gb = require_array("model.layers.20.mlp.gate_proj.biases");
    const ResidentArrayRecord* uw = require_array("model.layers.20.mlp.up_proj.weight");
    const ResidentArrayRecord* us = require_array("model.layers.20.mlp.up_proj.scales");
    const ResidentArrayRecord* ub = require_array("model.layers.20.mlp.up_proj.biases");
    const ResidentArrayRecord* dw = require_array("model.layers.20.mlp.down_proj.weight");
    const ResidentArrayRecord* ds = require_array("model.layers.20.mlp.down_proj.scales");
    const ResidentArrayRecord* db = require_array("model.layers.20.mlp.down_proj.biases");
    const ResidentArrayRecord* ga = require_array("model.layers.20.mlp.gate_proj.lora_a");
    const ResidentArrayRecord* glb = require_array("model.layers.20.mlp.gate_proj.lora_b");
    const ResidentArrayRecord* ua = require_array("model.layers.20.mlp.up_proj.lora_a");
    const ResidentArrayRecord* ulb = require_array("model.layers.20.mlp.up_proj.lora_b");
    const ResidentArrayRecord* da = require_array("model.layers.20.mlp.down_proj.lora_a");
    const ResidentArrayRecord* dlb = require_array("model.layers.20.mlp.down_proj.lora_b");
    const ResidentArrayRecord* post_norm = require_array("model.layers.20.post_attention_layernorm.weight");
    if (gw == nullptr || gs == nullptr || gb == nullptr || uw == nullptr || us == nullptr || ub == nullptr ||
        dw == nullptr || ds == nullptr || db == nullptr || ga == nullptr || glb == nullptr || ua == nullptr ||
        ulb == nullptr || da == nullptr || dlb == nullptr || post_norm == nullptr) {
        Napi::Error::New(env, "Selected layer20 MLP down projection arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    constexpr int hidden_size = 2560;
    constexpr int intermediate_size = 9728;
    constexpr int group_size = 64;
    constexpr int bits = 4;
    constexpr float adapter_scale = 20.0f;
    constexpr float eps = 1.0e-6f;

    std::vector<float> residual(static_cast<size_t>(hidden_size));
    for (int i = 0; i < hidden_size; ++i) {
        residual[static_cast<size_t>(i)] = static_cast<float>((i % 73) - 36) / 73.0f;
    }

    auto start = std::chrono::steady_clock::now();
    mlx::core::array residual_array(residual.begin(), mlx::core::Shape{1, hidden_size}, mlx::core::float32);
    mlx::core::array normalized = mlx::core::astype(
        mlx::core::fast::rms_norm(residual_array, std::make_optional(post_norm->array), eps),
        mlx::core::float32);
    auto project = [&](const mlx::core::array& x, const ResidentArrayRecord* w, const ResidentArrayRecord* s, const ResidentArrayRecord* b,
                       const ResidentArrayRecord* la, const ResidentArrayRecord* lb) {
        mlx::core::array base = mlx::core::astype(
            mlx::core::quantized_matmul(
                x,
                w->array,
                s->array,
                std::make_optional(b->array),
                true,
                group_size,
                bits,
                "affine"),
            mlx::core::float32);
        mlx::core::array delta = mlx::core::astype(
            adapter_scale * mlx::core::matmul(mlx::core::matmul(x, la->array), lb->array),
            mlx::core::float32);
        return mlx::core::astype(base + delta, mlx::core::float32);
    };
    mlx::core::array gate = project(normalized, gw, gs, gb, ga, glb);
    mlx::core::array up = project(normalized, uw, us, ub, ua, ulb);
    mlx::core::array activated = mlx::core::astype((gate * mlx::core::sigmoid(gate)) * up, mlx::core::float32);
    mlx::core::array down = project(activated, dw, ds, db, da, dlb);
    mlx::core::eval(activated, down);
    auto end = std::chrono::steady_clock::now();

    auto summarize = [&](const std::string& name, const mlx::core::array& arr, int expected_len) {
        const float* values = arr.data<float>();
        double checksum = 0.0;
        double abs_checksum = 0.0;
        for (int i = 0; i < expected_len; ++i) {
            const double v = static_cast<double>(values[i]);
            checksum += v;
            abs_checksum += std::abs(v);
        }
        Napi::Array first = Napi::Array::New(env, 8);
        for (int i = 0; i < 8; ++i) {
            first.Set(static_cast<uint32_t>(i), Napi::Number::New(env, values[i]));
        }
        Napi::Object out = Napi::Object::New(env);
        out.Set("name", Napi::String::New(env, name));
        out.Set("output_len", Napi::Number::New(env, static_cast<double>(arr.size())));
        out.Set("expected_output_len", Napi::Number::New(env, expected_len));
        out.Set("checksum", Napi::Number::New(env, checksum));
        out.Set("abs_checksum", Napi::Number::New(env, abs_checksum));
        out.Set("first_values", first);
        out.Set("backend", Napi::String::New(env, "mlx"));
        return out;
    };

    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_mlp_down_projection_probe_version", Napi::String::New(env, "gypsy-selected-mlp-down-projection-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_inputs_are_deterministic", Napi::Boolean::New(env, true));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer20_mlp_down_projection"));
    out.Set("adapter_layer", Napi::Number::New(env, 20));
    out.Set("activation_formula", Napi::String::New(env, "silu(gate) * up"));
    out.Set("adapter_delta_applied_to_down_proj", Napi::Boolean::New(env, true));
    out.Set("intermediate_len", Napi::Number::New(env, intermediate_size));
    out.Set("down_output_len", Napi::Number::New(env, hidden_size));
    out.Set("timing_ms", Napi::Number::New(env, timing_ms));
    out.Set("activated_mlp", summarize("activated_mlp", activated, intermediate_size));
    out.Set("down_projection", summarize("down_projection", down, hidden_size));
    out.Set("readback_count", Napi::Number::New(env, 2));
    Napi::Array reasons = Napi::Array::New(env, 2);
    reasons.Set(static_cast<uint32_t>(0), Napi::String::New(env, "layer20_mlp_activation_summary"));
    reasons.Set(static_cast<uint32_t>(1), Napi::String::New(env, "layer20_mlp_down_projection_summary"));
    out.Set("readback_reasons", reasons);
    out.Set("next_execution_step", Napi::String::New(env, "add_selected_mlp_residual_path"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedMlpResidualProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected MLP residual probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected MLP residual probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto require_array = [&](const std::string& name) -> const ResidentArrayRecord* {
        return FindResidentArray(session.selected_resident_arrays, name);
    };
    const ResidentArrayRecord* gw = require_array("model.layers.20.mlp.gate_proj.weight");
    const ResidentArrayRecord* gs = require_array("model.layers.20.mlp.gate_proj.scales");
    const ResidentArrayRecord* gb = require_array("model.layers.20.mlp.gate_proj.biases");
    const ResidentArrayRecord* uw = require_array("model.layers.20.mlp.up_proj.weight");
    const ResidentArrayRecord* us = require_array("model.layers.20.mlp.up_proj.scales");
    const ResidentArrayRecord* ub = require_array("model.layers.20.mlp.up_proj.biases");
    const ResidentArrayRecord* dw = require_array("model.layers.20.mlp.down_proj.weight");
    const ResidentArrayRecord* ds = require_array("model.layers.20.mlp.down_proj.scales");
    const ResidentArrayRecord* db = require_array("model.layers.20.mlp.down_proj.biases");
    const ResidentArrayRecord* ga = require_array("model.layers.20.mlp.gate_proj.lora_a");
    const ResidentArrayRecord* glb = require_array("model.layers.20.mlp.gate_proj.lora_b");
    const ResidentArrayRecord* ua = require_array("model.layers.20.mlp.up_proj.lora_a");
    const ResidentArrayRecord* ulb = require_array("model.layers.20.mlp.up_proj.lora_b");
    const ResidentArrayRecord* da = require_array("model.layers.20.mlp.down_proj.lora_a");
    const ResidentArrayRecord* dlb = require_array("model.layers.20.mlp.down_proj.lora_b");
    const ResidentArrayRecord* post_norm = require_array("model.layers.20.post_attention_layernorm.weight");
    if (gw == nullptr || gs == nullptr || gb == nullptr || uw == nullptr || us == nullptr || ub == nullptr ||
        dw == nullptr || ds == nullptr || db == nullptr || ga == nullptr || glb == nullptr || ua == nullptr ||
        ulb == nullptr || da == nullptr || dlb == nullptr || post_norm == nullptr) {
        Napi::Error::New(env, "Selected layer20 MLP residual arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    constexpr int hidden_size = 2560;
    constexpr int intermediate_size = 9728;
    constexpr int group_size = 64;
    constexpr int bits = 4;
    constexpr float adapter_scale = 20.0f;
    constexpr float eps = 1.0e-6f;

    std::vector<float> residual(static_cast<size_t>(hidden_size));
    for (int i = 0; i < hidden_size; ++i) {
        residual[static_cast<size_t>(i)] = static_cast<float>((i % 79) - 39) / 79.0f;
    }

    auto start = std::chrono::steady_clock::now();
    mlx::core::array residual_array(residual.begin(), mlx::core::Shape{1, hidden_size}, mlx::core::float32);
    mlx::core::array normalized = mlx::core::astype(
        mlx::core::fast::rms_norm(residual_array, std::make_optional(post_norm->array), eps),
        mlx::core::float32);
    auto project = [&](const mlx::core::array& x, const ResidentArrayRecord* w, const ResidentArrayRecord* s, const ResidentArrayRecord* b,
                       const ResidentArrayRecord* la, const ResidentArrayRecord* lb) {
        mlx::core::array base = mlx::core::astype(
            mlx::core::quantized_matmul(
                x,
                w->array,
                s->array,
                std::make_optional(b->array),
                true,
                group_size,
                bits,
                "affine"),
            mlx::core::float32);
        mlx::core::array delta = mlx::core::astype(
            adapter_scale * mlx::core::matmul(mlx::core::matmul(x, la->array), lb->array),
            mlx::core::float32);
        return mlx::core::astype(base + delta, mlx::core::float32);
    };
    mlx::core::array gate = project(normalized, gw, gs, gb, ga, glb);
    mlx::core::array up = project(normalized, uw, us, ub, ua, ulb);
    mlx::core::array activated = mlx::core::astype((gate * mlx::core::sigmoid(gate)) * up, mlx::core::float32);
    mlx::core::array down = project(activated, dw, ds, db, da, dlb);
    mlx::core::array mlp_residual_output = mlx::core::astype(residual_array + down, mlx::core::float32);
    mlx::core::eval(activated, down, mlp_residual_output);
    auto end = std::chrono::steady_clock::now();

    auto summarize = [&](const std::string& name, const mlx::core::array& arr, int expected_len) {
        const float* values = arr.data<float>();
        double checksum = 0.0;
        double abs_checksum = 0.0;
        for (int i = 0; i < expected_len; ++i) {
            const double v = static_cast<double>(values[i]);
            checksum += v;
            abs_checksum += std::abs(v);
        }
        Napi::Array first = Napi::Array::New(env, 8);
        for (int i = 0; i < 8; ++i) {
            first.Set(static_cast<uint32_t>(i), Napi::Number::New(env, values[i]));
        }
        Napi::Object out = Napi::Object::New(env);
        out.Set("name", Napi::String::New(env, name));
        out.Set("output_len", Napi::Number::New(env, static_cast<double>(arr.size())));
        out.Set("expected_output_len", Napi::Number::New(env, expected_len));
        out.Set("checksum", Napi::Number::New(env, checksum));
        out.Set("abs_checksum", Napi::Number::New(env, abs_checksum));
        out.Set("first_values", first);
        out.Set("backend", Napi::String::New(env, "mlx"));
        return out;
    };

    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_mlp_residual_probe_version", Napi::String::New(env, "gypsy-selected-mlp-residual-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_inputs_are_deterministic", Napi::Boolean::New(env, true));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer20_mlp_residual"));
    out.Set("adapter_layer", Napi::Number::New(env, 20));
    out.Set("activation_formula", Napi::String::New(env, "silu(gate) * up"));
    out.Set("adapter_delta_applied_to_down_proj", Napi::Boolean::New(env, true));
    out.Set("mlp_residual_add_applied", Napi::Boolean::New(env, true));
    out.Set("intermediate_len", Napi::Number::New(env, intermediate_size));
    out.Set("down_output_len", Napi::Number::New(env, hidden_size));
    out.Set("final_output_len", Napi::Number::New(env, hidden_size));
    out.Set("timing_ms", Napi::Number::New(env, timing_ms));
    out.Set("activated_mlp", summarize("activated_mlp", activated, intermediate_size));
    out.Set("down_projection", summarize("down_projection", down, hidden_size));
    out.Set("mlp_residual_output", summarize("mlp_residual_output", mlp_residual_output, hidden_size));
    out.Set("readback_count", Napi::Number::New(env, 3));
    Napi::Array reasons = Napi::Array::New(env, 3);
    reasons.Set(static_cast<uint32_t>(0), Napi::String::New(env, "layer20_mlp_activation_summary"));
    reasons.Set(static_cast<uint32_t>(1), Napi::String::New(env, "layer20_mlp_down_projection_summary"));
    reasons.Set(static_cast<uint32_t>(2), Napi::String::New(env, "layer20_mlp_residual_summary"));
    out.Set("readback_reasons", reasons);
    out.Set("next_execution_step", Napi::String::New(env, "add_selected_layer_output_contract"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedLayerOutputContractProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected layer output contract probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected layer output contract probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto require_array = [&](const std::string& name) -> const ResidentArrayRecord* {
        return FindResidentArray(session.selected_resident_arrays, name);
    };
    const std::array<std::string, 18> required = {
        "model.layers.20.self_attn.q_proj.weight",
        "model.layers.20.self_attn.k_proj.weight",
        "model.layers.20.self_attn.v_proj.weight",
        "model.layers.20.self_attn.o_proj.weight",
        "model.layers.20.mlp.gate_proj.weight",
        "model.layers.20.mlp.up_proj.weight",
        "model.layers.20.mlp.down_proj.weight",
        "model.layers.20.self_attn.q_norm.weight",
        "model.layers.20.self_attn.k_norm.weight",
        "model.layers.20.post_attention_layernorm.weight",
        "model.layers.20.self_attn.q_proj.lora_a",
        "model.layers.20.self_attn.k_proj.lora_a",
        "model.layers.20.self_attn.v_proj.lora_a",
        "model.layers.20.self_attn.o_proj.lora_a",
        "model.layers.20.mlp.gate_proj.lora_a",
        "model.layers.20.mlp.up_proj.lora_a",
        "model.layers.20.mlp.down_proj.lora_a",
        "model.norm.weight",
    };
    std::vector<std::string> missing;
    for (const auto& name : required) {
        if (require_array(name) == nullptr) {
            missing.push_back(name);
        }
    }
    if (!missing.empty()) {
        Napi::Array missing_arr = Napi::Array::New(env, missing.size());
        for (size_t i = 0; i < missing.size(); ++i) {
            missing_arr.Set(static_cast<uint32_t>(i), Napi::String::New(env, missing[i]));
        }
        Napi::Error err = Napi::Error::New(env, "Selected layer output contract arrays are incomplete");
        err.Value().Set("missing_arrays", missing_arr);
        err.ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::Array shape = Napi::Array::New(env, 2);
    shape.Set(static_cast<uint32_t>(0), Napi::Number::New(env, 1));
    shape.Set(static_cast<uint32_t>(1), Napi::Number::New(env, 2560));

    Napi::Array consumers = Napi::Array::New(env, 3);
    consumers.Set(static_cast<uint32_t>(0), Napi::String::New(env, "next_layer_input"));
    consumers.Set(static_cast<uint32_t>(1), Napi::String::New(env, "final_norm_input"));
    consumers.Set(static_cast<uint32_t>(2), Napi::String::New(env, "debug_summary_readback"));

    Napi::Array invariants = Napi::Array::New(env, 7);
    invariants.Set(static_cast<uint32_t>(0), Napi::String::New(env, "adapter_delta_added_before_q_norm"));
    invariants.Set(static_cast<uint32_t>(1), Napi::String::New(env, "adapter_delta_added_before_k_norm"));
    invariants.Set(static_cast<uint32_t>(2), Napi::String::New(env, "q_norm_before_rope"));
    invariants.Set(static_cast<uint32_t>(3), Napi::String::New(env, "k_norm_before_rope"));
    invariants.Set(static_cast<uint32_t>(4), Napi::String::New(env, "attention_o_projection_before_residual"));
    invariants.Set(static_cast<uint32_t>(5), Napi::String::New(env, "post_attention_rmsnorm_before_mlp"));
    invariants.Set(static_cast<uint32_t>(6), Napi::String::New(env, "mlp_down_projection_before_residual"));

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_layer_output_contract_probe_version", Napi::String::New(env, "gypsy-selected-layer-output-contract-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, false));
    out.Set("contract_only", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer20_output_contract"));
    out.Set("adapter_layer", Napi::Number::New(env, 20));
    out.Set("producer_probe", Napi::String::New(env, "runSelectedMlpResidualProbe"));
    out.Set("output_name", Napi::String::New(env, "layer20_output"));
    out.Set("output_dtype", Napi::String::New(env, "float32"));
    out.Set("output_shape", shape);
    out.Set("output_len", Napi::Number::New(env, 2560));
    out.Set("hidden_size", Napi::Number::New(env, 2560));
    out.Set("native_value_required", Napi::Boolean::New(env, true));
    out.Set("coffeescript_payload_allowed", Napi::Boolean::New(env, false));
    out.Set("valid_downstream_consumers", consumers);
    out.Set("ordering_invariants", invariants);
    out.Set("required_selected_array_count", Napi::Number::New(env, static_cast<double>(required.size())));
    out.Set("required_selected_arrays_present", Napi::Boolean::New(env, true));
    out.Set("readback_count", Napi::Number::New(env, 0));
    out.Set("next_execution_step", Napi::String::New(env, "add_selected_next_layer_handoff_probe"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedNextLayerHandoffProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected next-layer handoff probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected next-layer handoff probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    auto model_it = gypsy_models.find(session.model_handle);
    if (model_it == gypsy_models.end()) {
        Napi::Error::New(env, "Session model handle is no longer loaded").ThrowAsJavaScriptException();
        return env.Null();
    }
    auto adapter_it = gypsy_adapters.find(session.adapter_handle);
    if (adapter_it == gypsy_adapters.end()) {
        Napi::Error::New(env, "Session adapter handle is no longer loaded").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto has_descriptor = [](const std::vector<TensorDescriptor>& descriptors, const std::string& name) {
        return std::any_of(descriptors.begin(), descriptors.end(), [&](const TensorDescriptor& d) {
            return d.name == name;
        });
    };
    const std::array<std::string, 23> model_required = {
        "model.layers.21.input_layernorm.weight",
        "model.layers.21.post_attention_layernorm.weight",
        "model.layers.21.self_attn.q_norm.weight",
        "model.layers.21.self_attn.k_norm.weight",
        "model.layers.21.self_attn.q_proj.weight",
        "model.layers.21.self_attn.q_proj.scales",
        "model.layers.21.self_attn.q_proj.biases",
        "model.layers.21.self_attn.k_proj.weight",
        "model.layers.21.self_attn.k_proj.scales",
        "model.layers.21.self_attn.k_proj.biases",
        "model.layers.21.self_attn.v_proj.weight",
        "model.layers.21.self_attn.v_proj.scales",
        "model.layers.21.self_attn.v_proj.biases",
        "model.layers.21.self_attn.o_proj.weight",
        "model.layers.21.self_attn.o_proj.scales",
        "model.layers.21.self_attn.o_proj.biases",
        "model.layers.21.mlp.gate_proj.weight",
        "model.layers.21.mlp.gate_proj.scales",
        "model.layers.21.mlp.gate_proj.biases",
        "model.layers.21.mlp.up_proj.weight",
        "model.layers.21.mlp.up_proj.scales",
        "model.layers.21.mlp.up_proj.biases",
        "model.layers.21.mlp.down_proj.weight",
    };
    const std::array<std::string, 6> model_required_tail = {
        "model.layers.21.mlp.down_proj.scales",
        "model.layers.21.mlp.down_proj.biases",
        "model.norm.weight",
        "model.embed_tokens.weight",
        "model.embed_tokens.scales",
        "model.embed_tokens.biases",
    };
    const std::array<std::string, 14> adapter_required = {
        "model.layers.21.self_attn.q_proj.lora_a",
        "model.layers.21.self_attn.q_proj.lora_b",
        "model.layers.21.self_attn.k_proj.lora_a",
        "model.layers.21.self_attn.k_proj.lora_b",
        "model.layers.21.self_attn.v_proj.lora_a",
        "model.layers.21.self_attn.v_proj.lora_b",
        "model.layers.21.self_attn.o_proj.lora_a",
        "model.layers.21.self_attn.o_proj.lora_b",
        "model.layers.21.mlp.gate_proj.lora_a",
        "model.layers.21.mlp.gate_proj.lora_b",
        "model.layers.21.mlp.up_proj.lora_a",
        "model.layers.21.mlp.up_proj.lora_b",
        "model.layers.21.mlp.down_proj.lora_a",
        "model.layers.21.mlp.down_proj.lora_b",
    };

    std::vector<std::string> missing_model;
    for (const auto& name : model_required) {
        if (!has_descriptor(model_it->second.tensor_descriptors, name)) missing_model.push_back(name);
    }
    for (const auto& name : model_required_tail) {
        if (!has_descriptor(model_it->second.tensor_descriptors, name)) missing_model.push_back(name);
    }
    std::vector<std::string> missing_adapter;
    for (const auto& name : adapter_required) {
        if (!has_descriptor(adapter_it->second.tensor_descriptors, name)) missing_adapter.push_back(name);
    }
    if (!missing_model.empty() || !missing_adapter.empty()) {
        Napi::Object details = Napi::Object::New(env);
        details.Set("missing_model_descriptors", StringVectorToNapiArray(env, missing_model));
        details.Set("missing_adapter_descriptors", StringVectorToNapiArray(env, missing_adapter));
        Napi::Error err = Napi::Error::New(env, "Selected next-layer handoff descriptors are incomplete");
        err.Value().Set("details", details);
        err.ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::Array input_shape = Napi::Array::New(env, 2);
    input_shape.Set(static_cast<uint32_t>(0), Napi::Number::New(env, 1));
    input_shape.Set(static_cast<uint32_t>(1), Napi::Number::New(env, 2560));

    Napi::Array required_groups = Napi::Array::New(env, 9);
    required_groups.Set(static_cast<uint32_t>(0), Napi::String::New(env, "input_layernorm"));
    required_groups.Set(static_cast<uint32_t>(1), Napi::String::New(env, "q_proj"));
    required_groups.Set(static_cast<uint32_t>(2), Napi::String::New(env, "k_proj"));
    required_groups.Set(static_cast<uint32_t>(3), Napi::String::New(env, "v_proj"));
    required_groups.Set(static_cast<uint32_t>(4), Napi::String::New(env, "o_proj"));
    required_groups.Set(static_cast<uint32_t>(5), Napi::String::New(env, "post_attention_layernorm"));
    required_groups.Set(static_cast<uint32_t>(6), Napi::String::New(env, "gate_proj"));
    required_groups.Set(static_cast<uint32_t>(7), Napi::String::New(env, "up_proj"));
    required_groups.Set(static_cast<uint32_t>(8), Napi::String::New(env, "down_proj"));

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_next_layer_handoff_probe_version", Napi::String::New(env, "gypsy-selected-next-layer-handoff-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, false));
    out.Set("contract_only", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer20_to_layer21_handoff"));
    out.Set("source_layer", Napi::Number::New(env, 20));
    out.Set("target_layer", Napi::Number::New(env, 21));
    out.Set("handoff_value_name", Napi::String::New(env, "layer20_output"));
    out.Set("target_input_name", Napi::String::New(env, "layer21_input"));
    out.Set("target_input_shape", input_shape);
    out.Set("target_input_len", Napi::Number::New(env, 2560));
    out.Set("target_input_dtype", Napi::String::New(env, "float32"));
    out.Set("target_layer_descriptors_present", Napi::Boolean::New(env, true));
    out.Set("target_adapter_descriptors_present", Napi::Boolean::New(env, true));
    out.Set("target_model_descriptor_count_checked", Napi::Number::New(env, static_cast<double>(model_required.size() + model_required_tail.size())));
    out.Set("target_adapter_descriptor_count_checked", Napi::Number::New(env, static_cast<double>(adapter_required.size())));
    out.Set("target_required_groups", required_groups);
    out.Set("target_payload_loaded", Napi::Boolean::New(env, false));
    out.Set("target_resident_arrays_constructed", Napi::Boolean::New(env, false));
    out.Set("native_value_required", Napi::Boolean::New(env, true));
    out.Set("coffeescript_payload_allowed", Napi::Boolean::New(env, false));
    out.Set("readback_count", Napi::Number::New(env, 0));
    out.Set("next_execution_step", Napi::String::New(env, "add_selected_layer21_residency_probe"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedLayer21ResidencyProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected layer21 residency probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected layer21 residency probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    auto model_it = gypsy_models.find(session.model_handle);
    if (model_it == gypsy_models.end()) {
        Napi::Error::New(env, "Session model handle is no longer loaded").ThrowAsJavaScriptException();
        return env.Null();
    }
    auto adapter_it = gypsy_adapters.find(session.adapter_handle);
    if (adapter_it == gypsy_adapters.end()) {
        Napi::Error::New(env, "Session adapter handle is no longer loaded").ThrowAsJavaScriptException();
        return env.Null();
    }

    const std::vector<std::string> model_tensor_names = {
        "model.layers.21.input_layernorm.weight",
        "model.layers.21.self_attn.q_norm.weight",
        "model.layers.21.self_attn.k_norm.weight",
        "model.layers.21.post_attention_layernorm.weight",
        "model.layers.21.self_attn.q_proj.weight",
        "model.layers.21.self_attn.q_proj.scales",
        "model.layers.21.self_attn.q_proj.biases",
        "model.layers.21.self_attn.k_proj.weight",
        "model.layers.21.self_attn.k_proj.scales",
        "model.layers.21.self_attn.k_proj.biases",
        "model.layers.21.self_attn.v_proj.weight",
        "model.layers.21.self_attn.v_proj.scales",
        "model.layers.21.self_attn.v_proj.biases",
        "model.layers.21.self_attn.o_proj.weight",
        "model.layers.21.self_attn.o_proj.scales",
        "model.layers.21.self_attn.o_proj.biases",
        "model.layers.21.mlp.gate_proj.weight",
        "model.layers.21.mlp.gate_proj.scales",
        "model.layers.21.mlp.gate_proj.biases",
        "model.layers.21.mlp.up_proj.weight",
        "model.layers.21.mlp.up_proj.scales",
        "model.layers.21.mlp.up_proj.biases",
        "model.layers.21.mlp.down_proj.weight",
        "model.layers.21.mlp.down_proj.scales",
        "model.layers.21.mlp.down_proj.biases",
    };
    const std::vector<std::string> adapter_tensor_names = {
        "model.layers.21.self_attn.q_proj.lora_a",
        "model.layers.21.self_attn.q_proj.lora_b",
        "model.layers.21.self_attn.k_proj.lora_a",
        "model.layers.21.self_attn.k_proj.lora_b",
        "model.layers.21.self_attn.v_proj.lora_a",
        "model.layers.21.self_attn.v_proj.lora_b",
        "model.layers.21.self_attn.o_proj.lora_a",
        "model.layers.21.self_attn.o_proj.lora_b",
        "model.layers.21.mlp.gate_proj.lora_a",
        "model.layers.21.mlp.gate_proj.lora_b",
        "model.layers.21.mlp.up_proj.lora_a",
        "model.layers.21.mlp.up_proj.lora_b",
        "model.layers.21.mlp.down_proj.lora_a",
        "model.layers.21.mlp.down_proj.lora_b",
    };

    std::vector<ResidentArrayRecord> constructed;
    std::string error;
    for (const auto& name : model_tensor_names) {
        const TensorDescriptor* d = FindTensorDescriptor(model_it->second.tensor_descriptors, name);
        if (d == nullptr) {
            Napi::Error::New(env, "Missing selected layer21 model tensor: " + name).ThrowAsJavaScriptException();
            return env.Null();
        }
        auto record = ConstructMappedMlxArray(*d, session.model_mapped_files, error);
        if (!record) {
            Napi::Error::New(env, "Failed to construct selected layer21 model MLX array: " + error).ThrowAsJavaScriptException();
            return env.Null();
        }
        constructed.push_back(std::move(*record));
    }
    for (const auto& name : adapter_tensor_names) {
        const TensorDescriptor* d = FindTensorDescriptor(adapter_it->second.tensor_descriptors, name);
        if (d == nullptr) {
            Napi::Error::New(env, "Missing selected layer21 adapter tensor: " + name).ThrowAsJavaScriptException();
            return env.Null();
        }
        auto record = ConstructMappedMlxArray(*d, session.adapter_mapped_files, error);
        if (!record) {
            Napi::Error::New(env, "Failed to construct selected layer21 adapter MLX array: " + error).ThrowAsJavaScriptException();
            return env.Null();
        }
        constructed.push_back(std::move(*record));
    }

    std::uint64_t model_bytes = 0;
    std::uint64_t adapter_bytes = 0;
    Napi::Array arrays = Napi::Array::New(env, constructed.size());
    for (size_t i = 0; i < constructed.size(); ++i) {
        const auto& r = constructed[i];
        if (r.name.find(".lora_") != std::string::npos) {
            adapter_bytes += r.byte_size;
        } else {
            model_bytes += r.byte_size;
        }
        Napi::Object item = Napi::Object::New(env);
        item.Set("name", Napi::String::New(env, r.name));
        item.Set("role", Napi::String::New(env, r.role));
        item.Set("array_kind", Napi::String::New(env, r.array_kind));
        item.Set("dtype", Napi::String::New(env, r.dtype));
        item.Set("byte_size", Napi::Number::New(env, static_cast<double>(r.byte_size)));
        item.Set("mlx_array_constructed", Napi::Boolean::New(env, true));
        item.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
        item.Set("shape_rank", Napi::Number::New(env, static_cast<double>(r.array.ndim())));
        item.Set("element_count", Napi::Number::New(env, static_cast<double>(r.array.size())));
        item.Set("nbytes", Napi::Number::New(env, static_cast<double>(r.array.nbytes())));
        arrays.Set(static_cast<uint32_t>(i), item);
    }

    Napi::Array groups = Napi::Array::New(env, 9);
    groups.Set(static_cast<uint32_t>(0), Napi::String::New(env, "input_layernorm"));
    groups.Set(static_cast<uint32_t>(1), Napi::String::New(env, "q_proj"));
    groups.Set(static_cast<uint32_t>(2), Napi::String::New(env, "k_proj"));
    groups.Set(static_cast<uint32_t>(3), Napi::String::New(env, "v_proj"));
    groups.Set(static_cast<uint32_t>(4), Napi::String::New(env, "o_proj"));
    groups.Set(static_cast<uint32_t>(5), Napi::String::New(env, "post_attention_layernorm"));
    groups.Set(static_cast<uint32_t>(6), Napi::String::New(env, "gate_proj"));
    groups.Set(static_cast<uint32_t>(7), Napi::String::New(env, "up_proj"));
    groups.Set(static_cast<uint32_t>(8), Napi::String::New(env, "down_proj"));

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_layer21_residency_probe_version", Napi::String::New(env, "gypsy-selected-layer21-residency-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, false));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer21_arrays"));
    out.Set("source_layer", Napi::Number::New(env, 20));
    out.Set("target_layer", Napi::Number::New(env, 21));
    out.Set("target_input_contract", Napi::String::New(env, "layer20_output_float32_1x2560"));
    out.Set("target_groups", groups);
    out.Set("model_array_count", Napi::Number::New(env, static_cast<double>(model_tensor_names.size())));
    out.Set("adapter_array_count", Napi::Number::New(env, static_cast<double>(adapter_tensor_names.size())));
    out.Set("array_count", Napi::Number::New(env, static_cast<double>(constructed.size())));
    out.Set("model_array_bytes", Napi::Number::New(env, static_cast<double>(model_bytes)));
    out.Set("adapter_array_bytes", Napi::Number::New(env, static_cast<double>(adapter_bytes)));
    out.Set("array_bytes", Napi::Number::New(env, static_cast<double>(model_bytes + adapter_bytes)));
    out.Set("constructed_for_probe_only", Napi::Boolean::New(env, true));
    out.Set("stored_on_session", Napi::Boolean::New(env, false));
    out.Set("session_selected_resident_array_count", Napi::Number::New(env, static_cast<double>(session.selected_resident_arrays.size())));
    out.Set("native_payload_copied_to_mlx_allocator", Napi::Boolean::New(env, true));
    out.Set("arrays", arrays);
    out.Set("readback_count", Napi::Number::New(env, 0));
    out.Set("next_execution_step", Napi::String::New(env, "add_selected_layer21_input_norm_probe"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedLayer21InputQkvProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected layer21 input/qkv probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected layer21 input/qkv probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    auto model_it = gypsy_models.find(session.model_handle);
    auto adapter_it = gypsy_adapters.find(session.adapter_handle);
    if (model_it == gypsy_models.end() || adapter_it == gypsy_adapters.end()) {
        Napi::Error::New(env, "Session references unloaded model or adapter").ThrowAsJavaScriptException();
        return env.Null();
    }

    const std::vector<std::string> names = {
        "model.layers.21.input_layernorm.weight",
        "model.layers.21.self_attn.q_norm.weight",
        "model.layers.21.self_attn.k_norm.weight",
        "model.layers.21.self_attn.q_proj.weight",
        "model.layers.21.self_attn.q_proj.scales",
        "model.layers.21.self_attn.q_proj.biases",
        "model.layers.21.self_attn.k_proj.weight",
        "model.layers.21.self_attn.k_proj.scales",
        "model.layers.21.self_attn.k_proj.biases",
        "model.layers.21.self_attn.v_proj.weight",
        "model.layers.21.self_attn.v_proj.scales",
        "model.layers.21.self_attn.v_proj.biases",
    };
    const std::vector<std::string> adapter_names = {
        "model.layers.21.self_attn.q_proj.lora_a",
        "model.layers.21.self_attn.q_proj.lora_b",
        "model.layers.21.self_attn.k_proj.lora_a",
        "model.layers.21.self_attn.k_proj.lora_b",
        "model.layers.21.self_attn.v_proj.lora_a",
        "model.layers.21.self_attn.v_proj.lora_b",
    };

    std::vector<ResidentArrayRecord> arrays;
    std::string error;
    for (const auto& name : names) {
        const TensorDescriptor* d = FindTensorDescriptor(model_it->second.tensor_descriptors, name);
        if (d == nullptr) {
            Napi::Error::New(env, "Missing selected layer21 model tensor: " + name).ThrowAsJavaScriptException();
            return env.Null();
        }
        auto record = ConstructMappedMlxArray(*d, session.model_mapped_files, error);
        if (!record) {
            Napi::Error::New(env, "Failed to construct selected layer21 model MLX array: " + error).ThrowAsJavaScriptException();
            return env.Null();
        }
        arrays.push_back(std::move(*record));
    }
    for (const auto& name : adapter_names) {
        const TensorDescriptor* d = FindTensorDescriptor(adapter_it->second.tensor_descriptors, name);
        if (d == nullptr) {
            Napi::Error::New(env, "Missing selected layer21 adapter tensor: " + name).ThrowAsJavaScriptException();
            return env.Null();
        }
        auto record = ConstructMappedMlxArray(*d, session.adapter_mapped_files, error);
        if (!record) {
            Napi::Error::New(env, "Failed to construct selected layer21 adapter MLX array: " + error).ThrowAsJavaScriptException();
            return env.Null();
        }
        arrays.push_back(std::move(*record));
    }

    auto require_array = [&](const std::string& name) -> const ResidentArrayRecord* {
        return FindResidentArray(arrays, name);
    };
    const ResidentArrayRecord* input_norm = require_array("model.layers.21.input_layernorm.weight");
    const ResidentArrayRecord* q_norm = require_array("model.layers.21.self_attn.q_norm.weight");
    const ResidentArrayRecord* k_norm = require_array("model.layers.21.self_attn.k_norm.weight");
    const ResidentArrayRecord* qw = require_array("model.layers.21.self_attn.q_proj.weight");
    const ResidentArrayRecord* qs = require_array("model.layers.21.self_attn.q_proj.scales");
    const ResidentArrayRecord* qb = require_array("model.layers.21.self_attn.q_proj.biases");
    const ResidentArrayRecord* kw = require_array("model.layers.21.self_attn.k_proj.weight");
    const ResidentArrayRecord* ks = require_array("model.layers.21.self_attn.k_proj.scales");
    const ResidentArrayRecord* kb = require_array("model.layers.21.self_attn.k_proj.biases");
    const ResidentArrayRecord* vw = require_array("model.layers.21.self_attn.v_proj.weight");
    const ResidentArrayRecord* vs = require_array("model.layers.21.self_attn.v_proj.scales");
    const ResidentArrayRecord* vb = require_array("model.layers.21.self_attn.v_proj.biases");
    const ResidentArrayRecord* qa = require_array("model.layers.21.self_attn.q_proj.lora_a");
    const ResidentArrayRecord* qlb = require_array("model.layers.21.self_attn.q_proj.lora_b");
    const ResidentArrayRecord* ka = require_array("model.layers.21.self_attn.k_proj.lora_a");
    const ResidentArrayRecord* klb = require_array("model.layers.21.self_attn.k_proj.lora_b");
    const ResidentArrayRecord* va = require_array("model.layers.21.self_attn.v_proj.lora_a");
    const ResidentArrayRecord* vlb = require_array("model.layers.21.self_attn.v_proj.lora_b");
    if (input_norm == nullptr || q_norm == nullptr || k_norm == nullptr || qw == nullptr || qs == nullptr || qb == nullptr ||
        kw == nullptr || ks == nullptr || kb == nullptr || vw == nullptr || vs == nullptr || vb == nullptr ||
        qa == nullptr || qlb == nullptr || ka == nullptr || klb == nullptr || va == nullptr || vlb == nullptr) {
        Napi::Error::New(env, "Selected layer21 input/qkv arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    constexpr int hidden_size = 2560;
    constexpr int q_output_size = 4096;
    constexpr int kv_output_size = 1024;
    constexpr int q_heads = 32;
    constexpr int kv_heads = 8;
    constexpr int head_dim = 128;
    constexpr int group_size = 64;
    constexpr int bits = 4;
    constexpr int rope_position = 8;
    constexpr float adapter_scale = 20.0f;
    constexpr float eps = 1.0e-6f;
    constexpr float rope_theta = 5000000.0f;
    constexpr float rope_scale = 1.0f;
    constexpr bool rope_traditional = false;

    std::vector<float> layer20_output(static_cast<size_t>(hidden_size));
    for (int i = 0; i < hidden_size; ++i) {
        layer20_output[static_cast<size_t>(i)] = static_cast<float>((i % 83) - 41) / 83.0f;
    }

    auto start = std::chrono::steady_clock::now();
    mlx::core::array input_array(layer20_output.begin(), mlx::core::Shape{1, hidden_size}, mlx::core::float32);
    mlx::core::array normalized = mlx::core::astype(
        mlx::core::fast::rms_norm(input_array, std::make_optional(input_norm->array), eps),
        mlx::core::float32);
    auto project = [&](const ResidentArrayRecord* w, const ResidentArrayRecord* s, const ResidentArrayRecord* b,
                       const ResidentArrayRecord* la, const ResidentArrayRecord* lb) {
        mlx::core::array base = mlx::core::astype(
            mlx::core::quantized_matmul(
                normalized,
                w->array,
                s->array,
                std::make_optional(b->array),
                true,
                group_size,
                bits,
                "affine"),
            mlx::core::float32);
        mlx::core::array delta = mlx::core::astype(
            adapter_scale * mlx::core::matmul(mlx::core::matmul(normalized, la->array), lb->array),
            mlx::core::float32);
        return mlx::core::astype(base + delta, mlx::core::float32);
    };
    mlx::core::array q_combined = project(qw, qs, qb, qa, qlb);
    mlx::core::array k_combined = project(kw, ks, kb, ka, klb);
    mlx::core::array v_combined = project(vw, vs, vb, va, vlb);
    mlx::core::array q_normed = mlx::core::astype(
        mlx::core::fast::rms_norm(mlx::core::reshape(q_combined, mlx::core::Shape{q_heads, head_dim}), std::make_optional(q_norm->array), eps),
        mlx::core::float32);
    mlx::core::array k_normed = mlx::core::astype(
        mlx::core::fast::rms_norm(mlx::core::reshape(k_combined, mlx::core::Shape{kv_heads, head_dim}), std::make_optional(k_norm->array), eps),
        mlx::core::float32);
    mlx::core::array q_rope = mlx::core::flatten(mlx::core::astype(
        mlx::core::fast::rope(
            mlx::core::reshape(q_normed, mlx::core::Shape{1, q_heads, 1, head_dim}),
            head_dim,
            rope_traditional,
            std::make_optional(rope_theta),
            rope_scale,
            rope_position),
        mlx::core::float32));
    mlx::core::array k_rope = mlx::core::flatten(mlx::core::astype(
        mlx::core::fast::rope(
            mlx::core::reshape(k_normed, mlx::core::Shape{1, kv_heads, 1, head_dim}),
            head_dim,
            rope_traditional,
            std::make_optional(rope_theta),
            rope_scale,
            rope_position),
        mlx::core::float32));
    mlx::core::array v_flat = mlx::core::flatten(mlx::core::astype(v_combined, mlx::core::float32));
    mlx::core::eval(normalized, q_rope, k_rope, v_flat);
    auto end = std::chrono::steady_clock::now();

    auto summarize = [&](const std::string& name, const mlx::core::array& arr, int expected_len) {
        const float* values = arr.data<float>();
        double checksum = 0.0;
        double abs_checksum = 0.0;
        for (int i = 0; i < expected_len; ++i) {
            const double v = static_cast<double>(values[i]);
            checksum += v;
            abs_checksum += std::abs(v);
        }
        Napi::Array first = Napi::Array::New(env, 8);
        for (int i = 0; i < 8; ++i) {
            first.Set(static_cast<uint32_t>(i), Napi::Number::New(env, values[i]));
        }
        Napi::Object o = Napi::Object::New(env);
        o.Set("name", Napi::String::New(env, name));
        o.Set("output_len", Napi::Number::New(env, static_cast<double>(arr.size())));
        o.Set("expected_output_len", Napi::Number::New(env, expected_len));
        o.Set("checksum", Napi::Number::New(env, checksum));
        o.Set("abs_checksum", Napi::Number::New(env, abs_checksum));
        o.Set("first_values", first);
        o.Set("backend", Napi::String::New(env, "mlx"));
        return o;
    };

    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_layer21_input_qkv_probe_version", Napi::String::New(env, "gypsy-selected-layer21-input-qkv-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_inputs_are_deterministic", Napi::Boolean::New(env, true));
    out.Set("probe_scope", Napi::String::New(env, "selected_resident_layer21_input_norm_qkv"));
    out.Set("source_layer", Napi::Number::New(env, 20));
    out.Set("target_layer", Napi::Number::New(env, 21));
    out.Set("input_contract", Napi::String::New(env, "layer20_output_float32_1x2560"));
    out.Set("input_norm_applied", Napi::Boolean::New(env, true));
    out.Set("adapter_delta_applied_before_qk_norm", Napi::Boolean::New(env, true));
    out.Set("q_norm_applied_before_rope", Napi::Boolean::New(env, true));
    out.Set("k_norm_applied_before_rope", Napi::Boolean::New(env, true));
    out.Set("rope_applied_to_q", Napi::Boolean::New(env, true));
    out.Set("rope_applied_to_k", Napi::Boolean::New(env, true));
    out.Set("v_projection_has_no_norm_or_rope", Napi::Boolean::New(env, true));
    out.Set("q_heads", Napi::Number::New(env, q_heads));
    out.Set("kv_heads", Napi::Number::New(env, kv_heads));
    out.Set("head_dim", Napi::Number::New(env, head_dim));
    out.Set("rope_position", Napi::Number::New(env, rope_position));
    out.Set("timing_ms", Napi::Number::New(env, timing_ms));
    out.Set("normalized_input", summarize("layer21_normalized_input", normalized, hidden_size));
    out.Set("q_after_rope", summarize("layer21_q_after_rope", q_rope, q_output_size));
    out.Set("k_after_rope", summarize("layer21_k_after_rope", k_rope, kv_output_size));
    out.Set("v_projection", summarize("layer21_v_projection", v_flat, kv_output_size));
    out.Set("constructed_layer21_array_count", Napi::Number::New(env, static_cast<double>(arrays.size())));
    out.Set("constructed_for_probe_only", Napi::Boolean::New(env, true));
    out.Set("stored_on_session", Napi::Boolean::New(env, false));
    out.Set("session_selected_resident_array_count", Napi::Number::New(env, static_cast<double>(session.selected_resident_arrays.size())));
    out.Set("readback_count", Napi::Number::New(env, 4));
    Napi::Array reasons = Napi::Array::New(env, 4);
    reasons.Set(static_cast<uint32_t>(0), Napi::String::New(env, "layer21_input_rmsnorm_summary"));
    reasons.Set(static_cast<uint32_t>(1), Napi::String::New(env, "layer21_q_after_rope_summary"));
    reasons.Set(static_cast<uint32_t>(2), Napi::String::New(env, "layer21_k_after_rope_summary"));
    reasons.Set(static_cast<uint32_t>(3), Napi::String::New(env, "layer21_v_projection_summary"));
    out.Set("readback_reasons", reasons);
    out.Set("next_execution_step", Napi::String::New(env, "add_selected_layer21_attention_probe"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value RunSelectedFinalLogitsProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto session_it = gypsy_sessions.find(handle);
    if (session_it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = session_it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before selected final logits probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (session.selected_resident_arrays.empty()) {
        Napi::Error::New(env, "Selected resident arrays must be constructed before selected final logits probe can run").ThrowAsJavaScriptException();
        return env.Null();
    }
    auto model_it = gypsy_models.find(session.model_handle);
    if (model_it == gypsy_models.end()) {
        Napi::Error::New(env, "Session model handle is no longer loaded").ThrowAsJavaScriptException();
        return env.Null();
    }

    const std::vector<std::string> names = {
        "model.norm.weight",
        "model.embed_tokens.weight",
        "model.embed_tokens.scales",
        "model.embed_tokens.biases",
    };
    std::vector<ResidentArrayRecord> arrays;
    std::string error;
    for (const auto& name : names) {
        const TensorDescriptor* d = FindTensorDescriptor(model_it->second.tensor_descriptors, name);
        if (d == nullptr) {
            Napi::Error::New(env, "Missing selected final logits model tensor: " + name).ThrowAsJavaScriptException();
            return env.Null();
        }
        auto record = ConstructMappedMlxArray(*d, session.model_mapped_files, error);
        if (!record) {
            Napi::Error::New(env, "Failed to construct selected final logits MLX array: " + error).ThrowAsJavaScriptException();
            return env.Null();
        }
        arrays.push_back(std::move(*record));
    }

    auto require_array = [&](const std::string& name) -> const ResidentArrayRecord* {
        return FindResidentArray(arrays, name);
    };
    const ResidentArrayRecord* final_norm = require_array("model.norm.weight");
    const ResidentArrayRecord* ew = require_array("model.embed_tokens.weight");
    const ResidentArrayRecord* es = require_array("model.embed_tokens.scales");
    const ResidentArrayRecord* eb = require_array("model.embed_tokens.biases");
    if (final_norm == nullptr || ew == nullptr || es == nullptr || eb == nullptr) {
        Napi::Error::New(env, "Selected final logits arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    constexpr int hidden_size = 2560;
    constexpr int vocab_size = 151936;
    constexpr int group_size = 64;
    constexpr int bits = 4;
    constexpr float eps = 1.0e-6f;

    std::vector<float> hidden(static_cast<size_t>(hidden_size));
    for (int i = 0; i < hidden_size; ++i) {
        hidden[static_cast<size_t>(i)] = static_cast<float>((i % 89) - 44) / 89.0f;
    }

    auto start = std::chrono::steady_clock::now();
    mlx::core::array hidden_array(hidden.begin(), mlx::core::Shape{1, hidden_size}, mlx::core::float32);
    mlx::core::array normalized = mlx::core::astype(
        mlx::core::fast::rms_norm(hidden_array, std::make_optional(final_norm->array), eps),
        mlx::core::float32);
    mlx::core::array logits = mlx::core::flatten(mlx::core::astype(
        mlx::core::quantized_matmul(
            normalized,
            ew->array,
            es->array,
            std::make_optional(eb->array),
            true,
            group_size,
            bits,
            "affine"),
        mlx::core::float32));
    mlx::core::eval(normalized, logits);
    auto end = std::chrono::steady_clock::now();

    auto summarize = [&](const std::string& name, const mlx::core::array& arr, int expected_len) {
        const float* values = arr.data<float>();
        double checksum = 0.0;
        double abs_checksum = 0.0;
        for (int i = 0; i < expected_len; ++i) {
            const double v = static_cast<double>(values[i]);
            checksum += v;
            abs_checksum += std::abs(v);
        }
        Napi::Array first = Napi::Array::New(env, 8);
        for (int i = 0; i < 8; ++i) {
            first.Set(static_cast<uint32_t>(i), Napi::Number::New(env, values[i]));
        }
        Napi::Object o = Napi::Object::New(env);
        o.Set("name", Napi::String::New(env, name));
        o.Set("output_len", Napi::Number::New(env, static_cast<double>(arr.size())));
        o.Set("expected_output_len", Napi::Number::New(env, expected_len));
        o.Set("checksum", Napi::Number::New(env, checksum));
        o.Set("abs_checksum", Napi::Number::New(env, abs_checksum));
        o.Set("first_values", first);
        o.Set("backend", Napi::String::New(env, "mlx"));
        return o;
    };

    const float* values = logits.data<float>();
    std::vector<std::pair<float, int>> top;
    top.reserve(10);
    double logits_checksum = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
        const float v = values[i];
        logits_checksum += static_cast<double>(v);
        if (top.size() < 10) {
            top.emplace_back(v, i);
            std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        } else if (v > top.back().first) {
            top.back() = {v, i};
            std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        }
    }

    Napi::Array top_tokens = Napi::Array::New(env, top.size());
    for (size_t i = 0; i < top.size(); ++i) {
        Napi::Object item = Napi::Object::New(env);
        item.Set("rank", Napi::Number::New(env, static_cast<double>(i + 1)));
        item.Set("token_id", Napi::Number::New(env, top[i].second));
        item.Set("score", Napi::Number::New(env, top[i].first));
        top_tokens.Set(static_cast<uint32_t>(i), item);
    }

    const double timing_ms = std::chrono::duration<double, std::milli>(end - start).count();
    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("selected_final_logits_probe_version", Napi::String::New(env, "gypsy-selected-final-logits-probe/1"));
    out.Set("math_executed", Napi::Boolean::New(env, true));
    out.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("full_logits_returned_to_coffeescript", Napi::Boolean::New(env, false));
    out.Set("probe_inputs_are_deterministic", Napi::Boolean::New(env, true));
    out.Set("probe_scope", Napi::String::New(env, "selected_final_norm_tied_embedding_logits"));
    out.Set("input_contract", Napi::String::New(env, "layer_output_float32_1x2560"));
    out.Set("final_norm_applied", Napi::Boolean::New(env, true));
    out.Set("logits_projection", Napi::String::New(env, "tied_model_embed_tokens_quantized_matmul"));
    out.Set("logits_len", Napi::Number::New(env, vocab_size));
    out.Set("top_token_id", Napi::Number::New(env, top.empty() ? -1 : top[0].second));
    out.Set("top_token_score", Napi::Number::New(env, top.empty() ? 0.0f : top[0].first));
    out.Set("top_tokens", top_tokens);
    out.Set("logits_checksum", Napi::Number::New(env, logits_checksum));
    out.Set("timing_ms", Napi::Number::New(env, timing_ms));
    out.Set("normalized_hidden", summarize("final_normalized_hidden", normalized, hidden_size));
    out.Set("constructed_for_probe_only", Napi::Boolean::New(env, true));
    out.Set("stored_on_session", Napi::Boolean::New(env, false));
    out.Set("readback_count", Napi::Number::New(env, 2));
    Napi::Array reasons = Napi::Array::New(env, 2);
    reasons.Set(static_cast<uint32_t>(0), Napi::String::New(env, "final_norm_summary"));
    reasons.Set(static_cast<uint32_t>(1), Napi::String::New(env, "logits_top10_summary"));
    out.Set("readback_reasons", reasons);
    out.Set("next_execution_step", Napi::String::New(env, "wire_logits_probe_into_generate_scaffold"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value ProtocolStatus(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Array models = Napi::Array::New(env, gypsy_models.size());
    size_t model_index = 0;
    for (const auto& kv : gypsy_models) {
        const auto& model = kv.second;
        Napi::Object item = Napi::Object::New(env);
        item.Set("handle", Napi::String::New(env, model.handle));
        item.Set("model_dir", Napi::String::New(env, model.model_dir));
        item.Set("payload_loaded", Napi::Boolean::New(env, false));
        item.Set("resident_loaded", Napi::Boolean::New(env, false));
        item.Set("descriptor_index_cached", Napi::Boolean::New(env, !model.tensor_descriptors.empty()));
        item.Set("descriptor_count", Napi::Number::New(env, static_cast<double>(model.tensor_descriptors.size())));
        models.Set(static_cast<uint32_t>(model_index++), item);
    }

    Napi::Array tokenizers = Napi::Array::New(env, gypsy_tokenizers.size());
    size_t tokenizer_index = 0;
    for (const auto& kv : gypsy_tokenizers) {
        const auto& tokenizer = kv.second;
        Napi::Object item = Napi::Object::New(env);
        item.Set("handle", Napi::String::New(env, tokenizer.handle));
        item.Set("tokenizer_dir", Napi::String::New(env, tokenizer.tokenizer_dir));
        item.Set("payload_loaded", Napi::Boolean::New(env, false));
        tokenizers.Set(static_cast<uint32_t>(tokenizer_index++), item);
    }

    Napi::Array adapters = Napi::Array::New(env, gypsy_adapters.size());
    size_t adapter_index = 0;
    for (const auto& kv : gypsy_adapters) {
        const auto& adapter = kv.second;
        Napi::Object item = Napi::Object::New(env);
        item.Set("handle", Napi::String::New(env, adapter.handle));
        item.Set("adapter_dir", Napi::String::New(env, adapter.adapter_dir));
        item.Set("payload_loaded", Napi::Boolean::New(env, false));
        item.Set("resident_loaded", Napi::Boolean::New(env, false));
        item.Set("descriptor_index_cached", Napi::Boolean::New(env, !adapter.tensor_descriptors.empty()));
        item.Set("descriptor_count", Napi::Number::New(env, static_cast<double>(adapter.tensor_descriptors.size())));
        adapters.Set(static_cast<uint32_t>(adapter_index++), item);
    }

    Napi::Array sessions = Napi::Array::New(env, gypsy_sessions.size());
    size_t session_index = 0;
    std::uint64_t dangling_reference_count = 0;
    std::uint64_t ready_session_count = 0;
    std::uint64_t mapped_file_count = 0;
    std::uint64_t mapped_file_bytes = 0;
    std::uint64_t mapped_payload_bytes = 0;
    std::vector<std::string> dangling_sessions;
    std::vector<std::string> ready_sessions;

    for (const auto& kv : gypsy_sessions) {
        const auto& session = kv.second;
        const bool model_live = gypsy_models.find(session.model_handle) != gypsy_models.end();
        const bool tokenizer_live = gypsy_tokenizers.find(session.tokenizer_handle) != gypsy_tokenizers.end();
        const bool adapter_live = session.adapter_handle.empty() || gypsy_adapters.find(session.adapter_handle) != gypsy_adapters.end();
        const bool references_live = model_live && tokenizer_live && adapter_live;
        const bool ready = session.warmed && references_live;
        if (!references_live) {
            ++dangling_reference_count;
            dangling_sessions.push_back(session.handle);
        }
        if (ready) {
            ++ready_session_count;
            ready_sessions.push_back(session.handle);
        }
        const std::uint64_t session_mapped_file_count = session.model_mapped_files.size() + session.adapter_mapped_files.size();
        const std::uint64_t session_mapped_file_bytes =
            MappedFilesByteTotal(session.model_mapped_files) + MappedFilesByteTotal(session.adapter_mapped_files);
        const std::uint64_t session_mapped_payload_bytes =
            MappedFilesPayloadByteTotal(session.model_mapped_files) + MappedFilesPayloadByteTotal(session.adapter_mapped_files);
        mapped_file_count += session_mapped_file_count;
        mapped_file_bytes += session_mapped_file_bytes;
        mapped_payload_bytes += session_mapped_payload_bytes;

        Napi::Object item = Napi::Object::New(env);
        item.Set("handle", Napi::String::New(env, session.handle));
        item.Set("payload_loaded", Napi::Boolean::New(env, false));
        item.Set("resident_arrays_constructed", Napi::Boolean::New(env, false));
        item.Set("warmed", Napi::Boolean::New(env, session.warmed));
        item.Set("model", Napi::String::New(env, session.model_handle));
        item.Set("tokenizer", Napi::String::New(env, session.tokenizer_handle));
        item.Set("adapter", session.adapter_handle.empty() ? env.Null() : Napi::String::New(env, session.adapter_handle).As<Napi::Value>());
        item.Set("model_live", Napi::Boolean::New(env, model_live));
        item.Set("tokenizer_live", Napi::Boolean::New(env, tokenizer_live));
        item.Set("adapter_live", Napi::Boolean::New(env, adapter_live));
        item.Set("references_live", Napi::Boolean::New(env, references_live));
        item.Set("ready_for_generate", Napi::Boolean::New(env, ready));
        item.Set("mapped_file_count", Napi::Number::New(env, static_cast<double>(session_mapped_file_count)));
        item.Set("mapped_file_bytes", Napi::Number::New(env, static_cast<double>(session_mapped_file_bytes)));
        item.Set("mapped_payload_bytes", Napi::Number::New(env, static_cast<double>(session_mapped_payload_bytes)));
        item.Set("model_mapped_file_count", Napi::Number::New(env, static_cast<double>(session.model_mapped_files.size())));
        item.Set("adapter_mapped_file_count", Napi::Number::New(env, static_cast<double>(session.adapter_mapped_files.size())));
        item.Set("selected_resident_array_count", Napi::Number::New(env, static_cast<double>(session.selected_resident_arrays.size())));
        item.Set("selected_resident_array_bytes", Napi::Number::New(env, static_cast<double>(ResidentArraysByteTotal(session.selected_resident_arrays))));
        item.Set("execution_owner", Napi::String::New(env, "native"));
        item.Set("generation_loop_owner", Napi::String::New(env, "native"));
        item.Set("coffeescript_tensor_payload_allowed", Napi::Boolean::New(env, false));
        sessions.Set(static_cast<uint32_t>(session_index++), item);
    }

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, dangling_reference_count == 0));
    out.Set("status_type", Napi::String::New(env, "gypsy_protocol_status"));
    out.Set("payload_loaded_anywhere", Napi::Boolean::New(env, false));
    out.Set("resident_arrays_constructed_anywhere", Napi::Boolean::New(env, false));
    out.Set("mapped_file_count", Napi::Number::New(env, static_cast<double>(mapped_file_count)));
    out.Set("mapped_file_bytes", Napi::Number::New(env, static_cast<double>(mapped_file_bytes)));
    out.Set("mapped_payload_bytes", Napi::Number::New(env, static_cast<double>(mapped_payload_bytes)));
    out.Set("legacy_loader_enabled", Napi::Boolean::New(env, EnvFlagEnabled("GYPSY_ALLOW_LEGACY_LOAD_MODEL")));
    out.Set("execution_owner", Napi::String::New(env, "native"));
    out.Set("generation_loop_owner", Napi::String::New(env, "native"));
    out.Set("coffeescript_tensor_payload_allowed", Napi::Boolean::New(env, false));
    out.Set("handle_counts", GypsyHandleCounts(env));
    out.Set("ready_session_count", Napi::Number::New(env, static_cast<double>(ready_session_count)));
    out.Set("ready_sessions", StringVectorToNapiArray(env, ready_sessions));
    out.Set("dangling_session_reference_count", Napi::Number::New(env, static_cast<double>(dangling_reference_count)));
    out.Set("dangling_sessions", StringVectorToNapiArray(env, dangling_sessions));
    out.Set("models", models);
    out.Set("tokenizers", tokenizers);
    out.Set("adapters", adapters);
    out.Set("sessions", sessions);
    return out;
}

Napi::Value GenerateProtocol(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto it = gypsy_sessions.find(handle);
    if (it == gypsy_sessions.end()) {
        Napi::Error::New(env, "Unknown session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const GypsySessionRecord& session = it->second;
    if (!session.warmed) {
        Napi::Error::New(env, "Session must be warmed before generate").ThrowAsJavaScriptException();
        return env.Null();
    }
    auto model_it = gypsy_models.find(session.model_handle);
    auto tokenizer_it = gypsy_tokenizers.find(session.tokenizer_handle);
    if (model_it == gypsy_models.end() || tokenizer_it == gypsy_tokenizers.end()) {
        Napi::Error::New(env, "Session references unloaded model or tokenizer").ThrowAsJavaScriptException();
        return env.Null();
    }
    GypsyAdapterRecord* adapter = nullptr;
    if (!session.adapter_handle.empty()) {
        auto adapter_it = gypsy_adapters.find(session.adapter_handle);
        if (adapter_it == gypsy_adapters.end()) {
            Napi::Error::New(env, "Session references unloaded adapter").ThrowAsJavaScriptException();
            return env.Null();
        }
        adapter = &adapter_it->second;
    }

    std::string prompt_text;
    std::uint32_t prompt_token_count = 0;
    std::vector<std::uint32_t> prompt_tokens;
    bool prompt_text_provided = false;
    bool prompt_tokens_provided = false;
    bool native_text_tokenization_executed = false;
    std::string native_text_tokenization_mode = "not_applicable";
    double max_tokens = 0;
    double temperature = 0;
    double top_k = 0;
    double top_p = 1.0;
    double seed = 0;
    bool chat = false;

    if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object request = info[1].As<Napi::Object>();
        if (request.Has("prompt") && request.Get("prompt").IsString()) {
            prompt_text = request.Get("prompt").As<Napi::String>();
            prompt_text_provided = true;
        }
        if (request.Has("prompt_tokens") && request.Get("prompt_tokens").IsArray()) {
            prompt_tokens_provided = true;
            Napi::Array tokens = request.Get("prompt_tokens").As<Napi::Array>();
            prompt_token_count = tokens.Length();
            prompt_tokens.reserve(prompt_token_count);
            for (std::uint32_t i = 0; i < prompt_token_count; ++i) {
                Napi::Value value = tokens.Get(i);
                if (!value.IsNumber()) {
                    Napi::TypeError::New(env, "prompt_tokens must contain only numbers").ThrowAsJavaScriptException();
                    return env.Null();
                }
                const double token_value = value.As<Napi::Number>().DoubleValue();
                if (token_value < 0 || token_value >= 151936) {
                    Napi::RangeError::New(env, "prompt token id out of range").ThrowAsJavaScriptException();
                    return env.Null();
                }
                prompt_tokens.push_back(static_cast<std::uint32_t>(token_value));
            }
        }
        if (request.Has("max_tokens") && request.Get("max_tokens").IsNumber()) {
            max_tokens = request.Get("max_tokens").As<Napi::Number>().DoubleValue();
        }
        if (request.Has("temperature") && request.Get("temperature").IsNumber()) {
            temperature = request.Get("temperature").As<Napi::Number>().DoubleValue();
        }
        if (request.Has("top_k") && request.Get("top_k").IsNumber()) {
            top_k = request.Get("top_k").As<Napi::Number>().DoubleValue();
        }
        if (request.Has("top_p") && request.Get("top_p").IsNumber()) {
            top_p = request.Get("top_p").As<Napi::Number>().DoubleValue();
        }
        if (request.Has("seed") && request.Get("seed").IsNumber()) {
            seed = request.Get("seed").As<Napi::Number>().DoubleValue();
        }
        if (request.Has("chat") && request.Get("chat").IsBoolean()) {
            chat = request.Get("chat").As<Napi::Boolean>().Value();
        }
    }

    if (!prompt_text_provided && !prompt_tokens_provided) {
        Napi::TypeError::New(env, "generate request requires prompt or prompt_tokens").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (max_tokens < 0) {
        Napi::TypeError::New(env, "max_tokens must be non-negative").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (temperature < 0) {
        Napi::TypeError::New(env, "temperature must be non-negative").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (top_k < 0) {
        Napi::TypeError::New(env, "top_k must be non-negative").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (top_p <= 0 || top_p > 1.0) {
        Napi::TypeError::New(env, "top_p must be in (0, 1]").ThrowAsJavaScriptException();
        return env.Null();
    }

    if (prompt_text_provided && !prompt_tokens_provided) {
        const auto& tokenizer_record = tokenizer_it->second;
        auto encode_text = [&](const std::string& text) {
            return GreedyEncodeWithVocab(text, tokenizer_record.token_to_id);
        };
        if (!tokenizer_record.token_to_id.empty()) {
            if (chat) {
                auto get_token = [&](const std::string& piece, std::uint32_t fallback) {
                    auto it = tokenizer_record.token_to_id.find(piece);
                    return it == tokenizer_record.token_to_id.end() ? fallback : it->second;
                };
                prompt_tokens = {
                    get_token("<|im_start|>", 151644),
                    get_token("user", 872),
                    get_token("Ċ", 198)
                };
                auto text_tokens = encode_text(prompt_text);
                prompt_tokens.insert(prompt_tokens.end(), text_tokens.begin(), text_tokens.end());
                prompt_tokens.push_back(get_token("<|im_end|>", 151645));
                prompt_tokens.push_back(get_token("Ċ", 198));
                prompt_tokens.push_back(get_token("<|im_start|>", 151644));
                prompt_tokens.push_back(get_token("assistant", 77091));
                prompt_tokens.push_back(get_token("Ċ", 198));
                native_text_tokenization_mode = "qwen_greedy_vocab_chat";
            } else {
                prompt_tokens = encode_text(prompt_text);
                native_text_tokenization_mode = "qwen_greedy_vocab_raw";
            }
            prompt_token_count = static_cast<std::uint32_t>(prompt_tokens.size());
            native_text_tokenization_executed = !prompt_tokens.empty();
        } else {
            native_text_tokenization_mode = "unsupported_text_synthetic_fallback";
        }
    }

    std::uint64_t prompt_signature = 1469598103934665603ULL;
    auto mix_signature = [&](std::uint64_t value) {
        prompt_signature ^= value;
        prompt_signature *= 1099511628211ULL;
    };
    if (!prompt_tokens.empty()) {
        for (std::uint32_t token : prompt_tokens) {
            mix_signature(static_cast<std::uint64_t>(token) + 0x9e3779b97f4a7c15ULL);
        }
    } else {
        for (unsigned char c : prompt_text) {
            mix_signature(static_cast<std::uint64_t>(c));
        }
        mix_signature(chat ? 0x43484154ULL : 0x524157ULL);
    }

    Napi::Object request_summary = Napi::Object::New(env);
    request_summary.Set("prompt_text_provided", Napi::Boolean::New(env, prompt_text_provided));
    request_summary.Set("prompt_text_bytes", Napi::Number::New(env, static_cast<double>(prompt_text.size())));
    request_summary.Set("prompt_tokens_provided", Napi::Boolean::New(env, prompt_tokens_provided));
    request_summary.Set("prompt_token_count", Napi::Number::New(env, prompt_token_count));
    request_summary.Set("native_text_tokenization_executed", Napi::Boolean::New(env, native_text_tokenization_executed));
    request_summary.Set("native_text_tokenization_mode", Napi::String::New(env, native_text_tokenization_mode));
    request_summary.Set("prompt_signature", Napi::Number::New(env, static_cast<double>(prompt_signature)));
    request_summary.Set("max_tokens", Napi::Number::New(env, max_tokens));
    request_summary.Set("temperature", Napi::Number::New(env, temperature));
    request_summary.Set("top_k", Napi::Number::New(env, top_k));
    request_summary.Set("top_p", Napi::Number::New(env, top_p));
    request_summary.Set("seed", Napi::Number::New(env, seed));
    request_summary.Set("chat", Napi::Boolean::New(env, chat));

    Napi::Array phases = Napi::Array::New(env, 8);
    const std::vector<std::string> phase_names = {
        "format_or_tokenize_prompt",
        "full_prompt_prefill",
        "native_decode_loop",
        "native_kv_cache_update",
        "adapter_delta_application",
        "logits_projection",
        "native_sampling_or_greedy_selection",
        "eos_stop_cleanup_decode"
    };
    for (size_t i = 0; i < phase_names.size(); ++i) {
        phases.Set(static_cast<uint32_t>(i), Napi::String::New(env, phase_names[i]));
    }

    Napi::Object sampling = Napi::Object::New(env);
    sampling.Set("mode", Napi::String::New(env, temperature == 0 ? "greedy" : "sampled"));
    sampling.Set("temperature", Napi::Number::New(env, temperature));
    sampling.Set("top_k", Napi::Number::New(env, top_k));
    sampling.Set("top_p", Napi::Number::New(env, top_p));
    sampling.Set("seed", Napi::Number::New(env, seed));
    sampling.Set("owner", Napi::String::New(env, "native"));
    sampling.Set("full_logits_returned_to_coffeescript", Napi::Boolean::New(env, false));

    Napi::Object stop = Napi::Object::New(env);
    stop.Set("owner", Napi::String::New(env, "native"));
    stop.Set("eos_handled", Napi::Boolean::New(env, true));
    stop.Set("special_token_cleanup", Napi::Boolean::New(env, true));

    Napi::Object generation_plan = Napi::Object::New(env);
    generation_plan.Set("generation_plan_version", Napi::String::New(env, "gypsy-generation-plan/1"));
    generation_plan.Set("payload_loaded", Napi::Boolean::New(env, false));
    generation_plan.Set("request_validated", Napi::Boolean::New(env, true));
    generation_plan.Set("session_warmed", Napi::Boolean::New(env, session.warmed));
    generation_plan.Set("execution_owner", Napi::String::New(env, "native"));
    generation_plan.Set("generation_loop_owner", Napi::String::New(env, "native"));
    generation_plan.Set("coffeescript_drives_tokens", Napi::Boolean::New(env, false));
    generation_plan.Set("coffeescript_tensor_payload_allowed", Napi::Boolean::New(env, false));
    generation_plan.Set("prompt_source", Napi::String::New(env, prompt_tokens_provided ? "token_ids" : "text"));
    generation_plan.Set("chat_format_requested", Napi::Boolean::New(env, chat));
    generation_plan.Set("full_prompt_prefill_required", Napi::Boolean::New(env, true));
    generation_plan.Set("decode_loop_native", Napi::Boolean::New(env, true));
    generation_plan.Set("kv_cache_owner", Napi::String::New(env, "native_session"));
    generation_plan.Set("adapter_attached", Napi::Boolean::New(env, adapter != nullptr));
    generation_plan.Set("adapter_applied_in_prefill", Napi::Boolean::New(env, adapter != nullptr));
    generation_plan.Set("adapter_applied_in_decode", Napi::Boolean::New(env, adapter != nullptr));
    generation_plan.Set("model_descriptor_count", Napi::Number::New(env, static_cast<double>(model_it->second.tensor_descriptors.size())));
    generation_plan.Set("adapter_descriptor_count", Napi::Number::New(env, adapter == nullptr ? 0 : static_cast<double>(adapter->tensor_descriptors.size())));
    generation_plan.Set("max_tokens", Napi::Number::New(env, max_tokens));
    generation_plan.Set("phases", phases);
    generation_plan.Set("sampling", sampling);
    generation_plan.Set("stop", stop);

    constexpr int hidden_size = 2560;
    constexpr int vocab_size = 151936;
    constexpr int group_size = 64;
    constexpr int bits = 4;
    constexpr float eps = 1.0e-6f;
    // Compute precision for layer activations. bfloat16 halves activation bandwidth
    // versus float32 and matches Qwen's training precision.
    const mlx::core::Dtype compute_dtype = mlx::core::bfloat16;
    const std::uint32_t requested_count = static_cast<std::uint32_t>(std::max<double>(max_tokens, 0));
    // Safety upper bound (~144 MB extra KV at bf16 on Qwen3-4B). The caller's
    // requested max_tokens is honored up to this ceiling; raise it deliberately
    // if a use case actually needs longer generations.
    constexpr std::uint32_t generation_token_ceiling = 4096;
    const std::uint32_t generated_count = std::min<std::uint32_t>(requested_count, generation_token_ceiling);

    std::vector<ResidentArrayRecord> logits_arrays;
    std::string logits_error;
    std::vector<std::string> final_names = {
        "model.norm.weight",
        "model.embed_tokens.weight",
        "model.embed_tokens.scales",
        "model.embed_tokens.biases",
    };
    auto add_layer_names = [&](int layer) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        const std::vector<std::string> suffixes = {
            "input_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.scales",
            "self_attn.q_proj.biases",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.scales",
            "self_attn.k_proj.biases",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.scales",
            "self_attn.v_proj.biases",
            "self_attn.o_proj.weight",
            "self_attn.o_proj.scales",
            "self_attn.o_proj.biases",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.gate_proj.scales",
            "mlp.gate_proj.biases",
            "mlp.up_proj.weight",
            "mlp.up_proj.scales",
            "mlp.up_proj.biases",
            "mlp.down_proj.weight",
            "mlp.down_proj.scales",
            "mlp.down_proj.biases",
        };
        for (const auto& suffix : suffixes) final_names.push_back(p + suffix);
    };
    add_layer_names(0);
    add_layer_names(1);
    add_layer_names(2);
    add_layer_names(3);
    add_layer_names(4);
    add_layer_names(5);
    add_layer_names(6);
    add_layer_names(7);
    for (int layer = 8; layer <= 35; ++layer) {
        add_layer_names(layer);
    }
    auto add_adapter_layer_names = [&](int layer) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        const std::vector<std::string> suffixes = {
            "self_attn.q_proj.lora_a",
            "self_attn.q_proj.lora_b",
            "self_attn.k_proj.lora_a",
            "self_attn.k_proj.lora_b",
            "self_attn.v_proj.lora_a",
            "self_attn.v_proj.lora_b",
            "self_attn.o_proj.lora_a",
            "self_attn.o_proj.lora_b",
            "mlp.gate_proj.lora_a",
            "mlp.gate_proj.lora_b",
            "mlp.up_proj.lora_a",
            "mlp.up_proj.lora_b",
            "mlp.down_proj.lora_a",
            "mlp.down_proj.lora_b",
        };
        for (const auto& suffix : suffixes) final_names.push_back(p + suffix);
    };
    if (adapter != nullptr) {
        for (int layer = 20; layer <= 35; ++layer) {
            add_adapter_layer_names(layer);
        }
    }
    for (const auto& name : final_names) {
        const bool adapter_tensor = name.find(".lora_") != std::string::npos;
        const TensorDescriptor* d = adapter_tensor
            ? (adapter == nullptr ? nullptr : FindTensorDescriptor(adapter->tensor_descriptors, name))
            : FindTensorDescriptor(model_it->second.tensor_descriptors, name);
        if (d == nullptr) {
            Napi::Error::New(env, std::string("Missing generate tensor: ") + name).ThrowAsJavaScriptException();
            return env.Null();
        }
        auto record = ConstructMappedMlxArray(*d, adapter_tensor ? session.adapter_mapped_files : session.model_mapped_files, logits_error);
        if (!record) {
            Napi::Error::New(env, "Failed to construct generate MLX array: " + logits_error).ThrowAsJavaScriptException();
            return env.Null();
        }
        logits_arrays.push_back(std::move(*record));
    }

    auto require_logits_array = [&](const std::string& name) -> const ResidentArrayRecord* {
        return FindResidentArray(logits_arrays, name);
    };
    const ResidentArrayRecord* final_norm = require_logits_array("model.norm.weight");
    const ResidentArrayRecord* ew = require_logits_array("model.embed_tokens.weight");
    const ResidentArrayRecord* es = require_logits_array("model.embed_tokens.scales");
    const ResidentArrayRecord* eb = require_logits_array("model.embed_tokens.biases");
    const ResidentArrayRecord* l0_input_norm = require_logits_array("model.layers.0.input_layernorm.weight");
    const ResidentArrayRecord* l0_q_norm = require_logits_array("model.layers.0.self_attn.q_norm.weight");
    const ResidentArrayRecord* l0_k_norm = require_logits_array("model.layers.0.self_attn.k_norm.weight");
    const ResidentArrayRecord* l0_qw = require_logits_array("model.layers.0.self_attn.q_proj.weight");
    const ResidentArrayRecord* l0_qs = require_logits_array("model.layers.0.self_attn.q_proj.scales");
    const ResidentArrayRecord* l0_qb = require_logits_array("model.layers.0.self_attn.q_proj.biases");
    const ResidentArrayRecord* l0_kw = require_logits_array("model.layers.0.self_attn.k_proj.weight");
    const ResidentArrayRecord* l0_ks = require_logits_array("model.layers.0.self_attn.k_proj.scales");
    const ResidentArrayRecord* l0_kb = require_logits_array("model.layers.0.self_attn.k_proj.biases");
    const ResidentArrayRecord* l0_vw = require_logits_array("model.layers.0.self_attn.v_proj.weight");
    const ResidentArrayRecord* l0_vs = require_logits_array("model.layers.0.self_attn.v_proj.scales");
    const ResidentArrayRecord* l0_vb = require_logits_array("model.layers.0.self_attn.v_proj.biases");
    const ResidentArrayRecord* l0_ow = require_logits_array("model.layers.0.self_attn.o_proj.weight");
    const ResidentArrayRecord* l0_os = require_logits_array("model.layers.0.self_attn.o_proj.scales");
    const ResidentArrayRecord* l0_ob = require_logits_array("model.layers.0.self_attn.o_proj.biases");
    const ResidentArrayRecord* l0_post_norm = require_logits_array("model.layers.0.post_attention_layernorm.weight");
    const ResidentArrayRecord* l0_gw = require_logits_array("model.layers.0.mlp.gate_proj.weight");
    const ResidentArrayRecord* l0_gs = require_logits_array("model.layers.0.mlp.gate_proj.scales");
    const ResidentArrayRecord* l0_gb = require_logits_array("model.layers.0.mlp.gate_proj.biases");
    const ResidentArrayRecord* l0_uw = require_logits_array("model.layers.0.mlp.up_proj.weight");
    const ResidentArrayRecord* l0_us = require_logits_array("model.layers.0.mlp.up_proj.scales");
    const ResidentArrayRecord* l0_ub = require_logits_array("model.layers.0.mlp.up_proj.biases");
    const ResidentArrayRecord* l0_dw = require_logits_array("model.layers.0.mlp.down_proj.weight");
    const ResidentArrayRecord* l0_ds = require_logits_array("model.layers.0.mlp.down_proj.scales");
    const ResidentArrayRecord* l0_db = require_logits_array("model.layers.0.mlp.down_proj.biases");
    if (final_norm == nullptr || ew == nullptr || es == nullptr || eb == nullptr) {
        Napi::Error::New(env, "Generate final logits arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (l0_input_norm == nullptr || l0_q_norm == nullptr || l0_k_norm == nullptr ||
        l0_qw == nullptr || l0_qs == nullptr || l0_qb == nullptr ||
        l0_kw == nullptr || l0_ks == nullptr || l0_kb == nullptr ||
        l0_vw == nullptr || l0_vs == nullptr || l0_vb == nullptr ||
        l0_ow == nullptr || l0_os == nullptr || l0_ob == nullptr ||
        l0_post_norm == nullptr || l0_gw == nullptr || l0_gs == nullptr || l0_gb == nullptr ||
        l0_uw == nullptr || l0_us == nullptr || l0_ub == nullptr ||
        l0_dw == nullptr || l0_ds == nullptr || l0_db == nullptr) {
        Napi::Error::New(env, "Generate layer0 arrays are incomplete").ThrowAsJavaScriptException();
        return env.Null();
    }

    int top_token_id = -1;
    float top_token_score = 0.0f;
    double logits_checksum = 0.0;
    double final_norm_checksum = 0.0;
    double final_norm_abs_checksum = 0.0;
    bool token_embedding_lookup_executed = false;
    bool layer0_block_executed = false;
    bool layer1_block_executed = false;
    bool prompt_token_prefill_loop_executed = false;
    std::uint32_t prompt_prefill_tokens_processed = 0;
    int generated_layer_count = 0;
    std::int64_t embedding_token_id = -1;
    std::vector<std::uint32_t> generated_tokens;
    std::vector<std::pair<float, int>> top;
    top.reserve(10);
    struct LocalLayerKvCache {
        std::vector<float> keys;
        std::vector<float> values;
        std::uint32_t len = 0;
        bool has_mlx_cache = false;
        std::optional<mlx::core::array> mlx_expanded_keys;
        std::optional<mlx::core::array> mlx_expanded_values;
    };
    std::vector<LocalLayerKvCache> local_kv_cache(36);
    bool kv_cache_materialized = false;
    std::uint32_t kv_cache_positions = 0;
    struct ActiveTimingBuckets {
        double embedding_ms = 0.0;
        double input_norm_ms = 0.0;
        double qkv_projection_ms = 0.0;
        double qk_norm_rope_ms = 0.0;
        double kv_readback_append_ms = 0.0;
        double attention_cpu_ms = 0.0;
        double mlx_kv_append_ms = 0.0;
        double attention_mlx_ms = 0.0;
        double o_projection_ms = 0.0;
        double fused_attention_o_ms = 0.0;
        double direct_metal_layer35_attention_o_ms = 0.0;
        double direct_metal_layer35_attention_o_checksum = 0.0;
        double direct_metal_layer35_qkv_attention_o_ms = 0.0;
        double direct_qkv_attention_o_q_checksum = 0.0;
        double direct_qkv_attention_o_k_checksum = 0.0;
        double direct_qkv_attention_o_v_checksum = 0.0;
        double direct_qkv_attention_o_attention_checksum = 0.0;
        double direct_qkv_attention_o_output_checksum = 0.0;
        double direct_metal_layer35_post_residual_ms = 0.0;
        double direct_metal_layer35_post_residual_checksum = 0.0;
        double post_attention_residual_norm_ms = 0.0;
        double direct_metal_layer35_post_norm_ms = 0.0;
        double direct_metal_layer35_post_norm_checksum = 0.0;
        double mlp_gate_up_ms = 0.0;
        double direct_metal_layer35_gate_up_ms = 0.0;
        double direct_metal_layer35_gate_checksum = 0.0;
        double direct_metal_layer35_up_checksum = 0.0;
        double mlp_activation_ms = 0.0;
        double mlp_down_residual_ms = 0.0;
        double direct_metal_layer35_activation_ms = 0.0;
        double direct_metal_layer35_activation_checksum = 0.0;
        double direct_metal_layer35_down_ms = 0.0;
        double direct_metal_layer35_down_checksum = 0.0;
        double final_norm_ms = 0.0;
        double direct_metal_final_norm_ms = 0.0;
        double direct_metal_final_norm_checksum = 0.0;
        double logits_topk_ms = 0.0;
        double direct_metal_logits_ms = 0.0;
        double direct_metal_logits_checksum = 0.0;
        double layer_total_ms = 0.0;
        double readback_ms = 0.0;
        double direct_metal_layer35_input_norm_ms = 0.0;
        double direct_metal_layer35_input_norm_checksum = 0.0;
        double direct_metal_fused_mlp_tail_ms = 0.0;
        double direct_metal_fused_mlp_tail_checksum = 0.0;
        double direct_metal_full_layer_ms = 0.0;
        double direct_metal_full_layer_checksum = 0.0;
        std::uint64_t qkv_projection_calls = 0;
        std::uint64_t quantized_projection_calls = 0;
        std::uint64_t attention_calls = 0;
        std::uint64_t fused_attention_o_calls = 0;
        std::uint64_t direct_metal_layer35_attention_o_calls = 0;
        std::uint64_t direct_metal_layer35_qkv_attention_o_calls = 0;
        std::uint64_t direct_qkv_attention_o_calls = 0;
        std::uint64_t direct_metal_layer35_post_residual_calls = 0;
        std::uint64_t direct_metal_layer35_post_norm_calls = 0;
        std::uint64_t direct_metal_layer35_gate_up_calls = 0;
        std::uint64_t direct_metal_logits_calls = 0;
        std::uint64_t direct_metal_final_norm_calls = 0;
        std::uint64_t direct_metal_layer35_activation_calls = 0;
        std::uint64_t direct_metal_layer35_down_calls = 0;
        std::uint64_t mlx_attention_calls = 0;
        std::uint64_t cpu_attention_calls = 0;
        std::uint64_t readback_count = 0;
        std::uint64_t tokens_evaluated = 0;
        std::uint64_t direct_metal_layer35_input_norm_calls = 0;
        std::uint64_t direct_metal_fused_mlp_tail_calls = 0;
        std::uint64_t direct_metal_full_layer_calls = 0;
    } timing_buckets;
    auto elapsed_ms_since = [](const std::chrono::steady_clock::time_point& start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    bool generation_stopped_by_token = false;
    std::uint32_t generation_stop_token_id = 0;
    std::string generation_stop_reason = "max_tokens";
    bool mlx_attention_used_anywhere = false;
    bool cpu_attention_used_anywhere = false;
    bool direct_metal_attention_used_anywhere = false;
    std::uint64_t adapter_applied_projection_count = 0;
    constexpr float generation_adapter_scale = 20.0f;
    const char* full_logit_summary_env = std::getenv("GYPSY_FULL_LOGIT_SUMMARY");
    const bool full_logit_summary = full_logit_summary_env != nullptr &&
        std::string(full_logit_summary_env) == "1";
    const char* attention_backend_env = std::getenv("GYPSY_ATTENTION_BACKEND");
    const std::string attention_backend_requested = attention_backend_env == nullptr
        ? "mlx_prealloc_kv"
        : std::string(attention_backend_env);
    {
        const bool metal_ok = mlx::core::metal::is_available();
        const auto& dev = mlx::core::default_device();
        const bool dev_gpu = dev.type == mlx::core::Device::DeviceType::gpu;
        fprintf(stderr, "[mlx-diag] metal_available=%s default_device=%s backend=%s\n",
            metal_ok ? "YES" : "NO",
            dev_gpu ? "GPU" : "CPU",
            attention_backend_requested.c_str());
        // Cap MLX's free-buffer pool at 512 MB so released intermediates (e.g. the
        // ~1.56 GB embedding dequantize temp) are returned to the OS rather than held.
        mlx::core::set_cache_limit(512ULL * 1024 * 1024);
    }
    const char* direct_metal_resident_kv_env = std::getenv("GYPSY_DIRECT_METAL_RESIDENT_KV");
    const bool direct_metal_resident_kv = direct_metal_resident_kv_env != nullptr &&
        std::string(direct_metal_resident_kv_env) == "1";
    const char* direct_metal_keep_cpu_kv_env = std::getenv("GYPSY_DIRECT_METAL_KEEP_CPU_KV");
    const bool direct_metal_keep_cpu_kv = !direct_metal_resident_kv ||
        (direct_metal_keep_cpu_kv_env != nullptr && std::string(direct_metal_keep_cpu_kv_env) == "1");
    const char* direct_metal_concat_qkv_env = std::getenv("GYPSY_DIRECT_METAL_CONCAT_QKV_READBACK");
    const bool direct_metal_concat_qkv_readback = direct_metal_concat_qkv_env != nullptr &&
        std::string(direct_metal_concat_qkv_env) == "1";
    const char* projection_backend_env = std::getenv("GYPSY_PROJECTION_BACKEND");
    const std::string projection_backend_requested = projection_backend_env == nullptr ? "" : std::string(projection_backend_env);
    const bool use_compiled_metal_qkv_projection = projection_backend_requested == "compiled_metal_qkv";
    const char* fused_attention_o_env = std::getenv("GYPSY_FUSED_ATTENTION_O");
    const bool use_fused_attention_o_projection = fused_attention_o_env != nullptr &&
        std::string(fused_attention_o_env) == "1";
    const char* fused_attention_o_residual_env = std::getenv("GYPSY_FUSED_ATTENTION_O_RESIDUAL");
    const bool use_fused_attention_o_residual = fused_attention_o_residual_env != nullptr &&
        std::string(fused_attention_o_residual_env) == "1";
    const char* direct_metal_qkv_attention_o_env = std::getenv("GYPSY_DIRECT_METAL_QKV_ATTENTION_O");
    const bool use_direct_metal_qkv_attention_o = direct_metal_qkv_attention_o_env != nullptr &&
        std::string(direct_metal_qkv_attention_o_env) == "1";
    const char* direct_metal_logits_env = std::getenv("GYPSY_DIRECT_METAL_LOGITS");
    const bool use_direct_metal_logits = direct_metal_logits_env != nullptr &&
        std::string(direct_metal_logits_env) == "1";
    const char* direct_metal_final_norm_env = std::getenv("GYPSY_DIRECT_METAL_FINAL_NORM");
    const bool use_direct_metal_final_norm = direct_metal_final_norm_env != nullptr &&
        std::string(direct_metal_final_norm_env) == "1";
    const char* direct_metal_layer35_down_env = std::getenv("GYPSY_DIRECT_METAL_LAYER35_DOWN");
    const bool use_direct_metal_layer35_down = direct_metal_layer35_down_env != nullptr &&
        std::string(direct_metal_layer35_down_env) == "1";
    const char* direct_metal_layer35_activation_env = std::getenv("GYPSY_DIRECT_METAL_LAYER35_ACTIVATION");
    const bool use_direct_metal_layer35_activation = direct_metal_layer35_activation_env != nullptr &&
        std::string(direct_metal_layer35_activation_env) == "1";
    const char* direct_metal_layer35_gate_up_env = std::getenv("GYPSY_DIRECT_METAL_LAYER35_GATE_UP");
    const bool use_direct_metal_layer35_gate_up = direct_metal_layer35_gate_up_env != nullptr &&
        std::string(direct_metal_layer35_gate_up_env) == "1";
    const char* direct_metal_layer35_post_norm_env = std::getenv("GYPSY_DIRECT_METAL_LAYER35_POST_NORM");
    const bool use_direct_metal_layer35_post_norm = direct_metal_layer35_post_norm_env != nullptr &&
        std::string(direct_metal_layer35_post_norm_env) == "1";
    const char* direct_metal_layer35_post_residual_env = std::getenv("GYPSY_DIRECT_METAL_LAYER35_POST_RESIDUAL");
    const bool use_direct_metal_layer35_post_residual = direct_metal_layer35_post_residual_env != nullptr &&
        std::string(direct_metal_layer35_post_residual_env) == "1";
    const char* direct_metal_layer35_attention_o_env = std::getenv("GYPSY_DIRECT_METAL_LAYER35_ATTENTION_O");
    const bool use_direct_metal_layer35_attention_o = direct_metal_layer35_attention_o_env != nullptr &&
        std::string(direct_metal_layer35_attention_o_env) == "1";
    const char* direct_metal_layer35_qkv_attention_o_env = std::getenv("GYPSY_DIRECT_METAL_LAYER35_QKV_ATTENTION_O");
    const bool use_direct_metal_layer35_qkv_attention_o = direct_metal_layer35_qkv_attention_o_env != nullptr &&
        std::string(direct_metal_layer35_qkv_attention_o_env) == "1";
    const char* direct_metal_layer35_input_norm_env = std::getenv("GYPSY_DIRECT_METAL_LAYER35_INPUT_NORM");
    const bool use_direct_metal_layer35_input_norm = direct_metal_layer35_input_norm_env != nullptr &&
        std::string(direct_metal_layer35_input_norm_env) == "1";
    const char* direct_metal_fused_mlp_tail_env = std::getenv("GYPSY_DIRECT_METAL_FUSED_MLP_TAIL");
    const bool use_direct_metal_fused_mlp_tail = direct_metal_fused_mlp_tail_env != nullptr &&
        std::string(direct_metal_fused_mlp_tail_env) == "1";
    const char* direct_metal_full_layer_env = std::getenv("GYPSY_DIRECT_METAL_FULL_LAYER");
    const bool use_direct_metal_full_layer = direct_metal_full_layer_env != nullptr &&
        std::string(direct_metal_full_layer_env) == "1";
    const char* direct_metal_terminal_start_env = std::getenv("GYPSY_DIRECT_METAL_TERMINAL_LAYER_START");
    int direct_metal_terminal_layer_start = 35;
    if (direct_metal_terminal_start_env != nullptr) {
        try {
            direct_metal_terminal_layer_start = std::clamp(std::stoi(direct_metal_terminal_start_env), 0, 35);
        } catch (...) {
            direct_metal_terminal_layer_start = 35;
        }
    }
    auto is_generation_stop_token = [](std::uint32_t token_id) {
        return token_id == 151645U || token_id == 151643U;
    };
    auto generation_start = std::chrono::steady_clock::now();
    if (generated_count > 0) {
        const std::uint32_t loop_limit = generated_count;
        const bool use_mlx_attention = attention_backend_requested == "mlx" ||
            attention_backend_requested == "mlx_expanded_kv" ||
            attention_backend_requested == "mlx_prealloc_kv" ||
            attention_backend_requested == "metal_kernel_attention";
        const bool use_direct_metal_attention = attention_backend_requested == "direct_metal_attention";
        const bool use_mlx_prealloc_kv = attention_backend_requested == "mlx_prealloc_kv";
        const bool use_custom_metal_attention = attention_backend_requested == "metal_kernel_attention";
        const bool use_preallocated_mlx_kv = use_mlx_prealloc_kv || use_custom_metal_attention;
        if (use_direct_metal_attention && direct_metal_resident_kv) {
            GypsyResetDirectMetalAttentionKvCache();
        }
        const int mlx_prealloc_kv_capacity = static_cast<int>(
            std::max<std::uint32_t>(1, static_cast<std::uint32_t>(prompt_tokens.size()) + loop_limit + 1));
        std::uint32_t current_token_id = !prompt_tokens.empty()
            ? prompt_tokens.back()
            : static_cast<std::uint32_t>(prompt_signature % vocab_size);
        auto run_one_token = [&](std::uint32_t input_token_id, std::uint32_t position, bool keep_top_summary) -> std::uint32_t {
            std::vector<std::pair<float, int>> local_top;
            local_top.reserve(10);
            double local_logits_checksum = 0.0;
            double local_final_norm_checksum = 0.0;
            double local_final_norm_abs_checksum = 0.0;
            float local_top_token_score = 0.0f;
            int local_top_token_id = -1;
            int local_generated_layer_count = 0;
            bool local_layer0_block_executed = false;
            bool local_layer1_block_executed = false;

            auto build_hidden_array = [&]() -> mlx::core::array {
            if (!prompt_tokens.empty()) {
                const auto embedding_start = std::chrono::steady_clock::now();
                embedding_token_id = static_cast<std::int64_t>(input_token_id);
                token_embedding_lookup_executed = true;
                int32_t tok_val = static_cast<int32_t>(embedding_token_id);
                // Custom kernel: dequantize only the one needed row (avoids materializing
                // the full [vocab=151936, hidden=2560] float32 intermediate = 1.56 GB).
                static const std::string emb_source = R"METAL(
                    uint j = thread_position_in_grid.x;
                    if (j >= HIDDEN_SIZE) return;
                    constexpr uint GROUP_SIZE = 64;
                    constexpr uint VALUES_PER_WORD = 8;
                    constexpr uint WORDS_PER_GROUP = GROUP_SIZE / VALUES_PER_WORD;
                    constexpr uint PACKED_COLS = HIDDEN_SIZE / VALUES_PER_WORD;
                    constexpr uint SCALE_COLS = HIDDEN_SIZE / GROUP_SIZE;
                    uint row = tok_id[0];
                    uint g = j / GROUP_SIZE;
                    uint within_g = j % GROUP_SIZE;
                    uint word_idx = within_g / VALUES_PER_WORD;
                    uint nibble_idx = within_g % VALUES_PER_WORD;
                    uint word = weight[row * PACKED_COLS + g * WORDS_PER_GROUP + word_idx];
                    uint nibble = (word >> (nibble_idx * 4)) & 0xFu;
                    float scale = float(scales[row * SCALE_COLS + g]);
                    float bias = float(biases[row * SCALE_COLS + g]);
                    out[j] = float(nibble) * scale + bias;
                )METAL";
                static auto emb_kernel = mlx::core::fast::metal_kernel(
                    "gypsy_embedding_lookup",
                    {"tok_id", "weight", "scales", "biases"},
                    {"out"},
                    emb_source, "", true, false);
                mlx::core::array tok_id_arr(&tok_val, mlx::core::Shape{1}, mlx::core::int32);
                std::vector<mlx::core::array> emb_outputs = emb_kernel(
                    {tok_id_arr, ew->array, es->array, eb->array},
                    {mlx::core::Shape{1, hidden_size}},
                    {mlx::core::float32},
                    {static_cast<size_t>(hidden_size), 1, 1},
                    {1, 1, 1},
                    {{"HIDDEN_SIZE", hidden_size}},
                    std::nullopt, false, {});
                mlx::core::array embedding = mlx::core::astype(emb_outputs[0], compute_dtype);
                timing_buckets.embedding_ms += elapsed_ms_since(embedding_start);
                return embedding;
            }
            const auto embedding_start = std::chrono::steady_clock::now();
            std::vector<float> hidden(static_cast<size_t>(hidden_size));
            for (int i = 0; i < hidden_size; ++i) {
                const std::uint64_t shifted = prompt_signature >> ((i % 8) * 8);
                const float prompt_component = (static_cast<float>(shifted & 0xffU) - 127.5f) / 127.5f;
                const float position_component = static_cast<float>((i % 89) - 44) / 89.0f;
                hidden[static_cast<size_t>(i)] = position_component + (0.35f * prompt_component);
            }
            mlx::core::array synthetic_f32 = mlx::core::array(hidden.begin(), mlx::core::Shape{1, hidden_size}, mlx::core::float32);
            mlx::core::array synthetic = mlx::core::astype(synthetic_f32, compute_dtype);
            timing_buckets.embedding_ms += elapsed_ms_since(embedding_start);
            return synthetic;
        };
        mlx::core::array hidden_array = build_hidden_array();
        if (token_embedding_lookup_executed) {
            constexpr int q_heads = 32;
            constexpr int kv_heads = 8;
            constexpr int q_heads_per_kv = 4;
            constexpr int head_dim = 128;
            constexpr int intermediate_size = 9728;
            constexpr float rope_theta = 5000000.0f;
            constexpr float rope_scale = 1.0f;
            constexpr bool rope_traditional = false;
            auto apply_projection_adapter = [&](const mlx::core::array& x, const mlx::core::array& base, const std::string& projection_name) {
                if (adapter != nullptr && !projection_name.empty()) {
                    const ResidentArrayRecord* lora_a = require_logits_array(projection_name + ".lora_a");
                    const ResidentArrayRecord* lora_b = require_logits_array(projection_name + ".lora_b");
                    if (lora_a != nullptr && lora_b != nullptr) {
                        adapter_applied_projection_count += 1;
                        return mlx::core::astype(
                            base + generation_adapter_scale * mlx::core::matmul(mlx::core::matmul(x, lora_a->array), lora_b->array),
                            mlx::core::float32);
                    }
                }
                return base;
            };
            auto compiled_mlx_quantized_projection = [&](const mlx::core::array& x,
                                                         const ResidentArrayRecord* w,
                                                         const ResidentArrayRecord* s,
                                                         const ResidentArrayRecord* b,
                                                         int rows) -> mlx::core::array {
                static const std::string projection_source = R"METAL(
                    uint row = thread_position_in_grid.x;
                    if (row >= ROWS) {
                        return;
                    }
                    constexpr uint group_size = 64;
                    constexpr uint values_per_word = 8;
                    constexpr uint words_per_group = group_size / values_per_word;
                    constexpr uint groups = INPUT_LEN / group_size;
                    float acc = 0.0f;
                    for (uint g = 0; g < groups; ++g) {
                        const float scale = float(scales[row * groups + g]);
                        const float bias = float(biases[row * groups + g]);
                        for (uint word = 0; word < words_per_group; ++word) {
                            const uint packed = weight[row * PACKED_COLS + g * words_per_group + word];
                            for (uint n = 0; n < values_per_word; ++n) {
                                const uint q = (packed >> (n * 4)) & 0xFu;
                                const uint input_index = g * group_size + word * values_per_word + n;
                                acc += input[input_index] * (((float)q * scale) + bias);
                            }
                        }
                    }
                    out[row] = acc;
                )METAL";
                static auto projection_kernel = mlx::core::fast::metal_kernel(
                    "gypsy_quantized_projection_mlx_resident",
                    {"input", "weight", "scales", "biases"},
                    {"out"},
                    projection_source,
                    "",
                    true,
                    false);
                std::vector<mlx::core::array> outputs = projection_kernel(
                    {
                        mlx::core::flatten(mlx::core::astype(x, mlx::core::float32)),
                        w->array,
                        s->array,
                        b->array
                    },
                    {mlx::core::Shape{1, rows}},
                    {mlx::core::float32},
                    {static_cast<size_t>(rows), 1, 1},
                    {1, 1, 1},
                    {
                        {"ROWS", rows},
                        {"PACKED_COLS", 320},
                        {"INPUT_LEN", 2560}
                    },
                    std::nullopt,
                    false,
                    {});
                return mlx::core::astype(outputs[0], mlx::core::float32);
            };
            auto q_project = [&](const mlx::core::array& x, const ResidentArrayRecord* w, const ResidentArrayRecord* s, const ResidentArrayRecord* b, const std::string& projection_name = "") {
                mlx::core::array base = mlx::core::astype(
                    mlx::core::quantized_matmul(
                        x,
                        w->array,
                        s->array,
                        std::make_optional(b->array),
                        true,
                        group_size,
                        bits,
                        "affine"),
                    compute_dtype);
                return apply_projection_adapter(x, base, projection_name);
            };
            // Fused Q/K projection + RMS-norm + RoPE in one Metal kernel.
            // Grid: (num_heads, 1, 1) threadgroups. Each threadgroup has HEAD_DIM threads.
            // Each thread computes one quantized matmul output element, then the threadgroup
            // cooperates on the RMS reduction, then thread-pairs apply RoPE.
            auto fused_qk_proj_norm_rope = [&](const mlx::core::array& x,
                                                const ResidentArrayRecord* w,
                                                const ResidentArrayRecord* sc,
                                                const ResidentArrayRecord* bi,
                                                const ResidentArrayRecord* norm,
                                                int pos,
                                                int num_heads) -> mlx::core::array {
                static const std::string fused_source = R"METAL(
                    constexpr uint GROUP_SIZE = 64;
                    constexpr uint VALUES_PER_WORD = 8;
                    constexpr uint WORDS_PER_GROUP = GROUP_SIZE / VALUES_PER_WORD;
                    constexpr uint PACKED_COLS = HIDDEN_SIZE / VALUES_PER_WORD;
                    constexpr uint GROUPS_PER_ROW = HIDDEN_SIZE / GROUP_SIZE;
                    constexpr uint HALF_DIM = HEAD_DIM / 2;
                    constexpr float EPS = 1.0e-6f;
                    constexpr float THETA = 5000000.0f;

                    threadgroup float tg_values[HEAD_DIM];
                    threadgroup float tg_sum_sq[HEAD_DIM];

                    uint h = threadgroup_position_in_grid.x;
                    uint d = thread_position_in_threadgroup.x;

                    // 1. Quantized matmul: compute output row (h * HEAD_DIM + d)
                    uint row = h * HEAD_DIM + d;
                    float acc = 0.0f;
                    for (uint g = 0; g < GROUPS_PER_ROW; ++g) {
                        const float scl = float(scales[row * GROUPS_PER_ROW + g]);
                        const float bv = float(biases[row * GROUPS_PER_ROW + g]);
                        for (uint word = 0; word < WORDS_PER_GROUP; ++word) {
                            const uint packed = weight[row * PACKED_COLS + g * WORDS_PER_GROUP + word];
                            for (uint n = 0; n < VALUES_PER_WORD; ++n) {
                                const uint q = (packed >> (n * 4)) & 0xFu;
                                const uint input_index = g * GROUP_SIZE + word * VALUES_PER_WORD + n;
                                acc += x[input_index] * (float(q) * scl + bv);
                            }
                        }
                    }
                    tg_values[d] = acc;
                    tg_sum_sq[d] = acc * acc;
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    // 2. RMS-norm: tree-reduce sum of squares across threadgroup
                    for (uint stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
                        if (d < stride) {
                            tg_sum_sq[d] += tg_sum_sq[d + stride];
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    const float rms_inv = rsqrt(tg_sum_sq[0] / float(HEAD_DIM) + EPS);
                    const float normed = tg_values[d] * rms_inv * norm_w[d];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    tg_values[d] = normed;
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    // 3. RoPE (non-traditional pairing: d and d+HALF_DIM)
                    if (d < HALF_DIM) {
                        const float freq = exp(-log(THETA) * float(d) / float(HALF_DIM));
                        const float angle = float(position[0]) * freq;
                        const float cos_a = cos(angle);
                        const float sin_a = sin(angle);
                        const float v0 = tg_values[d];
                        const float v1 = tg_values[d + HALF_DIM];
                        out[h * HEAD_DIM + d] = v0 * cos_a - v1 * sin_a;
                        out[h * HEAD_DIM + d + HALF_DIM] = v0 * sin_a + v1 * cos_a;
                    }
                )METAL";
                static auto fused_kernel = mlx::core::fast::metal_kernel(
                    "gypsy_fused_qk_proj_norm_rope",
                    {"x", "weight", "scales", "biases", "norm_w", "position"},
                    {"out"},
                    fused_source, "", true, false);
                int32_t pos_val = pos;
                mlx::core::array pos_arr(&pos_val, mlx::core::Shape{1}, mlx::core::int32);
                mlx::core::array x_f32 = mlx::core::flatten(mlx::core::astype(x, mlx::core::float32));
                mlx::core::array norm_f32 = mlx::core::astype(norm->array, mlx::core::float32);
                std::vector<mlx::core::array> outputs = fused_kernel(
                    {x_f32, w->array, sc->array, bi->array, norm_f32, pos_arr},
                    {mlx::core::Shape{1, num_heads, 1, head_dim}},
                    {mlx::core::float32},
                    {static_cast<size_t>(num_heads) * head_dim, 1, 1},
                    {static_cast<size_t>(head_dim), 1, 1},
                    {{"HIDDEN_SIZE", static_cast<int>(hidden_size)}, {"HEAD_DIM", head_dim}},
                    std::nullopt, false, {});
                return outputs[0];
            };
            auto cached_attention = [&](int layer, const mlx::core::array& q_rope, const mlx::core::array& k_rope, const mlx::core::array& v_combined) {
                auto cpu_attention = [&]() {
                const auto readback_start = std::chrono::steady_clock::now();
                if (use_direct_metal_attention && direct_metal_resident_kv && direct_metal_concat_qkv_readback) {
                    mlx::core::array q_flat = mlx::core::flatten(mlx::core::astype(q_rope, mlx::core::float32));
                    mlx::core::array k_flat = mlx::core::flatten(mlx::core::astype(k_rope, mlx::core::float32));
                    mlx::core::array v_flat = mlx::core::flatten(mlx::core::astype(v_combined, mlx::core::float32));
                    mlx::core::array qkv_flat = mlx::core::concatenate({q_flat, k_flat, v_flat}, 0);
                    mlx::core::eval(qkv_flat);
                    const float* qkv_values = qkv_flat.data<float>();
                    const float* q_values = qkv_values;
                    const float* k_values = q_values + (q_heads * head_dim);
                    const float* v_values = k_values + (kv_heads * head_dim);
                    timing_buckets.readback_ms += elapsed_ms_since(readback_start);
                    timing_buckets.readback_count += 1;

                    const auto append_start = std::chrono::steady_clock::now();
                    LocalLayerKvCache& cache = local_kv_cache[static_cast<size_t>(layer)];
                    if (direct_metal_keep_cpu_kv) {
                        cache.keys.insert(cache.keys.end(), k_values, k_values + (kv_heads * head_dim));
                        cache.values.insert(cache.values.end(), v_values, v_values + (kv_heads * head_dim));
                    }
                    cache.len += 1;
                    kv_cache_materialized = true;
                    kv_cache_positions = std::max(kv_cache_positions, cache.len);
                    timing_buckets.kv_readback_append_ms += elapsed_ms_since(append_start);

                    const auto attention_start = std::chrono::steady_clock::now();
                    std::vector<float> attention(static_cast<size_t>(q_heads * head_dim), 0.0f);
                    GypsyDirectMetalProbeResult direct = GypsyRunDirectMetalAttentionAppendToHost(
                        static_cast<std::uint32_t>(layer),
                        q_values,
                        k_values,
                        v_values,
                        attention.data(),
                        cache.len);
                    timing_buckets.attention_mlx_ms += elapsed_ms_since(attention_start);
                    timing_buckets.attention_calls += 1;
                    timing_buckets.mlx_attention_calls += 1;
                    direct_metal_attention_used_anywhere = true;
                    if (direct.ok) {
                        return mlx::core::array(attention.begin(), mlx::core::Shape{1, q_heads * head_dim}, mlx::core::float32);
                    }
                    if (!direct_metal_keep_cpu_kv) {
                        throw std::runtime_error("direct Metal attention failed and CPU KV mirror is disabled");
                    }
                }
                mlx::core::array q_flat = mlx::core::flatten(mlx::core::astype(q_rope, mlx::core::float32));
                mlx::core::array k_flat = mlx::core::flatten(mlx::core::astype(k_rope, mlx::core::float32));
                mlx::core::array v_flat = mlx::core::flatten(mlx::core::astype(v_combined, mlx::core::float32));
                mlx::core::eval(q_flat, k_flat, v_flat);
                const float* q_values = q_flat.data<float>();
                const float* k_values = k_flat.data<float>();
                const float* v_values = v_flat.data<float>();
                timing_buckets.readback_ms += elapsed_ms_since(readback_start);
                timing_buckets.readback_count += 3;
                const auto append_start = std::chrono::steady_clock::now();
                LocalLayerKvCache& cache = local_kv_cache[static_cast<size_t>(layer)];
                if (!use_direct_metal_attention || direct_metal_keep_cpu_kv) {
                    cache.keys.insert(cache.keys.end(), k_values, k_values + (kv_heads * head_dim));
                    cache.values.insert(cache.values.end(), v_values, v_values + (kv_heads * head_dim));
                }
                cache.len += 1;
                kv_cache_materialized = true;
                kv_cache_positions = std::max(kv_cache_positions, cache.len);
                timing_buckets.kv_readback_append_ms += elapsed_ms_since(append_start);

                const auto attention_start = std::chrono::steady_clock::now();
                if (use_direct_metal_attention) {
                    std::vector<float> attention(static_cast<size_t>(q_heads * head_dim), 0.0f);
                    GypsyDirectMetalProbeResult direct = direct_metal_resident_kv
                        ? GypsyRunDirectMetalAttentionAppendToHost(
                            static_cast<std::uint32_t>(layer),
                            q_values,
                            k_values,
                            v_values,
                            attention.data(),
                            cache.len)
                        : GypsyRunDirectMetalAttentionToHost(
                            q_values,
                            cache.keys.data(),
                            cache.values.data(),
                            attention.data(),
                            cache.len);
                    timing_buckets.attention_mlx_ms += elapsed_ms_since(attention_start);
                    timing_buckets.attention_calls += 1;
                    timing_buckets.mlx_attention_calls += 1;
                    direct_metal_attention_used_anywhere = true;
                    if (direct.ok) {
                        return mlx::core::array(attention.begin(), mlx::core::Shape{1, q_heads * head_dim}, mlx::core::float32);
                    }
                    if (direct_metal_resident_kv && !direct_metal_keep_cpu_kv) {
                        throw std::runtime_error("direct Metal attention failed and CPU KV mirror is disabled");
                    }
                }
                std::vector<float> attention(static_cast<size_t>(q_heads * head_dim), 0.0f);
                std::vector<float> scores(static_cast<size_t>(cache.len));
                const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
                for (int qh = 0; qh < q_heads; ++qh) {
                    const int kvh = qh / q_heads_per_kv;
                    float max_score = -std::numeric_limits<float>::infinity();
                    for (std::uint32_t pos = 0; pos < cache.len; ++pos) {
                        const size_t k_base = (static_cast<size_t>(pos) * kv_heads + kvh) * head_dim;
                        const size_t q_base = static_cast<size_t>(qh) * head_dim;
                        float score = 0.0f;
                        for (int d = 0; d < head_dim; ++d) {
                            score += q_values[q_base + d] * cache.keys[k_base + d];
                        }
                        score *= scale;
                        scores[static_cast<size_t>(pos)] = score;
                        if (score > max_score) max_score = score;
                    }
                    float denom = 0.0f;
                    for (std::uint32_t pos = 0; pos < cache.len; ++pos) {
                        const float e = std::exp(scores[static_cast<size_t>(pos)] - max_score);
                        scores[static_cast<size_t>(pos)] = e;
                        denom += e;
                    }
                    if (denom == 0.0f) denom = 1.0f;
                    for (std::uint32_t pos = 0; pos < cache.len; ++pos) {
                        const float p = scores[static_cast<size_t>(pos)] / denom;
                        const size_t v_base = (static_cast<size_t>(pos) * kv_heads + kvh) * head_dim;
                        const size_t out_base = static_cast<size_t>(qh) * head_dim;
                        for (int d = 0; d < head_dim; ++d) {
                            attention[out_base + d] += p * cache.values[v_base + d];
                        }
                    }
                }
                timing_buckets.attention_cpu_ms += elapsed_ms_since(attention_start);
                timing_buckets.attention_calls += 1;
                timing_buckets.cpu_attention_calls += 1;
                cpu_attention_used_anywhere = true;
                return mlx::core::array(attention.begin(), mlx::core::Shape{1, q_heads * head_dim}, mlx::core::float32);
                };

                if (!use_mlx_attention) {
                    return cpu_attention();
                }

                try {
                    const auto append_start = std::chrono::steady_clock::now();
                    LocalLayerKvCache& cache = local_kv_cache[static_cast<size_t>(layer)];
                    // Expand kv_heads → q_heads by repeating each kv head q_heads_per_kv times.
                    // This gives standard MHA layout [1, q_heads, 1, head_dim] for SDPA.
                    mlx::core::array k_source = mlx::core::repeat(
                        mlx::core::reshape(k_rope, mlx::core::Shape{1, kv_heads, 1, head_dim}),
                        q_heads_per_kv, 1);  // [1, q_heads, 1, head_dim]
                    mlx::core::array v_source = mlx::core::repeat(
                        mlx::core::reshape(v_combined, mlx::core::Shape{1, kv_heads, 1, head_dim}),
                        q_heads_per_kv, 1);  // [1, q_heads, 1, head_dim]
                    if (use_preallocated_mlx_kv) {
                        if (!cache.has_mlx_cache || !cache.mlx_expanded_keys.has_value() || !cache.mlx_expanded_values.has_value()) {
                            cache.mlx_expanded_keys = mlx::core::zeros(
                                mlx::core::Shape{1, q_heads, mlx_prealloc_kv_capacity, head_dim},
                                compute_dtype);
                            cache.mlx_expanded_values = mlx::core::zeros(
                                mlx::core::Shape{1, q_heads, mlx_prealloc_kv_capacity, head_dim},
                                compute_dtype);
                            cache.has_mlx_cache = true;
                        }
                        const int write_pos = static_cast<int>(cache.len);
                        cache.mlx_expanded_keys = mlx::core::slice_update(
                            cache.mlx_expanded_keys.value(),
                            k_source,
                            mlx::core::Shape{0, 0, write_pos, 0},
                            mlx::core::Shape{1, q_heads, write_pos + 1, head_dim});
                        cache.mlx_expanded_values = mlx::core::slice_update(
                            cache.mlx_expanded_values.value(),
                            v_source,
                            mlx::core::Shape{0, 0, write_pos, 0},
                            mlx::core::Shape{1, q_heads, write_pos + 1, head_dim});
                    } else if (cache.has_mlx_cache && cache.mlx_expanded_keys.has_value() && cache.mlx_expanded_values.has_value()) {
                        cache.mlx_expanded_keys = mlx::core::concatenate({cache.mlx_expanded_keys.value(), k_source}, 2);
                        cache.mlx_expanded_values = mlx::core::concatenate({cache.mlx_expanded_values.value(), v_source}, 2);
                    } else {
                        cache.mlx_expanded_keys = k_source;
                        cache.mlx_expanded_values = v_source;
                        cache.has_mlx_cache = true;
                    }
                    cache.len += 1;
                    kv_cache_materialized = true;
                    kv_cache_positions = std::max(kv_cache_positions, cache.len);
                    timing_buckets.mlx_kv_append_ms += elapsed_ms_since(append_start);

                    const auto attention_start = std::chrono::steady_clock::now();
                    constexpr float scale = 0.08838834764831845f;
                    mlx::core::array keys_for_attention = use_preallocated_mlx_kv
                        ? mlx::core::slice(
                            cache.mlx_expanded_keys.value(),
                            mlx::core::Shape{0, 0, 0, 0},
                            mlx::core::Shape{1, q_heads, static_cast<int>(cache.len), head_dim})
                        : cache.mlx_expanded_keys.value();
                    mlx::core::array values_for_attention = use_preallocated_mlx_kv
                        ? mlx::core::slice(
                            cache.mlx_expanded_values.value(),
                            mlx::core::Shape{0, 0, 0, 0},
                            mlx::core::Shape{1, q_heads, static_cast<int>(cache.len), head_dim})
                        : cache.mlx_expanded_values.value();
                    mlx::core::array output = mlx::core::array(0.0f);
                    if (use_custom_metal_attention) {
                        static const std::string attention_source = R"METAL(
                            uint h = thread_position_in_grid.x;
                            if (h >= Q_HEADS) {
                                return;
                            }
                            int n = seq_len[0];
                            if (n <= 0) {
                                return;
                            }
                            if (n > MAX_SEQ) {
                                for (int d = 0; d < HEAD_DIM; ++d) {
                                    out[h * HEAD_DIM + d] = 0.0f;
                                }
                                return;
                            }
                            float scores[MAX_SEQ];
                            float max_score = -3.4028234663852886e38f;
                            for (int p = 0; p < n; ++p) {
                                float score = 0.0f;
                                for (int d = 0; d < HEAD_DIM; ++d) {
                                    score += q[h * HEAD_DIM + d] * k[((h * k_shape[2] + p) * HEAD_DIM) + d];
                                }
                                score *= 0.08838834764831845f;
                                scores[p] = score;
                                max_score = metal::max(max_score, score);
                            }
                            float denom = 0.0f;
                            for (int p = 0; p < n; ++p) {
                                scores[p] = metal::exp(scores[p] - max_score);
                                denom += scores[p];
                            }
                            denom = metal::max(denom, 1.0e-20f);
                            for (int d = 0; d < HEAD_DIM; ++d) {
                                float acc = 0.0f;
                                for (int p = 0; p < n; ++p) {
                                    float prob = scores[p] / denom;
                                    acc += prob * v[((h * v_shape[2] + p) * HEAD_DIM) + d];
                                }
                                out[h * HEAD_DIM + d] = acc;
                            }
                        )METAL";
                        static auto attention_kernel = mlx::core::fast::metal_kernel(
                            "gypsy_single_token_attention",
                            {"q", "k", "v", "seq_len"},
                            {"out"},
                            attention_source,
                            "",
                            true,
                            false);
                        std::vector<int32_t> seq_len_host{static_cast<int32_t>(cache.len)};
                        mlx::core::array seq_len_array(seq_len_host.begin(), mlx::core::Shape{1}, mlx::core::int32);
                        std::vector<mlx::core::array> attention_outputs = attention_kernel(
                            {
                                mlx::core::flatten(mlx::core::astype(q_rope, mlx::core::float32)),
                                keys_for_attention,
                                values_for_attention,
                                seq_len_array
                            },
                            {mlx::core::Shape{1, q_heads * head_dim}},
                            {mlx::core::float32},
                            {q_heads, 1, 1},
                            {q_heads, 1, 1},
                            {
                                {"Q_HEADS", q_heads},
                                {"HEAD_DIM", head_dim},
                                {"MAX_SEQ", 256}
                            },
                            std::nullopt,
                            false,
                            {});
                        output = attention_outputs[0];
                    } else {
                        mlx::core::array attention = mlx::core::fast::scaled_dot_product_attention(
                            mlx::core::reshape(q_rope, mlx::core::Shape{1, q_heads, 1, head_dim}),
                            keys_for_attention,
                            values_for_attention,
                            scale);
                        output = mlx::core::reshape(
                            mlx::core::astype(attention, compute_dtype),
                            mlx::core::Shape{1, q_heads * head_dim});
                    }
                    timing_buckets.attention_mlx_ms += elapsed_ms_since(attention_start);
                    timing_buckets.attention_calls += 1;
                    timing_buckets.mlx_attention_calls += 1;
                    mlx_attention_used_anywhere = true;
                    return output;
                } catch (...) {
                    return cpu_attention();
                }
            };
            auto cached_attention_o_projection = [&](int layer,
                                                     const mlx::core::array& q_rope,
                                                     const mlx::core::array& k_rope,
                                                     const mlx::core::array& v_combined,
                                                     const ResidentArrayRecord* ow_rec,
                                                     const ResidentArrayRecord* os_rec,
                                                     const ResidentArrayRecord* ob_rec) -> std::optional<mlx::core::array> {
                if (!use_fused_attention_o_projection ||
                    !use_direct_metal_attention ||
                    ow_rec == nullptr || os_rec == nullptr || ob_rec == nullptr ||
                    ow_rec->raw_data == nullptr || os_rec->raw_data == nullptr || ob_rec->raw_data == nullptr) {
                    return std::nullopt;
                }
                const auto readback_start = std::chrono::steady_clock::now();
                mlx::core::array q_flat = mlx::core::flatten(mlx::core::astype(q_rope, mlx::core::float32));
                mlx::core::array k_flat = mlx::core::flatten(mlx::core::astype(k_rope, mlx::core::float32));
                mlx::core::array v_flat = mlx::core::flatten(mlx::core::astype(v_combined, mlx::core::float32));
                mlx::core::eval(q_flat, k_flat, v_flat);
                const float* q_values = q_flat.data<float>();
                const float* k_values = k_flat.data<float>();
                const float* v_values = v_flat.data<float>();
                timing_buckets.readback_ms += elapsed_ms_since(readback_start);
                timing_buckets.readback_count += 3;

                const auto append_start = std::chrono::steady_clock::now();
                LocalLayerKvCache& cache = local_kv_cache[static_cast<size_t>(layer)];
                if (direct_metal_keep_cpu_kv) {
                    cache.keys.insert(cache.keys.end(), k_values, k_values + (kv_heads * head_dim));
                    cache.values.insert(cache.values.end(), v_values, v_values + (kv_heads * head_dim));
                }
                cache.len += 1;
                kv_cache_materialized = true;
                kv_cache_positions = std::max(kv_cache_positions, cache.len);
                timing_buckets.kv_readback_append_ms += elapsed_ms_since(append_start);

                const auto attention_start = std::chrono::steady_clock::now();
                std::vector<float> o_projection_values(static_cast<size_t>(hidden_size), 0.0f);
                GypsyDirectMetalProbeResult direct = GypsyRunDirectMetalAttentionOProjectionAppendToHost(
                    static_cast<std::uint32_t>(layer),
                    q_values,
                    k_values,
                    v_values,
                    static_cast<const std::uint32_t*>(ow_rec->raw_data),
                    static_cast<const std::uint16_t*>(os_rec->raw_data),
                    static_cast<const std::uint16_t*>(ob_rec->raw_data),
                    o_projection_values.data(),
                    cache.len);
                const double fused_elapsed = elapsed_ms_since(attention_start);
                timing_buckets.attention_mlx_ms += fused_elapsed;
                timing_buckets.fused_attention_o_ms += fused_elapsed;
                timing_buckets.attention_calls += 1;
                timing_buckets.fused_attention_o_calls += 1;
                if (layer >= direct_metal_terminal_layer_start && use_direct_metal_layer35_attention_o) {
                    timing_buckets.direct_metal_layer35_attention_o_ms += fused_elapsed;
                    timing_buckets.direct_metal_layer35_attention_o_calls += 1;
                    timing_buckets.direct_metal_layer35_attention_o_checksum = direct.checksum;
                }
                timing_buckets.mlx_attention_calls += 1;
                direct_metal_attention_used_anywhere = true;
                if (!direct.ok) {
                    return std::nullopt;
                }
                return mlx::core::array(o_projection_values.begin(), mlx::core::Shape{1, hidden_size}, mlx::core::float32);
            };
            auto cached_attention_o_residual = [&](int layer,
                                                   const mlx::core::array& layer_input,
                                                   const mlx::core::array& q_rope,
                                                   const mlx::core::array& k_rope,
                                                   const mlx::core::array& v_combined,
                                                   const ResidentArrayRecord* ow_rec,
                                                   const ResidentArrayRecord* os_rec,
                                                   const ResidentArrayRecord* ob_rec) -> std::optional<mlx::core::array> {
                if (!use_fused_attention_o_residual ||
                    !use_direct_metal_attention ||
                    ow_rec == nullptr || os_rec == nullptr || ob_rec == nullptr ||
                    ow_rec->raw_data == nullptr || os_rec->raw_data == nullptr || ob_rec->raw_data == nullptr) {
                    return std::nullopt;
                }
                const auto readback_start = std::chrono::steady_clock::now();
                mlx::core::array q_flat = mlx::core::flatten(mlx::core::astype(q_rope, mlx::core::float32));
                mlx::core::array k_flat = mlx::core::flatten(mlx::core::astype(k_rope, mlx::core::float32));
                mlx::core::array v_flat = mlx::core::flatten(mlx::core::astype(v_combined, mlx::core::float32));
                mlx::core::array residual_flat = mlx::core::flatten(mlx::core::astype(layer_input, mlx::core::float32));
                mlx::core::eval(q_flat, k_flat, v_flat, residual_flat);
                const float* q_values = q_flat.data<float>();
                const float* k_values = k_flat.data<float>();
                const float* v_values = v_flat.data<float>();
                const float* residual_values = residual_flat.data<float>();
                timing_buckets.readback_ms += elapsed_ms_since(readback_start);
                timing_buckets.readback_count += 4;

                const auto append_start = std::chrono::steady_clock::now();
                LocalLayerKvCache& cache = local_kv_cache[static_cast<size_t>(layer)];
                if (direct_metal_keep_cpu_kv) {
                    cache.keys.insert(cache.keys.end(), k_values, k_values + (kv_heads * head_dim));
                    cache.values.insert(cache.values.end(), v_values, v_values + (kv_heads * head_dim));
                }
                cache.len += 1;
                kv_cache_materialized = true;
                kv_cache_positions = std::max(kv_cache_positions, cache.len);
                timing_buckets.kv_readback_append_ms += elapsed_ms_since(append_start);

                const auto attention_start = std::chrono::steady_clock::now();
                std::vector<float> residual_output(static_cast<size_t>(hidden_size), 0.0f);
                GypsyDirectMetalProbeResult direct = GypsyRunDirectMetalAttentionOResidualAppendToHost(
                    static_cast<std::uint32_t>(layer),
                    q_values,
                    k_values,
                    v_values,
                    static_cast<const std::uint32_t*>(ow_rec->raw_data),
                    static_cast<const std::uint16_t*>(os_rec->raw_data),
                    static_cast<const std::uint16_t*>(ob_rec->raw_data),
                    residual_values,
                    residual_output.data(),
                    cache.len);
                const double fused_elapsed = elapsed_ms_since(attention_start);
                timing_buckets.attention_mlx_ms += fused_elapsed;
                timing_buckets.fused_attention_o_ms += fused_elapsed;
                timing_buckets.attention_calls += 1;
                timing_buckets.fused_attention_o_calls += 1;
                timing_buckets.mlx_attention_calls += 1;
                direct_metal_attention_used_anywhere = true;
                if (!direct.ok) {
                    return std::nullopt;
                }
                return mlx::core::array(residual_output.begin(), mlx::core::Shape{1, hidden_size}, mlx::core::float32);
            };

            auto run_plain_layer = [&](int layer, const mlx::core::array& layer_input) -> mlx::core::array {
                const auto layer_start = std::chrono::steady_clock::now();
                const std::string p = "model.layers." + std::to_string(layer) + ".";
                const ResidentArrayRecord* input_norm = require_logits_array(p + "input_layernorm.weight");
                const ResidentArrayRecord* q_norm = require_logits_array(p + "self_attn.q_norm.weight");
                const ResidentArrayRecord* k_norm = require_logits_array(p + "self_attn.k_norm.weight");
                const ResidentArrayRecord* qw = require_logits_array(p + "self_attn.q_proj.weight");
                const ResidentArrayRecord* qs = require_logits_array(p + "self_attn.q_proj.scales");
                const ResidentArrayRecord* qb = require_logits_array(p + "self_attn.q_proj.biases");
                const ResidentArrayRecord* kw = require_logits_array(p + "self_attn.k_proj.weight");
                const ResidentArrayRecord* ks = require_logits_array(p + "self_attn.k_proj.scales");
                const ResidentArrayRecord* kb = require_logits_array(p + "self_attn.k_proj.biases");
                const ResidentArrayRecord* vw = require_logits_array(p + "self_attn.v_proj.weight");
                const ResidentArrayRecord* vs = require_logits_array(p + "self_attn.v_proj.scales");
                const ResidentArrayRecord* vb = require_logits_array(p + "self_attn.v_proj.biases");
                const ResidentArrayRecord* ow = require_logits_array(p + "self_attn.o_proj.weight");
                const ResidentArrayRecord* os = require_logits_array(p + "self_attn.o_proj.scales");
                const ResidentArrayRecord* ob = require_logits_array(p + "self_attn.o_proj.biases");
                const ResidentArrayRecord* post_norm = require_logits_array(p + "post_attention_layernorm.weight");
                const ResidentArrayRecord* gw = require_logits_array(p + "mlp.gate_proj.weight");
                const ResidentArrayRecord* gs = require_logits_array(p + "mlp.gate_proj.scales");
                const ResidentArrayRecord* gb = require_logits_array(p + "mlp.gate_proj.biases");
                const ResidentArrayRecord* uw = require_logits_array(p + "mlp.up_proj.weight");
                const ResidentArrayRecord* us = require_logits_array(p + "mlp.up_proj.scales");
                const ResidentArrayRecord* ub = require_logits_array(p + "mlp.up_proj.biases");
                const ResidentArrayRecord* dw = require_logits_array(p + "mlp.down_proj.weight");
                const ResidentArrayRecord* ds = require_logits_array(p + "mlp.down_proj.scales");
                const ResidentArrayRecord* db = require_logits_array(p + "mlp.down_proj.biases");

                const auto input_norm_start = std::chrono::steady_clock::now();
                const bool direct_terminal_layer = layer >= direct_metal_terminal_layer_start;
                std::optional<std::vector<float>> direct_input_norm_values;
                mlx::core::array attn_norm = mlx::core::array(0.0f);
                if (use_direct_metal_layer35_input_norm &&
                    direct_terminal_layer &&
                    input_norm != nullptr &&
                    input_norm->raw_data != nullptr) {
                    const auto direct_input_norm_start = std::chrono::steady_clock::now();
                    const auto direct_input_norm_readback_start = std::chrono::steady_clock::now();
                    mlx::core::array layer_input_flat = mlx::core::flatten(mlx::core::astype(layer_input, mlx::core::float32));
                    mlx::core::eval(layer_input_flat);
                    timing_buckets.readback_ms += elapsed_ms_since(direct_input_norm_readback_start);
                    timing_buckets.readback_count += 1;
                    std::vector<float> norm_values(static_cast<size_t>(hidden_size), 0.0f);
                    GypsyDirectMetalProbeResult direct_input_norm = GypsyRunDirectMetalRmsNormToHost(
                        layer_input_flat.data<float>(),
                        static_cast<const std::uint16_t*>(input_norm->raw_data),
                        norm_values.data(),
                        static_cast<std::uint32_t>(hidden_size));
                    timing_buckets.direct_metal_layer35_input_norm_ms += elapsed_ms_since(direct_input_norm_start);
                    if (direct_input_norm.ok) {
                        timing_buckets.direct_metal_layer35_input_norm_calls += 1;
                        timing_buckets.direct_metal_layer35_input_norm_checksum = direct_input_norm.checksum;
                        direct_input_norm_values = std::move(norm_values);
                        attn_norm = mlx::core::array(
                            direct_input_norm_values->begin(),
                            mlx::core::Shape{1, hidden_size},
                            mlx::core::float32);
                    }
                }
                if (!direct_input_norm_values.has_value()) {
                    attn_norm = mlx::core::astype(
                        mlx::core::fast::rms_norm(layer_input, std::make_optional(input_norm->array), eps),
                        compute_dtype);
                }
                timing_buckets.input_norm_ms += elapsed_ms_since(input_norm_start);
                const auto qkv_start = std::chrono::steady_clock::now();
                mlx::core::array q_combined = mlx::core::array(0.0f);
                mlx::core::array k_combined = mlx::core::array(0.0f);
                mlx::core::array v_combined = mlx::core::array(0.0f);
                std::optional<mlx::core::array> direct_metal_o_projection;
                const bool projection_has_adapter =
                    adapter != nullptr &&
                    (require_logits_array(p + "self_attn.q_proj.lora_a") != nullptr ||
                     require_logits_array(p + "self_attn.k_proj.lora_a") != nullptr ||
                     require_logits_array(p + "self_attn.v_proj.lora_a") != nullptr ||
                     require_logits_array(p + "self_attn.o_proj.lora_a") != nullptr);
                const bool direct_qkv_attention_o_candidate =
                    use_direct_metal_qkv_attention_o ||
                    (use_direct_metal_layer35_qkv_attention_o && direct_terminal_layer);
                if (direct_qkv_attention_o_candidate &&
                    use_direct_metal_attention &&
                    !projection_has_adapter &&
                    qw != nullptr && qs != nullptr && qb != nullptr &&
                    kw != nullptr && ks != nullptr && kb != nullptr &&
                    vw != nullptr && vs != nullptr && vb != nullptr &&
                    ow != nullptr && os != nullptr && ob != nullptr &&
                    q_norm != nullptr && k_norm != nullptr &&
                    qw->raw_data != nullptr && qs->raw_data != nullptr && qb->raw_data != nullptr &&
                    kw->raw_data != nullptr && ks->raw_data != nullptr && kb->raw_data != nullptr &&
                    vw->raw_data != nullptr && vs->raw_data != nullptr && vb->raw_data != nullptr &&
                    ow->raw_data != nullptr && os->raw_data != nullptr && ob->raw_data != nullptr &&
                    q_norm->raw_data != nullptr && k_norm->raw_data != nullptr) {
                    try {
                        const float* attn_norm_values = nullptr;
                        mlx::core::array attn_norm_flat = mlx::core::array(0.0f);
                        if (direct_input_norm_values.has_value()) {
                            attn_norm_values = direct_input_norm_values->data();
                        } else {
                            const auto readback_start = std::chrono::steady_clock::now();
                            attn_norm_flat = mlx::core::flatten(mlx::core::astype(attn_norm, mlx::core::float32));
                            mlx::core::eval(attn_norm_flat);
                            attn_norm_values = attn_norm_flat.data<float>();
                            timing_buckets.readback_ms += elapsed_ms_since(readback_start);
                            timing_buckets.readback_count += 1;
                        }

                        LocalLayerKvCache& cache = local_kv_cache[static_cast<size_t>(layer)];
                        cache.len += 1;
                        kv_cache_materialized = true;
                        kv_cache_positions = std::max(kv_cache_positions, cache.len);

                        const auto direct_start = std::chrono::steady_clock::now();
                        std::vector<float> o_values(static_cast<size_t>(hidden_size), 0.0f);
                        GypsyDirectMetalProbeResult direct = GypsyRunDirectMetalQkvAttentionOAppendToHost(
                            static_cast<std::uint32_t>(layer),
                            attn_norm_values,
                            static_cast<const std::uint32_t*>(qw->raw_data),
                            static_cast<const std::uint16_t*>(qs->raw_data),
                            static_cast<const std::uint16_t*>(qb->raw_data),
                            static_cast<const std::uint32_t*>(kw->raw_data),
                            static_cast<const std::uint16_t*>(ks->raw_data),
                            static_cast<const std::uint16_t*>(kb->raw_data),
                            static_cast<const std::uint32_t*>(vw->raw_data),
                            static_cast<const std::uint16_t*>(vs->raw_data),
                            static_cast<const std::uint16_t*>(vb->raw_data),
                            static_cast<const std::uint16_t*>(q_norm->raw_data),
                            static_cast<const std::uint16_t*>(k_norm->raw_data),
                            static_cast<const std::uint32_t*>(ow->raw_data),
                            static_cast<const std::uint16_t*>(os->raw_data),
                            static_cast<const std::uint16_t*>(ob->raw_data),
                            o_values.data(),
                            cache.len,
                            static_cast<std::uint32_t>(position));
                        const double direct_elapsed = elapsed_ms_since(direct_start);
                        timing_buckets.attention_mlx_ms += direct_elapsed;
                        timing_buckets.fused_attention_o_ms += direct_elapsed;
                        timing_buckets.attention_calls += 1;
                        timing_buckets.fused_attention_o_calls += 1;
                        timing_buckets.direct_qkv_attention_o_calls += 1;
                        if (direct_terminal_layer && use_direct_metal_layer35_qkv_attention_o) {
                            timing_buckets.direct_metal_layer35_qkv_attention_o_calls += 1;
                            timing_buckets.direct_metal_layer35_qkv_attention_o_ms += direct_elapsed;
                        }
                        timing_buckets.mlx_attention_calls += 1;
                        timing_buckets.direct_qkv_attention_o_q_checksum = direct.q_checksum;
                        timing_buckets.direct_qkv_attention_o_k_checksum = direct.k_checksum;
                        timing_buckets.direct_qkv_attention_o_v_checksum = direct.v_checksum;
                        timing_buckets.direct_qkv_attention_o_attention_checksum = direct.attention_checksum;
                        timing_buckets.direct_qkv_attention_o_output_checksum = direct.checksum;
                        direct_metal_attention_used_anywhere = true;
                        if (direct.ok) {
                            direct_metal_o_projection = mlx::core::array(o_values.begin(), mlx::core::Shape{1, hidden_size}, mlx::core::float32);
                        }
                    } catch (...) {
                        direct_metal_o_projection = std::nullopt;
                    }
                }
                bool compiled_qkv_used = false;
                if (!direct_metal_o_projection.has_value() &&
                    use_compiled_metal_qkv_projection &&
                    qw != nullptr && qs != nullptr && qb != nullptr &&
                    kw != nullptr && ks != nullptr && kb != nullptr &&
                    vw != nullptr && vs != nullptr && vb != nullptr &&
                    true) {
                    try {
                        q_combined = apply_projection_adapter(attn_norm, compiled_mlx_quantized_projection(attn_norm, qw, qs, qb, 4096), p + "self_attn.q_proj");
                        k_combined = apply_projection_adapter(attn_norm, compiled_mlx_quantized_projection(attn_norm, kw, ks, kb, 1024), p + "self_attn.k_proj");
                        v_combined = apply_projection_adapter(attn_norm, compiled_mlx_quantized_projection(attn_norm, vw, vs, vb, 1024), p + "self_attn.v_proj");
                        compiled_qkv_used = true;
                    } catch (...) {
                        compiled_qkv_used = false;
                    }
                }
                if (!direct_metal_o_projection.has_value() && !compiled_qkv_used) {
                    q_combined = q_project(attn_norm, qw, qs, qb, p + "self_attn.q_proj");
                    k_combined = q_project(attn_norm, kw, ks, kb, p + "self_attn.k_proj");
                    v_combined = q_project(attn_norm, vw, vs, vb, p + "self_attn.v_proj");
                }
                timing_buckets.qkv_projection_ms += elapsed_ms_since(qkv_start);
                timing_buckets.qkv_projection_calls += 1;
                timing_buckets.quantized_projection_calls += 3;
                const auto qk_norm_rope_start = std::chrono::steady_clock::now();
                mlx::core::array q_rope = mlx::core::array(0.0f);
                mlx::core::array k_rope = mlx::core::array(0.0f);
                if (!direct_metal_o_projection.has_value()) {
                    mlx::core::array q_normed = mlx::core::astype(
                        mlx::core::fast::rms_norm(mlx::core::reshape(q_combined, mlx::core::Shape{q_heads, head_dim}), std::make_optional(q_norm->array), eps),
                        compute_dtype);
                    mlx::core::array k_normed = mlx::core::astype(
                        mlx::core::fast::rms_norm(mlx::core::reshape(k_combined, mlx::core::Shape{kv_heads, head_dim}), std::make_optional(k_norm->array), eps),
                        compute_dtype);
                    q_rope = mlx::core::astype(
                        mlx::core::fast::rope(
                            mlx::core::reshape(q_normed, mlx::core::Shape{1, q_heads, 1, head_dim}),
                            head_dim,
                            rope_traditional,
                            std::make_optional(rope_theta),
                            rope_scale,
                            static_cast<int>(position)),
                        compute_dtype);
                    k_rope = mlx::core::astype(
                        mlx::core::fast::rope(
                            mlx::core::reshape(k_normed, mlx::core::Shape{1, kv_heads, 1, head_dim}),
                            head_dim,
                            rope_traditional,
                            std::make_optional(rope_theta),
                            rope_scale,
                            static_cast<int>(position)),
                        compute_dtype);
                }
                timing_buckets.qk_norm_rope_ms += elapsed_ms_since(qk_norm_rope_start);
                const auto o_start = std::chrono::steady_clock::now();
                const bool o_projection_has_adapter =
                    adapter != nullptr &&
                    require_logits_array(p + "self_attn.o_proj.lora_a") != nullptr &&
                    require_logits_array(p + "self_attn.o_proj.lora_b") != nullptr;
                const bool direct_layer35_post_residual_candidate =
                    use_direct_metal_layer35_post_residual && direct_terminal_layer;
                const bool direct_layer35_attention_o_candidate =
                    use_direct_metal_layer35_attention_o && direct_terminal_layer;
                std::optional<mlx::core::array> fused_post_attention_residual = direct_metal_o_projection.has_value() || o_projection_has_adapter || direct_layer35_post_residual_candidate
                    ? std::nullopt
                    : cached_attention_o_residual(layer, layer_input, q_rope, k_rope, v_combined, ow, os, ob);
                std::optional<mlx::core::array> fused_o_projection = direct_metal_o_projection.has_value() || fused_post_attention_residual.has_value() || o_projection_has_adapter
                    ? std::nullopt
                    : (direct_terminal_layer
                        ? (direct_layer35_attention_o_candidate
                            ? cached_attention_o_projection(layer, q_rope, k_rope, v_combined, ow, os, ob)
                            : std::nullopt)
                        : cached_attention_o_projection(layer, q_rope, k_rope, v_combined, ow, os, ob));
                mlx::core::array o_projection = direct_metal_o_projection.has_value()
                    ? direct_metal_o_projection.value()
                    : (fused_o_projection.has_value()
                        ? fused_o_projection.value()
                        : (fused_post_attention_residual.has_value()
                            ? mlx::core::array(0.0f)
                            : q_project(cached_attention(layer, q_rope, k_rope, v_combined), ow, os, ob, p + "self_attn.o_proj")));
                timing_buckets.o_projection_ms += elapsed_ms_since(o_start);
                timing_buckets.quantized_projection_calls += 1;
                const bool gate_projection_has_adapter_for_tail =
                    adapter != nullptr &&
                    require_logits_array(p + "mlp.gate_proj.lora_a") != nullptr &&
                    require_logits_array(p + "mlp.gate_proj.lora_b") != nullptr;
                const bool up_projection_has_adapter_for_tail =
                    adapter != nullptr &&
                    require_logits_array(p + "mlp.up_proj.lora_a") != nullptr &&
                    require_logits_array(p + "mlp.up_proj.lora_b") != nullptr;
                const bool down_projection_has_adapter_for_tail =
                    adapter != nullptr &&
                    require_logits_array(p + "mlp.down_proj.lora_a") != nullptr &&
                    require_logits_array(p + "mlp.down_proj.lora_b") != nullptr;
                if (use_direct_metal_fused_mlp_tail &&
                    direct_terminal_layer &&
                    !fused_post_attention_residual.has_value() &&
                    !gate_projection_has_adapter_for_tail &&
                    !up_projection_has_adapter_for_tail &&
                    !down_projection_has_adapter_for_tail &&
                    post_norm != nullptr && post_norm->raw_data != nullptr &&
                    gw != nullptr && gs != nullptr && gb != nullptr &&
                    uw != nullptr && us != nullptr && ub != nullptr &&
                    dw != nullptr && ds != nullptr && db != nullptr &&
                    gw->raw_data != nullptr && gs->raw_data != nullptr && gb->raw_data != nullptr &&
                    uw->raw_data != nullptr && us->raw_data != nullptr && ub->raw_data != nullptr &&
                    dw->raw_data != nullptr && ds->raw_data != nullptr && db->raw_data != nullptr) {
                    const auto fused_tail_start = std::chrono::steady_clock::now();
                    const auto fused_tail_readback_start = std::chrono::steady_clock::now();
                    mlx::core::array layer_input_flat = mlx::core::flatten(mlx::core::astype(layer_input, mlx::core::float32));
                    mlx::core::array o_projection_flat = mlx::core::flatten(mlx::core::astype(o_projection, mlx::core::float32));
                    mlx::core::eval(layer_input_flat, o_projection_flat);
                    timing_buckets.readback_ms += elapsed_ms_since(fused_tail_readback_start);
                    timing_buckets.readback_count += 2;
                    std::vector<float> tail_values(static_cast<size_t>(hidden_size), 0.0f);
                    GypsyDirectMetalProbeResult direct_tail = GypsyRunDirectMetalMlpTailToHost(
                        layer_input_flat.data<float>(),
                        o_projection_flat.data<float>(),
                        static_cast<const std::uint16_t*>(post_norm->raw_data),
                        static_cast<const std::uint32_t*>(gw->raw_data),
                        static_cast<const std::uint16_t*>(gs->raw_data),
                        static_cast<const std::uint16_t*>(gb->raw_data),
                        static_cast<const std::uint32_t*>(uw->raw_data),
                        static_cast<const std::uint16_t*>(us->raw_data),
                        static_cast<const std::uint16_t*>(ub->raw_data),
                        static_cast<const std::uint32_t*>(dw->raw_data),
                        static_cast<const std::uint16_t*>(ds->raw_data),
                        static_cast<const std::uint16_t*>(db->raw_data),
                        tail_values.data());
                    timing_buckets.direct_metal_fused_mlp_tail_ms += elapsed_ms_since(fused_tail_start);
                    if (direct_tail.ok) {
                        timing_buckets.direct_metal_fused_mlp_tail_calls += 1;
                        timing_buckets.direct_metal_fused_mlp_tail_checksum = direct_tail.checksum;
                        timing_buckets.layer_total_ms += elapsed_ms_since(layer_start);
                        return mlx::core::array(tail_values.begin(), mlx::core::Shape{1, hidden_size}, mlx::core::float32);
                    }
                }
                const auto residual_norm_start = std::chrono::steady_clock::now();
                std::optional<std::vector<float>> direct_post_residual_values;
                mlx::core::array post_attention_residual = mlx::core::array(0.0f);
                if (direct_layer35_post_residual_candidate && !fused_post_attention_residual.has_value()) {
                    const auto direct_post_residual_start = std::chrono::steady_clock::now();
                    const auto direct_post_residual_readback_start = std::chrono::steady_clock::now();
                    mlx::core::array layer_input_flat = mlx::core::flatten(mlx::core::astype(layer_input, mlx::core::float32));
                    mlx::core::array o_projection_flat = mlx::core::flatten(mlx::core::astype(o_projection, mlx::core::float32));
                    mlx::core::eval(layer_input_flat, o_projection_flat);
                    timing_buckets.readback_ms += elapsed_ms_since(direct_post_residual_readback_start);
                    timing_buckets.readback_count += 2;
                    std::vector<float> residual_values(static_cast<size_t>(hidden_size), 0.0f);
                    GypsyDirectMetalProbeResult direct_post_residual = GypsyRunDirectMetalAddResidualToHost(
                        layer_input_flat.data<float>(),
                        o_projection_flat.data<float>(),
                        residual_values.data(),
                        static_cast<std::uint32_t>(hidden_size));
                    timing_buckets.direct_metal_layer35_post_residual_ms += elapsed_ms_since(direct_post_residual_start);
                    if (direct_post_residual.ok) {
                        timing_buckets.direct_metal_layer35_post_residual_calls += 1;
                        timing_buckets.direct_metal_layer35_post_residual_checksum = direct_post_residual.checksum;
                        direct_post_residual_values = std::move(residual_values);
                        post_attention_residual = mlx::core::array(
                            direct_post_residual_values->begin(),
                            mlx::core::Shape{1, hidden_size},
                            mlx::core::float32);
                    }
                }
                if (!direct_post_residual_values.has_value()) {
                    post_attention_residual = fused_post_attention_residual.has_value()
                        ? fused_post_attention_residual.value()
                        : mlx::core::astype(layer_input + o_projection, compute_dtype);
                }
                std::optional<std::vector<float>> direct_post_norm_values;
                mlx::core::array mlp_norm = mlx::core::array(0.0f);
                if (use_direct_metal_layer35_post_norm &&
                    direct_terminal_layer &&
                    post_norm != nullptr &&
                    post_norm->raw_data != nullptr) {
                    const auto direct_post_norm_start = std::chrono::steady_clock::now();
                    const float* post_attention_residual_values = nullptr;
                    mlx::core::array post_attention_residual_flat = mlx::core::array(0.0f);
                    if (direct_post_residual_values.has_value()) {
                        post_attention_residual_values = direct_post_residual_values->data();
                    } else {
                        const auto direct_post_norm_readback_start = std::chrono::steady_clock::now();
                        post_attention_residual_flat = mlx::core::flatten(mlx::core::astype(post_attention_residual, mlx::core::float32));
                        mlx::core::eval(post_attention_residual_flat);
                        timing_buckets.readback_ms += elapsed_ms_since(direct_post_norm_readback_start);
                        timing_buckets.readback_count += 1;
                        post_attention_residual_values = post_attention_residual_flat.data<float>();
                    }
                    std::vector<float> norm_values(static_cast<size_t>(hidden_size), 0.0f);
                    GypsyDirectMetalProbeResult direct_post_norm = GypsyRunDirectMetalRmsNormToHost(
                        post_attention_residual_values,
                        static_cast<const std::uint16_t*>(post_norm->raw_data),
                        norm_values.data(),
                        static_cast<std::uint32_t>(hidden_size));
                    timing_buckets.direct_metal_layer35_post_norm_ms += elapsed_ms_since(direct_post_norm_start);
                    if (direct_post_norm.ok) {
                        timing_buckets.direct_metal_layer35_post_norm_calls += 1;
                        timing_buckets.direct_metal_layer35_post_norm_checksum = direct_post_norm.checksum;
                        direct_post_norm_values = std::move(norm_values);
                        mlp_norm = mlx::core::array(
                            direct_post_norm_values->begin(),
                            mlx::core::Shape{1, hidden_size},
                            mlx::core::float32);
                    }
                }
                if (!direct_post_norm_values.has_value()) {
                    mlp_norm = mlx::core::astype(
                        mlx::core::fast::rms_norm(post_attention_residual, std::make_optional(post_norm->array), eps),
                        compute_dtype);
                }
                timing_buckets.post_attention_residual_norm_ms += elapsed_ms_since(residual_norm_start);
                const auto mlp_gate_up_start = std::chrono::steady_clock::now();
                std::optional<std::vector<float>> direct_gate_values;
                std::optional<std::vector<float>> direct_up_values;
                mlx::core::array gate = mlx::core::array(0.0f);
                mlx::core::array up = mlx::core::array(0.0f);
                const bool gate_projection_has_adapter =
                    adapter != nullptr &&
                    require_logits_array(p + "mlp.gate_proj.lora_a") != nullptr &&
                    require_logits_array(p + "mlp.gate_proj.lora_b") != nullptr;
                const bool up_projection_has_adapter =
                    adapter != nullptr &&
                    require_logits_array(p + "mlp.up_proj.lora_a") != nullptr &&
                    require_logits_array(p + "mlp.up_proj.lora_b") != nullptr;
                if (use_direct_metal_layer35_gate_up &&
                    direct_terminal_layer &&
                    direct_post_norm_values.has_value() &&
                    !gate_projection_has_adapter &&
                    !up_projection_has_adapter &&
                    gw != nullptr && gs != nullptr && gb != nullptr &&
                    uw != nullptr && us != nullptr && ub != nullptr &&
                    gw->raw_data != nullptr && gs->raw_data != nullptr && gb->raw_data != nullptr &&
                    uw->raw_data != nullptr && us->raw_data != nullptr && ub->raw_data != nullptr) {
                    const auto direct_gate_up_start = std::chrono::steady_clock::now();
                    const float* mlp_norm_values = direct_post_norm_values->data();
                    std::vector<float> gate_values(static_cast<size_t>(intermediate_size), 0.0f);
                    std::vector<float> up_values(static_cast<size_t>(intermediate_size), 0.0f);
                    GypsyDirectMetalProbeResult direct_gate = GypsyRunDirectMetalQuantizedProjectionTiledToHost(
                        mlp_norm_values,
                        static_cast<const std::uint32_t*>(gw->raw_data),
                        static_cast<const std::uint16_t*>(gs->raw_data),
                        static_cast<const std::uint16_t*>(gb->raw_data),
                        gate_values.data(),
                        static_cast<std::uint32_t>(intermediate_size),
                        static_cast<std::uint32_t>(hidden_size / 8),
                        static_cast<std::uint32_t>(hidden_size));
                    GypsyDirectMetalProbeResult direct_up = GypsyRunDirectMetalQuantizedProjectionTiledToHost(
                        mlp_norm_values,
                        static_cast<const std::uint32_t*>(uw->raw_data),
                        static_cast<const std::uint16_t*>(us->raw_data),
                        static_cast<const std::uint16_t*>(ub->raw_data),
                        up_values.data(),
                        static_cast<std::uint32_t>(intermediate_size),
                        static_cast<std::uint32_t>(hidden_size / 8),
                        static_cast<std::uint32_t>(hidden_size));
                    timing_buckets.direct_metal_layer35_gate_up_ms += elapsed_ms_since(direct_gate_up_start);
                    if (direct_gate.ok && direct_up.ok) {
                        timing_buckets.direct_metal_layer35_gate_up_calls += 1;
                        timing_buckets.direct_metal_layer35_gate_checksum = direct_gate.checksum;
                        timing_buckets.direct_metal_layer35_up_checksum = direct_up.checksum;
                        direct_gate_values = std::move(gate_values);
                        direct_up_values = std::move(up_values);
                        gate = mlx::core::array(direct_gate_values->begin(), mlx::core::Shape{1, intermediate_size}, mlx::core::float32);
                        up = mlx::core::array(direct_up_values->begin(), mlx::core::Shape{1, intermediate_size}, mlx::core::float32);
                    }
                }
                if (!direct_gate_values.has_value() || !direct_up_values.has_value()) {
                    gate = q_project(mlp_norm, gw, gs, gb, p + "mlp.gate_proj");
                    up = q_project(mlp_norm, uw, us, ub, p + "mlp.up_proj");
                }
                timing_buckets.mlp_gate_up_ms += elapsed_ms_since(mlp_gate_up_start);
                timing_buckets.quantized_projection_calls += 2;
                const auto mlp_activation_start = std::chrono::steady_clock::now();
                mlx::core::array activated = mlx::core::astype((gate * mlx::core::sigmoid(gate)) * up, compute_dtype);
                timing_buckets.mlp_activation_ms += elapsed_ms_since(mlp_activation_start);
                const auto mlp_down_start = std::chrono::steady_clock::now();
                mlx::core::array activated_input = mlx::core::reshape(activated, mlx::core::Shape{1, intermediate_size});
                std::optional<std::vector<float>> direct_activation_values;
                if (use_direct_metal_layer35_activation && direct_terminal_layer) {
                    const auto direct_activation_start = std::chrono::steady_clock::now();
                    const float* gate_values = nullptr;
                    const float* up_values = nullptr;
                    mlx::core::array gate_flat = mlx::core::array(0.0f);
                    mlx::core::array up_flat = mlx::core::array(0.0f);
                    if (direct_gate_values.has_value() && direct_up_values.has_value()) {
                        gate_values = direct_gate_values->data();
                        up_values = direct_up_values->data();
                    } else {
                        const auto direct_activation_readback_start = std::chrono::steady_clock::now();
                        gate_flat = mlx::core::flatten(mlx::core::astype(gate, mlx::core::float32));
                        up_flat = mlx::core::flatten(mlx::core::astype(up, mlx::core::float32));
                        mlx::core::eval(gate_flat, up_flat);
                        timing_buckets.readback_ms += elapsed_ms_since(direct_activation_readback_start);
                        timing_buckets.readback_count += 2;
                        gate_values = gate_flat.data<float>();
                        up_values = up_flat.data<float>();
                    }
                    std::vector<float> activation_values(static_cast<size_t>(intermediate_size), 0.0f);
                    GypsyDirectMetalProbeResult direct_activation = GypsyRunDirectMetalSiluMultiplyToHost(
                        gate_values,
                        up_values,
                        activation_values.data(),
                        static_cast<std::uint32_t>(intermediate_size));
                    timing_buckets.direct_metal_layer35_activation_ms += elapsed_ms_since(direct_activation_start);
                    if (direct_activation.ok) {
                        timing_buckets.direct_metal_layer35_activation_calls += 1;
                        timing_buckets.direct_metal_layer35_activation_checksum = direct_activation.checksum;
                        direct_activation_values = std::move(activation_values);
                        activated_input = mlx::core::array(
                            direct_activation_values->begin(),
                            mlx::core::Shape{1, intermediate_size},
                            mlx::core::float32);
                    }
                }
                mlx::core::array down = mlx::core::array(0.0f);
                bool direct_layer35_down_used = false;
                const bool down_projection_has_adapter =
                    adapter != nullptr &&
                    require_logits_array(p + "mlp.down_proj.lora_a") != nullptr &&
                    require_logits_array(p + "mlp.down_proj.lora_b") != nullptr;
                if (use_direct_metal_layer35_down &&
                    direct_terminal_layer &&
                    !down_projection_has_adapter &&
                    dw != nullptr && ds != nullptr && db != nullptr &&
                    dw->raw_data != nullptr && ds->raw_data != nullptr && db->raw_data != nullptr) {
                    const auto direct_down_start = std::chrono::steady_clock::now();
                    const auto direct_down_readback_start = std::chrono::steady_clock::now();
                    const float* activated_values = nullptr;
                    mlx::core::array activated_flat = mlx::core::array(0.0f);
                    if (direct_activation_values.has_value()) {
                        activated_values = direct_activation_values->data();
                    } else {
                        activated_flat = mlx::core::flatten(mlx::core::astype(activated_input, mlx::core::float32));
                        mlx::core::eval(activated_flat);
                        timing_buckets.readback_ms += elapsed_ms_since(direct_down_readback_start);
                        timing_buckets.readback_count += 1;
                        activated_values = activated_flat.data<float>();
                    }
                    std::vector<float> down_values(static_cast<size_t>(hidden_size), 0.0f);
                    GypsyDirectMetalProbeResult direct_down = GypsyRunDirectMetalQuantizedProjectionTiledToHost(
                        activated_values,
                        static_cast<const std::uint32_t*>(dw->raw_data),
                        static_cast<const std::uint16_t*>(ds->raw_data),
                        static_cast<const std::uint16_t*>(db->raw_data),
                        down_values.data(),
                        static_cast<std::uint32_t>(hidden_size),
                        static_cast<std::uint32_t>(intermediate_size / 8),
                        static_cast<std::uint32_t>(intermediate_size));
                    timing_buckets.direct_metal_layer35_down_ms += elapsed_ms_since(direct_down_start);
                    if (direct_down.ok) {
                        direct_layer35_down_used = true;
                        timing_buckets.direct_metal_layer35_down_calls += 1;
                        timing_buckets.direct_metal_layer35_down_checksum = direct_down.checksum;
                        down = mlx::core::array(down_values.begin(), mlx::core::Shape{1, hidden_size}, mlx::core::float32);
                    }
                }
                if (!direct_layer35_down_used) {
                    down = q_project(activated_input, dw, ds, db, p + "mlp.down_proj");
                }
                mlx::core::array output = mlx::core::astype(post_attention_residual + down, compute_dtype);
                timing_buckets.mlp_down_residual_ms += elapsed_ms_since(mlp_down_start);
                timing_buckets.quantized_projection_calls += 1;
                timing_buckets.layer_total_ms += elapsed_ms_since(layer_start);
                return output;
            };
            auto run_direct_metal_terminal_stack = [&](int start_layer, const mlx::core::array& stack_input) -> std::optional<mlx::core::array> {
                if (!use_direct_metal_full_layer || start_layer < 0 || start_layer > 35) {
                    return std::nullopt;
                }
                const auto stack_readback_start = std::chrono::steady_clock::now();
                mlx::core::array input_flat = mlx::core::flatten(mlx::core::astype(stack_input, mlx::core::float32));
                mlx::core::eval(input_flat);
                timing_buckets.readback_ms += elapsed_ms_since(stack_readback_start);
                timing_buckets.readback_count += 1;
                std::vector<float> final_values(static_cast<size_t>(hidden_size), 0.0f);
                const float* input_values = input_flat.data<float>();
                std::vector<GypsyDirectMetalLayerParams> layer_params;
                layer_params.reserve(static_cast<size_t>(36 - start_layer));

                for (int layer = start_layer; layer <= 35; ++layer) {
                    const std::string p = "model.layers." + std::to_string(layer) + ".";
                    const ResidentArrayRecord* input_norm = require_logits_array(p + "input_layernorm.weight");
                    const ResidentArrayRecord* q_norm = require_logits_array(p + "self_attn.q_norm.weight");
                    const ResidentArrayRecord* k_norm = require_logits_array(p + "self_attn.k_norm.weight");
                    const ResidentArrayRecord* qw = require_logits_array(p + "self_attn.q_proj.weight");
                    const ResidentArrayRecord* qs = require_logits_array(p + "self_attn.q_proj.scales");
                    const ResidentArrayRecord* qb = require_logits_array(p + "self_attn.q_proj.biases");
                    const ResidentArrayRecord* kw = require_logits_array(p + "self_attn.k_proj.weight");
                    const ResidentArrayRecord* ks = require_logits_array(p + "self_attn.k_proj.scales");
                    const ResidentArrayRecord* kb = require_logits_array(p + "self_attn.k_proj.biases");
                    const ResidentArrayRecord* vw = require_logits_array(p + "self_attn.v_proj.weight");
                    const ResidentArrayRecord* vs = require_logits_array(p + "self_attn.v_proj.scales");
                    const ResidentArrayRecord* vb = require_logits_array(p + "self_attn.v_proj.biases");
                    const ResidentArrayRecord* ow = require_logits_array(p + "self_attn.o_proj.weight");
                    const ResidentArrayRecord* os = require_logits_array(p + "self_attn.o_proj.scales");
                    const ResidentArrayRecord* ob = require_logits_array(p + "self_attn.o_proj.biases");
                    const ResidentArrayRecord* post_norm = require_logits_array(p + "post_attention_layernorm.weight");
                    const ResidentArrayRecord* gw = require_logits_array(p + "mlp.gate_proj.weight");
                    const ResidentArrayRecord* gs = require_logits_array(p + "mlp.gate_proj.scales");
                    const ResidentArrayRecord* gb = require_logits_array(p + "mlp.gate_proj.biases");
                    const ResidentArrayRecord* uw = require_logits_array(p + "mlp.up_proj.weight");
                    const ResidentArrayRecord* us = require_logits_array(p + "mlp.up_proj.scales");
                    const ResidentArrayRecord* ub = require_logits_array(p + "mlp.up_proj.biases");
                    const ResidentArrayRecord* dw = require_logits_array(p + "mlp.down_proj.weight");
                    const ResidentArrayRecord* ds = require_logits_array(p + "mlp.down_proj.scales");
                    const ResidentArrayRecord* db = require_logits_array(p + "mlp.down_proj.biases");
                    const bool projection_has_adapter =
                        adapter != nullptr &&
                        (require_logits_array(p + "self_attn.q_proj.lora_a") != nullptr ||
                         require_logits_array(p + "self_attn.k_proj.lora_a") != nullptr ||
                         require_logits_array(p + "self_attn.v_proj.lora_a") != nullptr ||
                         require_logits_array(p + "self_attn.o_proj.lora_a") != nullptr ||
                         require_logits_array(p + "mlp.gate_proj.lora_a") != nullptr ||
                         require_logits_array(p + "mlp.up_proj.lora_a") != nullptr ||
                         require_logits_array(p + "mlp.down_proj.lora_a") != nullptr);
                    if (projection_has_adapter ||
                        input_norm == nullptr || q_norm == nullptr || k_norm == nullptr ||
                        qw == nullptr || qs == nullptr || qb == nullptr ||
                        kw == nullptr || ks == nullptr || kb == nullptr ||
                        vw == nullptr || vs == nullptr || vb == nullptr ||
                        ow == nullptr || os == nullptr || ob == nullptr ||
                        post_norm == nullptr ||
                        gw == nullptr || gs == nullptr || gb == nullptr ||
                        uw == nullptr || us == nullptr || ub == nullptr ||
                        dw == nullptr || ds == nullptr || db == nullptr ||
                        input_norm->raw_data == nullptr || q_norm->raw_data == nullptr || k_norm->raw_data == nullptr ||
                        qw->raw_data == nullptr || qs->raw_data == nullptr || qb->raw_data == nullptr ||
                        kw->raw_data == nullptr || ks->raw_data == nullptr || kb->raw_data == nullptr ||
                        vw->raw_data == nullptr || vs->raw_data == nullptr || vb->raw_data == nullptr ||
                        ow->raw_data == nullptr || os->raw_data == nullptr || ob->raw_data == nullptr ||
                        post_norm->raw_data == nullptr ||
                        gw->raw_data == nullptr || gs->raw_data == nullptr || gb->raw_data == nullptr ||
                        uw->raw_data == nullptr || us->raw_data == nullptr || ub->raw_data == nullptr ||
                        dw->raw_data == nullptr || ds->raw_data == nullptr || db->raw_data == nullptr) {
                        return std::nullopt;
                    }

                    LocalLayerKvCache& cache = local_kv_cache[static_cast<size_t>(layer)];
                    cache.len += 1;
                    kv_cache_materialized = true;
                    kv_cache_positions = std::max(kv_cache_positions, cache.len);
                    GypsyDirectMetalLayerParams params;
                    params.layer = static_cast<std::uint32_t>(layer);
                    params.seq_len = cache.len;
                    params.position = static_cast<std::uint32_t>(position);
                    params.input_norm_weight = static_cast<const std::uint16_t*>(input_norm->raw_data);
                    params.q_weight = static_cast<const std::uint32_t*>(qw->raw_data);
                    params.q_scales = static_cast<const std::uint16_t*>(qs->raw_data);
                    params.q_biases = static_cast<const std::uint16_t*>(qb->raw_data);
                    params.k_weight = static_cast<const std::uint32_t*>(kw->raw_data);
                    params.k_scales = static_cast<const std::uint16_t*>(ks->raw_data);
                    params.k_biases = static_cast<const std::uint16_t*>(kb->raw_data);
                    params.v_weight = static_cast<const std::uint32_t*>(vw->raw_data);
                    params.v_scales = static_cast<const std::uint16_t*>(vs->raw_data);
                    params.v_biases = static_cast<const std::uint16_t*>(vb->raw_data);
                    params.q_norm_weight = static_cast<const std::uint16_t*>(q_norm->raw_data);
                    params.k_norm_weight = static_cast<const std::uint16_t*>(k_norm->raw_data);
                    params.o_weight = static_cast<const std::uint32_t*>(ow->raw_data);
                    params.o_scales = static_cast<const std::uint16_t*>(os->raw_data);
                    params.o_biases = static_cast<const std::uint16_t*>(ob->raw_data);
                    params.post_norm_weight = static_cast<const std::uint16_t*>(post_norm->raw_data);
                    params.gate_weight = static_cast<const std::uint32_t*>(gw->raw_data);
                    params.gate_scales = static_cast<const std::uint16_t*>(gs->raw_data);
                    params.gate_biases = static_cast<const std::uint16_t*>(gb->raw_data);
                    params.up_weight = static_cast<const std::uint32_t*>(uw->raw_data);
                    params.up_scales = static_cast<const std::uint16_t*>(us->raw_data);
                    params.up_biases = static_cast<const std::uint16_t*>(ub->raw_data);
                    params.down_weight = static_cast<const std::uint32_t*>(dw->raw_data);
                    params.down_scales = static_cast<const std::uint16_t*>(ds->raw_data);
                    params.down_biases = static_cast<const std::uint16_t*>(db->raw_data);
                    layer_params.push_back(params);
                }
                const auto direct_stack_start = std::chrono::steady_clock::now();
                GypsyDirectMetalProbeResult direct = GypsyRunDirectMetalTerminalStackResident(
                    layer_params.data(),
                    static_cast<std::uint32_t>(layer_params.size()),
                    input_values,
                    final_values.data());
                timing_buckets.direct_metal_full_layer_ms += elapsed_ms_since(direct_stack_start);
                if (!direct.ok) {
                    return std::nullopt;
                }
                timing_buckets.direct_metal_full_layer_calls += 1;
                timing_buckets.direct_metal_full_layer_checksum = direct.checksum;
                direct_metal_attention_used_anywhere = true;
                timing_buckets.mlx_attention_calls += static_cast<std::uint64_t>(36 - start_layer);
                timing_buckets.attention_calls += static_cast<std::uint64_t>(36 - start_layer);
                return mlx::core::array(final_values.begin(), mlx::core::Shape{1, hidden_size}, mlx::core::float32);
            };
            for (int layer = 0; layer <= 35; ++layer) {
                if (layer == direct_metal_terminal_layer_start) {
                    std::optional<mlx::core::array> terminal_output = run_direct_metal_terminal_stack(layer, hidden_array);
                    if (terminal_output.has_value()) {
                        hidden_array = terminal_output.value();
                        local_generated_layer_count = 36;
                        if (layer == 0) local_layer0_block_executed = true;
                        if (layer <= 1) local_layer1_block_executed = true;
                        break;
                    }
                }
                hidden_array = run_plain_layer(layer, hidden_array);
                if (layer == 0) local_layer0_block_executed = true;
                local_generated_layer_count = layer + 1;
                if (layer == 1) local_layer1_block_executed = true;
            }
            layer0_block_executed = layer0_block_executed || local_layer0_block_executed;
        }
        if (!keep_top_summary) {
            return input_token_id;
        }
        const auto final_norm_start = std::chrono::steady_clock::now();
        mlx::core::array normalized = mlx::core::array(0.0f);
        std::optional<std::vector<float>> direct_normalized_values;
        if (use_direct_metal_final_norm && final_norm != nullptr && final_norm->raw_data != nullptr) {
            const auto readback_start = std::chrono::steady_clock::now();
            mlx::core::array hidden_flat = mlx::core::flatten(mlx::core::astype(hidden_array, mlx::core::float32));
            mlx::core::eval(hidden_flat);
            timing_buckets.readback_ms += elapsed_ms_since(readback_start);
            timing_buckets.readback_count += 1;
            const float* hidden_values = hidden_flat.data<float>();
            std::vector<float> norm_values(static_cast<size_t>(hidden_size), 0.0f);
            const auto direct_norm_start = std::chrono::steady_clock::now();
            GypsyDirectMetalProbeResult direct_norm = GypsyRunDirectMetalRmsNormToHost(
                hidden_values,
                static_cast<const std::uint16_t*>(final_norm->raw_data),
                norm_values.data(),
                static_cast<std::uint32_t>(hidden_size));
            timing_buckets.direct_metal_final_norm_ms += elapsed_ms_since(direct_norm_start);
            if (direct_norm.ok) {
                timing_buckets.direct_metal_final_norm_calls += 1;
                timing_buckets.direct_metal_final_norm_checksum = direct_norm.checksum;
                direct_normalized_values = std::move(norm_values);
                normalized = mlx::core::array(direct_normalized_values->begin(), mlx::core::Shape{1, hidden_size}, mlx::core::float32);
            }
        }
        if (!direct_normalized_values.has_value()) {
            normalized = mlx::core::astype(
                mlx::core::fast::rms_norm(hidden_array, std::make_optional(final_norm->array), eps),
                compute_dtype);
        }
        timing_buckets.final_norm_ms += elapsed_ms_since(final_norm_start);
        const auto logits_start = std::chrono::steady_clock::now();
        bool direct_logits_used = false;
        if (use_direct_metal_logits &&
            ew != nullptr && es != nullptr && eb != nullptr &&
            ew->raw_data != nullptr && es->raw_data != nullptr && eb->raw_data != nullptr) {
            const auto direct_logits_start = std::chrono::steady_clock::now();
            const float* norm_values = nullptr;
            mlx::core::array normalized_flat = mlx::core::array(0.0f);
            if (direct_normalized_values.has_value()) {
                norm_values = direct_normalized_values->data();
            } else {
                const auto final_readback_start = std::chrono::steady_clock::now();
                normalized_flat = mlx::core::flatten(mlx::core::astype(normalized, mlx::core::float32));
                mlx::core::eval(normalized_flat);
                timing_buckets.readback_ms += elapsed_ms_since(final_readback_start);
                timing_buckets.readback_count += 1;
                norm_values = normalized_flat.data<float>();
            }
            local_final_norm_checksum = 0.0;
            local_final_norm_abs_checksum = 0.0;
            for (int i = 0; i < hidden_size; ++i) {
                const double v = static_cast<double>(norm_values[i]);
                local_final_norm_checksum += v;
                local_final_norm_abs_checksum += std::abs(v);
            }

            GypsyDirectMetalProbeResult direct = GypsyRunDirectMetalQuantizedProjectionTiledTop1(
                norm_values,
                static_cast<const std::uint32_t*>(ew->raw_data),
                static_cast<const std::uint16_t*>(es->raw_data),
                static_cast<const std::uint16_t*>(eb->raw_data),
                static_cast<std::uint32_t>(vocab_size),
                static_cast<std::uint32_t>(hidden_size / 8),
                static_cast<std::uint32_t>(hidden_size));
            timing_buckets.direct_metal_logits_ms += elapsed_ms_since(direct_logits_start);
            if (direct.ok) {
                timing_buckets.direct_metal_logits_calls += 1;
                timing_buckets.direct_metal_logits_checksum = direct.checksum;
                direct_logits_used = true;
                local_logits_checksum = static_cast<double>(direct.checksum);
                local_top_token_id = static_cast<int>(direct.top_id);
                local_top_token_score = direct.top_score;
                local_top.emplace_back(local_top_token_score, local_top_token_id);
            }
        }

        if (!direct_logits_used) {
            mlx::core::array logits = mlx::core::flatten(mlx::core::astype(
                mlx::core::quantized_matmul(
                    normalized,
                    ew->array,
                    es->array,
                    std::make_optional(eb->array),
                    true,
                    group_size,
                    bits,
                    "affine"),
                mlx::core::float32));
            timing_buckets.quantized_projection_calls += 1;

            if (full_logit_summary) {
            const auto final_readback_start = std::chrono::steady_clock::now();
            mlx::core::eval(normalized, logits);
            timing_buckets.readback_ms += elapsed_ms_since(final_readback_start);
            timing_buckets.readback_count += 2;

            const float* norm_values = normalized.data<float>();
            for (int i = 0; i < hidden_size; ++i) {
                const double v = static_cast<double>(norm_values[i]);
                local_final_norm_checksum += v;
                local_final_norm_abs_checksum += std::abs(v);
            }

            const float* values = logits.data<float>();
            for (int i = 0; i < vocab_size; ++i) {
                const float v = values[i];
                local_logits_checksum += static_cast<double>(v);
                if (local_top.size() < 10) {
                    local_top.emplace_back(v, i);
                    std::sort(local_top.begin(), local_top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                } else if (v > local_top.back().first) {
                    local_top.back() = {v, i};
                    std::sort(local_top.begin(), local_top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                }
            }
        } else {
            const auto final_readback_start = std::chrono::steady_clock::now();
            mlx::core::array top_id_array = mlx::core::astype(mlx::core::argmax(logits), mlx::core::int32);
            mlx::core::array top_score_array = mlx::core::flatten(mlx::core::take(logits, top_id_array));
            mlx::core::eval(top_id_array, top_score_array);
            timing_buckets.readback_ms += elapsed_ms_since(final_readback_start);
            timing_buckets.readback_count += 2;
            local_top_token_id = top_id_array.data<int32_t>()[0];
            local_top_token_score = top_score_array.data<float>()[0];
            local_top.emplace_back(local_top_token_score, local_top_token_id);
            }
        }
        if (!local_top.empty()) {
            local_top_token_score = local_top[0].first;
            local_top_token_id = local_top[0].second;
        }
        timing_buckets.logits_topk_ms += elapsed_ms_since(logits_start);
        timing_buckets.tokens_evaluated += 1;
            if (keep_top_summary) {
                top = std::move(local_top);
                top_token_id = local_top_token_id;
                top_token_score = local_top_token_score;
                logits_checksum = local_logits_checksum;
                final_norm_checksum = local_final_norm_checksum;
                final_norm_abs_checksum = local_final_norm_abs_checksum;
                layer0_block_executed = local_layer0_block_executed;
                layer1_block_executed = local_layer1_block_executed;
                generated_layer_count = local_generated_layer_count;
            }
            return static_cast<std::uint32_t>(std::max(local_top_token_id, 0));
        };
        if (prompt_tokens.size() > 1) {
            prompt_token_prefill_loop_executed = true;
            prompt_prefill_tokens_processed = static_cast<std::uint32_t>(prompt_tokens.size());
            std::uint32_t position = 0;
            for (size_t i = 0; i + 1 < prompt_tokens.size(); ++i) {
                (void)run_one_token(prompt_tokens[i], position++, false);
            }
            current_token_id = prompt_tokens.back();
            for (std::uint32_t i = 0; i < loop_limit; ++i) {
                current_token_id = run_one_token(current_token_id, position++, true);
                generated_tokens.push_back(current_token_id);
                if (is_generation_stop_token(current_token_id)) {
                    generation_stopped_by_token = true;
                    generation_stop_token_id = current_token_id;
                    generation_stop_reason = current_token_id == 151645U ? "eos_im_end" : "eos_endoftext";
                    break;
                }
            }
        } else if (prompt_tokens.size() == 1) {
            prompt_prefill_tokens_processed = 1;
            std::uint32_t position = 0;
            for (std::uint32_t i = 0; i < loop_limit; ++i) {
                current_token_id = run_one_token(current_token_id, position++, true);
                generated_tokens.push_back(current_token_id);
                if (is_generation_stop_token(current_token_id)) {
                    generation_stopped_by_token = true;
                    generation_stop_token_id = current_token_id;
                    generation_stop_reason = current_token_id == 151645U ? "eos_im_end" : "eos_endoftext";
                    break;
                }
            }
        } else {
            for (std::uint32_t i = 0; i < loop_limit; ++i) {
                current_token_id = run_one_token(current_token_id, i, true);
                generated_tokens.push_back(current_token_id);
                if (is_generation_stop_token(current_token_id)) {
                    generation_stopped_by_token = true;
                    generation_stop_token_id = current_token_id;
                    generation_stop_reason = current_token_id == 151645U ? "eos_im_end" : "eos_endoftext";
                    break;
                }
            }
        }
    }
    auto generation_end = std::chrono::steady_clock::now();

    Napi::Array generated_token_ids = Napi::Array::New(env, generated_tokens.size());
    for (size_t i = 0; i < generated_tokens.size(); ++i) {
        generated_token_ids.Set(static_cast<uint32_t>(i), Napi::Number::New(env, generated_tokens[i]));
    }
    Napi::Array top_tokens = Napi::Array::New(env, top.size());
    for (size_t i = 0; i < top.size(); ++i) {
        Napi::Object item = Napi::Object::New(env);
        item.Set("rank", Napi::Number::New(env, static_cast<double>(i + 1)));
        item.Set("token_id", Napi::Number::New(env, top[i].second));
        item.Set("score", Napi::Number::New(env, top[i].first));
        top_tokens.Set(static_cast<uint32_t>(i), item);
    }
    std::string generated_text;
    std::string raw_decoded_text;
    std::string cleaned_decoded_text;
    bool clean_stop_seen = false;
    for (size_t i = 0; i < generated_tokens.size(); ++i) {
        if (i > 0) generated_text += " ";
        generated_text += "<token:" + std::to_string(generated_tokens[i]) + ">";
        auto piece_it = tokenizer_it->second.id_to_token.find(generated_tokens[i]);
        const std::string decoded_piece = piece_it == tokenizer_it->second.id_to_token.end()
            ? DecodeKnownQwenToken(generated_tokens[i])
            : QwenVocabTextToPromptText(piece_it->second);
        raw_decoded_text += decoded_piece;
        if (!clean_stop_seen && is_generation_stop_token(generated_tokens[i])) {
            clean_stop_seen = true;
        } else if (!clean_stop_seen) {
            cleaned_decoded_text += decoded_piece;
        }
    }

    Napi::Object scaffold = Napi::Object::New(env);
    scaffold.Set("enabled", Napi::Boolean::New(env, false));
    scaffold.Set("model_math_executed", Napi::Boolean::New(env, true));
    scaffold.Set("logits_computed", Napi::Boolean::New(env, generated_count > 0));
    scaffold.Set("purpose", Napi::String::New(env, "disabled after first logits-backed generate milestone"));
    scaffold.Set("replacement_step", Napi::String::New(env, "replace_selected_hidden_contract_with_full_prompt_layer_stack"));

    Napi::Object logits_summary = Napi::Object::New(env);
    logits_summary.Set("owner", Napi::String::New(env, "native"));
    logits_summary.Set("source", Napi::String::New(env, token_embedding_lookup_executed
        ? "prompt_token_embedding_final_norm_tied_embedding_logits"
        : "prompt_conditioned_selected_final_norm_tied_embedding_logits"));
    logits_summary.Set("input_contract", Napi::String::New(env, token_embedding_lookup_executed
        ? "last_prompt_token_embedding_float32_1x2560"
        : "synthetic_prompt_conditioned_hidden_float32_1x2560"));
    logits_summary.Set("prompt_conditioned_hidden", Napi::Boolean::New(env, true));
    logits_summary.Set("token_embedding_lookup_executed", Napi::Boolean::New(env, token_embedding_lookup_executed));
    logits_summary.Set("native_text_tokenization_executed", Napi::Boolean::New(env, native_text_tokenization_executed));
    logits_summary.Set("native_text_tokenization_mode", Napi::String::New(env, native_text_tokenization_mode));
    logits_summary.Set("effective_prompt_token_count", Napi::Number::New(env, static_cast<double>(prompt_tokens.size())));
    logits_summary.Set("prompt_token_prefill_loop_executed", Napi::Boolean::New(env, prompt_token_prefill_loop_executed));
    logits_summary.Set("prompt_prefill_tokens_processed", Napi::Number::New(env, static_cast<double>(prompt_prefill_tokens_processed)));
    logits_summary.Set("prompt_prefill_logits_skipped", Napi::Boolean::New(env, prompt_token_prefill_loop_executed));
    logits_summary.Set("kv_cache_materialized", Napi::Boolean::New(env, kv_cache_materialized));
    logits_summary.Set("kv_cache_backend", Napi::String::New(env,
        direct_metal_attention_used_anywhere ? (direct_metal_resident_kv
            ? "native_metal_resident_per_layer_direct_metal_attention"
            : "native_cpu_float_per_layer_direct_metal_attention")
        : mlx_attention_used_anywhere ? "native_mlx_expanded_q_heads_per_layer"
        : (kv_cache_materialized ? "native_cpu_float_per_layer" : "none")));
    logits_summary.Set("direct_metal_cpu_kv_mirror_enabled", Napi::Boolean::New(env, direct_metal_keep_cpu_kv));
    logits_summary.Set("direct_metal_resident_kv_enabled", Napi::Boolean::New(env, direct_metal_resident_kv));
    logits_summary.Set("kv_cache_positions", Napi::Number::New(env, static_cast<double>(kv_cache_positions)));
    logits_summary.Set("layer0_block_executed", Napi::Boolean::New(env, layer0_block_executed));
    logits_summary.Set("layer1_block_executed", Napi::Boolean::New(env, layer1_block_executed));
    logits_summary.Set("layer_loop_start", Napi::Number::New(env, layer0_block_executed ? 0 : -1));
    logits_summary.Set("layer_loop_end_inclusive", Napi::Number::New(env, generated_layer_count > 0 ? generated_layer_count - 1 : -1));
    logits_summary.Set("layer0_attention_mode", Napi::String::New(env,
        layer0_block_executed
            ? (mlx_attention_used_anywhere ? "mlx_scaled_dot_product_attention_expanded_kv" : "cached_qk_softmax_v")
            : "not_executed"));
    logits_summary.Set("layers_executed", Napi::Number::New(env, generated_layer_count));
    logits_summary.Set("embedding_token_id", embedding_token_id >= 0 ? Napi::Number::New(env, static_cast<double>(embedding_token_id)).As<Napi::Value>() : env.Null());
    logits_summary.Set("prompt_signature", Napi::Number::New(env, static_cast<double>(prompt_signature)));
    logits_summary.Set("full_prompt_prefill_executed", Napi::Boolean::New(env, prompt_token_prefill_loop_executed && kv_cache_materialized));
    logits_summary.Set("full_layer_stack_executed", Napi::Boolean::New(env, generated_layer_count == 36));
    logits_summary.Set("adapter_requested", Napi::Boolean::New(env, adapter != nullptr));
    logits_summary.Set("adapter_active", Napi::Boolean::New(env, adapter != nullptr && adapter_applied_projection_count > 0));
    logits_summary.Set("adapter_scale", Napi::Number::New(env, adapter == nullptr ? 0.0 : generation_adapter_scale));
    logits_summary.Set("adapter_layer_range", Napi::String::New(env, adapter == nullptr ? "none" : "20..35"));
    logits_summary.Set("adapter_applied_projection_count", Napi::Number::New(env, static_cast<double>(adapter_applied_projection_count)));
    logits_summary.Set("adapter_projection_application_unit", Napi::String::New(env, "per_layer_per_token_projection_call"));
    logits_summary.Set("adapter_applied_to_prefill", Napi::Boolean::New(env, adapter != nullptr && prompt_prefill_tokens_processed > 1 && adapter_applied_projection_count > 0));
    logits_summary.Set("adapter_applied_to_decode", Napi::Boolean::New(env, adapter != nullptr && !generated_tokens.empty() && adapter_applied_projection_count > 0));
    logits_summary.Set("adapter_fallback_used", Napi::Boolean::New(env, false));
    logits_summary.Set("final_norm_applied", Napi::Boolean::New(env, generated_count > 0));
    logits_summary.Set("logits_projection", Napi::String::New(env, "tied_model_embed_tokens_quantized_matmul"));
    logits_summary.Set("logits_len", Napi::Number::New(env, vocab_size));
    logits_summary.Set("top_token_id", Napi::Number::New(env, top_token_id));
    logits_summary.Set("top_token_score", Napi::Number::New(env, top_token_score));
    logits_summary.Set("top_tokens", top_tokens);
    logits_summary.Set("logits_checksum", Napi::Number::New(env, logits_checksum));
    logits_summary.Set("final_norm_checksum", Napi::Number::New(env, final_norm_checksum));
    logits_summary.Set("final_norm_abs_checksum", Napi::Number::New(env, final_norm_abs_checksum));
    logits_summary.Set("payload_copied_to_coffeescript", Napi::Boolean::New(env, false));
    logits_summary.Set("full_logits_returned_to_coffeescript", Napi::Boolean::New(env, false));
    logits_summary.Set("readback_count", Napi::Number::New(env, static_cast<double>(timing_buckets.readback_count)));
    logits_summary.Set("full_logit_summary_enabled", Napi::Boolean::New(env, full_logit_summary));
    logits_summary.Set("greedy_selection_backend", Napi::String::New(env, full_logit_summary ? "host_full_logits_scan" : "mlx_argmax_scalar_readback"));

    Napi::Object active_timing = Napi::Object::New(env);
    active_timing.Set("direct_metal_terminal_layer_start", Napi::Number::New(env, direct_metal_terminal_layer_start));
    active_timing.Set("embedding_ms", Napi::Number::New(env, timing_buckets.embedding_ms));
    active_timing.Set("input_norm_ms", Napi::Number::New(env, timing_buckets.input_norm_ms));
    active_timing.Set("direct_metal_layer35_input_norm_enabled", Napi::Boolean::New(env, use_direct_metal_layer35_input_norm));
    active_timing.Set("direct_metal_layer35_input_norm_ms", Napi::Number::New(env, timing_buckets.direct_metal_layer35_input_norm_ms));
    active_timing.Set("direct_metal_layer35_input_norm_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.direct_metal_layer35_input_norm_calls)));
    active_timing.Set("direct_metal_layer35_input_norm_checksum", Napi::Number::New(env, timing_buckets.direct_metal_layer35_input_norm_checksum));
    active_timing.Set("direct_metal_fused_mlp_tail_enabled", Napi::Boolean::New(env, use_direct_metal_fused_mlp_tail));
    active_timing.Set("direct_metal_fused_mlp_tail_ms", Napi::Number::New(env, timing_buckets.direct_metal_fused_mlp_tail_ms));
    active_timing.Set("direct_metal_fused_mlp_tail_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.direct_metal_fused_mlp_tail_calls)));
    active_timing.Set("direct_metal_fused_mlp_tail_checksum", Napi::Number::New(env, timing_buckets.direct_metal_fused_mlp_tail_checksum));
    active_timing.Set("direct_metal_full_layer_enabled", Napi::Boolean::New(env, use_direct_metal_full_layer));
    active_timing.Set("direct_metal_full_layer_ms", Napi::Number::New(env, timing_buckets.direct_metal_full_layer_ms));
    active_timing.Set("direct_metal_full_layer_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.direct_metal_full_layer_calls)));
    active_timing.Set("direct_metal_full_layer_checksum", Napi::Number::New(env, timing_buckets.direct_metal_full_layer_checksum));
    active_timing.Set("direct_metal_full_layer_resident_stack", Napi::Boolean::New(env, timing_buckets.direct_metal_full_layer_calls > 0));
    active_timing.Set("qkv_projection_ms", Napi::Number::New(env, timing_buckets.qkv_projection_ms));
    active_timing.Set("qk_norm_rope_ms", Napi::Number::New(env, timing_buckets.qk_norm_rope_ms));
    active_timing.Set("kv_readback_append_ms", Napi::Number::New(env, timing_buckets.kv_readback_append_ms));
    active_timing.Set("attention_cpu_ms", Napi::Number::New(env, timing_buckets.attention_cpu_ms));
    active_timing.Set("mlx_kv_append_ms", Napi::Number::New(env, timing_buckets.mlx_kv_append_ms));
    active_timing.Set("attention_mlx_ms", Napi::Number::New(env, timing_buckets.attention_mlx_ms));
    active_timing.Set("o_projection_ms", Napi::Number::New(env, timing_buckets.o_projection_ms));
    active_timing.Set("fused_attention_o_ms", Napi::Number::New(env, timing_buckets.fused_attention_o_ms));
    active_timing.Set("direct_metal_layer35_attention_o_enabled", Napi::Boolean::New(env, use_direct_metal_layer35_attention_o));
    active_timing.Set("direct_metal_layer35_attention_o_ms", Napi::Number::New(env, timing_buckets.direct_metal_layer35_attention_o_ms));
    active_timing.Set("direct_metal_layer35_attention_o_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.direct_metal_layer35_attention_o_calls)));
    active_timing.Set("direct_metal_layer35_attention_o_checksum", Napi::Number::New(env, timing_buckets.direct_metal_layer35_attention_o_checksum));
    active_timing.Set("direct_metal_layer35_qkv_attention_o_enabled", Napi::Boolean::New(env, use_direct_metal_layer35_qkv_attention_o));
    active_timing.Set("direct_metal_layer35_qkv_attention_o_ms", Napi::Number::New(env, timing_buckets.direct_metal_layer35_qkv_attention_o_ms));
    active_timing.Set("direct_metal_layer35_qkv_attention_o_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.direct_metal_layer35_qkv_attention_o_calls)));
    active_timing.Set("direct_qkv_attention_o_q_checksum", Napi::Number::New(env, timing_buckets.direct_qkv_attention_o_q_checksum));
    active_timing.Set("direct_qkv_attention_o_k_checksum", Napi::Number::New(env, timing_buckets.direct_qkv_attention_o_k_checksum));
    active_timing.Set("direct_qkv_attention_o_v_checksum", Napi::Number::New(env, timing_buckets.direct_qkv_attention_o_v_checksum));
    active_timing.Set("direct_qkv_attention_o_attention_checksum", Napi::Number::New(env, timing_buckets.direct_qkv_attention_o_attention_checksum));
    active_timing.Set("direct_qkv_attention_o_output_checksum", Napi::Number::New(env, timing_buckets.direct_qkv_attention_o_output_checksum));
    active_timing.Set("direct_metal_layer35_post_residual_enabled", Napi::Boolean::New(env, use_direct_metal_layer35_post_residual));
    active_timing.Set("direct_metal_layer35_post_residual_ms", Napi::Number::New(env, timing_buckets.direct_metal_layer35_post_residual_ms));
    active_timing.Set("direct_metal_layer35_post_residual_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.direct_metal_layer35_post_residual_calls)));
    active_timing.Set("direct_metal_layer35_post_residual_checksum", Napi::Number::New(env, timing_buckets.direct_metal_layer35_post_residual_checksum));
    active_timing.Set("post_attention_residual_norm_ms", Napi::Number::New(env, timing_buckets.post_attention_residual_norm_ms));
    active_timing.Set("direct_metal_layer35_post_norm_enabled", Napi::Boolean::New(env, use_direct_metal_layer35_post_norm));
    active_timing.Set("direct_metal_layer35_post_norm_ms", Napi::Number::New(env, timing_buckets.direct_metal_layer35_post_norm_ms));
    active_timing.Set("direct_metal_layer35_post_norm_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.direct_metal_layer35_post_norm_calls)));
    active_timing.Set("direct_metal_layer35_post_norm_checksum", Napi::Number::New(env, timing_buckets.direct_metal_layer35_post_norm_checksum));
    active_timing.Set("mlp_gate_up_ms", Napi::Number::New(env, timing_buckets.mlp_gate_up_ms));
    active_timing.Set("direct_metal_layer35_gate_up_enabled", Napi::Boolean::New(env, use_direct_metal_layer35_gate_up));
    active_timing.Set("direct_metal_layer35_gate_up_ms", Napi::Number::New(env, timing_buckets.direct_metal_layer35_gate_up_ms));
    active_timing.Set("direct_metal_layer35_gate_up_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.direct_metal_layer35_gate_up_calls)));
    active_timing.Set("direct_metal_layer35_gate_checksum", Napi::Number::New(env, timing_buckets.direct_metal_layer35_gate_checksum));
    active_timing.Set("direct_metal_layer35_up_checksum", Napi::Number::New(env, timing_buckets.direct_metal_layer35_up_checksum));
    active_timing.Set("mlp_activation_ms", Napi::Number::New(env, timing_buckets.mlp_activation_ms));
    active_timing.Set("mlp_down_residual_ms", Napi::Number::New(env, timing_buckets.mlp_down_residual_ms));
    active_timing.Set("direct_metal_layer35_down_enabled", Napi::Boolean::New(env, use_direct_metal_layer35_down));
    active_timing.Set("direct_metal_layer35_down_ms", Napi::Number::New(env, timing_buckets.direct_metal_layer35_down_ms));
    active_timing.Set("direct_metal_layer35_down_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.direct_metal_layer35_down_calls)));
    active_timing.Set("direct_metal_layer35_down_checksum", Napi::Number::New(env, timing_buckets.direct_metal_layer35_down_checksum));
    active_timing.Set("direct_metal_layer35_activation_enabled", Napi::Boolean::New(env, use_direct_metal_layer35_activation));
    active_timing.Set("direct_metal_layer35_activation_ms", Napi::Number::New(env, timing_buckets.direct_metal_layer35_activation_ms));
    active_timing.Set("direct_metal_layer35_activation_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.direct_metal_layer35_activation_calls)));
    active_timing.Set("direct_metal_layer35_activation_checksum", Napi::Number::New(env, timing_buckets.direct_metal_layer35_activation_checksum));
    active_timing.Set("final_norm_ms", Napi::Number::New(env, timing_buckets.final_norm_ms));
    active_timing.Set("direct_metal_final_norm_enabled", Napi::Boolean::New(env, use_direct_metal_final_norm));
    active_timing.Set("direct_metal_final_norm_ms", Napi::Number::New(env, timing_buckets.direct_metal_final_norm_ms));
    active_timing.Set("direct_metal_final_norm_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.direct_metal_final_norm_calls)));
    active_timing.Set("direct_metal_final_norm_checksum", Napi::Number::New(env, timing_buckets.direct_metal_final_norm_checksum));
    active_timing.Set("logits_topk_ms", Napi::Number::New(env, timing_buckets.logits_topk_ms));
    active_timing.Set("direct_metal_logits_enabled", Napi::Boolean::New(env, use_direct_metal_logits));
    active_timing.Set("direct_metal_logits_ms", Napi::Number::New(env, timing_buckets.direct_metal_logits_ms));
    active_timing.Set("direct_metal_logits_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.direct_metal_logits_calls)));
    active_timing.Set("direct_metal_logits_checksum", Napi::Number::New(env, timing_buckets.direct_metal_logits_checksum));
    active_timing.Set("layer_total_ms", Napi::Number::New(env, timing_buckets.layer_total_ms));
    active_timing.Set("readback_ms", Napi::Number::New(env, timing_buckets.readback_ms));
    active_timing.Set("qkv_projection_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.qkv_projection_calls)));
    active_timing.Set("quantized_projection_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.quantized_projection_calls)));
    active_timing.Set("attention_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.attention_calls)));
    active_timing.Set("fused_attention_o_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.fused_attention_o_calls)));
    active_timing.Set("direct_qkv_attention_o_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.direct_qkv_attention_o_calls)));
    active_timing.Set("mlx_attention_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.mlx_attention_calls)));
    active_timing.Set("cpu_attention_calls", Napi::Number::New(env, static_cast<double>(timing_buckets.cpu_attention_calls)));
    active_timing.Set("readback_count_observed", Napi::Number::New(env, static_cast<double>(timing_buckets.readback_count)));
    active_timing.Set("tokens_evaluated", Napi::Number::New(env, static_cast<double>(timing_buckets.tokens_evaluated)));
    active_timing.Set("projection_backend_requested", Napi::String::New(env, projection_backend_requested.empty() ? "default_mlx" : projection_backend_requested));
    active_timing.Set("compiled_metal_qkv_projection_enabled", Napi::Boolean::New(env, use_compiled_metal_qkv_projection));
    active_timing.Set("fused_attention_o_projection_enabled", Napi::Boolean::New(env, use_fused_attention_o_projection));
    active_timing.Set("fused_attention_o_residual_enabled", Napi::Boolean::New(env, use_fused_attention_o_residual));
    active_timing.Set("direct_metal_qkv_attention_o_enabled", Napi::Boolean::New(env, use_direct_metal_qkv_attention_o));
    active_timing.Set("kv_cache_backend", Napi::String::New(env,
        direct_metal_attention_used_anywhere ? (direct_metal_resident_kv
            ? "native_metal_resident_per_layer_direct_metal_attention"
            : "native_cpu_float_per_layer_direct_metal_attention")
        : mlx_attention_used_anywhere ? (attention_backend_requested == "mlx_prealloc_kv" || attention_backend_requested == "metal_kernel_attention"
            ? "native_mlx_preallocated_expanded_q_heads_per_layer"
            : "native_mlx_expanded_q_heads_per_layer")
        : (kv_cache_materialized ? "native_cpu_float_per_layer" : "none")));
    active_timing.Set("direct_metal_cpu_kv_mirror_enabled", Napi::Boolean::New(env, direct_metal_keep_cpu_kv));
    active_timing.Set("direct_metal_resident_kv_enabled", Napi::Boolean::New(env, direct_metal_resident_kv));
    active_timing.Set("attention_backend", Napi::String::New(env,
        direct_metal_attention_used_anywhere && !cpu_attention_used_anywhere ? "direct_metal_objcxx_attention"
        : mlx_attention_used_anywhere && !cpu_attention_used_anywhere ? (attention_backend_requested == "metal_kernel_attention"
            ? "mlx_custom_metal_kernel_attention_preallocated_kv"
            : (attention_backend_requested == "mlx_prealloc_kv"
                ? "mlx_scaled_dot_product_attention_preallocated_expanded_kv"
                : "mlx_scaled_dot_product_attention_expanded_kv"))
        : (mlx_attention_used_anywhere ? "mixed_mlx_with_cpu_fallback" : (kv_cache_materialized ? "cpu_cached_qk_softmax_v" : "none"))));
    logits_summary.Set("active_timing_buckets", active_timing);

    Napi::Object stop_result = Napi::Object::New(env);
    stop_result.Set("stopped", Napi::Boolean::New(env, generation_stopped_by_token || generated_tokens.size() >= requested_count));
    stop_result.Set("stop_reason", Napi::String::New(env, generation_stopped_by_token
        ? generation_stop_reason
        : (requested_count == 0
        ? "max_tokens"
        : (generated_count < requested_count ? "temporary_short_greedy_loop_limit" : "max_tokens"))));
    stop_result.Set("stop_token_id", generation_stopped_by_token
        ? Napi::Number::New(env, generation_stop_token_id).As<Napi::Value>()
        : env.Null());

    Napi::Object timing = Napi::Object::New(env);
    const double generation_timing_ms = std::chrono::duration<double, std::milli>(generation_end - generation_start).count();
    timing.Set("generation_timing_ms", Napi::Number::New(env, generation_timing_ms));
    timing.Set("tokens_per_second", generated_count > 0 && generation_timing_ms > 0.0
        ? Napi::Number::New(env, (static_cast<double>(generated_tokens.size()) * 1000.0) / generation_timing_ms).As<Napi::Value>()
        : env.Null());
    timing.Set("native_loop_executed", Napi::Boolean::New(env, true));
    timing.Set("model_math_executed", Napi::Boolean::New(env, generated_count > 0));
    timing.Set("active_timing_buckets", active_timing);

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, true));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("status", Napi::String::New(env, generated_count > 0 ? "selected_logits_generated" : "no_tokens_requested"));
    out.Set("reason", Napi::String::New(env, generated_count > 0
        ? (token_embedding_lookup_executed
            ? "native generate selected a short greedy token sequence using full prompt prefill plus native per-layer KV cache; backend is correctness-first CPU KV attention"
            : "native generate selected a short greedy token sequence from prompt-conditioned synthetic hidden states; tokenization, KV cache, and full prompt prefill are not wired yet")
        : "max_tokens requested zero tokens"));
    out.Set("payload_loaded", Napi::Boolean::New(env, false));
    out.Set("request_validated", Napi::Boolean::New(env, true));
    out.Set("request_summary", request_summary);
    out.Set("generation_plan", generation_plan);
    out.Set("logits_summary", logits_summary);
    out.Set("generated_token_ids", generated_token_ids);
    out.Set("generated_token_count", Napi::Number::New(env, generated_tokens.size()));
    out.Set("text", Napi::String::New(env, generated_text));
    out.Set("raw_decoded_text", Napi::String::New(env, raw_decoded_text));
    out.Set("cleaned_decoded_text", Napi::String::New(env, cleaned_decoded_text));
    out.Set("stop", stop_result);
    out.Set("timing", timing);
    out.Set("scaffold_generation", scaffold);
    out.Set("model_math_executed", Napi::Boolean::New(env, generated_count > 0));
    out.Set("logits_computed", Napi::Boolean::New(env, generated_count > 0));
    out.Set("next_execution_step", Napi::String::New(env, "replace_selected_hidden_contract_with_full_prompt_layer_stack"));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value FreeSessionProtocol(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected session handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    auto it = gypsy_sessions.find(handle);
    const bool found = it != gypsy_sessions.end();
    std::uint64_t unmapped_file_count = 0;
    std::uint64_t unmapped_file_bytes = 0;
    if (found) {
        unmapped_file_count = it->second.model_mapped_files.size() + it->second.adapter_mapped_files.size();
        unmapped_file_bytes = MappedFilesByteTotal(it->second.model_mapped_files) + MappedFilesByteTotal(it->second.adapter_mapped_files);
        gypsy_sessions.erase(it);
    }

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, found));
    out.Set("session", Napi::String::New(env, handle));
    out.Set("freed", Napi::Boolean::New(env, found));
    out.Set("unmapped_file_count", Napi::Number::New(env, static_cast<double>(unmapped_file_count)));
    out.Set("unmapped_file_bytes", Napi::Number::New(env, static_cast<double>(unmapped_file_bytes)));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value UnloadModelProtocol(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected model handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    const auto referencing_sessions = SessionsReferencingHandle("model", handle);
    if (!referencing_sessions.empty()) {
        Napi::Error::New(env, "Cannot unload model while sessions reference it").ThrowAsJavaScriptException();
        return env.Null();
    }
    const size_t erased = gypsy_models.erase(handle);

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, erased > 0));
    out.Set("model", Napi::String::New(env, handle));
    out.Set("unloaded", Napi::Boolean::New(env, erased > 0));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value UnloadTokenizerProtocol(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected tokenizer handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    const auto referencing_sessions = SessionsReferencingHandle("tokenizer", handle);
    if (!referencing_sessions.empty()) {
        Napi::Error::New(env, "Cannot unload tokenizer while sessions reference it").ThrowAsJavaScriptException();
        return env.Null();
    }
    const size_t erased = gypsy_tokenizers.erase(handle);

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, erased > 0));
    out.Set("tokenizer", Napi::String::New(env, handle));
    out.Set("unloaded", Napi::Boolean::New(env, erased > 0));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

Napi::Value UnloadAdapterProtocol(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected adapter handle").ThrowAsJavaScriptException();
        return env.Null();
    }
    const std::string handle = info[0].As<Napi::String>();
    const auto referencing_sessions = SessionsReferencingHandle("adapter", handle);
    if (!referencing_sessions.empty()) {
        Napi::Error::New(env, "Cannot unload adapter while sessions reference it").ThrowAsJavaScriptException();
        return env.Null();
    }
    const size_t erased = gypsy_adapters.erase(handle);

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, erased > 0));
    out.Set("adapter", Napi::String::New(env, handle));
    out.Set("unloaded", Napi::Boolean::New(env, erased > 0));
    out.Set("handle_counts", GypsyHandleCounts(env));
    return out;
}

// ---------------------------------------------------------
// applyLora(model, loraPath, alpha, merge)
// ---------------------------------------------------------
Napi::Value ApplyLora(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto* m = getModel(info);
    if (!m) return env.Null();

    if (info.Length() < 4 ||
        !info[1].IsString() ||
        !info[2].IsNumber() ||
        !info[3].IsBoolean())
    {
        Napi::TypeError::New(env, "applyLora(model, path, alpha, merge)").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string path = info[1].As<Napi::String>();
    float alpha      = info[2].As<Napi::Number>().FloatValue();
    bool merge       = info[3].As<Napi::Boolean>();

    llm_status st = llm_apply_lora(m, path.c_str(), alpha, merge);

    return Napi::Number::New(env, st);
}

// ---------------------------------------------------------
// resetKV(model)
// ---------------------------------------------------------
Napi::Value ResetKV(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto* m = getModel(info);
    if (!m) return env.Null();

    llm_status st = llm_reset_kv_cache(m);
    return Napi::Number::New(env, st);
}

// ---------------------------------------------------------
// forwardStep(model, tokenId, pos)
// Returns Float32Array logits
// ---------------------------------------------------------
Napi::Value ForwardStep(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto* m = getModel(info);

    if (!m) {
        fprintf(stderr, "[node.forwardStep] EARLY EXIT: null model\n");
        return env.Null();
    }

    if (info.Length() < 3) {
        fprintf(stderr, "[node.forwardStep] EARLY EXIT: not enough args (argc=%zu)\n",
                info.Length());
        Napi::TypeError::New(env, "Expected tokenId, pos").ThrowAsJavaScriptException();
        return env.Null();
    }

    if (!info[1].IsNumber() || !info[2].IsNumber()) {
        fprintf(stderr, "[node.forwardStep] EARLY EXIT: tokenId/pos not numeric\n");
        Napi::TypeError::New(env, "Expected numeric tokenId/pos").ThrowAsJavaScriptException();
        return env.Null();
    }

    int32_t tokenId = info[1].As<Napi::Number>().Int32Value();
    int32_t pos     = info[2].As<Napi::Number>().Int32Value();
    int32_t vocab   = llm_get_vocab_size(m);

    if (vocab <= 0) {
        fprintf(stderr, "[node.forwardStep] EARLY EXIT: invalid vocab size=%d\n", vocab);
        Napi::Error::New(env, "Invalid vocab size").ThrowAsJavaScriptException();
        return env.Null();
    }

    fprintf(stderr, "[node.forwardStep] tokenId=%d pos=%d vocab=%d\n",
            tokenId, pos, vocab);

    // Allocate pinned C memory for logits
    size_t bytes = (size_t)vocab * sizeof(float);
    float* data = (float*)std::malloc(bytes);
    if (!data) {
        fprintf(stderr, "[node.forwardStep] EARLY EXIT: malloc failed (%zu bytes)\n", bytes);
        Napi::Error::New(env, "malloc failed in ForwardStep").ThrowAsJavaScriptException();
        return env.Null();
    }

    memset(data, 0, bytes);

    Napi::ArrayBuffer buf = Napi::ArrayBuffer::New(
        env,
        data,
        bytes,
        [](Napi::Env /*env*/, void* p) { std::free(p); }
    );

    Napi::Float32Array logits = Napi::Float32Array::New(env, vocab, buf, 0);

    llm_status st = llm_forward_step(m, tokenId, pos, data);
    if (st != LLM_OK) {
        fprintf(stderr, "[node.forwardStep] forward_step returned %d\n", st);
        Napi::Error::New(env, "forwardStep failed").ThrowAsJavaScriptException();
        return env.Null();
    }

    return logits;
}
// ---------------------------------------------------------
// forwardStep(model, tokenId, pos)
// Returns Float32Array logits
// ---------------------------------------------------------
Napi::Value oldForwardStep(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // ---------------- Argument Validation ----------------
    if (info.Length() < 3) {
        Napi::TypeError::New(
            env,
            "forwardStep: expected 3 args (model, tokenId, pos)"
        ).ThrowAsJavaScriptException();
        return env.Null();
    }
    if (!info[1].IsNumber()) {
        Napi::TypeError::New(env, "forwardStep: tokenId must be a number")
            .ThrowAsJavaScriptException();
        return env.Null();
    }
    if (!info[2].IsNumber()) {
        Napi::TypeError::New(env, "forwardStep: pos must be a number")
            .ThrowAsJavaScriptException();
        return env.Null();
    }

    auto* m = getModel(info);
    if (!m) {
        Napi::Error::New(env, "forwardStep: invalid model handle")
            .ThrowAsJavaScriptException();
        return env.Null();
    }

    int32_t tokenId = info[1].As<Napi::Number>().Int32Value();
    int32_t pos     = info[2].As<Napi::Number>().Int32Value();
    int32_t vocab   = llm_get_vocab_size(m);

    if (vocab <= 0) {
        Napi::Error::New(env, "forwardStep: invalid vocab size")
            .ThrowAsJavaScriptException();
        return env.Null();
    }

    // ---------------- Debug Flags ----------------
    const char* env_args   = std::getenv("MLX_DEBUG_ARGS");
    const char* env_wire   = std::getenv("MLX_DEBUG_WIRE");
    const char* env_logits = std::getenv("MLX_DEBUG_LOGITS");

    bool debug_args   = (env_args   && env_args[0]   != '\0');
    bool debug_wire   = (env_wire   && env_wire[0]   != '\0');
    bool debug_logits = (env_logits && env_logits[0] != '\0');

    // Global debug banner if any debug flag is active
    if (debug_args || debug_wire || debug_logits) {
        std::fprintf(
            stderr,
            "[DEBUG_FLAGS] wire=%d logits=%d args=%d\n",
            debug_wire ? 1 : 0,
            debug_logits ? 1 : 0,
            debug_args ? 1 : 0
        );
        std::fflush(stderr);
    }

    // ---------------- Debug: Print Args ----------------
    if (debug_args) {
        std::fprintf(
            stderr,
            "[DEBUG_ARGS] ForwardStep: argc=%zu tokenId=%d pos=%d vocab=%d\n",
            info.Length(), tokenId, pos, vocab
        );
        std::fflush(stderr);
    }

    // ---------------- Allocate Logits Buffer ----------------
    size_t bytes = static_cast<size_t>(vocab) * sizeof(float);
    float* data = static_cast<float*>(std::malloc(bytes));
    if (!data) {
        Napi::Error::New(env, "forwardStep: malloc failed").ThrowAsJavaScriptException();
        return env.Null();
    }
    std::memset(data, 0, bytes);

    Napi::ArrayBuffer buf = Napi::ArrayBuffer::New(
        env, data, bytes,
        [](Napi::Env /*env*/, void* finalizeData) {
            std::free(finalizeData);
        }
    );

    Napi::Float32Array logits = Napi::Float32Array::New(env, vocab, buf, 0);

    // ---------------- Debug Wire-Test Mode ----------------
    if (debug_wire) {
        std::fprintf(
            stderr,
            "[DEBUG_WIRE] Skipping llm_forward_step and returning test pattern\n"
        );
        std::fflush(stderr);

        if (tokenId >= 0 && tokenId < vocab)
            data[tokenId] = 1.0f;

        if (pos >= 0 && pos < vocab)
            data[pos] += 0.5f;  // additive so collisions are visible

        return logits;
    }

    // ---------------- Real Model Forward Step ----------------
    llm_status st = llm_forward_step(m, tokenId, pos, data);
    if (st != LLM_OK) {
        Napi::Error::New(env, "forwardStep: llm_forward_step failed")
            .ThrowAsJavaScriptException();
        return env.Null();
    }

    // ---------------- Debug Logits Stats ----------------
    if (debug_logits) {
        float minv = data[0];
        float maxv = data[0];
        int   argmax = 0;

        for (int i = 1; i < vocab; ++i) {
            float v = data[i];
            if (v < minv) minv = v;
            if (v > maxv) { maxv = v; argmax = i; }
        }

        std::fprintf(
            stderr,
            "[DEBUG_LOGITS] min=%g max=%g argmax=%d (pos=%d tokenId=%d)\n",
            minv, maxv, argmax, pos, tokenId
        );
        std::fflush(stderr);
    }

    return logits;
}
// ---------------------------------------------------------
// FreeModel(model)
// ---------------------------------------------------------
Napi::Value FreeModel(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto* m = getModel(info);
    if (!m) return env.Null();

    llm_free_model(m);

    return env.Undefined();
}

// ---------------------------------------------------------
// Metadata wrappers
// ---------------------------------------------------------
Napi::Value GetVocabSize(const Napi::CallbackInfo& info) {
    auto* m = getModel(info);
    return Napi::Number::New(info.Env(), llm_get_vocab_size(m));
}

Napi::Value GetHiddenSize(const Napi::CallbackInfo& info) {
    auto* m = getModel(info);
    return Napi::Number::New(info.Env(), llm_get_hidden_size(m));
}

Napi::Value GetNumLayers(const Napi::CallbackInfo& info) {
    auto* m = getModel(info);
    return Napi::Number::New(info.Env(), llm_get_num_layers(m));
}

// ---------------------------------------------------------
// listWeights(model) → array of weight names
// (useful for debugging HF loads)
// ---------------------------------------------------------
Napi::Value ListWeights(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto* m = getModel(info);
    if (!m) return env.Null();

    Napi::Array out = Napi::Array::New(env, m->weights.size());

    uint32_t i = 0;
    for (const auto &kv : m->weights) {
        out.Set(i++, Napi::String::New(env, kv.first));
    }

    return out;
}

Napi::Value RunDirectMetalProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    std::uint32_t n = 4096;
    if (info.Length() > 0 && info[0].IsNumber()) {
        n = info[0].As<Napi::Number>().Uint32Value();
    }
    GypsyDirectMetalProbeResult probe = GypsyRunDirectMetalAddProbe(n);
    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, probe.ok));
    out.Set("runtime", Napi::String::New(env, "direct_metal_objcxx"));
    out.Set("kernel", Napi::String::New(env, "gypsy_add_probe"));
    out.Set("device_available", Napi::Boolean::New(env, probe.device_available));
    out.Set("library_compiled", Napi::Boolean::New(env, probe.library_compiled));
    out.Set("pipeline_created", Napi::Boolean::New(env, probe.pipeline_created));
    out.Set("command_completed", Napi::Boolean::New(env, probe.command_completed));
    out.Set("element_count", Napi::Number::New(env, n));
    out.Set("checksum", Napi::Number::New(env, probe.checksum));
    out.Set("max_abs_diff", Napi::Number::New(env, probe.max_abs_diff));
    out.Set("elapsed_ms", Napi::Number::New(env, probe.elapsed_ms));
    out.Set("error", probe.error == nullptr ? env.Null().As<Napi::Value>() : Napi::String::New(env, probe.error).As<Napi::Value>());
    return out;
}

Napi::Value RunDirectMetalAttentionProbe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    std::uint32_t seq_len = 64;
    if (info.Length() > 0 && info[0].IsNumber()) {
        seq_len = info[0].As<Napi::Number>().Uint32Value();
    }
    GypsyDirectMetalProbeResult probe = GypsyRunDirectMetalAttentionProbe(seq_len);
    Napi::Object out = Napi::Object::New(env);
    out.Set("ok", Napi::Boolean::New(env, probe.ok));
    out.Set("runtime", Napi::String::New(env, "direct_metal_objcxx"));
    out.Set("kernel", Napi::String::New(env, "gypsy_attention_probe"));
    out.Set("device_available", Napi::Boolean::New(env, probe.device_available));
    out.Set("library_compiled", Napi::Boolean::New(env, probe.library_compiled));
    out.Set("pipeline_created", Napi::Boolean::New(env, probe.pipeline_created));
    out.Set("command_completed", Napi::Boolean::New(env, probe.command_completed));
    out.Set("seq_len", Napi::Number::New(env, seq_len));
    out.Set("checksum", Napi::Number::New(env, probe.checksum));
    out.Set("max_abs_diff", Napi::Number::New(env, probe.max_abs_diff));
    out.Set("elapsed_ms", Napi::Number::New(env, probe.elapsed_ms));
    out.Set("error", probe.error == nullptr ? env.Null().As<Napi::Value>() : Napi::String::New(env, probe.error).As<Napi::Value>());
    return out;
}

// ---------------------------------------------------------
// Module INIT
// ---------------------------------------------------------
Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("inspectModel",  Napi::Function::New(env, InspectModel));
    exports.Set("inspectAdapter",  Napi::Function::New(env, InspectAdapter));
    exports.Set("inspectTokenizer",  Napi::Function::New(env, InspectTokenizer));
    exports.Set("loadModelResident", Napi::Function::New(env, LoadModelResident));
    exports.Set("describeModelGroups", Napi::Function::New(env, DescribeModelGroups));
    exports.Set("loadTokenizer", Napi::Function::New(env, LoadTokenizer));
    exports.Set("loadAdapter", Napi::Function::New(env, LoadAdapter));
    exports.Set("describeAdapterGroups", Napi::Function::New(env, DescribeAdapterGroups));
    exports.Set("createSession", Napi::Function::New(env, CreateSession));
    exports.Set("describeSession", Napi::Function::New(env, DescribeSession));
    exports.Set("describeTensorViews", Napi::Function::New(env, DescribeTensorViews));
    exports.Set("describeTypedTensorPlan", Napi::Function::New(env, DescribeTypedTensorPlan));
    exports.Set("constructSelectedResidentArrays", Napi::Function::New(env, ConstructSelectedResidentArrays));
    exports.Set("describeSelectedResidentGroups", Napi::Function::New(env, DescribeSelectedResidentGroups));
    exports.Set("runSelectedNormProbe", Napi::Function::New(env, RunSelectedNormProbe));
    exports.Set("runSelectedQuantizedProjectionProbe", Napi::Function::New(env, RunSelectedQuantizedProjectionProbe));
    exports.Set("runSelectedLoraProjectionDeltaProbe", Napi::Function::New(env, RunSelectedLoraProjectionDeltaProbe));
    exports.Set("runSelectedBasePlusLoraProjectionProbe", Napi::Function::New(env, RunSelectedBasePlusLoraProjectionProbe));
    exports.Set("runSelectedQNormAfterLoraProbe", Napi::Function::New(env, RunSelectedQNormAfterLoraProbe));
    exports.Set("runSelectedRopeAfterQNormProbe", Napi::Function::New(env, RunSelectedRopeAfterQNormProbe));
    exports.Set("runSelectedKvProjectionPathProbe", Napi::Function::New(env, RunSelectedKvProjectionPathProbe));
    exports.Set("runSelectedSingleTokenAttentionProbe", Napi::Function::New(env, RunSelectedSingleTokenAttentionProbe));
    exports.Set("runSelectedOProjectionPathProbe", Napi::Function::New(env, RunSelectedOProjectionPathProbe));
    exports.Set("runSelectedPostAttentionResidualProbe", Napi::Function::New(env, RunSelectedPostAttentionResidualProbe));
    exports.Set("runSelectedPostAttentionRmsNormProbe", Napi::Function::New(env, RunSelectedPostAttentionRmsNormProbe));
    exports.Set("runSelectedMlpGateUpProbe", Napi::Function::New(env, RunSelectedMlpGateUpProbe));
    exports.Set("runSelectedMlpActivationProbe", Napi::Function::New(env, RunSelectedMlpActivationProbe));
    exports.Set("runSelectedMlpDownProjectionProbe", Napi::Function::New(env, RunSelectedMlpDownProjectionProbe));
    exports.Set("runSelectedMlpResidualProbe", Napi::Function::New(env, RunSelectedMlpResidualProbe));
    exports.Set("runSelectedLayerOutputContractProbe", Napi::Function::New(env, RunSelectedLayerOutputContractProbe));
    exports.Set("runSelectedNextLayerHandoffProbe", Napi::Function::New(env, RunSelectedNextLayerHandoffProbe));
    exports.Set("runSelectedLayer21ResidencyProbe", Napi::Function::New(env, RunSelectedLayer21ResidencyProbe));
    exports.Set("runSelectedLayer21InputQkvProbe", Napi::Function::New(env, RunSelectedLayer21InputQkvProbe));
    exports.Set("runSelectedFinalLogitsProbe", Napi::Function::New(env, RunSelectedFinalLogitsProbe));
    exports.Set("protocolStatus", Napi::Function::New(env, ProtocolStatus));
    exports.Set("warmSession", Napi::Function::New(env, WarmSession));
    exports.Set("generate", Napi::Function::New(env, GenerateProtocol));
    exports.Set("freeSession", Napi::Function::New(env, FreeSessionProtocol));
    exports.Set("unloadAdapter", Napi::Function::New(env, UnloadAdapterProtocol));
    exports.Set("unloadTokenizer", Napi::Function::New(env, UnloadTokenizerProtocol));
    exports.Set("unloadModel", Napi::Function::New(env, UnloadModelProtocol));

    exports.Set("loadModel",     Napi::Function::New(env, LoadModel));
    exports.Set("applyLora",     Napi::Function::New(env, ApplyLora));
    exports.Set("resetKV",       Napi::Function::New(env, ResetKV));
    exports.Set("forwardStep",   Napi::Function::New(env, ForwardStep));
    exports.Set("freeModel",     Napi::Function::New(env, FreeModel));

    exports.Set("getVocabSize",  Napi::Function::New(env, GetVocabSize));
    exports.Set("getHiddenSize", Napi::Function::New(env, GetHiddenSize));
    exports.Set("getNumLayers",  Napi::Function::New(env, GetNumLayers));

    exports.Set("listWeights",   Napi::Function::New(env, ListWeights));
    exports.Set("runDirectMetalProbe", Napi::Function::New(env, RunDirectMetalProbe));
    exports.Set("runDirectMetalAttentionProbe", Napi::Function::New(env, RunDirectMetalAttentionProbe));

    return exports;
}

NODE_API_MODULE(metal_llm, Init);
