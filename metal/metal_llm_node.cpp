// metal_llm_node.cpp — Node.js NAPI wrapper for metal_llm
// --------------------------------------------------------

#include <napi.h>
#include <cstdlib>
#include <cstring>
#include "metal_llm.h"

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

// ---------------------------------------------------------
// Module INIT
// ---------------------------------------------------------
Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("loadModel",     Napi::Function::New(env, LoadModel));
    exports.Set("applyLora",     Napi::Function::New(env, ApplyLora));
    exports.Set("resetKV",       Napi::Function::New(env, ResetKV));
    exports.Set("forwardStep",   Napi::Function::New(env, ForwardStep));
    exports.Set("freeModel",     Napi::Function::New(env, FreeModel));

    exports.Set("getVocabSize",  Napi::Function::New(env, GetVocabSize));
    exports.Set("getHiddenSize", Napi::Function::New(env, GetHiddenSize));
    exports.Set("getNumLayers",  Napi::Function::New(env, GetNumLayers));

    exports.Set("listWeights",   Napi::Function::New(env, ListWeights));

    return exports;
}

NODE_API_MODULE(metal_llm, Init);
