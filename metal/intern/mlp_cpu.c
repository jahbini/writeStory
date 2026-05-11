// ------------------------------------------------------------
// MLP block for layer L on CPU (Phi-1.5 style: fc1 -> GELU -> fc2)
// Uses:
//   model.layers.L.mlp.fc1.weight / bias
//   model.layers.L.mlp.fc2.weight / bias
// ------------------------------------------------------------
bool apply_mlp_layer_cpu(
    metal_llm_model* model,
    int layer,
    const std::vector<float>& x_in,
    std::vector<float>& x_out)
{
    if (!model) {
        log_err("apply_mlp_layer_cpu", "null model");
        return false;
    }

    const int hidden = (int)x_in.size();
    if (hidden == 0) {
        std::fprintf(stderr,
            "[metal_llm][apply_mlp_layer_cpu] empty input\n");
        return false;
    }

    std::string k_fc1_w = layer_key(layer, "mlp.fc1.weight");
    std::string k_fc1_b = layer_key(layer, "mlp.fc1.bias");
    std::string k_fc2_w = layer_key(layer, "mlp.fc2.weight");
    std::string k_fc2_b = layer_key(layer, "mlp.fc2.bias");

    auto it_fc1_w = model->weights.find(k_fc1_w);
    auto it_fc2_w = model->weights.find(k_fc2_w);

    if (it_fc1_w == model->weights.end() ||
        it_fc2_w == model->weights.end())
    {
        std::fprintf(stderr,
            "[metal_llm] ERROR: MLP fc1/fc2 weights missing for %s / %s\n",
            k_fc1_w.c_str(), k_fc2_w.c_str());
        return false;
    }

    mx::array W1 = it_fc1_w->second;  // [ff_dim, hidden]
    mx::array W2 = it_fc2_w->second;  // [hidden, ff_dim]

    mx::eval(W1);
    mx::eval(W2);

    if (W1.ndim() != 2 || W2.ndim() != 2) {
        std::fprintf(stderr,
            "[metal_llm] ERROR: MLP weights ndim != 2 (fc1_ndim=%d fc2_ndim=%d)\n",
            (int)W1.ndim(), (int)W2.ndim());
        return false;
    }

    int ff_dim = (int)W1.shape(0);
    int in_dim = (int)W1.shape(1);

    if (in_dim != hidden ||
        (int)W2.shape(0) != hidden ||
        (int)W2.shape(1) != ff_dim)
    {
        std::fprintf(stderr,
            "[metal_llm] ERROR: MLP weight shape mismatch (fc1=(%d,%d) fc2=(%d,%d) hidden=%d)\n",
            (int)W1.shape(0), (int)W1.shape(1),
            (int)W2.shape(0), (int)W2.shape(1),
            hidden);
        return false;
    }

    if (W1.dtype() != mx::float32 || W2.dtype() != mx::float32) {
        std::fprintf(stderr,
            "[metal_llm] ERROR: MLP weights not float32 after upcast\n");
        return false;
    }

    const float* W1_ptr = W1.data<float>();
    const float* W2_ptr = W2.data<float>();

    // ---- optional biases ----
    const float* b1_ptr = nullptr;
    const float* b2_ptr = nullptr;

    auto it_fc1_b = model->weights.find(k_fc1_b);
    if (it_fc1_b != model->weights.end()) {
        mx::array b1_arr = it_fc1_b->second;
        mx::eval(b1_arr);

        if (b1_arr.ndim() != 1 || (int)b1_arr.shape(0) != ff_dim ||
            b1_arr.dtype() != mx::float32)
        {
            std::fprintf(stderr,
                "[metal_llm] ERROR: fc1.bias shape/dtype mismatch (size=%d, ff_dim=%d)\n",
                (int)b1_arr.shape(0), ff_dim);
            return false;
        }
        b1_ptr = b1_arr.data<float>();
    }

    auto it_fc2_b = model->weights.find(k_fc2_b);
    if (it_fc2_b != model->weights.end()) {
        mx::array b2_arr = it_fc2_b->second;
        mx::eval(b2_arr);

        if (b2_arr.ndim() != 1 || (int)b2_arr.shape(0) != hidden ||
            b2_arr.dtype() != mx::float32)
        {
            std::fprintf(stderr,
                "[metal_llm] ERROR: fc2.bias shape/dtype mismatch (size=%d, hidden=%d)\n",
                (int)b2_arr.shape(0), hidden);
            return false;
        }
        b2_ptr = b2_arr.data<float>();
    }

    // ---- fc1 + GELU ----
    std::vector<float> h1(ff_dim);

    for (int j = 0; j < ff_dim; ++j) {
        const float* row = W1_ptr + (size_t)j * hidden;

        double acc = 0.0;
        for (int i = 0; i < hidden; ++i) {
            acc += (double)row[i] * (double)x_in[i];
        }
        if (b1_ptr) acc += (double)b1_ptr[j];

        float x = (float)acc;
        float u = x / std::sqrt(2.0f);
        float gelu = 0.5f * x * (1.0f + std::erff(u));

        h1[j] = gelu;
    }

    // ---- fc2 ----
    x_out.assign(hidden, 0.0f);

    for (int o = 0; o < hidden; ++o) {
        const float* row = W2_ptr + (size_t)o * ff_dim;

        double acc = 0.0;
        for (int j = 0; j < ff_dim; ++j) {
            acc += (double)row[j] * (double)h1[j];
        }
        if (b2_ptr) acc += (double)b2_ptr[o];

        x_out[o] = (float)acc;
    }

    return true;
}

