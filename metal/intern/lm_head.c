// ------------------------------------------------------------
// Helper: apply final lm_head
// ------------------------------------------------------------
bool apply_lm_head(
    metal_llm_model* model,
    const std::vector<float>& hvec,
    float* out_logits)
{
    // Find weight
    auto it_head = model->weights.find("lm_head.weight");
    if (it_head == model->weights.end()) {
        std::fprintf(stderr,
            "[metal_llm] ERROR: lm_head.weight missing\n");
        return false;
    }

    mx::array& head = it_head->second;
    mx::eval(head);

    // Correct dims: [vocab_size, hidden_size]
    int head_vocab  = (int)head.shape(0);
    int head_hidden = (int)head.shape(1);

    if (head_hidden != cfg.hidden_size) {
        std::fprintf(stderr,
            "[metal_llm] LM Head mismatch: head_hidden=%d cfg.hidden_size=%d\n",
            head_hidden, cfg.hidden_size);
        return false;
    }

    const float* W = head.data<float>();
    int outN = std::min(cfg.vocab_size, head_vocab);

    // CPU matvec
    for (int v = 0; v < outN; ++v) {
        const float* row = W + (size_t)v * head_hidden;
        double acc = 0.0;
        for (int h = 0; h < head_hidden; ++h) {
            acc += (double)row[h] * (double)hvec[h];
        }
        out_logits[v] = (float)acc;
    }

    return true;
}
