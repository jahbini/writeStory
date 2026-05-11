// ------------------------------------------------------------
// Single-token self-attention (degenerate: output = V)
// ------------------------------------------------------------
void apply_self_attention_layer0_cpu(
    const QKV& qkv,
    std::vector<float>& h_out)
{
    const int dim = static_cast<int>(qkv.V.size());
    h_out.resize(dim);
    for (int i = 0; i < dim; ++i) {
        h_out[i] = qkv.V[i];
    }
}

