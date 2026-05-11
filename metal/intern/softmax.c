// ------------------------------------------------------------
// softmax over a 1D vector (length = head_dim)
// src and dst may alias (in-place allowed)
// ------------------------------------------------------------
static void apply_softmax(float* dst, const float* src, int len) {
    // find max for numerical stability
    float maxv = src[0];
    for (int i = 1; i < len; ++i)
        if (src[i] > maxv) maxv = src[i];

    // subtract max and exp
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        float ex = expf(src[i] - maxv);
        dst[i]   = ex;
        sum     += ex;
    }

    // normalize
    float inv = 1.0f / sum;
    for (int i = 0; i < len; ++i)
        dst[i] *= inv;
}

