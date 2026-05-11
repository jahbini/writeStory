// ------------------------------------------------------------
// Shape/dtype helpers
// ------------------------------------------------------------

struct F16MatRef {
    const mx::float16_t* ptr = nullptr;
    int rows = 0;
    int cols = 0;
};

struct F32VecRef {
    const float* ptr = nullptr;
    int size = 0;
};

bool get_f16_matrix_2d(mx::array& A,
                       const char* name,
                       F16MatRef& out)
{
    if (A.ndim() != 2) {
        std::fprintf(stderr,
                     "[metal_llm] ERROR: %s ndim=%d (expected 2)\n",
                     name, static_cast<int>(A.ndim()));
        return false;
    }
    if (A.dtype() != mx::float16) {
        std::fprintf(stderr,
                     "[metal_llm] ERROR: %s dtype=%s (expected float16)\n",
                     name, dtype_to_string(A.dtype()));
        return false;
    }

    mx::eval(A);

    auto sh = A.shape();
    out.rows = static_cast<int>(sh[0]);
    out.cols = static_cast<int>(sh[1]);
    out.ptr  = A.data<mx::float16_t>();
    return true;
}

bool get_f32_vector_1d(mx::array& A,
                       const char* name,
                       F32VecRef& out)
{
    if (A.ndim() != 1) {
        std::fprintf(stderr,
                     "[metal_llm] ERROR: %s ndim=%d (expected 1)\n",
                     name, static_cast<int>(A.ndim()));
        return false;
    }
    if (A.dtype() != mx::float32) {
        std::fprintf(stderr,
                     "[metal_llm] ERROR: %s dtype=%s (expected float32)\n",
                     name, dtype_to_string(A.dtype()));
        return false;
    }

    mx::eval(A);

    out.size = static_cast<int>(A.shape(0));
    out.ptr  = A.data<float>();
    return true;
}

