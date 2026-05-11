// ---------------------------------------------------------------------------
//  APPLY LORA (stub)
// ---------------------------------------------------------------------------

extern "C"
llm_status llm_apply_lora(metal_llm_model* model,
                          const char* lora_path,
                          float alpha,
                          bool merge)
{
    if (!model || !lora_path) {
        return LLM_ERR_INVALID;
    }

    std::cout << "[metal_llm] apply_lora: " << lora_path
              << " alpha=" << alpha
              << " merge=" << (merge ? "true" : "false")
              << std::endl;

    // TODO: implement real LoRA merge if needed
    return LLM_OK;
}

