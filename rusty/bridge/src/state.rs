use std::collections::HashMap;
use std::collections::HashSet;

use serde_json::Value;

use crate::handles::HandleFactory;

// These fields are intentionally present for the upcoming real ML handle lifecycle.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ModelRecord {
    pub path: String,
}

// These fields are intentionally present for the upcoming real ML handle lifecycle.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ModelDescriptorRecord {
    pub model_dir: String,
    pub model_type: String,
    pub architecture: String,
    pub layer_count: u64,
    pub hidden_size: u64,
    pub vocab_size: u64,
    pub quantized: bool,
    pub tied_embeddings: bool,
    pub total_estimated_bytes: u64,
    pub loaded_layers: u64,
    pub loaded_weights: bool,
}

// These fields are intentionally present for the upcoming real ML handle lifecycle.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TokenizerRecord {
    pub path: String,
    pub detected_files: Vec<String>,
    pub tokenizer_kind: String,
    pub vocab: HashMap<String, u64>,
    pub model_type: Option<String>,
    pub normalizer_type: Option<String>,
    pub pre_tokenizer_type: Option<String>,
    pub decoder_type: Option<String>,
    pub vocab_size: Option<u64>,
    pub merges_count: Option<u64>,
    pub added_tokens_count: Option<u64>,
    pub model: Option<String>,
}

// These fields are intentionally present for the upcoming real ML handle lifecycle.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SessionRecord {
    pub model: Option<String>,
    pub tokenizer: Option<String>,
    pub kv_cache: Option<String>,
}

// These fields are intentionally present for the upcoming real ML handle lifecycle.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct KvCacheRecord {
    pub session: Option<String>,
    pub layers: u64,
    pub sequence_len: u64,
    pub cached_keys: Vec<String>,
    pub cached_values: Vec<String>,
}

// These fields are intentionally present for the upcoming real ML handle lifecycle.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TensorGroupRecord {
    pub model_dir: String,
    pub group: String,
}

// These fields are intentionally present for the upcoming real ML handle lifecycle.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct EmbeddingGroupRecord {
    pub model_dir: String,
    pub group: String,
}

// These fields are intentionally present for the upcoming real ML handle lifecycle.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct LayerGroupsRecord {
    pub model_dir: String,
    pub layer: u64,
    pub group_handles: HashMap<String, String>,
    pub total_byte_size: u64,
    pub quantized_group_count: u64,
    pub norm_group_count: u64,
}

// These fields are intentionally present for the upcoming real ML handle lifecycle.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct JobRecord {
    pub session: Option<String>,
    pub status: String,
    pub last_result: Option<Value>,
}

#[derive(Debug, Default)]
pub struct BridgeState {
    pub models: HashMap<String, ModelRecord>,
    pub native_models: HashMap<String, ModelRecord>,
    pub model_descriptors: HashMap<String, ModelDescriptorRecord>,
    pub tokenizers: HashMap<String, TokenizerRecord>,
    pub native_sessions: HashMap<String, SessionRecord>,
    pub sessions: HashMap<String, SessionRecord>,
    pub tensor_groups: HashMap<String, TensorGroupRecord>,
    pub embedding_groups: HashMap<String, EmbeddingGroupRecord>,
    pub layer_groups: HashMap<String, LayerGroupsRecord>,
    pub kv_caches: HashMap<String, KvCacheRecord>,
    pub jobs: HashMap<String, JobRecord>,
    pub greedy_probe_cache: HashMap<String, Value>,
    pub freed_models: HashSet<String>,
    pub freed_native_models: HashSet<String>,
    pub freed_model_descriptors: HashSet<String>,
    pub freed_tokenizers: HashSet<String>,
    pub freed_native_sessions: HashSet<String>,
    pub freed_sessions: HashSet<String>,
    pub freed_tensor_groups: HashSet<String>,
    pub freed_embedding_groups: HashSet<String>,
    pub freed_layer_groups: HashSet<String>,
    pub handles: HandleFactory,
    pub shutdown_requested: bool,
}

impl BridgeState {
    pub fn counts_json(&self) -> Value {
        serde_json::json!({
            "models": self.models.len(),
            "native_models": self.native_models.len(),
            "model_descriptors": self.model_descriptors.len(),
            "tokenizers": self.tokenizers.len(),
            "native_sessions": self.native_sessions.len(),
            "sessions": self.sessions.len(),
            "tensor_groups": self.tensor_groups.len(),
            "embedding_groups": self.embedding_groups.len(),
            "layer_groups": self.layer_groups.len(),
            "kv_caches": self.kv_caches.len(),
            "jobs": self.jobs.len(),
            "greedy_probe_cache": self.greedy_probe_cache.len()
        })
    }
}
