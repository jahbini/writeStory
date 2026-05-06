use std::collections::HashMap;
use std::collections::HashSet;

use serde_json::Value;

use crate::handles::HandleFactory;

#[derive(Debug, Clone)]
pub struct ModelRecord {
    pub path: String,
}

#[derive(Debug, Clone)]
pub struct TokenizerRecord {
    pub model: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SessionRecord {
    pub model: Option<String>,
    pub tokenizer: Option<String>,
    pub kv_cache: Option<String>,
}

#[derive(Debug, Clone)]
pub struct KvCacheRecord {
    pub session: Option<String>,
}

#[derive(Debug, Clone)]
pub struct JobRecord {
    pub session: Option<String>,
    pub status: String,
    pub last_result: Option<Value>,
}

#[derive(Debug, Default)]
pub struct BridgeState {
    pub models: HashMap<String, ModelRecord>,
    pub tokenizers: HashMap<String, TokenizerRecord>,
    pub sessions: HashMap<String, SessionRecord>,
    pub kv_caches: HashMap<String, KvCacheRecord>,
    pub jobs: HashMap<String, JobRecord>,
    pub freed_models: HashSet<String>,
    pub freed_sessions: HashSet<String>,
    pub handles: HandleFactory,
    pub shutdown_requested: bool,
}

impl BridgeState {
    pub fn counts_json(&self) -> Value {
        serde_json::json!({
            "models": self.models.len(),
            "tokenizers": self.tokenizers.len(),
            "sessions": self.sessions.len(),
            "kv_caches": self.kv_caches.len(),
            "jobs": self.jobs.len()
        })
    }
}
