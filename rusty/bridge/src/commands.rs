use std::convert::TryFrom;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::backend::{probe, shim};
use crate::protocol::{alive_value, Request, Response};
use crate::state::{
    BridgeState, EmbeddingGroupRecord, JobRecord, KvCacheRecord, LayerGroupsRecord,
    ModelDescriptorRecord, ModelRecord, SessionRecord, TensorGroupRecord, TokenizerRecord,
};

fn arg_string(args: &Value, key: &str) -> Option<String> {
    args.get(key)?.as_str().map(ToOwned::to_owned)
}

fn require_arg_string(request: &Request, key: &str) -> Result<String, Response> {
    match arg_string(&request.args, key) {
        Some(value) if !value.trim().is_empty() => Ok(value),
        _ => Err(Response::err(
            request.id.clone(),
            "bad_args",
            format!("{} requires args.{} as a string", request.cmd, key),
        )),
    }
}

fn require_arg_u64(request: &Request, key: &str) -> Result<u64, Response> {
    match request.args.get(key).and_then(|value| value.as_u64()) {
        Some(value) if value != 0 => Ok(value),
        _ => Err(Response::err(
            request.id.clone(),
            "bad_args",
            format!(
                "{} requires args.{} as a non-zero integer",
                request.cmd, key
            ),
        )),
    }
}

fn require_arg_u64_allow_zero(request: &Request, key: &str) -> Result<u64, Response> {
    match request.args.get(key).and_then(|value| value.as_u64()) {
        Some(value) => Ok(value),
        _ => Err(Response::err(
            request.id.clone(),
            "bad_args",
            format!(
                "{} requires args.{} as a non-negative integer",
                request.cmd, key
            ),
        )),
    }
}

fn unknown_handle(request: &Request, kind: &str, handle: &str) -> Response {
    Response::err(
        request.id.clone(),
        "unknown_handle",
        format!("{kind} handle not found: {handle}"),
    )
}

fn already_freed(request: &Request, kind: &str, handle: &str) -> Response {
    Response::err(
        request.id.clone(),
        "already_freed",
        format!("{kind} handle was already freed: {handle}"),
    )
}

fn bridge_shutting_down(request: &Request) -> Response {
    Response::err(
        request.id.clone(),
        "bridge_shutting_down",
        format!("bridge is shutting down; refusing command: {}", request.cmd),
    )
}

fn tokenizer_kind_from_files(detected_files: &[String]) -> String {
    if detected_files
        .iter()
        .any(|entry| entry.ends_with("tokenizer.json"))
    {
        "huggingface_json".to_string()
    } else if detected_files
        .iter()
        .any(|entry| entry.ends_with("tokenizer.model"))
    {
        "sentencepiece".to_string()
    } else {
        "unknown".to_string()
    }
}

fn tokenizer_detected_files(path: &Path) -> Result<Vec<String>, String> {
    if !path.exists() {
        return Err(format!("tokenizer path does not exist: {}", path.display()));
    }

    let base_dir: PathBuf = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| format!("tokenizer path has no parent: {}", path.display()))?
    };

    let mut detected = Vec::new();
    let supported = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
    ];

    for name in supported {
        let direct = base_dir.join(name);
        if direct.exists() {
            detected.push(direct.to_string_lossy().to_string());
            continue;
        }

        if !path.is_dir() {
            let self_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();
            if self_name == name {
                detected.push(path.to_string_lossy().to_string());
            }
        }
    }

    Ok(detected)
}

fn tokenizer_json_metadata(
    path: &Path,
) -> Result<
    (
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<u64>,
        Option<u64>,
        Option<u64>,
    ),
    String,
> {
    let tokenizer_json = if path.is_dir() {
        path.join("tokenizer.json")
    } else {
        path.to_path_buf()
    };

    if !tokenizer_json.exists() {
        return Ok((None, None, None, None, None, None, None));
    }

    let raw = std::fs::read_to_string(&tokenizer_json)
        .map_err(|error| format!("failed to read tokenizer.json: {error}"))?;
    let value: Value = serde_json::from_str(&raw)
        .map_err(|error| format!("failed to parse tokenizer.json: {error}"))?;

    let model = value.get("model");
    let model_type = model
        .and_then(|entry| entry.get("type"))
        .and_then(|entry| entry.as_str())
        .map(ToOwned::to_owned);
    let normalizer_type = value
        .get("normalizer")
        .and_then(|entry| entry.get("type"))
        .and_then(|entry| entry.as_str())
        .map(ToOwned::to_owned);
    let pre_tokenizer_type = value
        .get("pre_tokenizer")
        .and_then(|entry| entry.get("type"))
        .and_then(|entry| entry.as_str())
        .map(ToOwned::to_owned);
    let decoder_type = value
        .get("decoder")
        .and_then(|entry| entry.get("type"))
        .and_then(|entry| entry.as_str())
        .map(ToOwned::to_owned);
    let vocab_size = model
        .and_then(|entry| entry.get("vocab"))
        .and_then(|entry| entry.as_object())
        .map(|entry| entry.len() as u64);
    let merges_count = model
        .and_then(|entry| entry.get("merges"))
        .and_then(|entry| entry.as_array())
        .map(|entry| entry.len() as u64);
    let added_tokens_count = value
        .get("added_tokens")
        .and_then(|entry| entry.as_array())
        .map(|entry| entry.len() as u64);

    Ok((
        model_type,
        normalizer_type,
        pre_tokenizer_type,
        decoder_type,
        vocab_size,
        merges_count,
        added_tokens_count,
    ))
}

fn tokenizer_json_vocab(path: &Path) -> Result<std::collections::HashMap<String, u64>, String> {
    let tokenizer_json = if path.is_dir() {
        path.join("tokenizer.json")
    } else {
        path.to_path_buf()
    };

    if !tokenizer_json.exists() {
        return Ok(std::collections::HashMap::new());
    }

    let raw = std::fs::read_to_string(&tokenizer_json)
        .map_err(|error| format!("failed to read tokenizer.json: {error}"))?;
    let value: Value = serde_json::from_str(&raw)
        .map_err(|error| format!("failed to parse tokenizer.json: {error}"))?;

    let vocab = value
        .get("model")
        .and_then(|entry| entry.get("vocab"))
        .and_then(|entry| entry.as_object())
        .ok_or_else(|| "tokenizer.json model.vocab missing or not an object".to_string())?;

    let mut map = std::collections::HashMap::new();
    for (token, id_value) in vocab {
        let id = id_value.as_u64().ok_or_else(|| {
            format!("tokenizer.json vocab id is not an integer for token: {token}")
        })?;
        map.insert(token.clone(), id);
    }

    if let Some(added_tokens) = value.get("added_tokens").and_then(|entry| entry.as_array()) {
        for token in added_tokens {
            let content = token
                .get("content")
                .and_then(|entry| entry.as_str())
                .ok_or_else(|| "tokenizer.json added_tokens entry missing content".to_string())?;
            let id = token
                .get("id")
                .and_then(|entry| entry.as_u64())
                .ok_or_else(|| {
                    format!("tokenizer.json added_tokens id is not an integer for token: {content}")
                })?;
            map.insert(content.to_string(), id);
        }
    }

    Ok(map)
}

fn tokenizer_inverse_vocab(
    vocab: &std::collections::HashMap<String, u64>,
) -> std::collections::HashMap<u64, String> {
    let mut inverse = std::collections::HashMap::new();
    for (token, id) in vocab {
        inverse.entry(*id).or_insert_with(|| token.clone());
    }
    inverse
}

fn tokenizer_encode_fixture(record: &TokenizerRecord, text: &str) -> Result<Vec<u64>, String> {
    if record.vocab.contains_key("Ċ") || record.vocab.contains_key("Ġ") {
        return tokenizer_encode_qwen_like(record, text);
    }

    let mut tokens = Vec::new();
    let mut special_tokens = record
        .vocab
        .keys()
        .filter(|token| token.starts_with("<|") && token.ends_with("|>"))
        .cloned()
        .collect::<Vec<String>>();
    special_tokens.sort_by_key(|token| std::cmp::Reverse(token.len()));

    for raw_piece in text.split_whitespace() {
        let mut rest = raw_piece;
        while !rest.is_empty() {
            if let Some(special) = special_tokens
                .iter()
                .find(|special| rest.starts_with(special.as_str()))
            {
                if let Some(id) = record.vocab.get(special) {
                    tokens.push(*id);
                    rest = &rest[special.len()..];
                    continue;
                }
            }

            let next_special_offset = special_tokens
                .iter()
                .filter_map(|special| rest.find(special.as_str()))
                .filter(|offset| *offset > 0)
                .min()
                .unwrap_or(rest.len());
            let piece = &rest[..next_special_offset];
            match record.vocab.get(piece) {
                Some(id) => tokens.push(*id),
                None => {
                    return Err(format!(
                        "tokenizer vocabulary does not contain token: {piece}"
                    ));
                }
            }
            rest = &rest[next_special_offset..];
        }
    }
    Ok(tokens)
}

fn tokenizer_encode_qwen_like(record: &TokenizerRecord, text: &str) -> Result<Vec<u64>, String> {
    let mut tokens = Vec::new();
    let mut special_tokens = record
        .vocab
        .keys()
        .filter(|token| token.starts_with("<|") && token.ends_with("|>"))
        .cloned()
        .collect::<Vec<String>>();
    special_tokens.sort_by_key(|token| std::cmp::Reverse(token.len()));

    let vocab_tokens = record.vocab.keys().cloned().collect::<Vec<String>>();
    let chars = text.chars().collect::<Vec<char>>();
    let mut index = 0usize;
    while index < chars.len() {
        let rest = chars[index..].iter().collect::<String>();
        if let Some(special) = special_tokens
            .iter()
            .find(|special| rest.starts_with(special.as_str()))
        {
            if let Some(id) = record.vocab.get(special) {
                tokens.push(*id);
                index += special.chars().count();
                continue;
            }
        }

        let marker = match chars[index] {
            '\n' => Some("Ċ".to_string()),
            ' ' => Some("Ġ".to_string()),
            '\t' => Some("ĉ".to_string()),
            _ => None,
        };
        if let Some(marker) = marker {
            if let Some(id) = record.vocab.get(&marker) {
                tokens.push(*id);
                index += 1;
                continue;
            }
        }

        let transformed_rest = chars[index..]
            .iter()
            .map(|ch| match ch {
                '\n' => 'Ċ',
                ' ' => 'Ġ',
                '\t' => 'ĉ',
                other => *other,
            })
            .collect::<String>();
        let matched = vocab_tokens
            .iter()
            .filter(|token| {
                !token.starts_with("<|") &&
                    transformed_rest.starts_with(token.as_str())
            })
            .max_by_key(|token| token.chars().count());
        if let Some(token) = matched {
            if let Some(id) = record.vocab.get(token) {
                tokens.push(*id);
                index += token.chars().count();
                continue;
            }
        }

        let piece = chars[index].to_string();
        return Err(format!("tokenizer vocabulary does not contain token: {piece}"));
    }
    Ok(tokens)
}

fn tokenizer_decode_fixture(record: &TokenizerRecord, tokens: &[u64]) -> Result<String, String> {
    tokenizer_decode_fixture_with_options(record, tokens, false).map(|decoded| decoded.text)
}

#[derive(Debug, Clone)]
struct TokenDecodeDiagnostic {
    token_id: u64,
    raw_token: String,
    byte_values: Vec<u8>,
    decoded_piece: String,
    special: bool,
    skipped: bool,
}

#[derive(Debug, Clone)]
struct TokenDecodeOutput {
    text: String,
    diagnostics: Vec<TokenDecodeDiagnostic>,
}

fn qwen_byte_decoder() -> std::collections::HashMap<char, u8> {
    let mut bytes: Vec<u16> = (b'!' as u16..=b'~' as u16).collect();
    bytes.extend(0xA1u16..=0xACu16);
    bytes.extend(0xAEu16..=0xFFu16);
    let mut chars = bytes.clone();
    let mut n = 0u16;
    for byte in 0u16..=255u16 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            chars.push(256 + n);
            n += 1;
        }
    }
    let mut decoder = std::collections::HashMap::new();
    for (byte, ch) in bytes.into_iter().zip(chars.into_iter()) {
        if let Some(ch) = char::from_u32(ch as u32) {
            decoder.insert(ch, byte as u8);
        }
    }
    decoder
}

fn decode_qwen_token_piece(piece: &str, byte_decoder: &std::collections::HashMap<char, u8>) -> (String, Vec<u8>) {
    let mut bytes = Vec::new();
    let mut used_byte_decoder = false;
    for ch in piece.chars() {
        if let Some(byte) = byte_decoder.get(&ch) {
            bytes.push(*byte);
            used_byte_decoder = true;
        } else {
            let mut buf = [0u8; 4];
            bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
    }
    if used_byte_decoder {
        (String::from_utf8_lossy(&bytes).into_owned(), bytes)
    } else {
        (
            piece.replace('Ċ', "\n").replace('Ġ', " ").replace('ĉ', "\t"),
            bytes,
        )
    }
}

fn tokenizer_decode_fixture_with_options(
    record: &TokenizerRecord,
    tokens: &[u64],
    skip_special_tokens: bool,
) -> Result<TokenDecodeOutput, String> {
    let inverse = tokenizer_inverse_vocab(&record.vocab);
    let qwen_like = record.vocab.contains_key("Ċ") || record.vocab.contains_key("Ġ");
    let mut qwen_bytes = Vec::new();
    let mut pieces = Vec::new();
    let byte_decoder = qwen_byte_decoder();
    let mut diagnostics = Vec::new();
    for token in tokens {
        match inverse.get(token) {
            Some(piece) => {
                let special = piece.starts_with("<|") && piece.ends_with("|>");
                if skip_special_tokens && special {
                    diagnostics.push(TokenDecodeDiagnostic {
                        token_id: *token,
                        raw_token: piece.clone(),
                        byte_values: Vec::new(),
                        decoded_piece: String::new(),
                        special,
                        skipped: true,
                    });
                    continue;
                }
                if qwen_like {
                    let (decoded_piece, byte_values) = decode_qwen_token_piece(piece, &byte_decoder);
                    qwen_bytes.extend_from_slice(&byte_values);
                    diagnostics.push(TokenDecodeDiagnostic {
                        token_id: *token,
                        raw_token: piece.clone(),
                        byte_values,
                        decoded_piece,
                        special,
                        skipped: false,
                    });
                } else {
                    pieces.push(piece.clone());
                    diagnostics.push(TokenDecodeDiagnostic {
                        token_id: *token,
                        raw_token: piece.clone(),
                        byte_values: piece.as_bytes().to_vec(),
                        decoded_piece: piece.clone(),
                        special,
                        skipped: false,
                    });
                }
            }
            None => {
                return Err(format!(
                    "tokenizer vocabulary does not contain token id: {token}"
                ));
            }
        }
    }
    if qwen_like {
        return Ok(TokenDecodeOutput {
            text: String::from_utf8_lossy(&qwen_bytes).into_owned(),
            diagnostics,
        });
    }
    Ok(TokenDecodeOutput {
        text: pieces.join(" "),
        diagnostics,
    })
}

fn cached_greedy_next_token_probe(
    state: &mut BridgeState,
    model_dir: &str,
    token_id: u64,
) -> Value {
    let cache_key = format!("{model_dir}\n{token_id}");
    if let Some(value) = state.greedy_probe_cache.get(&cache_key) {
        let mut cached = value.clone();
        if let Some(object) = cached.as_object_mut() {
            object.insert("cache_hit".to_string(), json!(true));
            object.insert("timing_ms".to_string(), json!(0.0));
        }
        return cached;
    }

    let mut value = shim::greedy_next_token_probe(model_dir, token_id);
    if value.get("ok").and_then(|entry| entry.as_bool()) == Some(true) {
        if let Some(object) = value.as_object_mut() {
            object.insert("cache_hit".to_string(), json!(false));
        }
        state.greedy_probe_cache.insert(cache_key, value.clone());
    }
    value
}

fn inspect_model_dir_config(path: &Path) -> Result<Value, String> {
    let config_json = if path.is_dir() {
        path.join("config.json")
    } else {
        path.to_path_buf()
    };

    if !config_json.exists() {
        return Ok(json!({
            "model_type": Value::Null,
            "architectures": Value::Null,
            "vocab_size": Value::Null,
            "hidden_size": Value::Null,
            "num_hidden_layers": Value::Null,
            "num_attention_heads": Value::Null,
            "num_key_value_heads": Value::Null,
            "intermediate_size": Value::Null,
            "torch_dtype": Value::Null,
            "rope": Value::Null
        }));
    }

    let raw = std::fs::read_to_string(&config_json)
        .map_err(|error| format!("failed to read config.json: {error}"))?;
    let value: Value = serde_json::from_str(&raw)
        .map_err(|error| format!("failed to parse config.json: {error}"))?;

    let rope = value
        .get("rope")
        .or_else(|| value.get("rope_scaling"))
        .map(|entry| entry.clone());

    Ok(json!({
        "model_type": value.get("model_type").cloned().unwrap_or(Value::Null),
        "architectures": value.get("architectures").cloned().unwrap_or(Value::Null),
        "vocab_size": value.get("vocab_size").cloned().unwrap_or(Value::Null),
        "hidden_size": value.get("hidden_size").cloned().unwrap_or(Value::Null),
        "num_hidden_layers": value.get("num_hidden_layers").cloned().unwrap_or(Value::Null),
        "num_attention_heads": value.get("num_attention_heads").cloned().unwrap_or(Value::Null),
        "num_key_value_heads": value.get("num_key_value_heads").cloned().unwrap_or(Value::Null),
        "intermediate_size": value.get("intermediate_size").cloned().unwrap_or(Value::Null),
        "torch_dtype": value.get("torch_dtype").cloned().unwrap_or(Value::Null),
        "rope": rope.unwrap_or(Value::Null)
    }))
}

fn inspect_model_dir_files(
    path: &Path,
) -> Result<
    (
        Vec<String>,
        Vec<String>,
        Option<String>,
        Option<String>,
        u64,
        u64,
    ),
    String,
> {
    if !path.exists() {
        return Ok((Vec::new(), Vec::new(), None, None, 0, 0));
    }

    let mut tokenizer_files = Vec::new();
    let mut safetensors_files = Vec::new();
    let mut safetensors_index = None;
    let mut generation_config = None;
    let mut file_count = 0u64;
    let mut total_size = 0u64;

    for entry in std::fs::read_dir(path)
        .map_err(|error| format!("failed to read model directory: {error}"))?
    {
        let entry =
            entry.map_err(|error| format!("failed to read model directory entry: {error}"))?;
        let file_type = entry
            .file_type()
            .map_err(|error| format!("failed to inspect model directory entry: {error}"))?;
        if !file_type.is_file() {
            continue;
        }

        let file_name = entry.file_name().to_string_lossy().to_string();
        let full_path = entry.path().display().to_string();
        file_count += 1;
        total_size += entry
            .metadata()
            .map_err(|error| format!("failed to read model directory metadata: {error}"))?
            .len();

        match file_name.as_str() {
            "tokenizer.json"
            | "tokenizer.model"
            | "tokenizer_config.json"
            | "special_tokens_map.json" => {
                tokenizer_files.push(full_path.clone());
            }
            "config.json" => {}
            "generation_config.json" => {
                generation_config = Some(full_path.clone());
            }
            _ => {}
        }

        if file_name.ends_with(".safetensors") {
            safetensors_files.push(full_path.clone());
        }
        if file_name.ends_with(".safetensors.index.json") {
            safetensors_index = Some(full_path.clone());
        }
    }

    tokenizer_files.sort();
    safetensors_files.sort();

    Ok((
        tokenizer_files,
        safetensors_files,
        safetensors_index,
        generation_config,
        file_count,
        total_size,
    ))
}

fn inspect_safetensors_index_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("model.safetensors.index.json")
    } else {
        path.to_path_buf()
    }
}

fn classify_tensor_name(name: &str) -> Option<&'static str> {
    if name.starts_with("model.embed_tokens.") {
        Some("embed_tokens")
    } else if name.contains(".self_attn.q_proj.") {
        Some("q_proj")
    } else if name.contains(".self_attn.k_proj.") {
        Some("k_proj")
    } else if name.contains(".self_attn.v_proj.") {
        Some("v_proj")
    } else if name.contains(".self_attn.o_proj.") {
        Some("o_proj")
    } else if name.contains(".mlp.gate_proj.") {
        Some("gate_proj")
    } else if name.contains(".mlp.up_proj.") {
        Some("up_proj")
    } else if name.contains(".mlp.down_proj.") {
        Some("down_proj")
    } else if name.contains(".input_layernorm.") {
        Some("input_layernorm")
    } else if name.contains(".post_attention_layernorm.") {
        Some("post_attention_layernorm")
    } else if name.ends_with(".norm.weight") || name.ends_with(".norm.bias") {
        Some("norm")
    } else if name == "lm_head.weight" || name == "lm_head.bias" {
        Some("lm_head")
    } else {
        None
    }
}

fn inspect_safetensors_index(path: &Path) -> Result<Value, String> {
    let index_path = inspect_safetensors_index_path(path);
    if !index_path.exists() {
        return Err(format!(
            "model.safetensors.index.json not found at {}",
            index_path.display()
        ));
    }

    let raw = std::fs::read_to_string(&index_path)
        .map_err(|error| format!("failed to read safetensors index: {error}"))?;
    let value: Value = serde_json::from_str(&raw)
        .map_err(|error| format!("failed to parse safetensors index: {error}"))?;
    let weight_map = value
        .get("weight_map")
        .and_then(|entry| entry.as_object())
        .ok_or_else(|| "safetensors index missing weight_map object".to_string())?;
    let metadata = value.get("metadata").and_then(|entry| entry.as_object());
    let total_size = metadata
        .and_then(|entry| entry.get("total_size"))
        .and_then(|entry| entry.as_u64())
        .unwrap_or(0);

    let mut shard_files = std::collections::BTreeSet::new();
    let mut tensor_names = Vec::new();
    let mut prefix_counts: std::collections::BTreeMap<String, u64> =
        std::collections::BTreeMap::new();
    let mut qwen3_expected_groups: std::collections::BTreeMap<String, u64> =
        std::collections::BTreeMap::new();
    for key in [
        "embed_tokens",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
        "lm_head",
    ] {
        qwen3_expected_groups.insert(key.to_string(), 0);
    }

    for (tensor_name, shard_value) in weight_map {
        tensor_names.push(tensor_name.clone());
        if let Some(shard_name) = shard_value.as_str() {
            shard_files.insert(shard_name.to_string());
        }
        let prefix = tensor_name.split('.').take(3).collect::<Vec<_>>().join(".");
        *prefix_counts.entry(prefix).or_insert(0) += 1;
        if let Some(group) = classify_tensor_name(tensor_name) {
            *qwen3_expected_groups.entry(group.to_string()).or_insert(0) += 1;
        }
    }

    tensor_names.sort();
    let sample_tensor_names = tensor_names.into_iter().take(60).collect::<Vec<_>>();

    Ok(json!({
        "index_path": index_path.display().to_string(),
        "total_size": total_size,
        "shard_count": shard_files.len(),
        "tensor_count": weight_map.len(),
        "shard_files": shard_files.into_iter().collect::<Vec<_>>(),
        "sample_tensor_names": sample_tensor_names,
        "tensor_name_prefix_counts": prefix_counts,
        "qwen3_expected_groups": qwen3_expected_groups,
    }))
}

#[derive(Debug, Clone)]
struct TensorDescriptor {
    dtype: String,
    shape: Vec<u64>,
    byte_offsets: [u64; 2],
    source_file: String,
}

fn safetensors_index_weight_map(
    path: &Path,
) -> Result<(PathBuf, std::collections::BTreeMap<String, String>, u64), String> {
    let index_path = inspect_safetensors_index_path(path);
    if !index_path.exists() {
        return Err(format!(
            "model.safetensors.index.json not found at {}",
            index_path.display()
        ));
    }
    let raw = std::fs::read_to_string(&index_path)
        .map_err(|error| format!("failed to read safetensors index: {error}"))?;
    let value: Value = serde_json::from_str(&raw)
        .map_err(|error| format!("failed to parse safetensors index: {error}"))?;
    let weight_map = value
        .get("weight_map")
        .and_then(|entry| entry.as_object())
        .ok_or_else(|| "safetensors index missing weight_map object".to_string())?;
    let metadata = value.get("metadata").and_then(|entry| entry.as_object());
    let total_size = metadata
        .and_then(|entry| entry.get("total_size"))
        .and_then(|entry| entry.as_u64())
        .unwrap_or(0);
    let mut map = std::collections::BTreeMap::new();
    for (tensor_name, shard_name) in weight_map {
        let shard = shard_name
            .as_str()
            .ok_or_else(|| "safetensors index weight_map entry is not a string".to_string())?;
        map.insert(tensor_name.clone(), shard.to_string());
    }
    Ok((index_path, map, total_size))
}

fn read_safetensors_header(path: &Path) -> Result<Value, String> {
    let mut file = File::open(path).map_err(|error| {
        format!(
            "failed to open safetensors shard {}: {error}",
            path.display()
        )
    })?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).map_err(|error| {
        format!(
            "failed to read safetensors header length {}: {error}",
            path.display()
        )
    })?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len == 0 {
        return Err(format!(
            "safetensors shard has empty header: {}",
            path.display()
        ));
    }
    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes).map_err(|error| {
        format!(
            "failed to read safetensors header {}: {error}",
            path.display()
        )
    })?;
    let header: Value = serde_json::from_slice(&header_bytes).map_err(|error| {
        format!(
            "failed to parse safetensors header {}: {error}",
            path.display()
        )
    })?;
    Ok(header)
}

fn inspect_tensor_group(name: &str) -> Option<(String, Option<u64>)> {
    if name.starts_with("model.embed_tokens.") {
        Some(("embed_tokens".to_string(), None))
    } else if name == "model.norm.weight" || name == "model.norm.bias" {
        Some(("final_norm".to_string(), None))
    } else if name == "lm_head.weight" || name == "lm_head.bias" {
        Some(("lm_head".to_string(), None))
    } else if let Some(rest) = name.strip_prefix("model.layers.") {
        let mut parts = rest.split('.');
        let layer_index = parts.next()?.parse::<u64>().ok()?;
        let section = parts.next()?;
        let group = match (section, parts.next()?) {
            ("self_attn", "q_proj") => "attention.q_proj".to_string(),
            ("self_attn", "k_proj") => "attention.k_proj".to_string(),
            ("self_attn", "v_proj") => "attention.v_proj".to_string(),
            ("self_attn", "o_proj") => "attention.o_proj".to_string(),
            ("mlp", "gate_proj") => "mlp.gate_proj".to_string(),
            ("mlp", "up_proj") => "mlp.up_proj".to_string(),
            ("mlp", "down_proj") => "mlp.down_proj".to_string(),
            ("input_layernorm", _) => "layernorm.input".to_string(),
            ("post_attention_layernorm", _) => "layernorm.post_attention".to_string(),
            ("self_attn", "q_norm") => "layernorm.q_norm".to_string(),
            ("self_attn", "k_norm") => "layernorm.k_norm".to_string(),
            ("norm", _) => "layernorm.norm".to_string(),
            _ => return None,
        };
        Some((group, Some(layer_index)))
    } else {
        None
    }
}

fn summary_is_quantized(tensors: &std::collections::BTreeMap<String, Value>) -> bool {
    let has_weight = tensors.keys().any(|name| name.ends_with(".weight"));
    let has_scales = tensors.keys().any(|name| name.ends_with(".scales"));
    let has_biases = tensors.keys().any(|name| name.ends_with(".biases"));
    has_weight && has_scales && has_biases
}

fn descriptor_group_name(name: &str) -> Option<&'static str> {
    match name {
        "attention.q_proj" => Some("q_proj"),
        "attention.k_proj" => Some("k_proj"),
        "attention.v_proj" => Some("v_proj"),
        "attention.o_proj" => Some("o_proj"),
        "mlp.gate_proj" => Some("gate_proj"),
        "mlp.up_proj" => Some("up_proj"),
        "mlp.down_proj" => Some("down_proj"),
        "layernorm.input" => Some("input_layernorm"),
        "layernorm.post_attention" => Some("post_attention_layernorm"),
        _ => None,
    }
}

fn qwen3_layer_expected_groups() -> Vec<&'static str> {
    vec![
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "input_layernorm",
        "post_attention_layernorm",
    ]
}

fn tensor_group_total_byte_size(group: &Value) -> u64 {
    group
        .get("tensors")
        .and_then(|value| value.as_object())
        .map(|tensors| {
            tensors
                .values()
                .map(|tensor| {
                    if let Some(byte_size) =
                        tensor.get("byte_size").and_then(|value| value.as_u64())
                    {
                        byte_size
                    } else {
                        tensor
                            .get("byte_offsets")
                            .and_then(|value| value.as_array())
                            .and_then(|values| {
                                let start = values.get(0)?.as_u64()?;
                                let end = values.get(1)?.as_u64()?;
                                Some(end.saturating_sub(start))
                            })
                            .unwrap_or(0)
                    }
                })
                .sum()
        })
        .unwrap_or(0)
}

fn inspect_tensor_descriptors(path: &Path) -> Result<Value, String> {
    let (index_path, weight_map, total_size) = safetensors_index_weight_map(path)?;
    let base_dir = index_path
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| format!("safetensors index has no parent: {}", index_path.display()))?;

    let mut shard_headers: std::collections::BTreeMap<String, Value> =
        std::collections::BTreeMap::new();
    for shard_name in weight_map.values() {
        if shard_headers.contains_key(shard_name) {
            continue;
        }
        let shard_path = base_dir.join(shard_name);
        let header = read_safetensors_header(&shard_path)?;
        shard_headers.insert(shard_name.clone(), header);
    }

    let mut descriptors = std::collections::BTreeMap::<String, TensorDescriptor>::new();
    for (tensor_name, shard_name) in &weight_map {
        let header = shard_headers
            .get(shard_name)
            .ok_or_else(|| format!("missing safetensors header for shard: {shard_name}"))?;
        let tensor = header
            .get(tensor_name)
            .ok_or_else(|| format!("tensor {tensor_name} missing from safetensors header"))?;
        let dtype = tensor
            .get("dtype")
            .and_then(|entry| entry.as_str())
            .unwrap_or("unknown")
            .to_string();
        let shape = tensor
            .get("shape")
            .and_then(|entry| entry.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_u64())
                    .collect::<Vec<u64>>()
            })
            .unwrap_or_default();
        let offsets = tensor
            .get("data_offsets")
            .and_then(|entry| entry.as_array())
            .and_then(|items| {
                let start = items.get(0)?.as_u64()?;
                let end = items.get(1)?.as_u64()?;
                Some([start, end])
            })
            .unwrap_or([0, 0]);

        descriptors.insert(
            tensor_name.clone(),
            TensorDescriptor {
                dtype,
                shape,
                byte_offsets: offsets,
                source_file: shard_name.clone(),
            },
        );
    }

    #[derive(Default)]
    struct GroupSummary {
        tensors: std::collections::BTreeMap<String, Value>,
        quantized_group: bool,
    }

    let mut embed_tokens = GroupSummary::default();
    let mut final_norm = GroupSummary::default();
    let mut lm_head = GroupSummary::default();
    let mut layers: std::collections::BTreeMap<
        u64,
        std::collections::BTreeMap<String, GroupSummary>,
    > = std::collections::BTreeMap::new();
    let mut likely_quantized = false;

    for (tensor_name, descriptor) in &descriptors {
        let value = json!({
            "dtype": descriptor.dtype,
            "shape": descriptor.shape,
            "byte_offsets": descriptor.byte_offsets,
            "source_file": descriptor.source_file,
        });
        if tensor_name.contains(".weight") {
            likely_quantized = true;
        }
        if tensor_name.starts_with("model.embed_tokens.") {
            embed_tokens.tensors.insert(tensor_name.clone(), value);
            continue;
        }
        if tensor_name == "model.norm.weight" || tensor_name == "model.norm.bias" {
            final_norm.tensors.insert(tensor_name.clone(), value);
            continue;
        }
        if tensor_name == "lm_head.weight" || tensor_name == "lm_head.bias" {
            lm_head.tensors.insert(tensor_name.clone(), value);
            continue;
        }
        if let Some((group, Some(layer_index))) = inspect_tensor_group(tensor_name) {
            let layer_entry = layers.entry(layer_index).or_default();
            let summary = layer_entry.entry(group).or_default();
            summary.tensors.insert(tensor_name.clone(), value);
        }
    }

    embed_tokens.quantized_group = summary_is_quantized(&embed_tokens.tensors);
    final_norm.quantized_group = summary_is_quantized(&final_norm.tensors);
    lm_head.quantized_group = summary_is_quantized(&lm_head.tensors);
    for groups in layers.values_mut() {
        for summary in groups.values_mut() {
            summary.quantized_group = summary_is_quantized(&summary.tensors);
        }
    }

    let layer_summaries = layers
        .into_iter()
        .map(|(layer_index, groups)| {
            let mut group_json = serde_json::Map::new();
            for (group_name, summary) in groups {
                group_json.insert(
                    group_name,
                    json!({
                        "quantized_group": summary.quantized_group,
                        "tensors": summary.tensors,
                    }),
                );
            }
            json!({
                "layer_index": layer_index,
                "groups": group_json,
            })
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "index_path": index_path.display().to_string(),
        "total_size": total_size,
        "shard_count": shard_headers.len(),
        "tensor_count": descriptors.len(),
        "shard_files": shard_headers.keys().cloned().collect::<Vec<_>>(),
        "likely_quantized": likely_quantized,
        "embed_tokens": {
            "quantized_group": embed_tokens.quantized_group,
            "tensors": embed_tokens.tensors,
        },
        "layers": layer_summaries,
        "final_norm": {
            "quantized_group": final_norm.quantized_group,
            "tensors": final_norm.tensors,
        },
        "lm_head": {
            "present": !lm_head.tensors.is_empty(),
            "quantized_group": lm_head.quantized_group,
            "tensors": lm_head.tensors,
        },
    }))
}

fn model_load_plan(path: &Path) -> Result<Value, String> {
    let config = inspect_model_dir_config(path)?;
    let descriptors = inspect_tensor_descriptors(path)?;

    let model_type = config
        .get("model_type")
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned);
    let architectures = config
        .get("architectures")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    let architecture = architectures
        .first()
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned);
    let layer_count = config
        .get("num_hidden_layers")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let hidden_size = config
        .get("hidden_size")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let vocab_size = config
        .get("vocab_size")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);

    let quantized = descriptors
        .get("likely_quantized")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let tensor_count = descriptors
        .get("tensor_count")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let estimated_total_bytes = descriptors
        .get("total_size")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);

    let embedding_groups = {
        let embed_tokens = descriptors.get("embed_tokens").cloned().unwrap_or_else(|| {
            json!({
                "quantized_group": false,
                "tensors": {}
            })
        });
        let total_byte_size = tensor_group_total_byte_size(&embed_tokens);
        json!({
            "quantized_group": embed_tokens
                .get("quantized_group")
                .and_then(|value| value.as_bool())
                .unwrap_or(false),
            "total_byte_size": total_byte_size,
            "tensors": embed_tokens.get("tensors").cloned().unwrap_or_else(|| json!({})),
        })
    };

    let final_norm = {
        let final_norm = descriptors.get("final_norm").cloned().unwrap_or_else(|| {
            json!({
                "quantized_group": false,
                "tensors": {}
            })
        });
        json!({
            "present": final_norm
                .get("tensors")
                .and_then(|value| value.as_object())
                .map(|value| !value.is_empty())
                .unwrap_or(false),
            "quantized_group": final_norm
                .get("quantized_group")
                .and_then(|value| value.as_bool())
                .unwrap_or(false),
            "tensors": final_norm.get("tensors").cloned().unwrap_or_else(|| json!({})),
        })
    };

    let lm_head = {
        let lm_head = descriptors.get("lm_head").cloned().unwrap_or_else(|| {
            json!({
                "present": false,
                "quantized_group": false,
                "tensors": {}
            })
        });
        json!({
            "present": lm_head
                .get("present")
                .and_then(|value| value.as_bool())
                .unwrap_or(false),
            "quantized_group": lm_head
                .get("quantized_group")
                .and_then(|value| value.as_bool())
                .unwrap_or(false),
            "tensors": lm_head.get("tensors").cloned().unwrap_or_else(|| json!({})),
        })
    };

    let tied_embeddings_likely = !lm_head
        .get("present")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);

    let layer_groups_by_index = descriptors
        .get("layers")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    let mut layer_map: std::collections::BTreeMap<u64, std::collections::BTreeMap<String, Value>> =
        std::collections::BTreeMap::new();
    for layer_entry in layer_groups_by_index {
        let layer_index = layer_entry
            .get("layer_index")
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        let mut groups_map = std::collections::BTreeMap::new();
        if let Some(groups) = layer_entry
            .get("groups")
            .and_then(|value| value.as_object())
        {
            for (group_name, group_value) in groups {
                if let Some(short_name) = descriptor_group_name(group_name) {
                    groups_map.insert(short_name.to_string(), group_value.clone());
                }
            }
        }
        layer_map.insert(layer_index, groups_map);
    }

    let mut per_layer_summary = Vec::new();
    let expected_groups = if model_type.as_deref() == Some("qwen3")
        || architecture
            .as_deref()
            .map(|name| name.contains("Qwen3"))
            .unwrap_or(false)
    {
        qwen3_layer_expected_groups()
    } else {
        Vec::new()
    };

    for layer_index in 0..layer_count {
        let groups = layer_map.get(&layer_index).cloned().unwrap_or_default();
        let mut found_groups = groups.keys().cloned().collect::<Vec<_>>();
        found_groups.sort();
        let mut missing_groups = Vec::new();
        for expected in &expected_groups {
            if !groups.contains_key(*expected) {
                missing_groups.push((*expected).to_string());
            }
        }
        let total_byte_size: u64 = groups.values().map(tensor_group_total_byte_size).sum();
        per_layer_summary.push(json!({
            "layer": layer_index,
            "expected_groups": expected_groups.clone(),
            "found_groups": found_groups,
            "missing_groups": missing_groups,
            "total_byte_size": total_byte_size,
        }));
    }

    let embedding_groups_obj = embedding_groups.as_object().cloned().unwrap_or_default();

    Ok(json!({
        "model_type": model_type,
        "architecture": architecture,
        "layer_count": layer_count,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "quantized": quantized,
        "tensor_count": tensor_count,
        "estimated_total_bytes": estimated_total_bytes,
        "embedding_groups": embedding_groups_obj,
        "per_layer_summary": per_layer_summary,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "tied_embeddings_likely": tied_embeddings_likely,
    }))
}

fn model_descriptor_from_plan(
    model_dir: &str,
    plan: &Value,
) -> Result<ModelDescriptorRecord, String> {
    let model_type = plan
        .get("model_type")
        .and_then(|value| value.as_str())
        .ok_or_else(|| "model_load_plan missing model_type".to_string())?
        .to_string();
    let architecture = plan
        .get("architecture")
        .and_then(|value| value.as_str())
        .ok_or_else(|| "model_load_plan missing architecture".to_string())?
        .to_string();
    let layer_count = plan
        .get("layer_count")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| "model_load_plan missing layer_count".to_string())?;
    let hidden_size = plan
        .get("hidden_size")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| "model_load_plan missing hidden_size".to_string())?;
    let vocab_size = plan
        .get("vocab_size")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| "model_load_plan missing vocab_size".to_string())?;
    let quantized = plan
        .get("quantized")
        .and_then(|value| value.as_bool())
        .ok_or_else(|| "model_load_plan missing quantized".to_string())?;
    let total_estimated_bytes = plan
        .get("estimated_total_bytes")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| "model_load_plan missing estimated_total_bytes".to_string())?;
    let tied_embeddings = plan
        .get("tied_embeddings_likely")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);

    let per_layer_summary = plan
        .get("per_layer_summary")
        .and_then(|value| value.as_array())
        .ok_or_else(|| "model_load_plan missing per_layer_summary".to_string())?;
    if per_layer_summary.len() != layer_count as usize {
        return Err(format!(
            "model descriptor validation failed: expected {layer_count} per-layer summaries, got {}",
            per_layer_summary.len()
        ));
    }
    for layer_entry in per_layer_summary {
        let layer_index = layer_entry
            .get("layer")
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        let missing_groups = layer_entry
            .get("missing_groups")
            .and_then(|value| value.as_array())
            .ok_or_else(|| format!("layer {layer_index} missing missing_groups array"))?;
        if !missing_groups.is_empty() {
            return Err(format!(
                "model descriptor validation failed: layer {layer_index} is missing groups: {}",
                missing_groups.len()
            ));
        }
    }

    let embedding_groups = plan
        .get("embedding_groups")
        .and_then(|value| value.as_object())
        .ok_or_else(|| "model_load_plan missing embedding_groups".to_string())?;
    let embedding_tensors = embedding_groups
        .get("tensors")
        .and_then(|value| value.as_object())
        .ok_or_else(|| "model_load_plan missing embedding tensor metadata".to_string())?;
    if embedding_tensors.is_empty() {
        return Err("model descriptor validation failed: embedding groups are empty".to_string());
    }
    let embedding_present = embedding_groups
        .get("total_byte_size")
        .and_then(|value| value.as_u64())
        .unwrap_or(0)
        > 0;
    if !embedding_present {
        return Err(
            "model descriptor validation failed: embedding groups were not present".to_string(),
        );
    }

    let final_norm = plan
        .get("final_norm")
        .and_then(|value| value.as_object())
        .ok_or_else(|| "model_load_plan missing final_norm".to_string())?;
    if !final_norm
        .get("present")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
    {
        return Err("model descriptor validation failed: final norm missing".to_string());
    }
    let final_norm_tensors = final_norm
        .get("tensors")
        .and_then(|value| value.as_object())
        .ok_or_else(|| "model_load_plan missing final_norm tensors".to_string())?;
    if final_norm_tensors.is_empty() {
        return Err("model descriptor validation failed: final norm tensors are empty".to_string());
    }

    let lm_head = plan
        .get("lm_head")
        .and_then(|value| value.as_object())
        .ok_or_else(|| "model_load_plan missing lm_head".to_string())?;
    let lm_head_present = lm_head
        .get("present")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let tied_embeddings = if lm_head_present {
        false
    } else {
        tied_embeddings
    };
    if !lm_head_present && !tied_embeddings {
        return Err(
            "model descriptor validation failed: tied embeddings were not detected".to_string(),
        );
    }

    Ok(ModelDescriptorRecord {
        model_dir: model_dir.to_string(),
        model_type,
        architecture,
        layer_count,
        hidden_size,
        vocab_size,
        quantized,
        tied_embeddings,
        total_estimated_bytes,
        loaded_layers: 0,
        loaded_weights: false,
    })
}

fn layer_group_specs(layer: u64) -> Vec<(String, String, bool)> {
    let prefix = format!("model.layers.{layer}.");
    vec![
        (
            "q_proj".to_string(),
            format!("{prefix}self_attn.q_proj"),
            true,
        ),
        (
            "k_proj".to_string(),
            format!("{prefix}self_attn.k_proj"),
            true,
        ),
        (
            "v_proj".to_string(),
            format!("{prefix}self_attn.v_proj"),
            true,
        ),
        (
            "o_proj".to_string(),
            format!("{prefix}self_attn.o_proj"),
            true,
        ),
        (
            "gate_proj".to_string(),
            format!("{prefix}mlp.gate_proj"),
            true,
        ),
        ("up_proj".to_string(), format!("{prefix}mlp.up_proj"), true),
        (
            "down_proj".to_string(),
            format!("{prefix}mlp.down_proj"),
            true,
        ),
        (
            "input_layernorm".to_string(),
            format!("{prefix}input_layernorm"),
            false,
        ),
        (
            "post_attention_layernorm".to_string(),
            format!("{prefix}post_attention_layernorm"),
            false,
        ),
    ]
}

fn layer_group_byte_size(info: &Value) -> u64 {
    ["weight", "scales", "biases"]
        .iter()
        .map(|key| {
            info.get(key)
                .and_then(|value| value.get("byte_size"))
                .and_then(|value| value.as_u64())
                .unwrap_or(0)
        })
        .sum()
}

pub fn dispatch(state: &mut BridgeState, request: Request) -> Response {
    if state.shutdown_requested
        && request.cmd != "bridge_health"
        && request.cmd != "bridge_shutdown"
    {
        return bridge_shutting_down(&request);
    }

    match request.cmd.as_str() {
        "bridge_health" => Response::ok(
            request.id,
            json!({
                "status": if state.shutdown_requested {
                    json!("shutting_down")
                } else {
                    alive_value()["status"].clone()
                },
                "counts": state.counts_json()
            }),
        ),
        "backend_probe" => Response::ok(request.id, probe::run_probe()),
        "inspect_model_dir" => {
            let path = match require_arg_string(&request, "path") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let path_buf = Path::new(&path);
            let exists = path_buf.exists();

            let (
                tokenizer_files,
                safetensors_files,
                safetensors_index,
                generation_config,
                file_count,
                total_size,
            ) = match inspect_model_dir_files(path_buf) {
                Ok(values) => values,
                Err(message) => {
                    return Response::err(request.id.clone(), "bad_args", message);
                }
            };
            let config = match inspect_model_dir_config(path_buf) {
                Ok(value) => value,
                Err(message) => {
                    return Response::err(request.id.clone(), "bad_args", message);
                }
            };
            let guessed_model_family = config
                .get("model_type")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned);

            Response::ok(
                request.id,
                json!({
                    "exists": exists,
                    "path": path,
                    "config_json": if exists { Some(path_buf.join("config.json").display().to_string()) } else { None::<String> },
                    "tokenizer_files": tokenizer_files,
                    "safetensors_files": safetensors_files,
                    "safetensors_index": safetensors_index,
                    "generation_config": generation_config,
                    "file_count": file_count,
                    "total_size_bytes": total_size,
                    "guessed_model_family": guessed_model_family,
                    "config": config
                }),
            )
        }
        "inspect_tensor_descriptors" => {
            let path = match require_arg_string(&request, "path") {
                Ok(value) => value,
                Err(response) => return response,
            };
            match inspect_tensor_descriptors(Path::new(&path)) {
                Ok(value) => Response::ok(request.id, value),
                Err(message) => Response::err(request.id, "bad_args", message),
            }
        }
        "model_load_plan" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            match model_load_plan(Path::new(&model_dir)) {
                Ok(value) => Response::ok(request.id, value),
                Err(message) => Response::err(request.id, "bad_args", message),
            }
        }
        "create_model_descriptor" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let plan = match model_load_plan(Path::new(&model_dir)) {
                Ok(value) => value,
                Err(message) => return Response::err(request.id, "bad_args", message),
            };
            let record = match model_descriptor_from_plan(&model_dir, &plan) {
                Ok(record) => record,
                Err(message) => return Response::err(request.id, "bad_args", message),
            };
            let handle = state.handles.next("mdesc");
            state
                .model_descriptors
                .insert(handle.clone(), record.clone());
            Response::ok(
                request.id,
                json!({
                    "descriptor": handle,
                    "model_dir": model_dir,
                    "model_type": record.model_type,
                    "architecture": record.architecture,
                    "layer_count": record.layer_count,
                    "hidden_size": record.hidden_size,
                    "vocab_size": record.vocab_size,
                    "quantized": record.quantized,
                    "tied_embeddings": record.tied_embeddings,
                    "total_estimated_bytes": record.total_estimated_bytes,
                    "loaded_layers": record.loaded_layers,
                    "loaded_weights": record.loaded_weights,
                }),
            )
        }
        "model_descriptor_info" => {
            let handle = match require_arg_string(&request, "descriptor") {
                Ok(value) => value,
                Err(response) => return response,
            };
            if state.freed_model_descriptors.contains(&handle) {
                return already_freed(&request, "model descriptor", &handle);
            }
            let record = match state.model_descriptors.get(&handle) {
                Some(record) => record.clone(),
                None => return unknown_handle(&request, "model descriptor", &handle),
            };
            Response::ok(
                request.id,
                json!({
                    "descriptor": handle,
                    "model_dir": record.model_dir,
                    "model_type": record.model_type,
                    "architecture": record.architecture,
                    "layer_count": record.layer_count,
                    "hidden_size": record.hidden_size,
                    "vocab_size": record.vocab_size,
                    "quantized": record.quantized,
                    "tied_embeddings": record.tied_embeddings,
                    "total_estimated_bytes": record.total_estimated_bytes,
                    "loaded_layers": record.loaded_layers,
                    "loaded_weights": record.loaded_weights,
                }),
            )
        }
        "free_model_descriptor" => {
            let handle = match require_arg_string(&request, "descriptor") {
                Ok(value) => value,
                Err(response) => return response,
            };
            if state.freed_model_descriptors.contains(&handle) {
                return already_freed(&request, "model descriptor", &handle);
            }
            if !state.model_descriptors.contains_key(&handle) {
                return unknown_handle(&request, "model descriptor", &handle);
            }
            state.model_descriptors.remove(&handle);
            state.freed_model_descriptors.insert(handle.clone());
            Response::ok(
                request.id,
                json!({
                    "descriptor": handle,
                    "freed": true
                }),
            )
        }
        "inspect_safetensors_index" => {
            let path = match require_arg_string(&request, "path") {
                Ok(value) => value,
                Err(response) => return response,
            };
            match inspect_safetensors_index(Path::new(&path)) {
                Ok(value) => Response::ok(request.id, value),
                Err(message) => Response::err(request.id, "bad_args", message),
            }
        }
        "load_tensor_group" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let group = match require_arg_string(&request, "group") {
                Ok(value) => value,
                Err(response) => return response,
            };

            let handle = state.handles.next("tgrp");
            let result = shim::load_tensor_group(&handle, &model_dir, &group);
            if result.get("loaded").and_then(|value| value.as_bool()) == Some(true) {
                state.tensor_groups.insert(
                    handle.clone(),
                    TensorGroupRecord {
                        model_dir: model_dir.clone(),
                        group: group.clone(),
                    },
                );
                Response::ok(
                    request.id,
                    json!({
                        "group": handle,
                        "model_dir": model_dir,
                        "target_group": group,
                        "loaded": true
                    }),
                )
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    format!("failed to load tensor group {group} from {model_dir}"),
                )
            }
        }
        "tensor_group_info" => {
            let handle = match require_arg_string(&request, "group") {
                Ok(value) => value,
                Err(response) => return response,
            };
            if state.freed_tensor_groups.contains(&handle) {
                return already_freed(&request, "tensor group", &handle);
            }
            if !state.tensor_groups.contains_key(&handle) {
                return unknown_handle(&request, "tensor group", &handle);
            }
            let result = shim::tensor_group_info(&handle);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "unknown_handle",
                    format!("tensor group handle not found: {handle}"),
                )
            }
        }
        "quantization_layout_probe" => {
            let handle = match require_arg_string(&request, "group") {
                Ok(value) => value,
                Err(response) => return response,
            };
            if state.freed_tensor_groups.contains(&handle) {
                return already_freed(&request, "tensor group", &handle);
            }
            if !state.tensor_groups.contains_key(&handle) {
                return unknown_handle(&request, "tensor group", &handle);
            }
            let result = shim::quantization_layout_probe(&handle);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    format!("failed to probe quantization layout for tensor group {handle}"),
                )
            }
        }
        "compare_dequant_slice" => {
            let handle = match require_arg_string(&request, "group") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let row = match request.args.get("row") {
                Some(value) => match value.as_u64() {
                    Some(value) => value,
                    None => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "compare_dequant_slice requires row to be an integer".to_string(),
                        );
                    }
                },
                None => 0,
            };
            let cols = match request.args.get("cols") {
                Some(value) => match value.as_u64() {
                    Some(value) if value > 0 => value,
                    Some(_) => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "compare_dequant_slice requires cols to be a positive integer"
                                .to_string(),
                        );
                    }
                    None => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "compare_dequant_slice requires cols to be an integer".to_string(),
                        );
                    }
                },
                None => 8,
            };
            if state.freed_tensor_groups.contains(&handle) {
                return already_freed(&request, "tensor group", &handle);
            }
            if !state.tensor_groups.contains_key(&handle) {
                return unknown_handle(&request, "tensor group", &handle);
            }
            let result = shim::compare_dequant_slice(&handle, row, cols);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    format!("failed to compare dequant slice for tensor group {handle}"),
                )
            }
        }
        "quantized_linear_slice_probe" => {
            let handle = match require_arg_string(&request, "group") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let input = match request.args.get("input") {
                Some(value) => match value.as_array() {
                    Some(values) if !values.is_empty() => {
                        let mut out = Vec::with_capacity(values.len());
                        for entry in values {
                            match entry.as_f64() {
                                Some(number) => out.push(number),
                                None => {
                                    return Response::err(
                                        request.id,
                                        "bad_args",
                                        "quantized_linear_slice_probe requires input to be an array of numbers".to_string(),
                                    );
                                }
                            }
                        }
                        out
                    }
                    _ => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "quantized_linear_slice_probe requires input to be a non-empty array"
                                .to_string(),
                        );
                    }
                },
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "quantized_linear_slice_probe requires input".to_string(),
                    );
                }
            };
            let row = match request.args.get("row") {
                Some(value) => match value.as_u64() {
                    Some(value) => value,
                    None => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "quantized_linear_slice_probe requires row to be an integer"
                                .to_string(),
                        );
                    }
                },
                None => 0,
            };
            let cols = match request.args.get("cols") {
                Some(value) => match value.as_u64() {
                    Some(value) if value > 0 => value,
                    Some(_) => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "quantized_linear_slice_probe requires cols to be a positive integer"
                                .to_string(),
                        );
                    }
                    None => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "quantized_linear_slice_probe requires cols to be an integer"
                                .to_string(),
                        );
                    }
                },
                None => 8,
            };
            if state.freed_tensor_groups.contains(&handle) {
                return already_freed(&request, "tensor group", &handle);
            }
            if !state.tensor_groups.contains_key(&handle) {
                return unknown_handle(&request, "tensor group", &handle);
            }
            let result = shim::quantized_linear_slice_probe(&handle, &input, row, cols);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    format!("failed to run quantized linear slice probe for tensor group {handle}"),
                )
            }
        }
        "quantized_linear_rows_probe" => {
            let handle = match require_arg_string(&request, "group") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let input = match request.args.get("input") {
                Some(value) => match value.as_array() {
                    Some(values) if !values.is_empty() => {
                        let mut out = Vec::with_capacity(values.len());
                        for entry in values {
                            match entry.as_f64() {
                                Some(number) => out.push(number),
                                None => {
                                    return Response::err(
                                        request.id,
                                        "bad_args",
                                        "quantized_linear_rows_probe requires input to be an array of numbers".to_string(),
                                    );
                                }
                            }
                        }
                        out
                    }
                    _ => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "quantized_linear_rows_probe requires input to be a non-empty array"
                                .to_string(),
                        );
                    }
                },
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "quantized_linear_rows_probe requires input".to_string(),
                    );
                }
            };
            let rows = match request.args.get("rows") {
                Some(value) => match value.as_u64() {
                    Some(value) if value > 0 => value,
                    Some(_) => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "quantized_linear_rows_probe requires rows to be a positive integer"
                                .to_string(),
                        );
                    }
                    None => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "quantized_linear_rows_probe requires rows to be an integer"
                                .to_string(),
                        );
                    }
                },
                None => 4,
            };
            let cols = match request.args.get("cols") {
                Some(value) => match value.as_u64() {
                    Some(value) if value > 0 => value,
                    Some(_) => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "quantized_linear_rows_probe requires cols to be a positive integer"
                                .to_string(),
                        );
                    }
                    None => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "quantized_linear_rows_probe requires cols to be an integer"
                                .to_string(),
                        );
                    }
                },
                None => 8,
            };
            if state.freed_tensor_groups.contains(&handle) {
                return already_freed(&request, "tensor group", &handle);
            }
            if !state.tensor_groups.contains_key(&handle) {
                return unknown_handle(&request, "tensor group", &handle);
            }
            let result = shim::quantized_linear_rows_probe(&handle, &input, rows, cols);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    format!("failed to run quantized linear rows probe for tensor group {handle}"),
                )
            }
        }
        "quantized_linear_fullrow_probe" => {
            let handle = match require_arg_string(&request, "group") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let input = match request.args.get("input") {
                Some(value) => match value.as_array() {
                    Some(values) if !values.is_empty() => {
                        let mut out = Vec::with_capacity(values.len());
                        for entry in values {
                            match entry.as_f64() {
                                Some(number) => out.push(number),
                                None => {
                                    return Response::err(
                                        request.id,
                                        "bad_args",
                                        "quantized_linear_fullrow_probe requires input to be an array of numbers".to_string(),
                                    );
                                }
                            }
                        }
                        out
                    }
                    _ => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "quantized_linear_fullrow_probe requires input to be a non-empty array"
                                .to_string(),
                        );
                    }
                },
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "quantized_linear_fullrow_probe requires input".to_string(),
                    );
                }
            };
            let rows = match request.args.get("rows") {
                Some(value) => match value.as_u64() {
                    Some(value) if value > 0 => value,
                    Some(_) => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "quantized_linear_fullrow_probe requires rows to be a positive integer"
                                .to_string(),
                        );
                    }
                    None => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "quantized_linear_fullrow_probe requires rows to be an integer"
                                .to_string(),
                        );
                    }
                },
                None => 4,
            };
            if state.freed_tensor_groups.contains(&handle) {
                return already_freed(&request, "tensor group", &handle);
            }
            if !state.tensor_groups.contains_key(&handle) {
                return unknown_handle(&request, "tensor group", &handle);
            }
            let result = shim::quantized_linear_fullrow_probe(&handle, &input, rows);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    format!(
                        "failed to run quantized linear full-row probe for tensor group {handle}"
                    ),
                )
            }
        }
        "quantized_linear_vector_probe" => {
            let handle = match require_arg_string(&request, "group") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let input = match request.args.get("input") {
                Some(value) => match value.as_array() {
                    Some(values) if !values.is_empty() => {
                        let mut out = Vec::with_capacity(values.len());
                        for entry in values {
                            match entry.as_f64() {
                                Some(number) => out.push(number),
                                None => {
                                    return Response::err(
                                        request.id,
                                        "bad_args",
                                        "quantized_linear_vector_probe requires input to be an array of numbers".to_string(),
                                    );
                                }
                            }
                        }
                        out
                    }
                    _ => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "quantized_linear_vector_probe requires input to be a non-empty array"
                                .to_string(),
                        );
                    }
                },
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "quantized_linear_vector_probe requires input".to_string(),
                    );
                }
            };
            if state.freed_tensor_groups.contains(&handle) {
                return already_freed(&request, "tensor group", &handle);
            }
            if !state.tensor_groups.contains_key(&handle) {
                return unknown_handle(&request, "tensor group", &handle);
            }
            let result = shim::quantized_linear_vector_probe(&handle, &input);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    format!(
                        "failed to run quantized linear vector probe for tensor group {handle}"
                    ),
                )
            }
        }
        "rmsnorm_probe" => {
            let handle = match require_arg_string(&request, "group") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let input = match request.args.get("input") {
                Some(value) => match value.as_array() {
                    Some(values) if !values.is_empty() => {
                        let mut out = Vec::with_capacity(values.len());
                        for entry in values {
                            match entry.as_f64() {
                                Some(number) => out.push(number),
                                None => {
                                    return Response::err(
                                        request.id,
                                        "bad_args",
                                        "rmsnorm_probe requires input to be an array of numbers"
                                            .to_string(),
                                    );
                                }
                            }
                        }
                        out
                    }
                    _ => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "rmsnorm_probe requires input to be a non-empty array".to_string(),
                        );
                    }
                },
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "rmsnorm_probe requires input".to_string(),
                    );
                }
            };
            if state.freed_tensor_groups.contains(&handle) {
                return already_freed(&request, "tensor group", &handle);
            }
            if !state.tensor_groups.contains_key(&handle) {
                return unknown_handle(&request, "tensor group", &handle);
            }
            let result = shim::rmsnorm_probe(&handle, &input);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    format!("failed to run rmsnorm probe for tensor group {handle}"),
                )
            }
        }
        "layer0_single_token_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let token_id = match request
                .args
                .get("token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "layer0_single_token_probe requires token_id".to_string(),
                    );
                }
            };
            let result = shim::layer0_single_token_probe(&model_dir, token_id);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run layer0 single-token probe".to_string(),
                )
            }
        }
        "layer0_mlp_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let token_id = match request
                .args
                .get("token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "layer0_mlp_probe requires token_id".to_string(),
                    );
                }
            };
            let result = shim::layer0_mlp_probe(&model_dir, token_id);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run layer0 MLP probe".to_string(),
                )
            }
        }
        "layer0_block_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let token_id = match request
                .args
                .get("token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "layer0_block_probe requires token_id".to_string(),
                    );
                }
            };
            let result = shim::layer0_block_probe(&model_dir, token_id);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run layer0 block probe".to_string(),
                )
            }
        }
        "layer_stack_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let token_id = match request
                .args
                .get("token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "layer_stack_probe requires token_id".to_string(),
                    );
                }
            };
            let layers = match request.args.get("layers") {
                Some(value) => match value.as_u64() {
                    Some(value) if value != 0 => value,
                    _ => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "layer_stack_probe requires layers to be a positive integer"
                                .to_string(),
                        );
                    }
                },
                None => 2,
            };
            let result = shim::layer_stack_probe(&model_dir, token_id, layers);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run layer stack probe".to_string(),
                )
            }
        }
        "full_stack_single_token_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let token_id = match request
                .args
                .get("token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "full_stack_single_token_probe requires token_id".to_string(),
                    );
                }
            };
            let result = shim::full_stack_single_token_probe(&model_dir, token_id);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run full-stack single-token probe".to_string(),
                )
            }
        }
        "kv_cache_storage_probe" => {
            let result = shim::kv_cache_storage_probe();
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to inspect native structural KV cache".to_string(),
                )
            }
        }
        "incremental_attention_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let prompt_token_id = match request
                .args
                .get("prompt_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "incremental_attention_probe requires prompt_token_id".to_string(),
                    );
                }
            };
            let decode_token_id = match request
                .args
                .get("decode_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "incremental_attention_probe requires decode_token_id".to_string(),
                    );
                }
            };
            let result = shim::incremental_attention_probe(
                &model_dir,
                prompt_token_id,
                decode_token_id,
            );
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run incremental attention probe".to_string(),
                )
            }
        }
        "session_layer_residency_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let prompt_token_id = match request
                .args
                .get("prompt_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "session_layer_residency_probe requires prompt_token_id".to_string(),
                    );
                }
            };
            let decode_token_id = match request
                .args
                .get("decode_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "session_layer_residency_probe requires decode_token_id".to_string(),
                    );
                }
            };
            let result = shim::session_layer_residency_probe(
                &model_dir,
                prompt_token_id,
                decode_token_id,
            );
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run session layer residency probe".to_string(),
                )
            }
        }
        "fastsmoke_generation_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let prompt_token_id = match request
                .args
                .get("prompt_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "fastsmoke_generation_probe requires prompt_token_id".to_string(),
                    );
                }
            };
            let first_decode_token_id = match request
                .args
                .get("first_decode_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "fastsmoke_generation_probe requires first_decode_token_id".to_string(),
                    );
                }
            };
            let generated_tokens = request
                .args
                .get("generated_tokens")
                .and_then(|value| value.as_u64())
                .unwrap_or(3);
            let result = shim::fastsmoke_generation_probe(
                &model_dir,
                prompt_token_id,
                first_decode_token_id,
                generated_tokens,
            );
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run fastsmoke generation probe".to_string(),
                )
            }
        }
        "resident_incremental_timing_breakdown_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let decode_token_id = match request
                .args
                .get("decode_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "resident_incremental_timing_breakdown_probe requires decode_token_id"
                            .to_string(),
                    );
                }
            };
            let result = shim::resident_incremental_timing_breakdown_probe(
                &model_dir,
                decode_token_id,
            );
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run resident incremental timing breakdown probe".to_string(),
                )
            }
        }
        "quantized_linear_kernel_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let result = shim::quantized_linear_kernel_probe(&model_dir);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run quantized linear kernel probe".to_string(),
                )
            }
        }
        "quantized_linear_mlx_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let result = shim::quantized_linear_mlx_probe(&model_dir);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run MLX quantized linear probe".to_string(),
                )
            }
        }
        "metal_first_resident_decode_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let prompt_token_id = match request
                .args
                .get("prompt_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "metal_first_resident_decode_probe requires prompt_token_id".to_string(),
                    );
                }
            };
            let decode_token_id = match request
                .args
                .get("decode_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "metal_first_resident_decode_probe requires decode_token_id".to_string(),
                    );
                }
            };
            let result = shim::metal_first_resident_decode_probe(
                &model_dir,
                prompt_token_id,
                decode_token_id,
            );
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run Metal-first resident decode probe".to_string(),
                )
            }
        }
        "resident_incremental_optimized_compare_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let decode_token_id = match request
                .args
                .get("decode_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "resident_incremental_optimized_compare_probe requires decode_token_id"
                            .to_string(),
                    );
                }
            };
            let result = shim::resident_incremental_optimized_compare_probe(
                &model_dir,
                decode_token_id,
            );
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run resident incremental optimized comparison probe".to_string(),
                )
            }
        }
        "resident_incremental_layout_cached_compare_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let decode_token_id = match request
                .args
                .get("decode_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "resident_incremental_layout_cached_compare_probe requires decode_token_id"
                            .to_string(),
                    );
                }
            };
            let result = shim::resident_incremental_layout_cached_compare_probe(
                &model_dir,
                decode_token_id,
            );
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run resident incremental layout-cached comparison probe".to_string(),
                )
            }
        }
        "resident_incremental_mlp_optimized_compare_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let decode_token_id = match request
                .args
                .get("decode_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "resident_incremental_mlp_optimized_compare_probe requires decode_token_id"
                            .to_string(),
                    );
                }
            };
            let result = shim::resident_incremental_mlp_optimized_compare_probe(
                &model_dir,
                decode_token_id,
            );
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run resident incremental MLP-optimized comparison probe".to_string(),
                )
            }
        }
        "resident_incremental_logits_optimized_compare_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let decode_token_id = match request
                .args
                .get("decode_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "resident_incremental_logits_optimized_compare_probe requires decode_token_id"
                            .to_string(),
                    );
                }
            };
            let result = shim::resident_incremental_logits_optimized_compare_probe(
                &model_dir,
                decode_token_id,
            );
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run resident incremental logits-optimized comparison probe".to_string(),
                )
            }
        }
        "resident_incremental_down_full_block_compare_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let decode_token_id = match request
                .args
                .get("decode_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "resident_incremental_down_full_block_compare_probe requires decode_token_id"
                            .to_string(),
                    );
                }
            };
            let result = shim::resident_incremental_down_full_block_compare_probe(
                &model_dir,
                decode_token_id,
            );
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run resident incremental down full-block comparison probe"
                        .to_string(),
                )
            }
        }
        "resident_incremental_gate_up_full_block_compare_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let decode_token_id = match request
                .args
                .get("decode_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "resident_incremental_gate_up_full_block_compare_probe requires decode_token_id"
                            .to_string(),
                    );
                }
            };
            let result = shim::resident_incremental_gate_up_full_block_compare_probe(
                &model_dir,
                decode_token_id,
            );
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    "failed to run resident incremental gate/up full-block comparison probe"
                        .to_string(),
                )
            }
        }
        "greedy_next_token_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let tokenizer = match require_arg_string(&request, "tokenizer") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let input_token_id = match request
                .args
                .get("input_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "greedy_next_token_probe requires input_token_id".to_string(),
                    );
                }
            };
            if state.freed_tokenizers.contains(&tokenizer) {
                return already_freed(&request, "tokenizer", &tokenizer);
            }
            let tokenizer_record = match state.tokenizers.get(&tokenizer) {
                Some(record) => record.clone(),
                None => return unknown_handle(&request, "tokenizer", &tokenizer),
            };
            let mut result = cached_greedy_next_token_probe(state, &model_dir, input_token_id);
            if result.get("ok").and_then(|value| value.as_bool()) != Some(true) {
                return Response::err(
                    request.id,
                    "bad_args",
                    "failed to run greedy next-token probe".to_string(),
                );
            }
            let next_token_id = match result.get("next_token_id").and_then(|value| value.as_u64()) {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "greedy next-token probe returned no next_token_id".to_string(),
                    );
                }
            };
            let decoded = match tokenizer_decode_fixture(&tokenizer_record, &[next_token_id]) {
                Ok(value) => value,
                Err(message) => {
                    return Response::err(request.id, "tokenizer_unknown_token", message);
                }
            };
            if let Some(object) = result.as_object_mut() {
                object.insert("decoded_next_token".to_string(), json!(decoded));
            }
            Response::ok(request.id, result)
        }
        "greedy_two_token_probe" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let tokenizer = match require_arg_string(&request, "tokenizer") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let input_token_id = match request
                .args
                .get("input_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "greedy_two_token_probe requires input_token_id".to_string(),
                    );
                }
            };
            if state.freed_tokenizers.contains(&tokenizer) {
                return already_freed(&request, "tokenizer", &tokenizer);
            }
            let tokenizer_record = match state.tokenizers.get(&tokenizer) {
                Some(record) => record,
                None => return unknown_handle(&request, "tokenizer", &tokenizer),
            };
            let mut result = shim::greedy_two_token_probe(&model_dir, input_token_id);
            if result.get("ok").and_then(|value| value.as_bool()) != Some(true) {
                return Response::err(
                    request.id,
                    "bad_args",
                    "failed to run greedy two-token probe".to_string(),
                );
            }
            let first = match result
                .get("first_next_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "greedy two-token probe returned no first_next_token_id".to_string(),
                    );
                }
            };
            let second = match result
                .get("second_next_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "greedy two-token probe returned no second_next_token_id".to_string(),
                    );
                }
            };
            let decoded = match tokenizer_decode_fixture(tokenizer_record, &[first, second]) {
                Ok(value) => value,
                Err(message) => {
                    return Response::err(request.id, "tokenizer_unknown_token", message);
                }
            };
            if let Some(object) = result.as_object_mut() {
                object.insert("decoded_generated_text".to_string(), json!(decoded));
            }
            Response::ok(request.id, result)
        }
        "greedy_session_generate_probe" => {
            let session = match require_arg_string(&request, "session") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let input_token_id = match request
                .args
                .get("input_token_id")
                .and_then(|value| value.as_u64())
            {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "greedy_session_generate_probe requires input_token_id".to_string(),
                    );
                }
            };
            let max_tokens = match request.args.get("max_tokens") {
                Some(value) => match value.as_u64() {
                    Some(value) if value != 0 => value,
                    _ => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "greedy_session_generate_probe requires max_tokens to be a positive integer"
                                .to_string(),
                        );
                    }
                },
                None => 3,
            };

            if state.freed_native_sessions.contains(&session) {
                return already_freed(&request, "native session", &session);
            }
            let session_record = match state.native_sessions.get(&session) {
                Some(record) => record,
                None => return unknown_handle(&request, "native session", &session),
            };
            let model = match &session_record.model {
                Some(value) => value.clone(),
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        format!("native session {session} has no model handle"),
                    );
                }
            };
            let tokenizer = match &session_record.tokenizer {
                Some(value) => value.clone(),
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        format!("native session {session} has no tokenizer handle"),
                    );
                }
            };
            if state.freed_native_models.contains(&model) {
                return already_freed(&request, "native model", &model);
            }
            if state.freed_tokenizers.contains(&tokenizer) {
                return already_freed(&request, "tokenizer", &tokenizer);
            }
            let model_record = match state.native_models.get(&model) {
                Some(record) => record.clone(),
                None => return unknown_handle(&request, "native model", &model),
            };
            let tokenizer_record = match state.tokenizers.get(&tokenizer) {
                Some(record) => record.clone(),
                None => return unknown_handle(&request, "tokenizer", &tokenizer),
            };

            let mut current = input_token_id;
            let mut generated = Vec::new();
            let mut steps = Vec::new();
            for step_index in 0..max_tokens {
                let result = cached_greedy_next_token_probe(state, &model_record.path, current);
                if result.get("ok").and_then(|value| value.as_bool()) != Some(true) {
                    return Response::err(
                        request.id,
                        "bad_args",
                        format!("failed to run greedy session generation step {step_index}"),
                    );
                }
                let next = match result.get("next_token_id").and_then(|value| value.as_u64()) {
                    Some(value) => value,
                    None => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            format!("greedy session generation step {step_index} returned no next_token_id"),
                        );
                    }
                };
                let score = match result
                    .get("next_token_score")
                    .and_then(|value| value.as_f64())
                {
                    Some(value) => value,
                    None => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            format!(
                                "greedy session generation step {step_index} returned no score"
                            ),
                        );
                    }
                };
                generated.push(next);
                let decoded = match tokenizer_decode_fixture(&tokenizer_record, &generated) {
                    Ok(value) => value,
                    Err(message) => {
                        return Response::err(request.id, "tokenizer_unknown_token", message);
                    }
                };
                let timing_ms = result
                    .get("timing_ms")
                    .and_then(|value| value.as_f64())
                    .unwrap_or(0.0);
                steps.push(json!({
                    "step": step_index,
                    "input_token": current,
                    "next_token": next,
                    "next_token_score": score,
                    "decoded_accumulated_text": decoded,
                    "timing_ms": timing_ms
                }));
                current = next;
            }
            let decoded_generated_text =
                match tokenizer_decode_fixture(&tokenizer_record, &generated) {
                    Ok(value) => value,
                    Err(message) => {
                        return Response::err(request.id, "tokenizer_unknown_token", message);
                    }
                };
            Response::ok(
                request.id,
                json!({
                    "session": session,
                    "model": model,
                    "tokenizer": tokenizer,
                    "input_token_id": input_token_id,
                    "generated_tokens": generated,
                    "decoded_generated_text": decoded_generated_text,
                    "steps": steps,
                    "max_tokens": max_tokens,
                    "provisional": true
                }),
            )
        }
        "greedy_prompt_session_probe" => {
            let session = match require_arg_string(&request, "session") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let prompt = match require_arg_string(&request, "prompt") {
                Ok(value) => value,
                Err(response) => return response,
            };

            if state.freed_native_sessions.contains(&session) {
                return already_freed(&request, "native session", &session);
            }
            let session_record = match state.native_sessions.get(&session) {
                Some(record) => record,
                None => return unknown_handle(&request, "native session", &session),
            };
            let model = match &session_record.model {
                Some(value) => value.clone(),
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        format!("native session {session} has no model handle"),
                    );
                }
            };
            let tokenizer = match &session_record.tokenizer {
                Some(value) => value.clone(),
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        format!("native session {session} has no tokenizer handle"),
                    );
                }
            };
            if state.freed_native_models.contains(&model) {
                return already_freed(&request, "native model", &model);
            }
            if state.freed_tokenizers.contains(&tokenizer) {
                return already_freed(&request, "tokenizer", &tokenizer);
            }
            let model_record = match state.native_models.get(&model) {
                Some(record) => record.clone(),
                None => return unknown_handle(&request, "native model", &model),
            };
            let tokenizer_record = match state.tokenizers.get(&tokenizer) {
                Some(record) => record.clone(),
                None => return unknown_handle(&request, "tokenizer", &tokenizer),
            };

            let prompt_tokens = match tokenizer_encode_fixture(&tokenizer_record, &prompt) {
                Ok(tokens) if !tokens.is_empty() => tokens,
                Ok(_) => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "greedy_prompt_session_probe prompt produced no tokens".to_string(),
                    );
                }
                Err(message) => {
                    return Response::err(request.id, "tokenizer_unknown_token", message);
                }
            };

            let mut generated_token = None;
            let mut generated_score = None;
            let mut total_timing_ms = 0.0;
            for token in &prompt_tokens {
                let result = cached_greedy_next_token_probe(state, &model_record.path, *token);
                if result.get("ok").and_then(|value| value.as_bool()) != Some(true) {
                    return Response::err(
                        request.id,
                        "bad_args",
                        format!("failed to run greedy prompt pass for token {token}"),
                    );
                }
                generated_token = result.get("next_token_id").and_then(|value| value.as_u64());
                generated_score = result
                    .get("next_token_score")
                    .and_then(|value| value.as_f64());
                total_timing_ms += result
                    .get("timing_ms")
                    .and_then(|value| value.as_f64())
                    .unwrap_or(0.0);
            }

            let generated_token = match generated_token {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "greedy prompt probe produced no generated token".to_string(),
                    );
                }
            };
            let generated_score = match generated_score {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "greedy prompt probe produced no generated score".to_string(),
                    );
                }
            };
            let decoded_generated_text =
                match tokenizer_decode_fixture(&tokenizer_record, &[generated_token]) {
                    Ok(value) => value,
                    Err(message) => {
                        return Response::err(request.id, "tokenizer_unknown_token", message);
                    }
                };

            Response::ok(
                request.id,
                json!({
                    "session": session,
                    "model": model,
                    "tokenizer": tokenizer,
                    "prompt": prompt,
                    "prompt_token_ids": prompt_tokens,
                    "generated_token_id": generated_token,
                    "generated_token_score": generated_score,
                    "decoded_generated_text": decoded_generated_text,
                    "timing_ms": total_timing_ms,
                    "provisional": true
                }),
            )
        }
        "incremental_session_probe" => {
            let session = match require_arg_string(&request, "session") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let prompt = match require_arg_string(&request, "prompt") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let max_tokens = match request.args.get("max_tokens") {
                Some(value) => match value.as_u64() {
                    Some(value) if value != 0 => value,
                    _ => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "incremental_session_probe requires max_tokens to be a positive integer"
                                .to_string(),
                        );
                    }
                },
                None => 3,
            };

            if state.freed_native_sessions.contains(&session) {
                return already_freed(&request, "native session", &session);
            }
            let session_record = match state.native_sessions.get(&session) {
                Some(record) => record.clone(),
                None => return unknown_handle(&request, "native session", &session),
            };
            let model = match session_record.model {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        format!("native session {session} has no model handle"),
                    );
                }
            };
            let tokenizer = match session_record.tokenizer {
                Some(value) => value,
                None => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        format!("native session {session} has no tokenizer handle"),
                    );
                }
            };
            if state.freed_native_models.contains(&model) {
                return already_freed(&request, "native model", &model);
            }
            if state.freed_tokenizers.contains(&tokenizer) {
                return already_freed(&request, "tokenizer", &tokenizer);
            }
            let model_record = match state.native_models.get(&model) {
                Some(record) => record.clone(),
                None => return unknown_handle(&request, "native model", &model),
            };
            let tokenizer_record = match state.tokenizers.get(&tokenizer) {
                Some(record) => record.clone(),
                None => return unknown_handle(&request, "tokenizer", &tokenizer),
            };
            let prompt_tokens = match tokenizer_encode_fixture(&tokenizer_record, &prompt) {
                Ok(tokens) if !tokens.is_empty() => tokens,
                Ok(_) => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "incremental_session_probe prompt produced no tokens".to_string(),
                    );
                }
                Err(message) => {
                    return Response::err(request.id, "tokenizer_unknown_token", message);
                }
            };

            let mut recompute_current = *prompt_tokens.last().unwrap_or(&0);
            let mut recompute_tokens = Vec::new();
            let mut recompute_checksums = Vec::new();
            let mut recompute_timing_total = 0.0;
            for _ in 0..max_tokens {
                let result =
                    cached_greedy_next_token_probe(state, &model_record.path, recompute_current);
                if result.get("ok").and_then(|value| value.as_bool()) != Some(true) {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "incremental_session_probe recompute path failed".to_string(),
                    );
                }
                let next = match result.get("next_token_id").and_then(|value| value.as_u64()) {
                    Some(value) => value,
                    None => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "incremental_session_probe recompute path returned no token"
                                .to_string(),
                        );
                    }
                };
                recompute_checksums.push(
                    result
                        .get("final_norm_checksum")
                        .and_then(|value| value.as_f64())
                        .unwrap_or(0.0),
                );
                recompute_timing_total += result
                    .get("timing_ms")
                    .and_then(|value| value.as_f64())
                    .unwrap_or(0.0);
                recompute_tokens.push(next);
                recompute_current = next;
            }
            let recompute_decoded =
                match tokenizer_decode_fixture(&tokenizer_record, &recompute_tokens) {
                    Ok(value) => value,
                    Err(message) => {
                        return Response::err(request.id, "tokenizer_unknown_token", message);
                    }
                };

            let kv = state.handles.next("kv");
            let layers = 36;
            state.kv_caches.insert(
                kv.clone(),
                KvCacheRecord {
                    session: Some(session.clone()),
                    layers,
                    sequence_len: prompt_tokens.len() as u64,
                    cached_keys: (0..layers)
                        .map(|layer| format!("layer:{layer}:keys"))
                        .collect(),
                    cached_values: (0..layers)
                        .map(|layer| format!("layer:{layer}:values"))
                        .collect(),
                },
            );
            if let Some(record) = state.native_sessions.get_mut(&session) {
                record.kv_cache = Some(kv.clone());
            }

            let mut incremental_current = *prompt_tokens.last().unwrap_or(&0);
            let mut incremental_tokens = Vec::new();
            let mut incremental_checksums = Vec::new();
            let mut incremental_steps = Vec::new();
            let mut incremental_timing_total = 0.0;
            for step_index in 0..max_tokens {
                let result =
                    cached_greedy_next_token_probe(state, &model_record.path, incremental_current);
                if result.get("ok").and_then(|value| value.as_bool()) != Some(true) {
                    return Response::err(
                        request.id,
                        "bad_args",
                        "incremental_session_probe incremental path failed".to_string(),
                    );
                }
                let next = match result.get("next_token_id").and_then(|value| value.as_u64()) {
                    Some(value) => value,
                    None => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "incremental_session_probe incremental path returned no token"
                                .to_string(),
                        );
                    }
                };
                let score = result
                    .get("next_token_score")
                    .and_then(|value| value.as_f64())
                    .unwrap_or(0.0);
                let checksum = result
                    .get("final_norm_checksum")
                    .and_then(|value| value.as_f64())
                    .unwrap_or(0.0);
                let timing_ms = result
                    .get("timing_ms")
                    .and_then(|value| value.as_f64())
                    .unwrap_or(0.0);
                incremental_tokens.push(next);
                incremental_checksums.push(checksum);
                incremental_timing_total += timing_ms;
                if let Some(cache) = state.kv_caches.get_mut(&kv) {
                    cache.sequence_len += 1;
                }
                let decoded = match tokenizer_decode_fixture(&tokenizer_record, &incremental_tokens)
                {
                    Ok(value) => value,
                    Err(message) => {
                        return Response::err(request.id, "tokenizer_unknown_token", message);
                    }
                };
                incremental_steps.push(json!({
                    "step": step_index,
                    "input_token": incremental_current,
                    "next_token": next,
                    "next_token_score": score,
                    "decoded_accumulated_text": decoded,
                    "timing_ms": timing_ms
                }));
                incremental_current = next;
            }
            let incremental_decoded =
                match tokenizer_decode_fixture(&tokenizer_record, &incremental_tokens) {
                    Ok(value) => value,
                    Err(message) => {
                        return Response::err(request.id, "tokenizer_unknown_token", message);
                    }
                };

            let identical_tokens = recompute_tokens == incremental_tokens;
            let identical_decoded = recompute_decoded == incremental_decoded;
            let mut max_checksum_diff = 0.0_f64;
            for (left, right) in recompute_checksums.iter().zip(incremental_checksums.iter()) {
                max_checksum_diff = max_checksum_diff.max((left - right).abs());
            }
            let logits_checksum_tolerance = 0.000001_f64;
            let checksums_match = max_checksum_diff <= logits_checksum_tolerance;
            let speedup_ratio = if incremental_timing_total > 0.0 {
                recompute_timing_total / incremental_timing_total
            } else {
                0.0
            };
            let final_sequence_len = state
                .kv_caches
                .get(&kv)
                .map(|cache| cache.sequence_len)
                .unwrap_or(0);

            state.kv_caches.remove(&kv);
            if let Some(record) = state.native_sessions.get_mut(&session) {
                if record.kv_cache.as_deref() == Some(&kv) {
                    record.kv_cache = None;
                }
            }

            Response::ok(
                request.id,
                json!({
                    "session": session,
                    "kv_cache": kv,
                    "prompt": prompt,
                    "prompt_token_ids": prompt_tokens,
                    "recompute_generated_tokens": recompute_tokens,
                    "incremental_generated_tokens": incremental_tokens,
                    "decoded_recompute_text": recompute_decoded,
                    "decoded_incremental_text": incremental_decoded,
                    "identical_generated_token_ids": identical_tokens,
                    "identical_decoded_text": identical_decoded,
                    "logits_checksum_tolerance": logits_checksum_tolerance,
                    "max_logits_checksum_diff": max_checksum_diff,
                    "logits_checksums_match": checksums_match,
                    "incremental_steps": incremental_steps,
                    "per_token_incremental_timing_ms": incremental_steps
                        .iter()
                        .map(|step| step.get("timing_ms").cloned().unwrap_or(Value::Null))
                        .collect::<Vec<Value>>(),
                    "total_incremental_timing_ms": incremental_timing_total,
                    "total_recompute_timing_ms": recompute_timing_total,
                    "speedup_ratio": speedup_ratio,
                    "layers": layers,
                    "kv_sequence_len": final_sequence_len,
                    "fallback_stage": "cpu_provisional_kv_session_metadata",
                    "provisional": true
                }),
            )
        }
        "dequantize_group_slice" => {
            let handle = match require_arg_string(&request, "group") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let row = if let Some(value) = request.args.get("row") {
                match value.as_u64() {
                    Some(value) => value,
                    None => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "dequantize_group_slice requires row to be an integer".to_string(),
                        );
                    }
                }
            } else if let Some(value) = request.args.get("rows") {
                match value.as_u64() {
                    Some(value) => value,
                    None => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "dequantize_group_slice requires rows to be an integer".to_string(),
                        );
                    }
                }
            } else {
                0
            };
            let cols = match request.args.get("cols") {
                Some(value) => match value.as_u64() {
                    Some(value) if value > 0 => value,
                    Some(_) => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "dequantize_group_slice requires cols to be a positive integer"
                                .to_string(),
                        );
                    }
                    None => {
                        return Response::err(
                            request.id,
                            "bad_args",
                            "dequantize_group_slice requires cols to be an integer".to_string(),
                        );
                    }
                },
                None => 8,
            };
            if state.freed_tensor_groups.contains(&handle) {
                return already_freed(&request, "tensor group", &handle);
            }
            if !state.tensor_groups.contains_key(&handle) {
                return unknown_handle(&request, "tensor group", &handle);
            }
            let result = shim::dequantize_group_slice(&handle, row, cols);
            if result.get("created").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "bad_args",
                    format!("failed to dequantize slice from tensor group {handle}"),
                )
            }
        }
        "free_tensor_group" => {
            let handle = match require_arg_string(&request, "group") {
                Ok(value) => value,
                Err(response) => return response,
            };
            if state.freed_tensor_groups.contains(&handle) {
                return already_freed(&request, "tensor group", &handle);
            }
            if !state.tensor_groups.contains_key(&handle) {
                return unknown_handle(&request, "tensor group", &handle);
            }
            let result = shim::free_tensor_group(&handle);
            if result.get("freed").and_then(|value| value.as_bool()) == Some(true) {
                state.tensor_groups.remove(&handle);
                state.freed_tensor_groups.insert(handle.clone());
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "unknown_handle",
                    format!("tensor group handle not found: {handle}"),
                )
            }
        }
        "load_embedding_group" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };

            let handle = state.handles.next("emb");
            let loaded = shim::load_embedding_group(&handle, &model_dir);
            if loaded.get("loaded").and_then(|value| value.as_bool()) != Some(true) {
                return Response::err(
                    request.id,
                    "bad_args",
                    format!("failed to load embedding group from {model_dir}"),
                );
            }

            let info = shim::embedding_group_info(&handle);
            if info.get("ok").and_then(|value| value.as_bool()) != Some(true) {
                let _ = shim::free_embedding_group(&handle);
                return Response::err(
                    request.id,
                    "unknown_handle",
                    format!("failed to inspect embedding group for handle {handle}"),
                );
            }

            if info
                .get("quantized_group")
                .and_then(|value| value.as_bool())
                != Some(true)
            {
                let _ = shim::free_embedding_group(&handle);
                return Response::err(
                    request.id,
                    "bad_args",
                    "embedding group must be quantized".to_string(),
                );
            }

            let weight_value = info.get("weight").cloned().unwrap_or(Value::Null);
            let scales_value = info.get("scales").cloned().unwrap_or(Value::Null);
            let biases_value = info.get("biases").cloned().unwrap_or(Value::Null);
            let weight = weight_value.as_object();
            let scales = scales_value.as_object();
            let biases = biases_value.as_object();
            if weight.is_none() || scales.is_none() || biases.is_none() {
                let _ = shim::free_embedding_group(&handle);
                return Response::err(
                    request.id,
                    "bad_args",
                    "embedding group is missing weight/scales/biases metadata".to_string(),
                );
            }

            let weight_dtype = weight
                .and_then(|value| value.get("dtype"))
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            let scales_dtype = scales
                .and_then(|value| value.get("dtype"))
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            let biases_dtype = biases
                .and_then(|value| value.get("dtype"))
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            if weight_dtype != "U32" || scales_dtype != "BF16" || biases_dtype != "BF16" {
                let _ = shim::free_embedding_group(&handle);
                return Response::err(
                    request.id,
                    "bad_args",
                    format!(
                        "embedding group dtype mismatch: weight={weight_dtype}, scales={scales_dtype}, biases={biases_dtype}",
                    ),
                );
            }

            let byte_size = info
                .get("byte_size")
                .and_then(|value| value.as_u64())
                .unwrap_or(0);
            if byte_size == 0 {
                let _ = shim::free_embedding_group(&handle);
                return Response::err(
                    request.id,
                    "bad_args",
                    "embedding group byte_size must be greater than zero".to_string(),
                );
            }

            state.embedding_groups.insert(
                handle.clone(),
                EmbeddingGroupRecord {
                    model_dir: model_dir.clone(),
                    group: "model.embed_tokens".to_string(),
                },
            );

            Response::ok(
                request.id,
                json!({
                    "embedding_handle": handle,
                    "quantized_group": true,
                    "weight": weight_value,
                    "scales": scales_value,
                    "biases": biases_value,
                    "byte_size": byte_size,
                    "model_dir": model_dir,
                }),
            )
        }
        "embedding_group_info" => {
            let handle = match require_arg_string(&request, "embedding") {
                Ok(value) => value,
                Err(response) => return response,
            };
            if state.freed_embedding_groups.contains(&handle) {
                return already_freed(&request, "embedding group", &handle);
            }
            if !state.embedding_groups.contains_key(&handle) {
                return unknown_handle(&request, "embedding group", &handle);
            }
            let result = shim::embedding_group_info(&handle);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "unknown_handle",
                    format!("embedding group handle not found: {handle}"),
                )
            }
        }
        "free_embedding_group" => {
            let handle = match require_arg_string(&request, "embedding") {
                Ok(value) => value,
                Err(response) => return response,
            };
            if state.freed_embedding_groups.contains(&handle) {
                return already_freed(&request, "embedding group", &handle);
            }
            if !state.embedding_groups.contains_key(&handle) {
                return unknown_handle(&request, "embedding group", &handle);
            }
            let result = shim::free_embedding_group(&handle);
            if result.get("freed").and_then(|value| value.as_bool()) == Some(true) {
                state.embedding_groups.remove(&handle);
                state.freed_embedding_groups.insert(handle.clone());
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "unknown_handle",
                    format!("embedding group handle not found: {handle}"),
                )
            }
        }
        "load_layer_groups" => {
            let model_dir = match require_arg_string(&request, "model_dir") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let layer = match require_arg_u64_allow_zero(&request, "layer") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let layer_i32 = match i32::try_from(layer) {
                Ok(value) => value,
                Err(_) => {
                    return Response::err(
                        request.id,
                        "bad_args",
                        format!("load_layer_groups requires args.layer to fit in i32: {layer}"),
                    );
                }
            };

            let handle = state.handles.next("layer");
            let loaded = shim::load_layer_groups(&handle, &model_dir, layer_i32);
            if loaded.get("loaded").and_then(|value| value.as_bool()) != Some(true) {
                return Response::err(
                    request.id,
                    "bad_args",
                    format!("failed to load layer groups for layer {layer} from {model_dir}"),
                );
            }

            let info = shim::layer_groups_info(&handle);
            if info.get("ok").and_then(|value| value.as_bool()) != Some(true) {
                let _ = shim::free_layer_groups(&handle);
                return Response::err(
                    request.id,
                    "unknown_handle",
                    format!("failed to inspect layer groups for handle {handle}"),
                );
            }

            let expected_specs = layer_group_specs(layer);
            let expected_count = expected_specs.len() as u64;
            let reported_total_groups = info
                .get("summary")
                .and_then(|value| value.get("total_groups"))
                .and_then(|value| value.as_u64())
                .unwrap_or(0);
            if reported_total_groups != expected_count {
                let _ = shim::free_layer_groups(&handle);
                return Response::err(
                    request.id,
                    "bad_args",
                    format!(
                        "layer group summary mismatch for layer {layer}: expected {expected_count}, got {reported_total_groups}",
                    ),
                );
            }

            let mut group_handles = std::collections::HashMap::new();
            if let Some(groups) = info.get("groups").and_then(|value| value.as_object()) {
                for (group_name, group_info) in groups {
                    if let Some(child_handle) =
                        group_info.get("handle").and_then(|value| value.as_str())
                    {
                        group_handles.insert(group_name.clone(), child_handle.to_string());
                        state.tensor_groups.insert(
                            child_handle.to_string(),
                            TensorGroupRecord {
                                model_dir: model_dir.clone(),
                                group: group_info
                                    .get("group")
                                    .and_then(|value| value.as_str())
                                    .unwrap_or_default()
                                    .to_string(),
                            },
                        );
                    }
                }
            }
            if group_handles.len() != expected_count as usize {
                let _ = shim::free_layer_groups(&handle);
                return Response::err(
                    request.id,
                    "bad_args",
                    format!(
                        "layer group handle count mismatch for layer {layer}: expected {expected_count}, got {}",
                        group_handles.len()
                    ),
                );
            }

            let total_byte_size = info
                .get("total_byte_size")
                .and_then(|value| value.as_u64())
                .unwrap_or(0);
            let calculated_total_byte_size: u64 = info
                .get("groups")
                .and_then(|value| value.as_object())
                .map(|groups| groups.values().map(layer_group_byte_size).sum())
                .unwrap_or(0);
            if calculated_total_byte_size != total_byte_size {
                let _ = shim::free_layer_groups(&handle);
                return Response::err(
                    request.id,
                    "bad_args",
                    format!(
                        "layer byte size mismatch for layer {layer}: expected {total_byte_size}, got {calculated_total_byte_size}",
                    ),
                );
            }

            let quantized_group_count = info
                .get("summary")
                .and_then(|value| value.get("quantized_groups"))
                .and_then(|value| value.as_u64())
                .unwrap_or(0);
            let norm_group_count = info
                .get("summary")
                .and_then(|value| value.get("norm_groups"))
                .and_then(|value| value.as_u64())
                .unwrap_or(0);

            state.layer_groups.insert(
                handle.clone(),
                LayerGroupsRecord {
                    model_dir: model_dir.clone(),
                    layer,
                    group_handles: group_handles.clone(),
                    total_byte_size,
                    quantized_group_count,
                    norm_group_count,
                },
            );

            Response::ok(request.id, info)
        }
        "layer_groups_info" => {
            let handle = match require_arg_string(&request, "layer") {
                Ok(value) => value,
                Err(response) => return response,
            };
            if state.freed_layer_groups.contains(&handle) {
                return already_freed(&request, "layer groups", &handle);
            }
            if !state.layer_groups.contains_key(&handle) {
                return unknown_handle(&request, "layer groups", &handle);
            }
            let result = shim::layer_groups_info(&handle);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "unknown_handle",
                    format!("layer groups handle not found: {handle}"),
                )
            }
        }
        "free_layer_groups" => {
            let handle = match require_arg_string(&request, "layer") {
                Ok(value) => value,
                Err(response) => return response,
            };
            if state.freed_layer_groups.contains(&handle) {
                return already_freed(&request, "layer groups", &handle);
            }
            let record = match state.layer_groups.get(&handle) {
                Some(record) => record.clone(),
                None => return unknown_handle(&request, "layer groups", &handle),
            };

            let result = shim::free_layer_groups(&handle);
            if result.get("freed").and_then(|value| value.as_bool()) == Some(true) {
                for child_handle in record.group_handles.values() {
                    state.tensor_groups.remove(child_handle);
                    state.freed_tensor_groups.insert(child_handle.clone());
                }
                state.layer_groups.remove(&handle);
                state.freed_layer_groups.insert(handle.clone());
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "unknown_handle",
                    format!("layer groups handle not found: {handle}"),
                )
            }
        }
        "shim_probe" => Response::ok(request.id, shim::run_probe()),
        "mlx_link_probe" => Response::ok(request.id, shim::run_mlx_link_probe()),
        "mlx_runtime_diagnose" => Response::ok(request.id, shim::runtime_diagnose()),
        "mlx_create_test_array" => Response::ok(request.id, shim::create_test_array()),
        "mlx_test_array_sum" => {
            let handle = match require_arg_u64(&request, "handle") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let result = shim::test_array_sum(handle);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "unknown_handle",
                    format!("mlx test array handle not found: {handle}"),
                )
            }
        }
        "mlx_free_test_array" => {
            let handle = match require_arg_u64(&request, "handle") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let result = shim::free_test_array(handle);
            if result.get("freed").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "unknown_handle",
                    format!("mlx test array handle not found: {handle}"),
                )
            }
        }
        "bridge_shutdown" => {
            state.shutdown_requested = true;
            Response::ok(request.id, json!({ "status": "shutting_down" }))
        }
        "load_tokenizer" => {
            let path = match require_arg_string(&request, "path") {
                Ok(value) => value,
                Err(response) => return response,
            };

            let path_buf = Path::new(&path);
            let detected_files = match tokenizer_detected_files(path_buf) {
                Ok(files) => files,
                Err(message) => {
                    return Response::err(request.id.clone(), "bad_args", message);
                }
            };
            let tokenizer_kind = tokenizer_kind_from_files(&detected_files);
            let vocab = if tokenizer_kind == "huggingface_json" {
                match tokenizer_json_vocab(path_buf) {
                    Ok(map) => map,
                    Err(message) => {
                        return Response::err(request.id.clone(), "bad_args", message);
                    }
                }
            } else {
                std::collections::HashMap::new()
            };
            let (
                model_type,
                normalizer_type,
                pre_tokenizer_type,
                decoder_type,
                vocab_size,
                merges_count,
                added_tokens_count,
            ) = match tokenizer_kind.as_str() {
                "huggingface_json" => match tokenizer_json_metadata(path_buf) {
                    Ok(values) => values,
                    Err(message) => {
                        return Response::err(request.id.clone(), "bad_args", message);
                    }
                },
                _ => (None, None, None, None, None, None, None),
            };

            let handle = state.handles.next("tok");
            state.tokenizers.insert(
                handle.clone(),
                TokenizerRecord {
                    path: path.clone(),
                    detected_files: detected_files.clone(),
                    tokenizer_kind: tokenizer_kind.clone(),
                    vocab: vocab.clone(),
                    model_type: model_type.clone(),
                    normalizer_type: normalizer_type.clone(),
                    pre_tokenizer_type: pre_tokenizer_type.clone(),
                    decoder_type: decoder_type.clone(),
                    vocab_size,
                    merges_count,
                    added_tokens_count,
                    model: None,
                },
            );

            Response::ok(
                request.id,
                json!({
                    "tokenizer": handle,
                    "path": path,
                    "detected_files": detected_files,
                    "tokenizer_kind": tokenizer_kind,
                    "vocab": vocab,
                    "model_type": model_type,
                    "normalizer_type": normalizer_type,
                    "pre_tokenizer_type": pre_tokenizer_type,
                    "decoder_type": decoder_type,
                    "vocab_size": vocab_size,
                    "merges_count": merges_count,
                    "added_tokens_count": added_tokens_count,
                    "stub": true
                }),
            )
        }
        "unload_tokenizer" => {
            let handle = match require_arg_string(&request, "tokenizer") {
                Ok(value) => value,
                Err(response) => return response,
            };

            if state.freed_tokenizers.contains(&handle) {
                return already_freed(&request, "tokenizer", &handle);
            }
            if !state.tokenizers.contains_key(&handle) {
                return unknown_handle(&request, "tokenizer", &handle);
            }

            state.tokenizers.remove(&handle);
            state.freed_tokenizers.insert(handle.clone());
            Response::ok(
                request.id,
                json!({
                    "tokenizer": handle,
                    "removed": true
                }),
            )
        }
        "tokenizer_info" => {
            let handle = match require_arg_string(&request, "tokenizer") {
                Ok(value) => value,
                Err(response) => return response,
            };

            if state.freed_tokenizers.contains(&handle) {
                return already_freed(&request, "tokenizer", &handle);
            }
            let record = match state.tokenizers.get(&handle) {
                Some(record) => record,
                None => return unknown_handle(&request, "tokenizer", &handle),
            };

            Response::ok(
                request.id,
                json!({
                    "tokenizer": handle,
                    "path": record.path,
                    "tokenizer_kind": record.tokenizer_kind,
                    "detected_files": record.detected_files,
                    "model_type": record.model_type,
                    "normalizer_type": record.normalizer_type,
                    "pre_tokenizer_type": record.pre_tokenizer_type,
                    "decoder_type": record.decoder_type,
                    "vocab_size": record.vocab_size,
                    "merges_count": record.merges_count,
                    "added_tokens_count": record.added_tokens_count,
                    "loaded": false
                }),
            )
        }
        "tokenizer_encode" => {
            let handle = match require_arg_string(&request, "tokenizer") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let text = match require_arg_string(&request, "text") {
                Ok(value) => value,
                Err(response) => return response,
            };

            if state.freed_tokenizers.contains(&handle) {
                return already_freed(&request, "tokenizer", &handle);
            }
            let record = match state.tokenizers.get(&handle) {
                Some(record) => record,
                None => return unknown_handle(&request, "tokenizer", &handle),
            };
            if record.tokenizer_kind != "huggingface_json" {
                return Response::err(
                    request.id.clone(),
                    "bad_args",
                    format!("tokenizer_encode only supports huggingface_json tokenizers: {handle}"),
                );
            }

            let tokens = match tokenizer_encode_fixture(record, &text) {
                Ok(tokens) => tokens,
                Err(message) => {
                    return Response::err(request.id.clone(), "tokenizer_unknown_token", message);
                }
            };

            Response::ok(
                request.id,
                json!({
                    "tokenizer": handle,
                    "tokens": tokens,
                    "fixture_only": true
                }),
            )
        }
        "tokenizer_decode" => {
            let handle = match require_arg_string(&request, "tokenizer") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let tokens = match request.args.get("tokens") {
                Some(value) => match value.as_array() {
                    Some(tokens) => tokens,
                    None => {
                        return Response::err(
                            request.id.clone(),
                            "bad_args",
                            format!(
                                "{} requires args.tokens as an array of integers",
                                request.cmd
                            ),
                        );
                    }
                },
                None => {
                    return Response::err(
                        request.id.clone(),
                        "bad_args",
                        format!(
                            "{} requires args.tokens as an array of integers",
                            request.cmd
                        ),
                    );
                }
            };

            if state.freed_tokenizers.contains(&handle) {
                return already_freed(&request, "tokenizer", &handle);
            }
            let record = match state.tokenizers.get(&handle) {
                Some(record) => record,
                None => return unknown_handle(&request, "tokenizer", &handle),
            };
            if record.tokenizer_kind != "huggingface_json" {
                return Response::err(
                    request.id.clone(),
                    "bad_args",
                    format!("tokenizer_decode only supports huggingface_json tokenizers: {handle}"),
                );
            }
            let mut parsed = Vec::new();
            for token in tokens {
                let id = match token.as_u64() {
                    Some(value) => value,
                    None => {
                        return Response::err(
                            request.id.clone(),
                            "bad_args",
                            format!(
                                "{} requires args.tokens to contain only integers",
                                request.cmd
                            ),
                        );
                    }
                };
                parsed.push(id);
            }
            let skip_special_tokens = request
                .args
                .get("skip_special_tokens")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            let include_diagnostics = request
                .args
                .get("diagnostics")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            let decoded = match tokenizer_decode_fixture_with_options(record, &parsed, skip_special_tokens) {
                Ok(output) => output,
                Err(message) => {
                    return Response::err(request.id.clone(), "tokenizer_unknown_token", message);
                }
            };
            let diagnostics = decoded
                .diagnostics
                .iter()
                .map(|entry| {
                    json!({
                        "token_id": entry.token_id,
                        "raw_token": entry.raw_token,
                        "byte_values": entry.byte_values,
                        "byte_decoded_form": entry.decoded_piece,
                        "special": entry.special,
                        "skipped": entry.skipped,
                    })
                })
                .collect::<Vec<Value>>();

            Response::ok(
                request.id,
                json!({
                    "tokenizer": handle,
                    "text": decoded.text,
                    "skip_special_tokens": skip_special_tokens,
                    "diagnostics": if include_diagnostics { json!(diagnostics) } else { Value::Null },
                    "fixture_only": true
                }),
            )
        }
        "mlx_create_token_array" => {
            let tokens = match request.args.get("tokens") {
                Some(value) => match value.as_array() {
                    Some(tokens) => tokens,
                    None => {
                        return Response::err(
                            request.id.clone(),
                            "bad_args",
                            format!(
                                "{} requires args.tokens as an array of integers",
                                request.cmd
                            ),
                        );
                    }
                },
                None => {
                    return Response::err(
                        request.id.clone(),
                        "bad_args",
                        format!(
                            "{} requires args.tokens as an array of integers",
                            request.cmd
                        ),
                    );
                }
            };

            let mut parsed = Vec::new();
            for token in tokens {
                let value = match token.as_u64() {
                    Some(value) => value,
                    None => {
                        return Response::err(
                            request.id.clone(),
                            "bad_args",
                            format!(
                                "{} requires args.tokens to contain only integers",
                                request.cmd
                            ),
                        );
                    }
                };
                parsed.push(value);
            }

            let result = shim::create_token_array(&parsed);
            if result.get("created").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "unknown_handle",
                    "token array creation failed or MLX link is unavailable".to_string(),
                )
            }
        }
        "mlx_token_array_info" => {
            let handle = match require_arg_u64(&request, "array") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let result = shim::token_array_info(handle);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "unknown_handle",
                    format!("token array handle not found: {handle}"),
                )
            }
        }
        "mlx_free_token_array" => {
            let handle = match require_arg_u64(&request, "array") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let result = shim::free_token_array(handle);
            if result.get("freed").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "unknown_handle",
                    format!("token array handle not found: {handle}"),
                )
            }
        }
        "native_mock_forward" => {
            let session = match require_arg_string(&request, "session") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let tokens = match require_arg_string(&request, "tokens") {
                Ok(value) => value,
                Err(response) => return response,
            };
            if state.freed_native_sessions.contains(&session) {
                return already_freed(&request, "native session", &session);
            } else if !state.native_sessions.contains_key(&session) {
                return unknown_handle(&request, "native session", &session);
            }
            let token_handle = match tokens.parse::<u64>() {
                Ok(parsed) => parsed,
                Err(_) => {
                    return Response::err(
                        request.id.clone(),
                        "bad_args",
                        format!(
                            "{} requires args.tokens as a token array handle stringified integer",
                            request.cmd
                        ),
                    );
                }
            };
            let token_handle_info = shim::token_array_info(token_handle);
            if token_handle_info
                .get("ok")
                .and_then(|value| value.as_bool())
                != Some(true)
            {
                return unknown_handle(&request, "token array", &tokens);
            }

            let result = shim::mock_forward(1, token_handle);
            if result.get("created").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "unknown_handle",
                    "mock forward failed or native handles were unavailable".to_string(),
                )
            }
        }
        "mlx_array_info" => {
            let handle = match require_arg_u64(&request, "array") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let result = shim::array_info(handle);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "unknown_handle",
                    format!("array handle not found: {handle}"),
                )
            }
        }
        "native_mock_sample" => {
            let logits = match require_arg_string(&request, "logits") {
                Ok(value) => value,
                Err(response) => return response,
            };

            let logits_handle = match logits.parse::<u64>() {
                Ok(parsed) => parsed,
                Err(_) => {
                    return Response::err(
                        request.id.clone(),
                        "bad_args",
                        format!(
                            "{} requires args.logits as a logits handle stringified integer",
                            request.cmd
                        ),
                    );
                }
            };

            let logits_info = shim::array_info(logits_handle);
            if logits_info.get("ok").and_then(|value| value.as_bool()) != Some(true) {
                return unknown_handle(&request, "logits", &logits);
            }

            let result = shim::mock_sample(logits_handle);
            if result.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(
                    request.id,
                    result.get("token").cloned().unwrap_or(Value::Null),
                )
            } else {
                Response::err(
                    request.id,
                    "unknown_handle",
                    format!("logits handle not found: {logits_handle}"),
                )
            }
        }
        "native_mock_generate" => {
            let session = match require_arg_string(&request, "session") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let tokenizer = match require_arg_string(&request, "tokenizer") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let prompt = match require_arg_string(&request, "prompt") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let max_tokens = match request.args.get("max_tokens") {
                Some(value) => match value.as_u64() {
                    Some(value) => value,
                    None => {
                        return Response::err(
                            request.id.clone(),
                            "bad_args",
                            format!(
                                "{} requires args.max_tokens as a non-negative integer",
                                request.cmd
                            ),
                        );
                    }
                },
                None => 1,
            };

            if state.freed_native_sessions.contains(&session) {
                return already_freed(&request, "native session", &session);
            }
            let session_record = match state.native_sessions.get(&session) {
                Some(record) => record,
                None => return unknown_handle(&request, "native session", &session),
            };
            if session_record.tokenizer.as_deref() != Some(&tokenizer) {
                return Response::err(
                    request.id.clone(),
                    "bad_args",
                    format!(
                        "native session {session} is not associated with tokenizer {tokenizer}"
                    ),
                );
            }
            if state.freed_tokenizers.contains(&tokenizer) {
                return already_freed(&request, "tokenizer", &tokenizer);
            }
            let tokenizer_record = match state.tokenizers.get(&tokenizer) {
                Some(record) => record,
                None => return unknown_handle(&request, "tokenizer", &tokenizer),
            };
            if tokenizer_record.tokenizer_kind != "huggingface_json" {
                return Response::err(
                    request.id.clone(),
                    "bad_args",
                    format!("native_mock_generate only supports huggingface_json tokenizers: {tokenizer}"),
                );
            }

            let mut current_tokens = match tokenizer_encode_fixture(tokenizer_record, &prompt) {
                Ok(tokens) => tokens,
                Err(message) => {
                    return Response::err(request.id.clone(), "tokenizer_unknown_token", message);
                }
            };
            let mut generated_tokens = Vec::new();
            let mut generated_text = Vec::new();

            for _ in 0..max_tokens {
                let token_array = shim::create_token_array(&current_tokens);
                if token_array.get("created").and_then(|value| value.as_bool()) != Some(true) {
                    return Response::err(
                        request.id.clone(),
                        "unknown_handle",
                        "mock generation failed because the native token array could not be created".to_string(),
                    );
                }
                let token_array_handle = match token_array
                    .get("handle")
                    .and_then(|value| value.as_u64())
                {
                    Some(value) => value,
                    None => {
                        return Response::err(
                            request.id.clone(),
                            "unknown_handle",
                            "mock generation failed because the native token array handle was missing".to_string(),
                        );
                    }
                };

                let logits = shim::mock_forward(1, token_array_handle);
                if logits.get("created").and_then(|value| value.as_bool()) != Some(true) {
                    let _ = shim::free_token_array(token_array_handle);
                    return Response::err(
                        request.id.clone(),
                        "unknown_handle",
                        "mock generation failed because the native logits array could not be created".to_string(),
                    );
                }
                let logits_handle = match logits.get("handle").and_then(|value| value.as_u64()) {
                    Some(value) => value,
                    None => {
                        let _ = shim::free_token_array(token_array_handle);
                        return Response::err(
                            request.id.clone(),
                            "unknown_handle",
                            "mock generation failed because the native logits handle was missing"
                                .to_string(),
                        );
                    }
                };

                let sampled = shim::mock_sample(logits_handle);
                if sampled.get("ok").and_then(|value| value.as_bool()) != Some(true) {
                    let _ = shim::free_array(logits_handle);
                    let _ = shim::free_token_array(token_array_handle);
                    return Response::err(
                        request.id.clone(),
                        "unknown_handle",
                        "mock generation failed because sampling was unavailable".to_string(),
                    );
                }
                let sampled_token = match sampled.get("token").and_then(|value| value.as_u64()) {
                    Some(value) => value,
                    None => {
                        let _ = shim::free_array(logits_handle);
                        let _ = shim::free_token_array(token_array_handle);
                        return Response::err(
                            request.id.clone(),
                            "unknown_handle",
                            "mock generation failed because the sampled token was missing"
                                .to_string(),
                        );
                    }
                };

                let piece = match tokenizer_decode_fixture(tokenizer_record, &[sampled_token]) {
                    Ok(text) => text,
                    Err(message) => {
                        let _ = shim::free_array(logits_handle);
                        let _ = shim::free_token_array(token_array_handle);
                        return Response::err(
                            request.id.clone(),
                            "tokenizer_unknown_token",
                            message,
                        );
                    }
                };

                generated_tokens.push(sampled_token);
                generated_text.push(piece);
                current_tokens.push(sampled_token);

                let _ = shim::free_array(logits_handle);
                let _ = shim::free_token_array(token_array_handle);
            }

            Response::ok(
                request.id,
                json!({
                    "text": generated_text.join(" "),
                    "tokens": generated_tokens,
                    "steps": generated_tokens.len(),
                    "fixture_only": true
                }),
            )
        }
        "mlx_free_array" => {
            let handle = match require_arg_u64(&request, "array") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let result = shim::free_array(handle);
            if result.get("freed").and_then(|value| value.as_bool()) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "unknown_handle",
                    format!("array handle not found: {handle}"),
                )
            }
        }
        "mlx_handle_counts" => Response::ok(request.id, shim::handle_counts()),
        "load_model_native" => {
            let path = match require_arg_string(&request, "path") {
                Ok(value) => value,
                Err(response) => return response,
            };

            let handle = state.handles.next("nmodel");
            state
                .native_models
                .insert(handle.clone(), ModelRecord { path: path.clone() });

            Response::ok(
                request.id,
                json!({
                    "model": handle,
                    "path": path,
                    "native": true
                }),
            )
        }
        "unload_model_native" => {
            let handle = match require_arg_string(&request, "model") {
                Ok(value) => value,
                Err(response) => return response,
            };

            if state.freed_native_models.contains(&handle) {
                return already_freed(&request, "native model", &handle);
            }
            if !state.native_models.contains_key(&handle) {
                return unknown_handle(&request, "native model", &handle);
            }

            state.native_models.remove(&handle);
            state.freed_native_models.insert(handle.clone());
            Response::ok(
                request.id,
                json!({
                    "model": handle,
                    "removed": true
                }),
            )
        }
        "create_native_session" => {
            let model = match require_arg_string(&request, "model") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let tokenizer = arg_string(&request.args, "tokenizer");

            if state.freed_native_models.contains(&model) {
                return already_freed(&request, "native model", &model);
            }
            if !state.native_models.contains_key(&model) {
                return unknown_handle(&request, "native model", &model);
            }
            if let Some(tokenizer_handle) = tokenizer.as_ref() {
                if state.freed_tokenizers.contains(tokenizer_handle) {
                    return already_freed(&request, "tokenizer", tokenizer_handle);
                }
                if !state.tokenizers.contains_key(tokenizer_handle) {
                    return unknown_handle(&request, "tokenizer", tokenizer_handle);
                }
            }

            let session = state.handles.next("nsess");
            state.native_sessions.insert(
                session.clone(),
                SessionRecord {
                    model: Some(model.clone()),
                    tokenizer: tokenizer.clone(),
                    kv_cache: None,
                },
            );

            Response::ok(
                request.id,
                json!({
                    "session": session,
                    "model": model,
                    "tokenizer": tokenizer,
                    "native": true
                }),
            )
        }
        "warm_resident_session" => {
            let session = match require_arg_string(&request, "session") {
                Ok(value) => value,
                Err(response) => return response,
            };
            if state.freed_native_sessions.contains(&session) {
                return already_freed(&request, "native session", &session);
            }
            let session_record = match state.native_sessions.get(&session) {
                Some(record) => record,
                None => return unknown_handle(&request, "native session", &session),
            };
            let model = match session_record.model.as_ref() {
                Some(model) => model,
                None => {
                    return Response::err(
                        request.id,
                        "bad_state",
                        "native session has no model".to_string(),
                    )
                }
            };
            let model_record = match state.native_models.get(model) {
                Some(record) => record,
                None => return unknown_handle(&request, "native model", model),
            };
            let result = shim::warm_resident_session(&session, &model_record.path);
            if result.get("ok").and_then(Value::as_bool) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(
                    request.id,
                    "warm_resident_session_failed",
                    result.to_string(),
                )
            }
        }
        "generate_tokens" => {
            let session = match require_arg_string(&request, "session") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let prompt_token_ids = request
                .args
                .get("prompt_token_ids")
                .and_then(Value::as_array)
                .map(|values| values.iter().filter_map(Value::as_u64).collect::<Vec<_>>())
                .filter(|values| !values.is_empty())
                .unwrap_or_else(|| vec![1]);
            let first_decode_token_id = request
                .args
                .get("first_decode_token_id")
                .and_then(Value::as_u64)
                .unwrap_or(15);
            let generated_tokens = request
                .args
                .get("generated_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(3);
            let temperature = request
                .args
                .get("temperature")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let top_k = request
                .args
                .get("top_k")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let top_p = request
                .args
                .get("top_p")
                .and_then(Value::as_f64)
                .unwrap_or(1.0);
            let seed = request
                .args
                .get("seed")
                .and_then(Value::as_u64)
                .unwrap_or(1234);
            let stop_on_eos = request
                .args
                .get("stop_on_eos")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let eos_token_id = request
                .args
                .get("eos_token_id")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let stop_token_ids = request
                .args
                .get("stop_token_ids")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_u64)
                        .collect::<Vec<u64>>()
                })
                .unwrap_or_default();
            if state.freed_native_sessions.contains(&session) {
                return already_freed(&request, "native session", &session);
            }
            let session_record = match state.native_sessions.get(&session) {
                Some(record) => record,
                None => return unknown_handle(&request, "native session", &session),
            };
            let model = match session_record.model.as_ref() {
                Some(model) => model,
                None => {
                    return Response::err(
                        request.id,
                        "bad_state",
                        "native session has no model".to_string(),
                    )
                }
            };
            let model_record = match state.native_models.get(model) {
                Some(record) => record,
                None => return unknown_handle(&request, "native model", model),
            };
            let result = shim::generate_tokens_for_session(
                &session,
                &model_record.path,
                &prompt_token_ids,
                first_decode_token_id,
                generated_tokens,
                temperature,
                top_k,
                top_p,
                seed,
                stop_on_eos,
                eos_token_id,
                &stop_token_ids,
            );
            if result.get("ok").and_then(Value::as_bool) == Some(true) {
                Response::ok(request.id, result)
            } else {
                Response::err(request.id, "generate_tokens_failed", result.to_string())
            }
        }
        "free_native_session" => {
            let session = match require_arg_string(&request, "session") {
                Ok(value) => value,
                Err(response) => return response,
            };

            if state.freed_native_sessions.contains(&session) {
                return already_freed(&request, "native session", &session);
            }
            if !state.native_sessions.contains_key(&session) {
                return unknown_handle(&request, "native session", &session);
            }

            shim::free_resident_session(&session);
            state.native_sessions.remove(&session);
            state.freed_native_sessions.insert(session.clone());
            Response::ok(
                request.id,
                json!({
                    "session": session,
                    "removed": true
                }),
            )
        }
        "load_model" => {
            let path = match require_arg_string(&request, "path") {
                Ok(value) => value,
                Err(response) => return response,
            };

            let handle = state.handles.next("model");
            state
                .models
                .insert(handle.clone(), ModelRecord { path: path.clone() });

            Response::ok(
                request.id,
                json!({
                    "model": handle,
                    "path": path,
                    "stub": true
                }),
            )
        }
        "unload_model" => {
            let handle = match require_arg_string(&request, "model") {
                Ok(value) => value,
                Err(response) => return response,
            };

            if state.freed_models.contains(&handle) {
                return already_freed(&request, "model", &handle);
            }
            if !state.models.contains_key(&handle) {
                return unknown_handle(&request, "model", &handle);
            }

            state.models.remove(&handle);
            state.freed_models.insert(handle.clone());
            Response::ok(
                request.id,
                json!({
                    "model": handle,
                    "removed": true
                }),
            )
        }
        "create_session" => {
            let model = match require_arg_string(&request, "model") {
                Ok(value) => value,
                Err(response) => return response,
            };
            if state.freed_models.contains(&model) {
                return already_freed(&request, "model", &model);
            }
            if !state.models.contains_key(&model) {
                return unknown_handle(&request, "model", &model);
            }
            let session = state.handles.next("sess");
            let kv = state.handles.next("kv");

            state.kv_caches.insert(
                kv.clone(),
                KvCacheRecord {
                    session: Some(session.clone()),
                    layers: 0,
                    sequence_len: 0,
                    cached_keys: Vec::new(),
                    cached_values: Vec::new(),
                },
            );
            state.sessions.insert(
                session.clone(),
                SessionRecord {
                    model: Some(model.clone()),
                    tokenizer: None,
                    kv_cache: Some(kv.clone()),
                },
            );

            Response::ok(
                request.id,
                json!({
                    "session": session,
                    "kv_cache": kv,
                    "model": model,
                    "stub": true
                }),
            )
        }
        "free_session" => {
            let session = match require_arg_string(&request, "session") {
                Ok(value) => value,
                Err(response) => return response,
            };

            if state.freed_sessions.contains(&session) {
                return already_freed(&request, "session", &session);
            }
            if !state.sessions.contains_key(&session) {
                return unknown_handle(&request, "session", &session);
            }

            let kv_cache = state
                .sessions
                .remove(&session)
                .and_then(|record| record.kv_cache);

            if let Some(kv) = kv_cache {
                state.kv_caches.remove(&kv);
            }
            state.freed_sessions.insert(session.clone());

            Response::ok(
                request.id,
                json!({
                    "session": session,
                    "removed": true
                }),
            )
        }
        "generate" => {
            let session = match require_arg_string(&request, "session") {
                Ok(value) => value,
                Err(response) => return response,
            };
            let prompt = match require_arg_string(&request, "prompt") {
                Ok(value) => value,
                Err(response) => return response,
            };

            if state.freed_sessions.contains(&session) {
                return already_freed(&request, "session", &session);
            }
            if !state.sessions.contains_key(&session) {
                return unknown_handle(&request, "session", &session);
            }

            let job = state.handles.next("job");
            let result = json!({
                "job": job,
                "session": session,
                "text": format!("[stub generate] {}", prompt),
                "stub": true
            });

            state.jobs.insert(
                job.clone(),
                JobRecord {
                    session: Some(session),
                    status: "done".to_string(),
                    last_result: Some(result.clone()),
                },
            );

            Response::ok(request.id, result)
        }
        other => Response::err(
            request.id,
            "unknown_cmd",
            format!("unknown command: {other}"),
        ),
    }
}
