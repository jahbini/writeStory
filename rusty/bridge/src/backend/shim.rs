use std::env;
use std::ffi::{CStr, CString};
use std::num::FpCategory;
use std::os::raw::c_char;
use std::os::raw::c_ulonglong;
use std::path::Path;

use serde_json::{json, Value};

unsafe extern "C" {
    fn rusty_mlx_shim_version() -> *const c_char;
    fn rusty_mlx_shim_probe() -> i32;
    fn rusty_mlx_link_probe() -> i32;
    fn rusty_mlx_create_test_array() -> u64;
    fn rusty_mlx_test_array_sum(handle: u64) -> f64;
    fn rusty_mlx_free_test_array(handle: u64) -> i32;
    fn rusty_mlx_runtime_diagnose_json() -> *const c_char;
    fn rusty_mlx_create_token_array(tokens: *const c_ulonglong, length: c_ulonglong)
        -> c_ulonglong;
    fn rusty_mlx_token_array_info_json(handle: c_ulonglong) -> *const c_char;
    fn rusty_mlx_free_token_array(handle: c_ulonglong) -> i32;
    fn rusty_mlx_mock_forward(
        session_handle: c_ulonglong,
        token_array_handle: c_ulonglong,
    ) -> c_ulonglong;
    fn rusty_mlx_mock_sample(logits_handle: c_ulonglong) -> c_ulonglong;
    fn rusty_mlx_array_info_json(handle: c_ulonglong) -> *const c_char;
    fn rusty_mlx_free_array(handle: c_ulonglong) -> i32;
    fn rusty_mlx_handle_counts_json() -> *const c_char;
    fn rusty_mlx_load_tensor_group(
        handle: *const c_char,
        model_dir: *const c_char,
        group: *const c_char,
    ) -> i32;
    fn rusty_mlx_tensor_group_info_json(handle: *const c_char) -> *const c_char;
    fn rusty_mlx_quantization_layout_probe_json(handle: *const c_char) -> *const c_char;
    fn rusty_mlx_free_tensor_group(handle: *const c_char) -> i32;
    fn rusty_mlx_load_embedding_group(handle: *const c_char, model_dir: *const c_char) -> i32;
    fn rusty_mlx_embedding_group_info_json(handle: *const c_char) -> *const c_char;
    fn rusty_mlx_free_embedding_group(handle: *const c_char) -> i32;
    fn rusty_mlx_compare_dequant_slice_json(
        handle: *const c_char,
        row: c_ulonglong,
        cols: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_quantized_linear_slice_probe_json(
        handle: *const c_char,
        input: *const f64,
        input_length: c_ulonglong,
        row: c_ulonglong,
        cols: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_quantized_linear_rows_probe_json(
        handle: *const c_char,
        input: *const f64,
        input_length: c_ulonglong,
        rows: c_ulonglong,
        cols: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_quantized_linear_fullrow_probe_json(
        handle: *const c_char,
        input: *const f64,
        input_length: c_ulonglong,
        rows: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_quantized_linear_vector_probe_json(
        handle: *const c_char,
        input: *const f64,
        input_length: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_rmsnorm_probe_json(
        handle: *const c_char,
        input: *const f64,
        input_length: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_layer0_single_token_probe_json(
        model_dir: *const c_char,
        token_id: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_layer0_mlp_probe_json(
        model_dir: *const c_char,
        token_id: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_layer0_block_probe_json(
        model_dir: *const c_char,
        token_id: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_layer_stack_probe_json(
        model_dir: *const c_char,
        token_id: c_ulonglong,
        layers: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_full_stack_single_token_probe_json(
        model_dir: *const c_char,
        token_id: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_kv_cache_storage_probe_json() -> *const c_char;
    fn rusty_mlx_incremental_attention_probe_json(
        model_dir: *const c_char,
        prompt_token_id: c_ulonglong,
        decode_token_id: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_session_layer_residency_probe_json(
        model_dir: *const c_char,
        prompt_token_id: c_ulonglong,
        decode_token_id: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_fastsmoke_generation_probe_json(
        model_dir: *const c_char,
        prompt_token_id: c_ulonglong,
        prompt_token_ids_csv: *const c_char,
        first_decode_token_id: c_ulonglong,
        generated_tokens: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_warm_resident_session_json(
        session: *const c_char,
        model_dir: *const c_char,
        adapter_dir: *const c_char,
    ) -> *const c_char;
    fn rusty_mlx_generate_tokens_for_session_json(
        session: *const c_char,
        model_dir: *const c_char,
        adapter_dir: *const c_char,
        prompt_token_id: c_ulonglong,
        prompt_token_ids_csv: *const c_char,
        first_decode_token_id: c_ulonglong,
        generated_tokens: c_ulonglong,
        temperature: f64,
        top_k: c_ulonglong,
        top_p: f64,
        seed: c_ulonglong,
        stop_on_eos: i32,
        eos_token_id: c_ulonglong,
        stop_token_ids_csv: *const c_char,
    ) -> *const c_char;
    fn rusty_mlx_free_resident_session(session: *const c_char);
    fn rusty_mlx_resident_incremental_timing_breakdown_probe_json(
        model_dir: *const c_char,
        decode_token_id: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_quantized_linear_kernel_probe_json(model_dir: *const c_char) -> *const c_char;
    fn rusty_mlx_quantized_linear_mlx_probe_json(model_dir: *const c_char) -> *const c_char;
    fn rusty_mlx_metal_first_resident_decode_probe_json(
        model_dir: *const c_char,
        prompt_token_id: c_ulonglong,
        decode_token_id: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_resident_incremental_optimized_compare_probe_json(
        model_dir: *const c_char,
        decode_token_id: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_resident_incremental_layout_cached_compare_probe_json(
        model_dir: *const c_char,
        decode_token_id: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_resident_incremental_mlp_optimized_compare_probe_json(
        model_dir: *const c_char,
        decode_token_id: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_resident_incremental_logits_optimized_compare_probe_json(
        model_dir: *const c_char,
        decode_token_id: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_resident_incremental_down_full_block_compare_probe_json(
        model_dir: *const c_char,
        decode_token_id: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_resident_incremental_gate_up_full_block_compare_probe_json(
        model_dir: *const c_char,
        decode_token_id: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_greedy_next_token_probe_json(
        model_dir: *const c_char,
        token_id: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_greedy_two_token_probe_json(
        model_dir: *const c_char,
        token_id: c_ulonglong,
    ) -> *const c_char;
    fn rusty_mlx_dequantize_group_slice(
        handle: *const c_char,
        rows: c_ulonglong,
        cols: c_ulonglong,
    ) -> c_ulonglong;
    fn rusty_mlx_load_layer_groups(
        handle: *const c_char,
        model_dir: *const c_char,
        layer: i32,
    ) -> i32;
    fn rusty_mlx_layer_groups_info_json(handle: *const c_char) -> *const c_char;
    fn rusty_mlx_free_layer_groups(handle: *const c_char) -> i32;
}

pub fn run_probe() -> Value {
    let version = unsafe {
        let ptr = rusty_mlx_shim_version();
        if ptr.is_null() {
            "unknown".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    let reachable = unsafe { rusty_mlx_shim_probe() == 1 };

    json!({
        "version": version,
        "reachable": reachable
    })
}

pub fn run_mlx_link_probe() -> Value {
    let enabled = option_env!("RUSTY_MLX_LINK_ENABLED").unwrap_or("0") == "1";
    let selected_root = option_env!("RUSTY_MLX_SELECTED_ROOT").unwrap_or("");
    let selection_source = option_env!("RUSTY_MLX_SELECTION_SOURCE").unwrap_or("");
    let include_dir = option_env!("RUSTY_MLX_INCLUDE_DIR").unwrap_or("");
    let lib_dir = option_env!("RUSTY_MLX_LIB_DIR").unwrap_or("");

    let linked = if enabled {
        unsafe { rusty_mlx_link_probe() == 1 }
    } else {
        false
    };

    let mut notes = vec![
        "minimal native MLX compile/link probe only".to_string(),
        "no model loading or inference attempted".to_string(),
        "no tensors exposed to Rust or JavaScript".to_string(),
    ];

    if enabled {
        if !selected_root.is_empty() {
            notes.push(format!("using MLX root: {selected_root}"));
        }
        if !selection_source.is_empty() {
            notes.push(format!("MLX selection source: {selection_source}"));
        }
        if !include_dir.is_empty() {
            notes.push(format!("using MLX include dir: {include_dir}"));
        }
        if !lib_dir.is_empty() {
            notes.push(format!("using MLX lib dir: {lib_dir}"));
        }
    } else {
        notes.push("MLX native link probe disabled because include/lib path was not detected at build time".to_string());
    }

    json!({
        "linked": linked,
        "notes": notes
    })
}

pub fn create_test_array() -> Value {
    let handle = unsafe { rusty_mlx_create_test_array() };
    let created = handle != 0;

    let notes = if created {
        vec![
            "native MLX test array created inside C++ shim".to_string(),
            "Rust only received an opaque integer handle".to_string(),
        ]
    } else {
        vec!["native MLX test array creation failed or MLX link is unavailable".to_string()]
    };

    json!({
        "handle": handle,
        "created": created,
        "notes": notes
    })
}

pub fn test_array_sum(handle: u64) -> Value {
    let value = unsafe { rusty_mlx_test_array_sum(handle) };
    let ok = !matches!(value.classify(), FpCategory::Nan);

    json!({
        "handle": handle,
        "ok": ok,
        "sum": if ok { json!(value) } else { Value::Null }
    })
}

pub fn free_test_array(handle: u64) -> Value {
    let freed = unsafe { rusty_mlx_free_test_array(handle) == 1 };

    json!({
        "handle": handle,
        "freed": freed
    })
}

pub fn create_token_array(tokens: &[u64]) -> Value {
    let handle = unsafe {
        rusty_mlx_create_token_array(
            tokens.as_ptr() as *const c_ulonglong,
            tokens.len() as c_ulonglong,
        )
    };
    let created = handle != 0;

    let notes = if created {
        vec![
            "token IDs were converted into an MLX array inside the C++ shim".to_string(),
            "Rust only received an opaque integer handle".to_string(),
        ]
    } else {
        vec!["token array creation failed or MLX link is unavailable".to_string()]
    };

    json!({
        "handle": handle,
        "created": created,
        "notes": notes
    })
}

pub fn token_array_info(handle: u64) -> Value {
    let raw = unsafe {
        let ptr = rusty_mlx_token_array_info_json(handle);
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn free_token_array(handle: u64) -> Value {
    let freed = unsafe { rusty_mlx_free_token_array(handle) == 1 };

    json!({
        "handle": handle,
        "freed": freed
    })
}

pub fn mock_forward(session_handle: u64, token_array_handle: u64) -> Value {
    let handle = unsafe { rusty_mlx_mock_forward(session_handle, token_array_handle) };
    let created = handle != 0;

    let notes = if created {
        vec![
            "mock logits array created inside C++ shim".to_string(),
            "Rust only received an opaque integer handle".to_string(),
        ]
    } else {
        vec!["mock forward failed or native handles were unavailable".to_string()]
    };

    json!({
        "handle": handle,
        "created": created,
        "notes": notes
    })
}

pub fn mock_sample(logits_handle: u64) -> Value {
    let token = unsafe { rusty_mlx_mock_sample(logits_handle) };
    let ok = token != 0;

    json!({
        "ok": ok,
        "token": if ok { json!(token) } else { Value::Null }
    })
}

pub fn array_info(handle: u64) -> Value {
    let raw = unsafe {
        let ptr = rusty_mlx_array_info_json(handle);
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn free_array(handle: u64) -> Value {
    let freed = unsafe { rusty_mlx_free_array(handle) == 1 };

    json!({
        "handle": handle,
        "freed": freed
    })
}

pub fn handle_counts() -> Value {
    let raw = unsafe {
        let ptr = rusty_mlx_handle_counts_json();
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn load_tensor_group(handle: &str, model_dir: &str, group: &str) -> Value {
    let handle_c = CString::new(handle).unwrap_or_default();
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let group_c = CString::new(group).unwrap_or_default();
    let loaded = unsafe {
        rusty_mlx_load_tensor_group(handle_c.as_ptr(), model_dir_c.as_ptr(), group_c.as_ptr()) == 1
    };

    json!({
        "handle": handle,
        "loaded": loaded
    })
}

pub fn tensor_group_info(handle: &str) -> Value {
    let handle_c = CString::new(handle).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_tensor_group_info_json(handle_c.as_ptr());
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn quantization_layout_probe(handle: &str) -> Value {
    let handle_c = CString::new(handle).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_quantization_layout_probe_json(handle_c.as_ptr());
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn free_tensor_group(handle: &str) -> Value {
    let handle_c = CString::new(handle).unwrap_or_default();
    let freed = unsafe { rusty_mlx_free_tensor_group(handle_c.as_ptr()) == 1 };

    json!({
        "handle": handle,
        "freed": freed
    })
}

pub fn load_embedding_group(handle: &str, model_dir: &str) -> Value {
    let handle_c = CString::new(handle).unwrap_or_default();
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let loaded =
        unsafe { rusty_mlx_load_embedding_group(handle_c.as_ptr(), model_dir_c.as_ptr()) == 1 };

    json!({
        "handle": handle,
        "loaded": loaded
    })
}

pub fn embedding_group_info(handle: &str) -> Value {
    let handle_c = CString::new(handle).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_embedding_group_info_json(handle_c.as_ptr());
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn free_embedding_group(handle: &str) -> Value {
    let handle_c = CString::new(handle).unwrap_or_default();
    let freed = unsafe { rusty_mlx_free_embedding_group(handle_c.as_ptr()) == 1 };

    json!({
        "handle": handle,
        "freed": freed
    })
}

pub fn compare_dequant_slice(handle: &str, row: u64, cols: u64) -> Value {
    let handle_c = CString::new(handle).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_compare_dequant_slice_json(
            handle_c.as_ptr(),
            row as c_ulonglong,
            cols as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn quantized_linear_slice_probe(handle: &str, input: &[f64], row: u64, cols: u64) -> Value {
    let handle_c = CString::new(handle).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_quantized_linear_slice_probe_json(
            handle_c.as_ptr(),
            input.as_ptr(),
            input.len() as c_ulonglong,
            row as c_ulonglong,
            cols as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn quantized_linear_rows_probe(handle: &str, input: &[f64], rows: u64, cols: u64) -> Value {
    let handle_c = CString::new(handle).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_quantized_linear_rows_probe_json(
            handle_c.as_ptr(),
            input.as_ptr(),
            input.len() as c_ulonglong,
            rows as c_ulonglong,
            cols as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn quantized_linear_fullrow_probe(handle: &str, input: &[f64], rows: u64) -> Value {
    let handle_c = CString::new(handle).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_quantized_linear_fullrow_probe_json(
            handle_c.as_ptr(),
            input.as_ptr(),
            input.len() as c_ulonglong,
            rows as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn quantized_linear_vector_probe(handle: &str, input: &[f64]) -> Value {
    let handle_c = CString::new(handle).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_quantized_linear_vector_probe_json(
            handle_c.as_ptr(),
            input.as_ptr(),
            input.len() as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn rmsnorm_probe(handle: &str, input: &[f64]) -> Value {
    let handle_c = CString::new(handle).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_rmsnorm_probe_json(
            handle_c.as_ptr(),
            input.as_ptr(),
            input.len() as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn layer0_single_token_probe(model_dir: &str, token_id: u64) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr =
            rusty_mlx_layer0_single_token_probe_json(model_dir_c.as_ptr(), token_id as c_ulonglong);
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn layer0_mlp_probe(model_dir: &str, token_id: u64) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_layer0_mlp_probe_json(model_dir_c.as_ptr(), token_id as c_ulonglong);
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn layer0_block_probe(model_dir: &str, token_id: u64) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_layer0_block_probe_json(model_dir_c.as_ptr(), token_id as c_ulonglong);
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn layer_stack_probe(model_dir: &str, token_id: u64, layers: u64) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_layer_stack_probe_json(
            model_dir_c.as_ptr(),
            token_id as c_ulonglong,
            layers as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn full_stack_single_token_probe(model_dir: &str, token_id: u64) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_full_stack_single_token_probe_json(
            model_dir_c.as_ptr(),
            token_id as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn kv_cache_storage_probe() -> Value {
    let raw = unsafe {
        let ptr = rusty_mlx_kv_cache_storage_probe_json();
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn incremental_attention_probe(model_dir: &str, prompt_token_id: u64, decode_token_id: u64) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_incremental_attention_probe_json(
            model_dir_c.as_ptr(),
            prompt_token_id as c_ulonglong,
            decode_token_id as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn session_layer_residency_probe(
    model_dir: &str,
    prompt_token_id: u64,
    decode_token_id: u64,
) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_session_layer_residency_probe_json(
            model_dir_c.as_ptr(),
            prompt_token_id as c_ulonglong,
            decode_token_id as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn fastsmoke_generation_probe(
    model_dir: &str,
    prompt_token_id: u64,
    first_decode_token_id: u64,
    generated_tokens: u64,
) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let prompt_token_ids_c = CString::new("").unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_fastsmoke_generation_probe_json(
            model_dir_c.as_ptr(),
            prompt_token_id as c_ulonglong,
            prompt_token_ids_c.as_ptr(),
            first_decode_token_id as c_ulonglong,
            generated_tokens as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn warm_resident_session(session: &str, model_dir: &str, adapter_dir: Option<&str>) -> Value {
    let session_c = CString::new(session).unwrap_or_default();
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let adapter_dir_c = CString::new(adapter_dir.unwrap_or("")).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_warm_resident_session_json(
            session_c.as_ptr(),
            model_dir_c.as_ptr(),
            adapter_dir_c.as_ptr(),
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn generate_tokens_for_session(
    session: &str,
    model_dir: &str,
    adapter_dir: Option<&str>,
    prompt_token_ids: &[u64],
    first_decode_token_id: u64,
    generated_tokens: u64,
    temperature: f64,
    top_k: u64,
    top_p: f64,
    seed: u64,
    stop_on_eos: bool,
    eos_token_id: u64,
    stop_token_ids: &[u64],
) -> Value {
    let session_c = CString::new(session).unwrap_or_default();
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let adapter_dir_c = CString::new(adapter_dir.unwrap_or("")).unwrap_or_default();
    let prompt_token_id = prompt_token_ids.first().copied().unwrap_or(1);
    let prompt_token_ids_csv = prompt_token_ids
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let prompt_token_ids_c = CString::new(prompt_token_ids_csv).unwrap_or_default();
    let stop_token_ids_csv = stop_token_ids
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let stop_token_ids_c = CString::new(stop_token_ids_csv).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_generate_tokens_for_session_json(
            session_c.as_ptr(),
            model_dir_c.as_ptr(),
            adapter_dir_c.as_ptr(),
            prompt_token_id as c_ulonglong,
            prompt_token_ids_c.as_ptr(),
            first_decode_token_id as c_ulonglong,
            generated_tokens as c_ulonglong,
            temperature,
            top_k as c_ulonglong,
            top_p,
            seed as c_ulonglong,
            if stop_on_eos { 1 } else { 0 },
            eos_token_id as c_ulonglong,
            stop_token_ids_c.as_ptr(),
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn free_resident_session(session: &str) {
    let session_c = CString::new(session).unwrap_or_default();
    unsafe {
        rusty_mlx_free_resident_session(session_c.as_ptr());
    }
}

pub fn resident_incremental_timing_breakdown_probe(
    model_dir: &str,
    decode_token_id: u64,
) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_resident_incremental_timing_breakdown_probe_json(
            model_dir_c.as_ptr(),
            decode_token_id as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn quantized_linear_kernel_probe(model_dir: &str) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_quantized_linear_kernel_probe_json(model_dir_c.as_ptr());
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn quantized_linear_mlx_probe(model_dir: &str) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_quantized_linear_mlx_probe_json(model_dir_c.as_ptr());
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn metal_first_resident_decode_probe(
    model_dir: &str,
    prompt_token_id: u64,
    decode_token_id: u64,
) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_metal_first_resident_decode_probe_json(
            model_dir_c.as_ptr(),
            prompt_token_id as c_ulonglong,
            decode_token_id as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn resident_incremental_optimized_compare_probe(
    model_dir: &str,
    decode_token_id: u64,
) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_resident_incremental_optimized_compare_probe_json(
            model_dir_c.as_ptr(),
            decode_token_id as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn resident_incremental_layout_cached_compare_probe(
    model_dir: &str,
    decode_token_id: u64,
) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_resident_incremental_layout_cached_compare_probe_json(
            model_dir_c.as_ptr(),
            decode_token_id as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn resident_incremental_mlp_optimized_compare_probe(
    model_dir: &str,
    decode_token_id: u64,
) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_resident_incremental_mlp_optimized_compare_probe_json(
            model_dir_c.as_ptr(),
            decode_token_id as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn resident_incremental_logits_optimized_compare_probe(
    model_dir: &str,
    decode_token_id: u64,
) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_resident_incremental_logits_optimized_compare_probe_json(
            model_dir_c.as_ptr(),
            decode_token_id as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn resident_incremental_down_full_block_compare_probe(
    model_dir: &str,
    decode_token_id: u64,
) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_resident_incremental_down_full_block_compare_probe_json(
            model_dir_c.as_ptr(),
            decode_token_id as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn resident_incremental_gate_up_full_block_compare_probe(
    model_dir: &str,
    decode_token_id: u64,
) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_resident_incremental_gate_up_full_block_compare_probe_json(
            model_dir_c.as_ptr(),
            decode_token_id as c_ulonglong,
        );
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn greedy_next_token_probe(model_dir: &str, token_id: u64) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr =
            rusty_mlx_greedy_next_token_probe_json(model_dir_c.as_ptr(), token_id as c_ulonglong);
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn greedy_two_token_probe(model_dir: &str, token_id: u64) -> Value {
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let raw = unsafe {
        let ptr =
            rusty_mlx_greedy_two_token_probe_json(model_dir_c.as_ptr(), token_id as c_ulonglong);
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn dequantize_group_slice(handle: &str, row: u64, cols: u64) -> Value {
    let handle_c = CString::new(handle).unwrap_or_default();
    let array_handle = unsafe {
        rusty_mlx_dequantize_group_slice(handle_c.as_ptr(), row as c_ulonglong, cols as c_ulonglong)
    };
    let created = array_handle != 0;

    let mut value = json!({
        "array": array_handle,
        "created": created
    });
    if created {
        let info = array_info(array_handle);
        if let Some(obj) = value.as_object_mut() {
            obj.insert(
                "dtype".to_string(),
                info.get("dtype").cloned().unwrap_or(Value::Null),
            );
            obj.insert(
                "shape".to_string(),
                info.get("shape").cloned().unwrap_or(Value::Null),
            );
            obj.insert(
                "size".to_string(),
                info.get("size").cloned().unwrap_or(Value::Null),
            );
            obj.insert(
                "source_group".to_string(),
                info.get("source_group").cloned().unwrap_or(Value::Null),
            );
        }
    }

    value
}

pub fn load_layer_groups(handle: &str, model_dir: &str, layer: i32) -> Value {
    let handle_c = CString::new(handle).unwrap_or_default();
    let model_dir_c = CString::new(model_dir).unwrap_or_default();
    let loaded =
        unsafe { rusty_mlx_load_layer_groups(handle_c.as_ptr(), model_dir_c.as_ptr(), layer) == 1 };

    json!({
        "handle": handle,
        "loaded": loaded
    })
}

pub fn layer_groups_info(handle: &str) -> Value {
    let handle_c = CString::new(handle).unwrap_or_default();
    let raw = unsafe {
        let ptr = rusty_mlx_layer_groups_info_json(handle_c.as_ptr());
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "invalid_json"
        })
    })
}

pub fn free_layer_groups(handle: &str) -> Value {
    let handle_c = CString::new(handle).unwrap_or_default();
    let freed = unsafe { rusty_mlx_free_layer_groups(handle_c.as_ptr()) == 1 };

    json!({
        "handle": handle,
        "freed": freed
    })
}

pub fn runtime_diagnose() -> Value {
    let dyld_library_path = env::var("DYLD_LIBRARY_PATH").ok();
    let mlx_metal_path = env::var("MLX_METAL_PATH").ok();
    let cwd = env::current_dir()
        .ok()
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    let lib_dir = option_env!("RUSTY_MLX_LIB_DIR").unwrap_or("");
    let resolved_libmlx_path = if lib_dir.is_empty() {
        None
    } else {
        let dylib = Path::new(lib_dir).join("libmlx.dylib");
        let so = Path::new(lib_dir).join("libmlx.so");
        if dylib.exists() {
            Some(dylib.display().to_string())
        } else if so.exists() {
            Some(so.display().to_string())
        } else {
            None
        }
    };

    let metallib_near_libmlx = if lib_dir.is_empty() {
        false
    } else {
        Path::new(lib_dir).join("mlx.metallib").exists()
    };

    let mut metallib_source_build_dirs = Vec::new();
    let mut candidates = vec![
        "/opt/homebrew/lib/mlx.metallib".to_string(),
        "/usr/local/lib/mlx.metallib".to_string(),
    ];
    if let Ok(home) = env::var("HOME") {
        candidates.push(format!("{home}/development/mlx/build/mlx.metallib"));
        candidates.push(format!("{home}/development/mlx/build/Release/mlx.metallib"));
        candidates.push(format!("{home}/development/mlx/mlx.metallib"));
        candidates.push(format!(
            "{home}/writeStory/.venv/lib/python3.14/site-packages/mlx/lib/mlx.metallib"
        ));
        candidates.push(format!(
            "{home}/writeStory/.venv/lib/python3.13/site-packages/mlx/lib/mlx.metallib"
        ));
    }
    for candidate in candidates {
        if Path::new(&candidate).exists() {
            metallib_source_build_dirs.push(candidate);
        }
    }

    let diagnose_json = unsafe {
        let ptr = rusty_mlx_runtime_diagnose_json();
        if ptr.is_null() {
            "{}".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    let parsed = serde_json::from_str::<Value>(&diagnose_json).unwrap_or_else(|_| {
        json!({
            "primary_target": "unknown",
            "primary_success": false,
            "primary_failure_stage": "array construction",
            "primary_exception": diagnose_json,
            "cpu_attempted": false,
            "cpu_success": false,
            "cpu_failure_stage": "",
            "cpu_exception": ""
        })
    });

    json!({
        "dyld_library_path": dyld_library_path,
        "mlx_metal_path": mlx_metal_path,
        "resolved_libmlx_path": resolved_libmlx_path,
        "mlx_metallib_near_libmlx": metallib_near_libmlx,
        "mlx_metallib_source_build_dirs": metallib_source_build_dirs,
        "cwd": cwd,
        "runtime_probe": parsed
    })
}
