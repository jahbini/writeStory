use std::ffi::CStr;
use std::env;
use std::os::raw::c_char;
use std::num::FpCategory;
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
