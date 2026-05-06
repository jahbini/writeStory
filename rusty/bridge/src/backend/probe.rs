use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

fn push_if_exists(paths: &mut Vec<PathBuf>, path: impl AsRef<Path>) {
    let path = path.as_ref();
    if path.exists() {
        paths.push(path.to_path_buf());
    }
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(PathBuf::from)
}

fn canonical_display(path: &Path) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

fn collect_include_paths() -> Vec<String> {
    let mut paths = Vec::new();

    if let Some(home) = home_dir() {
        push_if_exists(&mut paths, home.join("development/mlx/mlx"));
        push_if_exists(&mut paths, home.join("development/mlx/build"));
        push_if_exists(&mut paths, home.join("development/node-mlx/deps/mlx/mlx"));
        push_if_exists(
            &mut paths,
            home.join("writeStory/.venv/lib/python3.14/site-packages/mlx/include"),
        );
        push_if_exists(
            &mut paths,
            home.join("writeStory/.venv/lib/python3.13/site-packages/mlx/include"),
        );
    }

    push_if_exists(&mut paths, "/opt/homebrew/include");
    push_if_exists(&mut paths, "/usr/local/include");

    if let Some(val) = env::var_os("MLX_INCLUDE_DIR") {
        push_if_exists(&mut paths, PathBuf::from(val));
    }
    if let Some(val) = env::var_os("CPATH") {
        for segment in env::split_paths(&val) {
            push_if_exists(&mut paths, segment);
        }
    }

    let mut strings = paths
        .into_iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    strings.sort();
    strings.dedup();
    strings
}

fn collect_library_paths() -> Vec<String> {
    let mut paths = Vec::new();

    if let Some(home) = home_dir() {
        push_if_exists(&mut paths, home.join("development/mlx/build"));
        push_if_exists(&mut paths, home.join("development/node-mlx/deps/mlx/build"));
        push_if_exists(
            &mut paths,
            home.join("writeStory/.venv/lib/python3.14/site-packages/mlx/lib"),
        );
        push_if_exists(
            &mut paths,
            home.join("writeStory/.venv/lib/python3.13/site-packages/mlx/lib"),
        );
    }

    push_if_exists(&mut paths, "/opt/homebrew/lib");
    push_if_exists(&mut paths, "/usr/local/lib");

    if let Some(val) = env::var_os("MLX_LIB_DIR") {
        push_if_exists(&mut paths, PathBuf::from(val));
    }
    if let Some(val) = env::var_os("MLX_LIBRARY_DIR") {
        push_if_exists(&mut paths, PathBuf::from(val));
    }
    if let Some(val) = env::var_os("LIBRARY_PATH") {
        for segment in env::split_paths(&val) {
            push_if_exists(&mut paths, segment);
        }
    }
    if let Some(val) = env::var_os("DYLD_LIBRARY_PATH") {
        for segment in env::split_paths(&val) {
            push_if_exists(&mut paths, segment);
        }
    }

    let mut strings = paths
        .into_iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    strings.sort();
    strings.dedup();
    strings
}

fn collect_python_package_paths() -> Vec<String> {
    let mut paths = Vec::new();

    if let Some(home) = home_dir() {
        push_if_exists(
            &mut paths,
            home.join("writeStory/.venv/lib/python3.14/site-packages/mlx"),
        );
        push_if_exists(
            &mut paths,
            home.join("writeStory/.venv/lib/python3.14/site-packages/mlx_lm"),
        );
        push_if_exists(
            &mut paths,
            home.join("writeStory/.venv/lib/python3.13/site-packages/mlx"),
        );
        push_if_exists(
            &mut paths,
            home.join("writeStory/.venv/lib/python3.13/site-packages/mlx_lm"),
        );
    }

    if let Some(val) = env::var_os("PYTHONPATH") {
        for segment in env::split_paths(&val) {
            push_if_exists(&mut paths, segment.join("mlx"));
            push_if_exists(&mut paths, segment.join("mlx_lm"));
        }
    }

    let mut strings = paths
        .into_iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    strings.sort();
    strings.dedup();
    strings
}

fn collect_candidate_header_files() -> Vec<String> {
    let mut files = Vec::new();

    if let Some(home) = home_dir() {
        for base in [
            home.join("development/mlx"),
            home.join("development/node-mlx/deps/mlx"),
            home.join("writeStory/.venv/lib/python3.14/site-packages/mlx/include"),
            home.join("writeStory/.venv/lib/python3.13/site-packages/mlx/include"),
        ] {
            for rel in ["mlx/mlx.h", "mlx/array.h", "mlx/ops.h"] {
                push_if_exists(&mut files, base.join(rel));
            }
        }
    }

    for base in [PathBuf::from("/opt/homebrew/include"), PathBuf::from("/usr/local/include")] {
        for rel in ["mlx/mlx.h", "mlx/array.h", "mlx/ops.h"] {
            push_if_exists(&mut files, base.join(rel));
        }
    }

    if let Some(val) = env::var_os("MLX_INCLUDE_DIR") {
        let base = PathBuf::from(val);
        for rel in ["mlx.h", "array.h", "ops.h", "mlx/mlx.h", "mlx/array.h", "mlx/ops.h"] {
            push_if_exists(&mut files, base.join(rel));
        }
    }

    let mut strings = files
        .into_iter()
        .map(|path| canonical_display(&path))
        .collect::<Vec<_>>();
    strings.sort();
    strings.dedup();
    strings
}

fn collect_candidate_library_files() -> Vec<String> {
    let mut files = Vec::new();

    if let Some(home) = home_dir() {
        for base in [
            home.join("development/mlx/build"),
            home.join("development/node-mlx/deps/mlx/build"),
            home.join("writeStory/.venv/lib/python3.14/site-packages/mlx/lib"),
            home.join("writeStory/.venv/lib/python3.13/site-packages/mlx/lib"),
        ] {
            for rel in ["libmlx.dylib", "libmlx.so", "libmlx.a"] {
                push_if_exists(&mut files, base.join(rel));
            }
        }
    }

    for base in [PathBuf::from("/opt/homebrew/lib"), PathBuf::from("/usr/local/lib")] {
        for rel in ["libmlx.dylib", "libmlx.so", "libmlx.a"] {
            push_if_exists(&mut files, base.join(rel));
        }
    }

    if let Some(val) = env::var_os("MLX_LIB_DIR") {
        let base = PathBuf::from(val);
        for rel in ["libmlx.dylib", "libmlx.so", "libmlx.a"] {
            push_if_exists(&mut files, base.join(rel));
        }
    }

    let mut strings = files
        .into_iter()
        .map(|path| canonical_display(&path))
        .collect::<Vec<_>>();
    strings.sort();
    strings.dedup();
    strings
}

fn collect_candidate_metallib_files() -> Vec<String> {
    let mut files = Vec::new();

    if let Some(home) = home_dir() {
        for path in [
            home.join("development/mlx/build/mlx.metallib"),
            home.join("development/mlx/build/Release/mlx.metallib"),
            home.join("development/node-mlx/deps/mlx/build/mlx.metallib"),
            home.join("writeStory/.venv/lib/python3.14/site-packages/mlx/lib/mlx.metallib"),
            home.join("writeStory/.venv/lib/python3.13/site-packages/mlx/lib/mlx.metallib"),
        ] {
            push_if_exists(&mut files, path);
        }
    }

    for path in [
        PathBuf::from("/opt/homebrew/lib/mlx.metallib"),
        PathBuf::from("/usr/local/lib/mlx.metallib"),
    ] {
        push_if_exists(&mut files, path);
    }

    if let Some(val) = env::var_os("MLX_LIB_DIR") {
        push_if_exists(&mut files, PathBuf::from(val).join("mlx.metallib"));
    }

    let mut strings = files
        .into_iter()
        .map(|path| canonical_display(&path))
        .collect::<Vec<_>>();
    strings.sort();
    strings.dedup();
    strings
}

fn collect_candidate_repo_paths() -> Vec<String> {
    let mut roots = Vec::new();
    if let Some(home) = home_dir() {
        for path in [
            home.join("development/mlx"),
            home.join("development/node-mlx"),
            home.join("development/node-mlx/deps/mlx"),
        ] {
            push_if_exists(&mut roots, path);
        }
    }
    let mut strings = roots
        .into_iter()
        .map(|path| canonical_display(&path))
        .collect::<Vec<_>>();
    strings.sort();
    strings.dedup();
    strings
}

fn installation_root_for(path: &str) -> String {
    if let Some(idx) = path.find("/opt/homebrew/Cellar/mlx/") {
        let prefix = &path[..idx + "/opt/homebrew/Cellar/mlx/".len()];
        let rest = &path[idx + "/opt/homebrew/Cellar/mlx/".len()..];
        if let Some(version) = rest.split('/').next() {
            return format!("{prefix}{version}");
        }
    }
    if let Some(idx) = path.find("/site-packages/mlx/") {
        return path[..idx + "/site-packages/mlx".len()].to_string();
    }
    if let Some(idx) = path.find("/development/node-mlx/deps/mlx/") {
        return path[..idx + "/development/node-mlx/deps/mlx".len()].to_string();
    }
    if let Some(idx) = path.find("/development/mlx/") {
        return path[..idx + "/development/mlx".len()].to_string();
    }
    if path.starts_with("/opt/homebrew/") {
        return "/opt/homebrew".to_string();
    }
    if path.starts_with("/usr/local/") {
        return "/usr/local".to_string();
    }
    Path::new(path)
        .parent()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| path.to_string())
}

fn version_evidence_for(root: &str, repo_paths: &[String]) -> Vec<String> {
    let mut evidence = Vec::new();
    if root.contains("/Cellar/mlx/") {
        evidence.push(format!("homebrew_cellar_root:{root}"));
    }
    if root.contains("/site-packages/mlx") {
        evidence.push(format!("python_package_root:{root}"));
    }
    if root.ends_with("/development/mlx") {
        evidence.push(format!("local_repo_root:{root}"));
    }
    if root.ends_with("/development/node-mlx/deps/mlx") {
        evidence.push(format!("node_mlx_deps_root:{root}"));
    }
    for repo in repo_paths {
        if root.starts_with(repo) {
            evidence.push(format!("repo_path:{repo}"));
        }
    }
    evidence.sort();
    evidence.dedup();
    evidence
}

fn build_installation_groups(
    header_files: &[String],
    library_files: &[String],
    metallib_files: &[String],
    repo_paths: &[String],
) -> Vec<Value> {
    let mut groups: BTreeMap<String, (Vec<String>, Vec<String>, Vec<String>)> = BTreeMap::new();

    for path in header_files {
        groups
            .entry(installation_root_for(path))
            .or_default()
            .0
            .push(path.clone());
    }
    for path in library_files {
        groups
            .entry(installation_root_for(path))
            .or_default()
            .1
            .push(path.clone());
    }
    for path in metallib_files {
        groups
            .entry(installation_root_for(path))
            .or_default()
            .2
            .push(path.clone());
    }

    groups
        .into_iter()
        .map(|(root, (mut headers, mut libraries, mut metallib))| {
            headers.sort();
            headers.dedup();
            libraries.sort();
            libraries.dedup();
            metallib.sort();
            metallib.dedup();

            let coherent = !headers.is_empty() && !libraries.is_empty();
            json!({
                "root": root,
                "headers": headers,
                "libraries": libraries,
                "metallib": metallib,
                "version_evidence": version_evidence_for(
                    &root,
                    repo_paths
                ),
                "coherent": coherent
            })
        })
        .collect()
}

fn select_preferred_installation(groups: &[Value]) -> (Option<String>, Vec<String>) {
    let mut notes = Vec::new();

    let explicit_include = env::var("MLX_INCLUDE_DIR").ok();
    let explicit_lib = env::var("MLX_LIB_DIR")
        .ok()
        .or_else(|| env::var("MLX_LIBRARY_DIR").ok());
    if let (Some(include_dir), Some(lib_dir)) = (explicit_include, explicit_lib) {
        let include_root = installation_root_for(&canonical_display(Path::new(&include_dir)));
        let lib_root = installation_root_for(&canonical_display(Path::new(&lib_dir)));
        if include_root == lib_root {
            notes.push("preferred explicit MLX_INCLUDE_DIR + MLX_LIB_DIR pair".to_string());
            return (Some(include_root), notes);
        }
        notes.push(format!(
            "explicit include/lib roots differ: include={include_root} lib={lib_root}"
        ));
    }

    let group_root = |pred: fn(&str) -> bool, need_metallib: bool| -> Option<String> {
        groups.iter().find_map(|group| {
            let root = group.get("root")?.as_str()?;
            let coherent = group.get("coherent")?.as_bool()?;
            let metallib_ok = !need_metallib
                || group
                    .get("metallib")
                    .and_then(|v| v.as_array())
                    .map(|arr| !arr.is_empty())
                    .unwrap_or(false);
            if coherent && metallib_ok && pred(root) {
                Some(root.to_string())
            } else {
                None
            }
        })
    };

    if let Some(root) = group_root(|root| root.ends_with("/development/mlx"), false) {
        notes.push("preferred local source/build repo pair".to_string());
        return (Some(root), notes);
    }
    if let Some(root) = group_root(|root| root.contains("/Cellar/mlx/") || root == "/opt/homebrew", true) {
        notes.push("preferred coherent Homebrew install".to_string());
        return (Some(root), notes);
    }
    if groups.iter().any(|g| {
        g.get("root")
            .and_then(|v| v.as_str())
            .map(|root| root.contains("/site-packages/mlx"))
            .unwrap_or(false)
    }) {
        notes.push("python package detected but not preferred as C++ link target".to_string());
    }
    if groups.iter().any(|g| {
        g.get("root")
            .and_then(|v| v.as_str())
            .map(|root| root.ends_with("/development/node-mlx/deps/mlx"))
            .unwrap_or(false)
    }) {
        notes.push("node-mlx deps detected but reserved for interface reference only".to_string());
    }
    (None, notes)
}

fn header_evidence(include_paths: &[String]) -> (bool, bool, Vec<String>) {
    let mut c_api_detected = false;
    let mut cpp_api_detected = false;
    let mut evidence = Vec::new();

    let header_patterns = [
        ("mlx/mlx.h", "umbrella MLX header"),
        ("mlx/array.h", "array header"),
        ("mlx/ops.h", "ops header"),
        ("mlx/api.h", "export/api visibility header"),
        ("mlx/export.h", "export header"),
    ];

    let c_api_patterns = [
        "mlx/c",
        "mlx/c_api.h",
        "mlx/c.h",
        "mlx/mlx_c.h",
        "mlx/api/c",
    ];

    for base in include_paths {
        let base_path = Path::new(base);

        for (relative, label) in header_patterns {
            let candidate = base_path.join(relative);
            if candidate.exists() {
                cpp_api_detected = true;
                evidence.push(format!("{label}: {}", canonical_display(&candidate)));
            }
        }

        for relative in c_api_patterns {
            let candidate = base_path.join(relative);
            if candidate.exists() {
                c_api_detected = true;
                evidence.push(format!("C API candidate: {}", canonical_display(&candidate)));
            }
        }
    }

    evidence.sort();
    evidence.dedup();
    (c_api_detected, cpp_api_detected, evidence)
}

fn library_evidence(library_paths: &[String]) -> Vec<String> {
    let mut evidence = Vec::new();

    let library_patterns = [
        "libmlx.dylib",
        "libmlx.so",
        "libmlx.a",
        "mlx.dylib",
        "mlx.so",
        "mlx.a",
        "libgguflib.a",
    ];

    for base in library_paths {
        let base_path = Path::new(base);
        for relative in library_patterns {
            let candidate = base_path.join(relative);
            if candidate.exists() {
                evidence.push(format!("library candidate: {}", canonical_display(&candidate)));
            }
        }
    }

    evidence.sort();
    evidence.dedup();
    evidence
}

fn python_evidence(python_paths: &[String]) -> (bool, Vec<String>) {
    let mut detected = false;
    let mut evidence = Vec::new();

    for base in python_paths {
        let path = Path::new(base);
        if path.exists() {
            detected = true;
            evidence.push(format!("python package candidate: {}", canonical_display(path)));
        }
    }

    evidence.sort();
    evidence.dedup();
    (detected, evidence)
}

fn classify_native_boundary(
    include_paths: &[String],
    library_paths: &[String],
    python_paths: &[String],
    installation_groups: &[Value],
) -> Value {
    let (c_api_detected, cpp_api_detected, mut evidence) = header_evidence(include_paths);
    evidence.extend(library_evidence(library_paths));
    let (python_package_detected, python_evidence) = python_evidence(python_paths);
    evidence.extend(python_evidence);
    evidence.sort();
    evidence.dedup();

    let recommended_path = if c_api_detected {
        "direct_c_api"
    } else if cpp_api_detected || python_package_detected {
        "rust_cpp_shim"
    } else {
        "unknown"
    };

    let (preferred_installation_root, installation_notes) =
        select_preferred_installation(installation_groups);

    json!({
        "c_api_detected": c_api_detected,
        "cpp_api_detected": cpp_api_detected,
        "python_package_detected": python_package_detected,
        "recommended_path": recommended_path,
        "preferred_installation_root": preferred_installation_root,
        "installation_notes": installation_notes,
        "evidence": evidence
    })
}

pub fn run_probe() -> Value {
    let include_paths = collect_include_paths();
    let library_paths = collect_library_paths();
    let python_paths = collect_python_package_paths();
    let header_files = collect_candidate_header_files();
    let library_files = collect_candidate_library_files();
    let metallib_files = collect_candidate_metallib_files();
    let repo_paths = collect_candidate_repo_paths();
    let installation_groups =
        build_installation_groups(&header_files, &library_files, &metallib_files, &repo_paths);
    let native_boundary = classify_native_boundary(
        &include_paths,
        &library_paths,
        &python_paths,
        &installation_groups,
    );

    let mut notes = Vec::new();
    let mlx_detected = include_paths.iter().any(|p| p.contains("/mlx"))
        || library_paths.iter().any(|p| p.contains("/mlx"))
        || python_paths.iter().any(|p| p.contains("/mlx"));
    let metal_detected = cfg!(target_os = "macos");

    notes.push("lightweight discovery only; no model load attempted".to_string());
    notes.push("no tensors, KV caches, or GPU buffers allocated".to_string());
    notes.push("Rusty must not mix MLX headers and libraries from different installs".to_string());
    if let Some(metal_env) = env::var_os("METAL_DEVICE_WRAPPER_TYPE") {
        notes.push(format!(
            "environment hint: METAL_DEVICE_WRAPPER_TYPE={}",
            metal_env.to_string_lossy()
        ));
    }
    if let Some(mlx_home) = env::var_os("MLX_HOME") {
        notes.push(format!("environment hint: MLX_HOME={}", mlx_home.to_string_lossy()));
    }
    if !mlx_detected {
        notes.push("no obvious local MLX include/library/package paths detected".to_string());
    }

    json!({
        "platform": env::consts::OS,
        "arch": env::consts::ARCH,
        "mlx_detected": mlx_detected,
        "metal_detected": metal_detected,
        "candidate_include_paths": include_paths,
        "candidate_library_paths": library_paths,
        "candidate_header_files": header_files,
        "candidate_library_files": library_files,
        "candidate_metallib_files": metallib_files,
        "candidate_repo_paths": repo_paths,
        "installation_groups": installation_groups,
        "native_boundary": native_boundary,
        "notes": notes
    })
}
