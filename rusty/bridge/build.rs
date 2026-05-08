use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Debug)]
struct InstallChoice {
    root: PathBuf,
    include_dir: PathBuf,
    lib_dir: PathBuf,
    metallib: Option<PathBuf>,
    source: &'static str,
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(PathBuf::from)
}

fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn root_for(path: &Path) -> PathBuf {
    let text = canonical(path).display().to_string();
    if let Some(idx) = text.find("/opt/homebrew/Cellar/mlx/") {
        let prefix = &text[..idx + "/opt/homebrew/Cellar/mlx/".len()];
        let rest = &text[idx + "/opt/homebrew/Cellar/mlx/".len()..];
        if let Some(version) = rest.split('/').next() {
            return PathBuf::from(format!("{prefix}{version}"));
        }
    }
    if let Some(idx) = text.find("/site-packages/mlx/") {
        return PathBuf::from(&text[..idx + "/site-packages/mlx".len()]);
    }
    if let Some(idx) = text.find("/development/mlx/") {
        return PathBuf::from(&text[..idx + "/development/mlx".len()]);
    }
    if let Some(idx) = text.find("/development/node-mlx/deps/mlx/") {
        return PathBuf::from(&text[..idx + "/development/node-mlx/deps/mlx".len()]);
    }
    path.parent().unwrap_or(path).to_path_buf()
}

fn explicit_choice() -> Result<Option<InstallChoice>, String> {
    let include = env::var_os("MLX_INCLUDE_DIR").map(PathBuf::from);
    let lib = env::var_os("MLX_LIB_DIR")
        .map(PathBuf::from)
        .or_else(|| env::var_os("MLX_LIBRARY_DIR").map(PathBuf::from));

    match (include, lib) {
        (None, None) => Ok(None),
        (Some(include_dir), Some(lib_dir)) => {
            let include_root = root_for(&include_dir);
            let lib_root = root_for(&lib_dir);
            if include_root != lib_root {
                return Err(format!(
                    "explicit MLX paths are mismatched: include_root={} lib_root={}",
                    include_root.display(),
                    lib_root.display()
                ));
            }
            let metallib = {
                let candidate = lib_dir.join("mlx.metallib");
                if candidate.exists() {
                    Some(candidate)
                } else {
                    None
                }
            };
            Ok(Some(InstallChoice {
                root: include_root,
                include_dir,
                lib_dir,
                metallib,
                source: "explicit_env",
            }))
        }
        _ => Err("both MLX_INCLUDE_DIR and MLX_LIB_DIR must be set together".to_string()),
    }
}

fn local_repo_choice() -> Option<InstallChoice> {
    let home = home_dir()?;
    let root = home.join("development/mlx");
    let include_dir = root.clone();
    let lib_dir = root.join("build");
    let dylib = lib_dir.join("libmlx.dylib");
    if include_dir.join("mlx/device.h").exists() && dylib.exists() {
        let metallib = {
            let direct = lib_dir.join("mlx.metallib");
            let release = lib_dir.join("Release/mlx.metallib");
            if direct.exists() {
                Some(direct)
            } else if release.exists() {
                Some(release)
            } else {
                None
            }
        };
        return Some(InstallChoice {
            root,
            include_dir,
            lib_dir,
            metallib,
            source: "local_repo",
        });
    }
    None
}

fn homebrew_choice() -> Option<InstallChoice> {
    let root = PathBuf::from("/opt/homebrew/Cellar/mlx/0.30.0");
    let include_dir = root.join("include");
    let lib_dir = root.join("lib");
    let dylib = lib_dir.join("libmlx.dylib");
    let metallib = lib_dir.join("mlx.metallib");
    if include_dir.join("mlx/device.h").exists() && dylib.exists() && metallib.exists() {
        return Some(InstallChoice {
            root,
            include_dir,
            lib_dir,
            metallib: Some(metallib),
            source: "homebrew",
        });
    }
    None
}

fn python_choice() -> Option<InstallChoice> {
    let home = home_dir()?;
    for root in [
        home.join("writeStory/.venv/lib/python3.14/site-packages/mlx"),
        home.join("writeStory/.venv/lib/python3.13/site-packages/mlx"),
    ] {
        let include_dir = root.join("include");
        let lib_dir = root.join("lib");
        let dylib = lib_dir.join("libmlx.dylib");
        if include_dir.join("mlx/device.h").exists() && dylib.exists() {
            let metallib = {
                let candidate = lib_dir.join("mlx.metallib");
                if candidate.exists() {
                    Some(candidate)
                } else {
                    None
                }
            };
            return Some(InstallChoice {
                root,
                include_dir,
                lib_dir,
                metallib,
                source: "python_package",
            });
        }
    }
    None
}

fn select_installation() -> Result<Option<InstallChoice>, String> {
    if let Some(choice) = explicit_choice()? {
        return Ok(Some(choice));
    }
    if let Some(choice) = local_repo_choice() {
        return Ok(Some(choice));
    }
    if let Some(choice) = homebrew_choice() {
        return Ok(Some(choice));
    }
    if let Some(choice) = python_choice() {
        return Ok(Some(choice));
    }
    Ok(None)
}

fn main() {
    println!("cargo:rerun-if-changed=native/mlx_shim.cpp");
    println!("cargo:rerun-if-changed=native/mlx_shim.h");
    println!("cargo:rerun-if-env-changed=MLX_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=MLX_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MLX_LIBRARY_DIR");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let object = out_dir.join("mlx_shim.o");
    let library = out_dir.join("libmlx_shim.a");

    let selected = match select_installation() {
        Ok(choice) => choice,
        Err(message) => {
            panic!("refusing mixed MLX installation selection: {message}");
        }
    };

    let mlx_enabled = selected.is_some();
    let selected_root = selected
        .as_ref()
        .map(|choice| choice.root.display().to_string())
        .unwrap_or_default();
    let include_dir = selected.as_ref().map(|choice| choice.include_dir.clone());
    let lib_dir = selected.as_ref().map(|choice| choice.lib_dir.clone());
    let source = selected
        .as_ref()
        .map(|choice| choice.source)
        .unwrap_or("none");

    println!(
        "cargo:warning=rusty selected MLX root: {}",
        if selected_root.is_empty() {
            "<none>"
        } else {
            &selected_root
        }
    );
    println!(
        "cargo:warning=rusty selected MLX include dir: {}",
        include_dir.as_deref().unwrap_or(Path::new("")).display()
    );
    println!(
        "cargo:warning=rusty selected MLX lib dir: {}",
        lib_dir.as_deref().unwrap_or(Path::new("")).display()
    );
    if let Some(choice) = &selected {
        if choice.metallib.is_none() {
            println!(
                "cargo:warning=rusty selected MLX root has no mlx.metallib alongside library: {}",
                choice.root.display()
            );
        }
    } else {
        println!("cargo:warning=rusty no coherent MLX installation selected");
    }

    println!(
        "cargo:rustc-env=RUSTY_MLX_LINK_ENABLED={}",
        if mlx_enabled { "1" } else { "0" }
    );
    println!("cargo:rustc-env=RUSTY_MLX_SELECTED_ROOT={selected_root}");
    println!("cargo:rustc-env=RUSTY_MLX_SELECTION_SOURCE={source}");
    println!(
        "cargo:rustc-env=RUSTY_MLX_INCLUDE_DIR={}",
        include_dir.as_deref().unwrap_or(Path::new("")).display()
    );
    println!(
        "cargo:rustc-env=RUSTY_MLX_LIB_DIR={}",
        lib_dir.as_deref().unwrap_or(Path::new("")).display()
    );

    let mut compile = Command::new("/usr/bin/c++");
    compile.args(["-std=c++17", "-c", "native/mlx_shim.cpp", "-o"]);
    compile.arg(&object);
    if let Some(include_dir) = &include_dir {
        compile.arg(format!("-I{}", include_dir.display()));
    }
    if mlx_enabled {
        compile.arg("-DRUSTY_MLX_HAVE_NATIVE_LINK=1");
    }

    let compile_status = compile.status().expect("failed to invoke C++ compiler");
    assert!(compile_status.success(), "C++ shim compile failed");

    let archive_status = Command::new("/usr/bin/ar")
        .args(["crus"])
        .arg(&library)
        .arg(&object)
        .status()
        .expect("failed to invoke ar");
    assert!(archive_status.success(), "C++ shim archive failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=mlx_shim");
    println!("cargo:rustc-link-lib=dylib=c++");
    if let Some(lib_dir) = &lib_dir {
        println!(
            "cargo:rustc-link-search=native={}",
            canonical(lib_dir).display()
        );
        println!("cargo:rustc-link-lib=dylib=mlx");
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            canonical(lib_dir).display()
        );
    }
}
