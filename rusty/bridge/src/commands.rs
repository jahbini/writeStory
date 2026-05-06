use serde_json::{json, Value};

use crate::backend::{probe, shim};
use crate::protocol::{alive_value, Request, Response};
use crate::state::{BridgeState, JobRecord, KvCacheRecord, ModelRecord, SessionRecord};

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
            format!("{} requires args.{} as a non-zero integer", request.cmd, key),
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

pub fn dispatch(state: &mut BridgeState, request: Request) -> Response {
    if state.shutdown_requested && request.cmd != "bridge_health" && request.cmd != "bridge_shutdown" {
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
        "load_model" => {
            let path = match require_arg_string(&request, "path") {
                Ok(value) => value,
                Err(response) => return response,
            };

            let handle = state.handles.next("model");
            state.models.insert(handle.clone(), ModelRecord { path: path.clone() });

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
