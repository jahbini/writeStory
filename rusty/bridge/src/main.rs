mod backend;
mod commands;
mod handles;
mod protocol;
mod state;

use std::io::{self, BufRead, Write};
use std::sync::mpsc;
use std::thread;

use protocol::{Request, Response};
use state::BridgeState;

struct WorkItem {
    request: Request,
    reply_to: mpsc::Sender<Response>,
}

fn main() {
    // Ownership boundary:
    // - JavaScript/CoffeeScript owns orchestration and opaque handles only.
    // - Rust owns future tensors, model lifetime, GPU memory, and KV caches.
    //
    // This scaffold keeps the bridge resident and restart-friendly, but does
    // not perform real MLX work yet.
    let (work_tx, work_rx) = mpsc::channel::<WorkItem>();

    let worker = thread::spawn(move || {
        let mut state = BridgeState::default();
        while let Ok(item) = work_rx.recv() {
            let response = commands::dispatch(&mut state, item.request);
            let shutdown = state.shutdown_requested;
            let _ = item.reply_to.send(response);
            if shutdown {
                break;
            }
        }
    });

    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(err) => {
                let response = Response::err("unknown", "stdin_read_failed", err.to_string());
                write_response(&mut stdout, &response);
                break;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let request: Request = match serde_json::from_str(trimmed) {
            Ok(req) => req,
            Err(err) => {
                let response = Response::err("unknown", "invalid_json", err.to_string());
                write_response(&mut stdout, &response);
                continue;
            }
        };

        if let Err(error) = request.validate() {
            let response = Response::err(request.id.clone(), error.code, error.message);
            write_response(&mut stdout, &response);
            continue;
        }

        let shutdown_requested = request.cmd == "bridge_shutdown";
        let (reply_tx, reply_rx) = mpsc::channel();

        if work_tx.send(WorkItem { request, reply_to: reply_tx }).is_err() {
            let response = Response::err(
                "unknown",
                "bridge_unavailable",
                "worker thread is not available",
            );
            write_response(&mut stdout, &response);
            break;
        }

        match reply_rx.recv() {
            Ok(response) => write_response(&mut stdout, &response),
            Err(err) => {
                let response = Response::err("unknown", "reply_failed", err.to_string());
                write_response(&mut stdout, &response);
                break;
            }
        }

        if shutdown_requested {
            break;
        }
    }

    drop(work_tx);
    let _ = worker.join();
}

fn write_response(stdout: &mut impl Write, response: &Response) {
    if let Ok(line) = serde_json::to_string(response) {
        let _ = writeln!(stdout, "{line}");
        let _ = stdout.flush();
    }
}
