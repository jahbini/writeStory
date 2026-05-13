#!/usr/bin/env bash
# Run MLX lazy generation probe.
#
# Prerequisites:
#   - Homebrew MLX installed: `brew install mlx` (or `brew upgrade mlx`).
#   - Optional dev aid: a local MLX source checkout at `mlx/` with a built
#     `mlx/build/libmlx.a` (produced by `build_mlx.sh`). When present,
#     `binding.gyp` prefers it; otherwise it links against Homebrew MLX.
#
# Usage:
#   ./test.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/test/logs/mlx_lazy_${RUN_ID}"
MANIFEST="$LOG_DIR/manifest.txt"

mkdir -p "$LOG_DIR"

cd "$ROOT"

# Report which MLX library will be used
LOCAL_LIB="$ROOT/mlx/build/libmlx.a"
if [[ -f "$LOCAL_LIB" ]]; then
  MLX_SOURCE="local_0.32.0 ($LOCAL_LIB)"
else
  MLX_SOURCE="homebrew_0.30.0 (/opt/homebrew/lib/libmlx.dylib)"
fi

{
  echo "run_id=$RUN_ID"
  echo "mlx_source=$MLX_SOURCE"
  echo "started_at=$(date -Iseconds)"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "node=$(node --version 2>/dev/null || true)"
  echo "coffee=$(coffee --version 2>/dev/null || true)"
  echo "uname=$(uname -a 2>/dev/null || true)"
  sw_vers 2>/dev/null || true
} >"$MANIFEST"

echo "mlx: $MLX_SOURCE"
echo ""

run_step() {
  local name="$1"
  shift
  local out="$LOG_DIR/${name}.stdout.log"
  local err="$LOG_DIR/${name}.stderr.log"
  echo "command_${name}=$*" >>"$MANIFEST"
  set +e
  { time "$@" ; } >"$out" 2>"$err"
  local status=$?
  set -e
  echo "exit_${name}=$status" >>"$MANIFEST"
  echo "stdout_${name}=$out" >>"$MANIFEST"
  echo "stderr_${name}=$err" >>"$MANIFEST"
  if [[ "$status" -ne 0 ]]; then
    echo "failed_step=$name" >>"$MANIFEST"
    echo "finished_at=$(date -Iseconds)" >>"$MANIFEST"
    cat "$err" >&2
    exit "$status"
  fi
}

# ── rebuild addon ──────────────────────────────────────────────────────────────
run_step build npx node-gyp rebuild
echo "addon built ok"
echo ""

# ── MLX lazy generation (pure lazy graph, one eval per token) ─────────────────
# Default backend is now 'mlx' in the compiled code, but we set it explicitly.
# No GYPSY_DIRECT_METAL_* flags → all CPU-readback fragments disabled.
run_step mlx_lazy_64 \
  env GYPSY_ATTENTION_BACKEND=mlx_prealloc_kv \
  coffee test/helpers/native_64_mlx_lazy_generation_probe.coffee

echo "=== MLX lazy generation result ==="
cat "$LOG_DIR/mlx_lazy_64.stdout.log"
echo ""
echo "=== wall-clock / CPU time ==="
cat "$LOG_DIR/mlx_lazy_64.stderr.log"
echo ""
echo "finished_at=$(date -Iseconds)" >>"$MANIFEST"
echo "log_dir=$LOG_DIR" >>"$MANIFEST"
echo "logs: $LOG_DIR"
