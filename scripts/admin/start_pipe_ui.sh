#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_NAME=""
MODEL_NAME=""
PIPE_NAME=""
PIPE_DIR=""
UI_BIND_MODE="${UI_BIND_MODE:-local}"
MODE="auto"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo "[start_pipe_ui] $*"
}

pick_default_pipe() {
  local pipe_root="$ROOT_DIR/pipes"
  local candidates=()

  [[ -d "$pipe_root" ]] || return 0

  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    candidates+=("$line")
  done < <(find "$pipe_root" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort)

  if [[ "${#candidates[@]}" -eq 0 ]]; then
    return 0
  fi

  if [[ "${#candidates[@]}" -eq 1 ]]; then
    basename "${candidates[0]}"
    return 0
  fi

  basename "$(ls -td "$pipe_root"/* 2>/dev/null | head -n 1)"
}

pipe_name_for_model() {
  local model_name="$1"
  local normalized

  normalized="${model_name//\//_}"
  normalized="${normalized// /_}"
  printf '%s\n' "$normalized"
}

legacy_pipe_name_for_model() {
  local model_name="$1"
  printf '%s\n' "${model_name##*/}"
}

ensure_parent_dir() {
  local target="$1"
  mkdir -p "$(dirname "$target")"
}

write_override_yaml() {
  ensure_parent_dir "$PIPE_DIR/override.yaml"
  cat > "$PIPE_DIR/override.yaml" <<EOF
run:
  model: $MODEL_NAME
EOF
}

read_override_model() {
  local override_path="$1"
  [[ -f "$override_path" ]] || return 0
  sed -n 's/^[[:space:]]*model:[[:space:]]*\(.*\)[[:space:]]*$/\1/p' "$override_path" | head -n 1
}

resolve_coffee_bin() {
  if [[ -x "$ROOT_DIR/node_modules/.bin/coffee" ]]; then
    printf '%s\n' "$ROOT_DIR/node_modules/.bin/coffee"
    return
  fi

  if command -v coffee >/dev/null 2>&1; then
    command -v coffee
    return
  fi

  die "Could not find a coffee executable"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    net|--net)
      UI_BIND_MODE="net"
      shift
      ;;
    local|--local)
      UI_BIND_MODE="local"
      shift
      ;;
    --pipe)
      MODE="pipe"
      shift
      [[ $# -gt 0 ]] || die "--pipe requires a value"
      TARGET_NAME="$1"
      shift
      ;;
    --model)
      MODE="model"
      shift
      [[ $# -gt 0 ]] || die "--model requires a value"
      TARGET_NAME="$1"
      shift
      ;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [net|--net] [local|--local] [--pipe PIPE_NAME | --model MODEL_ID | TARGET]

Rules:
  If no TARGET is given, the script auto-selects a pipe from ./pipes.
  If TARGET matches an existing pipes/<name> directory, it starts that pipe unchanged.
  Otherwise TARGET is treated as a model id and a pipe workspace is created/updated.
EOF
      exit 0
      ;;
    *)
      if [[ -n "$TARGET_NAME" ]]; then
        die "Unexpected extra argument: $1"
      fi
      TARGET_NAME="$1"
      shift
      ;;
  esac
done

if [[ -z "$TARGET_NAME" && "$MODE" != "model" ]]; then
  TARGET_NAME="$(pick_default_pipe)"
  [[ -n "$TARGET_NAME" ]] || die "No pipe specified and no existing pipes found under $ROOT_DIR/pipes"
  info "Auto-selected pipe: $TARGET_NAME"
fi

[[ -n "$TARGET_NAME" ]] || die "Usage: $(basename "$0") [net|--net] [local|--local] [--pipe PIPE_NAME | --model MODEL_ID | TARGET]"

if [[ "$MODE" == "pipe" || ( "$MODE" == "auto" && -d "$ROOT_DIR/pipes/$TARGET_NAME" ) ]]; then
  PIPE_NAME="$TARGET_NAME"
  PIPE_DIR="$ROOT_DIR/pipes/$PIPE_NAME"
  [[ -d "$PIPE_DIR" ]] || die "Pipe directory not found: $PIPE_DIR"
  MODEL_NAME="$(read_override_model "$PIPE_DIR/override.yaml")"
else
  MODEL_NAME="$TARGET_NAME"
  PIPE_NAME="$(pipe_name_for_model "$MODEL_NAME")"
  [[ -n "$PIPE_NAME" ]] || die "Could not derive pipe directory name from model name: $MODEL_NAME"
  PIPE_DIR="$ROOT_DIR/pipes/$PIPE_NAME"
  LEGACY_PIPE_NAME="$(legacy_pipe_name_for_model "$MODEL_NAME")"
  LEGACY_PIPE_DIR="$ROOT_DIR/pipes/$LEGACY_PIPE_NAME"
  mkdir -p "$PIPE_DIR"
  mkdir -p "$PIPE_DIR/logs" "$PIPE_DIR/state"
  write_override_yaml
fi

mkdir -p "$PIPE_DIR/logs" "$PIPE_DIR/state"

COFFEE_BIN="$(resolve_coffee_bin)"
UI_PORT_VALUE="${UI_PORT:-4311}"

info "Workspace: $PIPE_DIR"
if [[ -n "$MODEL_NAME" ]]; then
  info "Model: $MODEL_NAME"
else
  info "Model: (unchanged from pipe workspace)"
fi
info "Pipe name: $PIPE_NAME"
info "UI bind mode: $UI_BIND_MODE"
if [[ "${LEGACY_PIPE_NAME:-}" != "$PIPE_NAME" && -n "${LEGACY_PIPE_DIR:-}" && -d "$LEGACY_PIPE_DIR" && ! -d "$PIPE_DIR/build" ]]; then
  info "Legacy tail-only pipe also exists: $LEGACY_PIPE_DIR"
  info "New launches now use organization-qualified pipe names."
fi
info "UI port: $UI_PORT_VALUE"
info "Starting UI server"

cd "$PIPE_DIR"
exec env EXEC="$ROOT_DIR" CWD="$PIPE_DIR" UI_PORT="$UI_PORT_VALUE" UI_BIND_MODE="$UI_BIND_MODE" "$COFFEE_BIN" "$ROOT_DIR/ui_server.coffee"
