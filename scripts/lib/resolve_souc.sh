#!/usr/bin/env bash
# scripts/lib/resolve_souc.sh — shared souc binary resolution and cargo-skip logic.
# Source this file; do not execute directly.
#
# After sourcing:
#   SOUC_BIN            — path to the souc binary
#   SKIP_BUILD          — "1" if cargo should be skipped, "0" otherwise
#   sounio_cargo()      — wrapper: runs cargo if SKIP_BUILD=0, no-ops if 1
#   sounio_require_souc — asserts SOUC_BIN exists and is executable

# Guard against double-sourcing.
if [[ -n "${_SOUNIO_RESOLVE_SOUC_LOADED:-}" ]]; then
  return 0
fi
_SOUNIO_RESOLVE_SOUC_LOADED=1

_SOUNIO_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SOUNIO_ROOT_DIR="${_SOUNIO_ROOT_DIR:-$(cd "$_SOUNIO_LIB_DIR/../.." && pwd)}"

# Normalize boolean env vars: 1/true/yes/on → 1, 0/false/no/off → 0.
_sounio_normalize_bool() {
  local raw="$1"
  local name="$2"
  case "$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)  echo "1" ;;
    0|false|no|off) echo "0" ;;
    *)
      echo "error: invalid $name=$raw (expected 0/1/true/false/yes/no/on/off)" >&2
      return 1
      ;;
  esac
}

SOUNIO_REPO_HARD_NO_RUST="$(_sounio_normalize_bool "${SOUNIO_REPO_HARD_NO_RUST:-1}" "SOUNIO_REPO_HARD_NO_RUST")"
SKIP_BUILD="$(_sounio_normalize_bool "${SKIP_BUILD:-$SOUNIO_REPO_HARD_NO_RUST}" "SKIP_BUILD")"

# Resolve SOUC_BIN: explicit env → debug build → release build → PATH.
_sounio_resolve_bin() {
  local resolver="$_SOUNIO_ROOT_DIR/scripts/omega/omega_resolve_souc_bin.sh"
  if [[ -x "$resolver" ]]; then
    if _resolved_via_omega="$(
      OMEGA_SOUC_REQUIRE_PINNED="${OMEGA_SOUC_REQUIRE_PINNED:-$SOUNIO_REPO_HARD_NO_RUST}" \
      OMEGA_SOUC_ALLOW_LOCAL_FALLBACK="${OMEGA_SOUC_ALLOW_LOCAL_FALLBACK:-0}" \
        "$resolver" --print-path 2>/dev/null
    )"; then
      if [[ -n "$_resolved_via_omega" && -x "$_resolved_via_omega" ]]; then
        echo "$_resolved_via_omega"
        return 0
      fi
    fi
  fi
  if [[ -n "${SOUC_BIN:-}" && -x "$SOUC_BIN" ]]; then
    echo "$SOUC_BIN"
    return 0
  fi
  local root_bin="$_SOUNIO_ROOT_DIR/souc"
  if [[ -x "$root_bin" ]]; then
    echo "$root_bin"
    return 0
  fi
  local debug_bin="$_SOUNIO_ROOT_DIR/target/debug/souc"
  local release_bin="$_SOUNIO_ROOT_DIR/target/release/souc"
  if [[ -x "$debug_bin" ]]; then
    echo "$debug_bin"
    return 0
  fi
  if [[ -x "$release_bin" ]]; then
    echo "$release_bin"
    return 0
  fi
  if command -v souc >/dev/null 2>&1; then
    command -v souc
    return 0
  fi
  # Last resort: native compiler wrapper
  local native_wrapper="$_SOUNIO_ROOT_DIR/scripts/ci/souc-native-wrapper.sh"
  if [[ -f "$native_wrapper" ]]; then
    chmod +x "$native_wrapper"
    local native_bin="/tmp/souc-native.elf"
    if [[ ! -x "$native_bin" ]]; then
      bash "$_SOUNIO_ROOT_DIR/scripts/ci/build_native_souc.sh" "$native_bin" 2>/dev/null || true
    fi
    if [[ -x "$native_bin" ]]; then
      export SOUC_NATIVE_BIN="$native_bin"
      echo "$native_wrapper"
      return 0
    fi
  fi
  echo ""
  return 1
}

# Assert souc binary exists.
sounio_require_souc() {
  if [[ ! -x "$SOUC_BIN" ]]; then
    echo "error: souc binary not found at $SOUC_BIN" >&2
    echo "hint: build with 'cargo build -p souc' or set SOUC_BIN=/path/to/souc" >&2
    exit 1
  fi
}

# Wrapper: runs cargo normally if SKIP_BUILD=0, skips with message if SKIP_BUILD=1.
sounio_cargo() {
  if [[ "$SKIP_BUILD" = "1" ]]; then
    if [[ "$SOUNIO_REPO_HARD_NO_RUST" = "1" ]]; then
      echo "error: cargo invocation blocked in repo-hard no-rust mode: cargo $*" >&2
      return 2
    fi
    echo "[skip-build] skipping: cargo $*"
    return 0
  fi
  cargo "$@"
}

_SOUNIO_GPU_PROBE_REASON=""

# Probe whether a given souc binary can actually build the GPU fixture.
sounio_probe_gpu_backend() {
  local candidate="$1"
  local fixture="${2:-$_SOUNIO_ROOT_DIR/scripts/fixtures/gpu_minimal.sio}"
  _SOUNIO_GPU_PROBE_REASON=""

  if [[ -z "$candidate" || ! -x "$candidate" ]]; then
    _SOUNIO_GPU_PROBE_REASON="souc_unavailable"
    return 1
  fi
  if [[ ! -f "$fixture" ]]; then
    _SOUNIO_GPU_PROBE_REASON="gpu_fixture_missing"
    return 1
  fi

  local tmp_dir out_path log_path rc
  tmp_dir="$(mktemp -d)"
  out_path="$tmp_dir/gpu_probe.ptx"
  log_path="$tmp_dir/gpu_probe.log"
  set +e
  "$candidate" build "$fixture" --backend gpu -o "$out_path" >"$log_path" 2>&1
  rc=$?
  set -e

  if [[ $rc -eq 0 && -s "$out_path" ]] && grep -q '\.entry' "$out_path" >/dev/null 2>&1; then
    rm -rf "$tmp_dir"
    return 0
  fi

  if grep -qi "gpu backend not enabled\|not built with gpu support\|not built with gpu feature" "$log_path" >/dev/null 2>&1; then
    _SOUNIO_GPU_PROBE_REASON="gpu_backend_unavailable"
  elif grep -qi "unknown gpu target\|unsupported gpu target\|rocm unavailable\|hip unavailable\|amdgpu unavailable\|target unavailable" "$log_path" >/dev/null 2>&1; then
    _SOUNIO_GPU_PROBE_REASON="target_unavailable"
  else
    _SOUNIO_GPU_PROBE_REASON="gpu_probe_failed_rc_${rc}"
  fi

  rm -rf "$tmp_dir"
  return 1
}

sounio_gpu_probe_reason() {
  printf '%s' "${_SOUNIO_GPU_PROBE_REASON:-}"
}

# Resolve the first souc candidate that passes GPU backend probing.
sounio_resolve_gpu_souc() {
  local fixture="${1:-$_SOUNIO_ROOT_DIR/scripts/fixtures/gpu_minimal.sio}"
  local preferred="${2:-}"
  local -a candidates=()
  local -a extras=()

  _sounio_add_candidate() {
    local c="$1"
    local existing
    if [[ -z "$c" ]]; then
      return 0
    fi
    for existing in "${candidates[@]}"; do
      if [[ "$existing" == "$c" ]]; then
        return 0
      fi
    done
    candidates+=("$c")
  }

  _sounio_add_candidate "$preferred"
  _sounio_add_candidate "${SOUNIO_GPU_SOUC_BIN:-}"
  _sounio_add_candidate "${SOUC_BIN:-}"
  _sounio_add_candidate "$_SOUNIO_ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
  _sounio_add_candidate "$_SOUNIO_ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64"
  _sounio_add_candidate "$_SOUNIO_ROOT_DIR/souc"
  _sounio_add_candidate "$_SOUNIO_ROOT_DIR/target/release/souc"
  _sounio_add_candidate "$_SOUNIO_ROOT_DIR/target/debug/souc"

  if _resolved_fallback="$(_sounio_resolve_bin 2>/dev/null || true)"; then
    _sounio_add_candidate "$_resolved_fallback"
  fi
  unset _resolved_fallback

  if [[ -n "${SOUNIO_GPU_SOUC_CANDIDATES:-}" ]]; then
    IFS=':' read -r -a extras <<<"${SOUNIO_GPU_SOUC_CANDIDATES}"
    for c in "${extras[@]}"; do
      _sounio_add_candidate "$c"
    done
  fi

  local c last_reason
  last_reason="gpu_backend_unavailable"
  for c in "${candidates[@]}"; do
    if [[ ! -x "$c" ]]; then
      continue
    fi
    if sounio_probe_gpu_backend "$c" "$fixture"; then
      printf '%s\n' "$c"
      return 0
    fi
    if [[ -n "$(sounio_gpu_probe_reason)" ]]; then
      last_reason="$(sounio_gpu_probe_reason)"
    fi
  done

  _SOUNIO_GPU_PROBE_REASON="$last_reason"
  return 1
}

# If SOUC_BIN is not set or not executable, try to resolve it.
if [[ -z "${SOUC_BIN:-}" ]] || [[ ! -x "${SOUC_BIN:-}" ]]; then
  _resolved="$(_sounio_resolve_bin 2>/dev/null || true)"
  if [[ -n "$_resolved" ]]; then
    SOUC_BIN="$_resolved"
  fi
  unset _resolved
fi
