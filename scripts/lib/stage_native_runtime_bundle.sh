#!/usr/bin/env bash
# scripts/lib/stage_native_runtime_bundle.sh — restore the checked native runtime C shim path.
#
# The pinned souc artifact currently bakes an absolute temp release-build path for
# chiuratto_ffi.c. This helper resolves the real source of that shim and stages it
# into the baked location before invoking `build --backend native`.

set -euo pipefail

if [[ -n "${_SOUNIO_STAGE_NATIVE_RUNTIME_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
_SOUNIO_STAGE_NATIVE_RUNTIME_LOADED=1

_SOUNIO_STAGE_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SOUNIO_STAGE_ROOT_DIR="${_SOUNIO_STAGE_ROOT_DIR:-$(cd "$_SOUNIO_STAGE_LIB_DIR/../.." && pwd)}"

sounio_native_runtime_bundle_dir() {
  local souc_bin="${1:-${SOUC_BIN:-}}"
  if [[ -z "$souc_bin" ]]; then
    return 1
  fi
  local bin_dir
  bin_dir="$(cd "$(dirname "$souc_bin")" && pwd)"
  printf '%s\n' "$bin_dir/runtime/native"
}

sounio_resolve_native_runtime_source() {
  local souc_bin="${1:-${SOUC_BIN:-}}"
  local candidate=""

  if [[ -n "${SOUNIO_NATIVE_RUNTIME_DIR:-}" ]]; then
    candidate="${SOUNIO_NATIVE_RUNTIME_DIR%/}/chiuratto_ffi.c"
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  if [[ -n "$souc_bin" ]]; then
    candidate="$(sounio_native_runtime_bundle_dir "$souc_bin")/chiuratto_ffi.c"
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  candidate="$_SOUNIO_STAGE_ROOT_DIR/crates/souc/src/backend/native/chiuratto_ffi.c"
  if [[ -f "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  echo "error: unable to resolve chiuratto_ffi.c runtime shim" >&2
  return 1
}

sounio_native_runtime_baked_path() {
  local souc_bin="${1:-${SOUC_BIN:-}}"
  if [[ -z "$souc_bin" || ! -x "$souc_bin" ]]; then
    return 1
  fi

  local release_root=""
  release_root="$(
    strings "$souc_bin" \
      | grep -m1 -oE '/tmp/sounio-release-build\.[^/[:space:]]+' \
      || true
  )"

  if [[ -z "$release_root" ]]; then
    return 0
  fi

  printf '%s\n' "$release_root/crates/souc/src/backend/native/chiuratto_ffi.c"
}

sounio_stage_native_runtime_bundle() {
  local souc_bin="${1:-${SOUC_BIN:-}}"
  if [[ -z "$souc_bin" || ! -x "$souc_bin" ]]; then
    echo "error: souc binary not found for native runtime staging" >&2
    return 1
  fi

  local runtime_src baked_path
  runtime_src="$(sounio_resolve_native_runtime_source "$souc_bin")"
  baked_path="$(sounio_native_runtime_baked_path "$souc_bin")"

  if [[ -z "$baked_path" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "$baked_path")"
  ln -sf "$runtime_src" "$baked_path"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  sounio_stage_native_runtime_bundle "${1:-${SOUC_BIN:-}}"
fi
