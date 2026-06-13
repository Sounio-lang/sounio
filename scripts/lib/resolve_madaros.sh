#!/usr/bin/env bash
# scripts/lib/resolve_madaros.sh - shared Madaros (Stage1 modular compiler)
# resolution. Source this file; do not execute directly.

if [[ -n "${_SOUNIO_RESOLVE_MADAROS_LOADED:-}" ]]; then
  return 0
fi
_SOUNIO_RESOLVE_MADAROS_LOADED=1

_SOUNIO_MADAROS_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SOUNIO_MADAROS_ROOT_DIR="${_SOUNIO_MADAROS_ROOT_DIR:-$(cd "$_SOUNIO_MADAROS_LIB_DIR/../.." && pwd)}"

_sounio_madaros_is_wrapper() {
  local path="$1"
  [[ -f "$path" ]] && [[ "$(head -c 2 "$path" 2>/dev/null || true)" == "#!" ]]
}

_sounio_resolve_madaros_raw_bin() {
  local cand
  for cand in \
    "${MADAROS_RAW_BIN:-}" \
    "${SOUNIO_MADAROS_BIN:-}" \
    "$_SOUNIO_MADAROS_ROOT_DIR/artifacts/self-hosted/madaros" \
    "$_SOUNIO_MADAROS_ROOT_DIR/bin/madaros-linux-x86_64"; do
    if [[ -n "$cand" && -x "$cand" ]] && ! _sounio_madaros_is_wrapper "$cand"; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

_sounio_resolve_madaros_bin() {
  if [[ -x "$_SOUNIO_MADAROS_ROOT_DIR/bin/madaros" ]]; then
    echo "$_SOUNIO_MADAROS_ROOT_DIR/bin/madaros"
    return 0
  fi
  if command -v madaros >/dev/null 2>&1; then
    command -v madaros
    return 0
  fi
  _sounio_resolve_madaros_raw_bin
}

MADAROS_RAW_BIN="$(_sounio_resolve_madaros_raw_bin 2>/dev/null || true)"
MADAROS_BIN="$(_sounio_resolve_madaros_bin 2>/dev/null || true)"

sounio_require_madaros_raw() {
  if [[ -z "${MADAROS_RAW_BIN:-}" || ! -x "$MADAROS_RAW_BIN" ]]; then
    echo "error: Madaros raw ELF not found" >&2
    echo "  tried: MADAROS_RAW_BIN, SOUNIO_MADAROS_BIN, artifacts/self-hosted/madaros, bin/madaros-linux-x86_64" >&2
    return 1
  fi
}

sounio_require_madaros() {
  if [[ -z "${MADAROS_BIN:-}" || ! -x "$MADAROS_BIN" ]]; then
    echo "error: Madaros launcher not found" >&2
    return 1
  fi
}
