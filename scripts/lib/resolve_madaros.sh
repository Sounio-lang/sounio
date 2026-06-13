#!/usr/bin/env bash
# scripts/lib/resolve_madaros.sh — shared Madaros (modular compiler) resolution.
# Source this file; do not execute directly.
#
# After sourcing:
#   MADAROS_BIN            — path to the Madaros modular compiler binary
#   sounio_require_madaros — asserts MADAROS_BIN exists and is executable
#
# The modular compiler is built from self-hosted/compiler/main.sio (Stage1).
# It is distinct from the Stage0 bootstrap compiler resolved by
# scripts/lib/resolve_souc.sh.

# Guard against double-sourcing.
if [[ -n "${_SOUNIO_RESOLVE_MADAROS_LOADED:-}" ]]; then
  return 0
fi
_SOUNIO_RESOLVE_MADAROS_LOADED=1

_SOUNIO_MADAROS_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SOUNIO_MADAROS_ROOT_DIR="${_SOUNIO_MADAROS_ROOT_DIR:-$(cd "$_SOUNIO_MADAROS_LIB_DIR/../.." && pwd)}"

# Resolve MADAROS_BIN: explicit env → repo wrapper → repo raw ELF → PATH fallback.
_sounio_resolve_madaros_bin() {
  if [[ -n "${MADAROS_BIN:-}" && -x "$MADAROS_BIN" ]]; then
    echo "$MADAROS_BIN"
    return 0
  fi
  if [[ -n "${SOUNIO_MADAROS_BIN:-}" && -x "$SOUNIO_MADAROS_BIN" ]]; then
    echo "$SOUNIO_MADAROS_BIN"
    return 0
  fi
  local wrapper="$_SOUNIO_MADAROS_ROOT_DIR/bin/madaros"
  if [[ -x "$wrapper" ]]; then
    echo "$wrapper"
    return 0
  fi
  local raw_elf="$_SOUNIO_MADAROS_ROOT_DIR/artifacts/self-hosted/madaros"
  if [[ -x "$raw_elf" ]]; then
    echo "$raw_elf"
    return 0
  fi
  local prebuilt="$_SOUNIO_MADAROS_ROOT_DIR/bin/madaros-linux-x86_64"
  if [[ -x "$prebuilt" ]]; then
    echo "$prebuilt"
    return 0
  fi
  if command -v madaros >/dev/null 2>&1; then
    command -v madaros
    return 0
  fi
  echo "error: resolve_madaros: no Madaros binary found" >&2
  echo "  tried: MADAROS_BIN, SOUNIO_MADAROS_BIN, $wrapper, $raw_elf, $prebuilt, PATH" >&2
  echo "  run: make build-madaros" >&2
  return 1
}

MADAROS_BIN="$(_sounio_resolve_madaros_bin)"

sounio_require_madaros() {
  if [[ -z "${MADAROS_BIN:-}" || ! -x "$MADAROS_BIN" ]]; then
    echo "error: Madaros binary not found or not executable: ${MADAROS_BIN:-<unset>}" >&2
    return 1
  fi
}
