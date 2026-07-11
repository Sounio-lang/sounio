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

# A raw artifacts/self-hosted/madaros is trusted only when its gate receipt
# binds the current ELF and proves SMT plus EISA native behavioral gates.
# Otherwise it is demoted below relocgate.
_sounio_madaros_receipt_ok() {
  local elf="$1"
  local receipt="$elf.gate-receipt"
  [[ -f "$receipt" ]] || return 1
  local want
  want="$(sha256sum "$elf" 2>/dev/null | cut -d' ' -f1)"
  [[ -n "$want" ]] || return 1
  grep -Fq "$want" "$receipt" || return 1
  grep -Fxq "smt_skip=0" "$receipt" || return 1
  grep -Fxq "eisa_native_conformance=39/39" "$receipt" || return 1
  grep -Fxq "eisa_native_tamper=pass" "$receipt" || return 1
  grep -Fxq "eisa_native_anti_vacuity=pass" "$receipt" || return 1
}

# Resolve MADAROS_BIN: explicit env → repo wrapper → gated raw ELF →
# relocgate (verified-green) → checked-in prebuilt → explicit PATH opt-in.
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
    if _sounio_madaros_receipt_ok "$raw_elf"; then
      echo "$raw_elf"
      return 0
    fi
    echo "warning: using ungated $raw_elf skipped; export MADAROS_RAW_BIN=$_SOUNIO_MADAROS_ROOT_DIR/bin/madaros-relocgate for the verified-green binary" >&2
  fi
  local relocgate="$_SOUNIO_MADAROS_ROOT_DIR/bin/madaros-relocgate"
  if [[ -x "$relocgate" ]]; then
    echo "$relocgate"
    return 0
  fi
  local prebuilt="$_SOUNIO_MADAROS_ROOT_DIR/bin/madaros-linux-x86_64"
  if [[ -x "$prebuilt" ]]; then
    echo "$prebuilt"
    return 0
  fi
  if [[ "${SOUNIO_MADAROS_ALLOW_PATH_FALLBACK:-0}" == "1" ]] && command -v madaros >/dev/null 2>&1; then
    echo "warning: resolve_madaros using PATH fallback because SOUNIO_MADAROS_ALLOW_PATH_FALLBACK=1" >&2
    command -v madaros
    return 0
  fi
  echo "error: resolve_madaros: no Madaros binary found" >&2
  echo "  tried: MADAROS_BIN, SOUNIO_MADAROS_BIN, $wrapper, $raw_elf (receipt-gated), $relocgate, $prebuilt, PATH opt-in" >&2
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
