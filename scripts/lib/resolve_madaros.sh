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

# An explicit override that is set but unusable is refused, not skipped.
#
# The chain below tests each candidate with -x and moves on when it fails, so
# MADAROS_BIN pointing at a build that left its output non-executable resolves
# silently to the committed prebuilt -- which lags self-hosted/ source. Measured
# 2026-08-28: the same ELF gave "check: OK" at mode 0600 and error[E245] at
# 0700. Fixed in bin/souc and bin/madaros by #2256; this library is the third
# door onto the same defect.
_sounio_refuse_madaros_override() {  # <var-name> <path>
  local var="$1" path="$2" why
  if   [[ ! -e "$path" ]]; then why="no such file"
  elif [[ -d "$path" ]];   then why="is a directory"
  else                          why="not executable (chmod +x it)"; fi
  echo "error: $var is set but cannot be used: $why" >&2
  echo "  $var=$path" >&2
  echo "  refusing to fall back to another compiler: this run would measure a" >&2
  echo "  binary you did not name. Fix the path, or unset $var." >&2
  return 78
}

# Resolve MADAROS_BIN: explicit env → repo wrapper → repo raw ELF → PATH fallback.
_sounio_resolve_madaros_bin() {
  local _v
  for _v in MADAROS_BIN SOUNIO_MADAROS_BIN; do
    if [[ -n "${!_v:-}" && ! -x "${!_v}" ]]; then
      _sounio_refuse_madaros_override "$_v" "${!_v}"
      return 78
    fi
  done
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
