#!/usr/bin/env bash
# scripts/ci/souc-native-wrapper.sh
#
# Subcommand wrapper around the raw Sounio compiler ELF.
#
# The raw `bin/souc` ELF only accepts the positional interface
# `<source.sio> <output>` (plus a few flags: --show-ast, --show-types,
# --r15-monitor, --target). The test harness at
# `scripts/dev/run_sio_test_suite.sh` and the workflow gates at
# `scripts/ci/*.sh` invoke a higher-level subcommand interface:
#
#   <souc> check <file>             typecheck only, no execute
#   <souc> run   <file>             typecheck + compile + execute
#   <souc> compile <file> -o <out>  typecheck + compile to <out>
#   <souc> info                     print this wrapper's resolution path
#   <souc> <file> <out>             raw positional passthrough
#
# This wrapper translates those subcommands to the raw ELF interface and
# patches two harness-visible behaviors that the raw ELF does not perform
# on its own:
#
#   1. Exit code: the raw ELF exits 0 even when it fails to produce an
#      output ELF (silent lexer/parser failure on bad source). This
#      wrapper detects "no ELF produced" and exits 1, which the harness
#      needs to classify compile-fail tests.
#
#   2. Diagnostic marker: the harness greps the compiler output for the
#      literal string `typecheck: failed` to disambiguate
#      "compile succeeded" from "compile failed but the compiler still
#      exited 0" (line 292 of run_sio_test_suite.sh). The raw ELF does
#      not emit this token; this wrapper appends it on failure.
#
# Environment:
#   SOUNIO_STDLIB_PATH  — stdlib path forwarded to the raw ELF unchanged
#   SOUC_NATIVE_BIN     — override the underlying raw ELF (defaults to
#                         ROOT_DIR/bin/souc)
#   SOUNIO_SOUC_BIN     — same as SOUC_NATIVE_BIN; honored for symmetry
#                         with the ontology-validation wrapper protocol
#                         documented in scripts/dev/run_sio_test_suite.sh
#
# Selection by the harness:
#   The harness selects this wrapper when SOUNIO_TEST_SOUC_BIN points at
#   a file whose first two bytes are `#!`. Example:
#       export SOUNIO_TEST_SOUC_BIN="$ROOT_DIR/scripts/ci/souc-native-wrapper.sh"
#       bash scripts/run_sio_test_suite.sh --filter pbpk

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Resolve the underlying raw ELF.
#
# Resolution order:
#   1. SOUNIO_SOUC_BIN — the canonical name in the ontology-validation
#      wrapper protocol, but ONLY if it points to a non-wrapper ELF
#      (i.e. does not start with `#!`). When the harness routes through
#      us, it sets SOUNIO_SOUC_BIN to this wrapper's own path, which we
#      must skip to avoid infinite recursion.
#   2. SOUC_NATIVE_BIN — older alias, same shebang-skip rule.
#   3. ROOT_DIR/bin/souc — the checked-in raw ELF (default).
#
# The shebang-skip rule is what makes the wrapper composable: callers
# can pass us through `SOUNIO_SOUC_BIN` and we'll still find the raw
# ELF underneath.
_resolve_raw_elf() {
  local cand
  for cand in "${SOUNIO_SOUC_BIN:-}" "${SOUC_NATIVE_BIN:-}" ""; do
    if [[ -n "$cand" && -x "$cand" ]] \
       && [[ "$(head -c 2 "$cand" 2>/dev/null)" != "#!" ]]; then
      echo "$cand"
      return 0
    fi
  done
  if [[ -x "$ROOT_DIR/bin/souc" ]]; then
    echo "$ROOT_DIR/bin/souc"
    return 0
  fi
  return 1
}

if ! RAW_SOUC="$(_resolve_raw_elf)"; then
  echo "error: souc-native-wrapper: no raw ELF found" >&2
  echo "  tried: SOUNIO_SOUC_BIN (skipped if wrapper), SOUC_NATIVE_BIN (skipped if wrapper), $ROOT_DIR/bin/souc" >&2
  exit 127
fi

# Madaros's own type-checker/lowerer needs a large stack for real
# multi-module programs (deep recursion during typecheck/lowering); the
# default 8 MiB shell stack ulimit segfaults (rc=139) partway through
# lowering a program with even one cross-module import (reproduced
# directly: bare `raw_elf src -o out` on a trivial two-file `use mod::fn`
# program crashes under the default ulimit and compiles cleanly once the
# stack limit is raised — see self-hosted/compiler/module_frontend.sio's
# `lower_array: seed_begin` trace, where it dies). bin/madaros already
# raises this before invoking its own raw ELF (MADAROS_STACK_KB, default
# 524288 KiB) — this wrapper must do the same, since CI and the test
# harness invoke the raw ELF directly through it, bypassing bin/madaros
# entirely.
MADAROS_STACK_KB="${MADAROS_STACK_KB:-524288}"
if [[ "$MADAROS_STACK_KB" == "0" ]]; then
  ulimit -s unlimited 2>/dev/null || true
else
  ulimit -s "$MADAROS_STACK_KB" 2>/dev/null || true
fi

_detect_raw_mode() {
  if [[ -n "${SOUNIO_SOUC_RAW_MODE:-}" ]]; then
    echo "$SOUNIO_SOUC_RAW_MODE"
    return 0
  fi
  local base
  base="$(basename "$RAW_SOUC")"
  if [[ "$base" == *madaros* ]]; then
    echo "modular"
    return 0
  fi
  local ident
  ident="$("$RAW_SOUC" /dev/null /dev/null 2>/dev/null | head -n 1 || true)"
  if grep -q "Madaros v" <<<"$ident"; then
    echo "modular"
    return 0
  fi
  echo "legacy"
}

RAW_MODE="$(_detect_raw_mode)"

usage() {
  cat <<USAGE
souc-native-wrapper.sh — subcommand wrapper around $RAW_SOUC

Usage:
  $(basename "$0") check <file.sio>             typecheck only
  $(basename "$0") run   <file.sio>             compile + execute
  $(basename "$0") compile <file.sio> -o <out>  compile to <out>
  $(basename "$0") info                          print resolution path
  $(basename "$0") <file.sio> <out>              raw positional passthrough

Environment:
  SOUNIO_STDLIB_PATH=<dir>   forwarded to raw ELF
  SOUNIO_SOUC_BIN=<path>     override raw ELF
  SOUC_NATIVE_BIN=<path>     override raw ELF (alias)
USAGE
}

info() {
  echo "wrapper:   $0"
  echo "raw_elf:   $RAW_SOUC"
  echo "raw_mode:  $RAW_MODE"
  echo "stdlib:    ${SOUNIO_STDLIB_PATH:-<unset>}"
}

# Compile $1 (source) to $2 (output ELF), capture stdout+stderr, return 0
# only when the ELF was actually produced, is non-empty, and the raw ELF
# emitted its real-success marker (`compile: fns=...`). The raw ELF
# emits an "error: no main" or "typecheck: failed" line for some failure
# modes and a stub ~35 kB ELF for other failure modes (e.g. partial
# parse) without a corresponding `compile: fns=...` line. On any of
# those outcomes, append `typecheck: failed` to the captured output
# (matches the harness grep at line 292 of run_sio_test_suite.sh) and
# return 1.
souc_compile() {
  local src="$1"
  local out="$2"
  local log
  if [[ "$RAW_MODE" == "modular" ]]; then
    log="$("$RAW_SOUC" "$src" -o "$out" 2>&1)" || true
    if [[ ! -s "$out" ]]; then
      printf '%s\ntypecheck: failed\n' "$log"
      return 1
    fi
    printf '%s' "$log"
    return 0
  fi
  log="$("$RAW_SOUC" "$src" "$out" 2>&1)" || true
  if [[ ! -s "$out" ]] || ! grep -qF "compile: fns=" <<<"$log"; then
    printf '%s\ntypecheck: failed\n' "$log"
    return 1
  fi
  printf '%s' "$log"
  return 0
}

# Subcommand dispatch. Default: pass through to the raw ELF
# (e.g. `bin/souc <file> <out>`).
case "${1:-}" in
  check)
    shift
    if [[ $# -ne 1 ]]; then
      echo "error: check takes exactly 1 argument (file.sio)" >&2
      usage >&2
      exit 2
    fi
    if [[ "$RAW_MODE" == "modular" ]]; then
      set +e
      "$RAW_SOUC" --check "$1"
      rc=$?
      set -e
      exit "$rc"
    fi
    tmp="$(mktemp /tmp/sounio-check-XXXXXX.elf)"
    trap 'rm -f "$tmp"' EXIT
    souc_compile "$1" "$tmp"
    rc=$?
    rm -f "$tmp"
    trap - EXIT
    exit "$rc"
    ;;

  run)
    shift
    if [[ $# -ne 1 ]]; then
      echo "error: run takes exactly 1 argument (file.sio)" >&2
      usage >&2
      exit 2
    fi
    tmp="$(mktemp /tmp/sounio-run-XXXXXX.elf)"
    trap 'rm -f "$tmp"' EXIT
    set +e
    souc_compile "$1" "$tmp"
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      rm -f "$tmp"
      trap - EXIT
      exit "$rc"
    fi
    chmod +x "$tmp"
    set +e
    "$tmp"
    rc=$?
    set -e
    rm -f "$tmp"
    trap - EXIT
    exit "$rc"
    ;;

  compile)
    shift
    src=""
    out=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        -o)
          shift
          [[ $# -ge 1 ]] || { echo "error: -o requires an argument" >&2; exit 2; }
          out="$1"
          shift
          ;;
        -*)
          echo "error: unknown flag: $1" >&2
          exit 2
          ;;
        *)
          if [[ -z "$src" ]]; then
            src="$1"
          else
            echo "error: compile takes one source and one -o <out>" >&2
            exit 2
          fi
          shift
          ;;
      esac
    done
    if [[ -z "$src" || -z "$out" ]]; then
      echo "error: compile requires <file.sio> -o <out>" >&2
      usage >&2
      exit 2
    fi
    souc_compile "$src" "$out"
    ;;

  info)
    info
    ;;

  -h|--help|help|"")
    usage
    ;;

  *)
    # Positional passthrough: `bin/souc <file> <out>`.
    exec "$RAW_SOUC" "$@"
    ;;
esac
