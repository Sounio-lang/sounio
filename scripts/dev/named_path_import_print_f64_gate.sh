#!/usr/bin/env bash
# scripts/dev/named_path_import_print_f64_gate.sh
#
# Regression gate for path-form named imports under multi-module Madaros.
#
# Covers the residual D4 papercut after #1239 (visibility builtin allow-list):
# bare `use module::symbol` (no braces) combined with print_f64 and a local
# helper function. Pre-fix: AST closure treated the symbol as a path segment
# (mod/half.sio missing) → E137 / incomplete closure. Brace form
# `use mod::{half}` already worked.
#
# Also covers multi-module -O full peels (Wave9): opt_cleanup_module_inplace
# must not SEGV after lower_done and must still print 1.500000, plus integer
# const-fold surface 42 and SCCP-lite branch surface 7. Root cause was by-value
# IrFunction copies (Box/call_args, A8 family) plus miscompiled
# `&! functions[fi]` array-element exclusive refs; cleanup mutates via
# module+fi field peels only (dedup/copy/DSE/control/CSE/DCE + const-fold +
# SCCP-lite + compact_nops).
#
# PASS = check rc=0, default compile+run rc=0 with 1.500000/42/7, -O same.
# See docs/audit/MADAROS_MULTIMODULE_PRINT_IMPORT_BUGS_2026-07-13.md.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

FIXTURE_DIR="$REPO_ROOT/tests/compiler/named_path_import_print_f64_gate"
MAIN_SRC="$FIXTURE_DIR/main.sio"
SCRATCH_DIR="$(mktemp -d /tmp/named-path-import-gate.XXXXXX)"
trap 'rm -rf "$SCRATCH_DIR"' EXIT

log()  { printf '[gate] %s\n' "$*" >&2; }

RAW=""
for cand in "$REPO_ROOT/artifacts/self-hosted/madaros" "$REPO_ROOT/bin/madaros-linux-x86_64"; do
  if [[ -x "$cand" && "$(head -c2 "$cand" 2>/dev/null)" != '#!' ]]; then
    RAW="$cand"; break
  fi
done

if [[ -z "$RAW" ]]; then
  echo "NAMED_PATH_IMPORT_GATE_BLOCKED reason=no_raw_madaros" >&2
  exit 1
fi

RAW_SHA256="$(sha256sum "$RAW" | awk '{print $1}')"
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "raw_elf=$RAW"
log "raw_elf_sha256=$RAW_SHA256"
log "git_sha=$GIT_SHA"

echo "=== path-form named import check ==="
"$RAW" --check "$MAIN_SRC" >"$SCRATCH_DIR/check.stdout" 2>"$SCRATCH_DIR/check.stderr"
CHECK_RC=$?
if [[ $CHECK_RC -ne 0 ]]; then
  echo "NAMED_PATH_IMPORT_GATE_BLOCKED reason=check_failed check_rc=$CHECK_RC" >&2
  tail -15 "$SCRATCH_DIR/check.stderr" >&2
  exit 1
fi
if grep -qE 'error\[E[0-9]+\]|AST closure incomplete' "$SCRATCH_DIR/check.stdout" "$SCRATCH_DIR/check.stderr"; then
  echo "NAMED_PATH_IMPORT_GATE_BLOCKED reason=check_emitted_error_or_incomplete_closure" >&2
  tail -15 "$SCRATCH_DIR/check.stderr" >&2
  exit 1
fi
log "check: PASS"

run_compile_and_execute() {
  local label="$1"
  shift
  local out="$SCRATCH_DIR/${label}.elf"
  local cc_stdout="$SCRATCH_DIR/${label}.cc.stdout"
  local cc_stderr="$SCRATCH_DIR/${label}.cc.stderr"
  local run_out="$SCRATCH_DIR/${label}.run.out"

  echo "=== path-form named import compile + run (${label}) ==="
  "$RAW" "$@" "$MAIN_SRC" -o "$out" >"$cc_stdout" 2>"$cc_stderr"
  local cc_rc=$?
  if [[ $cc_rc -ne 0 ]]; then
    echo "NAMED_PATH_IMPORT_GATE_BLOCKED reason=compile_fail label=${label} cc_rc=$cc_rc" >&2
    tail -20 "$cc_stderr" >&2
    exit 1
  fi
  if [[ ! -s "$out" ]]; then
    echo "NAMED_PATH_IMPORT_GATE_BLOCKED reason=no_elf_emitted label=${label}" >&2
    tail -20 "$cc_stderr" >&2
    exit 1
  fi

  chmod +x "$out"
  "$out" >"$run_out" 2>/dev/null
  local run_rc=$?
  if [[ $run_rc -ne 0 ]]; then
    echo "NAMED_PATH_IMPORT_GATE_BLOCKED reason=run_fail label=${label} run_rc=$run_rc" >&2
    exit 1
  fi

  if ! grep -q '1\.500000' "$run_out"; then
    echo "NAMED_PATH_IMPORT_GATE_BLOCKED reason=stdout_missing_expected_value label=${label}" >&2
    cat "$run_out" >&2
    exit 1
  fi
  # Const-fold surface (folded_sum → 42) and SCCP-lite branch surface (7).
  local line2 line3
  line2="$(sed -n '2p' "$run_out" | tr -d '\r')"
  line3="$(sed -n '3p' "$run_out" | tr -d '\r')"
  if [[ "$line2" != "42" ]]; then
    echo "NAMED_PATH_IMPORT_GATE_BLOCKED reason=stdout_missing_constfold_42 label=${label} line2=${line2}" >&2
    cat "$run_out" >&2
    exit 1
  fi
  if [[ "$line3" != "7" ]]; then
    echo "NAMED_PATH_IMPORT_GATE_BLOCKED reason=stdout_missing_sccp_7 label=${label} line3=${line3}" >&2
    cat "$run_out" >&2
    exit 1
  fi

  log "${label}: PASS stdout=$(tr '\n' '|' < "$run_out")"
}

# Default (no -O) must stay green.
run_compile_and_execute "default"

# -O multi-module opt_cleanup full peels: 1.500000 + constfold 42 + sccp 7.
run_compile_and_execute "opt" -O

echo "NAMED_PATH_IMPORT_GATE_PASS raw_sha256=${RAW_SHA256:0:16} default+opt full_peels"
exit 0
