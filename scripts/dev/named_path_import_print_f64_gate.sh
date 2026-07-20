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
# PASS = check rc=0, compile+run rc=0, stdout contains 1.500000.
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

# Default (non -O) compile is the acceptance path. The -O cleanup lane currently
# SIGSEGVs after lower for this multi-module shape — that is a separate optimizer
# defect, not the named-import/E137 subject of this gate.
echo "=== path-form named import compile + run (default native) ==="
OUT="$SCRATCH_DIR/named_path_import.elf"
"$RAW" "$MAIN_SRC" -o "$OUT" >"$SCRATCH_DIR/cc.stdout" 2>"$SCRATCH_DIR/cc.stderr"
CC_RC=$?
if [[ $CC_RC -ne 0 ]]; then
  echo "NAMED_PATH_IMPORT_GATE_BLOCKED reason=compile_fail cc_rc=$CC_RC" >&2
  tail -15 "$SCRATCH_DIR/cc.stderr" >&2
  exit 1
fi
if [[ ! -s "$OUT" ]]; then
  echo "NAMED_PATH_IMPORT_GATE_BLOCKED reason=no_elf_emitted" >&2
  exit 1
fi

chmod +x "$OUT"
"$OUT" >"$SCRATCH_DIR/run.out" 2>/dev/null
RUN_RC=$?
if [[ $RUN_RC -ne 0 ]]; then
  echo "NAMED_PATH_IMPORT_GATE_BLOCKED reason=run_fail run_rc=$RUN_RC" >&2
  exit 1
fi

if ! grep -q '1\.500000' "$SCRATCH_DIR/run.out"; then
  echo "NAMED_PATH_IMPORT_GATE_BLOCKED reason=stdout_missing_expected_value" >&2
  cat "$SCRATCH_DIR/run.out" >&2
  exit 1
fi

echo "NAMED_PATH_IMPORT_GATE_PASS raw_sha256=${RAW_SHA256:0:16} stdout=$(tr -d '\n' < "$SCRATCH_DIR/run.out")"
exit 0
