#!/usr/bin/env bash
# A `&Seq<T>` parameter must deliver the Seq.
#
# Seq methods are intrinsics taking the handle BY VALUE, but a `&Seq<T>`
# receiver holds the ADDRESS of the handle slot. lower.sio passed that address
# straight through, so the callee read one indirection level off while the
# program type-checked clean: `.len()` returned a static address, `.get(i)` a
# stack address, and a loop over both segfaulted.
#
# Reachable only since the checker began unwrapping a TyRef receiver to its
# TyNamed pointee (PR #2413); before that the call was refused with E019. The
# type-level unwrap shipped without its lowering-level counterpart, so a clean
# rejection became silent wrong code. This gate is what stops that recurring.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# Always pin this worktree's stdlib (never inherit a foreign SOUNIO_STDLIB_PATH).
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="tests/run-pass/seq_ref_param_delivers_handle.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
ELF="$OUT/seq_ref_param.elf"

echo "== madaros_seq_ref_param_gate =="

if ! "$SOUC" compile "$SRC" -o "$ELF" >"$OUT/compile.log" 2>&1; then
  echo "FAIL: compile"
  tail -40 "$OUT/compile.log" || true
  exit 1
fi
chmod +x "$ELF"

LOG="$OUT/run.log"
# Capture rc BEFORE anything else runs: `$?` inside an `if !` body is the `if`'s
# own status, not the command's, and reading it there reports 0 for a crash.
set +e
"$ELF" >"$LOG" 2>&1
RUN_RC=$?
set -e
if [ "$RUN_RC" -ne 0 ]; then
  # rc 139 here is the original defect: the accumulating loop segfaulted.
  echo "FAIL: run (rc $RUN_RC)"
  cat "$LOG" || true
  exit 1
fi

grep -q 'SEQ_REF_PARAM_OK' "$LOG" || {
  echo "FAIL: missing sentinel"
  cat "$LOG" || true
  exit 1
}

echo "MADAROS_SEQ_REF_PARAM_GATE_OK"
