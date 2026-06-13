#!/usr/bin/env bash
# Gate for promoting a candidate as bin/madaros-linux-x86_64.
#
# This is intentionally separate from project_spine_gate.sh: it validates a
# candidate binary before swapping it into the worktree.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CANDIDATE="${MADAROS_CANDIDATE:-$ROOT_DIR/bin/madaros-linux-x86_64}"
IMPORTS="${MADAROS_IMPORTS_CANDIDATE:-$ROOT_DIR/bin/madaros-imports-linux-x86_64}"
TMP_DIR="$(mktemp -d /tmp/sounio-madaros-main-candidate.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
  echo "[madaros-main-candidate] FAIL: $*" >&2
  exit 1
}

[[ -x "$CANDIDATE" ]] || fail "candidate is not executable: $CANDIDATE"
[[ -x "$IMPORTS" ]] || fail "imports candidate is not executable: $IMPORTS"

echo "[madaros-main-candidate] 1/4 identity"
IDENTITY_OUT="$("$CANDIDATE" --version 2>&1)" || fail "candidate --version failed"
[[ "$IDENTITY_OUT" == *"Madares"* || "$IDENTITY_OUT" == *"Madaros"* ]] || {
  printf '%s\n' "$IDENTITY_OUT" >&2
  fail "candidate does not identify as Madaros/Madares"
}

cat >"$TMP_DIR/native_v2_42.sio" <<'SRC'
fn main() -> i64 {
    42
}
SRC

echo "[madaros-main-candidate] 2/4 source native-v2 returns 42"
"$CANDIDATE" --native-v2-compile "$TMP_DIR/native_v2_42.sio" -o "$TMP_DIR/native_v2_42.elf" >/tmp/madaros_main_candidate_native_v2.log 2>&1 || {
  cat /tmp/madaros_main_candidate_native_v2.log >&2
  fail "candidate --native-v2-compile failed"
}
[[ -s "$TMP_DIR/native_v2_42.elf" ]] || {
  cat /tmp/madaros_main_candidate_native_v2.log >&2
  fail "candidate --native-v2-compile produced no ELF"
}
chmod +x "$TMP_DIR/native_v2_42.elf"
set +e
"$TMP_DIR/native_v2_42.elf"
NATIVE_V2_RC=$?
set -e
[[ "$NATIVE_V2_RC" -eq 42 ]] || {
  cat /tmp/madaros_main_candidate_native_v2.log >&2
  fail "candidate native-v2 source ELF expected exit 42, got $NATIVE_V2_RC"
}

echo "[madaros-main-candidate] 3/4 backend native-v2 scalar returns 42"
"$CANDIDATE" --native-v2-emit-scalar 42 "$TMP_DIR/emit_scalar_42.elf" >/tmp/madaros_main_candidate_emit_scalar.log 2>&1 || {
  cat /tmp/madaros_main_candidate_emit_scalar.log >&2
  fail "candidate --native-v2-emit-scalar failed"
}
[[ -s "$TMP_DIR/emit_scalar_42.elf" ]] || {
  cat /tmp/madaros_main_candidate_emit_scalar.log >&2
  fail "candidate --native-v2-emit-scalar produced no ELF"
}
chmod +x "$TMP_DIR/emit_scalar_42.elf"
set +e
"$TMP_DIR/emit_scalar_42.elf"
EMIT_RC=$?
set -e
[[ "$EMIT_RC" -eq 42 ]] || {
  cat /tmp/madaros_main_candidate_emit_scalar.log >&2
  fail "candidate emit-scalar ELF expected exit 42, got $EMIT_RC"
}

echo "[madaros-main-candidate] 4/4 imports full modular aggregate fallback"
"$IMPORTS" --native-compile tests/multimodule/abi_name_many_args_main.sio -o "$TMP_DIR/imported_aggregate.elf" >/tmp/madaros_main_candidate_imports.log 2>&1 || {
  cat /tmp/madaros_main_candidate_imports.log >&2
  fail "imports candidate --native-compile aggregate failed"
}
[[ -s "$TMP_DIR/imported_aggregate.elf" ]] || {
  cat /tmp/madaros_main_candidate_imports.log >&2
  fail "imports candidate aggregate produced no ELF"
}
if ! grep -q 'falling back to full IR path' /tmp/madaros_main_candidate_imports.log; then
  cat /tmp/madaros_main_candidate_imports.log >&2
  fail "imports candidate aggregate did not exercise full IR fallback"
fi
if grep -q 'stage0 full compiler' /tmp/madaros_main_candidate_imports.log; then
  cat /tmp/madaros_main_candidate_imports.log >&2
  fail "imports candidate aggregate used stage0 rescue"
fi
chmod +x "$TMP_DIR/imported_aggregate.elf"
set +e
"$TMP_DIR/imported_aggregate.elf"
AGG_RC=$?
set -e
[[ "$AGG_RC" -eq 0 ]] || {
  cat /tmp/madaros_main_candidate_imports.log >&2
  fail "imports candidate aggregate expected exit 0, got $AGG_RC"
}

echo "MADAROS_MAIN_CANDIDATE_PASS"
