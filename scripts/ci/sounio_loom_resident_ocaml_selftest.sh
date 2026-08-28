#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-resident-ocaml.XXXXXX")"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
RECEIPTS="$TEST_ROOT/resident.tsv"
VALID_FRAME="$TEST_ROOT/valid.frame"
PYTHON_FRAME="$TEST_ROOT/python.frame"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-resident-ocaml-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane.sh" >/dev/null
dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null

one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'
printf '%s\n' \
  "9023 3 1 3 10 1 1 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $one $one $one $zero $one $zero $zero" \
  > "$VALID_FRAME"
printf '%s\n' \
  "9023 3 1 3 7 1 1 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $one $one $one $zero $one $zero $zero" \
  > "$PYTHON_FRAME"

probe() {
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_RESIDENT_RECEIPT_LOG="$RECEIPTS" \
    "$LOOM" resident-authority-probe --root "$ROOT_DIR" --mode "$1" \
      --frame "$2" --deadline-ms 5000
}

happy="$(probe happy "$VALID_FRAME")"
[[ "$happy" == LOOM_RESIDENT_OCAML_PROBE\ mode=happy* && \
  "$happy" == *'semantic_authority=Sounio operational_realization=OCaml'* && \
  "$happy" == *'process_identity=stable'* && "$happy" == *'sequence=1 decision_code=0'* && \
  "$happy" == *'poisoned=false' ]] || fail "happy route failed: $happy"

python="$(probe happy "$PYTHON_FRAME")"
[[ "$python" == *'sequence=1 decision_code=410'* && "$python" == *'poisoned=false' ]] ||
  fail "Python decision diverged: $python"

replay="$(probe replay "$VALID_FRAME")"
[[ "$replay" == 'LOOM_RESIDENT_OCAML_PROBE mode=replay semantic_authority=Sounio decision_code=442 poisoned=true reuse_refused=true' ]] ||
  fail "replay did not poison: $replay"

mismatch="$(probe mismatch "$VALID_FRAME")"
[[ "$mismatch" == 'LOOM_RESIDENT_OCAML_PROBE mode=mismatch semantic_authority=Sounio decision_code=443 poisoned=true reuse_refused=true' ]] ||
  fail "mismatch did not poison: $mismatch"

timeout="$(probe timeout "$VALID_FRAME")"
[[ "$timeout" == 'LOOM_RESIDENT_OCAML_PROBE mode=timeout semantic_authority=Sounio refused=true poisoned=true reuse_refused=true' ]] ||
  fail "timeout did not poison: $timeout"

eof="$(probe eof "$VALID_FRAME")"
[[ "$eof" == 'LOOM_RESIDENT_OCAML_PROBE mode=eof semantic_authority=Sounio refused=true poisoned=true reuse_refused=true' ]] ||
  fail "EOF did not poison: $eof"

[[ -s "$RECEIPTS" ]] || fail 'resident receipt log is empty'
grep -Fq $'event=EFFECT\t' "$RECEIPTS" || fail 'effect receipt is missing'
grep -Fq $'event=POISON\t' "$RECEIPTS" || fail 'poison receipt is missing'
grep -Fq 'parent_9023_manifest_sha256=0024178b8928f0c82d794d390244e83e5ce431054587fc7dd609c0f25c2e5b4f' "$RECEIPTS" ||
  fail '9023 parent hash is missing from receipts'
grep -Fq 'parent_9024_manifest_sha256=7ba1917b083b61f6c335c1e64764085d93474dd67896c9dea58516ef9f0e16f2' "$RECEIPTS" ||
  fail '9024 parent hash is missing from receipts'
grep -Fq 'resident_runtime_sha256=5c432f4c56fb0be5c157fb12147566a5f74f2cc4cc1e25b46f37050eff1ac12b' "$RECEIPTS" ||
  fail 'resident runtime hash is missing from receipts'

tampered_manifest="$TEST_ROOT/tampered.runtime.v1"
cp "$ROOT_DIR/tools/loom/resident_membrane.runtime.v1" "$tampered_manifest"
printf '%s\n' 'tamper=1' >> "$tampered_manifest"
set +e
manifest_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_MANIFEST="$tampered_manifest" \
  "$LOOM" resident-authority-probe --root "$ROOT_DIR" --mode happy \
    --frame "$VALID_FRAME" --deadline-ms 5000 2>&1)"
manifest_rc=$?
set -e
[[ "$manifest_rc" -eq 1 && "$manifest_output" == *'resident-runtime-manifest-hash-mismatch'* ]] ||
  fail "tampered manifest was not refused: $manifest_output"

tampered_runtime="$TEST_ROOT/tampered-runtime"
cp "$ROOT_DIR/tools/loom/.runtime/sounio-loom-resident-membrane-runtime" "$tampered_runtime"
printf 'x' >> "$tampered_runtime"
chmod 0755 "$tampered_runtime"
set +e
runtime_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_RUNTIME="$tampered_runtime" \
  "$LOOM" resident-authority-probe --root "$ROOT_DIR" --mode happy \
    --frame "$VALID_FRAME" --deadline-ms 5000 2>&1)"
runtime_rc=$?
set -e
[[ "$runtime_rc" -eq 1 && "$runtime_output" == *'resident-runtime-hash-mismatch'* ]] ||
  fail "tampered runtime was not refused: $runtime_output"

printf '%s\n' \
  'sounio-loom-resident-ocaml-selftest: PASS semantic_authority=Sounio operational_realization=OCaml+resident-Sounio happy=ALLOW python=DENY410 process_identity=stable sequence=correlated replay=DENY442+poison mismatch=DENY443+poison timeout=refused+poison eof=refused+poison reuse=refused receipts=hash-bound manifest_tamper=refused runtime_tamper=refused performance_gate=false membrane_integration=false'
