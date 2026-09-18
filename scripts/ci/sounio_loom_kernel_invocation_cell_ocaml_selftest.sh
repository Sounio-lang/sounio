#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_PARENT="$ROOT_DIR/tools/loom/_build"
mkdir -p "$TEST_PARENT"
TEST_ROOT="$(mktemp -d "$TEST_PARENT/invocation-cell-ocaml.XXXXXX")"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
RUNTIME="$TEST_ROOT/resident-v3"
RECEIPTS="$TEST_ROOT/resident.tsv"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-invocation-cell-ocaml-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

SOUNIO_LOOM_RESIDENT_MEMBRANE_V3_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v3.sh" >/dev/null
(
  flock -x 9
  dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
) 9>"$ROOT_DIR/tools/loom/_build/.dune-build.lock"

parent_9028='1991017987 113822720 1367310835 4264184359 1117900107 2622180275 1259621157 4224578159'
parent_9025='3253784467 4165106381 4153681002 298013982 643434942 312724736 195896759 132696721'
parent_9023='2365323 2301161672 762924345 38070334 1558458629 1166539901 3590963442 1546541903'
one='1 1 1 1 1 1 1 1'
bindings="$parent_9028 $parent_9025 $parent_9023 $one $one $one $one $one $one $one $one $one"
prepare_join='1 1 1 1 1 1'
admit_join='2 2 1 1 1 1'
close_join='3 3 1 1 1 1'
abort_join='4 4 1 1 1 1'
capsule='1 1 5 6 7 1 1 0 0 1 1 1'
membrane='1 8 9 10 11 1 1 1 1'
scope='1 1 1 1 1 1'
coverage='1 100 1 50 1 1 1 1'
open_lifecycle='1 1 1 12 13 1 0 0 0'
close_lifecycle='1 1 1 12 13 1 0 1 0'
abort_lifecycle='1 1 1 12 13 1 0 0 1'
open_outcome='0 0 0 0 0 0 0 0 0 0'
close_outcome='14 1 1 1 1 1 0 0 0 0'
abort_outcome='14 0 1 1 1 1 1 1 1 1'
authority='1 1 1 1 1 1'
python_authority='0 0 1 1 0 1'
evidence='1 1 10 10'

valid_prepare="9029 3 1 $prepare_join $capsule $membrane $scope $coverage $open_lifecycle $open_outcome $authority $evidence $bindings"
valid_admit="9029 3 1 $admit_join $capsule $membrane $scope $coverage $open_lifecycle $open_outcome $authority $evidence $bindings"
valid_close="9029 3 1 $close_join $capsule $membrane $scope $coverage $close_lifecycle $close_outcome $authority $evidence $bindings"
valid_abort="9029 3 1 $abort_join $capsule $membrane $scope $coverage $abort_lifecycle $abort_outcome $authority $evidence $bindings"
current_material="9029 3 1 1 1 0 0 1 0 $capsule $membrane $scope $coverage $open_lifecycle $open_outcome $authority $evidence $bindings"
python_oracle="9029 3 1 $prepare_join $capsule $membrane $scope $coverage $open_lifecycle $open_outcome $python_authority $evidence $bindings"

printf '%s\n' "$valid_prepare" > "$TEST_ROOT/prepare.frame"
printf '%s\n' "$valid_admit" > "$TEST_ROOT/admit.frame"
printf '%s\n' "$valid_close" > "$TEST_ROOT/close.frame"
printf '%s\n' "$valid_abort" > "$TEST_ROOT/abort.frame"
printf '%s\n' "$current_material" > "$TEST_ROOT/current.frame"
printf '%s\n' "$python_oracle" > "$TEST_ROOT/python.frame"

probe() {
  local mode="$1" prepare="$2"
  shift 2
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_RESIDENT_MEMBRANE_V3_RUNTIME="$RUNTIME" \
    SOUNIO_LOOM_RESIDENT_RECEIPT_LOG="$RECEIPTS" \
    "$LOOM" invocation-cell-probe --root "$ROOT_DIR" --mode "$mode" \
      --prepare "$prepare" --deadline-ms 15000 "$@"
}

current_output="$(probe current "$TEST_ROOT/current.frame")"
[[ "$current_output" == *'codes=481 state=UNPREPARED poisoned=false'* && \
  "$current_output" == *'material_invocation=false'* ]] ||
  fail "current material was not refused without state advance: $current_output"

python_output="$(probe python "$TEST_ROOT/python.frame")"
[[ "$python_output" == *'codes=488 state=UNPREPARED poisoned=false'* ]] ||
  fail "Python oracle data was not refused by Sounio: $python_output"

happy_output="$(probe happy "$TEST_ROOT/prepare.frame" \
  --admit "$TEST_ROOT/admit.frame" --close "$TEST_ROOT/close.frame")"
[[ "$happy_output" == *'codes=0,0,0 state=CLOSED poisoned=false'* && \
  "$happy_output" == *'sequence=3 '* ]] ||
  fail "happy lifecycle did not close exactly once: $happy_output"

abort_output="$(probe abort "$TEST_ROOT/prepare.frame" \
  --abort "$TEST_ROOT/abort.frame")"
[[ "$abort_output" == *'codes=0,0 state=POISONED poisoned=true'* ]] ||
  fail "typed abort did not poison the cell: $abort_output"

for mode in replay mismatch timeout eof; do
  output="$(probe "$mode" "$TEST_ROOT/prepare.frame" \
    --admit "$TEST_ROOT/admit.frame")"
  [[ "$output" == *'codes=0 state=POISONED poisoned=true control_refused=true reuse_refused=true'* ]] ||
    fail "$mode did not fail closed and refuse reuse: $output"
done

grep -Fq $'\tevent=INVOCATION_CELL\t' "$RECEIPTS" ||
  fail 'invocation-cell receipt is missing'
grep -Fq $'\tparent_9029_manifest_sha256=61918604bf177753c6141f6cd0f05d342a1869ab8fc08d187306a481de33d70e\t' \
  "$RECEIPTS" || fail 'receipt omitted frozen action 9029'
grep -Fq $'\tresident_manifest_sha256=6d5e8d1fd0d6b3badf707ed438804a9e1b46dc74862e09e2f98d143c40665431\t' \
  "$RECEIPTS" || fail 'receipt omitted frozen resident v3 manifest'

tampered_manifest="$TEST_ROOT/kernel-invocation-cell.freeze.v1"
cp "$ROOT_DIR/tools/loom/kernel_invocation_cell_authority.freeze.v1" \
  "$tampered_manifest"
printf '\n' >> "$tampered_manifest"
set +e
manifest_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V3_RUNTIME="$RUNTIME" \
  SOUNIO_LOOM_KERNEL_INVOCATION_CELL_MANIFEST="$tampered_manifest" \
  "$LOOM" invocation-cell-probe --root "$ROOT_DIR" --mode current \
    --prepare "$TEST_ROOT/current.frame" --deadline-ms 2000 2>&1)"
manifest_rc=$?
set -e
[[ "$manifest_rc" -eq 1 && \
  "$manifest_output" == *'invocation-cell-manifest-hash-mismatch'* ]] ||
  fail "action 9029 manifest tamper did not fail before spawn: $manifest_output"

tampered_resident_manifest="$TEST_ROOT/resident-v3.runtime"
cp "$ROOT_DIR/tools/loom/resident_membrane.runtime.v3" \
  "$tampered_resident_manifest"
printf '\n' >> "$tampered_resident_manifest"
set +e
resident_manifest_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V3_RUNTIME="$RUNTIME" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V3_MANIFEST="$tampered_resident_manifest" \
  "$LOOM" invocation-cell-probe --root "$ROOT_DIR" --mode current \
    --prepare "$TEST_ROOT/current.frame" --deadline-ms 2000 2>&1)"
resident_manifest_rc=$?
set -e
[[ "$resident_manifest_rc" -eq 1 && \
  "$resident_manifest_output" == *'resident-runtime-v3-manifest-hash-mismatch'* ]] ||
  fail "resident v3 manifest tamper did not fail before spawn: $resident_manifest_output"

tampered_runtime="$TEST_ROOT/resident-v3-tampered"
cp "$RUNTIME" "$tampered_runtime"
printf '\n' >> "$tampered_runtime"
chmod 0755 "$tampered_runtime"
set +e
runtime_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V3_RUNTIME="$tampered_runtime" \
  "$LOOM" invocation-cell-probe --root "$ROOT_DIR" --mode current \
    --prepare "$TEST_ROOT/current.frame" --deadline-ms 2000 2>&1)"
runtime_rc=$?
set -e
[[ "$runtime_rc" -eq 1 && \
  "$runtime_output" == *'resident-runtime-hash-mismatch'* ]] ||
  fail "resident v3 runtime tamper did not fail before spawn: $runtime_output"

printf '%s\n' \
  'sounio-loom-kernel-invocation-cell-ocaml-selftest: PASS semantic_authority=Sounio operational_kernel=OCaml resident=Sounio-v3 lifecycle=UNPREPARED-PREPARED-EFFECT_STOPPED-CLOSED-POISONED happy=ALLOWx3 abort=ALLOWx2+POISON current_material=DENY481 python_oracle=DENY488 replay=POISON+REUSE_REFUSED mismatch=POISON+REUSE_REFUSED timeout=POISON+REUSE_REFUSED eof=POISON+REUSE_REFUSED receipts=hash-bound action_manifest_tamper=refused-before-spawn resident_manifest_tamper=refused-before-spawn runtime_tamper=refused-before-spawn material_invocation=false material_coverage=false same_uid_peer_isolation=false exec_attached=false commit_attached=false ci_attached=false'
