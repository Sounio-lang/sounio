#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_PARENT="$ROOT_DIR/tools/loom/_build"
mkdir -p "$TEST_PARENT"
TEST_ROOT="$(mktemp -d "$TEST_PARENT/effect-closure-ocaml.XXXXXX")"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
MEMBRANE_RUNTIME="$TEST_ROOT/subprocess-membrane"
RESIDENT_V2_RUNTIME="$TEST_ROOT/resident-membrane-v2"
SCOPE="$TEST_ROOT/scope"
DECISIONS="$TEST_ROOT/decisions.tsv"
RECEIPTS="$TEST_ROOT/resident.tsv"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-effect-closure-ocaml-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

SOUNIO_LOOM_SUBPROCESS_MEMBRANE_OUTPUT="$MEMBRANE_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_subprocess_membrane.sh" >/dev/null
SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_OUTPUT="$RESIDENT_V2_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v2.sh" >/dev/null
(
  flock -x 9
  dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
) 9>"$ROOT_DIR/tools/loom/_build/.dune-build.lock"
mkdir -p "$SCOPE"

probe() {
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_SUBPROCESS_MEMBRANE_RUNTIME="$MEMBRANE_RUNTIME" \
    SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_RUNTIME="$RESIDENT_V2_RUNTIME" \
    SOUNIO_LOOM_SUBPROCESS_MEMBRANE_LOG="$DECISIONS" \
    SOUNIO_LOOM_RESIDENT_RECEIPT_LOG="$RECEIPTS" \
    "$LOOM" subprocess-membrane-probe --root "$ROOT_DIR" --cwd "$ROOT_DIR" \
      --scope "$SCOPE" --deadline-ms 15000 -- /usr/bin/true
}

output="$(probe)"
[[ "$output" == *'kind=1 exit=0'* && "$output" == *'decision_code=0'* ]] ||
  fail "diagnostic effect did not complete: $output"
[[ "$output" == *'authority=resident-Sounio-v2'* && \
  "$output" == *'closure_authority=Sounio'* && \
  "$output" == *'closure_code=447'* && \
  "$output" == *'closure_material=refused'* && \
  "$output" == *'attachment=refused'* ]] ||
  fail "current material closure was not refused: $output"
[[ "$output" != *'closure_code=0'* ]] || fail 'diagnostic coverage was promoted to closure'
grep -Fq $'\tevent=EFFECT_CLOSURE\t' "$RECEIPTS" || fail 'closure receipt is missing'
grep -Fq $'\tcode=447\t' "$RECEIPTS" || fail 'closure receipt omitted DENY447'
grep -Fq $'\tparent_9025_manifest_sha256=c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91\t' \
  "$RECEIPTS" || fail 'closure receipt omitted the frozen action 9025 manifest'
grep -Fq $'\tresident_runtime_sha256=da1de5041588f722e5b1904af3d13ac435a29e7bc254ec6b0a5df375116b0b44\t' \
  "$RECEIPTS" || fail 'closure receipt omitted the frozen resident v2 runtime'

runtime_sentinel="$TEST_ROOT/runtime-tamper-executed"
set +e
runtime_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_RUNTIME="$MEMBRANE_RUNTIME" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_RUNTIME=/usr/bin/true \
  "$LOOM" subprocess-membrane-probe --root "$ROOT_DIR" --cwd "$ROOT_DIR" \
    --scope "$SCOPE" --deadline-ms 2000 -- /bin/sh -c \
    "printf BAD > '$runtime_sentinel'" 2>&1)"
runtime_rc=$?
set -e
[[ "$runtime_rc" -eq 1 && "$runtime_output" == *'resident-runtime-hash-mismatch'* ]] ||
  fail "resident v2 runtime tamper did not fail closed: rc=$runtime_rc output=$runtime_output"
[[ ! -e "$runtime_sentinel" ]] || fail 'runtime tamper executed the child'

tampered_manifest="$TEST_ROOT/resident-v2.runtime"
cp "$ROOT_DIR/tools/loom/resident_membrane.runtime.v2" "$tampered_manifest"
printf '\n' >> "$tampered_manifest"
manifest_sentinel="$TEST_ROOT/manifest-tamper-executed"
set +e
manifest_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_RUNTIME="$MEMBRANE_RUNTIME" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_RUNTIME="$RESIDENT_V2_RUNTIME" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_MANIFEST="$tampered_manifest" \
  "$LOOM" subprocess-membrane-probe --root "$ROOT_DIR" --cwd "$ROOT_DIR" \
    --scope "$SCOPE" --deadline-ms 2000 -- /bin/sh -c \
    "printf BAD > '$manifest_sentinel'" 2>&1)"
manifest_rc=$?
set -e
[[ "$manifest_rc" -eq 1 && \
  "$manifest_output" == *'resident-runtime-v2-manifest-hash-mismatch'* ]] ||
  fail "resident v2 manifest tamper did not fail closed: rc=$manifest_rc output=$manifest_output"
[[ ! -e "$manifest_sentinel" ]] || fail 'manifest tamper executed the child'

printf '%s\n' \
  'sounio-loom-effect-closure-ocaml-selftest: PASS semantic_authority=Sounio operational_realization=OCaml+resident-Sounio-v2 current_material=DENY447 validation=pre-effect receipt=EFFECT_CLOSURE+hash-bound runtime_tamper=refused-before-spawn manifest_tamper=refused-before-spawn diagnostic_effect=completed attachment=refused material_coverage=false'
