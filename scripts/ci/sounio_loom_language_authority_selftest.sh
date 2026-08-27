#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-language-authority-selftest.XXXXXX")"
RUNTIME="$TEST_ROOT/sounio-language-authority"
SABOTAGED="$TEST_ROOT/sounio-language-authority-sabotaged"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-language-authority-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

run_frame() {
  local runtime="$1" frame="$2"
  set +e
  FRAME_OUTPUT="$(printf '%s\n' "$frame" | "$runtime" 2>&1)"
  FRAME_RC=$?
  set -e
}

SOUNIO_LOOM_LANGUAGE_AUTHORITY_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_language_authority.sh" >/dev/null

selftest="$(printf '0\n' | "$RUNTIME")"
[[ "$selftest" == 'SOUNIO_LANGUAGE_AUTHORITY_SELFTEST PASS cases=33' ]] ||
  fail "Sounio-owned selftest did not pass: $selftest"

one='1 1 1 1 1 1 1 1'
two='2 2 2 2 2 2 2 2'
zero='0 0 0 0 0 0 0 0'

# schema stage action language role policy semantic_write expected_write
# parity_valid review_promoted exception waiver{founder,scope,purpose,live}
# guardian{transitional,declarative,fixture}, then eight SHA-256 limb vectors.
python_attempt="9020 3 4 7 7 1 0 0 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $zero $zero"
run_frame "$RUNTIME" "$python_attempt"
[[ "$FRAME_RC" -eq 110 ]] || fail "Python oracle attempt returned $FRAME_RC instead of 110: $FRAME_OUTPUT"
[[ "$FRAME_OUTPUT" == *'reason=forbidden-language'* ]] ||
  fail "Python oracle refusal omitted its reason: $FRAME_OUTPUT"

rust_attempt="9020 3 4 8 7 1 0 0 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $zero $zero"
run_frame "$RUNTIME" "$rust_attempt"
[[ "$FRAME_RC" -eq 110 && "$FRAME_OUTPUT" == *'reason=forbidden-language'* ]] ||
  fail "Rust oracle attempt was not refused: rc=$FRAME_RC output=$FRAME_OUTPUT"

missing_policy="9020 3 4 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $zero $zero"
run_frame "$RUNTIME" "$missing_policy"
[[ "$FRAME_RC" -eq 101 && "$FRAME_OUTPUT" == *'reason=policy-missing'* ]] ||
  fail "missing policy did not fail closed: rc=$FRAME_RC output=$FRAME_OUTPUT"

llm_promotion="9020 1 5 6 6 1 0 0 0 1 0 0 0 0 0 0 0 0 $zero $zero $zero $zero $zero $zero $zero $zero"
run_frame "$RUNTIME" "$llm_promotion"
[[ "$FRAME_RC" -eq 119 && "$FRAME_OUTPUT" == *'reason=review-promoted-to-authority'* ]] ||
  fail "LLM authority promotion was not refused: rc=$FRAME_RC output=$FRAME_OUTPUT"

cpp_guard="9020 3 6 4 4 1 0 0 0 0 0 0 0 0 0 1 1 1 $one $one $one $one $one $one $zero $zero"
run_frame "$RUNTIME" "$cpp_guard"
[[ "$FRAME_RC" -eq 0 && "$FRAME_OUTPUT" == SOUNIO_LANGUAGE_AUTHORITY_ALLOW* ]] ||
  fail "transitional declarative C++ guardian was not admitted: rc=$FRAME_RC output=$FRAME_OUTPUT"

ocaml_realization="9020 3 12 9 8 1 0 0 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $zero $zero"
run_frame "$RUNTIME" "$ocaml_realization"
[[ "$FRAME_RC" -eq 0 && "$FRAME_OUTPUT" == SOUNIO_LANGUAGE_AUTHORITY_ALLOW* ]] ||
  fail "OCaml operational realization was not admitted after freeze: rc=$FRAME_RC output=$FRAME_OUTPUT"

ocaml_prefreeze="9020 2 12 9 8 1 0 0 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $zero $zero"
run_frame "$RUNTIME" "$ocaml_prefreeze"
[[ "$FRAME_RC" -eq 112 && "$FRAME_OUTPUT" == *'reason=wrong-stage'* ]] ||
  fail "OCaml operational realization was not refused before freeze: rc=$FRAME_RC output=$FRAME_OUTPUT"

ocaml_wrong_parent="9020 3 12 9 8 1 0 0 0 0 0 0 0 0 0 0 0 0 $one $one $two $one $one $one $zero $zero"
run_frame "$RUNTIME" "$ocaml_wrong_parent"
[[ "$FRAME_RC" -eq 117 && "$FRAME_OUTPUT" == *'reason=parent-semantics-hash-mismatch'* ]] ||
  fail "OCaml parent-hash laundering was not refused: rc=$FRAME_RC output=$FRAME_OUTPUT"

ocaml_guard="9020 3 6 9 8 1 0 0 0 0 0 0 0 0 0 1 1 1 $one $one $one $one $one $one $zero $zero"
run_frame "$RUNTIME" "$ocaml_guard"
[[ "$FRAME_RC" -eq 0 && "$FRAME_OUTPUT" == SOUNIO_LANGUAGE_AUTHORITY_ALLOW* ]] ||
  fail "transitional declarative OCaml guardian was not admitted: rc=$FRAME_RC output=$FRAME_OUTPUT"

ocaml_parity="9020 3 4 9 8 1 0 0 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $zero $zero"
run_frame "$RUNTIME" "$ocaml_parity"
[[ "$FRAME_RC" -eq 123 && "$FRAME_OUTPUT" == *'reason=action-forbidden-for-role'* ]] ||
  fail "OCaml operational role was allowed to execute parity: rc=$FRAME_RC output=$FRAME_OUTPUT"

wrong_parent="9020 3 4 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 $one $one $two $one $one $one $zero $zero"
run_frame "$RUNTIME" "$wrong_parent"
[[ "$FRAME_RC" -eq 117 && "$FRAME_OUTPUT" == *'reason=parent-semantics-hash-mismatch'* ]] ||
  fail "parity parent laundering was not refused: rc=$FRAME_RC output=$FRAME_OUTPUT"

malformed='9020 3'
run_frame "$RUNTIME" "$malformed"
[[ "$FRAME_RC" -eq 124 && "$FRAME_OUTPUT" == *'reason=malformed-frame'* ]] ||
  fail "malformed policy frame did not fail closed: rc=$FRAME_RC output=$FRAME_OUTPUT"

# Causal control: remove only the Sounio-owned Python prohibition, rebuild the
# same Sounio program, and replay the unchanged Python frame. The refusal must
# disappear. This proves the named Sounio rule, not parser noise, caused E110.
module="$ROOT_DIR/stdlib/coordination/loom_language_authority.sio"
entrypoint="$ROOT_DIR/tools/loom/language_authority_main.sio"
needle='if language == 7 || language == 8 { return 110 }'
[[ "$(grep -Fc "$needle" "$module")" -eq 1 ]] ||
  fail 'Python prohibition sabotage point is not unique'
sed "s/$needle/if language == 8 { return 110 }/" "$module" > "$TEST_ROOT/module-sabotaged.sio"
sed -n '1,$p' "$TEST_ROOT/module-sabotaged.sio" "$entrypoint" > "$TEST_ROOT/runtime-sabotaged.sio"
SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" compile \
  "$TEST_ROOT/runtime-sabotaged.sio" -o "$SABOTAGED" >/dev/null
chmod 0755 "$SABOTAGED"
run_frame "$SABOTAGED" "$python_attempt"
[[ "$FRAME_RC" -eq 0 && "$FRAME_OUTPUT" == SOUNIO_LANGUAGE_AUTHORITY_ALLOW* ]] ||
  fail "removing only the Python rule did not admit the unchanged control: rc=$FRAME_RC output=$FRAME_OUTPUT"

printf '%s\n' \
  'sounio-loom-language-authority-selftest: PASS language=Sounio cases=33 python=refused rust=refused policy_missing=refused llm_promotion=refused parent_laundering=refused ocaml_realization=admitted ocaml_prefreeze=refused ocaml_parent_laundering=refused ocaml_guardian=admitted ocaml_parity=refused cpp_bootstrap=admitted malformed=refused sabotage_python_rule=admits'
