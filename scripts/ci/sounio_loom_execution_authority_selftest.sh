#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-execution-authority-selftest.XXXXXX")"
RUNTIME="$TEST_ROOT/sounio-execution-authority"
SABOTAGED="$TEST_ROOT/sounio-execution-authority-sabotaged"
SENTINEL_DIR="$TEST_ROOT/sentinel-bin"
SENTINEL_MARKER="$TEST_ROOT/prohibited-runtime-executed"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-execution-authority-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

run_frame() {
  local runtime="$1" frame="$2"
  set +e
  FRAME_OUTPUT="$(printf '%s\n' "$frame" | "$runtime" 2>&1)"
  FRAME_RC=$?
  set -e
}

mkdir -p "$SENTINEL_DIR"
for forbidden in python python3 pypy pypy3 cargo rustc; do
  printf '#!/usr/bin/env bash\nprintf prohibited >%q\nexit 97\n' "$SENTINEL_MARKER" \
    >"$SENTINEL_DIR/$forbidden"
  chmod 0755 "$SENTINEL_DIR/$forbidden"
done
export PATH="$SENTINEL_DIR:$PATH"

SOUNIO_LOOM_EXECUTION_AUTHORITY_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_execution_authority.sh" >/dev/null

selftest="$(printf '0\n' | "$RUNTIME")"
[[ "$selftest" == 'SOUNIO_EXECUTION_AUTHORITY_SELFTEST PASS cases=32' ]] ||
  fail "Sounio-owned selftest did not pass: $selftest"

one='1 1 1 1 1 1 1 1'
two='2 2 2 2 2 2 2 2'
zero='0 0 0 0 0 0 0 0'

# schema, stage, surface, class, language, purpose, policy, preexec, closure,
# semantic-write, expected-write, review-promoted, parity-open, receipt-chain,
# exception, waiver founder/scope/purpose/live, then eight digest vectors.
python_attempt="9021 3 1 1 7 1 1 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $zero $zero"
run_frame "$RUNTIME" "$python_attempt"
[[ "$FRAME_RC" -eq 210 && "$FRAME_OUTPUT" == *'reason=forbidden-language'* ]] ||
  fail "Python oracle attempt was not refused: rc=$FRAME_RC output=$FRAME_OUTPUT"

python_waiver="9021 3 1 1 7 1 1 1 1 0 0 0 0 0 1 1 1 1 1 $one $one $one $one $one $one $zero $one"
run_frame "$RUNTIME" "$python_waiver"
[[ "$FRAME_RC" -eq 210 && "$FRAME_OUTPUT" == *'reason=forbidden-language'* ]] ||
  fail "Python waiver bypassed the non-waivable rule: rc=$FRAME_RC output=$FRAME_OUTPUT"

rust_attempt="9021 3 4 2 8 1 1 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $zero $zero"
run_frame "$RUNTIME" "$rust_attempt"
[[ "$FRAME_RC" -eq 210 && "$FRAME_OUTPUT" == *'reason=forbidden-language'* ]] ||
  fail "Rust oracle attempt was not refused: rc=$FRAME_RC output=$FRAME_OUTPUT"

ocaml_transport="9021 3 1 1 9 1 1 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $zero $zero"
run_frame "$RUNTIME" "$ocaml_transport"
[[ "$FRAME_RC" -eq 0 && "$FRAME_OUTPUT" == SOUNIO_EXECUTION_AUTHORITY_ALLOW* ]] ||
  fail "measured OCaml transport was not admitted: rc=$FRAME_RC output=$FRAME_OUTPUT"

dynamic_shell="9021 3 1 3 10 1 1 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $zero $zero"
run_frame "$RUNTIME" "$dynamic_shell"
[[ "$FRAME_RC" -eq 226 && "$FRAME_OUTPUT" == *'reason=dynamic-execution-unclassified'* ]] ||
  fail "dynamic shell was not refused: rc=$FRAME_RC output=$FRAME_OUTPUT"

missing_policy="9021 3 1 1 9 1 0 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $zero $zero"
run_frame "$RUNTIME" "$missing_policy"
[[ "$FRAME_RC" -eq 201 && "$FRAME_OUTPUT" == *'reason=policy-missing'* ]] ||
  fail "missing policy did not fail closed: rc=$FRAME_RC output=$FRAME_OUTPUT"

wrong_parent="9021 3 1 1 9 1 1 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $two $one $one $one $zero $zero"
run_frame "$RUNTIME" "$wrong_parent"
[[ "$FRAME_RC" -eq 217 && "$FRAME_OUTPUT" == *'reason=parent-semantics-hash-mismatch'* ]] ||
  fail "parent-hash laundering was not refused: rc=$FRAME_RC output=$FRAME_OUTPUT"

commit_without_chain="9021 3 2 1 11 1 1 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $zero $zero"
run_frame "$RUNTIME" "$commit_without_chain"
[[ "$FRAME_RC" -eq 229 && "$FRAME_OUTPUT" == *'reason=receipt-chain-missing'* ]] ||
  fail "commit without a receipt chain was not refused: rc=$FRAME_RC output=$FRAME_OUTPUT"

commit_with_chain="9021 3 2 1 11 1 1 1 1 0 0 0 0 1 0 0 0 0 0 $one $one $one $one $one $one $zero $zero"
run_frame "$RUNTIME" "$commit_with_chain"
[[ "$FRAME_RC" -eq 0 && "$FRAME_OUTPUT" == SOUNIO_EXECUTION_AUTHORITY_ALLOW* ]] ||
  fail "receipt-bound Git commit was not admitted: rc=$FRAME_RC output=$FRAME_OUTPUT"

llm_promotion="9021 3 1 1 6 4 1 1 1 0 0 1 0 0 0 0 0 0 0 $one $one $one $one $one $one $zero $zero"
run_frame "$RUNTIME" "$llm_promotion"
[[ "$FRAME_RC" -eq 219 && "$FRAME_OUTPUT" == *'reason=review-promoted-to-authority'* ]] ||
  fail "LLM promotion was not refused: rc=$FRAME_RC output=$FRAME_OUTPUT"

parity_closed="9021 3 1 1 2 3 1 1 1 0 0 0 0 1 0 0 0 0 0 $one $one $one $one $one $one $zero $zero"
run_frame "$RUNTIME" "$parity_closed"
[[ "$FRAME_RC" -eq 222 && "$FRAME_OUTPUT" == *'reason=parity-closed'* ]] ||
  fail "pre-open Lean parity was not refused: rc=$FRAME_RC output=$FRAME_OUTPUT"

run_frame "$RUNTIME" '9021 3'
[[ "$FRAME_RC" -eq 224 && "$FRAME_OUTPUT" == *'reason=malformed-frame'* ]] ||
  fail "malformed frame did not fail closed: rc=$FRAME_RC output=$FRAME_OUTPUT"

# Causal control: remove only Python from the Sounio prohibition and replay the
# unchanged Python frame. The decision must become ALLOW.
module="$ROOT_DIR/stdlib/coordination/loom_execution_authority.sio"
entrypoint="$ROOT_DIR/tools/loom/execution_authority_main.sio"
needle='if measured_language == 7 || measured_language == 8 { return 210 }'
[[ "$(grep -Fc "$needle" "$module")" -eq 1 ]] ||
  fail 'Python prohibition sabotage point is not unique'
sed "s/$needle/if measured_language == 8 { return 210 }/" "$module" \
  >"$TEST_ROOT/module-sabotaged.sio"
sed -n '1,$p' "$TEST_ROOT/module-sabotaged.sio" "$entrypoint" \
  >"$TEST_ROOT/runtime-sabotaged.sio"
SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" compile \
  "$TEST_ROOT/runtime-sabotaged.sio" -o "$SABOTAGED" >/dev/null
chmod 0755 "$SABOTAGED"
run_frame "$SABOTAGED" "$python_attempt"
[[ "$FRAME_RC" -eq 0 && "$FRAME_OUTPUT" == SOUNIO_EXECUTION_AUTHORITY_ALLOW* ]] ||
  fail "removing only the Python rule did not admit the unchanged control: rc=$FRAME_RC output=$FRAME_OUTPUT"

[[ ! -e "$SENTINEL_MARKER" ]] ||
  fail "a Python or Rust executable ran during the authority gate"

printf '%s\n' \
  'sounio-loom-execution-authority-selftest: PASS language=Sounio cases=32 python=refused rust=refused waiver_bypass=refused dynamic_shell=refused policy_missing=refused parent_laundering=refused commit_without_receipt=refused commit_with_receipt=admitted llm_promotion=refused parity_closed=refused malformed=refused sabotage_python_rule=admits python_executed=false rust_executed=false'
