#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-operation-ingress.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
SOURCE="tests/verify-ir/call_b.sio"
COMMAND="loom-exec-cell-v2 sounio-check source=$SOURCE"
COMMAND_SHA256="b5566b9ef6aa68866db784bbb33792d0dda4506932fbb10240db96ce99e1a27d"
EVENT="6017e4c6e745560696f78836f9cc07ec71a9106f13ad1bfdb16d7e342f0840a9"
GENERATION="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
PRINCIPAL="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
DESCRIPTOR="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
GRANT="dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
RECORD_MANIFEST_SHA256="58d4a49c5b2462261ee53cd06f3ca8e29d363c1a38bf47274fa98a67b79cc569"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-exec-operation-ingress-v1-20260830.txt"

fail() {
  printf 'sounio-loom-exec-operation-ingress-selftest: FAIL: %s test_root=%s\n' \
    "$*" "$TEST_ROOT" >&2
  exit 1
}

mutate_digest() {
  local value="$1"
  printf '%s%s' "${value:0:63}" "$([[ "${value:63:1}" == 0 ]] && printf 1 || printf 0)"
}

evidence_value() {
  local key="$1" line count
  count="$(grep -c "^${key}=" "$EVIDENCE" || true)"
  [[ "$count" == 1 ]] || fail "evidence field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$EVIDENCE")"
  printf '%s' "${line#*=}"
}

(cd "$ROOT_DIR/tools/loom" && dune build src/loom.exe) >/dev/null
bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_operation_catalog_fixture.sh" \
  >/dev/null
bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_result_record_fixture.sh" \
  >/dev/null
bash "$ROOT_DIR/scripts/ci/sounio_loom_exec_operation_grant_fixture_freeze_selftest.sh" \
  >/dev/null

SOURCE_SHA256="$(sha256sum "$ROOT_DIR/$SOURCE" | cut -d ' ' -f 1)"
OUTPUT="$TEST_ROOT/loom-sounio-check-${SOURCE_SHA256:0:16}.elf"
SOUNIO_LOOM_HOOK_TEST_MODE=1 "$LOOM" exec-result-record-probe \
  --root "$ROOT_DIR" --mode issue --source "$SOURCE" --output "$OUTPUT" \
  --event "$EVENT" --generation "$GENERATION" --principal "$PRINCIPAL" \
  --descriptor-binding "$DESCRIPTOR" --grant-receipt "$GRANT" \
  >"$TEST_ROOT/issued"
HEADER="$(head -n1 "$TEST_ROOT/issued")"
HANDLE="$(sed -n '1s/.* handle=\([^ ]*\) .*/\1/p' "$TEST_ROOT/issued")"
RECORD_SHA256="$(sed -n '1s/.* record_sha256=\([^ ]*\) .*/\1/p' "$TEST_ROOT/issued")"
tail -n +2 "$TEST_ROOT/issued" >"$TEST_ROOT/record"
[[ "$HEADER" == *'semantic_authority=Sounio action=9036'* && \
   "$HANDLE" == "loom-result-v2:$EVENT:$GENERATION:$RECORD_SHA256" && \
   "$(sha256sum "$TEST_ROOT/record" | cut -d ' ' -f 1)" == "$RECORD_SHA256" ]] ||
  fail 'issued record or handle diverged'

LANGUAGE_MANIFEST="$ROOT_DIR/tools/loom/language_authority.freeze.v1"
LANGUAGE_RUNTIME="$TEST_ROOT/sounio-loom-language-authority-runtime"
RESIDENT_RUNTIME="$TEST_ROOT/sounio-loom-resident-membrane-runtime-v5"
TOOLCHAIN_ROOT="$TEST_ROOT/toolchain"
FROZEN_COMMIT="$(sed -n 's/^sounio_executable_commit=//p' "$LANGUAGE_MANIFEST")"
[[ -n "$FROZEN_COMMIT" ]] || fail 'language authority commit absent'
mkdir -p "$TOOLCHAIN_ROOT"
git -C "$ROOT_DIR" archive "$FROZEN_COMMIT" \
  bin/souc bin/souc-lean-single-x86_64 | tar -x -C "$TOOLCHAIN_ROOT"
SOUNIO_LOOM_LANGUAGE_AUTHORITY_SOUC="$TOOLCHAIN_ROOT/bin/souc" \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_OUTPUT="$LANGUAGE_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_language_authority.sh" >/dev/null
SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_OUTPUT="$RESIDENT_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v5.sh" >/dev/null

printf '{"hook_event_name":"PreToolUse","session_id":"operation-ingress","cwd":"%s","tool_name":"Bash","tool_input":{"command":"%s"}}' \
  "$ROOT_DIR" "$COMMAND" >"$TEST_ROOT/event.json"
printf '{"hook_event_name":"PreToolUse","session_id":"operation-command-control","cwd":"%s","tool_name":"Bash","tool_input":{"command":"%s"}}' \
  "$ROOT_DIR" "loom-exec-cell-v2 sounio-check source=tests/verify-ir/call_a.sio" \
  >"$TEST_ROOT/command-control.json"
printf '{"hook_event_name":"PreToolUse","session_id":"operation-source-control","cwd":"%s","tool_name":"Bash","tool_input":{"command":"%s"}}' \
  "$ROOT_DIR" "loom-exec-cell-v2 sounio-check source=../call_b.sio" \
  >"$TEST_ROOT/source-control.json"

probe() {
  local tag="$1" mode="$2" event_path="${3:-$TEST_ROOT/event.json}"
  SOUNIO_COORD_DIR="$TEST_ROOT/coord" \
  SOUNIO_COORD_RUNTIME_MODE=local \
  SOUNIO_COORD_NATIVE_HOOK_SELFTEST=1 \
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME="$LANGUAGE_RUNTIME" \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_LOG="$TEST_ROOT/$tag.hook.tsv" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_RUNTIME="$RESIDENT_RUNTIME" \
  SOUNIO_LOOM_RESIDENT_RECEIPT_LOG="$TEST_ROOT/$tag.resident.tsv" \
  SOUNIO_LOOM_EXEC_INGRESS_DARK_LOG="$TEST_ROOT/$tag.ingress.tsv" \
    "$LOOM" exec-ingress-probe --root "$ROOT_DIR" --mode "$mode" \
      --event "$event_path" --record "$TEST_ROOT/record" --handle "$HANDLE"
}

POSITIVE="$(probe positive record)"
[[ "$POSITIVE" == *'mode=record hook_code=0 broker_code=0'* && \
   "$POSITIVE" == *'result_returned=true'* && \
   "$POSITIVE" == *'Sounio 9036 returned a verified ExecCell operation record'* && \
   "$POSITIVE" == *'exec-result-record-present'* && \
   "$POSITIVE" == *"$HANDLE"* && "$POSITIVE" != *'exec-capability'* ]] ||
  fail "positive operation transport diverged: $POSITIVE"
grep -Fq $'reason=descriptor-result-bound-actions-9030+9031+9035+9036\tauthorizing=false\tproduction_activation=false\texec_attached=true' \
  "$TEST_ROOT/positive.ingress.tsv" ||
  fail 'positive ingress audit omitted the four-action binding'
grep -Fq $'exec_intent_projected=true\texec_intent_action=9035\t' \
  "$TEST_ROOT/positive.ingress.tsv" ||
  fail 'positive ingress audit omitted Sounio 9035'
grep -Fq $'exec_projection_kind=operation-catalog\t' \
  "$TEST_ROOT/positive.ingress.tsv" ||
  fail 'positive ingress audit mislabeled the operation projection'
grep -Fq $'result_returned=true\tresult_action=9036\tresult_kind=operation-record\t' \
  "$TEST_ROOT/positive.ingress.tsv" ||
  fail 'positive ingress audit omitted Sounio 9036'
raw_event="$(sed -n 's/.*\traw_event_sha256=\([^[:space:]]*\).*/\1/p' \
  "$TEST_ROOT/positive.ingress.tsv")"
semantic_event="$(sed -n 's/.*\tevent_sha256=\([^[:space:]]*\).*/\1/p' \
  "$TEST_ROOT/positive.ingress.tsv")"
[[ "$raw_event" =~ ^[0-9a-f]{64}$ && "$semantic_event" == "$EVENT" && \
   "$raw_event" != "$semantic_event" ]] ||
  fail 'raw hook event and Sounio operation identity were not separated'

RECORD_HEX="$(od -An -tx1 -v "$TEST_ROOT/record" | tr -d ' \n')"
SOUNIO_LOOM_HOOK_TEST_MODE=1 "$LOOM" exec-result-record-present \
  --root "$ROOT_DIR" --event "$EVENT" --command "$COMMAND_SHA256" \
  --handle "$HANDLE" --record-sha256 "$RECORD_SHA256" \
  --record-hex "$RECORD_HEX" --manifest-sha256 "$RECORD_MANIFEST_SHA256" \
  >"$TEST_ROOT/presented"
cmp "$TEST_ROOT/record" "$TEST_ROOT/presented" ||
  fail 'read-only record presenter changed canonical bytes'

for control in \
  'binding:record-binding:product-exec-ingress-response-binding-mismatch' \
  'digest:record-digest:exec-result-record-transport-record-hash-mismatch' \
  'manifest:record-manifest:exec-result-record-transport-manifest-mismatch'
do
  IFS=: read -r tag mode reason <<<"$control"
  output="$(probe "$tag" "$mode")"
  [[ "$output" == *"mode=$mode hook_code=2 broker_code=0"* && \
     "$output" == *"$reason"* && "$output" != *'exec-result-record-present'* && \
     "$output" != *'exec-capability'* ]] ||
    fail "$tag sabotage did not fail closed: $output"
done

COMMAND_CONTROL="$(probe command record "$TEST_ROOT/command-control.json")"
[[ "$COMMAND_CONTROL" == *'hook_code=2 broker_code=0'* && \
   "$COMMAND_CONTROL" == *'exec-result-record-transport-command-not-frozen-operation'* && \
   "$COMMAND_CONTROL" != *'exec-result-record-present'* ]] ||
  fail "operation command substitution was admitted: $COMMAND_CONTROL"
SOURCE_CONTROL="$(probe source record "$TEST_ROOT/source-control.json")"
[[ "$SOURCE_CONTROL" == *'hook_code=2'* && \
   "$SOURCE_CONTROL" == *'SOUNIO_EXEC_OPERATION_CATALOG DENY563'* && \
   "$SOURCE_CONTROL" != *'exec-result-record-present'* ]] ||
  fail "invalid source did not causally reach Sounio DENY563: $SOURCE_CONTROL"

BAD_MANIFEST="$(mutate_digest "$RECORD_MANIFEST_SHA256")"
set +e
ORIGINAL_REFUSAL="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  "$LOOM" exec-result-record-present --root "$ROOT_DIR" --event "$EVENT" \
  --command "$COMMAND_SHA256" --handle "$HANDLE" \
  --record-sha256 "$RECORD_SHA256" --record-hex "$RECORD_HEX" \
  --manifest-sha256 "$BAD_MANIFEST" 2>&1)"
ORIGINAL_CODE=$?
set -e
[[ $ORIGINAL_CODE -eq 1 && "$ORIGINAL_REFUSAL" == \
   'error: exec-result-record-transport-manifest-mismatch' ]] ||
  fail 'the original transport rule did not refuse its manifest sabotage'

cp "$ROOT_DIR/tools/loom/src/loom_exec_intent.ml" "$TEST_ROOT/loom_exec_intent.ml"
cp "$ROOT_DIR/tools/loom/src/loom_exec_result.ml" "$TEST_ROOT/loom_exec_result.ml"
cp "$ROOT_DIR/tools/loom/src/loom_exec_catalog.ml" "$TEST_ROOT/loom_exec_catalog.ml"
sed 's/if manifest_sha256 <> policy.manifest_sha256 then/if false then/' \
  "$ROOT_DIR/tools/loom/src/loom_exec_result_record.ml" \
  >"$TEST_ROOT/loom_exec_result_record.ml"
grep -Fq 'if false then' "$TEST_ROOT/loom_exec_result_record.ml" ||
  fail 'causal mutant did not delete the 9036 manifest equality rule'
cat >"$TEST_ROOT/mutant_main.ml" <<'EOF'
let () =
  let result =
    Loom_exec_result_record.validate_transport ~root:Sys.argv.(1)
      ~event_sha256:Sys.argv.(2) ~command_sha256:Sys.argv.(3)
      ~handle:Sys.argv.(4) ~record_sha256:Sys.argv.(5)
      ~record_hex:Sys.argv.(6) ~manifest_sha256:Sys.argv.(7)
  in
  Printf.printf "MUTANT_ADMITTED record_sha256=%s\n" result.record_sha256
EOF
(
  cd "$TEST_ROOT"
  ocamlfind ocamlopt -package unix,cryptokit -linkpkg \
    loom_exec_intent.ml loom_exec_result.ml loom_exec_catalog.ml \
    loom_exec_result_record.ml mutant_main.ml -o mutant-presenter
)
MUTANT="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  "$TEST_ROOT/mutant-presenter" "$ROOT_DIR" "$EVENT" "$COMMAND_SHA256" \
  "$HANDLE" "$RECORD_SHA256" "$RECORD_HEX" "$BAD_MANIFEST")"
[[ "$MUTANT" == MUTANT_ADMITTED* ]] ||
  fail "manifest-rule deletion did not admit the same sabotage: $MUTANT"

mkdir -p "$TEST_ROOT/oracles"
ORACLE_EXECUTED="$TEST_ROOT/prohibited-oracle-executed"
for name in python python3 rustc cargo; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$ORACLE_EXECUTED" \
    >"$TEST_ROOT/oracles/$name"
  chmod 0755 "$TEST_ROOT/oracles/$name"
done
PATH="$TEST_ROOT/oracles:$PATH" probe oracle record >/dev/null
[[ ! -e "$ORACLE_EXECUTED" ]] || fail 'a prohibited Python or Rust oracle executed'
DEPENDENCIES="$(ldd "$LOOM" 2>&1 || true; \
  ldd "$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-exec-operation-catalog" 2>&1 || true; \
  ldd "$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-exec-result-record" 2>&1 || true)"
printf '%s\n' "$DEPENDENCIES" | grep -Eqi 'python|rust' &&
  fail 'operation-ingress runtime has a prohibited dependency'

RESULT='sounio-loom-exec-operation-ingress-selftest: PASS semantic_authority=Sounio actions=9030+9031+9035+9036 operational_attachment=OCaml event_projection=Sounio-9035 transport=authenticated-inherited-descriptor result_record_returned=true presenter=read-only canonical_record_sha256=dynamic+verified handle_is_bearer=false handle_is_execution_authority=false raw_event_separate=true binding_sabotage=REFUSED digest_sabotage=REFUSED manifest_sabotage=REFUSED command_substitution=REFUSED invalid_source=DENY563 causal_rule=manifest_sha256_equal causal_mutant=ADMITTED python_executed=false rust_executed=false provider_lifecycle_attached=false dynamic_user_host_attached=false production_activation=false parity_open=false claim_ready=false'
printf '%s\n' "$RESULT"

if [[ "${SOUNIO_LOOM_EXEC_OPERATION_INGRESS_SKIP_EVIDENCE:-0}" == 1 ]]; then
  exit 0
fi
[[ -f "$EVIDENCE" ]] || fail 'checked-in evidence absent'
IMPLEMENTATION_COMMIT="$(evidence_value implementation_commit)"
[[ "$IMPLEMENTATION_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
  fail 'checked-in implementation commit is malformed'
git -C "$ROOT_DIR" cat-file -e "${IMPLEMENTATION_COMMIT}^{commit}" 2>/dev/null ||
  fail 'checked-in implementation commit is absent'
git -C "$ROOT_DIR" merge-base --is-ancestor "$IMPLEMENTATION_COMMIT" HEAD ||
  fail 'checked-in implementation commit is not an ancestor of HEAD'
for binding in \
  "result:$RESULT" \
  "sounio_operation_manifest_sha256:$(sha256sum "$ROOT_DIR/tools/loom/exec_operation_catalog.freeze.v1" | cut -d ' ' -f 1)" \
  "sounio_result_manifest_sha256:$(sha256sum "$ROOT_DIR/tools/loom/exec_result_record.freeze.v1" | cut -d ' ' -f 1)" \
  "sounio_grant_manifest_sha256:$(sha256sum "$ROOT_DIR/tools/loom/exec_operation_grant_fixture.freeze.v1" | cut -d ' ' -f 1)" \
  "operational_ingress_source_sha256:$(sha256sum "$ROOT_DIR/tools/loom/src/loom_exec_ingress.ml" | cut -d ' ' -f 1)" \
  "operational_record_source_sha256:$(sha256sum "$ROOT_DIR/tools/loom/src/loom_exec_result_record.ml" | cut -d ' ' -f 1)" \
  "operational_hook_source_sha256:$(sha256sum "$ROOT_DIR/tools/loom/src/loom_hook.ml" | cut -d ' ' -f 1)" \
  "gate_sha256:$(sha256sum "$ROOT_DIR/scripts/ci/sounio_loom_exec_operation_ingress_selftest.sh" | cut -d ' ' -f 1)" \
  "loom_executable_sha256:$(sha256sum "$LOOM" | cut -d ' ' -f 1)"
do
  key="${binding%%:*}"
  expected="${binding#*:}"
  [[ "$(evidence_value "$key")" == "$expected" ]] ||
    fail "checked-in evidence drifted: $key"
done
