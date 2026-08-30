#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_PARENT="$ROOT_DIR/tools/loom/_build"
mkdir -p "$TEST_PARENT"
TEST_ROOT="$(mktemp -d "$TEST_PARENT/product-exec-result-ingress.XXXXXX")"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-product-exec-result-ingress-v1-20260830.txt"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-product-exec-result-ingress-selftest: FAIL: %s test_root=%s\n' \
    "$*" "$TEST_ROOT" >&2
  exit 1
}

manifest_value() {
  local key="$1" line count
  count="$(grep -c "^${key}=" "$ROOT_DIR/tools/loom/exec_result_handle.freeze.v1" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$ROOT_DIR/tools/loom/exec_result_handle.freeze.v1")"
  printf '%s' "${line#*=}"
}

fixture_value() {
  local key="$1" line count
  count="$(grep -c "^${key}=" "$ROOT_DIR/tools/loom/product_exec_cell_fixture.freeze.v1" || true)"
  [[ "$count" == 1 ]] || fail "fixture field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$ROOT_DIR/tools/loom/product_exec_cell_fixture.freeze.v1")"
  printf '%s' "${line#*=}"
}

evidence_value() {
  local key="$1" line count
  count="$(grep -c "^${key}=" "$EVIDENCE" || true)"
  [[ "$count" == 1 ]] || fail "evidence field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$EVIDENCE")"
  printf '%s' "${line#*=}"
}

AUTHORITY_RUNTIME="$TEST_ROOT/sounio-loom-exec-result-handle"
SOUNIO_LOOM_EXEC_RESULT_HANDLE_OUTPUT="$AUTHORITY_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_result_handle_fixture.sh" \
  >/dev/null
[[ "$(sha256sum "$AUTHORITY_RUNTIME" | cut -d ' ' -f 1)" == \
   "$(manifest_value executable_sha256)" ]] ||
  fail 'source-fresh Sounio 9033 runtime hash drifted'

INTENT_RUNTIME="$TEST_ROOT/sounio-loom-exec-intent-envelope"
SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_OUTPUT="$INTENT_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_intent_envelope_fixture.sh" \
  >/dev/null
intent_runtime_sha256="$(sed -n 's/^executable_sha256=//p' \
  "$ROOT_DIR/tools/loom/exec_intent_envelope.freeze.v1")"
[[ "$(sha256sum "$INTENT_RUNTIME" | cut -d ' ' -f 1)" == \
   "$intent_runtime_sha256" ]] ||
  fail 'source-fresh Sounio 9034 runtime hash drifted'

PAYLOAD="$TEST_ROOT/sounio-process-witness-handshake"
SOUNIO_LOOM_PROCESS_WITNESS_HANDSHAKE_OUTPUT="$PAYLOAD" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_process_witness_handshake_payload.sh" \
  >/dev/null
printf 'CLOSE\n' | env -i "$PAYLOAD" >"$TEST_ROOT/result.receipt"
[[ "$(sha256sum "$TEST_ROOT/result.receipt" | cut -d ' ' -f 1)" == \
   "$(manifest_value result_receipt_sha256)" ]] ||
  fail 'actual Sounio payload receipt hash drifted'

LANGUAGE_MANIFEST="$ROOT_DIR/tools/loom/language_authority.freeze.v1"
LANGUAGE_RUNTIME="$TEST_ROOT/sounio-loom-language-authority-runtime"
RESIDENT_RUNTIME="$TEST_ROOT/sounio-loom-resident-membrane-runtime-v5"
TOOLCHAIN_ROOT="$TEST_ROOT/toolchain"
frozen_commit="$(sed -n 's/^sounio_executable_commit=//p' "$LANGUAGE_MANIFEST")"
[[ -n "$frozen_commit" ]] || fail 'language-authority manifest omitted its commit'
mkdir -p "$TOOLCHAIN_ROOT"
git -C "$ROOT_DIR" archive "$frozen_commit" \
  bin/souc bin/souc-lean-single-x86_64 | tar -x -C "$TOOLCHAIN_ROOT"
SOUNIO_LOOM_LANGUAGE_AUTHORITY_SOUC="$TOOLCHAIN_ROOT/bin/souc" \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_OUTPUT="$LANGUAGE_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_language_authority.sh" >/dev/null
SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_OUTPUT="$RESIDENT_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v5.sh" >/dev/null

dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
COMMAND="$(sed -n 's/^command=//p' "$ROOT_DIR/tools/loom/product_exec_cell_fixture.freeze.v1")"
[[ -n "$COMMAND" ]] || fail 'product fixture command absent'
printf '{"hook_event_name":"PreToolUse","session_id":"exec-result-v1","cwd":"%s","tool_name":"Bash","tool_input":{"command":"%s"}}' \
  "$ROOT_DIR" "$COMMAND" >"$TEST_ROOT/event.json"

probe() {
  local tag="$1" mode="$2"
  SOUNIO_COORD_DIR="$TEST_ROOT/coord" \
  SOUNIO_COORD_RUNTIME_MODE=local \
  SOUNIO_COORD_NATIVE_HOOK_SELFTEST=1 \
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME="$LANGUAGE_RUNTIME" \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_LOG="$TEST_ROOT/$tag.hook.tsv" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_RUNTIME="$RESIDENT_RUNTIME" \
  SOUNIO_LOOM_RESIDENT_RECEIPT_LOG="$TEST_ROOT/$tag.resident.tsv" \
  SOUNIO_LOOM_EXEC_RESULT_HANDLE_RUNTIME="$AUTHORITY_RUNTIME" \
  SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_RUNTIME="$INTENT_RUNTIME" \
  SOUNIO_LOOM_EXEC_INGRESS_DARK_LOG="$TEST_ROOT/$tag.ingress.tsv" \
    "$LOOM" exec-ingress-probe --root "$ROOT_DIR" --mode "$mode" \
      --event "$TEST_ROOT/event.json" --receipt "$TEST_ROOT/result.receipt"
}

positive="$(probe positive result)"
[[ "$positive" == *'mode=result hook_code=0 broker_code=0'* && \
   "$positive" == *'result_returned=true exact_fixture_hook_switched=true'* && \
   "$positive" == *'Sounio 9033 returned a read-only ExecCell result'* && \
   "$positive" == *'exec-result-present'* && \
   "$positive" == *"$(manifest_value canonical_handle)"* && \
   "$positive" != *'exec-capability'* ]] ||
  fail "positive result transport diverged: $positive"
grep -Fq $'reason=descriptor-result-bound-actions-9030+9031+9033+9034\tauthorizing=false\tproduction_activation=false\texec_attached=true' \
  "$TEST_ROOT/positive.ingress.tsv" ||
  fail 'positive ingress receipt did not join the returned result'
grep -Fq $'exec_intent_projected=true\texec_intent_action=9034\texec_intent_manifest_sha256=8a95e587ccc81c16da17d56b9649d04bc9c3e764d66fc938c195d95568e7608e\t' \
  "$TEST_ROOT/positive.ingress.tsv" ||
  fail 'positive ingress receipt omitted the Sounio 9034 projection'
grep -Fq $'result_returned=true\t' "$TEST_ROOT/positive.ingress.tsv" ||
  fail 'positive ingress receipt omitted result_returned=true'
raw_event="$(sed -n 's/.*\traw_event_sha256=\([^[:space:]]*\).*/\1/p' "$TEST_ROOT/positive.ingress.tsv")"
semantic_event="$(sed -n 's/.*\tevent_sha256=\([^[:space:]]*\).*/\1/p' "$TEST_ROOT/positive.ingress.tsv")"
[[ "$raw_event" =~ ^[0-9a-f]{64}$ && \
   "$semantic_event" == "$(manifest_value event_sha256)" && \
   "$raw_event" != "$semantic_event" ]] ||
  fail 'Sounio semantic event projection was not explicit and separate'

receipt_hex="$(od -An -tx1 -v "$TEST_ROOT/result.receipt" | tr -d ' \n')"
SOUNIO_LOOM_HOOK_TEST_MODE=1 \
SOUNIO_LOOM_EXEC_RESULT_HANDLE_RUNTIME="$AUTHORITY_RUNTIME" \
  "$LOOM" exec-result-present --root "$ROOT_DIR" \
    --event "$(manifest_value event_sha256)" \
    --command "$(fixture_value command_sha256)" \
    --handle "$(manifest_value canonical_handle)" \
    --receipt-sha256 "$(manifest_value result_receipt_sha256)" \
    --receipt-hex "$receipt_hex" \
    --manifest-sha256 "$(sha256sum "$ROOT_DIR/tools/loom/exec_result_handle.freeze.v1" | cut -d ' ' -f 1)" \
    >"$TEST_ROOT/presented.receipt"
cmp "$TEST_ROOT/result.receipt" "$TEST_ROOT/presented.receipt" ||
  fail 'read-only presenter changed the receipt bytes'

for item in \
  'binding:result-binding:product-exec-ingress-response-binding-mismatch' \
  'receipt:result-receipt:result-transport-receipt-not-frozen-fixture' \
  'manifest:result-manifest:result-transport-manifest-not-frozen-fixture'
do
  IFS=: read -r tag mode reason <<<"$item"
  output="$(probe "$tag" "$mode")"
  [[ "$output" == *"mode=$mode hook_code=2 broker_code=0"* && \
     "$output" == *"$reason"* && "$output" != *'exec-result-present'* && \
     "$output" != *'exec-capability'* ]] ||
    fail "$tag sabotage did not fail closed: $output"
done

bad_manifest="$(sha256sum "$ROOT_DIR/tools/loom/exec_result_handle.freeze.v1" | cut -d ' ' -f 1)"
bad_manifest="${bad_manifest%?}$( [[ "${bad_manifest: -1}" == 0 ]] && printf 1 || printf 0 )"
set +e
original_refusal="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_RESULT_HANDLE_RUNTIME="$AUTHORITY_RUNTIME" \
  "$LOOM" exec-result-present --root "$ROOT_DIR" \
    --event "$(manifest_value event_sha256)" \
    --command "$(fixture_value command_sha256)" \
    --handle "$(manifest_value canonical_handle)" \
    --receipt-sha256 "$(manifest_value result_receipt_sha256)" \
    --receipt-hex "$receipt_hex" --manifest-sha256 "$bad_manifest" 2>&1)"
original_code=$?
set -e
[[ $original_code -eq 1 && \
   "$original_refusal" == *'result-transport-manifest-not-frozen-fixture'* ]] ||
  fail 'the original manifest rule did not refuse its sabotage'

sed 's/if manifest_sha256 <> policy.manifest_sha256 then/if false then/' \
  "$ROOT_DIR/tools/loom/src/loom_exec_result.ml" >"$TEST_ROOT/loom_exec_result.ml"
grep -Fq 'if false then' "$TEST_ROOT/loom_exec_result.ml" ||
  fail 'causal mutant did not delete the manifest equality rule'
cat >"$TEST_ROOT/mutant_main.ml" <<'EOF'
let () =
  let result =
    Loom_exec_result.validate_transport ~root:Sys.argv.(1)
      ~event_sha256:Sys.argv.(2) ~command_sha256:Sys.argv.(3)
      ~handle:Sys.argv.(4) ~receipt_sha256:Sys.argv.(5)
      ~receipt_hex:Sys.argv.(6) ~manifest_sha256:Sys.argv.(7)
  in
  Printf.printf "MUTANT_ADMITTED receipt_sha256=%s\n" result.receipt_sha256
EOF
(
  cd "$TEST_ROOT"
  ocamlfind ocamlopt -package unix,cryptokit -linkpkg loom_exec_result.ml \
    mutant_main.ml -o mutant-presenter
)
mutant="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_RESULT_HANDLE_RUNTIME="$AUTHORITY_RUNTIME" \
  "$TEST_ROOT/mutant-presenter" "$ROOT_DIR" \
    "$(manifest_value event_sha256)" "$(fixture_value command_sha256)" \
    "$(manifest_value canonical_handle)" \
    "$(manifest_value result_receipt_sha256)" "$receipt_hex" "$bad_manifest")"
[[ "$mutant" == MUTANT_ADMITTED* ]] ||
  fail "rule-deletion mutant did not admit the unchanged sabotage: $mutant"

oracle_sentinel="$TEST_ROOT/prohibited-oracle-executed"
mkdir "$TEST_ROOT/oracles"
for name in python python3 rustc cargo; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$oracle_sentinel" \
    >"$TEST_ROOT/oracles/$name"
  chmod 0755 "$TEST_ROOT/oracles/$name"
done
PATH="$TEST_ROOT/oracles:$PATH" probe oracle result >/dev/null
[[ ! -e "$oracle_sentinel" ]] || fail 'a prohibited Python or Rust oracle executed'
dependencies="$(ldd "$LOOM" 2>&1 || true; ldd "$AUTHORITY_RUNTIME" 2>&1 || true; \
  ldd "$INTENT_RUNTIME" 2>&1 || true)"
printf '%s\n' "$dependencies" | grep -Eqi 'python|rust' &&
  fail 'a runtime has a prohibited Python or Rust dependency'

result="$(printf 'sounio-loom-product-exec-result-ingress-selftest: PASS semantic_authority=Sounio actions=9030+9031+9033+9034 operational_attachment=OCaml transport=authenticated-inherited-descriptor result_returned=true presenter=read-only receipt_sha256=%s manifest_sha256=%s raw_event_separate=true event_projection=Sounio-9034 event_override=false exact_fixture_hook_switched=true local_exec_capability_used=false binding_sabotage=REFUSED receipt_sabotage=REFUSED manifest_sabotage=REFUSED causal_rule=manifest_sha256_equal causal_mutant=ADMITTED python_executed=false rust_executed=false material_exec_cell=false production_activation=false parity_open=false claim_ready=false' \
  "$(manifest_value result_receipt_sha256)" \
  "$(sha256sum "$ROOT_DIR/tools/loom/exec_result_handle.freeze.v1" | cut -d ' ' -f 1)")"
for binding in \
  "sounio_semantics_manifest_sha256:tools/loom/exec_result_handle.freeze.v1" \
  "sounio_intent_manifest_sha256:tools/loom/exec_intent_envelope.freeze.v1" \
  "operational_result_source_sha256:tools/loom/src/loom_exec_result.ml" \
  "operational_intent_source_sha256:tools/loom/src/loom_exec_intent.ml" \
  "operational_ingress_source_sha256:tools/loom/src/loom_exec_ingress.ml" \
  "operational_hook_source_sha256:tools/loom/src/loom_hook.ml" \
  "material_ingress_source_sha256:tools/loom/src/loom_product_exec_ingress_host_canary.inc" \
  "material_exec_cell_source_sha256:tools/loom/src/loom_product_exec_cell_host_canary.inc" \
  "material_broker_source_sha256:tools/loom/src/loom_kernel_principal_broker.cpp" \
  "gate_sha256:scripts/ci/sounio_loom_product_exec_result_ingress_selftest.sh" \
  "host_evidence_sha256:tools/loom/evidence/loom-provider-lifecycle-exec-cell-host-v1-20260830.txt"
do
  IFS=: read -r key relative <<<"$binding"
  [[ "$(evidence_value "$key")" == \
     "$(sha256sum "$ROOT_DIR/$relative" | cut -d ' ' -f 1)" ]] ||
    fail "evidence binding drifted: $key"
done
[[ "$(evidence_value result)" == "$result" &&
   "$(evidence_value material_exec_cell)" == true &&
   "$(evidence_value exact_fixture_result_attached)" == true &&
   "$(evidence_value general_exec_attached)" == false &&
   "$(evidence_value provider_hook_switched)" == true &&
   "$(evidence_value provider_lifecycle_attached)" == true &&
   "$(evidence_value production_activation)" == false ]] ||
  fail 'evidence posture or result drifted'
printf '%s\n' "$result"
