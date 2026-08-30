#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-result-store.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-exec-result-store-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

manifest_value() {
  local key="$1" line count
  count="$(grep -c "^${key}=" "$ROOT_DIR/tools/loom/exec_result_handle.freeze.v1" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$ROOT_DIR/tools/loom/exec_result_handle.freeze.v1")"
  printf '%s' "${line#*=}"
}

evidence_value() {
  local key="$1" line count
  count="$(grep -c "^${key}=" "$ROOT_DIR/tools/loom/evidence/loom-exec-result-store-v1-20260830.txt" || true)"
  [[ "$count" == 1 ]] || fail "evidence field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$ROOT_DIR/tools/loom/evidence/loom-exec-result-store-v1-20260830.txt")"
  printf '%s' "${line#*=}"
}

run_probe() {
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_RESULT_HANDLE_RUNTIME="$AUTHORITY_RUNTIME" \
    "$LOOM" exec-result-probe --root "$ROOT_DIR" --store "$STORE" "$@"
}

AUTHORITY_RUNTIME="$TEST_ROOT/sounio-loom-exec-result-handle"
SOUNIO_LOOM_EXEC_RESULT_HANDLE_OUTPUT="$AUTHORITY_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_result_handle_fixture.sh" >/dev/null
[[ "$(sha256sum "$AUTHORITY_RUNTIME" | cut -d ' ' -f 1)" == \
   "$(manifest_value executable_sha256)" ]] || fail 'source-fresh authority hash drifted'

PAYLOAD="$TEST_ROOT/sounio-process-witness-handshake"
SOUNIO_LOOM_PROCESS_WITNESS_HANDSHAKE_OUTPUT="$PAYLOAD" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_process_witness_handshake_payload.sh" >/dev/null
printf 'CLOSE\n' | env -i "$PAYLOAD" > "$TEST_ROOT/result.receipt"
[[ "$(sha256sum "$TEST_ROOT/result.receipt" | cut -d ' ' -f 1)" == \
   "$(manifest_value result_receipt_sha256)" ]] || fail 'material receipt hash drifted'

dune build --root "$ROOT_DIR/tools/loom" src/loom.exe
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
STORE="$TEST_ROOT/store"
mkdir -m 0700 "$STORE"

publish="$(run_probe --mode publish --receipt "$TEST_ROOT/result.receipt")"
printf '%s\n' "$publish" | grep -Fq 'mode=publish semantic_authority=Sounio action=9033 operational_kernel=OCaml' ||
  fail "publish output diverged: $publish"
printf '%s\n' "$publish" | grep -Fq 'material_result_store=true result_store_attached=false' ||
  fail 'publish widened the attachment claim'
handle="$(printf '%s\n' "$publish" | sed -n 's/.* handle=\([^ ]*\) .*/\1/p')"
record="$(printf '%s\n' "$publish" | sed -n 's/.* record_path=\([^ ]*\) .*/\1/p')"
record_sha256="$(printf '%s\n' "$publish" | sed -n 's/.* record_sha256=\([^ ]*\) .*/\1/p')"
authority_output_sha256="$(printf '%s\n' "$publish" | sed -n 's/.* authority_output_sha256=\([^ ]*\) .*/\1/p')"
[[ "$handle" == "$(manifest_value canonical_handle)" ]] || fail 'published handle drifted'
[[ "$record_sha256" =~ ^[0-9a-f]{64}$ && "$authority_output_sha256" =~ ^[0-9a-f]{64}$ ]] ||
  fail 'material record or authority output digest is malformed'
[[ -f "$record" && ! -L "$record" ]] || fail 'result record is absent or linked'
[[ "$(stat -c '%a:%u:%h' "$record")" == "400:$(id -u):1" ]] ||
  fail 'result record is not immutable, singly linked, and owner-bound'

resolve="$(run_probe --mode resolve --handle "$handle")"
printf '%s\n' "$resolve" | grep -Fq 'mode=resolve semantic_authority=Sounio action=9033 operational_kernel=OCaml' ||
  fail "resolve output diverged: $resolve"
printf '%s\n' "$resolve" | grep -Fq "receipt_sha256=$(manifest_value result_receipt_sha256)" ||
  fail 'resolved receipt digest drifted'

set +e
duplicate="$(run_probe --mode publish --receipt "$TEST_ROOT/result.receipt" 2>&1)"
duplicate_code=$?
set -e
[[ $duplicate_code -eq 1 && "$duplicate" == *'result-record-already-exists'* ]] ||
  fail "duplicate publish was not refused: $duplicate"

printf 'wrong\n' > "$TEST_ROOT/wrong.receipt"
set +e
wrong_receipt="$(run_probe --mode publish --receipt "$TEST_ROOT/wrong.receipt" 2>&1)"
wrong_receipt_code=$?
set -e
[[ $wrong_receipt_code -eq 1 && "$wrong_receipt" == *'result-receipt-hash-mismatch'* ]] ||
  fail "wrong receipt was not refused: $wrong_receipt"

command_mismatch="$(run_probe --mode command-mismatch)"
printf '%s\n' "$command_mismatch" | grep -Fq \
  "decision=$(manifest_value command_mismatch_decision) control_refused=true material_mutation=false" ||
  fail "Sounio command mismatch control diverged: $command_mismatch"

set +e
promotion="$(run_probe --mode promote-authority --handle "$handle" 2>&1)"
promotion_code=$?
set -e
[[ $promotion_code -eq 1 && "$promotion" == *'result-handle-authority-promotion-refused'* ]] ||
  fail "authority promotion was not refused: $promotion"

tampered_manifest="$TEST_ROOT/tampered.freeze.v1"
cp "$ROOT_DIR/tools/loom/exec_result_handle.freeze.v1" "$tampered_manifest"
printf 'tampered=true\n' >> "$tampered_manifest"
set +e
manifest_refusal="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_RESULT_HANDLE_RUNTIME="$AUTHORITY_RUNTIME" \
  SOUNIO_LOOM_EXEC_RESULT_HANDLE_MANIFEST="$tampered_manifest" \
  "$LOOM" exec-result-probe --root "$ROOT_DIR" --store "$STORE" \
    --mode resolve --handle "$handle" 2>&1)"
manifest_code=$?
set -e
[[ $manifest_code -eq 1 && "$manifest_refusal" == *'exec-result-manifest-hash-mismatch'* ]] ||
  fail "tampered manifest was not refused: $manifest_refusal"

tampered_runtime="$TEST_ROOT/tampered-runtime"
cp "$AUTHORITY_RUNTIME" "$tampered_runtime"
printf 'x' >> "$tampered_runtime"
chmod 0755 "$tampered_runtime"
set +e
runtime_refusal="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_RESULT_HANDLE_RUNTIME="$tampered_runtime" \
  "$LOOM" exec-result-probe --root "$ROOT_DIR" --store "$STORE" \
    --mode resolve --handle "$handle" 2>&1)"
runtime_code=$?
set -e
[[ $runtime_code -eq 1 && "$runtime_refusal" == *'exec-result-runtime-hash-mismatch'* ]] ||
  fail "tampered runtime was not refused: $runtime_refusal"

cp "$record" "$TEST_ROOT/record.original"
chmod 0600 "$record"
sed 's/^receipt_hex=./receipt_hex=f/' "$TEST_ROOT/record.original" > "$record"
chmod 0400 "$record"
set +e
receipt_tamper="$(run_probe --mode resolve --handle "$handle" 2>&1)"
receipt_tamper_code=$?
set -e
[[ $receipt_tamper_code -eq 1 && "$receipt_tamper" == *'result-record-receipt-hash-mismatch'* ]] ||
  fail "tampered receipt bytes were not refused: $receipt_tamper"

chmod 0600 "$record"
{
  sed -n '2p' "$TEST_ROOT/record.original"
  sed -n '1p' "$TEST_ROOT/record.original"
  sed -n '3,$p' "$TEST_ROOT/record.original"
} > "$record"
chmod 0400 "$record"
set +e
canonical_refusal="$(run_probe --mode resolve --handle "$handle" 2>&1)"
canonical_code=$?
set -e
[[ $canonical_code -eq 1 && "$canonical_refusal" == *'result-record-canonical-form-mismatch'* ]] ||
  fail "noncanonical record was not refused: $canonical_refusal"

sed 's/let canonical_record_rule record_text canonical = record_text = canonical/let canonical_record_rule _record_text _canonical = true/' \
  "$ROOT_DIR/tools/loom/src/loom_exec_result.ml" > "$TEST_ROOT/loom_exec_result.ml"
grep -Fq 'let canonical_record_rule _record_text _canonical = true' \
  "$TEST_ROOT/loom_exec_result.ml" || fail 'causal mutation did not alter the OCaml rule'
cat > "$TEST_ROOT/mutant_main.ml" <<'EOF'
let () =
  let result =
    Loom_exec_result.resolve ~root:Sys.argv.(1) ~store_root:Sys.argv.(2)
      ~handle:Sys.argv.(3) ~purpose:Loom_exec_result.Result_read
  in
  Printf.printf "MUTANT_ADMITTED record_sha256=%s\n" result.record_sha256
EOF
(
  cd "$TEST_ROOT"
  ocamlfind ocamlopt -package unix,cryptokit -linkpkg loom_exec_result.ml \
    mutant_main.ml -o mutant-resolver
)
mutant="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_RESULT_HANDLE_RUNTIME="$AUTHORITY_RUNTIME" \
  "$TEST_ROOT/mutant-resolver" "$ROOT_DIR" "$STORE" "$handle")"
[[ "$mutant" == MUTANT_ADMITTED* ]] ||
  fail "isolated rule deletion did not admit unchanged noncanonical record: $mutant"

chmod 0600 "$record"
cp "$TEST_ROOT/record.original" "$record"
chmod 0400 "$record"
run_probe --mode resolve --handle "$handle" >/dev/null

oracle_executed="$TEST_ROOT/oracle-executed"
mkdir "$TEST_ROOT/oracles"
for name in python python3 rustc cargo; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$oracle_executed" > "$TEST_ROOT/oracles/$name"
  chmod 0755 "$TEST_ROOT/oracles/$name"
done
PATH="$TEST_ROOT/oracles:$PATH" run_probe --mode command-mismatch >/dev/null
[[ ! -e "$oracle_executed" ]] || fail 'a prohibited Python or Rust oracle executed'
dependencies="$(ldd "$LOOM" 2>&1 || true; ldd "$AUTHORITY_RUNTIME" 2>&1 || true)"
printf '%s\n' "$dependencies" | grep -Eqi 'python|rust' &&
  fail 'the material or authority runtime has a prohibited dependency'

result="$(printf 'sounio-loom-exec-result-store-selftest: PASS semantic_authority=Sounio action=9033 operational_kernel=OCaml material_result_store=true result_store_attached=false publish=PASS resolve=PASS duplicate_publish=REFUSED receipt_tamper=REFUSED manifest_tamper=REFUSED runtime_tamper=REFUSED authority_promotion=REFUSED command_mismatch=DENY534 canonical_record_sabotage=PASS record_mode=0400 atomic_commit=link-no-replace+fsync receipt_sha256=%s manifest_sha256=%s record_sha256=%s authority_output_sha256=%s python_executed=false rust_executed=false exec_cell_attached=false provider_hook_switched=false production_activation=false parity_open=false claim_ready=false' \
  "$(manifest_value result_receipt_sha256)" \
  "$(sha256sum "$ROOT_DIR/tools/loom/exec_result_handle.freeze.v1" | cut -d ' ' -f 1)" \
  "$record_sha256" "$authority_output_sha256")"
[[ "$(evidence_value operational_source_sha256)" == \
   "$(sha256sum "$ROOT_DIR/tools/loom/src/loom_exec_result.ml" | cut -d ' ' -f 1)" ]] ||
  fail 'evidence operational source hash drifted'
[[ "$(evidence_value gate_sha256)" == \
   "$(sha256sum "$ROOT_DIR/scripts/ci/sounio_loom_exec_result_store_selftest.sh" | cut -d ' ' -f 1)" ]] ||
  fail 'evidence gate hash drifted'
[[ "$(evidence_value sounio_semantics_manifest_sha256)" == \
   "$(sha256sum "$ROOT_DIR/tools/loom/exec_result_handle.freeze.v1" | cut -d ' ' -f 1)" ]] ||
  fail 'evidence Sounio freeze hash drifted'
[[ "$(evidence_value loom_executable_sha256)" == \
   "$(sha256sum "$LOOM" | cut -d ' ' -f 1)" ]] ||
  fail 'evidence Loom executable hash drifted'
[[ "$(evidence_value result)" == "$result" ]] || fail 'evidence result drifted'
[[ "$(evidence_value result_sha256)" == \
   "$(printf '%s' "$result" | sha256sum | cut -d ' ' -f 1)" ]] ||
  fail 'evidence result hash drifted'
printf '%s\n' "$result"
