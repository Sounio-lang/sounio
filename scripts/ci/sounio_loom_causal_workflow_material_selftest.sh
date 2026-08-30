#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CELL="$ROOT_DIR/tools/loom/_build/default/src/loom-causal-workflow-material-cell"
SOURCE="$ROOT_DIR/tests/verify-ir/call_b.sio"
COMPILER="$ROOT_DIR/bin/souc-lean-single-x86_64"
SEMANTICS="$ROOT_DIR/tools/loom/causal_workflow_kernel.freeze.v1"
EXPECTED_SOURCE='899d05ffe60528a6b71871e24fa0d1bc105cd033b7ae2c5a0a6d2bb808cdcad9'
EXPECTED_ARTIFACT='eff2ac0ef28b34d6cc4f008cfb08a30ba18a0874c8654c06a3c62ec2f48a249c'

fail() {
  printf 'sounio-loom-causal-workflow-material-selftest: FAIL reason=%s material_execution=false\n' "$*" >&2
  exit 1
}

field() {
  local line="$1" key="$2" token
  for token in $line; do
    [[ "$token" == "$key="* ]] && {
      printf '%s\n' "${token#*=}"
      return 0
    }
  done
  fail "field omitted: $key"
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

for tool in c++ sha256sum mktemp chmod stat; do
  command -v "$tool" >/dev/null 2>&1 || fail "required tool absent: $tool"
done
bash "$ROOT_DIR/scripts/dev/build_loom_causal_workflow_material_cell.sh" >/dev/null
[[ "$(sha256_file "$SOURCE")" == "$EXPECTED_SOURCE" ]] || fail 'frozen source drifted'

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-causal-material-selftest.XXXXXX")"
cleanup() {
  set +e
  [[ -n "${RUN_CELL_PID:-}" ]] && kill "$RUN_CELL_PID" 2>/dev/null || true
  [[ -n "${ATTEST_CELL_PID:-}" ]] && kill "$ATTEST_CELL_PID" 2>/dev/null || true
  chmod -R u+rwX "$WORK" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

"$COMPILER" "$SOURCE" "$WORK/artifact.elf"
chmod 0555 "$WORK/artifact.elf"
[[ "$(sha256_file "$WORK/artifact.elf")" == "$EXPECTED_ARTIFACT" ]] ||
  fail 'compiled artifact diverged from frozen Sounio result'

SOURCE_SHA256="$(sha256_file "$SOURCE")"
COMPILER_SHA256="$(sha256_file "$COMPILER")"
SEMANTICS_SHA256="$(sha256_file "$SEMANTICS")"
cat > "$WORK/compile.record" <<EOF
loom-exec-result-record-v1
operation=sounio-check
source_sha256=$SOURCE_SHA256
compiler_sha256=$COMPILER_SHA256
artifact_sha256=$EXPECTED_ARTIFACT
exit_code=0
EOF
chmod 0400 "$WORK/compile.record"
cat > "$WORK/hardware.record" <<EOF
schema=loom-causal-hardware-record-v1
architecture=$(uname -m)
kernel=$(uname -r)
EOF
chmod 0400 "$WORK/hardware.record"

RUN_TICKET="$(printf 'local-material-run-ticket-v1' | sha256sum | cut -d ' ' -f 1)"
coproc RUN_CELL {
  exec 3<"$WORK/artifact.elf"
  LISTEN_FDS=1 LISTEN_FDNAMES=artifact LISTEN_PID="$BASHPID" \
    exec "$CELL" --mode RUN_EXACT
}
IFS= read -r -u "${RUN_CELL[0]}" run_ready || fail 'RUN_EXACT READY absent'
[[ "$run_ready" == 'LOOM_CAUSAL_MATERIAL_CELL_READY_V1 mode=RUN_EXACT '* ]] ||
  fail "RUN_EXACT READY malformed: $run_ready"
printf 'ARM RUN_EXACT %s %s\n' "$RUN_TICKET" "$EXPECTED_ARTIFACT" >&"${RUN_CELL[1]}"
IFS= read -r -u "${RUN_CELL[0]}" run_result || fail 'RUN_EXACT result absent'
[[ "$run_result" == 'LOOM_CAUSAL_MATERIAL_CELL_RESULT_V1 mode=RUN_EXACT '* ]] ||
  fail "RUN_EXACT result malformed: $run_result"
RUN_RECORD_SHA256="$(field "$run_result" record_sha256)"
[[ "$(field "$run_result" exit_code)" == 0 && \
   "$(field "$run_result" stdout_sha256)" == e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 && \
   "$(field "$run_result" stderr_sha256)" == e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 ]] ||
  fail 'RUN_EXACT observation diverged'
IFS= read -r -u "${RUN_CELL[0]}" run_record_begin || fail 'RUN_EXACT record begin absent'
[[ "$run_record_begin" == LOOM_CAUSAL_MATERIAL_RECORD_BEGIN ]] ||
  fail 'RUN_EXACT record begin malformed'
: > "$WORK/result.record"
while IFS= read -r -u "${RUN_CELL[0]}" line; do
  [[ "$line" == LOOM_CAUSAL_MATERIAL_RECORD_END ]] && break
  printf '%s\n' "$line" >> "$WORK/result.record"
done
chmod 0400 "$WORK/result.record"
[[ "$(sha256_file "$WORK/result.record")" == "$RUN_RECORD_SHA256" ]] ||
  fail 'RUN_EXACT record transport drifted'
printf 'CLOSE RUN_EXACT %s\n' "$RUN_RECORD_SHA256" >&"${RUN_CELL[1]}"
IFS= read -r -u "${RUN_CELL[0]}" run_closed || fail 'RUN_EXACT CLOSE absent'
[[ "$run_closed" == 'LOOM_CAUSAL_MATERIAL_CELL_CLOSED_V1 mode=RUN_EXACT '* ]] ||
  fail 'RUN_EXACT CLOSE malformed'
wait "$RUN_CELL_PID"
unset RUN_CELL_PID

COMPILE_RECORD_SHA256="$(sha256_file "$WORK/compile.record")"
HARDWARE_SHA256="$(sha256_file "$WORK/hardware.record")"
coproc ATTEST_CELL {
  exec 3<"$WORK/compile.record" 4<"$WORK/hardware.record" \
    5<"$WORK/result.record" 6<"$SEMANTICS"
  LISTEN_FDS=4 \
    LISTEN_FDNAMES=compile_record:hardware_record:result_record:semantics_manifest \
    LISTEN_PID="$BASHPID" exec "$CELL" --mode ATTEST
}
IFS= read -r -u "${ATTEST_CELL[0]}" attest_ready || fail 'ATTEST READY absent'
[[ "$attest_ready" == 'LOOM_CAUSAL_MATERIAL_CELL_READY_V1 mode=ATTEST '* ]] ||
  fail "ATTEST READY malformed: $attest_ready"
printf 'ARM ATTEST %s %s %s %s\n' \
  "$COMPILE_RECORD_SHA256" "$RUN_RECORD_SHA256" "$SEMANTICS_SHA256" \
  "$HARDWARE_SHA256" >&"${ATTEST_CELL[1]}"
IFS= read -r -u "${ATTEST_CELL[0]}" attest_result || fail 'ATTEST result absent'
[[ "$attest_result" == 'LOOM_CAUSAL_MATERIAL_CELL_RESULT_V1 mode=ATTEST '* ]] ||
  fail "ATTEST result malformed: $attest_result"
ATTEST_RECORD_SHA256="$(field "$attest_result" record_sha256)"
IFS= read -r -u "${ATTEST_CELL[0]}" attest_record_begin || fail 'ATTEST record begin absent'
[[ "$attest_record_begin" == LOOM_CAUSAL_MATERIAL_RECORD_BEGIN ]] ||
  fail 'ATTEST record begin malformed'
: > "$WORK/attestation.record"
while IFS= read -r -u "${ATTEST_CELL[0]}" line; do
  [[ "$line" == LOOM_CAUSAL_MATERIAL_RECORD_END ]] && break
  printf '%s\n' "$line" >> "$WORK/attestation.record"
done
[[ "$(sha256_file "$WORK/attestation.record")" == "$ATTEST_RECORD_SHA256" ]] ||
  fail 'ATTEST record transport drifted'
printf 'CLOSE ATTEST %s\n' "$ATTEST_RECORD_SHA256" >&"${ATTEST_CELL[1]}"
IFS= read -r -u "${ATTEST_CELL[0]}" attest_closed || fail 'ATTEST CLOSE absent'
[[ "$attest_closed" == 'LOOM_CAUSAL_MATERIAL_CELL_CLOSED_V1 mode=ATTEST '* ]] ||
  fail 'ATTEST CLOSE malformed'
wait "$ATTEST_CELL_PID"
unset ATTEST_CELL_PID

printf 'sounio-loom-causal-workflow-material-selftest: PASS semantic_authority=Sounio action=9037 material_language=C++20 material_role=MATERIAL_PARITY compile_actual=true run_exact_actual=true attest_actual=true source_sha256=%s artifact_sha256=%s result_record_sha256=%s attestation_record_sha256=%s inherited_descriptors=true arbitrary_path=false handle_is_bearer=false handle_is_execution_authority=false dynamic_user=false hostguardian_attached=false controller_recovery=false pod_loss_measured=false python_executed=false rust_executed=false parity_open=false claim_ready=false\n' \
  "$SOURCE_SHA256" "$EXPECTED_ARTIFACT" "$RUN_RECORD_SHA256" "$ATTEST_RECORD_SHA256"
