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

descriptor_binding_one_sha256() {
  local process_fd="$1" name="$2" mode_hex mode
  mode_hex="$(stat -Lc '%f' "$process_fd")"
  mode="$((16#$mode_hex))"
  printf 'LOOM_CAUSAL_MATERIAL_DESCRIPTORS/2|name=%s|dev=%s|ino=%s|size=%s|mode=%s|uid=%s|gid=%s|content_sha256=%s' \
    "$name" \
    "$(stat -Lc '%d' "$process_fd")" \
    "$(stat -Lc '%i' "$process_fd")" \
    "$(stat -Lc '%s' "$process_fd")" \
    "$mode" \
    "$(stat -Lc '%u' "$process_fd")" \
    "$(stat -Lc '%g' "$process_fd")" \
    "$(sha256_file "$process_fd")" | sha256sum | cut -d ' ' -f 1
}

for tool in c++ sha256sum mktemp chmod stat; do
  command -v "$tool" >/dev/null 2>&1 || fail "required tool absent: $tool"
done
bash "$ROOT_DIR/scripts/dev/build_loom_causal_workflow_material_cell.sh" >/dev/null
[[ "$(sha256_file "$SOURCE")" == "$EXPECTED_SOURCE" ]] || fail 'frozen source drifted'

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-causal-material-selftest.XXXXXX")"
cleanup() {
  set +e
  [[ -n "${LARGE_CELL_PID:-}" ]] && kill "$LARGE_CELL_PID" 2>/dev/null || true
  [[ -n "${OVERFLOW_PID:-}" ]] && kill "$OVERFLOW_PID" 2>/dev/null || true
  [[ -n "${DESCENDANT_CELL_PID:-}" ]] && kill "$DESCENDANT_CELL_PID" 2>/dev/null || true
  [[ -n "${DESCENDANT_PID:-}" ]] && kill "$DESCENDANT_PID" 2>/dev/null || true
  [[ -n "${PROBE_CELL_PID:-}" ]] && kill "$PROBE_CELL_PID" 2>/dev/null || true
  [[ -n "${EARLY_CELL_PID:-}" ]] && kill "$EARLY_CELL_PID" 2>/dev/null || true
  [[ -n "${NONCE_CELL_PID:-}" ]] && kill "$NONCE_CELL_PID" 2>/dev/null || true
  [[ -n "${WITNESS_CELL_PID:-}" ]] && kill "$WITNESS_CELL_PID" 2>/dev/null || true
  [[ -n "${DUPLICATE_CELL_PID:-}" ]] && kill "$DUPLICATE_CELL_PID" 2>/dev/null || true
  [[ -n "${TIMEOUT_CELL_PID:-}" ]] && kill "$TIMEOUT_CELL_PID" 2>/dev/null || true
  [[ -n "${KILLED_CELL_PID:-}" ]] && kill "$KILLED_CELL_PID" 2>/dev/null || true
  [[ -n "${REPLACEMENT_A_CELL_PID:-}" ]] && kill "$REPLACEMENT_A_CELL_PID" 2>/dev/null || true
  [[ -n "${REPLACEMENT_B_CELL_PID:-}" ]] && kill "$REPLACEMENT_B_CELL_PID" 2>/dev/null || true
  [[ -n "${RUN_CELL_PID:-}" ]] && kill "$RUN_CELL_PID" 2>/dev/null || true
  [[ -n "${ATTEST_CELL_PID:-}" ]] && kill "$ATTEST_CELL_PID" 2>/dev/null || true
  chmod -R u+rwX "$WORK" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

MID_EXEC_RUNTIME="$WORK/sounio-mid-exec-authority"
SOUNIO_LOOM_CAUSAL_MID_EXEC_OUTPUT="$MID_EXEC_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_mid_exec_fixture.sh" \
  >/dev/null
SOUNIO_RELEASE_OUTPUT="$(printf '9037 1 3071 1045\n' | "$MID_EXEC_RUNTIME")"
[[ "${SOUNIO_RELEASE_OUTPUT%%$'\n'*}" == \
   'SOUNIO_CAUSAL_WORKFLOW_MID_EXEC RELEASE semantic_authority=Sounio action=9037 subordinate_contract=mid-exec-v1' ]] ||
  fail 'frozen Sounio release authority refused its positive frame'
SOUNIO_RELEASE_RECEIPT_SHA256="$(printf '%s' "$SOUNIO_RELEASE_OUTPUT" | sha256sum | cut -d ' ' -f 1)"
[[ "$SOUNIO_RELEASE_RECEIPT_SHA256" == \
   9002a805c0eb565fc2cc4706c278200bc0712a1059b6c0d2cb7178d132dfd31b ]] ||
  fail 'Sounio release receipt diverged from the frozen manifest'

arm_to_exec_barrier() {
  local output_fd="$1" input_fd="$2" ticket="$3" artifact="$4" label="$5"
  local canonical premature
  BARRIER_RUN_GRANT_GENERATION="$(printf 'run-grant:%s' "$label" | sha256sum | cut -d ' ' -f 1)"
  BARRIER_RAW_NONCE="$(printf 'raw-nonce:%s' "$label" | sha256sum | cut -d ' ' -f 1)"
  BARRIER_NONCE_SHA256="$(printf '%s' "$BARRIER_RAW_NONCE" | sha256sum | cut -d ' ' -f 1)"
  BARRIER_GUARDIAN_GENERATION="$(printf 'guardian:%s' "$label" | sha256sum | cut -d ' ' -f 1)"
  BARRIER_UNIT="loom-material-$label.service"
  BARRIER_INVOCATION_ID="$(printf 'invocation:%s' "$label" | sha256sum | cut -c1-32)"
  printf 'ARM RUN_EXACT %s %s %s %s %s %s %s\n' \
    "$ticket" "$artifact" "$BARRIER_RUN_GRANT_GENERATION" \
    "$BARRIER_NONCE_SHA256" "$BARRIER_GUARDIAN_GENERATION" \
    "$BARRIER_UNIT" "$BARRIER_INVOCATION_ID" >&"$input_fd"
  IFS= read -r -u "$output_fd" BARRIER_LINE || fail "$label exec barrier absent"
  [[ "$BARRIER_LINE" == 'LOOM_CAUSAL_MATERIAL_EXEC_BARRIER_V1 mode=RUN_EXACT '* && \
     "$(field "$BARRIER_LINE" semantic_authority)" == Sounio && \
     "$(field "$BARRIER_LINE" action)" == 9037 && \
     "$(field "$BARRIER_LINE" state)" == MATERIAL_RUNNING_IN_EXEC && \
     "$(field "$BARRIER_LINE" material_pid)" =~ ^[1-9][0-9]*$ && \
     "$(field "$BARRIER_LINE" material_start_tick)" =~ ^[1-9][0-9]*$ && \
     "$(field "$BARRIER_LINE" material_cgroup_sha256)" =~ ^[0-9a-f]{64}$ && \
     "$(field "$BARRIER_LINE" material_executable_sha256)" == "$artifact" && \
     "$(field "$BARRIER_LINE" run_ticket_sha256)" == "$ticket" && \
     "$(field "$BARRIER_LINE" run_grant_generation)" == "$BARRIER_RUN_GRANT_GENERATION" && \
     "$(field "$BARRIER_LINE" guardian_generation)" == "$BARRIER_GUARDIAN_GENERATION" && \
     "$(field "$BARRIER_LINE" unit)" == "$BARRIER_UNIT" && \
     "$(field "$BARRIER_LINE" unit_invocation_id)" == "$BARRIER_INVOCATION_ID" && \
     "$(field "$BARRIER_LINE" barrier_nonce_sha256)" == "$BARRIER_NONCE_SHA256" && \
     "$(field "$BARRIER_LINE" ptrace_event)" == PTRACE_EVENT_EXEC && \
     "$(field "$BARRIER_LINE" artifact_instruction_observed)" == false ]] ||
    fail "$label exec barrier binding invalid: $BARRIER_LINE"
  BARRIER_MATERIAL_PID="$(field "$BARRIER_LINE" material_pid)"
  BARRIER_MATERIAL_START_TICK="$(field "$BARRIER_LINE" material_start_tick)"
  BARRIER_MATERIAL_CGROUP_SHA256="$(field "$BARRIER_LINE" material_cgroup_sha256)"
  BARRIER_WITNESS_SHA256="$(field "$BARRIER_LINE" running_witness_sha256)"
  canonical="LOOM_CAUSAL_MATERIAL_RUNNING_IN_EXEC/1|guardian_generation=$BARRIER_GUARDIAN_GENERATION|unit=$BARRIER_UNIT|unit_invocation_id=$BARRIER_INVOCATION_ID|material_pid=$BARRIER_MATERIAL_PID|material_start_tick=$BARRIER_MATERIAL_START_TICK|material_cgroup_sha256=$BARRIER_MATERIAL_CGROUP_SHA256|run_grant_generation=$BARRIER_RUN_GRANT_GENERATION|barrier_nonce_sha256=$BARRIER_NONCE_SHA256|run_ticket_sha256=$ticket|artifact_sha256=$artifact|principal_sha256=$(field "$BARRIER_LINE" principal_sha256)|descriptor_binding_sha256=$(field "$BARRIER_LINE" descriptor_binding_sha256)"
  [[ "$(printf '%s' "$canonical" | sha256sum | cut -d ' ' -f 1)" == \
     "$BARRIER_WITNESS_SHA256" ]] || fail "$label canonical witness drifted"
  if IFS= read -r -t 0.05 -u "$output_fd" premature; then
    fail "$label emitted output/result before RELEASE: $premature"
  fi
  kill -0 "$BARRIER_MATERIAL_PID" 2>/dev/null ||
    fail "$label material process was not live at exec barrier"
}

release_exec_barrier() {
  local input_fd="$1"
  printf 'RELEASE RUN_EXACT %s %s %s\n' \
    "$BARRIER_WITNESS_SHA256" "$BARRIER_RAW_NONCE" \
    "$SOUNIO_RELEASE_RECEIPT_SHA256" >&"$input_fd"
}

"$COMPILER" "$SOURCE" "$WORK/artifact.elf"
chmod 0555 "$WORK/artifact.elf"
[[ "$(sha256_file "$WORK/artifact.elf")" == "$EXPECTED_ARTIFACT" ]] ||
  fail 'compiled artifact diverged from frozen Sounio result'

SOURCE_SHA256="$(sha256_file "$SOURCE")"
COMPILER_SHA256="$(sha256_file "$COMPILER")"
SEMANTICS_SHA256="$(sha256_file "$SEMANTICS")"
ARTIFACT_BYTES="$(stat -c '%s' "$WORK/artifact.elf")"
EMPTY_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
cat > "$WORK/compile.record" <<EOF
LOOM_EXEC_RESULT_RECORD/1
operation=sounio-check
event_sha256=$SOURCE_SHA256
command_template_sha256=$SEMANTICS_SHA256
generation_sha256=$COMPILER_SHA256
source_sha256=$SOURCE_SHA256
compiler_sha256=$COMPILER_SHA256
argv_sha256=$SOURCE_SHA256
artifact_sha256=$EXPECTED_ARTIFACT
artifact_bytes=$ARTIFACT_BYTES
stdout_sha256=$EMPTY_SHA256
stderr_sha256=$EMPTY_SHA256
diagnostics_sha256=$EMPTY_SHA256
sandbox_profile_sha256=$SEMANTICS_SHA256
principal_sha256=$SOURCE_SHA256
descriptor_binding_sha256=$COMPILER_SHA256
grant_receipt_sha256=$EXPECTED_ARTIFACT
exit_code=0
EOF
chmod 0400 "$WORK/compile.record"
cat > "$WORK/hardware.record" <<EOF
schema=loom-causal-hardware-record-v1
hostname=$(cat /etc/hostname)
kernel=$(uname -r)
boot_id=$(cat /proc/sys/kernel/random/boot_id)
EOF
chmod 0400 "$WORK/hardware.record"

cat > "$WORK/exec-barrier-probe.cpp" <<CPP
#include <fcntl.h>
#include <unistd.h>

int main() {
  const int marker = open("$WORK/artifact-instruction.marker",
                          O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (marker < 0) return 2;
  const char observed[] = "artifact-instruction-observed\n";
  const ssize_t written = write(marker, observed, sizeof(observed) - 1);
  return written == static_cast<ssize_t>(sizeof(observed) - 1) &&
                 close(marker) == 0
             ? 0
             : 3;
}
CPP
c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  "$WORK/exec-barrier-probe.cpp" -o "$WORK/exec-barrier-probe.elf"
chmod 0555 "$WORK/exec-barrier-probe.elf"
PROBE_ARTIFACT_SHA256="$(sha256_file "$WORK/exec-barrier-probe.elf")"
PROBE_TICKET="$(printf 'local-material-exec-barrier-probe-v1' | sha256sum | cut -d ' ' -f 1)"
coproc PROBE_CELL {
  exec 3<"$WORK/exec-barrier-probe.elf"
  LISTEN_FDS=1 LISTEN_FDNAMES=artifact LISTEN_PID="$BASHPID" \
    exec "$CELL" --mode RUN_EXACT
}
IFS= read -r -u "${PROBE_CELL[0]}" probe_ready || fail 'barrier probe READY absent'
[[ "$probe_ready" == 'LOOM_CAUSAL_MATERIAL_CELL_READY_V1 mode=RUN_EXACT '* ]] ||
  fail 'barrier probe READY malformed'
arm_to_exec_barrier "${PROBE_CELL[0]}" "${PROBE_CELL[1]}" \
  "$PROBE_TICKET" "$PROBE_ARTIFACT_SHA256" instruction-probe
[[ ! -e "$WORK/artifact-instruction.marker" ]] ||
  fail 'artifact instruction executed before authorized RELEASE'
release_exec_barrier "${PROBE_CELL[1]}"
IFS= read -r -u "${PROBE_CELL[0]}" probe_result || fail 'barrier probe result absent'
[[ "$probe_result" == 'LOOM_CAUSAL_MATERIAL_CELL_RESULT_V1 mode=RUN_EXACT '* && \
   "$(field "$probe_result" exit_code)" == 0 ]] ||
  fail 'barrier probe result malformed'
[[ -f "$WORK/artifact-instruction.marker" ]] ||
  fail 'artifact did not execute after authorized RELEASE'
PROBE_RECORD_SHA256="$(field "$probe_result" record_sha256)"
IFS= read -r -u "${PROBE_CELL[0]}" probe_begin || fail 'barrier probe record absent'
[[ "$probe_begin" == LOOM_CAUSAL_MATERIAL_RECORD_BEGIN ]] || fail 'barrier probe record malformed'
while IFS= read -r -u "${PROBE_CELL[0]}" line; do
  [[ "$line" == LOOM_CAUSAL_MATERIAL_RECORD_END ]] && break
done
printf 'CLOSE RUN_EXACT %s\n' "$PROBE_RECORD_SHA256" >&"${PROBE_CELL[1]}"
IFS= read -r -u "${PROBE_CELL[0]}" probe_closed || fail 'barrier probe CLOSE absent'
[[ "$probe_closed" == 'LOOM_CAUSAL_MATERIAL_CELL_CLOSED_V1 mode=RUN_EXACT '* ]] ||
  fail 'barrier probe CLOSE malformed'
wait "$PROBE_CELL_PID"
unset PROBE_CELL_PID

coproc EARLY_CELL {
  exec 2>"$WORK/early-release.stderr" 3<"$WORK/exec-barrier-probe.elf"
  LISTEN_FDS=1 LISTEN_FDNAMES=artifact LISTEN_PID="$BASHPID" \
    exec "$CELL" --mode RUN_EXACT
}
EARLY_PID="$EARLY_CELL_PID"
IFS= read -r -u "${EARLY_CELL[0]}" early_ready || fail 'early-release READY absent'
printf 'RELEASE RUN_EXACT %s %s %s\n' \
  "$PROBE_TICKET" "$PROBE_TICKET" "$PROBE_TICKET" >&"${EARLY_CELL[1]}"
if IFS= read -r -u "${EARLY_CELL[0]}" early_output; then
  fail "early RELEASE unexpectedly produced output: $early_output"
fi
if wait "$EARLY_PID"; then fail 'early RELEASE unexpectedly succeeded'; fi
unset EARLY_CELL_PID EARLY_PID
[[ "$(<"$WORK/early-release.stderr")" == *'RUN_EXACT arm frame malformed'* ]] ||
  fail 'early RELEASE refusal diverged'

coproc NONCE_CELL {
  exec 2>"$WORK/wrong-nonce.stderr" 3<"$WORK/exec-barrier-probe.elf"
  LISTEN_FDS=1 LISTEN_FDNAMES=artifact LISTEN_PID="$BASHPID" \
    exec "$CELL" --mode RUN_EXACT
}
NONCE_PID="$NONCE_CELL_PID"
IFS= read -r -u "${NONCE_CELL[0]}" nonce_ready || fail 'wrong-nonce READY absent'
arm_to_exec_barrier "${NONCE_CELL[0]}" "${NONCE_CELL[1]}" \
  "$PROBE_TICKET" "$PROBE_ARTIFACT_SHA256" wrong-nonce
NONCE_MATERIAL_PID="$BARRIER_MATERIAL_PID"
WRONG_RAW_NONCE="$(printf 'wrong-raw-nonce' | sha256sum | cut -d ' ' -f 1)"
printf 'RELEASE RUN_EXACT %s %s %s\n' \
  "$BARRIER_WITNESS_SHA256" "$WRONG_RAW_NONCE" \
  "$SOUNIO_RELEASE_RECEIPT_SHA256" >&"${NONCE_CELL[1]}"
if IFS= read -r -u "${NONCE_CELL[0]}" nonce_output; then
  fail "wrong nonce unexpectedly produced result: $nonce_output"
fi
if wait "$NONCE_PID"; then fail 'wrong nonce unexpectedly succeeded'; fi
unset NONCE_CELL_PID NONCE_PID
kill -0 "$NONCE_MATERIAL_PID" 2>/dev/null && fail 'wrong-nonce child survived refusal'
[[ "$(<"$WORK/wrong-nonce.stderr")" == *'release frame binding invalid'* ]] ||
  fail 'wrong nonce refusal diverged'

coproc WITNESS_CELL {
  exec 2>"$WORK/wrong-witness.stderr" 3<"$WORK/exec-barrier-probe.elf"
  LISTEN_FDS=1 LISTEN_FDNAMES=artifact LISTEN_PID="$BASHPID" \
    exec "$CELL" --mode RUN_EXACT
}
WITNESS_PID="$WITNESS_CELL_PID"
IFS= read -r -u "${WITNESS_CELL[0]}" witness_ready || fail 'wrong-witness READY absent'
arm_to_exec_barrier "${WITNESS_CELL[0]}" "${WITNESS_CELL[1]}" \
  "$PROBE_TICKET" "$PROBE_ARTIFACT_SHA256" wrong-witness
WITNESS_MATERIAL_PID="$BARRIER_MATERIAL_PID"
WRONG_WITNESS="$(printf 'wrong-running-witness' | sha256sum | cut -d ' ' -f 1)"
printf 'RELEASE RUN_EXACT %s %s %s\n' \
  "$WRONG_WITNESS" "$BARRIER_RAW_NONCE" "$SOUNIO_RELEASE_RECEIPT_SHA256" \
  >&"${WITNESS_CELL[1]}"
if IFS= read -r -u "${WITNESS_CELL[0]}" witness_output; then
  fail "wrong witness unexpectedly produced result: $witness_output"
fi
if wait "$WITNESS_PID"; then fail 'wrong witness unexpectedly succeeded'; fi
unset WITNESS_CELL_PID WITNESS_PID
kill -0 "$WITNESS_MATERIAL_PID" 2>/dev/null && fail 'wrong-witness child survived refusal'
[[ "$(<"$WORK/wrong-witness.stderr")" == *'release frame binding invalid'* ]] ||
  fail 'wrong witness refusal diverged'

coproc DUPLICATE_CELL {
  exec 2>"$WORK/duplicate-release.stderr" 3<"$WORK/exec-barrier-probe.elf"
  LISTEN_FDS=1 LISTEN_FDNAMES=artifact LISTEN_PID="$BASHPID" \
    exec "$CELL" --mode RUN_EXACT
}
DUPLICATE_PID="$DUPLICATE_CELL_PID"
IFS= read -r -u "${DUPLICATE_CELL[0]}" duplicate_ready || fail 'duplicate READY absent'
arm_to_exec_barrier "${DUPLICATE_CELL[0]}" "${DUPLICATE_CELL[1]}" \
  "$PROBE_TICKET" "$PROBE_ARTIFACT_SHA256" duplicate-release
DUPLICATE_MATERIAL_PID="$BARRIER_MATERIAL_PID"
printf 'RELEASE RUN_EXACT %s %s %s\nRELEASE RUN_EXACT %s %s %s\n' \
  "$BARRIER_WITNESS_SHA256" "$BARRIER_RAW_NONCE" "$SOUNIO_RELEASE_RECEIPT_SHA256" \
  "$BARRIER_WITNESS_SHA256" "$BARRIER_RAW_NONCE" "$SOUNIO_RELEASE_RECEIPT_SHA256" \
  >&"${DUPLICATE_CELL[1]}"
if IFS= read -r -u "${DUPLICATE_CELL[0]}" duplicate_output; then
  fail "duplicate RELEASE unexpectedly produced result: $duplicate_output"
fi
if wait "$DUPLICATE_PID"; then fail 'duplicate RELEASE unexpectedly succeeded'; fi
unset DUPLICATE_CELL_PID DUPLICATE_PID
kill -0 "$DUPLICATE_MATERIAL_PID" 2>/dev/null && fail 'duplicate-release child survived refusal'
[[ "$(<"$WORK/duplicate-release.stderr")" == *'duplicate or early post-release frame refused'* ]] ||
  fail 'duplicate RELEASE refusal diverged'

TIMEOUT_CELL="$WORK/material-cell-short-barrier-timeout"
c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -DLOOM_CAUSAL_BARRIER_HOLD_TIMEOUT_MS=150 \
  "$ROOT_DIR/tools/loom/src/loom_causal_workflow_material_cell.cpp" \
  -lcrypto -o "$TIMEOUT_CELL"
coproc TIMEOUT_RUN {
  exec 2>"$WORK/barrier-timeout.stderr" 3<"$WORK/exec-barrier-probe.elf"
  LISTEN_FDS=1 LISTEN_FDNAMES=artifact LISTEN_PID="$BASHPID" \
    exec "$TIMEOUT_CELL" --mode RUN_EXACT
}
TIMEOUT_CELL_PID="$TIMEOUT_RUN_PID"
IFS= read -r -u "${TIMEOUT_RUN[0]}" timeout_ready || fail 'timeout READY absent'
arm_to_exec_barrier "${TIMEOUT_RUN[0]}" "${TIMEOUT_RUN[1]}" \
  "$PROBE_TICKET" "$PROBE_ARTIFACT_SHA256" hold-timeout
TIMEOUT_MATERIAL_PID="$BARRIER_MATERIAL_PID"
if IFS= read -r -u "${TIMEOUT_RUN[0]}" timeout_output; then
  fail "hold timeout unexpectedly produced result: $timeout_output"
fi
if wait "$TIMEOUT_CELL_PID"; then fail 'barrier hold timeout unexpectedly succeeded'; fi
unset TIMEOUT_CELL_PID
kill -0 "$TIMEOUT_MATERIAL_PID" 2>/dev/null && fail 'hold-timeout child survived refusal'
[[ "$(<"$WORK/barrier-timeout.stderr")" == *'barrier hold timed out'* ]] ||
  fail 'barrier hold timeout refusal diverged'

coproc KILLED_CELL {
  exec 2>"$WORK/killed-child.stderr" 3<"$WORK/exec-barrier-probe.elf"
  LISTEN_FDS=1 LISTEN_FDNAMES=artifact LISTEN_PID="$BASHPID" \
    exec "$CELL" --mode RUN_EXACT
}
KILLED_PID="$KILLED_CELL_PID"
IFS= read -r -u "${KILLED_CELL[0]}" killed_ready || fail 'killed-child READY absent'
arm_to_exec_barrier "${KILLED_CELL[0]}" "${KILLED_CELL[1]}" \
  "$PROBE_TICKET" "$PROBE_ARTIFACT_SHA256" killed-child
KILLED_MATERIAL_PID="$BARRIER_MATERIAL_PID"
kill -KILL "$KILLED_MATERIAL_PID"
release_exec_barrier "${KILLED_CELL[1]}"
if IFS= read -r -u "${KILLED_CELL[0]}" killed_output; then
  fail "killed child unexpectedly produced result: $killed_output"
fi
if wait "$KILLED_PID"; then fail 'killed child unexpectedly succeeded'; fi
unset KILLED_CELL_PID KILLED_PID
kill -0 "$KILLED_MATERIAL_PID" 2>/dev/null && fail 'killed child remained live'
KILLED_REFUSAL="$(<"$WORK/killed-child.stderr")"
[[ ( "$KILLED_REFUSAL" == *'identity changed while exec barrier held'* ||
      "$KILLED_REFUSAL" == *'cannot read process identity:'* ||
      "$KILLED_REFUSAL" == *'cannot open process executable:'* ) ]] ||
  fail "killed child refusal diverged: $KILLED_REFUSAL"

coproc REPLACEMENT_A_CELL {
  exec 2>"$WORK/replacement-a.stderr" 3<"$WORK/exec-barrier-probe.elf"
  LISTEN_FDS=1 LISTEN_FDNAMES=artifact LISTEN_PID="$BASHPID" \
    exec "$CELL" --mode RUN_EXACT
}
REPLACEMENT_A_PID="$REPLACEMENT_A_CELL_PID"
IFS= read -r -u "${REPLACEMENT_A_CELL[0]}" replacement_a_ready || fail 'replacement A READY absent'
arm_to_exec_barrier "${REPLACEMENT_A_CELL[0]}" "${REPLACEMENT_A_CELL[1]}" \
  "$PROBE_TICKET" "$PROBE_ARTIFACT_SHA256" replacement-a
REPLACED_WITNESS_SHA256="$BARRIER_WITNESS_SHA256"
REPLACED_MATERIAL_PID="$BARRIER_MATERIAL_PID"
kill -KILL "$REPLACED_MATERIAL_PID"
release_exec_barrier "${REPLACEMENT_A_CELL[1]}"
if IFS= read -r -u "${REPLACEMENT_A_CELL[0]}" replacement_a_output; then
  fail "replacement A unexpectedly produced result: $replacement_a_output"
fi
if wait "$REPLACEMENT_A_PID"; then fail 'replacement A unexpectedly succeeded'; fi
unset REPLACEMENT_A_CELL_PID REPLACEMENT_A_PID

coproc REPLACEMENT_B_CELL {
  exec 2>"$WORK/replacement-b.stderr" 3<"$WORK/exec-barrier-probe.elf"
  LISTEN_FDS=1 LISTEN_FDNAMES=artifact LISTEN_PID="$BASHPID" \
    exec "$CELL" --mode RUN_EXACT
}
REPLACEMENT_B_PID="$REPLACEMENT_B_CELL_PID"
IFS= read -r -u "${REPLACEMENT_B_CELL[0]}" replacement_b_ready || fail 'replacement B READY absent'
arm_to_exec_barrier "${REPLACEMENT_B_CELL[0]}" "${REPLACEMENT_B_CELL[1]}" \
  "$PROBE_TICKET" "$PROBE_ARTIFACT_SHA256" replacement-b
REPLACEMENT_MATERIAL_PID="$BARRIER_MATERIAL_PID"
[[ "$REPLACEMENT_MATERIAL_PID" != "$REPLACED_MATERIAL_PID" || \
   "$BARRIER_WITNESS_SHA256" != "$REPLACED_WITNESS_SHA256" ]] ||
  fail 'replacement process did not change its witness identity'
printf 'RELEASE RUN_EXACT %s %s %s\n' \
  "$REPLACED_WITNESS_SHA256" "$BARRIER_RAW_NONCE" \
  "$SOUNIO_RELEASE_RECEIPT_SHA256" >&"${REPLACEMENT_B_CELL[1]}"
if IFS= read -r -u "${REPLACEMENT_B_CELL[0]}" replacement_b_output; then
  fail "replacement child unexpectedly produced result: $replacement_b_output"
fi
if wait "$REPLACEMENT_B_PID"; then fail 'replacement child unexpectedly succeeded'; fi
unset REPLACEMENT_B_CELL_PID REPLACEMENT_B_PID
kill -0 "$REPLACEMENT_MATERIAL_PID" 2>/dev/null && fail 'replacement child survived stale-witness refusal'
SOUNIO_REPLACEMENT_REFUSAL="$(printf '9037 1 3063 1045\n' | "$MID_EXEC_RUNTIME" || true)"
[[ "$SOUNIO_REPLACEMENT_REFUSAL" == \
   'SOUNIO_CAUSAL_WORKFLOW_MID_EXEC DENY593 semantic_authority=Sounio action=9037 subordinate_contract=mid-exec-v1' ]] ||
  fail 'Sounio replacement-process rule did not refuse with DENY593'

cat > "$WORK/large-output.cpp" <<'CPP'
#include <array>
#include <cstddef>
#include <unistd.h>

bool write_all(int descriptor, const char* bytes, std::size_t size) {
  while (size > 0) {
    const ssize_t written = write(descriptor, bytes, size);
    if (written < 0) return false;
    bytes += written;
    size -= static_cast<std::size_t>(written);
  }
  return true;
}

int main() {
  std::array<char, 4096> stdout_chunk{};
  std::array<char, 4096> stderr_chunk{};
  stdout_chunk.fill('O');
  stderr_chunk.fill('E');
#ifndef LOOM_OUTPUT_CHUNKS
#define LOOM_OUTPUT_CHUNKS 128
#endif
  for (std::size_t index = 0; index < LOOM_OUTPUT_CHUNKS; ++index) {
    if (!write_all(STDOUT_FILENO, stdout_chunk.data(), stdout_chunk.size()) ||
        !write_all(STDERR_FILENO, stderr_chunk.data(), stderr_chunk.size())) {
      return 2;
    }
  }
  return 0;
}
CPP
c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  "$WORK/large-output.cpp" -o "$WORK/large-output.elf"
chmod 0555 "$WORK/large-output.elf"
LARGE_ARTIFACT_SHA256="$(sha256_file "$WORK/large-output.elf")"
LARGE_TICKET="$(printf 'local-material-large-output-ticket-v1' | sha256sum | cut -d ' ' -f 1)"
coproc LARGE_CELL {
  exec 3<"$WORK/large-output.elf"
  LISTEN_FDS=1 LISTEN_FDNAMES=artifact LISTEN_PID="$BASHPID" \
    exec "$CELL" --mode RUN_EXACT
}
IFS= read -r -u "${LARGE_CELL[0]}" large_ready || fail 'large-output READY absent'
[[ "$large_ready" == 'LOOM_CAUSAL_MATERIAL_CELL_READY_V1 mode=RUN_EXACT '* && \
   "$(field "$large_ready" descriptor_binding_schema)" == LOOM_CAUSAL_MATERIAL_DESCRIPTORS/2 && \
   "$(field "$large_ready" executable_sha256)" == "$(sha256_file "$CELL")" && \
   "$(field "$large_ready" cgroup_sha256)" == "$(sha256_file "/proc/$LARGE_CELL_PID/cgroup")" && \
   "$(field "$large_ready" descriptor_binding_sha256)" == "$(descriptor_binding_one_sha256 "/proc/$LARGE_CELL_PID/fd/3" artifact)" ]] ||
  fail 'large-output READY identity/binding metadata invalid'
arm_to_exec_barrier "${LARGE_CELL[0]}" "${LARGE_CELL[1]}" \
  "$LARGE_TICKET" "$LARGE_ARTIFACT_SHA256" large-output
LARGE_WITNESS_SHA256="$BARRIER_WITNESS_SHA256"
release_exec_barrier "${LARGE_CELL[1]}"
IFS= read -r -u "${LARGE_CELL[0]}" large_result || fail 'large-output result absent'
[[ "$large_result" == 'LOOM_CAUSAL_MATERIAL_CELL_RESULT_V1 mode=RUN_EXACT '* ]] ||
  fail "large-output result malformed: $large_result"
LARGE_RECORD_SHA256="$(field "$large_result" record_sha256)"
LARGE_HANDLE="$(field "$large_result" handle)"
[[ "$(field "$large_result" exit_code)" == 0 && \
   "$(field "$large_result" stdout_bytes)" == 524288 && \
   "$(field "$large_result" stderr_bytes)" == 524288 && \
   "$(field "$large_result" stdout_sha256)" == 4d29dc1064614b31d5fafbaf147c02c81f8671f9d5f021b48322958427a772f6 && \
   "$(field "$large_result" stderr_sha256)" == 6166f0b4f9f52c4ab2fcb0825d86a538bd871248dbfe3d692f804c0c7a2454d4 && \
   "$(field "$large_result" stdout_limit_bytes)" == 1048576 && \
   "$(field "$large_result" stderr_limit_bytes)" == 1048576 && \
   "$(field "$large_result" execution_timeout_milliseconds)" == 15000 && \
   "$(field "$large_result" extinction_timeout_milliseconds)" == 2000 && \
   "$(field "$large_result" barrier_hold_timeout_milliseconds)" == 600000 && \
   "$(field "$large_result" running_witness_sha256)" == "$LARGE_WITNESS_SHA256" && \
   "$(field "$large_result" sounio_release_receipt_sha256)" == "$SOUNIO_RELEASE_RECEIPT_SHA256" && \
   "$(field "$large_result" post_exec_barrier)" == true && \
   "$(field "$large_result" release_bound)" == true && \
   "$(field "$large_result" process_group_owned)" == true && \
   "$(field "$large_result" process_group_extinct)" == true && \
   "$(field "$large_result" cell_local_descendants_extinct)" == true && \
   "$(field "$large_result" host_cgroup_extinction_measured)" == false && \
   "$(field "$large_result" handle_type)" == loom-result-v3 && \
   "$LARGE_HANDLE" == "loom-result-v3:$LARGE_ARTIFACT_SHA256:$LARGE_RECORD_SHA256" && \
   "$(printf '%s' "$LARGE_HANDLE" | sha256sum | cut -d ' ' -f 1)" == "$(field "$large_result" handle_sha256)" ]] ||
  fail 'large-output concurrent capture/extinction contract diverged'
IFS= read -r -u "${LARGE_CELL[0]}" large_record_begin || fail 'large-output record begin absent'
[[ "$large_record_begin" == LOOM_CAUSAL_MATERIAL_RECORD_BEGIN ]] ||
  fail 'large-output record begin malformed'
: > "$WORK/large-output.record"
while IFS= read -r -u "${LARGE_CELL[0]}" line; do
  [[ "$line" == LOOM_CAUSAL_MATERIAL_RECORD_END ]] && break
  printf '%s\n' "$line" >> "$WORK/large-output.record"
done
[[ "$(sha256_file "$WORK/large-output.record")" == "$LARGE_RECORD_SHA256" ]] ||
  fail 'large-output record transport drifted'
printf 'CLOSE RUN_EXACT %s\n' "$LARGE_RECORD_SHA256" >&"${LARGE_CELL[1]}"
IFS= read -r -u "${LARGE_CELL[0]}" large_closed || fail 'large-output CLOSE absent'
[[ "$large_closed" == 'LOOM_CAUSAL_MATERIAL_CELL_CLOSED_V1 mode=RUN_EXACT '* ]] ||
  fail 'large-output CLOSE malformed'
wait "$LARGE_CELL_PID"
unset LARGE_CELL_PID

c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -DLOOM_OUTPUT_CHUNKS=257 "$WORK/large-output.cpp" \
  -o "$WORK/output-overflow.elf"
chmod 0555 "$WORK/output-overflow.elf"
OVERFLOW_ARTIFACT_SHA256="$(sha256_file "$WORK/output-overflow.elf")"
OVERFLOW_TICKET="$(printf 'local-material-output-overflow-ticket-v1' | sha256sum | cut -d ' ' -f 1)"
coproc OVERFLOW_CELL {
  exec 2>"$WORK/output-overflow-cell.stderr"
  exec 3<"$WORK/output-overflow.elf"
  LISTEN_FDS=1 LISTEN_FDNAMES=artifact LISTEN_PID="$BASHPID" \
    exec "$CELL" --mode RUN_EXACT
}
OVERFLOW_PID="$OVERFLOW_CELL_PID"
IFS= read -r -u "${OVERFLOW_CELL[0]}" overflow_ready || fail 'output-overflow READY absent'
[[ "$overflow_ready" == 'LOOM_CAUSAL_MATERIAL_CELL_READY_V1 mode=RUN_EXACT '* ]] ||
  fail 'output-overflow READY malformed'
arm_to_exec_barrier "${OVERFLOW_CELL[0]}" "${OVERFLOW_CELL[1]}" \
  "$OVERFLOW_TICKET" "$OVERFLOW_ARTIFACT_SHA256" output-overflow
release_exec_barrier "${OVERFLOW_CELL[1]}"
if IFS= read -r -u "${OVERFLOW_CELL[0]}" overflow_result; then
  fail "output-overflow unexpectedly emitted result: $overflow_result"
fi
if wait "$OVERFLOW_PID"; then
  fail 'output-overflow execution unexpectedly succeeded'
fi
unset OVERFLOW_CELL_PID OVERFLOW_PID
OVERFLOW_REFUSAL="$(<"$WORK/output-overflow-cell.stderr")"
[[ ( "$OVERFLOW_REFUSAL" == *'LOOM_CAUSAL_MATERIAL_CELL_REFUSED reason=artifact stdout exceeded 1048576-byte limit material_execution=false'* ||
     "$OVERFLOW_REFUSAL" == *'LOOM_CAUSAL_MATERIAL_CELL_REFUSED reason=artifact stderr exceeded 1048576-byte limit material_execution=false'* ) ]] ||
  fail "output-overflow refusal diverged: $OVERFLOW_REFUSAL"

cat > "$WORK/escaped-descendant.cpp" <<CPP
#include <csignal>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

int main() {
  const pid_t child = fork();
  if (child < 0) return 2;
  if (child == 0) {
    if (setsid() < 0) _exit(3);
    const int marker = open("$WORK/escaped-descendant.pid",
                            O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (marker < 0) _exit(4);
    char text[64];
    const int size = snprintf(text, sizeof(text), "%lld\\n",
                              static_cast<long long>(getpid()));
    const ssize_t written =
        size > 0 ? write(marker, text, static_cast<size_t>(size)) : -1;
    if (size <= 0 || written != static_cast<ssize_t>(size) ||
        fsync(marker) != 0 || close(marker) != 0) {
      _exit(5);
    }
    for (;;) pause();
  }
  for (int attempt = 0; attempt < 2000; ++attempt) {
    if (access("$WORK/escaped-descendant.pid", F_OK) == 0) return 0;
    usleep(1000);
  }
  kill(child, SIGKILL);
  return 6;
}
CPP
c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  "$WORK/escaped-descendant.cpp" -o "$WORK/escaped-descendant.elf"
chmod 0555 "$WORK/escaped-descendant.elf"
DESCENDANT_ARTIFACT_SHA256="$(sha256_file "$WORK/escaped-descendant.elf")"
DESCENDANT_TICKET="$(printf 'local-material-escaped-descendant-ticket-v1' | sha256sum | cut -d ' ' -f 1)"
coproc DESCENDANT_CELL {
  exec 3<"$WORK/escaped-descendant.elf"
  LISTEN_FDS=1 LISTEN_FDNAMES=artifact LISTEN_PID="$BASHPID" \
    exec "$CELL" --mode RUN_EXACT
}
IFS= read -r -u "${DESCENDANT_CELL[0]}" descendant_ready || fail 'descendant READY absent'
[[ "$descendant_ready" == 'LOOM_CAUSAL_MATERIAL_CELL_READY_V1 mode=RUN_EXACT '* ]] ||
  fail 'descendant READY malformed'
arm_to_exec_barrier "${DESCENDANT_CELL[0]}" "${DESCENDANT_CELL[1]}" \
  "$DESCENDANT_TICKET" "$DESCENDANT_ARTIFACT_SHA256" escaped-descendant
release_exec_barrier "${DESCENDANT_CELL[1]}"
IFS= read -r -u "${DESCENDANT_CELL[0]}" descendant_result || fail 'descendant result absent'
[[ "$descendant_result" == 'LOOM_CAUSAL_MATERIAL_CELL_RESULT_V1 mode=RUN_EXACT '* && \
   "$(field "$descendant_result" exit_code)" == 0 && \
   "$(field "$descendant_result" process_group_extinct)" == true && \
   "$(field "$descendant_result" cell_local_descendants_extinct)" == true && \
   "$(field "$descendant_result" host_cgroup_extinction_measured)" == false ]] ||
  fail 'escaped descendant local-extinction contract diverged'
DESCENDANT_PID="$(<"$WORK/escaped-descendant.pid")"
[[ "$DESCENDANT_PID" =~ ^[1-9][0-9]*$ ]] || fail 'escaped descendant pid malformed'
if kill -0 "$DESCENDANT_PID" 2>/dev/null; then
  fail 'escaped descendant survived cell-local extinction'
fi
DESCENDANT_RECORD_SHA256="$(field "$descendant_result" record_sha256)"
IFS= read -r -u "${DESCENDANT_CELL[0]}" descendant_record_begin || fail 'descendant record begin absent'
[[ "$descendant_record_begin" == LOOM_CAUSAL_MATERIAL_RECORD_BEGIN ]] ||
  fail 'descendant record begin malformed'
while IFS= read -r -u "${DESCENDANT_CELL[0]}" line; do
  [[ "$line" == LOOM_CAUSAL_MATERIAL_RECORD_END ]] && break
done
printf 'CLOSE RUN_EXACT %s\n' "$DESCENDANT_RECORD_SHA256" >&"${DESCENDANT_CELL[1]}"
IFS= read -r -u "${DESCENDANT_CELL[0]}" descendant_closed || fail 'descendant CLOSE absent'
[[ "$descendant_closed" == 'LOOM_CAUSAL_MATERIAL_CELL_CLOSED_V1 mode=RUN_EXACT '* ]] ||
  fail 'descendant CLOSE malformed'
wait "$DESCENDANT_CELL_PID"
unset DESCENDANT_CELL_PID

RUN_TICKET="$(printf 'local-material-run-ticket-v1' | sha256sum | cut -d ' ' -f 1)"
coproc RUN_CELL {
  exec 3<"$WORK/artifact.elf"
  LISTEN_FDS=1 LISTEN_FDNAMES=artifact LISTEN_PID="$BASHPID" \
    exec "$CELL" --mode RUN_EXACT
}
IFS= read -r -u "${RUN_CELL[0]}" run_ready || fail 'RUN_EXACT READY absent'
[[ "$run_ready" == 'LOOM_CAUSAL_MATERIAL_CELL_READY_V1 mode=RUN_EXACT '* ]] ||
  fail "RUN_EXACT READY malformed: $run_ready"
arm_to_exec_barrier "${RUN_CELL[0]}" "${RUN_CELL[1]}" \
  "$RUN_TICKET" "$EXPECTED_ARTIFACT" frozen-run
RUN_WITNESS_SHA256="$BARRIER_WITNESS_SHA256"
RUN_GRANT_GENERATION="$BARRIER_RUN_GRANT_GENERATION"
RUN_NONCE_SHA256="$BARRIER_NONCE_SHA256"
RUN_MATERIAL_PID="$BARRIER_MATERIAL_PID"
RUN_MATERIAL_START_TICK="$BARRIER_MATERIAL_START_TICK"
RUN_MATERIAL_CGROUP_SHA256="$BARRIER_MATERIAL_CGROUP_SHA256"
release_exec_barrier "${RUN_CELL[1]}"
IFS= read -r -u "${RUN_CELL[0]}" run_result || fail 'RUN_EXACT result absent'
[[ "$run_result" == 'LOOM_CAUSAL_MATERIAL_CELL_RESULT_V1 mode=RUN_EXACT '* ]] ||
  fail "RUN_EXACT result malformed: $run_result"
RUN_RECORD_SHA256="$(field "$run_result" record_sha256)"
RUN_HANDLE="$(field "$run_result" handle)"
[[ "$(field "$run_result" exit_code)" == 0 && \
   "$(field "$run_result" stdout_sha256)" == e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 && \
   "$(field "$run_result" stderr_sha256)" == e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 && \
   "$(field "$run_result" running_witness_sha256)" == "$RUN_WITNESS_SHA256" && \
   "$(field "$run_result" run_grant_generation)" == "$RUN_GRANT_GENERATION" && \
   "$(field "$run_result" barrier_nonce_sha256)" == "$RUN_NONCE_SHA256" && \
   "$(field "$run_result" material_pid)" == "$RUN_MATERIAL_PID" && \
   "$(field "$run_result" material_start_tick)" == "$RUN_MATERIAL_START_TICK" && \
   "$(field "$run_result" material_cgroup_sha256)" == "$RUN_MATERIAL_CGROUP_SHA256" && \
   "$(field "$run_result" sounio_release_receipt_sha256)" == "$SOUNIO_RELEASE_RECEIPT_SHA256" && \
   "$(field "$run_result" post_exec_barrier)" == true && \
   "$(field "$run_result" release_bound)" == true && \
   "$(field "$run_result" handle_type)" == loom-result-v3 && \
   "$RUN_HANDLE" == "loom-result-v3:$EXPECTED_ARTIFACT:$RUN_RECORD_SHA256" ]] ||
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

cp "$WORK/compile.record" "$WORK/compile-extra.record"
chmod 0600 "$WORK/compile-extra.record"
printf 'injected_authority=true\n' >> "$WORK/compile-extra.record"
chmod 0400 "$WORK/compile-extra.record"
COMPILE_EXTRA_SHA256="$(sha256_file "$WORK/compile-extra.record")"
set +e
compile_extra_output="$({
  exec 3<"$WORK/compile-extra.record" 4<"$WORK/hardware.record" \
    5<"$WORK/result.record" 6<"$SEMANTICS"
  LISTEN_FDS=4 \
    LISTEN_FDNAMES=compile_record:hardware_record:result_record:semantics_manifest \
    LISTEN_PID="$BASHPID" exec "$CELL" --mode ATTEST <<< \
      "ARM ATTEST $COMPILE_EXTRA_SHA256 $RUN_RECORD_SHA256 $SEMANTICS_SHA256 $HARDWARE_SHA256"
} 2>&1)"
compile_extra_code=$?
set -e
[[ $compile_extra_code -eq 70 && \
   "$compile_extra_output" == *'ATTEST compile record schema cardinality invalid'* ]] ||
  fail "compile extra-field sabotage did not trigger exact-schema refusal: $compile_extra_output"

sed 's/semantic_action=9037/semantic_action=9036/' \
  "$WORK/result.record" > "$WORK/result-wrong-action.record"
chmod 0400 "$WORK/result-wrong-action.record"
RESULT_WRONG_ACTION_SHA256="$(sha256_file "$WORK/result-wrong-action.record")"
set +e
wrong_action_output="$({
  exec 3<"$WORK/compile.record" 4<"$WORK/hardware.record" \
    5<"$WORK/result-wrong-action.record" 6<"$SEMANTICS"
  LISTEN_FDS=4 \
    LISTEN_FDNAMES=compile_record:hardware_record:result_record:semantics_manifest \
    LISTEN_PID="$BASHPID" exec "$CELL" --mode ATTEST <<< \
      "ARM ATTEST $COMPILE_RECORD_SHA256 $RESULT_WRONG_ACTION_SHA256 $SEMANTICS_SHA256 $HARDWARE_SHA256"
} 2>&1)"
wrong_action_code=$?
set -e
[[ $wrong_action_code -eq 70 && \
   "$wrong_action_output" == *'ATTEST result posture invalid'* ]] ||
  fail "wrong semantic action sabotage did not trigger posture refusal: $wrong_action_output"

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
ATTEST_HANDLE="$(field "$attest_result" handle)"
[[ "$(field "$attest_result" handle_type)" == loom-attestation-v1 && \
   "$ATTEST_HANDLE" == "loom-attestation-v1:$SEMANTICS_SHA256:$RUN_RECORD_SHA256:$ATTEST_RECORD_SHA256" && \
   "$(field "$attest_result" running_witness_sha256)" == "$RUN_WITNESS_SHA256" && \
   "$(field "$attest_result" run_grant_generation)" == "$RUN_GRANT_GENERATION" && \
   "$(field "$attest_result" barrier_nonce_sha256)" == "$RUN_NONCE_SHA256" && \
   "$(field "$attest_result" sounio_release_receipt_sha256)" == "$SOUNIO_RELEASE_RECEIPT_SHA256" && \
   "$(printf '%s' "$ATTEST_HANDLE" | sha256sum | cut -d ' ' -f 1)" == "$(field "$attest_result" handle_sha256)" ]] ||
  fail 'ATTEST typed handle contract diverged'
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

printf 'sounio-loom-causal-workflow-material-selftest: PASS semantic_authority=Sounio action=9037 material_language=C++20 material_role=MATERIAL_PARITY compile_actual=true run_exact_actual=true attest_actual=true source_sha256=%s artifact_sha256=%s result_record_sha256=%s attestation_record_sha256=%s post_fexecve_barrier=PTRACE_EVENT_EXEC ptrace_exitkill=true artifact_instruction_pre_release=false release_receipt_sha256=%s wrong_nonce=REFUSED wrong_witness=REFUSED duplicate_release=REFUSED early_release=REFUSED barrier_hold_timeout=REFUSED killed_child=REFUSED replacement_child=REFUSED sounio_replacement_rule=DENY593 concurrent_capture_large_output=true output_limit_negative=true stdout_limit_bytes=1048576 stderr_limit_bytes=1048576 execution_timeout_milliseconds=15000 extinction_timeout_milliseconds=2000 barrier_hold_timeout_milliseconds=600000 process_group_owned=true process_group_extinct=true escaped_descendant_extinct=true cell_local_descendants_extinct=true host_cgroup_extinction_measured=false inherited_descriptors=true descriptor_binding_schema=LOOM_CAUSAL_MATERIAL_DESCRIPTORS/2 arbitrary_path=false compile_extra_field_sabotage=REFUSED wrong_semantic_action_sabotage=REFUSED exact_attest_schema=true canonical_typed_handle_output=true typed_handle_persistence=false handle_is_bearer=false handle_is_execution_authority=false dynamic_user=false hostguardian_attached=false controller_recovery=false pod_loss_measured=false python_executed=false rust_executed=false parity_open=false claim_ready=false\n' \
  "$SOURCE_SHA256" "$EXPECTED_ARTIFACT" "$RUN_RECORD_SHA256" \
  "$ATTEST_RECORD_SHA256" "$SOUNIO_RELEASE_RECEIPT_SHA256"
