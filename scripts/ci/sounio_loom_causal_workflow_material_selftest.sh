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
printf 'ARM RUN_EXACT %s %s\n' "$LARGE_TICKET" "$LARGE_ARTIFACT_SHA256" >&"${LARGE_CELL[1]}"
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
printf 'ARM RUN_EXACT %s %s\n' "$OVERFLOW_TICKET" "$OVERFLOW_ARTIFACT_SHA256" >&"${OVERFLOW_CELL[1]}"
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
printf 'ARM RUN_EXACT %s %s\n' "$DESCENDANT_TICKET" "$DESCENDANT_ARTIFACT_SHA256" >&"${DESCENDANT_CELL[1]}"
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
printf 'ARM RUN_EXACT %s %s\n' "$RUN_TICKET" "$EXPECTED_ARTIFACT" >&"${RUN_CELL[1]}"
IFS= read -r -u "${RUN_CELL[0]}" run_result || fail 'RUN_EXACT result absent'
[[ "$run_result" == 'LOOM_CAUSAL_MATERIAL_CELL_RESULT_V1 mode=RUN_EXACT '* ]] ||
  fail "RUN_EXACT result malformed: $run_result"
RUN_RECORD_SHA256="$(field "$run_result" record_sha256)"
RUN_HANDLE="$(field "$run_result" handle)"
[[ "$(field "$run_result" exit_code)" == 0 && \
   "$(field "$run_result" stdout_sha256)" == e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 && \
   "$(field "$run_result" stderr_sha256)" == e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 && \
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

printf 'sounio-loom-causal-workflow-material-selftest: PASS semantic_authority=Sounio action=9037 material_language=C++20 material_role=MATERIAL_PARITY compile_actual=true run_exact_actual=true attest_actual=true source_sha256=%s artifact_sha256=%s result_record_sha256=%s attestation_record_sha256=%s concurrent_capture_large_output=true output_limit_negative=true stdout_limit_bytes=1048576 stderr_limit_bytes=1048576 execution_timeout_milliseconds=15000 extinction_timeout_milliseconds=2000 process_group_owned=true process_group_extinct=true escaped_descendant_extinct=true cell_local_descendants_extinct=true host_cgroup_extinction_measured=false inherited_descriptors=true descriptor_binding_schema=LOOM_CAUSAL_MATERIAL_DESCRIPTORS/2 arbitrary_path=false compile_extra_field_sabotage=REFUSED wrong_semantic_action_sabotage=REFUSED exact_attest_schema=true canonical_typed_handle_output=true typed_handle_persistence=false handle_is_bearer=false handle_is_execution_authority=false dynamic_user=false hostguardian_attached=false controller_recovery=false pod_loss_measured=false python_executed=false rust_executed=false parity_open=false claim_ready=false\n' \
  "$SOURCE_SHA256" "$EXPECTED_ARTIFACT" "$RUN_RECORD_SHA256" "$ATTEST_RECORD_SHA256"
