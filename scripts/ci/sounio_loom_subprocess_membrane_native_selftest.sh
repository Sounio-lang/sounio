#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_PARENT="$ROOT_DIR/tools/loom/_build"
mkdir -p "$TEST_PARENT"
TEST_ROOT="$(mktemp -d "$TEST_PARENT/subprocess-membrane-native.XXXXXX")"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
SCOPE="$TEST_ROOT/scope"
LOG="$TEST_ROOT/decisions.tsv"
RESIDENT_LOG="$TEST_ROOT/resident.tsv"
PYTHON_PATH="$(command -v python3 || true)"
export SOUNIO_LOOM_RESIDENT_RECEIPT_LOG="$RESIDENT_LOG"

cleanup() {
  if [[ "${SOUNIO_LOOM_KEEP_TEST_ROOT:-0}" != 1 ]]; then
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-subprocess-membrane-native-selftest: FAIL: %s test_root=%s\n' \
    "$*" "$TEST_ROOT" >&2
  exit 1
}

probe() {
  local deadline_ms="$1" scope="$2"
  shift 2
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_SUBPROCESS_MEMBRANE_LOG="$LOG" \
    SOUNIO_LOOM_RESIDENT_RECEIPT_LOG="$RESIDENT_LOG" \
    "$LOOM" subprocess-membrane-probe \
      --root "$ROOT_DIR" --cwd "$ROOT_DIR" --scope "$scope" \
      --deadline-ms "$deadline_ms" -- "$@"
}

field() {
  local output="$1" key="$2" token
  for token in $output; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s' "${token#*=}"
      return 0
    fi
  done
  fail "probe field is missing: $key"
}

expect_probe() {
  local expected_rc="$1" expected="$2" deadline_ms="$3" scope="$4"
  shift 4
  local output rc
  set +e
  output="$(probe "$deadline_ms" "$scope" "$@" 2>&1)"
  rc=$?
  set -e
  [[ "$rc" -eq "$expected_rc" ]] ||
    fail "probe rc=$rc expected=$expected_rc output=$output"
  [[ "$output" == *"$expected"* ]] ||
    fail "probe omitted $expected: $output"
  [[ "$output" == *'attachment=refused'* ]] ||
    fail "probe promoted attachment: $output"
  printf '%s' "$output"
}

[[ -n "$PYTHON_PATH" && -x "$PYTHON_PATH" ]] ||
  fail 'python3 path is required for the deliberate non-execution control'

bash "$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null
resident_runtime="$ROOT_DIR/tools/loom/.runtime/sounio-loom-resident-membrane-runtime"
resident_runtime_sha='5c432f4c56fb0be5c157fb12147566a5f74f2cc4cc1e25b46f37050eff1ac12b'
resident_runtime_target="sha256-$resident_runtime_sha/sounio-loom-resident-membrane-runtime"
[[ -L "$resident_runtime" && "$(readlink "$resident_runtime")" == "$resident_runtime_target" ]] ||
  fail 'resident runtime was not promoted through its content-addressed symlink'
resident_runtime_sum="$(sha256sum "$resident_runtime")"
[[ "${resident_runtime_sum%% *}" == "$resident_runtime_sha" ]] ||
  fail 'promoted resident runtime hash drifted'
[[ ! -w "$(realpath "$resident_runtime")" ]] ||
  fail 'content-addressed resident runtime remained writable'
resident_v2_runtime="$ROOT_DIR/tools/loom/.runtime/sounio-loom-resident-membrane-runtime-v2"
resident_v2_runtime_sha='da1de5041588f722e5b1904af3d13ac435a29e7bc254ec6b0a5df375116b0b44'
resident_v2_runtime_target="sha256-$resident_v2_runtime_sha/sounio-loom-resident-membrane-runtime-v2"
[[ -L "$resident_v2_runtime" && "$(readlink "$resident_v2_runtime")" == "$resident_v2_runtime_target" ]] ||
  fail 'resident v2 runtime was not promoted through its content-addressed symlink'
resident_v2_runtime_sum="$(sha256sum "$resident_v2_runtime")"
[[ "${resident_v2_runtime_sum%% *}" == "$resident_v2_runtime_sha" ]] ||
  fail 'promoted resident v2 runtime hash drifted'
[[ ! -w "$(realpath "$resident_v2_runtime")" ]] ||
  fail 'content-addressed resident v2 runtime remained writable'
mkdir -p "$SCOPE"

positive="$(expect_probe 0 'kind=1 exit=0' 15000 "$SCOPE" /usr/bin/true)"
[[ "$positive" == *'decision_code=0'* && "$positive" == *'timed_out=false'* ]] ||
  fail "positive leaf was not admitted cleanly: $positive"
[[ "$positive" == *'authority=resident-Sounio-v2'* && \
  "$positive" == *'closure_authority=Sounio'* && \
  "$positive" == *'closure_code=447'* && \
  "$positive" == *'closure_material=refused'* ]] ||
  fail "positive leaf did not materialize resident Sounio closure refusal: $positive"
closure_result_sha256="$(field "$positive" closure_result_sha256)"
[[ "$closure_result_sha256" =~ ^[0-9a-f]{64}$ ]] ||
  fail "closure result digest is malformed: $positive"
authority_pid="$(field "$positive" authority_pid)"
authority_generation="$(field "$positive" authority_generation_sha256)"
authority_sequence="$(field "$positive" authority_sequence)"
[[ "$authority_pid" =~ ^[1-9][0-9]*$ && \
  "$authority_generation" =~ ^[0-9a-f]{64}$ && \
  "$authority_sequence" =~ ^[1-9][0-9]*$ ]] ||
  fail "resident identity receipt is malformed: $positive"
[[ -s "$RESIDENT_LOG" ]] || fail 'resident authority receipt log is empty'
while IFS= read -r receipt; do
  [[ "$receipt" == *$'\tgeneration_sha256='"$authority_generation"$'\t'* && \
    "$receipt" == *$'\tpid='"$authority_pid"$'\t'* ]] ||
    fail "resident identity drifted inside one probe: $receipt"
done < "$RESIDENT_LOG"
grep -Fq $'\tevent=START\t' "$RESIDENT_LOG" || fail 'resident START receipt is missing'
grep -Fq $'\tevent=EFFECT_CLOSURE\t' "$RESIDENT_LOG" || fail 'resident effect-closure receipt is missing'
grep -Fq $'\tevent=STOP\t' "$RESIDENT_LOG" || fail 'resident STOP receipt is missing'
grep -Fq $'\tparent_9025_manifest_sha256=c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91\t' \
  "$RESIDENT_LOG" || fail 'resident action 9025 binding is missing'
[[ "$positive" == *'sandbox=bubblewrap'* && \
   "$positive" == *'sandbox_ready=true'* && \
   "$positive" == *'rootfs=readonly'* && \
   "$positive" == *'network=isolated'* && \
   "$positive" == *'inherited_fds=closed'* ]] ||
  fail "kernel sandbox receipt was incomplete: $positive"

inside="$SCOPE/inside-write"
inside_result="$(expect_probe 0 'decision_code=0' 15000 "$SCOPE" \
  /bin/sh -c "printf OK > '$inside'")"
[[ -f "$inside" && "$(<"$inside")" == OK ]] ||
  fail "in-scope mechanical write did not materialize: $inside_result"

python_sentinel="$TEST_ROOT/python-executed"
python_result="$(expect_probe 126 'decision_code=410' 2000 "$SCOPE" \
  "$PYTHON_PATH" -c "open('$python_sentinel', 'w').write('BAD')")"
[[ ! -e "$python_sentinel" ]] || fail 'direct Python control executed'
[[ "$python_result" == *'kind=5'* ]] || fail 'direct Python did not stop at policy'

hidden_python_sentinel="$TEST_ROOT/hidden-python-executed"
hidden_python_result="$(expect_probe 126 'decision_code=410' 15000 "$SCOPE" \
  /bin/sh -c "'$PYTHON_PATH' -c \"open('$hidden_python_sentinel', 'w').write('BAD')\"")"
[[ ! -e "$hidden_python_sentinel" ]] || fail 'Python behind shell executed'
[[ "$hidden_python_result" == *'kind=5'* ]] ||
  fail 'Python behind shell did not stop at its exec event'

rust_sentinel="$TEST_ROOT/rust-executed"
fake_rust="$TEST_ROOT/rustc"
printf '#!/bin/sh\nprintf BAD > %s\n' "$rust_sentinel" > "$fake_rust"
chmod 0755 "$fake_rust"
rust_result="$(expect_probe 126 'decision_code=410' 2000 "$SCOPE" "$fake_rust")"
[[ ! -e "$rust_sentinel" ]] || fail 'Rust-named control executed'
[[ "$rust_result" == *'kind=5'* ]] || fail 'Rust-named control did not stop at policy'

outside="$TEST_ROOT/outside-write"
outside_result="$(expect_probe 126 'decision_code=422' 15000 "$SCOPE" \
  /bin/sh -c "printf BAD > '$outside'")"
[[ ! -e "$outside" ]] || fail 'out-of-scope write reached the filesystem'
[[ "$outside_result" == *'kind=5'* ]] || fail 'out-of-scope write was not stopped'

kernel_outside="$TEST_ROOT/kernel-outside-write"
set +e
kernel_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_LOG="$LOG" \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_TEST_DISABLE_FS_OBSERVER=1 \
  "$LOOM" subprocess-membrane-probe --root "$ROOT_DIR" --cwd "$ROOT_DIR" \
    --scope "$SCOPE" --deadline-ms 15000 -- /bin/sh -c \
    "printf BAD > '$kernel_outside'" 2>&1)"
kernel_rc=$?
set -e
[[ "$kernel_rc" -ne 0 && "$kernel_output" == *'sandbox_ready=true'* && \
   "$kernel_output" == *'decision_code=0'* ]] ||
  fail "kernel filesystem backstop did not independently refuse: rc=$kernel_rc output=$kernel_output"
[[ ! -e "$kernel_outside" ]] ||
  fail 'filesystem observer sabotage escaped the read-only root'

inherited_target="$TEST_ROOT/inherited-fd-write"
: > "$inherited_target"
exec 9>>"$inherited_target"
set +e
inherited_output="$(probe 15000 "$SCOPE" /bin/sh -c 'printf BAD >&9' 2>&1)"
inherited_rc=$?
set -e
exec 9>&-
[[ "$inherited_rc" -ne 0 && "$inherited_output" == *'sandbox_ready=true'* && \
   "$inherited_output" == *'inherited_fds=closed'* ]] ||
  fail "inherited descriptor was not refused: rc=$inherited_rc output=$inherited_output"
[[ ! -s "$inherited_target" ]] || fail 'inherited writable descriptor escaped'

parent_net_namespace="$(readlink /proc/self/ns/net)"
network_output="$(expect_probe 0 'sandbox_ready=true' 15000 "$SCOPE" \
  /usr/bin/readlink /proc/self/ns/net)"
child_net_namespace="$(printf '%s\n' "$network_output" | sed -n '1p')"
[[ "$child_net_namespace" == net:\[*\] && \
   "$child_net_namespace" != "$parent_net_namespace" ]] ||
  fail "network namespace was not isolated: parent=$parent_net_namespace child=$child_net_namespace"

semantic="$SCOPE/semantic.sio"
semantic_result="$(expect_probe 126 'decision_code=413' 15000 "$SCOPE" \
  /bin/sh -c "printf BAD > '$semantic'")"
[[ ! -e "$semantic" ]] || fail 'non-Sounio semantic write reached the filesystem'
[[ "$semantic_result" == *'kind=5'* ]] || fail 'semantic write was not stopped'

mutation_target="$TEST_ROOT/mutation-target"
printf 'PRESERVE\n' > "$mutation_target"
mutation_result="$(expect_probe 126 'decision_code=422' 15000 "$SCOPE" \
  /usr/bin/rm "$mutation_target")"
[[ -f "$mutation_target" && "$(<"$mutation_target")" == PRESERVE ]] ||
  fail 'out-of-scope path mutation changed its target'
[[ "$mutation_result" == *'kind=5'* ]] || fail 'path mutation was not stopped'

fd_helper_source="$TEST_ROOT/fd-mutation.c"
fd_helper="$TEST_ROOT/fd-mutation"
cat > "$fd_helper_source" <<'EOF'
#include <fcntl.h>
#include <unistd.h>
int main(int argc, char **argv) {
  int descriptor;
  if (argc != 2) return 64;
  descriptor = open(argv[1], O_RDONLY);
  if (descriptor < 0) return 65;
  return ftruncate(descriptor, 0) == 0 ? 0 : 66;
}
EOF
cc -O2 -o "$fd_helper" "$fd_helper_source"
fd_target="$SCOPE/fd-target"
printf 'PRESERVE\n' > "$fd_target"
fd_result="$(expect_probe 126 'decision_code=415' 15000 "$SCOPE" \
  "$fd_helper" "$fd_target")"
[[ -f "$fd_target" && "$(<"$fd_target")" == PRESERVE ]] ||
  fail 'unsupported fd mutation changed its target'
[[ "$fd_result" == *'kind=5'* ]] || fail 'unsupported fd mutation was not stopped'

late_sentinel="$SCOPE/late-descendant-write"
started_ns="$(date +%s%N)"
timeout_result="$(expect_probe 124 'timed_out=true' 8000 "$SCOPE" \
  /bin/sh -c "(sleep 12; printf LATE > '$late_sentinel') & wait")"
ended_ns="$(date +%s%N)"
wall_ms="$(( (ended_ns - started_ns) / 1000000 ))"
[[ "$timeout_result" == *'kind=4'* && "$timeout_result" == *'signal=9'* ]] ||
  fail "timeout did not report terminated tree: $timeout_result"
[[ "$wall_ms" -lt 12000 ]] || fail "timeout exceeded wall bound: ${wall_ms}ms"
sleep 5
[[ ! -e "$late_sentinel" ]] || fail 'descendant survived timeout and wrote later'

missing_sentinel="$TEST_ROOT/missing-policy-executed"
set +e
missing_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_LOG="$LOG" \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_MANIFEST="$TEST_ROOT/missing.freeze.v1" \
  "$LOOM" subprocess-membrane-probe --root "$ROOT_DIR" --cwd "$ROOT_DIR" \
    --scope "$SCOPE" --deadline-ms 2000 -- /bin/sh -c \
    "printf BAD > '$missing_sentinel'" 2>&1)"
missing_rc=$?
set -e
[[ "$missing_rc" -eq 1 && "$missing_output" == *'subprocess-membrane-policy-missing'* ]] ||
  fail "missing policy did not fail closed: rc=$missing_rc output=$missing_output"
[[ ! -e "$missing_sentinel" ]] || fail 'missing policy executed the child'

tampered_manifest="$TEST_ROOT/tampered.freeze.v1"
cp "$ROOT_DIR/tools/loom/subprocess_membrane.freeze.v1" "$tampered_manifest"
printf '\n' >> "$tampered_manifest"
set +e
tamper_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_LOG="$LOG" \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_MANIFEST="$tampered_manifest" \
  "$LOOM" subprocess-membrane-probe --root "$ROOT_DIR" --cwd "$ROOT_DIR" \
    --scope "$SCOPE" --deadline-ms 15000 -- /usr/bin/true 2>&1)"
tamper_rc=$?
set -e
[[ "$tamper_rc" -eq 1 && "$tamper_output" == *'policy-hash-mismatch'* ]] ||
  fail "tampered policy did not fail closed: rc=$tamper_rc output=$tamper_output"

set +e
runtime_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_LOG="$LOG" \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_RUNTIME=/usr/bin/true \
  "$LOOM" subprocess-membrane-probe --root "$ROOT_DIR" --cwd "$ROOT_DIR" \
    --scope "$SCOPE" --deadline-ms 2000 -- /usr/bin/true 2>&1)"
runtime_rc=$?
set -e
[[ "$runtime_rc" -eq 1 && "$runtime_output" == *'runtime-hash-mismatch'* ]] ||
  fail "tampered runtime did not fail closed: rc=$runtime_rc output=$runtime_output"

resident_runtime_sentinel="$TEST_ROOT/resident-runtime-tamper-executed"
set +e
resident_runtime_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_LOG="$LOG" \
  SOUNIO_LOOM_RESIDENT_RECEIPT_LOG="$RESIDENT_LOG" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_RUNTIME=/usr/bin/true \
  "$LOOM" subprocess-membrane-probe --root "$ROOT_DIR" --cwd "$ROOT_DIR" \
    --scope "$SCOPE" --deadline-ms 2000 -- /bin/sh -c \
    "printf BAD > '$resident_runtime_sentinel'" 2>&1)"
resident_runtime_rc=$?
set -e
[[ "$resident_runtime_rc" -eq 1 && \
  "$resident_runtime_output" == *'resident-runtime-hash-mismatch'* ]] ||
  fail "tampered resident runtime did not fail closed: rc=$resident_runtime_rc output=$resident_runtime_output"
[[ ! -e "$resident_runtime_sentinel" ]] ||
  fail 'tampered resident runtime executed the child'

resident_v2_manifest_sentinel="$TEST_ROOT/resident-v2-manifest-tamper-executed"
tampered_resident_v2_manifest="$TEST_ROOT/resident-v2.runtime"
cp "$ROOT_DIR/tools/loom/resident_membrane.runtime.v2" "$tampered_resident_v2_manifest"
printf '\n' >> "$tampered_resident_v2_manifest"
set +e
resident_v2_manifest_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_LOG="$LOG" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_MANIFEST="$tampered_resident_v2_manifest" \
  "$LOOM" subprocess-membrane-probe --root "$ROOT_DIR" --cwd "$ROOT_DIR" \
    --scope "$SCOPE" --deadline-ms 2000 -- /bin/sh -c \
    "printf BAD > '$resident_v2_manifest_sentinel'" 2>&1)"
resident_v2_manifest_rc=$?
set -e
[[ "$resident_v2_manifest_rc" -eq 1 && \
  "$resident_v2_manifest_output" == *'resident-runtime-v2-manifest-hash-mismatch'* ]] ||
  fail "tampered resident v2 manifest did not fail closed: rc=$resident_v2_manifest_rc output=$resident_v2_manifest_output"
[[ ! -e "$resident_v2_manifest_sentinel" ]] ||
  fail 'tampered resident v2 manifest executed the child'

closure_manifest_sentinel="$TEST_ROOT/closure-manifest-tamper-executed"
tampered_closure_manifest="$TEST_ROOT/effect-closure.freeze"
cp "$ROOT_DIR/tools/loom/effect_closure_authority.freeze.v1" "$tampered_closure_manifest"
printf '\n' >> "$tampered_closure_manifest"
set +e
closure_manifest_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_LOG="$LOG" \
  SOUNIO_LOOM_EFFECT_CLOSURE_MANIFEST="$tampered_closure_manifest" \
  "$LOOM" subprocess-membrane-probe --root "$ROOT_DIR" --cwd "$ROOT_DIR" \
    --scope "$SCOPE" --deadline-ms 2000 -- /bin/sh -c \
    "printf BAD > '$closure_manifest_sentinel'" 2>&1)"
closure_manifest_rc=$?
set -e
[[ "$closure_manifest_rc" -eq 1 && \
  "$closure_manifest_output" == *'effect-closure-manifest-hash-mismatch'* ]] ||
  fail "tampered effect-closure manifest did not fail closed: rc=$closure_manifest_rc output=$closure_manifest_output"
[[ ! -e "$closure_manifest_sentinel" ]] ||
  fail 'tampered effect-closure manifest executed the child'

sandbox_sentinel="$TEST_ROOT/sandbox-tamper-executed"
set +e
sandbox_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_LOG="$LOG" \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_SANDBOX=/usr/bin/true \
  "$LOOM" subprocess-membrane-probe --root "$ROOT_DIR" --cwd "$ROOT_DIR" \
    --scope "$SCOPE" --deadline-ms 2000 -- /bin/sh -c \
    "printf BAD > '$sandbox_sentinel'" 2>&1)"
sandbox_rc=$?
set -e
[[ "$sandbox_rc" -eq 1 && "$sandbox_output" == *'sandbox-hash-mismatch'* ]] ||
  fail "tampered sandbox did not fail closed: rc=$sandbox_rc output=$sandbox_output"
[[ ! -e "$sandbox_sentinel" ]] || fail 'tampered sandbox executed the child'

set +e
final_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_LOG="$LOG" \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_TEST_FINAL_OUTCOME_INCOMPLETE=1 \
  "$LOOM" subprocess-membrane-probe --root "$ROOT_DIR" --cwd "$ROOT_DIR" \
    --scope "$SCOPE" --deadline-ms 15000 -- /usr/bin/true 2>&1)"
final_rc=$?
set -e
[[ "$final_rc" -eq 126 && "$final_output" == *'kind=5'* && \
   "$final_output" == *'decision_code=426'* ]] ||
  fail "final Sounio refusal did not override child success: rc=$final_rc output=$final_output"

for code in 0 410 413 415 422 426; do
  grep -q $'\tcode='"$code"$'\t' "$LOG" || fail "decision log omitted code $code"
done
grep -q $'\tmanifest_sha256=0024178b8928f0c82d794d390244e83e5ce431054587fc7dd609c0f25c2e5b4f\t' \
  "$LOG" || fail 'decision log omitted frozen manifest binding'
grep -q $'\tsource_sha256=d72aa2e11d36ec0f6ff1e0048d2957aff5a1fb55ef2f960b9b3e13d0c25a992c\t' \
  "$LOG" || fail 'decision log omitted Sounio source binding'
grep -q $'\tsandbox_sha256=52231e1caf55bcbc667b269f49c63599a6f7db4767ae6a039580d0ff853db712\t' \
  "$LOG" || fail 'decision log omitted Bubblewrap mechanism binding'

printf '%s\n' \
  "sounio-loom-subprocess-membrane-native-selftest: PASS semantic_authority=Sounio operational_realization=OCaml+resident-Sounio-v2+C+Bubblewrap platform=linux-x86_64 runtime_promotion=v1+v2-content-addressed+readonly+atomic-symlink resident_identity=stable resident_sequence=correlated resident_receipts=START+EFFECT+EFFECT_CLOSURE+STOP closure_current=DENY447 closure_same_uid=unproven closure_material=refused resident_v2_runtime_tamper=refused-before-spawn resident_v2_manifest_tamper=refused-before-spawn closure_manifest_tamper=refused-before-spawn root_exec=ALLOW hidden_python=DENY410+not-executed direct_python=DENY410+not-executed rust=DENY410+not-executed in_scope_write=ALLOW out_of_scope_write=DENY422+not-written kernel_fs_backstop=observer-sabotaged+not-written inherited_fd=closed+not-written network_namespace=distinct semantic_write=DENY413+not-written path_mutation=DENY422+preserved fd_mutation=DENY415+preserved process_tree_timeout=SIGKILL+no-late-write timeout_wall_ms=$wall_ms missing_policy=refused-before-spawn policy_tamper=refused runtime_tamper=refused sandbox_tamper=refused-before-spawn final_outcome_sabotage=DENY426+child-zero-overridden decision_log=authority+mechanism-hash-bound native_coverage_attested=false exec_attached=false commit_attached=false ci_attached=false"
