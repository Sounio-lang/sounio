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
PYTHON_PATH="$(command -v python3 || true)"

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
    "$LOOM" subprocess-membrane-probe \
      --root "$ROOT_DIR" --cwd "$ROOT_DIR" --scope "$scope" \
      --deadline-ms "$deadline_ms" -- "$@"
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
mkdir -p "$SCOPE"

positive="$(expect_probe 0 'kind=1 exit=0' 2000 "$SCOPE" /usr/bin/true)"
[[ "$positive" == *'decision_code=0'* && "$positive" == *'timed_out=false'* ]] ||
  fail "positive leaf was not admitted cleanly: $positive"

inside="$SCOPE/inside-write"
inside_result="$(expect_probe 0 'decision_code=0' 3000 "$SCOPE" \
  /bin/sh -c "printf OK > '$inside'")"
[[ -f "$inside" && "$(<"$inside")" == OK ]] ||
  fail "in-scope mechanical write did not materialize: $inside_result"

python_sentinel="$TEST_ROOT/python-executed"
python_result="$(expect_probe 126 'decision_code=410' 2000 "$SCOPE" \
  "$PYTHON_PATH" -c "open('$python_sentinel', 'w').write('BAD')")"
[[ ! -e "$python_sentinel" ]] || fail 'direct Python control executed'
[[ "$python_result" == *'kind=5'* ]] || fail 'direct Python did not stop at policy'

hidden_python_sentinel="$TEST_ROOT/hidden-python-executed"
hidden_python_result="$(expect_probe 126 'decision_code=410' 5000 "$SCOPE" \
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
outside_result="$(expect_probe 126 'decision_code=422' 3000 "$SCOPE" \
  /bin/sh -c "printf BAD > '$outside'")"
[[ ! -e "$outside" ]] || fail 'out-of-scope write reached the filesystem'
[[ "$outside_result" == *'kind=5'* ]] || fail 'out-of-scope write was not stopped'

semantic="$SCOPE/semantic.sio"
semantic_result="$(expect_probe 126 'decision_code=413' 3000 "$SCOPE" \
  /bin/sh -c "printf BAD > '$semantic'")"
[[ ! -e "$semantic" ]] || fail 'non-Sounio semantic write reached the filesystem'
[[ "$semantic_result" == *'kind=5'* ]] || fail 'semantic write was not stopped'

mutation_target="$TEST_ROOT/mutation-target"
printf 'PRESERVE\n' > "$mutation_target"
mutation_result="$(expect_probe 126 'decision_code=422' 3000 "$SCOPE" \
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
fd_result="$(expect_probe 126 'decision_code=415' 3000 "$SCOPE" \
  "$fd_helper" "$fd_target")"
[[ -f "$fd_target" && "$(<"$fd_target")" == PRESERVE ]] ||
  fail 'unsupported fd mutation changed its target'
[[ "$fd_result" == *'kind=5'* ]] || fail 'unsupported fd mutation was not stopped'

late_sentinel="$SCOPE/late-descendant-write"
started_ns="$(date +%s%N)"
timeout_result="$(expect_probe 124 'timed_out=true' 300 "$SCOPE" \
  /bin/sh -c "(sleep 2; printf LATE > '$late_sentinel') & wait")"
ended_ns="$(date +%s%N)"
wall_ms="$(( (ended_ns - started_ns) / 1000000 ))"
[[ "$timeout_result" == *'kind=4'* && "$timeout_result" == *'signal=9'* ]] ||
  fail "timeout did not report terminated tree: $timeout_result"
[[ "$wall_ms" -lt 3000 ]] || fail "timeout exceeded wall bound: ${wall_ms}ms"
sleep 3
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
    --scope "$SCOPE" --deadline-ms 2000 -- /usr/bin/true 2>&1)"
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

set +e
final_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_LOG="$LOG" \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_TEST_FINAL_OUTCOME_INCOMPLETE=1 \
  "$LOOM" subprocess-membrane-probe --root "$ROOT_DIR" --cwd "$ROOT_DIR" \
    --scope "$SCOPE" --deadline-ms 2000 -- /usr/bin/true 2>&1)"
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

printf '%s\n' \
  "sounio-loom-subprocess-membrane-native-selftest: PASS semantic_authority=Sounio operational_realization=OCaml+C platform=linux-x86_64 root_exec=ALLOW hidden_python=DENY410+not-executed direct_python=DENY410+not-executed rust=DENY410+not-executed in_scope_write=ALLOW out_of_scope_write=DENY422+not-written semantic_write=DENY413+not-written path_mutation=DENY422+preserved fd_mutation=DENY415+preserved process_tree_timeout=SIGKILL+no-late-write timeout_wall_ms=$wall_ms missing_policy=refused-before-spawn policy_tamper=refused runtime_tamper=refused final_outcome_sabotage=DENY426+child-zero-overridden decision_log=hash-bound native_coverage_attested=false exec_attached=false commit_attached=false ci_attached=false"
