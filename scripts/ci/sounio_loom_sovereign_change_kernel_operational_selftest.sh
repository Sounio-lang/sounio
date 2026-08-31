#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
POLICY_ROOT="${SOUNIO_LOOM_TEST_POLICY_ROOT:-$ROOT_DIR}"
LANGUAGE_AUTHORITY_ROOT="${SOUNIO_LOOM_TEST_LANGUAGE_AUTHORITY_ROOT:-$ROOT_DIR}"
MANIFEST="$POLICY_ROOT/tools/loom/sovereign_change_kernel.freeze.v1"
MATERIAL_MANIFEST="$POLICY_ROOT/tools/loom/sovereign_material_change.freeze.v2"
NATIVE_HOOK_MANIFEST="$ROOT_DIR/tools/loom/native_hook_cutover.freeze.v1"
LOOM="${SOUNIO_LOOM_TEST_RUNTIME:-$ROOT_DIR/tools/loom/_build/default/src/loom.exe}"
FIXTURE="$ROOT_DIR/tools/loom/_build/default/src/loom_change_provider_fixture.exe"
AUTHORITY="${SOUNIO_LOOM_TEST_CHANGE_AUTHORITY:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-sovereign-change-kernel}"
MATERIAL_AUTHORITY="${SOUNIO_LOOM_TEST_MATERIAL_CHANGE_AUTHORITY:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-sovereign-material-change}"
CI_ADMIT="${SOUNIO_LOOM_TEST_CI_ADMIT:-$ROOT_DIR/scripts/ci/sounio_loom_sovereign_change_receipt_admit.sh}"
EXPECTED_MANIFEST_SHA256=c84c5e7ff608f86ac51872de143516b0feb0981d0ee962583e2c62f66cbbacfb
EXPECTED_MATERIAL_MANIFEST_SHA256=662e01af4aed45ab22a0cfce283fd7aa9ec8775a65a2fb5a7a94a02c2c174c00
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-native-hook.XXXXXX")"
NATIVE_HOOK_AUTHORITY="$TEST_ROOT/sounio-loom-native-hook-cutover"
NATIVE_HOOK_TOOLCHAIN="$TEST_ROOT/native-hook-toolchain"
STATE_DIR="$TEST_ROOT/state"
COORD_DIR="$TEST_ROOT/coord"
ORACLE_DIR="$TEST_ROOT/no-oracle"
ORACLE_MARKER="$TEST_ROOT/prohibited-oracle-invoked"
REPORT="$TEST_ROOT/report.txt"
WORKTREE="$TEST_ROOT/worktree"
LANE=change-operational
SESSION_ID=change-operational-session

fail() {
  printf 'sounio-loom-sovereign-change-kernel-operational-selftest: FAIL: %s test_root=%s\n' \
    "$*" "$TEST_ROOT" >&2
  exit 1
}

session_dir() {
  printf '%s/sessions/codex--%s' "$STATE_DIR" "$LANE"
}

cleanup() {
  local state pid target
  while IFS= read -r state; do
    pid="$(sed -n 's/^daemon_pid=//p' "$state" | head -1)"
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || continue
    kill -KILL "$pid" 2>/dev/null || true
  done < <(find "$STATE_DIR" -name session.state -type f 2>/dev/null)
  if [[ -f "$REPORT.targets" ]]; then
    while IFS= read -r target; do
      case "$target" in
        "$WORKTREE"/tools/loom/.change-fixture-*) rm -f -- "$target" ;;
        *) printf 'refusing unexpected fixture cleanup path: %s\n' "$target" >&2 ;;
      esac
    done <"$REPORT.targets"
  fi
  if [[ "${SOUNIO_LOOM_KEEP_TEST_ROOT:-0}" == 1 ]]; then
    printf 'sounio-loom-sovereign-change-kernel-operational-selftest: retained=%s\n' \
      "$TEST_ROOT" >&2
  else
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT

wait_for_exit() {
  local attempt attempts state
  state="$(session_dir)/session.state"
  attempts="$((${SOUNIO_LOOM_TEST_EXIT_TIMEOUT_SECONDS:-120} * 20))"
  for attempt in $(seq 1 "$attempts"); do
    grep -q '^state=exited$' "$state" 2>/dev/null && return 0
    sleep 0.05
  done
  return 1
}

manifest_value() {
  sed -n "s/^$1=//p" "$MANIFEST" | head -1
}

material_manifest_value() {
  sed -n "s/^$1=//p" "$MATERIAL_MANIFEST" | head -1
}

decoded_events() {
  local kind="$1" hex escaped
  while IFS= read -r hex; do
    escaped="$(printf '%s' "$hex" | sed 's/../\\x&/g')"
    printf '%b' "$escaped"
    printf '\n'
  done < <(awk -F '\t' -v kind="$kind" '$5 == kind { print $6 }' "$journal")
}

mkdir -p "$STATE_DIR" "$COORD_DIR" "$ORACLE_DIR" "$WORKTREE/tools/loom"
for oracle in python python3 pypy pypy3 cargo rustc; do
  printf '#!/bin/sh\nprintf invoked >%q\nexit 97\n' "$ORACLE_MARKER" \
    >"$ORACLE_DIR/$oracle"
  chmod 0555 "$ORACLE_DIR/$oracle"
done

git -C "$WORKTREE" init --initial-branch=main >/dev/null
git -C "$WORKTREE" config user.name 'Loom Fixture'
git -C "$WORKTREE" config user.email 'loom-fixture@sounio.local'
printf 'sovereign change fixture\n' >"$WORKTREE/README.md"
git -C "$WORKTREE" add README.md
git -C "$WORKTREE" commit -m 'fixture seed' >/dev/null
seed_commit="$(git -C "$WORKTREE" rev-parse HEAD)"

[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "$EXPECTED_MANIFEST_SHA256" ]] ||
  fail 'frozen Sounio manifest hash diverged'
[[ "$(sha256sum "$MATERIAL_MANIFEST" | awk '{print $1}')" == "$EXPECTED_MATERIAL_MANIFEST_SHA256" ]] ||
  fail 'frozen Sounio material manifest hash diverged'
[[ "$(manifest_value semantic_authority)" == Sounio &&
   "$(manifest_value stage)" == SEMANTICS_FROZEN &&
   "$(manifest_value action)" == 9043 &&
   "$(manifest_value grant_resident_memory)" == true &&
   "$(manifest_value grant_is_bearer)" == false &&
   "$(manifest_value grant_single_use)" == true &&
   "$(manifest_value consume_atomic)" == true ]] ||
  fail 'frozen Sounio change contract is incomplete'
[[ "$(material_manifest_value semantic_authority)" == Sounio &&
   "$(material_manifest_value stage)" == SEMANTICS_FROZEN &&
   "$(material_manifest_value action)" == 9044 &&
   "$(material_manifest_value provider_root_readonly)" == true &&
   "$(material_manifest_value ci_policy)" == consume-not-reinterpret ]] ||
  fail 'frozen Sounio material change contract is incomplete'

(
  flock -x 9
  dune build --root "$ROOT_DIR/tools/loom" \
    src/loom.exe src/loom_change_provider_fixture.exe >/dev/null
) 9>"$ROOT_DIR/tools/loom/_build/.dune-build.lock"

# Dune owns _build and may remove undeclared artifacts, so the frozen Sounio
# executable is deliberately installed after the OCaml material runtime build.
SOUNIO_SOURCE_ROOT="$ROOT_DIR" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_sovereign_change_kernel.sh" >/dev/null
SOUNIO_SOURCE_ROOT="$ROOT_DIR" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_sovereign_material_change.sh" >/dev/null
native_hook_executable_commit="$(sed -n 's/^sounio_executable_commit=//p' "$NATIVE_HOOK_MANIFEST")"
mkdir -p "$NATIVE_HOOK_TOOLCHAIN"
git -C "$ROOT_DIR" archive "$native_hook_executable_commit" \
  bin/souc bin/souc-lean-single-x86_64 | tar -x -C "$NATIVE_HOOK_TOOLCHAIN"
SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_SOUC="$NATIVE_HOOK_TOOLCHAIN/bin/souc" \
  SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_OUTPUT="$NATIVE_HOOK_AUTHORITY" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_native_hook_cutover.sh" >/dev/null

[[ -x "$LOOM" && -x "$FIXTURE" && -x "$AUTHORITY" && -x "$MATERIAL_AUTHORITY" &&
   -x "$CI_ADMIT" ]] ||
  fail 'operational binaries are absent'
authority_sha256="$(sha256sum "$AUTHORITY" | awk '{print $1}')"
expected_authority_sha256="$(manifest_value executable_sha256)"
[[ "$authority_sha256" == "$expected_authority_sha256" ]] ||
  fail 'executed Sounio authority does not match the freeze manifest'

env -u BASH_ENV -u ENV \
  PATH="$ORACLE_DIR:$PATH" \
  SOUNIO_COORD_DIR="$COORD_DIR" \
  SOUNIO_COORD_RUNTIME_MODE=local \
  SOUNIO_COORD_NATIVE_HOOK_SELFTEST=1 \
  SOUNIO_COORD_NATIVE_HOOK_WAKE_SELFTEST=1 \
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_SOVEREIGN_CHANGE_MEDIATED=1 \
  SOUNIO_LOOM_SOVEREIGN_CHANGE_ROOT="$POLICY_ROOT" \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_ROOT="$LANGUAGE_AUTHORITY_ROOT" \
  SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_ROOT="$ROOT_DIR" \
  SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_RUNTIME="$NATIVE_HOOK_AUTHORITY" \
  SOUNIO_LOOM_NATIVE_HOOK_CONFIG="$ROOT_DIR/.codex/hooks.json" \
  SOUNIO_LOOM_TOOL_ROOT="$ROOT_DIR" \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_LOG="$TEST_ROOT/language-authority.tsv" \
  SOUNIO_LOOM_COORD_AUTO=0 \
  SOUNIO_LOOM_START_READY_TIMEOUT_SECONDS=120 \
  "$LOOM" start --state-dir "$STATE_DIR" --agent codex --lane "$LANE" \
    --session-id "$SESSION_ID" --cwd "$WORKTREE" -- \
    "$FIXTURE" "$LOOM" "$WORKTREE" "$REPORT" >/dev/null

wait_for_exit || fail 'Loom generation did not terminate'
output="$(find "$(session_dir)" -name output.bin -type f -print -quit)"
journal="$(find "$(session_dir)" -name journal.tsv -type f -print -quit)"
[[ -f "$output" && -f "$journal" ]] || fail 'Loom generation evidence is absent'
if [[ ! -f "$REPORT" ]]; then
  fail "provider fixture omitted its PASS report: $(tr '\n' ' ' <"$output")"
fi

expected_report='loom-change-provider-fixture: PASS language=OCaml role=OPERATIONAL_ATTACHMENT semantic_authority=false provider_root=READ_ONLY write=KERNEL_MATERIALIZED edit=KERNEL_MATERIALIZED apply_patch=KERNEL_MATERIALIZED replay=REFUSED wrong_stage=REFUSED+GRANT_BURNED direct_root_write=EROFS direct_git_commit=REFUSED widened_commit=REFUSED commit_replay=REFUSED kernel_commit=ADMITTED receipt=ISSUED python_executed=false rust_executed=false'
[[ "$(<"$REPORT")" == "$expected_report" ]] ||
  fail "provider fixture diverged: $(<"$REPORT")"
[[ ! -s "$output" ]] || fail "provider emitted unexpected output: $(tr '\n' ' ' <"$output")"

prepared="$(grep -c $'\tCHANGE_GRANT_PREPARED\t' "$journal")"
consumed="$(grep -c $'\tCHANGE_GRANT_CONSUMED\t' "$journal")"
refused="$(grep -c $'\tCHANGE_GRANT_REFUSED\t' "$journal")"
committed="$(grep -c $'\tCHANGE_COMMIT_ADMITTED\t' "$journal")"
commit_refused="$(grep -c $'\tCHANGE_COMMIT_REFUSED\t' "$journal")"
[[ "$prepared" -eq 5 && "$consumed" -eq 4 && "$refused" -eq 3 ]] ||
  fail "journal counts diverged: prepare=$prepared consume=$consumed deny=$refused"
[[ "$committed" -eq 1 ]] || fail "kernel commit count diverged: commit=$committed"
[[ "$commit_refused" -eq 1 ]] ||
  fail "kernel commit refusal count diverged: refused=$commit_refused"
prepare_audit="$(decoded_events CHANGE_GRANT_PREPARED)"
consume_audit="$(decoded_events CHANGE_GRANT_CONSUMED)"
deny_audit="$(decoded_events CHANGE_GRANT_REFUSED)"
[[ "$(grep -c '^decision=ALLOW$' <<<"$prepare_audit")" -eq 5 &&
   "$prepare_audit" == *'reason=action-9044-material-prepare-admit'* &&
   "$prepare_audit" == *'material_decision=SOUNIO_SOVEREIGN_MATERIAL_CHANGE MATERIAL_PREPARED'* &&
   "$prepare_audit" == *'provider_root_readonly=true'* &&
   "$prepare_audit" == *'git_common_readonly=true'* &&
   "$prepare_audit" == *'decision_authority=Sounio'* ]] ||
  fail 'prepare decisions are not fully auditable as Sounio ALLOW'
[[ "$(grep -c '^decision=ALLOW$' <<<"$consume_audit")" -eq 4 &&
   "$consume_audit" == *'reason=action-9044-material-consume-admit'* &&
   "$consume_audit" == *'material_decision=SOUNIO_SOVEREIGN_MATERIAL_CHANGE MATERIAL_CONSUMED'* &&
   "$consume_audit" == *'decision_authority=Sounio'* ]] ||
  fail 'consume decisions are not fully auditable as Sounio ALLOW'
[[ "$(grep -c '^decision=DENY$' <<<"$deny_audit")" -eq 3 &&
   "$deny_audit" == *'reason=change-grant-missing-or-replayed'* &&
   "$deny_audit" == *'reason=change-staged-post-image-mismatch'* &&
   "$deny_audit" == *'decision_authority=OCaml-structural-precondition'* ]] ||
  fail 'structural DENY decisions or reasons are not fully auditable'
verify_output="$("$LOOM" verify-journal --journal "$journal")"
[[ "$verify_output" == JOURNAL_OK*'phase=exited'* ]] ||
  fail "journal verification failed: $verify_output"

receipt_path="$(<"$REPORT.receipt")"
[[ -f "$receipt_path" ]] || fail 'kernel commit receipt is absent after provider exit'
commit_oid="$(git -C "$WORKTREE" rev-parse HEAD)"
[[ "$commit_oid" != "$seed_commit" ]] || fail 'kernel did not advance the Git branch'
[[ "$(git -C "$WORKTREE" status --porcelain)" == '' ]] ||
  fail 'kernel commit left the authorized file set dirty'
changed_paths="$(git -C "$WORKTREE" diff-tree --no-commit-id --name-only -r "$commit_oid")"
expected_paths="$(
  sed -n '1p;3p;5p' "$REPORT.targets" | sed "s#^$WORKTREE/##" | LC_ALL=C sort
)"
[[ "$(LC_ALL=C sort <<<"$changed_paths")" == "$expected_paths" ]] ||
  fail "kernel commit tree widened beyond the authorized file set: $changed_paths"
[[ "$(git -C "$WORKTREE" show "$commit_oid:$(sed -n '1s#^.*/worktree/##p' "$REPORT.targets")")" == after ]] ||
  fail 'committed Write/Edit bytes diverged'
[[ "$(git -C "$WORKTREE" show "$commit_oid:$(sed -n '3s#^.*/worktree/##p' "$REPORT.targets")")" == probe ]] ||
  fail 'committed apply_patch bytes diverged'
[[ "$(git -C "$WORKTREE" show "$commit_oid:$(sed -n '5s#^.*/worktree/##p' "$REPORT.targets")")" == bound ]] ||
  fail 'committed final Write bytes diverged'
admission_output="$(
  SOUNIO_LOOM_SOVEREIGN_CHANGE_ROOT="$POLICY_ROOT" \
    SOUNIO_LOOM_LANGUAGE_AUTHORITY_ROOT="$LANGUAGE_AUTHORITY_ROOT" \
    SOUNIO_LOOM_BIN="$LOOM" \
    "$CI_ADMIT" "$WORKTREE" "$receipt_path"
)"
[[ "$admission_output" == LOOM_CHANGE_CI_ADMITTED*'policy_executed_by_ci=false claim_ready=false'* &&
   "$admission_output" == *LOOM_CHANGE_CLAIM_READY*'semantic_authority=Sounio action=9044'*'claim_ready=true'* &&
   "$admission_output" == *SOUNIO_LOOM_CHANGE_ADMISSION*'PASS'* ]] ||
  fail "CI/claim admission chain diverged: $admission_output"

tampered="$receipt_path.tampered.receipt"
cp "$receipt_path" "$tampered"
printf 'tampered=31\n' >>"$tampered"
chmod 0600 "$tampered"
if SOUNIO_LOOM_SOVEREIGN_CHANGE_ROOT="$POLICY_ROOT" \
     "$LOOM" change-ci-admit --root "$WORKTREE" --receipt "$tampered" \
     >"$TEST_ROOT/tampered.out" 2>&1; then
  fail 'CI admitted a tampered receipt'
fi
grep -q 'change-ci-receipt-hash-mismatch' "$TEST_ROOT/tampered.out" ||
  fail "tampered receipt failed for the wrong reason: $(<"$TEST_ROOT/tampered.out")"

[[ ! -e "$ORACLE_MARKER" ]] || fail 'a prohibited Python or Rust oracle executed'

printf '%s\n' \
  "sounio-loom-sovereign-change-kernel-operational-selftest: PASS semantic_authority=Sounio action=9044 stage=CLAIM_READY parent_manifest_sha256=$EXPECTED_MANIFEST_SHA256 material_manifest_sha256=$EXPECTED_MATERIAL_MANIFEST_SHA256 provider_root=READ_ONLY grant_residency=Loom_kernel_memory grant_is_bearer=false grant_single_use=true consume_atomic=true exact_call_id=true exact_patch_hash=true exact_worktree_state=true authenticated_peer=true exact_file_set=true prepare=$prepared consume=$consumed deny=$refused commit=$committed commit_refused=$commit_refused write=KERNEL_MATERIALIZED edit=KERNEL_MATERIALIZED apply_patch=KERNEL_MATERIALIZED replay=REFUSED wrong_stage=REFUSED+GRANT_BURNED direct_root_write=EROFS direct_git_commit=REFUSED widened_commit=REFUSED commit_replay=REFUSED kernel_commit=ADMITTED receipt=ISSUED receipt_tamper=REFUSED ci=CONSUMED_NOT_REINTERPRETED journal=VERIFIED python_executed=false rust_executed=false write_attached=true commit_attached=true ci_attached=true claim_ready=true"
