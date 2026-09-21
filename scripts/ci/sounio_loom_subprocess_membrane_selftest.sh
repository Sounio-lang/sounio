#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-subprocess-membrane.XXXXXX")"
RUNTIME="$TEST_ROOT/subprocess-membrane"
MODULE="$ROOT_DIR/stdlib/coordination/loom_subprocess_membrane_authority.sio"
ENTRYPOINT="$ROOT_DIR/tools/loom/subprocess_membrane_main.sio"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-subprocess-membrane-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

SOUNIO_LOOM_SUBPROCESS_MEMBRANE_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_subprocess_membrane.sh" >/dev/null

selftest="$(printf '0\n' | "$RUNTIME")"
[[ "$selftest" == 'SOUNIO_SUBPROCESS_MEMBRANE_SELFTEST PASS cases=43' ]] ||
  fail "unexpected Sounio selftest: $selftest"

one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'

valid_exec="9023 3 1 3 10 1 1 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $one $one $one $zero $one $zero $zero"
hidden_python="9023 3 1 3 7 1 1 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $one $one $one $zero $one $zero $zero"
out_of_scope_write="9023 3 1 4 10 1 1 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $one $one $one $one $one $zero $zero"
live_descendant_commit="9023 3 2 8 12 1 1 1 1 1 1 1 0 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $one $one $zero $zero $one $one $zero"
no_deadline_root="9023 3 1 1 11 1 1 1 1 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $one $one $one $zero $zero $zero $zero"

valid="$(printf '%s\n' "$valid_exec" | "$RUNTIME")"
[[ "$valid" == 'SOUNIO_SUBPROCESS_MEMBRANE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN' ]] ||
  fail "valid exec event refused: $valid"

python_denied="$(printf '%s\n' "$hidden_python" | "$RUNTIME" || true)"
[[ "$python_denied" == 'SOUNIO_SUBPROCESS_MEMBRANE_DENY code=410 reason=prohibited-language stage=SEMANTICS_FROZEN' ]] ||
  fail "hidden Python did not hit code 410: $python_denied"
write_denied="$(printf '%s\n' "$out_of_scope_write" | "$RUNTIME" || true)"
[[ "$write_denied" == 'SOUNIO_SUBPROCESS_MEMBRANE_DENY code=422 reason=write-scope-missing stage=SEMANTICS_FROZEN' ]] ||
  fail "out-of-scope write did not hit code 422: $write_denied"
commit_denied="$(printf '%s\n' "$live_descendant_commit" | "$RUNTIME" || true)"
[[ "$commit_denied" == 'SOUNIO_SUBPROCESS_MEMBRANE_DENY code=423 reason=tree-not-quiescent stage=SEMANTICS_FROZEN' ]] ||
  fail "live-descendant commit did not hit code 423: $commit_denied"
deadline_denied="$(printf '%s\n' "$no_deadline_root" | "$RUNTIME" || true)"
[[ "$deadline_denied" == 'SOUNIO_SUBPROCESS_MEMBRANE_DENY code=421 reason=deadline-missing stage=SEMANTICS_FROZEN' ]] ||
  fail "no-deadline root did not hit code 421: $deadline_denied"

sabotage() {
  local label="$1" rule="$2" frame="$3" expected="$4"
  local sabotaged_module="$TEST_ROOT/$label.sio"
  local combined="$TEST_ROOT/$label-combined.sio"
  local sabotaged_runtime="$TEST_ROOT/$label-runtime"
  grep -Fqx "$rule" "$MODULE" || fail "$label rule is absent or changed"
  grep -Fvx "$rule" "$MODULE" > "$sabotaged_module"
  sed -n '1,$p' "$sabotaged_module" "$ENTRYPOINT" > "$combined"
  SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" compile "$combined" \
    -o "$sabotaged_runtime" >/dev/null
  chmod 0755 "$sabotaged_runtime"
  local actual
  actual="$(printf '%s\n' "$frame" | "$sabotaged_runtime")"
  [[ "$actual" == "$expected" ]] ||
    fail "$label sabotage did not admit its unchanged witness: $actual"
}

sabotage prohibited-language \
  '    if facts.actor_language == 7 || facts.actor_language == 8 { return 410 }' \
  "$hidden_python" \
  'SOUNIO_SUBPROCESS_MEMBRANE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN'
sabotage write-scope \
  '    if (facts.effect_kind == 4 || facts.effect_kind == 5) && (facts.scope_bound != 1 || !subprocess_membrane_digest_nonzero(bindings.target_hash) || !subprocess_membrane_digest_nonzero(bindings.claim_scope_hash)) { return 422 }' \
  "$out_of_scope_write" \
  'SOUNIO_SUBPROCESS_MEMBRANE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN'
sabotage tree-quiescence \
  '    if (facts.effect_kind == 8 || facts.effect_kind == 9) && facts.tree_quiescent != 1 { return 423 }' \
  "$live_descendant_commit" \
  'SOUNIO_SUBPROCESS_MEMBRANE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN'
sabotage deadline \
  '    if facts.deadline_bound != 1 || !subprocess_membrane_digest_nonzero(bindings.deadline_hash) { return 421 }' \
  "$no_deadline_root" \
  'SOUNIO_SUBPROCESS_MEMBRANE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN'

printf '%s\n' \
  'sounio-loom-subprocess-membrane-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY action=9023 cases=43 hidden_python=DENY410 out_of_scope_write=DENY422 live_descendant_commit=DENY423 no_deadline=DENY421 causal_sabotage=ALLOWx4'
