#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-native-hook-promotion.XXXXXX")"
RUNTIME_ROOT="${SOUNIO_COORD_RUNTIME_DIR:-$(git -C "$ROOT_DIR" rev-parse --path-format=absolute --git-common-dir)/sounio-coord-runtime}"
STATE_ROOT="${SOUNIO_COORD_DIR:-$(git -C "$ROOT_DIR" rev-parse --path-format=absolute --git-common-dir)/sounio-coord-state}"
DRAIN_STATE_ROOT="$TEST_ROOT/drain-state"
INSTALLER="$ROOT_DIR/scripts/dev/install_sounio_loom_native_hooks.sh"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-native-hook-promotion-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

create_target() {
  local target="$1"
  mkdir -p "$target/.codex" "$target/.claude" "$target/bin"
  git init -q "$target"
  printf '{"legacy":"codex"}\n' > "$target/.codex/hooks.json"
  printf '{"legacy":"claude"}\n' > "$target/.claude/settings.json"
  cp "$ROOT_DIR/bin/sounio-coord" "$target/bin/"
  git -C "$target" add .codex/hooks.json .claude/settings.json bin/sounio-coord
  git -C "$target" -c user.name='LOOM selftest' -c user.email='loom-selftest@invalid' \
    commit -q -m 'fixture'
  ln -s "$RUNTIME_ROOT" "$target/.git/sounio-coord-runtime"
  ln -s "$STATE_ROOT" "$target/.git/sounio-coord-state"
}

config_sha() {
  local root="$1" path
  for path in .codex/hooks.json .claude/settings.json .cursor/hooks.json \
      .grok/hooks/loom-native.json; do
    if [[ -f "$root/$path" ]]; then
      sha256sum "$root/$path" | awk '{print $1}'
    else
      printf 'absent\n'
    fi
  done | paste -sd ':' -
}

[[ -x "$INSTALLER" ]] || fail 'promotion installer is not executable'
[[ -L "$RUNTIME_ROOT/current" ]] || fail 'shared runtime is not active'
[[ -L "$RUNTIME_ROOT/native-next" ]] || fail 'staged native runtime is not selected'
legacy_before="$(readlink -f "$RUNTIME_ROOT/current")"

positive="$TEST_ROOT/positive"
create_target "$positive"
output="$(
  SOUNIO_COORD_RUNTIME_DIR="$RUNTIME_ROOT" SOUNIO_COORD_DIR="$STATE_ROOT" \
    SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_NATIVE_HOOK_DRAIN_STATE_DIR="$DRAIN_STATE_ROOT" \
    "$INSTALLER" --source-root "$ROOT_DIR" --target-root "$positive" --activate
)"
grep -q '^LOOM_NATIVE_HOOKS_ACTIVATED ' <<< "$output" ||
  fail 'positive promotion omitted its activation receipt'
[[ "$(sha256sum "$positive/.codex/hooks.json" | awk '{print $1}')" == \
  "$(sha256sum "$ROOT_DIR/.codex/hooks.json" | awk '{print $1}')" ]] ||
  fail 'positive promotion installed the wrong Codex configuration'
[[ "$(sha256sum "$positive/.claude/settings.json" | awk '{print $1}')" == \
  "$(sha256sum "$ROOT_DIR/.claude/settings.json" | awk '{print $1}')" ]] ||
  fail 'positive promotion installed the wrong Claude configuration'
[[ "$(sha256sum "$positive/.cursor/hooks.json" | awk '{print $1}')" == \
  "$(sha256sum "$ROOT_DIR/.cursor/hooks.json" | awk '{print $1}')" ]] ||
  fail 'positive promotion installed the wrong Cursor configuration'
[[ "$(sha256sum "$positive/.grok/hooks/loom-native.json" | awk '{print $1}')" == \
  "$(sha256sum "$ROOT_DIR/.grok/hooks/loom-native.json" | awk '{print $1}')" ]] ||
  fail 'positive promotion installed the wrong Grok configuration'
[[ ! -e "$positive/.git/index.lock" ]] ||
  fail 'positive promotion retained the target Git index lock'
receipt="$(sed -n 's/.* receipt=\(.*\)$/\1/p' <<< "$output")"
[[ -f "$receipt" ]] || fail 'positive promotion receipt is missing'
grep -q '^result=ACTIVATED$' "$receipt" || fail 'positive promotion receipt is not activated'
grep -q '^canary_lifecycle_receipts=12$' "$receipt" ||
  fail 'positive promotion did not retain its twelve lifecycle receipts'
grep -q '^canary_cleanup_receipts=1$' "$receipt" ||
  fail 'positive promotion did not retain its Codex cleanup receipt'
grep -q '^canary_allow_receipts=13$' "$receipt" ||
  fail 'positive promotion did not retain its thirteen-ALLOW canary proof'
grep -q '^canary_runtime_capsule_receipts=13$' "$receipt" ||
  fail 'positive promotion did not retain its runtime-capsule proof'
grep -q '^canary_action_9045_receipts=13$' "$receipt" ||
  fail 'positive promotion did not retain its Sounio action 9045 proof'
grep -q '^canary_action_9046_mask=15$' "$receipt" ||
  fail 'positive promotion did not retain its Sounio action 9046 canary mask'
grep -q '^guardian_action_9046_prepared=true$' "$receipt" ||
  fail 'positive promotion did not retain its Sounio action 9046 guardian proof'
grep -q '^runtime_selector=native-next$' "$receipt" ||
  fail 'positive promotion did not bind native-next'
grep -q '^legacy_current_unchanged=true$' "$receipt" ||
  fail 'positive promotion did not prove the legacy current remained unchanged'
grep -q '^bridge_free_current=false$' "$receipt" ||
  fail 'positive promotion prematurely claimed bridge-free current'
[[ "$(readlink -f "$RUNTIME_ROOT/current")" == "$legacy_before" ]] ||
  fail 'positive promotion changed the current runtime selector'
grep -q '^canary_providers=codex+claude+cursor+grok$' "$receipt" ||
  fail 'positive promotion did not retain its four-provider binding'
for provider in codex claude cursor grok; do
  grep -q "^${provider}_config_sha256=[0-9a-f]\{64\}$" "$receipt" ||
    fail "positive promotion receipt omitted the $provider configuration"
  grep -q "^canary_${provider}_lane=session-" "$receipt" ||
    fail "positive promotion receipt omitted the $provider lane"
done

rollback="$TEST_ROOT/rollback"
create_target "$rollback"
rollback_before="$(config_sha "$rollback")"
set +e
rollback_output="$(
  SOUNIO_COORD_RUNTIME_DIR="$RUNTIME_ROOT" SOUNIO_COORD_DIR="$STATE_ROOT" \
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_NATIVE_HOOK_DRAIN_STATE_DIR="$DRAIN_STATE_ROOT" \
  SOUNIO_LOOM_NATIVE_HOOK_PROMOTION_SABOTAGE_AFTER_SWAP=1 \
    "$INSTALLER" --source-root "$ROOT_DIR" --target-root "$rollback" --activate 2>&1
)"
rollback_rc=$?
set -e
[[ "$rollback_rc" -ne 0 ]] || fail 'after-swap sabotage did not refuse promotion'
grep -q 'error: sabotage-after-swap' <<< "$rollback_output" ||
  fail 'after-swap sabotage did not cause the refusal'
grep -q '^ROLLED_BACK transaction=' <<< "$rollback_output" ||
  fail 'after-swap sabotage did not report rollback'
[[ "$(config_sha "$rollback")" == "$rollback_before" ]] ||
  fail 'after-swap sabotage did not restore all four configurations exactly'
[[ ! -e "$rollback/.git/index.lock" ]] ||
  fail 'rollback retained the target Git index lock'

dirty="$TEST_ROOT/dirty"
create_target "$dirty"
printf 'dirty\n' >> "$dirty/.codex/hooks.json"
dirty_before="$(config_sha "$dirty")"
if SOUNIO_COORD_RUNTIME_DIR="$RUNTIME_ROOT" SOUNIO_COORD_DIR="$STATE_ROOT" \
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_NATIVE_HOOK_DRAIN_STATE_DIR="$DRAIN_STATE_ROOT" \
  "$INSTALLER" --source-root "$ROOT_DIR" --target-root "$dirty" --activate \
  >/dev/null 2>&1; then
  fail 'promotion accepted a dirty target hook configuration'
fi
[[ "$(config_sha "$dirty")" == "$dirty_before" ]] ||
  fail 'dirty-target refusal changed a configuration'

locked="$TEST_ROOT/locked"
create_target "$locked"
printf 'foreign-index-lock\n' > "$locked/.git/index.lock"
locked_before="$(config_sha "$locked")"
if SOUNIO_COORD_RUNTIME_DIR="$RUNTIME_ROOT" SOUNIO_COORD_DIR="$STATE_ROOT" \
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_NATIVE_HOOK_DRAIN_STATE_DIR="$DRAIN_STATE_ROOT" \
  "$INSTALLER" --source-root "$ROOT_DIR" --target-root "$locked" --activate \
  >/dev/null 2>&1; then
  fail 'promotion ignored an existing target Git index lock'
fi
[[ "$(config_sha "$locked")" == "$locked_before" ]] ||
  fail 'index-lock refusal changed a configuration'
grep -q '^foreign-index-lock$' "$locked/.git/index.lock" ||
  fail 'promotion removed a foreign target Git index lock'

printf '%s\n' \
  'sounio-loom-native-hook-promotion-selftest: PASS promotion=atomic selector=native-next legacy_current=unchanged runtime=manifest-bound git_index=locked policyless_canary=13-ALLOW lifecycle_receipts=12 codex_cleanup_receipts=1 providers=codex+claude+cursor+grok action=9045+9046 guardian=prepared canary_mask=15 rollback=exact dirty_target=refused foreign_lock=preserved python_oracle=absent rust_oracle=absent'
