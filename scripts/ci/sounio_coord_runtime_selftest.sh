#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-coord-runtime-selftest.XXXXXX")"
REPO="$TEST_ROOT/repo"
SECOND="$TEST_ROOT/second-worktree"
STATE="$TEST_ROOT/state"
ALT="$TEST_ROOT/upgrade-source"
BAD="$TEST_ROOT/bad-source"

cleanup() {
  git -C "$REPO" worktree remove --force "$SECOND" >/dev/null 2>&1 || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  echo "sounio-coord-runtime-selftest: FAIL: $*" >&2
  exit 1
}

mkdir -p "$REPO/bin" "$REPO/scripts/dev"
cp "$ROOT_DIR/bin/sounio-coord" "$REPO/bin/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agent_hook.py" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agent_hook_runtime.py" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_causal_runtime.py" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/install_sounio_coord_runtime.sh" "$REPO/scripts/dev/"
chmod +x "$REPO/bin/sounio-coord" "$REPO/scripts/dev/"*.sh "$REPO/scripts/dev/"*.py
git -C "$REPO" init -q
git -C "$REPO" config user.name 'Sounio Runtime Selftest'
git -C "$REPO" config user.email 'coord-runtime-selftest@sounio.local'
git -C "$REPO" add .
git -C "$REPO" commit -qm seed
git -C "$REPO" worktree add -q -b second-lane "$SECOND"
RUNTIME_ROOT="$REPO/.git/sounio-coord-runtime"

output="$(cd "$REPO" && SOUNIO_COORD_RUNTIME_MODE=local bin/sounio-coord runtime-info)"
grep -q '^selection=local$' <<< "$output" || fail 'launcher did not report its local fallback'
grep -q '^protocol_version=3$' <<< "$output" || fail 'local runtime protocol is wrong'

output="$(cd "$REPO" && bin/sounio-coord install-runtime)"
first_id="$(sed -n 's/^INSTALLED runtime_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$first_id" ]] || fail 'installer did not return the first runtime id'
grep -q "^ACTIVATED runtime_id=$first_id " <<< "$output" || fail 'first runtime was not activated'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-coord-causal-runtime" ]] || \
  fail 'installed runtime omitted the causal receipt verifier'

output="$(cd "$SECOND" && bin/sounio-coord runtime-info)"
grep -q '^selection=shared$' <<< "$output" || fail 'second worktree did not select shared runtime'
grep -q "^runtime_id=$first_id$" <<< "$output" || fail 'worktrees selected different runtime ids'
grep -q "$RUNTIME_ROOT/versions/$first_id/bin/sounio-coord-runtime" <<< "$output" || \
  fail 'runtime path is not anchored in the Git common directory'

printf '#!/usr/bin/env bash\nexit 97\n' > "$SECOND/scripts/dev/sounio_coord_runtime.sh"
printf '#!/usr/bin/env python3\nraise SystemExit(98)\n' > \
  "$SECOND/scripts/dev/sounio_coord_agent_hook_runtime.py"
printf '#!/usr/bin/env python3\nraise SystemExit(99)\n' > \
  "$SECOND/scripts/dev/sounio_coord_causal_runtime.py"
chmod +x "$SECOND/scripts/dev/sounio_coord_runtime.sh" \
  "$SECOND/scripts/dev/sounio_coord_agent_hook_runtime.py" \
  "$SECOND/scripts/dev/sounio_coord_causal_runtime.py"
output="$(cd "$SECOND" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || \
  fail 'sabotaged worktree fallback displaced the shared CLI runtime'
output="$(
  cd "$SECOND"
  printf '%s\n' \
    "{\"session_id\":\"runtime-test\",\"cwd\":\"$SECOND\",\"hook_event_name\":\"SessionStart\"}" | \
    SOUNIO_COORD_DIR="$STATE" python3 scripts/dev/sounio_coord_agent_hook.py --agent claude
)"
grep -q 'agent=claude lane=session-runtime-test' <<< "$output" || \
  fail 'sabotaged worktree fallback displaced the shared hook runtime'

mkdir -p "$ALT/scripts/dev"
cp "$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agent_hook_runtime.py" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_causal_runtime.py" "$ALT/scripts/dev/"
sed -i 's/SOUNIO_COORD_RUNTIME_VERSION=2026\.08\.23\.4/SOUNIO_COORD_RUNTIME_VERSION=2026.08.23.5-test/' \
  "$ALT/scripts/dev/sounio_coord_runtime.sh"
chmod +x "$ALT/scripts/dev/"*
output="$(cd "$REPO" && bin/sounio-coord install-runtime --source-root "$ALT")"
second_id="$(sed -n 's/^INSTALLED runtime_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$second_id" && "$second_id" != "$first_id" ]] || fail 'upgrade did not create a new runtime id'
output="$(cd "$SECOND" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$second_id$" <<< "$output" || fail 'worktree did not observe atomic runtime upgrade'
grep -q '^runtime_version=2026.08.23.5-test$' <<< "$output" || fail 'upgraded runtime version is wrong'

output="$(cd "$REPO" && bin/sounio-coord install-runtime --activate "$first_id")"
grep -q "^ACTIVATED runtime_id=$first_id " <<< "$output" || fail 'runtime rollback failed'
output="$(cd "$SECOND" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || fail 'worktree did not observe runtime rollback'

mkdir -p "$BAD/scripts/dev"
cp "$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" "$BAD/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agent_hook_runtime.py" "$BAD/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_causal_runtime.py" "$BAD/scripts/dev/"
sed -i 's/SOUNIO_COORD_PROTOCOL_VERSION=3/SOUNIO_COORD_PROTOCOL_VERSION=4/' \
  "$BAD/scripts/dev/sounio_coord_runtime.sh"
chmod +x "$BAD/scripts/dev/"*
if (cd "$REPO" && bin/sounio-coord install-runtime --source-root "$BAD") >/dev/null 2>&1; then
  fail 'installer accepted an incompatible protocol'
fi
mkdir -p "$RUNTIME_ROOT/versions/incomplete"
if (cd "$REPO" && bin/sounio-coord install-runtime --activate incomplete) >/dev/null 2>&1; then
  fail 'installer activated an incomplete runtime'
fi
output="$(cd "$REPO" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || fail 'failed activation changed the current runtime'

output="$(cd "$REPO" && bin/sounio-coord install-runtime --list)"
grep -q "runtime_id=$first_id current=yes" <<< "$output" || fail 'runtime list lost the current marker'
grep -q "runtime_id=$second_id current=no" <<< "$output" || fail 'runtime list lost installed upgrade'

unlink "$RUNTIME_ROOT/current"
ln -s versions/missing "$RUNTIME_ROOT/current"
if (cd "$REPO" && bin/sounio-coord runtime-info) >/dev/null 2>&1; then
  fail 'CLI launcher silently fell back across a broken shared-runtime link'
fi
if (
  cd "$REPO"
  printf '%s\n' \
    "{\"session_id\":\"broken-link\",\"cwd\":\"$REPO\",\"hook_event_name\":\"SessionStart\"}" | \
    SOUNIO_COORD_DIR="$STATE" python3 scripts/dev/sounio_coord_agent_hook.py --agent claude
) >/dev/null 2>&1; then
  fail 'hook launcher silently fell back across a broken shared-runtime link'
fi
output="$(cd "$REPO" && scripts/dev/install_sounio_coord_runtime.sh --activate "$first_id")"
grep -q "^ACTIVATED runtime_id=$first_id " <<< "$output" || \
  fail 'installer did not recover a broken current link atomically'

echo 'sounio-coord-runtime-selftest: PASS'
