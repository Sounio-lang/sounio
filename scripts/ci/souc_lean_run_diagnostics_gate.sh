#!/usr/bin/env bash
# `SOUNIO_SOUC_ENGINE=lean_single bin/souc run` must print the compiler's
# diagnostics and fail when the compile fails, and must never execute what a
# failed compile left behind.
#
# Measured 2026-10-05 (P0.6): the run branch discarded both streams of the
# compile and ran under `set -e`, so a type error ended the launcher with rc 1
# and zero bytes of output; 123 of 359 swept programs failed that way while
# `souc check` on the same files printed the errors.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-lean-run-diag.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "[lean-run-diagnostics] FAIL: $*" >&2; exit 1; }

mkdir -p "$WORK/repo/bin"
cp "$ROOT_DIR/bin/souc" "$WORK/repo/bin/souc"

# Fake compiler: chatter, one diagnostic, rc 1, and a runnable leftover that
# must not be executed.
cat >"$WORK/repo/bin/souc-lean-single-x86_64" <<'FAKE'
#!/usr/bin/env bash
out="${2:?missing output path}"
echo "source: $1 63 bytes"
echo "error[E200]: undefined identifier \`nope\` at <main>:2" >&2
echo "typecheck: failed" >&2
printf '#!/usr/bin/env bash\necho LEFTOVER_WAS_EXECUTED\n' >"$out"
chmod +x "$out"
exit 1
FAKE
chmod +x "$WORK/repo/bin/souc-lean-single-x86_64"
printf 'fn main() -> i32 { 0 }\n' >"$WORK/prog.sio"

set +e
SOUNIO_SOUC_ENGINE=lean_single "$WORK/repo/bin/souc" run "$WORK/prog.sio" >"$WORK/out" 2>"$WORK/err"
rc=$?
set -e

[[ $rc -ne 0 ]] || fail "failed compile returned rc 0"
grep -q 'error\[E200\]' "$WORK/err" || fail "diagnostic not shown; stderr was: $(cat "$WORK/err")"
grep -q 'nothing was run' "$WORK/err" || fail "no 'nothing was run' notice"
! grep -q LEFTOVER_WAS_EXECUTED "$WORK/out" "$WORK/err" || fail "the failed compile's leftover was executed"

echo "[lean-run-diagnostics] PASS (rc=$rc, diagnostic surfaced, leftover not executed)"
