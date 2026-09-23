#!/usr/bin/env bash
# Recheck suite-visible `//@ known-failure` files that name Madaros
# (`//@ requires: madaros`) on the current Madaros ELF.
#
# Why this exists: Full Test Suite is lean_single and *skips* these files.
# madaros_changed_tests_gate.sh only runs requires:madaros files that appear
# in the PR diff. A compiler-only PR that makes a tagged failure start
# passing never re-runs that file, so the tag rots. #1890 cleaned 240 such
# tags after a census. This gate is the mechanism that replaces the census.
#
# A pass here is a signal (stale tag), not a test FAIL. The harness prints
# XPAS and, with SOUNIO_XPAS_FATAL=1, exits 1.
#
# Instrument: the Madaros ELF the caller already built. Never
# `souc <file> -o`. Never an E230-patched local ELF unless that is what
# the caller passed — the f64 job passes its current-source binary.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS_BIN="${SOUNIO_MADAROS_CHANGED_TESTS_BIN:-${MADAROS_BIN:-}}"

fail() {
  echo "KNOWN_FAILURE_MADAROS_RECHECK_FAIL reason=$1" >&2
  exit 1
}

[[ -n "$MADAROS_BIN" && -x "$MADAROS_BIN" ]] || fail "missing_madaros_bin"

mapfile -t selected < <(
  python3 - "$ROOT_DIR" <<'PY'
import re, sys
from pathlib import Path
root = Path(sys.argv[1])
patterns = [
    "tests/run-pass/*.sio",
    "tests/compile-fail/*.sio",
    "tests/ui/type/*.sio",
    "tests/ui/effect/*.sio",
    "tests/ui/ownership/*.sio",
    "tests/ui/resolve/*.sio",
    "tests/ui/pattern/*.sio",
    "tests/stdlib/*/test_*.sio",
    "tests/gpu/*.sio",
]
for pat in patterns:
    for path in sorted(root.glob(pat)):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"//@\s*known-failure", text):
            continue
        if not re.search(r"//@\s*requires:\s*madaros\b", text):
            continue
        print(path.relative_to(root))
PY
)

if ((${#selected[@]} == 0)); then
  echo "KNOWN_FAILURE_MADAROS_RECHECK_PASS count=0"
  exit 0
fi

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/sounio-kf-madaros.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT
test_list="$work_dir/tests.txt"
printf '%s\n' "${selected[@]}" >"$test_list"

echo "KNOWN_FAILURE_MADAROS_RECHECK_START count=${#selected[@]} compiler=$MADAROS_BIN"
printf 'test=%s\n' "${selected[@]}"

# `//@ compile-fail` files need a check the harness-based XPAS pass below
# cannot make: for these, the harness's own `//@ error-pattern:` text match
# folds TWO different outcomes into the same `xfail` bucket --
#   (a) Madaros still correctly REJECTS, but with different wording than the
#       pinned pattern (today's actual state for at least one file here), and
#   (b) Madaros WRONGLY ACCEPTS the invalid program (a real regression) --
# because a wrong-acceptance also ends with the harness's own
# "expected typecheck failure but passed" -> exit_code forced to 1 -> same
# xfail path as (a). SOUNIO_XPAS_FATAL only fires on XPAS (exit_code 0), so
# it cannot tell (a) from (b) and would stay green through (b). Recheck
# compiler rejection directly, bypassing the harness's pattern-match layer
# entirely: a compile-fail file must still exit nonzero from `check`, full
# stop, regardless of what the emitted diagnostic text says.
compile_fail_regressions=""
for f in "${selected[@]}"; do
  grep -qE '^//@[[:space:]]*compile-fail' "$ROOT_DIR/$f" || continue
  if "$MADAROS_BIN" --check "$ROOT_DIR/$f" >/dev/null 2>&1; then
    compile_fail_regressions="${compile_fail_regressions}${f}
"
  fi
done
if [[ -n "$compile_fail_regressions" ]]; then
  echo "KNOWN_FAILURE_MADAROS_RECHECK_COMPILE_FAIL_REGRESSION:" >&2
  printf '%s' "$compile_fail_regressions" >&2
  fail "compile_fail_now_accepted (Madaros wrongly accepted a //@ compile-fail file that a known-failure annotation was masking as xfail)"
fi

# ulimit is the caller's problem (the f64 job already raises the stack).
SOUNIO_MADAROS_AVAILABLE=1 \
SOUNIO_XPAS_FATAL=1 \
SOUNIO_SOUC_RAW_MODE=modular \
SOUNIO_TEST_SOUC_BIN="$MADAROS_BIN" \
  bash "$ROOT_DIR/scripts/run_sio_test_suite.sh" \
    --test-list "$test_list" \
    --verbose \
    --jobs "${SOUNIO_TEST_JOBS:-4}"

echo "KNOWN_FAILURE_MADAROS_RECHECK_PASS count=${#selected[@]}"
