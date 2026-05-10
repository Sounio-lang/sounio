#!/usr/bin/env bash
# Phase B deep-dive -- sedenion ZD empirical foundation gate.
#
# Compiles and runs tests/math/test_zd_deep_dive.sio:
#   B.4 enumerate primitive ZDs in {1..15}
#   B.5 h-sweep partition stability
#   B.6 multi-invariant equivalence spectrum
#   B.7 iterated trajectory drift between different-z pairs
#
# This is *evidence-recording* -- the test always asserts ALL PASS as long
# as it runs cleanly. The synthesis block at the end is the durable artifact
# downstream readers (and future butterfly-shapers) consume.
#
# Reference baseline (2026-05-09):
#   - 92 primitive ZDs enumerated
#   - trivial dedup (same-z): structural, 2 per z, h-stable
#   - cross-z dedup: h-coincidental; diverges ~2x per iterated step
#   - recommendation: z-canonical hash, not orbit-class metadata

ulimit -s unlimited 2>/dev/null || true
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="$ROOT_DIR/bin/souc-linux-x86_64"
[[ -x "$SOUC_BIN" ]] || { echo "FAIL: $SOUC_BIN missing" >&2; exit 2; }
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

TEST_SRC="tests/math/test_zd_deep_dive.sio"
[[ -f "$TEST_SRC" ]] || { echo "FAIL: $TEST_SRC missing" >&2; exit 2; }

OUT_ELF="$(mktemp -t zd-deep.XXXXXX.elf)"
OUT_LOG="$(mktemp -t zd-deep.XXXXXX.log)"
trap 'rm -f "$OUT_ELF" "$OUT_LOG"' EXIT INT TERM

if ! "$SOUC_BIN" "$TEST_SRC" "$OUT_ELF" >/dev/null 2>&1; then
    echo "FAIL: compile $TEST_SRC failed" >&2
    "$SOUC_BIN" "$TEST_SRC" "$OUT_ELF" >&2 || true
    exit 1
fi
chmod +x "$OUT_ELF"

set +e
"$OUT_ELF" >"$OUT_LOG" 2>&1
rc=$?
set -e

echo "=== Phase B deep-dive findings ==="
cat "$OUT_LOG"
echo "=== end ==="

if [[ "$rc" -ne 0 ]]; then echo "FAIL: test exited $rc" >&2; exit 1; fi
grep -q "ALL PASS" "$OUT_LOG" || { echo "FAIL: 'ALL PASS' missing" >&2; exit 1; }
grep -q "Synthesis" "$OUT_LOG" || { echo "FAIL: synthesis block missing" >&2; exit 1; }

# Surface the headline numbers for grep-ability.
zd_count="$(grep -E '^  ZDs found:' "$OUT_LOG" | awk '{print $NF}' | tr -d '\n')"
echo "Phase B deep-dive: ZDs=$zd_count, $(grep -m1 recommendation "$OUT_LOG" | sed 's/^[[:space:]]*//')"

exit 0
