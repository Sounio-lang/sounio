#!/usr/bin/env bash
# W047 once per call site.
#
# tests/run-pass/epistemic_hessian_of.sio has 3 hessian_of call sites.
# A single `madaros build` walks each function body three times with three
# separate Checker instances (visibility preflight, lower verdict pass,
# epistemic export), and checker_report_warning_at_inplace used to print
# on every walk: 9 W047 blocks for 3 call sites. The warning sink now
# dedups by (code, file_id, start, end), so each call site warns once.
#
# This gate counts W047 lines for that file and requires exactly 3.
# It fails on a Madaros built before the fix (which prints 9).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

case "$(uname -s 2>/dev/null || echo unknown)/$(uname -m 2>/dev/null || echo unknown)" in
  Linux/x86_64|Linux/amd64) ;;
  *)
    echo "[w047-once] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

MADAROS="${MADAROS:-$ROOT/bin/madaros}"
SRC="$ROOT/tests/run-pass/epistemic_hessian_of.sio"
OUT="$(mktemp -d /tmp/sounio-w047-once.XXXXXX)"
trap 'rm -rf "$OUT"' EXIT

echo "== w047_once_gate =="
echo "madaros: $MADAROS"
echo "source:  $SRC"

# Build (not just check): the triple walk lives on the build path.
"$MADAROS" build "$SRC" -o "$OUT/epistemic_hessian_of" > "$OUT/build.log" 2>&1 || {
  echo "[w047-once] FAIL: madaros build exited non-zero" >&2
  tail -20 "$OUT/build.log" >&2
  exit 1
}

count="$(grep -c 'warning\[W047\]' "$OUT/build.log" || true)"
echo "W047 lines: $count (expected 3, one per hessian_of call site)"

if [ "$count" != "3" ]; then
  echo "[w047-once] FAIL: expected exactly 3 W047 lines, got $count" >&2
  grep 'warning\[W047\]' "$OUT/build.log" >&2 || true
  exit 1
fi

# Distinct call sites keep distinct spans: no site was swallowed whole.
sites="$(grep 'warning\[W047\]' "$OUT/build.log" | sort -u | wc -l)"
if [ "$sites" != "3" ]; then
  echo "[w047-once] FAIL: expected 3 distinct W047 spans, got $sites" >&2
  exit 1
fi

echo "[w047-once] PASS: 3 W047 lines, one per call site"
