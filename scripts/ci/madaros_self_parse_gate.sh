#!/usr/bin/env bash
# The compiler must be able to read its own source tree.
#
# On 2026-08-03 a Madaros built from main could not parse
# self-hosted/resolve/scope.sio — a file in its own sources. 417 gates did not
# catch it; it was found by accident. Re-measured afterwards against the
# pre-fix binary: 118 of 559 files under self-hosted/ failed to parse. Not one
# file. A hundred and eighteen.
#
# Nothing here builds a compiler. build_modular_madaros.sh locks internally and
# flock is not reentrant, so a gate that built one would deadlock; and the CI
# job this runs in has already built the ELF and kept it. If no binary is
# supplied this gate REFUSES rather than building.
#
#   A1  the closure of main.sio is complete
#   A2  the closure has capacity headroom
#   A3  every tracked .sio under self-hosted/ parses, ratcheted against a baseline
#
# Falsifiability is proven separately by madaros_self_parse_selftest.sh.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9

VERDICT="$ROOT_DIR/scripts/lib/boundary_closure_verdict.sh"
BASELINE="$ROOT_DIR/scripts/ci/fixtures/madaros_self_parse_baseline.txt"
REPORT_DIR="${SOUNIO_MADAROS_SELF_PARSE_REPORT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/sounio-self-parse.XXXXXX")}"
mkdir -p "$REPORT_DIR"

fail() { echo "MADAROS_SELF_PARSE_GATE_FAIL: $*" >&2; exit 1; }

[[ -x "$VERDICT" ]] || fail "missing $VERDICT"

MADAROS="${SOUNIO_MADAROS_SELF_PARSE_BIN:-${MADAROS_RAW_BIN:-$ROOT_DIR/artifacts/self-hosted/madaros}}"
[[ -x "$MADAROS" ]] \
  || fail "no Madaros ELF at $MADAROS. This gate does not build one — set SOUNIO_MADAROS_SELF_PARSE_BIN."
if head -c2 "$MADAROS" 2>/dev/null | grep -q '#!'; then
  fail "$MADAROS is a wrapper script, not a raw ELF. A wrapper resolves to whatever prebuilt is lying around, which is the opposite of what this gate measures."
fi

ulimit -s 524288 2>/dev/null || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

echo "  madaros = $MADAROS"

# ------------------------------------------------------------------ A1 + A2
MAIN_SRC="self-hosted/compiler/main.sio"
[[ -f "$MAIN_SRC" ]] || fail "$MAIN_SRC absent — nothing to measure"

timeout 300 "$MADAROS" --science-boundary-closure "$MAIN_SRC" >"$REPORT_DIR/main.report" 2>&1 || true
# Deliberately ignoring the exit code: closure mode exits 0 on `status
# incomplete`. The verdict comes from the report body or it comes from nothing.
if ! "$VERDICT" "$REPORT_DIR/main.report" >"$REPORT_DIR/main.verdict" 2>&1; then
  # The closure is currently blocked by a file the compiler cannot parse. That
  # is allowed ONLY while that file is a known, listed entry in the baseline —
  # the same ratchet A3 uses. Measured 2026-08-04: main.sio's closure stops at
  # self-hosted/check/check.sio (23 592 lines), which fails at node 0, i.e. it
  # cannot parse itself as an entry point either.
  blocker=$(awk -F'\t' '$1 == "failed_path" { print $2; exit }' "$REPORT_DIR/main.report")
  if [[ -n "$blocker" ]] && [[ -f "$BASELINE" ]] && grep -qxF "$blocker" "$BASELINE"; then
    echo "  A1/A2 closure blocked at $blocker (known, in the baseline)"
    echo "        the closure cannot be complete until that file parses"
  else
    echo "  A1/A2 FAIL on $MAIN_SRC:"
    sed 's/^/    /' "$REPORT_DIR/main.verdict"
    grep -vE '^(node|edge)\b' "$REPORT_DIR/main.report" | sed 's/^/    /'
    fail "the compiler cannot walk its own import closure, and the blocker is not a known one"
  fi
else
  echo "  A1/A2 $(cat "$REPORT_DIR/main.verdict")"
fi

# ------------------------------------------------------------------ A3
# git ls-files, never find: find also returns untracked LSP scratch snapshots
# (self-hosted/lsp/.sounio-lsp-snapshot.*.sio) that exist only on one machine,
# and a gate whose input set depends on local scratch is a flaky gate.
mapfile -t SOURCES < <(git ls-files 'self-hosted/**/*.sio' 'self-hosted/*.sio' 2>/dev/null | sort -u)
(( ${#SOURCES[@]} > 0 )) \
  || fail "git ls-files matched no .sio under self-hosted/ — the tree moved and this sweep is measuring nothing"
(( ${#SOURCES[@]} >= 400 )) \
  || fail "only ${#SOURCES[@]} sources found; expected ~511. Either the tree shrank drastically or the glob broke."
echo "  A3 sweeping ${#SOURCES[@]} tracked sources"

SWEEP_OUT="$REPORT_DIR/sweep"
: >"$SWEEP_OUT.failed"
: >"$SWEEP_OUT.crashed"

probe_one() {
  local src="$1" out
  out=$(timeout 60 "$MADAROS" --science-boundary-closure "$src" 2>&1)
  if ! grep -qxF 'SOUNIO_BOUNDARY_CLOSURE_V1' <<<"$out"; then
    printf '%s\n' "$src" >>"$SWEEP_OUT.crashed"
    return
  fi
  if awk -F'\t' '$1 == "parse_failed" { print $2; exit }' <<<"$out" | grep -qx true; then
    printf '%s\n' "$src" >>"$SWEEP_OUT.failed"
  fi
}
export -f probe_one
export MADAROS SWEEP_OUT

printf '%s\n' "${SOURCES[@]}" | xargs -P4 -I{} bash -c 'probe_one "$@"' _ {}

sort -u -o "$SWEEP_OUT.failed" "$SWEEP_OUT.failed"
sort -u -o "$SWEEP_OUT.crashed" "$SWEEP_OUT.crashed"

if [[ -s "$SWEEP_OUT.crashed" ]]; then
  echo "  A3 the compiler produced no report at all for:"
  sed 's/^/    /' "$SWEEP_OUT.crashed"
  fail "$(wc -l <"$SWEEP_OUT.crashed") source(s) crashed or hung the compiler"
fi

if [[ ! -f "$BASELINE" ]]; then
  echo "  A3 no baseline at $BASELINE. Current failures ($(wc -l <"$SWEEP_OUT.failed")):"
  sed 's/^/    /' "$SWEEP_OUT.failed"
  fail "seed the baseline from THIS output and commit it — never from a guess"
fi

# The baseline is a ratchet. A new failure is red; a baseline entry that now
# parses is ALSO red, so the allowlist can only shrink. That is what stops it
# rotting into a permanent excuse.
grep -vE '^\s*(#|$)' "$BASELINE" | sort -u >"$SWEEP_OUT.baseline"

NEW=$(comm -23 "$SWEEP_OUT.failed" "$SWEEP_OUT.baseline")
FIXED=$(comm -13 "$SWEEP_OUT.failed" "$SWEEP_OUT.baseline")

if [[ -n "$NEW" ]]; then
  echo "  A3 REGRESSION — these parse on the baseline compiler and not on this one:"
  sed 's/^/    /' <<<"$NEW"
  fail "$(grep -c . <<<"$NEW") source(s) newly unparseable"
fi
if [[ -n "$FIXED" ]]; then
  echo "  A3 these are in the baseline but now PARSE — remove them from $BASELINE:"
  sed 's/^/    /' <<<"$FIXED"
  fail "the baseline may only shrink, and it is now out of date"
fi

remaining=$(grep -c . "$SWEEP_OUT.failed" || true)
echo "  A3 ${#SOURCES[@]} sources, ${remaining:-0} known-unparseable (all in the baseline)"
echo "MADAROS_SELF_PARSE_GATE_OK"
