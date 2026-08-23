#!/usr/bin/env bash
# The committed Madaros binary must exercise what its source implements.
#
# lean_single has canonical_compiler_gate.sh: the shipped ELF must be the
# byte-identical fixed point of lean_single.sio. Madaros has no equivalent,
# because it is built from ~120 modules by build_modular_madaros.sh in about
# four minutes -- too slow to rebuild on every PR, so nothing watches it.
#
# WHY THIS EXISTS, measured 2026-08-23. self-hosted/check/check.sio implements
# the EpistemicComplete floor and reports E215 from two sites. Built from that
# source, Madaros refuses tests/compile-fail/dissertation_pbpk28_overclaim.sio
# with `error[E215] EpistemicComplete violation`, exactly as the fixture's own
# error-pattern demands. The COMMITTED bin/madaros-linux-x86_64 returns rc=0 and
# accepts it.
#
# Two agents independently concluded from that rc=0 that the compiler does not
# enforce the gate and that Dissertation Contribution 2 was inert. Both had
# carefully unset the poisoned SOUC_BIN and SOUNIO_STDLIB_PATH exports first.
# Clearing the environment disarms one trap; the committed binary is a second
# one, and a clean environment running a stale binary answers yesterday with
# every appearance of rigour.
#
# So this gate does not rebuild and does not compare bytes. It asks, per
# capability: the source implements this -- does the shipped binary do it?
# A source marker without the matching behaviour means the binary is behind.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 9
ART=artifacts/gates/madaros_binary_source_drift.v1.json
mkdir -p "$(dirname "$ART")"

# The measurement must not inherit another checkout's compiler or stdlib.
unset SOUC_BIN
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="${SOUNIO_DRIFT_SOUC:-$ROOT/bin/souc}"

fail_json() {
  printf '{"status":"fail","reason":"%s","metrics":{"total":%s,"passed":%s,"failed":%s,"not_run":0}}\n' \
    "$1" "${2:-0}" "${3:-0}" "${4:-1}" > "$ART"
}

[[ -x "$SOUC" ]] || { echo "MADAROS_DRIFT_FAIL reason=no_compiler at $SOUC" >&2; fail_json no_compiler; exit 1; }

# Each row: label | source marker (grep -E) | file | probe | expected rc test
# `expected` is `refuse` (probe must exit non-zero) or `accept` (must exit 0).
run_probe() { timeout 300 "$SOUC" check "$1" >/dev/null 2>&1; echo $?; }

total=0; behind=0; skipped=0

check_row() {
  local label="$1" marker="$2" srcfile="$3" fixture="$4" expected="$5"
  total=$((total + 1))
  if [[ ! -f "$srcfile" ]] || ! grep -qE "$marker" "$srcfile"; then
    # The source does not implement it, so the binary is not behind for it.
    # This is a SKIP, never a pass: if the marker moved, the row measured
    # nothing and must say so rather than report agreement.
    echo "  SKIP  $label -- source marker not found in $srcfile"
    skipped=$((skipped + 1))
    return
  fi
  [[ -f "$fixture" ]] || { echo "  SKIP  $label -- fixture $fixture missing"; skipped=$((skipped+1)); return; }
  local rc; rc=$(run_probe "$fixture")
  case "$expected" in
    refuse) if [[ "$rc" == "0" ]]; then
              echo "  BEHIND  $label -- source implements it, shipped binary accepts $fixture (rc=0)"
              behind=$((behind + 1))
            else
              echo "  ok      $label -- refused, rc=$rc"
            fi ;;
    accept) if [[ "$rc" != "0" ]]; then
              echo "  BEHIND  $label -- source implements it, shipped binary rejects $fixture (rc=$rc)"
              behind=$((behind + 1))
            else
              echo "  ok      $label -- accepted"
            fi ;;
  esac
}

echo "compiler under test: $SOUC"
echo "source tree:         $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo

check_row "E215 EpistemicComplete floor" \
          '215, confidence_milli, minimum' \
          self-hosted/check/check.sio \
          tests/compile-fail/dissertation_pbpk28_overclaim.sio \
          refuse

echo
echo "  capabilities probed: $total   behind: $behind   skipped: $skipped"

# A gate over zero rows is the failure this gate exists to name elsewhere.
if [[ "$((total - skipped))" -lt 1 ]]; then
  echo "MADAROS_DRIFT_FAIL reason=every_row_skipped -- the gate measured nothing" >&2
  fail_json every_row_skipped "$total" 0 "$total"
  exit 1
fi

if [[ "$behind" -gt 0 ]]; then
  cat >&2 <<'MSG'

MADAROS_DRIFT_FAIL: the committed bin/madaros-linux-x86_64 is behind self-hosted/.

  This is not a source defect. The source implements the capability above; the
  shipped binary predates it. Anyone measuring compiler behaviour with the
  committed binary will get a wrong answer that looks careful.

  To clear: rebuild and commit the binary --
      bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros
  (call it DIRECTLY; wrapping it in souc-build-lock.sh self-deadlocks silently),
  or run it off the pod with scripts/dev/souc-build-remote.sh.
MSG
  fail_json binary_behind_source "$total" "$((total - behind - skipped))" "$behind"
  exit 1
fi

printf '{"status":"pass","metrics":{"total":%s,"passed":%s,"failed":0,"not_run":%s}}\n' \
  "$total" "$((total - skipped))" "$skipped" > "$ART"
echo "MADAROS_BINARY_SOURCE_DRIFT_OK"
