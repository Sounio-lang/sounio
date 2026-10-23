#!/usr/bin/env bash
# region_provenance_gate.sh — Spine B: EpistemicInRegion runs under souc
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "region_provenance_gate"

unset SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_SOUC_ENGINE || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
[[ -x "$SOUC" ]] || gate_fail "souc not executable: $SOUC"

SRC="$ROOT/stdlib/epistemic/region_provenance.sio"
LIVE="$ROOT/tests/run-pass/epistemic_region_live_heap_stamp.sio"
[[ -f "$SRC" ]] || gate_fail "missing $SRC"
[[ -f "$LIVE" ]] || gate_fail "missing $LIVE"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/region-prov-gate.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

set +e
timeout 180 "$SOUC" run "$SRC" >"$TMP/out" 2>"$TMP/err"
RC=$?
set -e
echo "run_rc=$RC"
if [[ "$RC" != "0" ]]; then
  sed -n '1,40p' "$TMP/err" | sed 's/^/    /' >&2
  sed -n '1,20p' "$TMP/out" | sed 's/^/    /' >&2
  gate_fail "region_provenance.sio run rc=$RC"
fi
grep -q "ALL PASS" "$TMP/out" || gate_fail "missing ALL PASS in stdout"

# Live __rgn_cursor stamp needs a Madaros whose checker binds the intrinsic
# (check.sio checker_bind_import_unknown). Committed prebuilt may lag; prefer
# from-source artifacts/self-hosted/madaros when present.
MADAROS_FS="$ROOT/artifacts/self-hosted/madaros"
if [[ -x "$MADAROS_FS" ]]; then
  set +e
  timeout 180 env SOUNIO_MADAROS_BIN="$MADAROS_FS" SOUNIO_SOUC_ENGINE=madaros \
    "$SOUC" run "$LIVE" >"$TMP/live.out" 2>"$TMP/live.err"
  LRC=$?
  set -e
  echo "live_rc=$LRC"
  if [[ "$LRC" != "0" ]]; then
    sed -n '1,40p' "$TMP/live.err" | sed 's/^/    /' >&2
    sed -n '1,20p' "$TMP/live.out" | sed 's/^/    /' >&2
    gate_fail "epistemic_region_live_heap_stamp.sio run rc=$LRC"
  fi
  grep -q "LIVE_HEAP_STAMP_OK" "$TMP/live.out" || gate_fail "missing LIVE_HEAP_STAMP_OK"
  gate_pass "EpistemicInRegion + live __rgn_cursor stamp"
else
  gate_pass "EpistemicInRegion selftest (live cursor deferred until from-source Madaros)"
fi
echo "measured=yes"
exit 0
