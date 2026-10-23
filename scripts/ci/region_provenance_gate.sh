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
[[ -f "$SRC" ]] || gate_fail "missing $SRC"

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

gate_pass "EpistemicInRegion selftest"
echo "measured=yes"
exit 0
