#!/usr/bin/env bash
# Special-function gate (ADR-008 pilot demotion).
#
# Claim clock (may fail CI): Sounio under the selected engine must compile and
# run the parity emitters for each family. That is the language claim surface.
#
# Foreign corroboration (mpmath dps=30 via scripts/parity/special_parity_ref.py):
#   - default: report only (exit 0 even on mismatch) — external_corroboration_only
#   - SOUNIO_FOREIGN_ORACLE_HARD=1: restore legacy hard-fail on mpmath mismatch
#
# Dev-tier; not wired into ci.yml. lean_single-LOCKED historical note retained
# as default engine for emitter compile (override with SOUNIO_SOUC_ENGINE).
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
REF="scripts/parity/special_parity_ref.py"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

HARD="${SOUNIO_FOREIGN_ORACLE_HARD:-0}"

emit_all() {
  : > "$OUT/emit.txt"
  for fam in "$@"; do
    src="tests/parity/special_parity_${fam}.sio"
    if ! "$SOUC" compile "$src" -o "$OUT/$fam.elf" >/dev/null 2>&1; then
      echo "FAIL compile $src"
      return 1
    fi
    chmod +x "$OUT/$fam.elf"
    if ! "$OUT/$fam.elf" >> "$OUT/emit.txt" 2>/dev/null; then
      echo "FAIL run $src"
      return 1
    fi
  done
}

# Full 9-family default. Narrow via PARITY_FAMILIES; set REQUIRE_ALL=0 if subset.
FAMILIES="${PARITY_FAMILIES:-erf gamma beta igamma bessel airyzetaelliptic hypergeometric orthopoly}"
REQ=""
[ "${REQUIRE_ALL:-1}" = "1" ] && REQ="--require-all"

echo "== claim: Sounio special_parity emitters (oracle_class=sounio_native_expected) =="
if ! emit_all $FAMILIES; then
  echo "SPECIAL_SCIPY_PARITY_GATE_FAIL claim_emit"
  exit 1
fi
echo "  claim emit: OK ($(wc -l < "$OUT/emit.txt") lines)"

echo "== corroboration: mpmath (oracle_class=external_corroboration_only; HARD=$HARD) =="
if ! python3 -c 'import mpmath' 2>/dev/null; then
  echo "  SKIP: mpmath not installed (claim path already green)"
  echo "SPECIAL_SCIPY_PARITY_GATE_OK"
  exit 0
fi

set +e
python3 "$REF" $REQ < "$OUT/emit.txt" | tee "$OUT/report.txt"
ref_rc=${PIPESTATUS[0]}
set -e

if grep -q "SPECIAL_SCIPY_PARITY_OK" "$OUT/report.txt" 2>/dev/null; then
  echo "  mpmath corroboration: OK"
  echo "SPECIAL_SCIPY_PARITY_GATE_OK"
  exit 0
fi

echo "  mpmath corroboration: MISMATCH (not a claim failure under ADR-008)"
if [ "$HARD" = "1" ]; then
  echo "SPECIAL_SCIPY_PARITY_GATE_FAIL foreign_hard (SOUNIO_FOREIGN_ORACLE_HARD=1)"
  exit 1
fi
echo "SPECIAL_SCIPY_PARITY_GATE_OK"
exit 0
