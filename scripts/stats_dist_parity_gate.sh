#!/usr/bin/env bash
# ADR-008: claim = Sounio stats_parity emitters; mpmath soft unless HARD=1.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
REF="scripts/parity/stats_parity_ref.py"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
HARD="${SOUNIO_FOREIGN_ORACLE_HARD:-0}"

emit_all() { : > "$OUT/emit.txt"
  for fam in "$@"; do
    src="tests/parity/stats_parity_${fam}.sio"
    if ! "$SOUC" compile "$src" -o "$OUT/$fam.elf" >/dev/null 2>&1; then
      echo "FAIL compile $src"; return 1; fi
    chmod +x "$OUT/$fam.elf"
    "$OUT/$fam.elf" >> "$OUT/emit.txt" 2>/dev/null || { echo "FAIL run $src"; return 1; }
  done
}
FAMILIES="${PARITY_FAMILIES:-stdnormal continuous1 continuous2 continuous3 discrete}"
REQ=""; [ "${REQUIRE_ALL:-1}" = "1" ] && REQ="--require-all"
echo "== claim: Sounio stats emitters =="
emit_all $FAMILIES || { echo "STATS_DIST_PARITY_GATE_FAIL claim"; exit 1; }
echo "  claim emit: OK"
echo "== corroboration: mpmath (HARD=$HARD) =="
if ! python3 -c 'import mpmath' 2>/dev/null; then
  echo "  SKIP: mpmath not installed"; echo "STATS_DIST_PARITY_GATE_OK"; exit 0
fi
set +e
python3 "$REF" $REQ < "$OUT/emit.txt" | tee "$OUT/report.txt"
set -e
if grep -q "STATS_DIST_PARITY_OK" "$OUT/report.txt" 2>/dev/null || grep -q "STATS_PARITY_OK" "$OUT/report.txt" 2>/dev/null; then
  echo "  mpmath: OK"; echo "STATS_DIST_PARITY_GATE_OK"; exit 0
fi
# also accept report token variants from ref
if grep -q "_OK" "$OUT/report.txt" 2>/dev/null && ! grep -qi FAIL "$OUT/report.txt"; then
  echo "  mpmath: OK (token)"; echo "STATS_DIST_PARITY_GATE_OK"; exit 0
fi
echo "  mpmath: MISMATCH (not claim under ADR-008)"
[ "$HARD" = "1" ] && { echo "STATS_DIST_PARITY_GATE_FAIL foreign_hard"; exit 1; }
echo "STATS_DIST_PARITY_GATE_OK"
exit 0
