#!/usr/bin/env bash
# SciPy↔Sounio special-function parity gate. Reference = mpmath (dps=30).
# lean_single-LOCKED (Phase-0). Dev-tier; not wired into ci.yml.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
export SOUNIO_SOUC_ENGINE=lean_single
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
REF="scripts/parity/special_parity_ref.py"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
python3 -c 'import mpmath' 2>/dev/null || { echo "SKIP: mpmath not installed"; exit 0; }

emit_all() { : > "$OUT/emit.txt"
  for fam in "$@"; do
    src="tests/parity/special_parity_${fam}.sio"
    if ! "$SOUC" compile "$src" -o "$OUT/$fam.elf" >/dev/null 2>&1; then
      echo "FAIL compile $src"; return 1; fi
    chmod +x "$OUT/$fam.elf"; "$OUT/$fam.elf" >> "$OUT/emit.txt" 2>/dev/null || { echo "FAIL run $src"; return 1; }
  done
}

# Full 9-family default. A dev may narrow it via PARITY_FAMILIES and set
# REQUIRE_ALL=0 (a subset would otherwise fail the coverage assertion).
FAMILIES="${PARITY_FAMILIES:-erf gamma beta igamma bessel airyzetaelliptic hypergeometric orthopoly}"
REQ=""; [ "${REQUIRE_ALL:-1}" = "1" ] && REQ="--require-all"
emit_all $FAMILIES || exit 1
python3 "$REF" $REQ < "$OUT/emit.txt" | tee "$OUT/report.txt"
grep -q "SPECIAL_SCIPY_PARITY_OK" "$OUT/report.txt" \
  && echo "SPECIAL_SCIPY_PARITY_GATE_OK" || { echo "GATE FAILED"; exit 1; }
