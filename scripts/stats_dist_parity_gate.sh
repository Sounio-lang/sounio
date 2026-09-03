#!/usr/bin/env bash
# stats↔mpmath distribution parity gate. Reference = mpmath (dps=30).
# lean_single-LOCKED (reuses the #1210 bit-exact bridge). Dev-tier; not in ci.yml.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
export SOUNIO_SOUC_ENGINE=lean_single
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
REF="scripts/parity/stats_parity_ref.py"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
python3 -c 'import mpmath' 2>/dev/null || { echo "SKIP: mpmath not installed"; exit 0; }

emit_all() { : > "$OUT/emit.txt"
  for fam in "$@"; do
    src="tests/parity/stats_parity_${fam}.sio"
    if ! "$SOUC" compile "$src" -o "$OUT/$fam.elf" >/dev/null 2>&1; then
      echo "FAIL compile $src"; return 1; fi
    chmod +x "$OUT/$fam.elf"; "$OUT/$fam.elf" >> "$OUT/emit.txt" 2>/dev/null || { echo "FAIL run $src"; return 1; }
  done
}

FAMILIES="${PARITY_FAMILIES:-continuous1 stdnormal continuous2 continuous3 discrete}"
REQ=""; [ "${REQUIRE_ALL:-1}" = "1" ] && REQ="--require-all"
emit_all $FAMILIES || exit 1
python3 "$REF" $REQ < "$OUT/emit.txt" | tee "$OUT/report.txt"
grep -q "STATS_DIST_PARITY_OK" "$OUT/report.txt" \
  && echo "STATS_DIST_PARITY_GATE_OK" || { echo "GATE FAILED"; exit 1; }
