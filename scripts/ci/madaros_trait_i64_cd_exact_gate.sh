#!/usr/bin/env bash
# Madaros trait-for-i64 method lower + cd_exact_generic_i64 closeout (1→2→3).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE MADAROS_RAW_BIN SOUNIO_MADAROS_BIN || true
SOUC="${SOUC:-$ROOT/bin/souc}"

echo "== trait-i64 method lower =="
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
ELF="$OUT/trait.elf"
SRC="docs/handoff/repros/madaros_trait_i64_method_lower.sio"
if ! "$SOUC" compile "$SRC" -o "$ELF" >"$OUT/compile.log" 2>&1; then
  echo "FAIL: trait compile"
  tail -40 "$OUT/compile.log" || true
  exit 1
fi
chmod +x "$ELF"
if ! "$ELF" >"$OUT/run.log" 2>&1; then
  echo "FAIL: trait run"
  cat "$OUT/run.log" || true
  exit 1
fi
grep -q 'ER_TRAIT_I64_OK' "$OUT/run.log" || {
  echo "FAIL: missing ER_TRAIT_I64_OK"
  cat "$OUT/run.log" || true
  exit 1
}

echo "== cd_exact_generic_i64 =="
bash scripts/dev/madaros_cd_exact_generic_i64_gate.sh

echo "MADAROS_TRAIT_I64_CD_EXACT_GATE_OK"
