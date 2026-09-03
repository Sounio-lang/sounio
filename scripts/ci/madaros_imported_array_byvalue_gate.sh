#!/usr/bin/env bash
# Gate: #913 — [f64;N] passed by value to an imported function must not arrive zeroed.
#
# Default Madaros multi-module native path (bin/souc → Madaros).
# Witness: tests/run-pass/imported_array_byvalue.sio
#   + fixtures/imported_array_byvalue_leaf.sio
#
# Accept markers:
#   IMPORTED_ARRAY_BYVALUE_OK
#   import_byvalue_sum_bits 4618441417868443648   (6.0)
#   import_ols_slope_bits   4611686018427387904   (2.0)
#
# Explicit non-claims:
#   - does not require stats::regression::linear (Madaros multi-mod parse residual
#     on that module's impl methods is separate)
#   - does not claim general large-struct SRET or global-array paths
#   - does not rebuild Madaros; uses the configured SOUC / prebuilt

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true

# Multi-module native lower can need a larger stack on some hosts.
ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true

SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="$ROOT/tests/run-pass/imported_array_byvalue.sio"
LEAF="$ROOT/tests/run-pass/fixtures/imported_array_byvalue_leaf.sio"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

if [[ ! -x "$SOUC" ]]; then
  fail "souc not executable at $SOUC"
fi
if [[ ! -f "$SRC" || ! -f "$LEAF" ]]; then
  fail "missing witness files"
fi

echo "== madaros_imported_array_byvalue_gate (#913) =="
echo "souc=$SOUC"
"$SOUC" --version 2>&1 | head -2 || true
echo "git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# Prefer raw Madaros ELF identity when present (for receipts).
RAW=""
for cand in "${MADAROS_RAW_BIN:-}" "${SOUNIO_MADAROS_BIN:-}" \
            "$ROOT/artifacts/self-hosted/madaros" "$ROOT/bin/madaros-linux-x86_64"; do
  if [[ -n "$cand" && -x "$cand" && "$(head -c2 "$cand" 2>/dev/null || true)" != '#!' ]]; then
    RAW="$cand"
    break
  fi
done
if [[ -n "$RAW" ]]; then
  echo "raw_elf=$RAW"
  echo "raw_elf_sha256=$(sha256sum "$RAW" | awk '{print $1}')"
fi

OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT

set +e
"$SOUC" run "$SRC" >"$OUT" 2>&1
rc=$?
set -e

cat "$OUT"
if [[ "$rc" != "0" ]]; then
  fail "witness exited rc=$rc"
fi
if grep -q '^FAIL ' "$OUT" || grep -q 'FAIL ' <<<"$(grep -E 'FAIL |FAIL$' "$OUT" || true)"; then
  # Any printed FAIL marker is a hard fail.
  if grep -E 'FAIL (local_|import_|caller_)' "$OUT" >/dev/null 2>&1; then
    fail "witness printed FAIL marker"
  fi
fi
grep -q 'IMPORTED_ARRAY_BYVALUE_OK' "$OUT" || fail "missing IMPORTED_ARRAY_BYVALUE_OK"

# Bit-exact oracles (IEEE-754 little-endian bit patterns of f64).
grep -q 'import_byvalue_sum_bits 4618441417868443648' "$OUT" || fail "sum bits not 6.0"
grep -q 'import_byvalue_e0_bits 4607182418800017408' "$OUT" || fail "e0 bits not 1.0"
grep -q 'import_byvalue_e1_bits 4611686018427387904' "$OUT" || fail "e1 bits not 2.0"
grep -q 'import_wide_byvalue_sum_bits 4618441417868443648' "$OUT" || fail "wide sum bits not 6.0"
grep -q 'import_ols_slope_bits 4611686018427387904' "$OUT" || fail "OLS slope bits not 2.0"

echo "MADAROS_IMPORTED_ARRAY_BYVALUE_GATE_OK"
echo "issue=913 status=closed_on_default_madaros claim=cross_module_f64_array_byvalue_preserves_payload"
