#!/usr/bin/env bash
# scripts/madaros_imported_f64_bss_arith_gate.sh
#
# Wave15 D — science multi-mod pillar: f64 BSS arithmetic inside imported
# modules (stats::densities::lognormal_pdf residual under default Madaros).
#
# Pre-fix: seed Wave13 external BSS preseed allocates DE, then into-acc
# lowerer_from_acc_module starts with empty global_types and skips re-record
# when the slot already exists → IrLoadGlobal without ir_mark_float_reg →
# native binops cvtsi2sd the IEEE bit pattern (DE+1 → ~4.6e18;
# lognormal_pdf → ~1e-300 via de_exp overflow clamp).
#
# Arms:
#   1. micro multi-mod leaf arith (add/sub/mul on module-level f64)
#   2. stats::densities lognormal science vertical
#   3. dual + cd_exact non-regression (optional if SKIP_REGRESSION=1)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true
ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true

SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

MICRO="$ROOT/tests/run-pass/imported_f64_bss_arith_main.sio"
SCI="$ROOT/tests/run-pass/imported_f64_lognormal_science.sio"
RECEIPT="$ROOT/artifacts/compiler/madaros_imported_f64_bss_arith_receipt.v1.json"

echo "== madaros_imported_f64_bss_arith_gate =="
"$SOUC" --version 2>&1 | head -2 || true

RAW=""
for cand in "${MADAROS_RAW_BIN:-}" "${SOUNIO_MADAROS_BIN:-}" \
            "$ROOT/artifacts/self-hosted/madaros" \
            "$ROOT/bin/madaros-linux-x86_64"; do
  if [[ -n "$cand" && -x "$cand" && "$(head -c2 "$cand" 2>/dev/null || true)" != '#!' ]]; then
    RAW="$cand"
    break
  fi
done
if [[ -z "$RAW" ]]; then
  echo "MADAROS_IMPORTED_F64_BSS_ARITH_GATE_BLOCKED reason=no_raw_madaros" >&2
  exit 1
fi
echo "raw_elf=$RAW"
echo "raw_elf_sha256=$(sha256sum "$RAW" | awk '{print $1}')"
echo "git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# Ensure wrapper resolves the chosen raw ELF (source fix requires rebuilt Madaros).
export SOUNIO_MADAROS_BIN="$RAW"
export MADAROS_RAW_BIN="$RAW"

run_ok() {
  local name="$1" src="$2" marker="$3"
  echo "== run: $name =="
  local log="$OUT/${name}.log"
  if ! "$SOUC" run "$src" >"$log" 2>&1; then
    echo "FAIL: run $src"
    tail -40 "$log" || true
    return 1
  fi
  if ! grep -q "$marker" "$log"; then
    echo "FAIL: missing marker $marker"
    tail -40 "$log" || true
    return 1
  fi
  if grep -q 'FAIL ' "$log"; then
    echo "FAIL: assertion marker in output"
    tail -40 "$log" || true
    return 1
  fi
  # Surface key lines
  grep -E '^(de_bits|neg_bits|plus1_bits|times2_bits|lnpdf_bits|exp_bits|IMPORTED_)' "$log" || true
  echo "PASS: $name"
  return 0
}

fail=0
run_ok "micro_bss_arith" "$MICRO" "IMPORTED_F64_BSS_ARITH_OK" || fail=1
run_ok "lognormal_science" "$SCI" "IMPORTED_F64_LOGNORMAL_SCIENCE_OK" || fail=1

if [[ "${SKIP_REGRESSION:-0}" != "1" ]]; then
  echo "== regression: dual_import =="
  if ! bash "$ROOT/scripts/madaros_dual_import_gate.sh" >"$OUT/dual.log" 2>&1; then
    echo "FAIL: dual_import regression"
    tail -30 "$OUT/dual.log" || true
    fail=1
  else
    echo "PASS: dual_import"
  fi
  echo "== regression: cd_exact_e2e =="
  if ! bash "$ROOT/scripts/madaros_cd_exact_e2e_gate.sh" >"$OUT/cd.log" 2>&1; then
    echo "FAIL: cd_exact_e2e regression"
    tail -30 "$OUT/cd.log" || true
    fail=1
  else
    echo "PASS: cd_exact_e2e"
  fi
fi

if [[ "$fail" -ne 0 ]]; then
  echo "MADAROS_IMPORTED_F64_BSS_ARITH_GATE_FAIL" >&2
  exit 1
fi

mkdir -p "$(dirname "$RECEIPT")"
cat >"$RECEIPT" <<JSON
{
  "schema": "madaros_imported_f64_bss_arith_receipt.v1",
  "gate": "scripts/madaros_imported_f64_bss_arith_gate.sh",
  "marker": "MADAROS_IMPORTED_F64_BSS_ARITH_GATE_OK",
  "git_sha": "$(git rev-parse HEAD 2>/dev/null || echo unknown)",
  "raw_elf": "$RAW",
  "raw_elf_sha256": "$(sha256sum "$RAW" | awk '{print $1}')",
  "engine": "$("$SOUC" --version 2>&1 | head -1 | tr -d '\n')",
  "proven": [
    "imported_module_f64_bss_add_sub_mul",
    "stats_densities_lognormal_pdf_multi_mod",
    "dual_import_non_regression",
    "cd_exact_e2e_non_regression"
  ],
  "claim_boundary": "Imported-module same-module f64 BSS arithmetic under into-acc multi-mod lower is float-typed (no cvtsi2sd of IEEE bits). lognormal_pdf(1,0,1) ≈ 1/sqrt(2π). Does not claim full stats::regression multi-mod, OLS denser path, or print_f64."
}
JSON
echo "receipt: $RECEIPT"
echo "MADAROS_IMPORTED_F64_BSS_ARITH_GATE_OK"
