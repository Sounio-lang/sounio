#!/usr/bin/env bash
# scripts/madaros_dual_import_gate.sh
#
# Dual-module import under default Madaros: epistemic::gum + epistemic::knowledge
# in one program must type-check, compile, and run with DUAL_GUM_KNOWLEDGE_OK.
#
# Check path (pre-#1245): 51× false E175 on private helpers that share names
# across the two stdlib modules (chk / near / test_combine). Fix:
# self-hosted/check/defs.sio fn_sig_table_find_prefer_module +
# self-hosted/check/check.sio checker_fn_sigs_find_inplace.
#
# Run path (pre-2026-07-20): witness called non-existent e.mean(); multi-module
# check accepted it and native lower SEGV'd at lower_array:seed_begin. Correct
# API is e.val() (or ep_val free fn). Dual modules with real methods already
# compile+run; gate now *requires* run success (no longer best-effort WARN).
#
# Also re-runs knowledge-alone, gum-alone, and Root 2 multi-module method so a
# dual-import patch cannot regress those gates. Does NOT touch clinical/ousadia.
#
# Requires a current-source Madaros (artifacts/self-hosted/madaros or
# MADAROS_RAW_BIN). Checked-in bin/madaros-linux-x86_64 may predate the check fix.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true

# Dual multi-module lower+codegen needs ~120000 KiB soft stack (65536 KiB
# lowers cleanly then fails to emit ELF). Prefer unlimited; fall back to 128 MiB.
ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true

SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
fail=0

echo "== madaros_dual_import_gate =="
"$SOUC" --version 2>&1 | head -2 || true

RAW=""
for cand in "${MADAROS_RAW_BIN:-}" "${SOUNIO_MADAROS_BIN:-}" \
            "$ROOT/artifacts/self-hosted/madaros" "$ROOT/bin/madaros-linux-x86_64"; do
  if [[ -n "$cand" && -x "$cand" && "$(head -c2 "$cand" 2>/dev/null || true)" != '#!' ]]; then
    RAW="$cand"
    break
  fi
done
if [[ -z "$RAW" ]]; then
  echo "MADAROS_DUAL_IMPORT_GATE_BLOCKED reason=no_raw_madaros" >&2
  exit 1
fi
echo "raw_elf=$RAW"
echo "raw_elf_sha256=$(sha256sum "$RAW" | awk '{print $1}')"
echo "git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

check_ok() {
  local name="$1" src="$2"
  echo "== check: $name =="
  if ! "$SOUC" check "$src" >"$OUT/c.log" 2>&1; then
    echo "FAIL: check $src"
    # Surface E175 specifically — the historical dual-import failure mode.
    if grep -q 'error\[E175' "$OUT/c.log" 2>/dev/null; then
      echo "  (observed E175 — dual-import / private-name collision regression?)"
    fi
    tail -20 "$OUT/c.log" || true
    fail=1
    return
  fi
  if ! grep -q 'check: OK' "$OUT/c.log" 2>/dev/null && ! grep -q 'verdict=0' "$OUT/c.log" 2>/dev/null; then
    # Some engines print only verdict=0
    if grep -q 'verdict=1' "$OUT/c.log" 2>/dev/null; then
      echo "FAIL: check $src (verdict=1)"
      tail -20 "$OUT/c.log" || true
      fail=1
      return
    fi
  fi
  echo "PASS: check $name"
}

# Primary dual-import witness
check_ok "dual gum+knowledge" tests/run-pass/madaros_dual_gum_knowledge.sio

# Alone controls (must stay green)
cat >"$OUT/knowledge_alone.sio" <<'EOF'
use epistemic::knowledge::{Epistemic}
fn main() -> i64 with IO, Mut, Div, Panic {
    let e = Epistemic::measured(10.0, 0.5)
    let m = e.val()
    if m > 0.0 { 0 } else { 1 }
}
EOF
cat >"$OUT/gum_alone.sio" <<'EOF'
use epistemic::gum::{gum_type_a, gum_type_b, gum_combine2, gum_k95}
fn main() -> i64 with IO, Mut, Div, Panic {
    let ua = gum_type_a(0.070710, 5)
    let ub = gum_type_b(0.05)
    let r = gum_combine2(98.3, ua, ub)
    let k = gum_k95(r)
    if k > 0.0 { 0 } else { 1 }
}
EOF
check_ok "knowledge alone" "$OUT/knowledge_alone.sio"
check_ok "gum alone" "$OUT/gum_alone.sio"

# Root 2 multi-module methods must not regress (#1227)
if [[ -f tests/run-pass/madaros_root2_multimodule_method.sio ]]; then
  check_ok "root2 multimodule method" tests/run-pass/madaros_root2_multimodule_method.sio
fi

# Negative control: true private cross-module access still E175
echo "== negative: visibility_fn_private =="
if "$SOUC" check tests/multimodule/visibility_fn_private_main.sio >"$OUT/neg.log" 2>&1; then
  echo "FAIL: private cross-module call was accepted (E175 gate broken)"
  fail=1
else
  if grep -q 'error\[E175' "$OUT/neg.log"; then
    echo "PASS: private cross-module still E175"
  else
    echo "FAIL: private call rejected but without E175"
    tail -15 "$OUT/neg.log" || true
    fail=1
  fi
fi

# Required run of dual witness (native multi-module compile + execute)
echo "== run (required): dual gum+knowledge =="
run_ok=0
if ! "$SOUC" compile tests/run-pass/madaros_dual_gum_knowledge.sio -o "$OUT/dual.elf" >"$OUT/compile.log" 2>&1; then
  echo "FAIL: dual compile (native multi-module)"
  # SEGV during lower often exits 139 with log ending at lower_array:seed_begin
  if grep -q 'lower_array: seed_begin' "$OUT/compile.log" 2>/dev/null \
     && ! grep -q 'lower_array: seed_done\|Written to\|Compilation successful' "$OUT/compile.log" 2>/dev/null; then
    echo "  (log ends at lower_array:seed_begin — missing method / Root-2 residual?)"
  fi
  tail -20 "$OUT/compile.log" || true
  fail=1
else
  chmod +x "$OUT/dual.elf"
  set +e
  "$OUT/dual.elf" >"$OUT/run.log" 2>&1
  run_ec=$?
  set -e
  if [[ $run_ec -eq 0 ]] && grep -q 'DUAL_GUM_KNOWLEDGE_OK' "$OUT/run.log"; then
    echo "PASS: run dual gum+knowledge"
    cat "$OUT/run.log" || true
    run_ok=1
  else
    echo "FAIL: dual run exit=$run_ec (expected 0 and DUAL_GUM_KNOWLEDGE_OK)"
    cat "$OUT/run.log" 2>/dev/null || true
    fail=1
  fi
fi

mkdir -p "$ROOT/artifacts/compiler"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
STATUS=fail
[[ $fail -eq 0 ]] && STATUS=pass
CLAIMS_JSON='[
    "dual_gum_knowledge_check_ok",
    "knowledge_alone_check_ok",
    "gum_alone_check_ok",
    "private_cross_module_still_e175"'
if [[ $run_ok -eq 1 ]]; then
  CLAIMS_JSON+=',
    "dual_gum_knowledge_native_run_ok"'
fi
CLAIMS_JSON+='
  ]'
cat >"$ROOT/artifacts/compiler/madaros_dual_import_receipt.v1.json" <<EOF
{
  "schema": "madaros_dual_import_receipt.v1",
  "status": "$STATUS",
  "engine": "madaros_default",
  "commit": "$COMMIT",
  "raw_elf": "$RAW",
  "native_run": $run_ok,
  "claims": $CLAIMS_JSON,
  "claims_not_made": [
    "all_stdlib_dual_pairs",
    "unknown_method_rejected_at_check",
    "scalar_kind_global_change"
  ]
}
EOF
echo "receipt: $ROOT/artifacts/compiler/madaros_dual_import_receipt.v1.json"

if [[ $fail -eq 0 ]]; then
  echo "MADAROS_DUAL_IMPORT_GATE_OK"
  exit 0
fi
exit 1
