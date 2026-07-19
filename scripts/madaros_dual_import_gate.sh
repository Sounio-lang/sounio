#!/usr/bin/env bash
# scripts/madaros_dual_import_gate.sh
#
# Dual-module import under default Madaros: epistemic::gum + epistemic::knowledge
# in one program must type-check and (when the native path is live) run.
#
# Pre-fix: 51× false E175 on private helpers that share names across the two
# stdlib modules (chk / near / test_combine). Each module alone was fine.
# Fix: self-hosted/check/defs.sio fn_sig_table_find_prefer_module +
# self-hosted/check/check.sio checker_fn_sigs_find_inplace.
#
# Also re-runs knowledge-alone, gum-alone, and Root 2 multi-module method so a
# dual-import patch cannot regress those gates. Does NOT touch clinical/ousadia.
#
# Requires a current-source Madaros (artifacts/self-hosted/madaros or
# MADAROS_RAW_BIN). Checked-in bin/madaros-linux-x86_64 may predate the fix.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true

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
    let m = e.mean()
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

# Best-effort run of dual witness (native multi-module may still be fragile)
echo "== run (best-effort): dual gum+knowledge =="
if "$SOUC" compile tests/run-pass/madaros_dual_gum_knowledge.sio -o "$OUT/dual.elf" >"$OUT/compile.log" 2>&1; then
  chmod +x "$OUT/dual.elf"
  if "$OUT/dual.elf" >"$OUT/run.log" 2>&1 && grep -q 'DUAL_GUM_KNOWLEDGE_OK' "$OUT/run.log"; then
    echo "PASS: run dual gum+knowledge"
    cat "$OUT/run.log" || true
  else
    echo "WARN: compile ok but run missing DUAL_GUM_KNOWLEDGE_OK (native residual; check still required)"
    cat "$OUT/run.log" 2>/dev/null || true
    # Do not fail the gate on native run residual — the bug under repair is E175 preflight.
  fi
else
  echo "WARN: dual compile failed (native multi-module residual; check still required)"
  tail -10 "$OUT/compile.log" || true
fi

mkdir -p "$ROOT/artifacts/compiler"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
STATUS=fail
[[ $fail -eq 0 ]] && STATUS=pass
cat >"$ROOT/artifacts/compiler/madaros_dual_import_receipt.v1.json" <<EOF
{
  "schema": "madaros_dual_import_receipt.v1",
  "status": "$STATUS",
  "engine": "madaros_default",
  "commit": "$COMMIT",
  "raw_elf": "$RAW",
  "claims": [
    "dual_gum_knowledge_check_ok",
    "knowledge_alone_check_ok",
    "gum_alone_check_ok",
    "private_cross_module_still_e175"
  ],
  "claims_not_made": [
    "native_run_guaranteed",
    "all_stdlib_dual_pairs",
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
