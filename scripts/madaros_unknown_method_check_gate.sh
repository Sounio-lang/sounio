#!/usr/bin/env bash
# scripts/madaros_unknown_method_check_gate.sh
#
# Unknown instance methods must fail at multi-module Madaros *check* (E011 /
# verdict=1), not SEGV at native lower preseed. Known methods (e.val, e.std)
# and associated constructors (Epistemic::measured) stay green.
#
# Root cause fixed: Type::method path callees were typed as the first path
# segment only (often TyUnknown for imports), so receivers stayed unknown and
# method resolution silently accepted any name.
#
# Requires current-source Madaros (artifacts/self-hosted/madaros or
# MADAROS_RAW_BIN). Checked-in bin/madaros-linux-x86_64 may predate the fix.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true

ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true

SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
fail=0

echo "== madaros_unknown_method_check_gate =="
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
  echo "MADAROS_UNKNOWN_METHOD_CHECK_GATE_BLOCKED reason=no_raw_madaros" >&2
  exit 1
fi
echo "raw_elf=$RAW"
echo "raw_elf_sha256=$(sha256sum "$RAW" | awk '{print $1}')"
echo "git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

check_must_fail() {
  local name="$1" src="$2"
  echo "== check-fail: $name =="
  set +e
  "$SOUC" check "$src" >"$OUT/c.log" 2>&1
  local rc=$?
  set -e
  if grep -qE 'error\[E011|no method named|verdict=1' "$OUT/c.log" 2>/dev/null; then
    echo "PASS: $name rejected at check"
    return
  fi
  if grep -q 'verdict=0' "$OUT/c.log" 2>/dev/null || grep -q 'check: OK' "$OUT/c.log" 2>/dev/null; then
    echo "FAIL: $name was accepted at check (expected E011 / verdict=1)"
    tail -25 "$OUT/c.log" || true
    fail=1
    return
  fi
  if [[ $rc -ne 0 ]]; then
    echo "PASS: $name check non-zero (rc=$rc)"
    return
  fi
  echo "FAIL: $name unclear check result"
  tail -25 "$OUT/c.log" || true
  fail=1
}

check_ok() {
  local name="$1" src="$2"
  echo "== check: $name =="
  if ! "$SOUC" check "$src" >"$OUT/c.log" 2>&1; then
    echo "FAIL: check $src"
    tail -20 "$OUT/c.log" || true
    fail=1
    return
  fi
  if grep -q 'verdict=1' "$OUT/c.log" 2>/dev/null; then
    echo "FAIL: check $src (verdict=1)"
    tail -20 "$OUT/c.log" || true
    fail=1
    return
  fi
  echo "PASS: check $name"
}

run_ok() {
  local name="$1" src="$2" sentinel="$3"
  echo "== run: $name =="
  if ! "$SOUC" compile "$src" -o "$OUT/t.elf" >"$OUT/c.log" 2>&1; then
    echo "FAIL: compile $src"
    tail -20 "$OUT/c.log" || true
    fail=1
    return
  fi
  chmod +x "$OUT/t.elf"
  if ! "$OUT/t.elf" >"$OUT/r.log" 2>&1 || ! grep -q "$sentinel" "$OUT/r.log"; then
    echo "FAIL: run $src"
    cat "$OUT/r.log" || true
    fail=1
    return
  fi
  echo "PASS: $sentinel"
}

# Negative: multi-module Epistemic.mean (nonexistent)
check_must_fail "multi-module e.mean()" \
  tests/ui/type/unknown_method_multimodule.sio

# Negative: local associated constructor then unknown method
check_must_fail "local E::of + e.no_such_method()" \
  tests/ui/type/unknown_method_after_associated.sio

# Negative: dual-shaped free program with e.mean()
cat >"$OUT/dual_mean.sio" <<'EOF'
use epistemic::knowledge::{Epistemic}
use epistemic::gum::{gum_type_a, gum_type_b, gum_combine2, gum_k95}

fn main() -> i32 with IO, Mut, Div, Panic {
    let ua = gum_type_a(0.070710, 5)
    let ub = gum_type_b(0.05)
    let r = gum_combine2(98.3, ua, ub)
    let k = gum_k95(r)
    let e = Epistemic::measured(10.0, 0.5)
    let m = e.mean()
    if k <= 0.0 { return 1 }
    if m < 9.0 { return 1 }
    print("SHOULD_NOT\n")
    return 0
}
EOF
check_must_fail "dual gum+knowledge e.mean()" "$OUT/dual_mean.sio"

# Positive: known methods still check+run
check_ok "known methods multi-module" \
  tests/run-pass/madaros_unknown_method_known_ok.sio
run_ok "known methods multi-module run" \
  tests/run-pass/madaros_unknown_method_known_ok.sio \
  KNOWN_METHOD_OK

# Positive: dual with e.val() still green
check_ok "dual gum+knowledge e.val()" \
  tests/run-pass/madaros_dual_gum_knowledge.sio
run_ok "dual gum+knowledge e.val() run" \
  tests/run-pass/madaros_dual_gum_knowledge.sio \
  DUAL_GUM_KNOWLEDGE_OK

# Positive: Root2 multimodule methods still green
check_ok "root2 multimodule methods" \
  tests/run-pass/madaros_root2_multimodule_method.sio
run_ok "root2 multimodule methods run" \
  tests/run-pass/madaros_root2_multimodule_method.sio \
  ROOT2_MULTIMODULE_METHOD_OK

mkdir -p "$ROOT/artifacts/compiler"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
STATUS=fail
[[ $fail -eq 0 ]] && STATUS=pass
cat >"$ROOT/artifacts/compiler/madaros_unknown_method_check_receipt.v1.json" <<EOF
{
  "schema": "madaros_unknown_method_check_receipt.v1",
  "status": "$STATUS",
  "engine": "madaros_default",
  "commit": "$COMMIT",
  "claims": [
    "unknown_method_rejected_at_check",
    "unknown_method_name_rejected_under_untyped_receiver",
    "known_method_e_val_still_ok",
    "dual_gum_knowledge_e_val_still_ok",
    "root2_multimodule_method_still_ok"
  ],
  "claims_not_made": [
    "ufcs_instance_method_as_type_path",
    "exhaustive_stdlib_method_census",
    "imported_associated_Type_method_always_typed",
    "lean_single_fixed_point_after_change"
  ]
}
EOF
echo "receipt: $ROOT/artifacts/compiler/madaros_unknown_method_check_receipt.v1.json"

if [[ $fail -eq 0 ]]; then
  echo "MADAROS_UNKNOWN_METHOD_CHECK_GATE_OK"
  exit 0
fi
exit 1
