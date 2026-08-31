#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${E2B_BASE_REF:-}"
if [[ -z "$BASE_REF" ]]; then
  BASE_REF="$(git merge-base HEAD "${ENIR_GATE_BASE:-origin/main}")"
fi
SEED="${SOUC_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
KEEP_DIR="${E2B_RECEIPT_DIR:-}"
TMP_DIR="$(mktemp -d /tmp/madaros-e2b-enir-cfg.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

DRIVER="$TMP_DIR/madaros-enir"
ORACLE="$TMP_DIR/eisa-v1-oracle"
ORACLE_OUT="$TMP_DIR/eisa-v1-oracle.out"
RECEIPT="$TMP_DIR/e2b-enir-cfg.receipt.json"

fail() {
  echo "E2B_ENIR_V1_FINITE_CFG_GATE_FAIL: $*" >&2
  exit 1
}

. scripts/dev/madaros_v2_enir_gate_scope.sh

[[ -x "$SEED" ]] || fail "missing executable Stage0 seed: $SEED"
git rev-parse --verify "$BASE_REF" >/dev/null 2>&1 || fail "base ref not found: $BASE_REF"
E2B_PROTECTED=(
  self-hosted/compiler/main.sio self-hosted/ir self-hosted/native self-hosted/wasm \
  self-hosted/gpu stdlib/runtime stdlib/eisa stdlib/math/dd64.sio \
)
madaros_v2_enir_gate_scope_or_skip "$BASE_REF" "E2B_ENIR_V1_FINITE_CFG_GATE" \
  "E2B changed a compiler codegen/ABI/runtime or canonical EISA-oracle surface" "${E2B_PROTECTED[@]}"

scripts/dev/souc-build-lock.sh "$SEED" self-hosted/enir/driver.sio "$DRIVER" >"$TMP_DIR/driver-build.log" 2>&1
[[ -s "$DRIVER" ]] || fail "native ENIR driver build produced no ELF"
if grep -Eq '^error(\[E[0-9]+\])?:|unknown identifier|typecheck: failed|assignment type mismatch' "$TMP_DIR/driver-build.log"; then
  tail -80 "$TMP_DIR/driver-build.log" >&2
  fail "Stage0 reported diagnostics while building ENIR driver"
fi
chmod +x "$DRIVER"

scripts/dev/souc-build-lock.sh "$SEED" tools/eisa/eisa_enir_v1_oracle.sio "$ORACLE" >"$TMP_DIR/oracle-build.log" 2>&1
[[ -s "$ORACLE" ]] || fail "source-fresh EISA v1 oracle build produced no ELF"
if grep -Eq '^error(\[E[0-9]+\])?:|unknown identifier|typecheck: failed|assignment type mismatch' "$TMP_DIR/oracle-build.log"; then
  tail -80 "$TMP_DIR/oracle-build.log" >&2
  fail "Stage0 reported diagnostics while building EISA v1 oracle"
fi
chmod +x "$ORACLE"
"$ORACLE" >"$ORACLE_OUT"
grep -Fq 'E2B_EISA_V1_ORACLE_PASS' "$ORACLE_OUT" || fail "EISA v1 oracle did not finish cleanly"
[[ "$(grep -c '^e2b-case-begin|' "$ORACLE_OUT")" == "8" ]] || fail "EISA v1 oracle program manifest drifted"
[[ "$(grep -c '^eisa-receipt:' "$ORACLE_OUT")" == "11" ]] || fail "EISA v1 oracle observation manifest drifted"

python3 scripts/dev/madaros_v2_e2b_enir_cfg_verify.py \
  --driver "$DRIVER" \
  --corpus tools/eisa/eisa_enir_v1_oracle.sio \
  --oracle "$ORACLE_OUT" \
  --out-dir "$TMP_DIR/cases" \
  --receipt "$RECEIPT" \
  --root "$ROOT_DIR"

mkdir -p "$TMP_DIR/negative"
cat >"$TMP_DIR/negative/v0_rejects_v1.eisa" <<'EOF'
epistemic fn v0_rejects_v1() {
let x=0.0
while x != 0.0 {
}
gate x
}
EOF
if "$DRIVER" lower "$TMP_DIR/negative/v0_rejects_v1.eisa" >/dev/null 2>&1; then
  fail "v0 entrypoint accepted v1 control flow"
fi
v0_negative_count=1

cat >"$TMP_DIR/negative/malformed_if.eisa" <<'EOF'
epistemic fn malformed_if() {
let x=0.0
if x <= 0.0 {
}
gate x
}
EOF
cat >"$TMP_DIR/negative/undefined_branch.eisa" <<'EOF'
epistemic fn undefined_branch() {
if missing < 0.0 {
}
let x=1.0
gate x
}
EOF
cat >"$TMP_DIR/negative/nonempty_while.eisa" <<'EOF'
epistemic fn nonempty_while() {
let x=0.0
while x != 0.0 {
gate x
}
gate x
}
EOF
cat >"$TMP_DIR/negative/conditional_definition.eisa" <<'EOF'
epistemic fn conditional_definition() {
let x=0.0-1.0
if x < 0.0 {
let y=1.0
}
gate x
}
EOF
cat >"$TMP_DIR/negative/conditional_gate.eisa" <<'EOF'
epistemic fn conditional_gate() {
let x=0.0-1.0
if x < 0.0 {
gate x
}
gate x
}
EOF
cat >"$TMP_DIR/negative/missing_root_close.eisa" <<'EOF'
epistemic fn missing_root_close() {
let x=0.0-1.0
if x < 0.0 {
gate x
}
EOF
cat >"$TMP_DIR/negative/fractional_fuel.eisa" <<'EOF'
epistemic fn fractional_fuel() {
fuel 1.5
let x=1.0
gate x
}
EOF
cat >"$TMP_DIR/negative/fuel_out_of_range.eisa" <<'EOF'
epistemic fn fuel_out_of_range() {
fuel 1000001
let x=1.0
gate x
}
EOF
cat >"$TMP_DIR/negative/late_fuel.eisa" <<'EOF'
epistemic fn late_fuel() {
let x=1.0
fuel 12
gate x
}
EOF
cat >"$TMP_DIR/negative/missing_gate.eisa" <<'EOF'
epistemic fn missing_gate() {
let x=1.0
}
EOF
cat >"$TMP_DIR/negative/scoped_duplicate.eisa" <<'EOF'
epistemic fn scoped_duplicate() {
let x=0.0-1.0
if x < 0.0 {
let x=1.0
}
gate x
}
EOF
python3 - "$TMP_DIR/negative/depth_capacity.eisa" <<'PY'
from pathlib import Path
import sys
lines = ["epistemic fn depth_capacity() {", "let x=0.0-1.0"]
lines += ["if x < 0.0 {"] * 17
lines += ["}"] * 17
lines += ["gate x", "}"]
Path(sys.argv[1]).write_text("\n".join(lines) + "\n", encoding="ascii")
PY
python3 - "$TMP_DIR/negative/symbol_capacity.eisa" <<'PY'
from pathlib import Path
import sys
lines = ["epistemic fn symbol_capacity() {"]
lines += [f"let x{i}={i}" for i in range(65)]
lines += ["gate x63", "}"]
Path(sys.argv[1]).write_text("\n".join(lines) + "\n", encoding="ascii")
PY

negative_count=0
for source in "$TMP_DIR"/negative/*.eisa; do
  [[ "$(basename "$source")" == "v0_rejects_v1.eisa" ]] && continue
  negative_count=$((negative_count + 1))
  if "$DRIVER" lower-v1 "$source" >"$source.out" 2>&1; then
    fail "native v1 source lowerer accepted negative fixture: $(basename "$source")"
  fi
  grep -Fq 'enir-lower-error|' "$source.out" || fail "negative source lacks classified lowering error: $(basename "$source")"
done
[[ "$negative_count" == "13" ]] || fail "v1 negative source count drift: $negative_count"
source_negative_count=$((v0_negative_count + negative_count))
[[ "$source_negative_count" == "14" ]] || fail "total source negative count drift: $source_negative_count"

tamper_count=0
tamper_rejected() {
  local name="$1"
  local path="$TMP_DIR/$name.enir"
  tamper_count=$((tamper_count + 1))
  if "$DRIVER" run "$path" >"$TMP_DIR/$name.log" 2>&1; then
    fail "interpreter accepted artifact tamper: $name"
  fi
}

cp "$TMP_DIR/cases/v1_loop.enir" "$TMP_DIR/target_oob.enir"
sed -i 's/op|2|11|-1|-1|0|-1|4|0|-1|1/op|2|11|-1|-1|0|-1|99|0|-1|1/' "$TMP_DIR/target_oob.enir"
tamper_rejected target_oob

cp "$TMP_DIR/cases/v1_loop.enir" "$TMP_DIR/backedge_wrong.enir"
sed -i 's/op|3|10|-1|-1|-1|-1|2|-1|-1|1/op|3|10|-1|-1|-1|-1|1|-1|-1|1/' "$TMP_DIR/backedge_wrong.enir"
tamper_rejected backedge_wrong

cp "$TMP_DIR/cases/v1_if_both.enir" "$TMP_DIR/policy_missing.enir"
sed -i 's/op|4|12|-1|-1|2|-1|6|0|-1|1/op|4|12|-1|-1|2|-1|6|-1|-1|1/' "$TMP_DIR/policy_missing.enir"
tamper_rejected policy_missing

cp "$TMP_DIR/cases/v1_loop.enir" "$TMP_DIR/halt_missing.enir"
sed -i 's/op|5|13|-1|-1|-1|-1|-1|-1|-1|1/op|5|10|-1|-1|-1|-1|0|-1|-1|1/' "$TMP_DIR/halt_missing.enir"
tamper_rejected halt_missing
[[ "$tamper_count" == "4" ]] || fail "artifact tamper count drift: $tamper_count"

cp "$TMP_DIR/cases/v1_loop.enir" "$TMP_DIR/noncanonical.enir"
sed -i 's/type|0|4|/type|00|4|/' "$TMP_DIR/noncanonical.enir"
if "$DRIVER" roundtrip "$TMP_DIR/noncanonical.enir" >/dev/null 2>&1; then
  fail "roundtrip accepted non-canonical v1 integer spelling"
fi

E2_BASE_REF=HEAD bash scripts/dev/madaros_v2_e2_enir_lowering_gate.sh >/dev/null

if [[ -n "$KEEP_DIR" ]]; then
  mkdir -p "$KEEP_DIR"
  cp "$RECEIPT" "$KEEP_DIR/e2b-enir-cfg.receipt.json"
  cp "$TMP_DIR"/cases/*.enir "$KEEP_DIR/"
fi

RECEIPT_SHA="$(sha256sum "$RECEIPT" | cut -d' ' -f1)"
echo "E2B_ENIR_V1_FINITE_CFG_FULL_GATE_PASS receipt_sha256=$RECEIPT_SHA programs=8 observations=11 cumulative=13/30,17/39 source_negatives=$source_negative_count artifact_tampers=4 canonicalization_tampers=1 control=taken,not-taken,poisoned,frail high_ids=20,23,20 native_independent=exact evm_observable=exact e2a_regression=pass e1_regression=pass codegen_diff=0"
