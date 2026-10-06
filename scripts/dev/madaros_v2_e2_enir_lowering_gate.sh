#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${E2_BASE_REF:-}"
if [[ -z "$BASE_REF" ]]; then
  BASE_REF="$(git merge-base HEAD "${ENIR_GATE_BASE:-origin/main}")"
fi
SEED="${SOUC_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
KEEP_DIR="${E2_RECEIPT_DIR:-}"
TMP_DIR="$(mktemp -d /tmp/madaros-e2-enir-lowering.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

DRIVER="$TMP_DIR/madaros-enir"
ORACLE="$TMP_DIR/eisa-evm-oracle"
ORACLE_OUT="$TMP_DIR/eisa-evm-oracle.out"
RECEIPT="$TMP_DIR/e2-enir-lowering.receipt.json"

fail() {
  echo "E2_ENIR_LOWERING_GATE_FAIL: $*" >&2
  exit 1
}

. scripts/dev/madaros_v2_enir_gate_scope.sh

[[ -x "$SEED" ]] || fail "missing executable Stage0 seed: $SEED"
git rev-parse --verify "$BASE_REF" >/dev/null 2>&1 || fail "base ref not found: $BASE_REF"
E2_PROTECTED=(
  self-hosted/compiler/main.sio self-hosted/ir self-hosted/native self-hosted/wasm \
  self-hosted/gpu stdlib/runtime stdlib/eisa stdlib/math/dd64.sio \
)
madaros_v2_enir_gate_scope_or_skip "$BASE_REF" "E2_ENIR_LOWERING_GATE" \
  "E2 changed a compiler codegen/ABI/runtime or EVM-oracle surface" "${E2_PROTECTED[@]}"

scripts/dev/souc-build-lock.sh "$SEED" self-hosted/enir/driver.sio "$DRIVER" >"$TMP_DIR/driver-build.log" 2>&1
[[ -s "$DRIVER" ]] || fail "native ENIR driver build produced no ELF"
if grep -Eq '^error(\[E[0-9]+\])?:|unknown identifier|typecheck: failed|assignment type mismatch' "$TMP_DIR/driver-build.log"; then
  tail -80 "$TMP_DIR/driver-build.log" >&2
  fail "Stage0 reported diagnostics while building ENIR driver"
fi
chmod +x "$DRIVER"

scripts/dev/souc-build-lock.sh "$SEED" tools/eisa/eisa_evm_run.sio "$ORACLE" >"$TMP_DIR/oracle-build.log" 2>&1
[[ -s "$ORACLE" ]] || fail "source-fresh EVM oracle build produced no ELF"
if grep -Eq '^error(\[E[0-9]+\])?:|unknown identifier|typecheck: failed|assignment type mismatch' "$TMP_DIR/oracle-build.log"; then
  tail -80 "$TMP_DIR/oracle-build.log" >&2
  fail "Stage0 reported diagnostics while building EVM oracle"
fi
chmod +x "$ORACLE"
"$ORACLE" >"$ORACLE_OUT"
[[ "$(grep -c '^eisa-receipt:' "$ORACLE_OUT")" == "39" ]] || fail "EVM oracle did not emit exact 39-observation corpus"

python3 scripts/dev/madaros_v2_e2_enir_lowering_verify.py \
  --driver "$DRIVER" \
  --corpus tools/eisa/eisa_evm_run.sio \
  --oracle "$ORACLE_OUT" \
  --out-dir "$TMP_DIR/cases" \
  --receipt "$RECEIPT" \
  --root "$ROOT_DIR"

mkdir -p "$TMP_DIR/negative"
cat >"$TMP_DIR/negative/bad_header.eisa" <<'EOF'
fn bad_header() {
let x = 1.0
gate x
}
EOF
cat >"$TMP_DIR/negative/unsupported_control.eisa" <<'EOF'
epistemic fn unsupported_control() {
let x = 1.0
while x != 0.0 {
gate x
}
}
EOF
cat >"$TMP_DIR/negative/undefined_name.eisa" <<'EOF'
epistemic fn undefined_name() {
let x = y + 1.0
gate x
}
EOF
cat >"$TMP_DIR/negative/duplicate_name.eisa" <<'EOF'
epistemic fn duplicate_name() {
let x = 1.0
let x = 2.0
gate x
}
EOF
cat >"$TMP_DIR/negative/missing_close.eisa" <<'EOF'
epistemic fn missing_close() {
let x = 1.0
gate x
EOF
cat >"$TMP_DIR/negative/missing_gate.eisa" <<'EOF'
epistemic fn missing_gate() {
let x = 1.0
}
EOF
cat >"$TMP_DIR/negative/bad_store.eisa" <<'EOF'
epistemic fn bad_store() {
let x = 1.0
store dose <- x
gate x
}
EOF
python3 - "$TMP_DIR/negative/symbol_capacity.eisa" <<'PY'
from pathlib import Path
import sys
lines = ["epistemic fn symbol_capacity() {"]
lines += [f"let x{i} = {i}.0" for i in range(33)]
lines += ["gate x31", "}"]
Path(sys.argv[1]).write_text("\n".join(lines) + "\n", encoding="ascii")
PY
python3 - "$TMP_DIR/negative/slot_capacity.eisa" <<'PY'
from pathlib import Path
import sys
lines = ["epistemic fn slot_capacity() {", "let x = 1.0"]
lines += [f"store [s{i}] <- x" for i in range(17)]
lines += ["gate x", "}"]
Path(sys.argv[1]).write_text("\n".join(lines) + "\n", encoding="ascii")
PY

negative_count=0
for source in "$TMP_DIR"/negative/*.eisa; do
  negative_count=$((negative_count + 1))
  if "$DRIVER" lower "$source" >"$source.out" 2>&1; then
    fail "native source lowerer accepted negative fixture: $(basename "$source")"
  fi
  grep -Fq 'enir-lower-error|' "$source.out" || fail "negative source lacks classified lowering error: $(basename "$source")"
done
[[ "$negative_count" == "9" ]] || fail "negative source count drift: $negative_count"

cp "$TMP_DIR/cases/golden_mul.enir" "$TMP_DIR/bad-artifact.enir"
python3 - "$TMP_DIR/bad-artifact.enir" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
text = p.read_text(encoding="ascii")
text = text.replace("op|2|4|2|0|0|1|", "op|2|4|2|0|99|1|", 1)
p.write_text(text, encoding="ascii")
PY
if "$DRIVER" run "$TMP_DIR/bad-artifact.enir" >"$TMP_DIR/bad-artifact.log" 2>&1; then
  fail "interpreter accepted invalid SSA artifact tamper"
fi

cp "$TMP_DIR/cases/golden_mul.enir" "$TMP_DIR/noncanonical.enir"
python3 - "$TMP_DIR/noncanonical.enir" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
p.write_text(p.read_text(encoding="ascii").replace("type|0|4|", "type|00|4|", 1), encoding="ascii")
PY
if "$DRIVER" roundtrip "$TMP_DIR/noncanonical.enir" >/dev/null 2>&1; then
  fail "roundtrip accepted non-canonical integer spelling"
fi

E1_BASE_REF=HEAD bash scripts/dev/madaros_v2_e1_enir_shadow_gate.sh >/dev/null

if [[ -n "$KEEP_DIR" ]]; then
  mkdir -p "$KEEP_DIR"
  cp "$RECEIPT" "$KEEP_DIR/e2-enir-lowering.receipt.json"
  cp "$TMP_DIR"/cases/*.enir "$KEEP_DIR/"
fi

RECEIPT_SHA="$(sha256sum "$RECEIPT" | cut -d' ' -f1)"
echo "E2A_ENIR_V0_STRAIGHT_LINE_FULL_GATE_PASS receipt_sha256=$RECEIPT_SHA programs=5 observations=6 memory_events=3 source_negatives=9 artifact_tampers=2 native_independent=exact evm_observable=exact e1_regression=pass codegen_diff=0"
