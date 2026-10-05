#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${E2C_BASE_REF:-}"
if [[ -z "$BASE_REF" ]]; then
  BASE_REF="$(git merge-base HEAD "${ENIR_GATE_BASE:-origin/main}")"
fi
SEED="${SOUC_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
KEEP_DIR="${E2C_RECEIPT_DIR:-}"
TMP_DIR="$(mktemp -d /tmp/madaros-e2c-enir-fuel-blockargs.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

DRIVER="$TMP_DIR/madaros-enir"
ORACLE="$TMP_DIR/eisa-v1-loop-oracle"
ORACLE_OUT="$TMP_DIR/eisa-v1-loop-oracle.out"
RECEIPT="$TMP_DIR/e2c-enir-fuel-blockargs.receipt.json"

fail() {
  echo "E2C_ENIR_FUEL_BLOCKARGS_GATE_FAIL: $*" >&2
  exit 1
}

. scripts/dev/madaros_v2_enir_gate_scope.sh

[[ -x "$SEED" ]] || fail "missing executable Stage0 seed: $SEED"
git rev-parse --verify "$BASE_REF" >/dev/null 2>&1 || fail "base ref not found: $BASE_REF"
E2C_PROTECTED=(
  self-hosted/compiler/main.sio self-hosted/ir self-hosted/native self-hosted/wasm \
  self-hosted/gpu stdlib/runtime stdlib/eisa stdlib/math/dd64.sio \
)
madaros_v2_enir_gate_scope_or_skip "$BASE_REF" "E2C_ENIR_FUEL_BLOCKARGS_GATE" \
  "E2C changed compiler codegen/ABI/runtime or canonical EISA implementation" "${E2C_PROTECTED[@]}"

if [[ -n "${E2C_PREBUILT_DRIVER:-}" ]]; then
  cp "$E2C_PREBUILT_DRIVER" "$DRIVER"
  : >"$TMP_DIR/driver-build.log"
else
  scripts/dev/souc-build-lock.sh "$SEED" self-hosted/enir/driver.sio "$DRIVER" >"$TMP_DIR/driver-build.log" 2>&1
fi
[[ -s "$DRIVER" ]] || fail "native ENIR driver build produced no ELF"
if grep -Eq '^error(\[E[0-9]+\])?:|unknown identifier|typecheck: failed|assignment type mismatch' "$TMP_DIR/driver-build.log"; then
  tail -100 "$TMP_DIR/driver-build.log" >&2
  fail "Stage0 reported diagnostics while building ENIR driver"
fi
chmod +x "$DRIVER"

if [[ -n "${E2C_PREBUILT_ORACLE:-}" ]]; then
  cp "$E2C_PREBUILT_ORACLE" "$ORACLE"
  : >"$TMP_DIR/oracle-build.log"
else
  scripts/dev/souc-build-lock.sh "$SEED" tools/eisa/eisa_enir_v1_loop_oracle.sio "$ORACLE" >"$TMP_DIR/oracle-build.log" 2>&1
fi
[[ -s "$ORACLE" ]] || fail "source-fresh EISA/METRON oracle build produced no ELF"
if grep -Eq '^error(\[E[0-9]+\])?:|unknown identifier|typecheck: failed|assignment type mismatch' "$TMP_DIR/oracle-build.log"; then
  tail -100 "$TMP_DIR/oracle-build.log" >&2
  fail "Stage0 reported diagnostics while building EISA/METRON oracle"
fi
chmod +x "$ORACLE"
"$ORACLE" >"$ORACLE_OUT"
grep -Fq 'E2C_EISA_V1_LOOP_ORACLE_PASS' "$ORACLE_OUT" || fail "EISA/METRON oracle did not finish cleanly"
[[ "$(grep -c '^e2c-case-begin|' "$ORACLE_OUT")" == "3" ]] || fail "oracle program manifest drifted"
[[ "$(grep -c '^eisa-receipt:' "$ORACLE_OUT")" == "3" ]] || fail "oracle observation manifest drifted"

python3 scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_verify.py \
  --driver "$DRIVER" \
  --corpus tools/eisa/eisa_enir_v1_loop_oracle.sio \
  --oracle "$ORACLE_OUT" \
  --out-dir "$TMP_DIR/cases" \
  --receipt "$RECEIPT" \
  --root "$ROOT_DIR"

mkdir -p "$TMP_DIR/negative"
cat >"$TMP_DIR/negative/duplicate_fuel.eisa" <<'EOF'
epistemic fn duplicate_fuel() {
fuel 5
fuel 6
let x=1
while x != 0.0 {
}
}
EOF
cat >"$TMP_DIR/negative/zero_fuel.eisa" <<'EOF'
epistemic fn zero_fuel() {
fuel 0
let x=1
while x != 0.0 {
}
}
EOF
cat >"$TMP_DIR/negative/fractional_fuel.eisa" <<'EOF'
epistemic fn fractional_fuel() {
fuel 5.5
let x=1
while x != 0.0 {
}
}
EOF
cat >"$TMP_DIR/negative/late_fuel.eisa" <<'EOF'
epistemic fn late_fuel() {
let x=1
fuel 5
while x != 0.0 {
}
}
EOF
cat >"$TMP_DIR/negative/no_observation.eisa" <<'EOF'
epistemic fn no_observation() {
fuel 5
let x=1
}
EOF
cat >"$TMP_DIR/negative/nested_loop.eisa" <<'EOF'
epistemic fn nested_loop() {
fuel 10
let x=1
while x != 0.0 {
while x != 0.0 {
}
}
}
EOF
cat >"$TMP_DIR/negative/second_loop.eisa" <<'EOF'
epistemic fn second_loop() {
fuel 10
let x=1
while x != 0.0 {
}
while x != 0.0 {
}
}
EOF
cat >"$TMP_DIR/negative/set_undefined.eisa" <<'EOF'
epistemic fn set_undefined() {
fuel 10
let x=1
while x != 0.0 {
set y=0
}
}
EOF
cat >"$TMP_DIR/negative/set_local.eisa" <<'EOF'
epistemic fn set_local() {
fuel 10
let x=1
while x != 0.0 {
let y=1
set y=0
}
}
EOF
cat >"$TMP_DIR/negative/gate_inside.eisa" <<'EOF'
epistemic fn gate_inside() {
fuel 10
let x=1
while x != 0.0 {
gate x
}
}
EOF
cat >"$TMP_DIR/negative/two_gates.eisa" <<'EOF'
epistemic fn two_gates() {
fuel 20
let x=0
while x != 0.0 {
}
gate x
gate x
}
EOF
cat >"$TMP_DIR/negative/high_body.eisa" <<'EOF'
epistemic fn high_body() {
fuel 20
let a=0
let b=0
let c=0
let d=0
let e=1
while e != 0.0 {
let x=1
}
}
EOF
cat >"$TMP_DIR/negative/unsupported_if.eisa" <<'EOF'
epistemic fn unsupported_if() {
fuel 20
let x=1
while x != 0.0 {
if x < 0.0 {
}
}
}
EOF
cat >"$TMP_DIR/negative/unsupported_store.eisa" <<'EOF'
epistemic fn unsupported_store() {
fuel 20
let x=1
while x != 0.0 {
}
store [m] <- x
}
EOF
cat >"$TMP_DIR/negative/missing_root_close.eisa" <<'EOF'
epistemic fn missing_root_close() {
fuel 5
let x=1
while x != 0.0 {
}
EOF
cat >"$TMP_DIR/negative/malformed_while.eisa" <<'EOF'
epistemic fn malformed_while() {
fuel 5
let x=1
while x > 0.0 {
}
}
EOF

source_negative_count=0
for source in "$TMP_DIR"/negative/*.eisa; do
  source_negative_count=$((source_negative_count + 1))
  if "$DRIVER" lower-v1 "$source" >"$source.out" 2>&1; then
    fail "source lowerer accepted E2C negative: $(basename "$source")"
  fi
  grep -Fq 'enir-lower-error|' "$source.out" || fail "negative lacks classified lowering error: $(basename "$source")"
done
[[ "$source_negative_count" == "16" ]] || fail "source negative count drifted: $source_negative_count"

python3 - "$TMP_DIR/cases/v1e_fixedpoint.enir" "$TMP_DIR/cases/v1_fuel.enir" "$TMP_DIR/tampers" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="ascii")
fuel_source = Path(sys.argv[2]).read_text(encoding="ascii")
out = Path(sys.argv[3])
out.mkdir()

def write(name, text):
    (out / f"{name}.enir").write_text(text, encoding="ascii")

write("fuel_zero", source.replace("resource|100\n", "resource|0\n", 1))
write("edge_arity", source.replace("edge|0|0|1|3|", "edge|0|0|1|2|", 1))
write("edge_arg_oob", source.replace("edge|3|2|1|3|16|5|14|-1", "edge|3|2|1|3|99|5|14|-1", 1))
write("barg_ordinal", source.replace("barg|1|1|1|", "barg|1|1|0|", 1))
write("arg_range_overlap", source.replace("block|3|11|1|3|3|0|", "block|3|11|1|0|3|0|", 1))
write("wrong_predecessor", source.replace("edge|3|2|1|", "edge|3|3|1|", 1))
write("term_edge_owner", source.replace("block|2|3|8|3|0|1|-1|3|-1|1", "block|2|3|8|3|0|1|-1|0|-1|1", 1))
write("term_fuel", source.replace("block|1|3|0|0|3|2|7|1|2|1", "block|1|3|0|0|3|2|7|1|2|0", 1))
write("block_overlap", source.replace("block|2|3|8|", "block|2|2|9|", 1))
write("dominance", source.replace("op|3|5|9|0|5|3|", "op|3|5|9|0|5|4|", 1))
write("unsupported_eload", source.replace("op|3|5|9|0|5|3|-1|-1|-1|1", "op|3|1|9|0|-1|-1|-1|-1|0|1", 1))
write("fuel_zero_condition", fuel_source.replace("value|0|0|1|4607182418800017408|3|", "value|0|0|1|0|0|", 1))
write("footer_count", source.replace("end2|1|17|17|1|4|6|4|12|1|1", "end2|1|17|17|1|4|5|4|12|1|1", 1))
write("noncanonical_integer", source.replace("resource|100\n", "resource|0100\n", 1))
write("duplicate_resource", source.replace("resource|100\n", "resource|100\nresource|100\n", 1))
write("crlf", source.replace("\n", "\r\n"))
PY

artifact_tamper_count=0
for artifact in "$TMP_DIR"/tampers/*.enir; do
  artifact_tamper_count=$((artifact_tamper_count + 1))
  if "$DRIVER" verify "$artifact" >"$artifact.out" 2>&1; then
    fail "verifier accepted E2C artifact tamper: $(basename "$artifact")"
  fi
done
[[ "$artifact_tamper_count" == "16" ]] || fail "artifact tamper count drifted: $artifact_tamper_count"

E2B_BASE_REF=HEAD bash scripts/dev/madaros_v2_e2b_enir_cfg_gate.sh >"$TMP_DIR/e2b-regression.log"
grep -Fq 'E2B_ENIR_V1_FINITE_CFG_FULL_GATE_PASS' "$TMP_DIR/e2b-regression.log" || fail "E2B/E2A/E1 regression chain failed"

if [[ -n "$KEEP_DIR" ]]; then
  mkdir -p "$KEEP_DIR"
  cp "$RECEIPT" "$KEEP_DIR/e2c-enir-fuel-blockargs.receipt.json"
  cp "$TMP_DIR"/cases/*.enir "$KEEP_DIR/"
fi

RECEIPT_SHA="$(sha256sum "$RECEIPT" | cut -d' ' -f1)"
echo "E2C_ENIR_FUEL_BLOCKARGS_FULL_GATE_PASS receipt_sha256=$RECEIPT_SHA programs=3 observations=3 blocks=12 block_args=10 edges=12 source_negatives=$source_negative_count artifact_tampers=$artifact_tamper_count fuel=5,100,25 outcomes=stop,gate,stop fixedpoint_ops=15 high_last_value=20 native_independent=exact evm_observable=exact e2b_regression=pass e2a_regression=pass e1_regression=pass codegen_diff=0"
