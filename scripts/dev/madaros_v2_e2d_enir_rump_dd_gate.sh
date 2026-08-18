#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${E2D_BASE_REF:-}"
if [[ -z "$BASE_REF" ]]; then
  BASE_REF="$(git merge-base HEAD "${ENIR_GATE_BASE:-origin/main}")"
fi
SEED="${SOUC_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
KEEP_DIR="${E2D_RECEIPT_DIR:-}"
TMP_DIR="$(mktemp -d /tmp/madaros-e2d-enir-rump-dd.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

DRIVER="$TMP_DIR/madaros-enir"
ORACLE="$TMP_DIR/eisa-e2d-oracle"
ORACLE_OUT="$TMP_DIR/eisa-e2d-oracle.out"
SOURCE="tools/eisa/eisa_enir_v1_rump_dd.eisa"
CORPUS="tools/eisa/eisa_evm_run.sio"
RECEIPT="$TMP_DIR/e2d-enir-rump-dd.receipt.json"

fail() {
  echo "E2D_ENIR_V1_RUMP_DD_GATE_FAIL: $*" >&2
  exit 1
}

. scripts/dev/madaros_v2_enir_gate_scope.sh

[[ -x "$SEED" ]] || fail "missing executable Stage0 seed: $SEED"
git rev-parse --verify "$BASE_REF" >/dev/null 2>&1 || fail "base ref not found: $BASE_REF"
E2D_PROTECTED=(
  self-hosted/compiler/main.sio self-hosted/ir self-hosted/native self-hosted/wasm \
  self-hosted/gpu stdlib/runtime stdlib/eisa stdlib/math/dd64.sio "$CORPUS" \
)
madaros_v2_enir_gate_scope_or_skip "$BASE_REF" "E2D_ENIR_V1_RUMP_DD_GATE" \
  "E2D changed compiler codegen/ABI/runtime or the frozen METRON oracle" "${E2D_PROTECTED[@]}"

scripts/dev/souc-build-lock.sh "$SEED" self-hosted/enir/driver.sio "$DRIVER" >"$TMP_DIR/driver-build.log" 2>&1
[[ -s "$DRIVER" ]] || fail "native ENIR driver build produced no ELF"
if grep -Eq '^error:|unknown identifier|typecheck: failed|assignment type mismatch' "$TMP_DIR/driver-build.log"; then
  tail -100 "$TMP_DIR/driver-build.log" >&2
  fail "Stage0 reported diagnostics while building ENIR driver"
fi
chmod +x "$DRIVER"

scripts/dev/souc-build-lock.sh "$SEED" "$CORPUS" "$ORACLE" >"$TMP_DIR/oracle-build.log" 2>&1
[[ -s "$ORACLE" ]] || fail "source-fresh frozen METRON corpus build produced no ELF"
if grep -Eq '^error:|unknown identifier|typecheck: failed|assignment type mismatch' "$TMP_DIR/oracle-build.log"; then
  tail -100 "$TMP_DIR/oracle-build.log" >&2
  fail "Stage0 reported diagnostics while building METRON corpus"
fi
chmod +x "$ORACLE"
"$ORACLE" >"$ORACLE_OUT"
[[ "$(grep -c '^eisa-receipt:' "$ORACLE_OUT")" == "39" ]] || fail "METRON corpus observation manifest drifted"

python3 scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py \
  --driver "$DRIVER" \
  --source "$SOURCE" \
  --corpus "$CORPUS" \
  --oracle "$ORACLE_OUT" \
  --out-dir "$TMP_DIR/cases" \
  --receipt "$RECEIPT" \
  --root "$ROOT_DIR"

mkdir -p "$TMP_DIR/negative"
cat >"$TMP_DIR/negative/no_observation.eisa" <<'EOF'
epistemic fn no_observation() {
fuel 64
let x=1
}
EOF
cat >"$TMP_DIR/negative/duplicate_fuel.eisa" <<'EOF'
epistemic fn duplicate_fuel() {
fuel 64
fuel 64
let x=1
gate x
}
EOF
cat >"$TMP_DIR/negative/zero_fuel.eisa" <<'EOF'
epistemic fn zero_fuel() {
fuel 0
let x=1
gate x
}
EOF
cat >"$TMP_DIR/negative/fractional_fuel.eisa" <<'EOF'
epistemic fn fractional_fuel() {
fuel 64.5
let x=1
gate x
}
EOF
cat >"$TMP_DIR/negative/late_fuel.eisa" <<'EOF'
epistemic fn late_fuel() {
let x=1
fuel 64
gate x
}
EOF
cat >"$TMP_DIR/negative/fuel_out_of_range.eisa" <<'EOF'
epistemic fn fuel_out_of_range() {
fuel 1000001
let x=1
gate x
}
EOF
cat >"$TMP_DIR/negative/fuel_insufficient.eisa" <<'EOF'
epistemic fn fuel_insufficient() {
fuel 1
let x=1
gate x
}
EOF
cat >"$TMP_DIR/negative/undefined_gate.eisa" <<'EOF'
epistemic fn undefined_gate() {
fuel 64
gate missing
}
EOF
cat >"$TMP_DIR/negative/duplicate_symbol.eisa" <<'EOF'
epistemic fn duplicate_symbol() {
fuel 64
let x=1
let x=2
gate x
}
EOF
cat >"$TMP_DIR/negative/nested_expression.eisa" <<'EOF'
epistemic fn nested_expression() {
fuel 64
let x=1+2+3
gate x
}
EOF
cat >"$TMP_DIR/negative/content_after_close.eisa" <<'EOF'
epistemic fn content_after_close() {
fuel 64
let x=1
gate x
}
gate x
EOF
cat >"$TMP_DIR/negative/missing_close.eisa" <<'EOF'
epistemic fn missing_close() {
fuel 64
let x=1
gate x
EOF
cat >"$TMP_DIR/negative/set_outside_loop.eisa" <<'EOF'
epistemic fn set_outside_loop() {
fuel 64
let x=1
set x=2
gate x
}
EOF
cat >"$TMP_DIR/negative/unsupported_if.eisa" <<'EOF'
epistemic fn unsupported_if() {
fuel 64
let x=1
if x < 0.0 {
}
gate x
}
EOF
cat >"$TMP_DIR/negative/load_before_store.eisa" <<'EOF'
epistemic fn load_before_store() {
fuel 64
let x=load [m]
gate x
}
EOF
cat >"$TMP_DIR/negative/malformed_header.eisa" <<'EOF'
fn malformed_header() {
fuel 64
let x=1
gate x
}
EOF
python3 - "$TMP_DIR/negative/symbol_capacity.eisa" "$TMP_DIR/negative/gate_capacity.eisa" <<'PY'
from pathlib import Path
import sys
symbols = ["epistemic fn symbol_capacity() {", "fuel 1000"]
symbols += [f"let x{i}={i}" for i in range(65)]
symbols += ["gate x63", "}"]
Path(sys.argv[1]).write_text("\n".join(symbols) + "\n", encoding="ascii")
gates = ["epistemic fn gate_capacity() {", "fuel 1000", "let x=1"]
gates += ["gate x"] * 65
gates += ["}"]
Path(sys.argv[2]).write_text("\n".join(gates) + "\n", encoding="ascii")
PY

source_negative_count=0
for source in "$TMP_DIR"/negative/*.eisa; do
  source_negative_count=$((source_negative_count + 1))
  if "$DRIVER" lower-v1 "$source" >"$source.out" 2>&1; then
    fail "source lowerer accepted E2D negative: $(basename "$source")"
  fi
  grep -Fq 'enir-lower-error|' "$source.out" || fail "negative lacks classified lowering error: $(basename "$source")"
done
[[ "$source_negative_count" == "18" ]] || fail "source negative count drifted: $source_negative_count"

python3 - "$TMP_DIR/cases/v1_rump_dd.enir" "$TMP_DIR/tampers" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="ascii")
out = Path(sys.argv[2])
out.mkdir()

def write(name, text):
    (out / f"{name}.enir").write_text(text, encoding="ascii")

write("fuel_zero", source.replace("resource|64\n", "resource|0\n", 1))
write("fuel_insufficient", source.replace("resource|64\n", "resource|29\n", 1))
write("halt_cost_zero", source.replace("block|0|0|29|0|0|0|-1|-1|-1|1", "block|0|0|29|0|0|0|-1|-1|-1|0", 1))
write("block_op_count", source.replace("block|0|0|29|", "block|0|0|28|", 1))
write("footer_edge_count", source.replace("end2|1|26|26|1|1|0|0|29|3|1", "end2|1|26|26|1|1|0|1|29|3|1", 1))
write("footer_observation_count", source.replace("end2|1|26|26|1|1|0|0|29|3|1", "end2|1|26|26|1|1|0|0|29|2|1", 1))
write("observation_ordinal", source.replace("obs|1|v1_rump_dd|1|0", "obs|1|v1_rump_dd|0|0", 1))
write("observation_kind", source.replace("obs|2|v1_rump_dd|2|0", "obs|2|v1_rump_dd|2|1", 1))
write("future_use", source.replace("op|2|4|2|0|1|1|", "op|2|4|2|0|25|1|", 1))
write("duplicate_result", source.replace("op|3|4|3|0|2|2|", "op|3|4|2|0|2|2|", 1))
write("provenance_transform", source.replace("prov|2|5|-1|4|-1|-1", "prov|2|5|-1|3|-1|-1", 1))
write("unknown_constant", source.replace("value|0|0|1|4680070212836392960|", "value|0|0|0|4680070212836392960|", 1))
write("gate_policy", source.replace("op|26|7|-1|-1|24|-1|-1|0|-1|1", "op|26|7|-1|-1|24|-1|-1|-1|-1|1", 1))
write("unsupported_eload", source.replace("op|2|4|2|0|1|1|-1|-1|-1|1", "op|2|1|2|0|-1|-1|-1|-1|0|1", 1))
write("noncanonical_integer", source.replace("resource|64\n", "resource|064\n", 1))
write("crlf", source.replace("\n", "\r\n"))
PY

artifact_tamper_count=0
for artifact in "$TMP_DIR"/tampers/*.enir; do
  artifact_tamper_count=$((artifact_tamper_count + 1))
  if "$DRIVER" run "$artifact" >"$artifact.out" 2>&1; then
    fail "interpreter accepted E2D artifact tamper: $(basename "$artifact")"
  fi
done
[[ "$artifact_tamper_count" == "16" ]] || fail "artifact tamper count drifted: $artifact_tamper_count"

sed 's/let five5=5\.5/let five5=5.25/' "$SOURCE" >"$TMP_DIR/source-tamper.eisa"
"$DRIVER" lower-v1 "$TMP_DIR/source-tamper.eisa" >"$TMP_DIR/source-tamper.enir"
"$DRIVER" run "$TMP_DIR/cases/v1_rump_dd.enir" | grep '^enir-exec|' >"$TMP_DIR/source-original.receipts"
"$DRIVER" run "$TMP_DIR/source-tamper.enir" | grep '^enir-exec|' >"$TMP_DIR/source-tamper.receipts"
if cmp -s "$TMP_DIR/source-original.receipts" "$TMP_DIR/source-tamper.receipts"; then
  fail "causal source tamper did not change Rump observations"
fi

E2C_BASE_REF=HEAD bash scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_gate.sh >"$TMP_DIR/e2c-regression.log"
grep -Fq 'E2C_ENIR_FUEL_BLOCKARGS_FULL_GATE_PASS' "$TMP_DIR/e2c-regression.log" || fail "E2C/E2B/E2A/E1 regression chain failed"

if [[ -n "$KEEP_DIR" ]]; then
  mkdir -p "$KEEP_DIR"
  cp "$RECEIPT" "$KEEP_DIR/e2d-enir-rump-dd.receipt.json"
  cp "$TMP_DIR/cases/v1_rump_dd.enir" "$KEEP_DIR/"
fi

RECEIPT_SHA="$(sha256sum "$RECEIPT" | cut -d' ' -f1)"
echo "E2D_ENIR_V1_RUMP_DD_FULL_GATE_PASS receipt_sha256=$RECEIPT_SHA programs=1 observations=3 cumulative=17/30,23/39 ops=29 semantic_instructions=30 values=26 fuel=64->34 source_negatives=$source_negative_count artifact_tampers=$artifact_tamper_count causal_source_tamper=pass graph=source==frozen-image dd64_words=independent metron_receipts=exact e2c_regression=pass e2b_regression=pass e2a_regression=pass e1_regression=pass codegen_diff=0"
