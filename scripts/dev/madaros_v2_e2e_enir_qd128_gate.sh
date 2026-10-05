#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${E2E_BASE_REF:-}"
if [[ -z "$BASE_REF" ]]; then
  BASE_REF="$(git merge-base HEAD "${ENIR_GATE_BASE:-origin/main}")"
fi
SEED="${SOUC_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
KEEP_DIR="${E2E_RECEIPT_DIR:-}"
TMP_DIR="$(mktemp -d /tmp/madaros-e2e-enir-qd128.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

DRIVER="$TMP_DIR/madaros-enir"
ORACLE="$TMP_DIR/eisa-e2e-oracle"
ORACLE_OUT="$TMP_DIR/eisa-e2e-oracle.out"
CORPUS="tools/eisa/eisa_evm_run.sio"
RECEIPT="$TMP_DIR/e2e-enir-qd128.receipt.json"

fail() {
  echo "E2E_ENIR_QD128_GATE_FAIL: $*" >&2
  exit 1
}

. scripts/dev/madaros_v2_enir_gate_scope.sh

[[ -x "$SEED" ]] || fail "missing executable Stage0 seed: $SEED"
git rev-parse --verify "$BASE_REF" >/dev/null 2>&1 || fail "base ref not found: $BASE_REF"
E2E_PROTECTED=(
  self-hosted/compiler/main.sio self-hosted/ir self-hosted/native self-hosted/wasm \
  self-hosted/gpu stdlib/runtime stdlib/eisa stdlib/math/dd64.sio stdlib/math/qd128.sio "$CORPUS" \
)
madaros_v2_enir_gate_scope_or_skip "$BASE_REF" "E2E_ENIR_QD128_GATE" \
  "E2E changed compiler codegen/ABI/runtime, shared qd128, or the frozen METRON oracle" "${E2E_PROTECTED[@]}"

scripts/dev/souc-build-lock.sh "$SEED" self-hosted/enir/driver.sio "$DRIVER" >"$TMP_DIR/driver-build.log" 2>&1
[[ -s "$DRIVER" ]] || fail "native ENIR driver build produced no ELF"
if grep -Eq '^error(\[E[0-9]+\])?:|unknown identifier|typecheck: failed|assignment type mismatch|private struct field' "$TMP_DIR/driver-build.log"; then
  tail -100 "$TMP_DIR/driver-build.log" >&2
  fail "Stage0 reported diagnostics while building ENIR driver"
fi
chmod +x "$DRIVER"

scripts/dev/souc-build-lock.sh "$SEED" "$CORPUS" "$ORACLE" >"$TMP_DIR/oracle-build.log" 2>&1
[[ -s "$ORACLE" ]] || fail "source-fresh METRON corpus build produced no ELF"
if grep -Eq '^error(\[E[0-9]+\])?:|unknown identifier|typecheck: failed|assignment type mismatch' "$TMP_DIR/oracle-build.log"; then
  tail -100 "$TMP_DIR/oracle-build.log" >&2
  fail "Stage0 reported diagnostics while building METRON corpus"
fi
chmod +x "$ORACLE"
"$ORACLE" >"$ORACLE_OUT"
[[ "$(grep -c '^eisa-receipt:' "$ORACLE_OUT")" == "39" ]] || fail "METRON observation manifest drifted"

verify=(python3 scripts/dev/madaros_v2_e2e_enir_qd128_verify.py
  --driver "$DRIVER"
  --source-dir tools/eisa
  --corpus "$CORPUS"
  --oracle "$ORACLE_OUT"
  --out-dir "$TMP_DIR/cases"
  --receipt "$RECEIPT"
  --root "$ROOT_DIR")
"${verify[@]}"

mkdir -p "$TMP_DIR/negative"
python3 - "$TMP_DIR/negative" <<'PY'
from pathlib import Path
import sys

out = Path(sys.argv[1])
cases = {
    "no_fuel": "epistemic fn no_fuel() {\nlet x=1\ngate x\n}\n",
    "zero_fuel": "epistemic fn zero_fuel() {\nfuel 0\nlet x=1\ngate x\n}\n",
    "insufficient_fuel": "epistemic fn insufficient_fuel() {\nfuel 2\nlet x=1\ngate x\n}\n",
    "duplicate_fuel": "epistemic fn duplicate_fuel() {\nfuel 12\nfuel 12\nlet x=1\ngate x\n}\n",
    "late_fuel": "epistemic fn late_fuel() {\nlet x=1\nfuel 12\ngate x\n}\n",
    "fractional_fuel": "epistemic fn fractional_fuel() {\nfuel 12.5\nlet x=1\ngate x\n}\n",
    "fuel_range": "epistemic fn fuel_range() {\nfuel 1000001\nlet x=1\ngate x\n}\n",
    "no_gate": "epistemic fn no_gate() {\nfuel 12\nlet x=1\n}\n",
    "undefined_gate": "epistemic fn undefined_gate() {\nfuel 12\ngate x\n}\n",
    "duplicate_symbol": "epistemic fn duplicate_symbol() {\nfuel 12\nlet x=1\nlet x=2\ngate x\n}\n",
    "nested_expression": "epistemic fn nested_expression() {\nfuel 12\nlet x=1+2+3\ngate x\n}\n",
    "nested_while_control": "epistemic fn nested_while_control() {\nfuel 12\nlet x=1\nwhile x != 0.0 {\nwhile x != 0.0 {\n}\n}\ngate x\n}\n",
    "load_before_store": "epistemic fn load_before_store() {\nfuel 12\nlet x=load [m]\ngate x\n}\n",
    "malformed_decimal": "epistemic fn malformed_decimal() {\nfuel 12\nlet x=0.\ngate x\n}\n",
    "non_finite": "epistemic fn non_finite() {\nfuel 12\nlet x=999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999\ngate x\n}\n",
    "after_close": "epistemic fn after_close() {\nfuel 12\nlet x=1\ngate x\n}\ngate x\n",
}
for name, text in cases.items():
    (out / f"{name}.eisa").write_text(text, encoding="ascii")
PY

source_negative_count=0
for source in "$TMP_DIR"/negative/*.eisa; do
  source_negative_count=$((source_negative_count + 1))
  if "$DRIVER" lower-v2 "$source" >"$source.out" 2>&1; then
    fail "source lowerer accepted E2E negative: $(basename "$source")"
  fi
  grep -Fq 'enir-lower-error|' "$source.out" || fail "negative lacks classified lowering error: $(basename "$source")"
done
[[ "$source_negative_count" == "16" ]] || fail "source negative count drifted: $source_negative_count"

cat >"$TMP_DIR/div-zero.eisa" <<'EOF'
epistemic fn div_zero() {
    fuel 12
    let x = 1.0
    let y = 0.0
    let z = x / y
    gate z
}
EOF
"$DRIVER" lower-v2 "$TMP_DIR/div-zero.eisa" >"$TMP_DIR/div-zero.enir"
"$DRIVER" run "$TMP_DIR/div-zero.enir" >"$TMP_DIR/div-zero.out"
grep -Eq '^enir-exec\|.*error0_bits=0\|error1_bits=0\|error2_bits=0\|error3_bits=0\|uncertainty_bits=[0-9-]+\|status=1\|gate_class=2\|' "$TMP_DIR/div-zero.out" \
  || fail "v2 divide-by-zero did not fail closed as a poisoned all-word observation"

python3 - "$TMP_DIR/cases/v2_div.enir" "$TMP_DIR/tampers" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="ascii")
out = Path(sys.argv[2]); out.mkdir()

def write(name, text):
    (out / f"{name}.enir").write_text(text, encoding="ascii")

write("profile", source.replace("enir|2|2|v2_div|2", "enir|2|2|v2_div|1", 1))
write("type_error_kind", source.replace("type|0|4|2|1|1|1|2", "type|0|4|1|1|1|1|2", 1))
write("type_value_kind", source.replace("type|0|4|2|1|1|1|2", "type|0|3|2|1|1|1|2", 1))
for index, field in enumerate(("error0", "error1", "error2", "error3"), 6):
    row = source.splitlines()[3]
    parts = row.split("|")
    parts[index] = "1"
    write(field, source.replace(row, "|".join(parts), 1))
row = source.splitlines()[3]; parts = row.split("|"); parts[10] = "1"
write("uncertainty", source.replace(row, "|".join(parts), 1))
write("fuel_zero", source.replace("resource|12", "resource|0", 1))
write("fuel_insufficient", source.replace("resource|12", "resource|4", 1))
write("halt_cost", source.replace("block|0|0|4|0|0|0|-1|-1|-1|1", "block|0|0|4|0|0|0|-1|-1|-1|0", 1))
write("future_use", source.replace("op|2|5|2|0|0|1|", "op|2|5|2|0|2|1|", 1))
write("provenance", source.replace("prov|2|5|-1|5|-1|-1", "prov|2|5|-1|4|-1|-1", 1))
write("observation", source.replace("obs|0|v2_div|0|0", "obs|0|v2_div|1|0", 1))
write("footer", source.replace("end2|1|3|3|1|1|0|0|4|1|1", "end2|1|3|3|1|1|0|0|4|2|1", 1))
write("noncanonical", source.replace("resource|12", "resource|012", 1))
write("crlf", source.replace("\n", "\r\n"))
PY

artifact_tamper_count=0
for artifact in "$TMP_DIR"/tampers/*.enir; do
  artifact_tamper_count=$((artifact_tamper_count + 1))
  if "$DRIVER" run "$artifact" >"$artifact.out" 2>&1; then
    fail "interpreter accepted E2E artifact tamper: $(basename "$artifact")"
  fi
done
[[ "$artifact_tamper_count" == "17" ]] || fail "artifact tamper count drifted: $artifact_tamper_count"

receipt_tamper_count=0
for field in error0_bits error1_bits error2_bits error3_bits; do
  wrapper="$TMP_DIR/tamper-$field"
  cat >"$wrapper" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" == "run" ]]; then
  "$DRIVER" "\$@" | sed "0,/|$field=[^|]*/s//|$field=1/"
else
  exec "$DRIVER" "\$@"
fi
EOF
  chmod +x "$wrapper"
  receipt_tamper_count=$((receipt_tamper_count + 1))
  if python3 scripts/dev/madaros_v2_e2e_enir_qd128_verify.py \
      --driver "$wrapper" --source-dir tools/eisa --corpus "$CORPUS" --oracle "$ORACLE_OUT" \
      --out-dir "$TMP_DIR/tampered-$field" --receipt "$TMP_DIR/tampered-$field.json" --root "$ROOT_DIR" \
      >"$TMP_DIR/tampered-$field.log" 2>&1; then
    fail "independent verifier accepted runtime $field receipt tamper"
  fi
done
[[ "$receipt_tamper_count" == "4" ]] || fail "runtime receipt tamper count drifted"

sed 's/let y = 0\.2/let y = 0.25/' tools/eisa/eisa_enir_v2_add.eisa >"$TMP_DIR/source-tamper.eisa"
"$DRIVER" lower-v2 "$TMP_DIR/source-tamper.eisa" >"$TMP_DIR/source-tamper.enir"
"$DRIVER" run "$TMP_DIR/cases/v2_add.enir" | grep '^enir-exec|' >"$TMP_DIR/source-original.receipt"
"$DRIVER" run "$TMP_DIR/source-tamper.enir" | grep '^enir-exec|' >"$TMP_DIR/source-tamper.receipt"
cmp -s "$TMP_DIR/source-original.receipt" "$TMP_DIR/source-tamper.receipt" \
  && fail "causal source tamper did not change qd128 observation"

E2D_BASE_REF=HEAD bash scripts/dev/madaros_v2_e2d_enir_rump_dd_gate.sh >"$TMP_DIR/e2d-regression.log"
grep -Fq 'E2D_ENIR_V1_RUMP_DD_FULL_GATE_PASS' "$TMP_DIR/e2d-regression.log" \
  || fail "E2D/E2C/E2B/E2A/E1 regression chain failed"

if [[ -n "$KEEP_DIR" ]]; then
  mkdir -p "$KEEP_DIR"
  cp "$RECEIPT" "$KEEP_DIR/e2e-enir-qd128.receipt.json"
  cp "$TMP_DIR"/cases/v2_*.enir "$KEEP_DIR/"
fi

RECEIPT_SHA="$(sha256sum "$RECEIPT" | cut -d' ' -f1)"
echo "E2E_ENIR_QD128_ARITHMETIC_FULL_GATE_PASS receipt_sha256=$RECEIPT_SHA programs=6 observations=6 cumulative=23/30,29/39 ops=21 semantic_instructions=27 values=15 fuel=72->45 source_negatives=$source_negative_count artifact_tampers=$artifact_tamper_count runtime_word_tampers=$receipt_tamper_count divide_zero=poison causal_source_tamper=pass graph=source==frozen-image qd128_words=independent high_precision=pass metron_receipts=exact e2d_regression=pass e2c_regression=pass e2b_regression=pass e2a_regression=pass e1_regression=pass codegen_diff=0"
