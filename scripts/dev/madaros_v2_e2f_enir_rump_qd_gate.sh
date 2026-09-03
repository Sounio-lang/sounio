#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${E2F_BASE_REF:-}"
if [[ -z "$BASE_REF" ]]; then
  BASE_REF="$(git merge-base HEAD "${ENIR_GATE_BASE:-origin/main}")"
fi
SEED="${SOUC_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
KEEP_DIR="${E2F_RECEIPT_DIR:-}"
TMP_DIR="$(mktemp -d /tmp/madaros-e2f-enir-rump-qd.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

DRIVER="$TMP_DIR/madaros-enir"
ORACLE="$TMP_DIR/eisa-e2f-oracle"
ORACLE_OUT="$TMP_DIR/eisa-e2f-oracle.out"
SOURCE="tools/eisa/eisa_enir_v2_rump_qd.eisa"
CORPUS="tools/eisa/eisa_evm_run.sio"
RECEIPT="$TMP_DIR/e2f-enir-rump-qd.receipt.json"

fail() {
  echo "E2F_ENIR_V2_RUMP_QD_GATE_FAIL: $*" >&2
  exit 1
}

. scripts/dev/madaros_v2_enir_gate_scope.sh

[[ -x "$SEED" ]] || fail "missing executable Stage0 seed: $SEED"
git rev-parse --verify "$BASE_REF" >/dev/null 2>&1 || fail "base ref not found: $BASE_REF"
E2F_PROTECTED=(
  self-hosted/compiler/main.sio self-hosted/ir self-hosted/native self-hosted/wasm \
  self-hosted/gpu stdlib/runtime stdlib/eisa stdlib/math/dd64.sio stdlib/math/qd128.sio \
  self-hosted/enir/qd.sio "$CORPUS" \
)
madaros_v2_enir_gate_scope_or_skip "$BASE_REF" "E2F_ENIR_V2_RUMP_QD_GATE" \
  "E2F changed compiler codegen/ABI/runtime, pinned qd semantics, or frozen METRON oracle" "${E2F_PROTECTED[@]}"

scripts/dev/souc-build-lock.sh "$SEED" self-hosted/enir/driver.sio "$DRIVER" >"$TMP_DIR/driver-build.log" 2>&1
[[ -s "$DRIVER" ]] || fail "native ENIR driver build produced no ELF"
if grep -Eq '^error:|unknown identifier|typecheck: failed|assignment type mismatch|private struct field' "$TMP_DIR/driver-build.log"; then
  tail -100 "$TMP_DIR/driver-build.log" >&2
  fail "Stage0 reported diagnostics while building ENIR driver"
fi
chmod +x "$DRIVER"

scripts/dev/souc-build-lock.sh "$SEED" "$CORPUS" "$ORACLE" >"$TMP_DIR/oracle-build.log" 2>&1
[[ -s "$ORACLE" ]] || fail "source-fresh METRON corpus build produced no ELF"
if grep -Eq '^error:|unknown identifier|typecheck: failed|assignment type mismatch' "$TMP_DIR/oracle-build.log"; then
  tail -100 "$TMP_DIR/oracle-build.log" >&2
  fail "Stage0 reported diagnostics while building METRON corpus"
fi
chmod +x "$ORACLE"
"$ORACLE" >"$ORACLE_OUT"
[[ "$(grep -c '^eisa-receipt:' "$ORACLE_OUT")" == "39" ]] || fail "METRON observation manifest drifted"

python3 scripts/dev/madaros_v2_e2f_enir_rump_qd_verify.py \
  --driver "$DRIVER" --source "$SOURCE" --corpus "$CORPUS" --oracle "$ORACLE_OUT" \
  --out-dir "$TMP_DIR/cases" --receipt "$RECEIPT" --root "$ROOT_DIR"

mkdir -p "$TMP_DIR/negative"
python3 - "$TMP_DIR/negative" <<'PY'
from pathlib import Path
import sys

out = Path(sys.argv[1])
cases = {
    "no_observation": "epistemic fn no_observation() {\nfuel 64\nlet x=1\n}\n",
    "duplicate_fuel": "epistemic fn duplicate_fuel() {\nfuel 64\nfuel 64\nlet x=1\ngate x\n}\n",
    "zero_fuel": "epistemic fn zero_fuel() {\nfuel 0\nlet x=1\ngate x\n}\n",
    "fractional_fuel": "epistemic fn fractional_fuel() {\nfuel 64.5\nlet x=1\ngate x\n}\n",
    "late_fuel": "epistemic fn late_fuel() {\nlet x=1\nfuel 64\ngate x\n}\n",
    "fuel_range": "epistemic fn fuel_range() {\nfuel 1000001\nlet x=1\ngate x\n}\n",
    "fuel_insufficient": "epistemic fn fuel_insufficient() {\nfuel 1\nlet x=1\ngate x\n}\n",
    "undefined_gate": "epistemic fn undefined_gate() {\nfuel 64\ngate missing\n}\n",
    "duplicate_symbol": "epistemic fn duplicate_symbol() {\nfuel 64\nlet x=1\nlet x=2\ngate x\n}\n",
    "nested_expression": "epistemic fn nested_expression() {\nfuel 64\nlet x=1+2+3\ngate x\n}\n",
    "after_close": "epistemic fn after_close() {\nfuel 64\nlet x=1\ngate x\n}\ngate x\n",
    "missing_close": "epistemic fn missing_close() {\nfuel 64\nlet x=1\ngate x\n",
    "set": "epistemic fn set_value() {\nfuel 64\nlet x=1\nset x=2\ngate x\n}\n",
    "if_control": "epistemic fn if_control() {\nfuel 64\nlet x=1\nif x < 0.0 {\n}\ngate x\n}\n",
    "load_before_store": "epistemic fn load_before_store() {\nfuel 64\nlet x=load [m]\ngate x\n}\n",
    "malformed_header": "fn malformed_header() {\nfuel 64\nlet x=1\ngate x\n}\n",
}
for name, text in cases.items():
    (out / f"{name}.eisa").write_text(text, encoding="ascii")
symbols = ["epistemic fn symbol_capacity() {", "fuel 1000"] + [f"let x{i}={i}" for i in range(65)] + ["gate x63", "}"]
(out / "symbol_capacity.eisa").write_text("\n".join(symbols) + "\n", encoding="ascii")
gates = ["epistemic fn gate_capacity() {", "fuel 1000", "let x=1"] + ["gate x"] * 65 + ["}"]
(out / "gate_capacity.eisa").write_text("\n".join(gates) + "\n", encoding="ascii")
PY

source_negative_count=0
for source in "$TMP_DIR"/negative/*.eisa; do
  source_negative_count=$((source_negative_count + 1))
  if "$DRIVER" lower-v2 "$source" >"$source.out" 2>&1; then
    fail "source lowerer accepted E2F negative: $(basename "$source")"
  fi
  grep -Fq 'enir-lower-error|' "$source.out" || fail "negative lacks classified lowering error: $(basename "$source")"
done
[[ "$source_negative_count" == "18" ]] || fail "source negative count drifted: $source_negative_count"

python3 - "$TMP_DIR/cases/v2_rump_qd.enir" "$TMP_DIR/tampers" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="ascii")
out = Path(sys.argv[2]); out.mkdir()

def write(name, text):
    (out / f"{name}.enir").write_text(text, encoding="ascii")

write("fuel_zero", source.replace("resource|64", "resource|0", 1))
write("fuel_insufficient", source.replace("resource|64", "resource|29", 1))
write("halt_cost", source.replace("block|0|0|29|0|0|0|-1|-1|-1|1", "block|0|0|29|0|0|0|-1|-1|-1|0", 1))
write("block_count", source.replace("block|0|0|29|", "block|0|0|28|", 1))
write("footer_edge", source.replace("end2|1|26|26|1|1|0|0|29|3|1", "end2|1|26|26|1|1|0|1|29|3|1", 1))
write("footer_observation", source.replace("end2|1|26|26|1|1|0|0|29|3|1", "end2|1|26|26|1|1|0|0|29|2|1", 1))
write("observation_ordinal", source.replace("obs|1|v2_rump_qd|1|0", "obs|1|v2_rump_qd|0|0", 1))
write("observation_kind", source.replace("obs|2|v2_rump_qd|2|0", "obs|2|v2_rump_qd|2|1", 1))
write("future_use", source.replace("op|2|4|2|0|1|1|", "op|2|4|2|0|25|1|", 1))
write("duplicate_result", source.replace("op|3|4|3|0|2|2|", "op|3|4|2|0|2|2|", 1))
write("provenance", source.replace("prov|2|5|-1|4|-1|-1", "prov|2|5|-1|3|-1|-1", 1))
write("unknown_constant", source.replace("value|0|0|1|4680070212836392960|", "value|0|0|0|4680070212836392960|", 1))
write("gate_policy", source.replace("op|26|7|-1|-1|24|-1|-1|0|-1|1", "op|26|7|-1|-1|24|-1|-1|-1|-1|1", 1))
write("unsupported_eload", source.replace("op|2|4|2|0|1|1|-1|-1|-1|1", "op|2|1|2|0|-1|-1|-1|-1|0|1", 1))
write("type_dd64", source.replace("type|0|4|2|1|1|1|2", "type|0|4|1|1|1|1|2", 1))
row = source.splitlines()[3]
for index, field in enumerate(("error0", "error1", "error2", "error3"), 6):
    parts = row.split("|"); parts[index] = "1"
    write(field, source.replace(row, "|".join(parts), 1))
write("noncanonical", source.replace("resource|64", "resource|064", 1))
write("crlf", source.replace("\n", "\r\n"))
PY

artifact_tamper_count=0
for artifact in "$TMP_DIR"/tampers/*.enir; do
  artifact_tamper_count=$((artifact_tamper_count + 1))
  if "$DRIVER" run "$artifact" >"$artifact.out" 2>&1; then
    fail "interpreter accepted E2F artifact tamper: $(basename "$artifact")"
  fi
done
[[ "$artifact_tamper_count" == "21" ]] || fail "artifact tamper count drifted: $artifact_tamper_count"

runtime_tamper_count=0
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
  runtime_tamper_count=$((runtime_tamper_count + 1))
  if python3 scripts/dev/madaros_v2_e2f_enir_rump_qd_verify.py \
      --driver "$wrapper" --source "$SOURCE" --corpus "$CORPUS" --oracle "$ORACLE_OUT" \
      --out-dir "$TMP_DIR/tampered-$field" --receipt "$TMP_DIR/tampered-$field.json" --root "$ROOT_DIR" \
      >"$TMP_DIR/tampered-$field.log" 2>&1; then
    fail "independent verifier accepted runtime $field tamper"
  fi
done
[[ "$runtime_tamper_count" == "4" ]] || fail "runtime word tamper count drifted"

sed 's/let five5=5\.5/let five5=5.25/' "$SOURCE" >"$TMP_DIR/source-tamper.eisa"
"$DRIVER" lower-v2 "$TMP_DIR/source-tamper.eisa" >"$TMP_DIR/source-tamper.enir"
"$DRIVER" run "$TMP_DIR/cases/v2_rump_qd.enir" | grep '^enir-exec|' >"$TMP_DIR/source-original.receipts"
"$DRIVER" run "$TMP_DIR/source-tamper.enir" | grep '^enir-exec|' >"$TMP_DIR/source-tamper.receipts"
cmp -s "$TMP_DIR/source-original.receipts" "$TMP_DIR/source-tamper.receipts" \
  && fail "causal source tamper did not change v2 Rump observations"

E2E_BASE_REF=HEAD bash scripts/dev/madaros_v2_e2e_enir_qd128_gate.sh >"$TMP_DIR/e2e-regression.log"
grep -Fq 'E2E_ENIR_QD128_ARITHMETIC_FULL_GATE_PASS' "$TMP_DIR/e2e-regression.log" \
  || fail "E2E/E2D/E2C/E2B/E2A/E1 regression chain failed"

if [[ -n "$KEEP_DIR" ]]; then
  mkdir -p "$KEEP_DIR"
  cp "$RECEIPT" "$KEEP_DIR/e2f-enir-rump-qd.receipt.json"
  cp "$TMP_DIR/cases/v2_rump_qd.enir" "$KEEP_DIR/"
fi

RECEIPT_SHA="$(sha256sum "$RECEIPT" | cut -d' ' -f1)"
echo "E2F_ENIR_V2_RUMP_QD_FULL_GATE_PASS receipt_sha256=$RECEIPT_SHA programs=1 observations=3 cumulative=24/30,32/39 ops=29 semantic_instructions=30 values=26 fuel=64->34 source_negatives=$source_negative_count artifact_tampers=$artifact_tamper_count runtime_word_tampers=$runtime_tamper_count causal_source_tamper=pass graph=source==frozen-image qd128_words=independent pair_reconstruction=exact single_register_boundary=honest target_relative_bound=2^-210 final_relative_bound=2^-162 metron_receipts=exact e2e_regression=pass e2d_regression=pass e2c_regression=pass e2b_regression=pass e2a_regression=pass e1_regression=pass codegen_diff=0"
