#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${E2G_BASE_REF:-}"
if [[ -z "$BASE_REF" ]]; then
  BASE_REF="$(git merge-base HEAD "${ENIR_GATE_BASE:-origin/main}")"
fi
SEED="${SOUC_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
KEEP_DIR="${E2G_RECEIPT_DIR:-}"
TMP_DIR="$(mktemp -d /tmp/madaros-e2g-enir-fuel-control-frail.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

DRIVER="$TMP_DIR/madaros-enir"
ORACLE="$TMP_DIR/eisa-metron-corpus"
ORACLE_OUT="$TMP_DIR/eisa-metron-corpus.out"
RECEIPT="$TMP_DIR/e2g-enir-fuel-control-frail.receipt.json"

fail() {
  echo "E2G_ENIR_V2_FUEL_CONTROL_FRAIL_GATE_FAIL: $*" >&2
  exit 1
}

. scripts/dev/madaros_v2_enir_gate_scope.sh

[[ -x "$SEED" ]] || fail "missing executable Stage0 seed: $SEED"
git rev-parse --verify "$BASE_REF" >/dev/null 2>&1 || fail "base ref not found: $BASE_REF"
E2G_PROTECTED=(self-hosted/compiler/main.sio self-hosted/ir self-hosted/native self-hosted/wasm self-hosted/gpu stdlib/runtime stdlib/eisa stdlib/math/dd64.sio stdlib/math/qd128.sio self-hosted/enir/qd.sio tools/eisa/eisa_evm_run.sio)
madaros_v2_enir_gate_scope_or_skip "$BASE_REF" "E2G_ENIR_V2_FUEL_CONTROL_FRAIL_GATE" \
  "E2G changed codegen/ABI/runtime, pinned qd semantics, or frozen METRON oracle" "${E2G_PROTECTED[@]}"

scripts/dev/souc-build-lock.sh "$SEED" self-hosted/enir/driver.sio "$DRIVER" >"$TMP_DIR/driver-build.log" 2>&1
[[ -s "$DRIVER" ]] || fail "native ENIR driver build produced no ELF"
if grep -Eq '^error:|unknown identifier|typecheck: failed|assignment type mismatch|private struct field' "$TMP_DIR/driver-build.log"; then
  tail -100 "$TMP_DIR/driver-build.log" >&2
  fail "Stage0 reported diagnostics while building ENIR driver"
fi
chmod +x "$DRIVER"

scripts/dev/souc-build-lock.sh "$SEED" tools/eisa/eisa_evm_run.sio "$ORACLE" >"$TMP_DIR/oracle-build.log" 2>&1
[[ -s "$ORACLE" ]] || fail "source-fresh METRON corpus build produced no ELF"
if grep -Eq '^error:|unknown identifier|typecheck: failed|assignment type mismatch' "$TMP_DIR/oracle-build.log"; then
  tail -100 "$TMP_DIR/oracle-build.log" >&2
  fail "Stage0 reported diagnostics while building METRON corpus"
fi
chmod +x "$ORACLE"
"$ORACLE" >"$ORACLE_OUT"
[[ "$(grep -c '^eisa-receipt:' "$ORACLE_OUT")" == "39" ]] || fail "METRON observation manifest drifted"

python3 scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_verify.py --driver "$DRIVER" --corpus tools/eisa/eisa_evm_run.sio --oracle "$ORACLE_OUT" --out-dir "$TMP_DIR/cases" --receipt "$RECEIPT" --root "$ROOT_DIR"

mkdir -p "$TMP_DIR/negative"
python3 - "$TMP_DIR/negative" <<'PY'
from pathlib import Path
import sys

out = Path(sys.argv[1])
cases = {
    "duplicate_fuel": "epistemic fn duplicate_fuel() {\nfuel 5\nfuel 6\nlet x=1\nwhile x != 0.0 {\n}\n}\n",
    "zero_fuel": "epistemic fn zero_fuel() {\nfuel 0\nlet x=1\nwhile x != 0.0 {\n}\n}\n",
    "fractional_fuel": "epistemic fn fractional_fuel() {\nfuel 5.5\nlet x=1\nwhile x != 0.0 {\n}\n}\n",
    "late_fuel": "epistemic fn late_fuel() {\nlet x=1\nfuel 5\nwhile x != 0.0 {\n}\n}\n",
    "nested_loop": "epistemic fn nested_loop() {\nfuel 10\nlet x=1\nwhile x != 0.0 {\nwhile x != 0.0 {\n}\n}\n}\n",
    "second_loop": "epistemic fn second_loop() {\nfuel 10\nlet x=1\nwhile x != 0.0 {\n}\nwhile x != 0.0 {\n}\n}\n",
    "set_undefined": "epistemic fn set_undefined() {\nfuel 10\nlet x=1\nwhile x != 0.0 {\nset y=0\n}\n}\n",
    "set_local": "epistemic fn set_local() {\nfuel 10\nlet x=1\nwhile x != 0.0 {\nlet y=1\nset y=0\n}\n}\n",
    "gate_inside": "epistemic fn gate_inside() {\nfuel 10\nlet x=1\nwhile x != 0.0 {\ngate x\n}\n}\n",
    "two_gates": "epistemic fn two_gates() {\nfuel 20\nlet x=0\nwhile x != 0.0 {\n}\ngate x\ngate x\n}\n",
    "high_body": "epistemic fn high_body() {\nfuel 20\nlet a=0\nlet b=0\nlet c=0\nlet d=0\nlet e=1\nwhile e != 0.0 {\nlet x=1\n}\n}\n",
    "unsupported_if": "epistemic fn unsupported_if() {\nfuel 20\nlet x=1\nwhile x != 0.0 {\nif x < 0.0 {\n}\n}\n}\n",
    "unsupported_store": "epistemic fn unsupported_store() {\nfuel 20\nlet x=1\nwhile x != 0.0 {\n}\nstore [m] <- x\n}\n",
    "missing_root_close": "epistemic fn missing_root_close() {\nfuel 5\nlet x=1\nwhile x != 0.0 {\n}\n",
    "malformed_while": "epistemic fn malformed_while() {\nfuel 5\nlet x=1\nwhile x > 0.0 {\n}\n}\n",
    "undefined_condition": "epistemic fn undefined_condition() {\nfuel 5\nwhile x != 0.0 {\n}\n}\n",
    "gate_undefined": "epistemic fn gate_undefined() {\nfuel 10\nlet x=0\nwhile x != 0.0 {\n}\ngate y\n}\n",
    "after_close": "epistemic fn after_close() {\nfuel 10\nlet x=0\nwhile x != 0.0 {\n}\n}\ngate x\n",
}
for name, text in cases.items():
    (out / f"{name}.eisa").write_text(text, encoding="ascii")
PY

source_negative_count=0
for source in "$TMP_DIR"/negative/*.eisa; do
  source_negative_count=$((source_negative_count + 1))
  if "$DRIVER" lower-v2 "$source" >"$source.out" 2>&1; then
    fail "source lowerer accepted E2G negative: $(basename "$source")"
  fi
  grep -Fq 'enir-lower-error|' "$source.out" || fail "negative lacks classified lowering error: $(basename "$source")"
done
[[ "$source_negative_count" == "18" ]] || fail "source negative count drifted: $source_negative_count"

python3 - "$TMP_DIR/cases/v2_frail.enir" "$TMP_DIR/cases/v2_fuel.enir" "$TMP_DIR/cases/v2_loop.enir" "$TMP_DIR/tampers" <<'PY'
from pathlib import Path
import sys

frail = Path(sys.argv[1]).read_text(encoding="ascii")
fuel = Path(sys.argv[2]).read_text(encoding="ascii")
loop = Path(sys.argv[3]).read_text(encoding="ascii")
out = Path(sys.argv[4]); out.mkdir()

def write(name, text):
    (out / f"{name}.enir").write_text(text, encoding="ascii")

write("profile_v1", frail.replace("enir|2|2|v2_frail|2", "enir|2|2|v2_frail|1", 1))
write("type_dd64", frail.replace("type|0|4|2|1|1|1|2", "type|0|4|1|1|1|1|2", 1))
write("block_count", frail.replace("end2|1|12|12|1|4|8|4|5|1|1", "end2|1|12|12|1|3|8|4|5|1|1", 1))
write("edge_count", frail.replace("end2|1|12|12|1|4|8|4|5|1|1", "end2|1|12|12|1|4|8|3|5|1|1", 1))
write("entry_cost", frail.replace("block|0|0|4|0|0|1|-1|0|-1|0", "block|0|0|4|0|0|1|-1|0|-1|1", 1))
write("header_kind", frail.replace("block|1|4|0|0|4|2|10|1|2|1", "block|1|4|0|0|4|3|10|1|2|1", 1))
write("body_cost", frail.replace("block|2|4|0|4|0|1|-1|3|-1|1", "block|2|4|0|4|0|1|-1|3|-1|0", 1))
write("exit_cost", frail.replace("block|3|4|1|4|4|0|-1|-1|-1|1", "block|3|4|1|4|4|0|-1|-1|-1|0", 1))
write("edge_arity", frail.replace("edge|0|0|1|4|", "edge|0|0|1|3|", 1))
write("edge_owner", frail.replace("edge|3|2|1|4|", "edge|3|3|1|4|", 1))
write("edge_target", frail.replace("edge|2|1|2|0|", "edge|2|1|0|0|", 1))
write("condition_oob", frail.replace("block|1|4|0|0|4|2|10|", "block|1|4|0|0|4|2|99|", 1))
write("dominance", frail.replace("op|3|3|3|0|2|0|", "op|3|3|3|0|2|10|", 1))
write("fuel_zero", frail.replace("resource|30", "resource|0", 1))
write("fuel_condition_zero", fuel.replace("value|0|0|1|4607182418800017408|3|", "value|0|0|1|0|0|", 1))
row = frail.splitlines()[3]
for index, field in enumerate(("error0", "error1", "error2", "error3"), 6):
    parts = row.split("|"); parts[index] = "1"
    write(field, frail.replace(row, "|".join(parts), 1))
write("loop_exit_cost", loop.replace("block|3|2|1|2|2|0|-1|-1|-1|1", "block|3|2|1|2|2|0|-1|-1|-1|0", 1))
write("noncanonical", frail.replace("resource|30", "resource|030", 1))
write("crlf", frail.replace("\n", "\r\n"))
PY

artifact_tamper_count=0
for artifact in "$TMP_DIR"/tampers/*.enir; do
  artifact_tamper_count=$((artifact_tamper_count + 1))
  if "$DRIVER" verify "$artifact" >"$artifact.out" 2>&1; then
    fail "verifier accepted E2G artifact tamper: $(basename "$artifact")"
  fi
done
[[ "$artifact_tamper_count" == "22" ]] || fail "artifact tamper count drifted: $artifact_tamper_count"

runtime_tamper_count=0
for field in error0_bits error1_bits error2_bits error3_bits frail_branches; do
  wrapper="$TMP_DIR/tamper-$field"
  cat >"$wrapper" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" == "run" ]]; then
  "$DRIVER" "\$@" | sed "0,/$field=/s/$field=[^|]*/$field=777/"
else
  exec "$DRIVER" "\$@"
fi
EOF
  chmod +x "$wrapper"
  runtime_tamper_count=$((runtime_tamper_count + 1))
  if python3 scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_verify.py --driver "$wrapper" --corpus tools/eisa/eisa_evm_run.sio --oracle "$ORACLE_OUT" --out-dir "$TMP_DIR/tampered-$field" --receipt "$TMP_DIR/tampered-$field.json" --root "$ROOT_DIR" >"$TMP_DIR/tampered-$field.log" 2>&1; then
    fail "independent verifier accepted runtime $field tamper"
  fi
done
[[ "$runtime_tamper_count" == "5" ]] || fail "runtime tamper count drifted"

sed 's/let one=1$/let one=2/' tools/eisa/eisa_enir_v2_frail.eisa >"$TMP_DIR/source-tamper.eisa"
"$DRIVER" lower-v2 "$TMP_DIR/source-tamper.eisa" >"$TMP_DIR/source-tamper.enir"
"$DRIVER" run "$TMP_DIR/cases/v2_frail.enir" >"$TMP_DIR/source-original.receipt"
set +e
"$DRIVER" run "$TMP_DIR/source-tamper.enir" >"$TMP_DIR/source-tamper.receipt" 2>&1
tamper_rc=$?
set -e
[[ "$tamper_rc" != "0" ]] || fail "causal source tamper unexpectedly reached the declared gate"
grep -Fq 'enir-exec-error|code=70' "$TMP_DIR/source-tamper.receipt" || fail "causal source tamper lacked fail-closed missing-gate evidence"
cmp -s "$TMP_DIR/source-original.receipt" "$TMP_DIR/source-tamper.receipt" && fail "causal source tamper did not change v2 frail execution"

E2F_BASE_REF=HEAD bash scripts/dev/madaros_v2_e2f_enir_rump_qd_gate.sh >"$TMP_DIR/e2f-regression.log"
grep -Fq 'E2F_ENIR_V2_RUMP_QD_FULL_GATE_PASS' "$TMP_DIR/e2f-regression.log" || fail "E2F/E2E/E2D/E2C/E2B/E2A/E1 regression chain failed"

if [[ -n "$KEEP_DIR" ]]; then
  mkdir -p "$KEEP_DIR"
  cp "$RECEIPT" "$KEEP_DIR/e2g-enir-fuel-control-frail.receipt.json"
  cp "$TMP_DIR"/cases/*.enir "$KEEP_DIR/"
fi

RECEIPT_SHA="$(sha256sum "$RECEIPT" | cut -d' ' -f1)"
echo "E2G_ENIR_V2_FUEL_CONTROL_FRAIL_FULL_GATE_PASS receipt_sha256=$RECEIPT_SHA programs=3 observations=3 cumulative=27/30,35/39 blocks=12 edges=12 fuel=5->0,24->19,30->23 source_negatives=$source_negative_count artifact_tampers=$artifact_tamper_count runtime_tampers=$runtime_tamper_count causal_source_tamper=pass control=zero,nonzero,frail qd128_words=independent frail_true_value=exact1 metron_receipts=exact e2f_regression=pass e2e_regression=pass e2d_regression=pass e2c_regression=pass e2b_regression=pass e2a_regression=pass e1_regression=pass codegen_diff=0"
