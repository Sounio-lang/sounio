#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${E2H_BASE_REF:-}"
if [[ -z "$BASE_REF" ]]; then
  BASE_REF="$(git merge-base HEAD "${ENIR_GATE_BASE:-origin/main}")"
fi
SEED="${SOUC_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
KEEP_DIR="${E2H_RECEIPT_DIR:-}"
TMP_DIR="$(mktemp -d /tmp/madaros-e2h-enir-memory-move-poison.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

DRIVER="$TMP_DIR/madaros-enir"
ORACLE="$TMP_DIR/eisa-metron-corpus"
ORACLE_OUT="$TMP_DIR/eisa-metron-corpus.out"
RECEIPT="$TMP_DIR/e2h-enir-memory-move-poison.receipt.json"

fail() {
  echo "E2H_ENIR_MEMORY_MOVE_POISON_GATE_FAIL: $*" >&2
  exit 1
}

. scripts/dev/madaros_v2_enir_gate_scope.sh

[[ -x "$SEED" ]] || fail "missing executable Stage0 seed: $SEED"
git rev-parse --verify "$BASE_REF" >/dev/null 2>&1 || fail "base ref not found: $BASE_REF"
E2H_PROTECTED=(
  self-hosted/compiler/main.sio self-hosted/ir self-hosted/native self-hosted/wasm \
  self-hosted/gpu stdlib/runtime stdlib/eisa stdlib/math/dd64.sio stdlib/math/qd128.sio \
  self-hosted/enir/qd.sio tools/eisa/eisa_evm_run.sio \
)
madaros_v2_enir_gate_scope_or_skip "$BASE_REF" "E2H_ENIR_MEMORY_MOVE_POISON_GATE" \
  "E2H changed codegen/ABI/runtime, pinned qd semantics, or frozen METRON oracle" "${E2H_PROTECTED[@]}"

scripts/dev/souc-build-lock.sh "$SEED" self-hosted/enir/driver.sio "$DRIVER" >"$TMP_DIR/driver-build.log" 2>&1
[[ -s "$DRIVER" ]] || fail "native ENIR driver build produced no ELF"
if grep -Eq '^error(\[E[0-9]+\])?:|unknown identifier|typecheck: failed|assignment type mismatch|private struct field' "$TMP_DIR/driver-build.log"; then
  tail -100 "$TMP_DIR/driver-build.log" >&2
  fail "Stage0 reported diagnostics while building ENIR driver"
fi
chmod +x "$DRIVER"

scripts/dev/souc-build-lock.sh "$SEED" tools/eisa/eisa_evm_run.sio "$ORACLE" >"$TMP_DIR/oracle-build.log" 2>&1
[[ -s "$ORACLE" ]] || fail "source-fresh METRON corpus build produced no ELF"
if grep -Eq '^error(\[E[0-9]+\])?:|unknown identifier|typecheck: failed|assignment type mismatch' "$TMP_DIR/oracle-build.log"; then
  tail -100 "$TMP_DIR/oracle-build.log" >&2
  fail "Stage0 reported diagnostics while building METRON corpus"
fi
chmod +x "$ORACLE"
"$ORACLE" >"$ORACLE_OUT"
[[ "$(grep -c '^eisa-receipt:' "$ORACLE_OUT")" == "39" ]] || fail "METRON observation manifest drifted"

python3 scripts/dev/madaros_v2_e2h_enir_memory_move_poison_verify.py \
  --driver "$DRIVER" --source-dir tools/eisa --corpus tools/eisa/eisa_evm_run.sio \
  --oracle "$ORACLE_OUT" --out-dir "$TMP_DIR/cases" --receipt "$RECEIPT" --root "$ROOT_DIR"

# A second store must dominate the later load, and the full product must come
# from that store rather than the first slot value.
cat >"$TMP_DIR/two-store.eisa" <<'EOF'
epistemic fn two_store() {
fuel 20
let first=7.25
store [m] <- first
let second=8.5
store [m] <- second
let loaded=load [m]
gate loaded
}
EOF
"$DRIVER" lower-v2 "$TMP_DIR/two-store.eisa" >"$TMP_DIR/two-store.enir"
grep -Fq 'prov|2|7|3|1|-1|-1' "$TMP_DIR/two-store.enir" || fail "load provenance does not name latest dominating store site"
"$DRIVER" run "$TMP_DIR/two-store.enir" >"$TMP_DIR/two-store.out"
grep -Fq 'value_bits=4620974692658839552' "$TMP_DIR/two-store.out" || fail "second store did not determine loaded value"
grep -Fq 'enir-memory|module=two_store|slot=0|site=3|value_bits=4620974692658839552' "$TMP_DIR/two-store.out" || fail "latest store receipt missing"

cat >"$TMP_DIR/negative-zero-memory.eisa" <<'EOF'
epistemic fn negative_zero_memory() {
fuel 20
let nz=-0
store [m] <- nz
let loaded=load [m]
let moved=loaded
gate loaded
gate moved
}
EOF
"$DRIVER" lower-v2 "$TMP_DIR/negative-zero-memory.eisa" >"$TMP_DIR/negative-zero-memory.enir"
"$DRIVER" run "$TMP_DIR/negative-zero-memory.enir" >"$TMP_DIR/negative-zero-memory.out"
[[ "$(grep -c '^enir-exec|.*value_bits=0|' "$TMP_DIR/negative-zero-memory.out")" == "2" ]] || fail "literal negative zero was not canonical at both gates"
grep -Fq 'enir-memory|module=negative_zero_memory|slot=0|site=1|value_bits=0|' "$TMP_DIR/negative-zero-memory.out" || fail "literal negative zero was not canonical in memory"

mkdir -p "$TMP_DIR/negative"
python3 - "$TMP_DIR/negative" <<'PY'
from pathlib import Path
import sys

out = Path(sys.argv[1])
cases = {
    "load_before_store": "epistemic fn load_before_store() {\nfuel 12\nlet x=load [m]\ngate x\n}\n",
    "undefined_store_value": "epistemic fn undefined_store_value() {\nfuel 12\nstore [m] <- x\n}\n",
    "bad_store_brackets": "epistemic fn bad_store_brackets() {\nfuel 12\nlet x=1\nstore m <- x\ngate x\n}\n",
    "bad_store_arrow": "epistemic fn bad_store_arrow() {\nfuel 12\nlet x=1\nstore [m] = x\ngate x\n}\n",
    "bad_load_brackets": "epistemic fn bad_load_brackets() {\nfuel 12\nlet x=load m\ngate x\n}\n",
    "bad_load_tail": "epistemic fn bad_load_tail() {\nfuel 12\nlet x=1\nstore [m] <- x\nlet y=load [m] junk\ngate y\n}\n",
    "empty_slot": "epistemic fn empty_slot() {\nfuel 12\nlet x=1\nstore [] <- x\ngate x\n}\n",
    "numeric_slot": "epistemic fn numeric_slot() {\nfuel 12\nlet x=1\nstore [7] <- x\ngate x\n}\n",
    "duplicate_symbol": "epistemic fn duplicate_symbol() {\nfuel 12\nlet x=1\nstore [m] <- x\nlet x=load [m]\ngate x\n}\n",
    "memory_in_loop": "epistemic fn memory_in_loop() {\nfuel 20\nlet x=1\nwhile x != 0.0 {\n}\nstore [m] <- x\n}\n",
    "load_in_loop": "epistemic fn load_in_loop() {\nfuel 20\nlet x=1\nwhile x != 0.0 {\n}\nlet y=load [m]\n}\n",
    "missing_gate": "epistemic fn missing_gate() {\nfuel 12\nlet x=1\nstore [m] <- x\nlet y=load [m]\n}\n",
    "store_literal": "epistemic fn store_literal() {\nfuel 12\nstore [m] <- 1\n}\n",
    "load_expression": "epistemic fn load_expression() {\nfuel 12\nlet x=1\nstore [m] <- x\nlet y=load [m]+x\ngate y\n}\n",
    "missing_close": "epistemic fn missing_close() {\nfuel 12\nlet x=1\nstore [m] <- x\n",
}
for name, text in cases.items():
    (out / f"{name}.eisa").write_text(text, encoding="ascii")

lines = ["epistemic fn too_many_slots() {", "fuel 80", "let x=1"]
lines += [f"store [s{i}] <- x" for i in range(17)]
lines += ["gate x", "}"]
(out / "too_many_slots.eisa").write_text("\n".join(lines) + "\n", encoding="ascii")
PY

source_negative_count=0
for source in "$TMP_DIR"/negative/*.eisa; do
  source_negative_count=$((source_negative_count + 1))
  if "$DRIVER" lower-v2 "$source" >"$source.out" 2>&1; then
    fail "source lowerer accepted E2H negative: $(basename "$source")"
  fi
  grep -Fq 'enir-lower-error|' "$source.out" || fail "negative lacks classified lowering error: $(basename "$source")"
done
[[ "$source_negative_count" == "16" ]] || fail "source negative count drifted: $source_negative_count"

python3 - "$TMP_DIR/cases/v2_mem.enir" "$TMP_DIR/cases/v2_emov.enir" "$TMP_DIR/cases/v2_mem_poison.enir" "$TMP_DIR/tampers" <<'PY'
from pathlib import Path
import sys

mem = Path(sys.argv[1]).read_text(encoding="ascii")
emov = Path(sys.argv[2]).read_text(encoding="ascii")
poison = Path(sys.argv[3]).read_text(encoding="ascii")
out = Path(sys.argv[4]); out.mkdir()

def write(name, text):
    (out / f"{name}.enir").write_text(text, encoding="ascii")

write("profile_v1", mem.replace("enir|2|2|v2_mem|2", "enir|2|2|v2_mem|1", 1))
write("type_dd64", mem.replace("type|0|4|2|1|1|1|2", "type|0|4|1|1|1|1|2", 1))
write("zero_fuel", mem.replace("resource|12", "resource|0", 1))
write("block_count", mem.replace("end2|1|2|2|1|1|0|0|4|1|1", "end2|1|2|2|1|2|0|0|4|1|1", 1))
write("load_slot_oob", mem.replace("op|2|1|1|0|-1|-1|-1|-1|0|1", "op|2|1|1|0|-1|-1|-1|-1|16|1", 1))
write("store_slot_oob", mem.replace("op|1|8|-1|-1|0|-1|-1|-1|0|1", "op|1|8|-1|-1|0|-1|-1|-1|16|1", 1))
write("load_wrong_slot", mem.replace("op|2|1|1|0|-1|-1|-1|-1|0|1", "op|2|1|1|0|-1|-1|-1|-1|1|1", 1))
write("load_before_store", mem.replace("op|1|8|-1|-1|0|-1|-1|-1|0|1\nop|2|1|1|0|-1|-1|-1|-1|0|1", "op|1|1|1|0|-1|-1|-1|-1|0|1\nop|2|8|-1|-1|0|-1|-1|-1|0|1", 1))
write("load_provenance", mem.replace("prov|1|5|1|1|-1|-1", "prov|1|5|0|1|-1|-1", 1))
write("load_has_operand", mem.replace("op|2|1|1|0|-1|-1|-1|-1|0|1", "op|2|1|1|0|0|-1|-1|-1|0|1", 1))
write("store_has_result", mem.replace("op|1|8|-1|-1|0|-1|-1|-1|0|1", "op|1|8|1|0|0|-1|-1|-1|0|1", 1))
write("store_operand_oob", mem.replace("op|1|8|-1|-1|0|-1|-1|-1|0|1", "op|1|8|-1|-1|99|-1|-1|-1|0|1", 1))
write("emov_operand_oob", emov.replace("op|4|9|4|0|3|-1|-1|-1|-1|1", "op|4|9|4|0|99|-1|-1|-1|-1|1", 1))
write("gate_operand_oob", emov.replace("op|6|7|-1|-1|4|-1|-1|0|-1|1", "op|6|7|-1|-1|99|-1|-1|0|-1|1", 1))
write("move_transform", emov.replace("prov|4|7|-1|9|-1|-1", "prov|4|7|-1|1|-1|-1", 1))
write("poison_status_descriptor", poison.replace("value|4|0|0|0|-1|0|0|0|0|0|0|-1|4", "value|4|0|0|0|-1|0|0|0|0|1|0|-1|4", 1))
row = poison.splitlines()[4]
for index, field in enumerate(("error0", "error1", "error2", "error3"), 6):
    parts = row.split("|"); parts[index] = "1"
    write(field, poison.replace(row, "|".join(parts), 1))
write("uncertainty_descriptor", poison.replace("value|2|0|0|0|-1|0|0|0|0|0|0|-1|2", "value|2|0|0|0|-1|0|0|0|0|0|1|-1|2", 1))
write("policy_reference", mem.replace("op|3|7|-1|-1|1|-1|-1|0|-1|1", "op|3|7|-1|-1|1|-1|-1|1|-1|1", 1))
write("footer_ops", mem.replace("end2|1|2|2|1|1|0|0|4|1|1", "end2|1|2|2|1|1|0|0|3|1|1", 1))
write("noncanonical", mem.replace("resource|12", "resource|012", 1))
write("crlf", mem.replace("\n", "\r\n"))
PY

artifact_tamper_count=0
for artifact in "$TMP_DIR"/tampers/*.enir; do
  artifact_tamper_count=$((artifact_tamper_count + 1))
  if "$DRIVER" verify "$artifact" >"$artifact.out" 2>&1; then
    fail "verifier accepted E2H artifact tamper: $(basename "$artifact")"
  fi
done
[[ "$artifact_tamper_count" == "25" ]] || fail "artifact tamper count drifted: $artifact_tamper_count"

runtime_tamper_count=0
for field in site value_bits error0_bits error1_bits error2_bits error3_bits uncertainty_bits status; do
  wrapper="$TMP_DIR/tamper-memory-$field"
  cat >"$wrapper" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" == "run" ]]; then
  "$DRIVER" "\$@" | sed "/^enir-memory|/s/$field=[^|]*/$field=777/"
else
  exec "$DRIVER" "\$@"
fi
EOF
  chmod +x "$wrapper"
  runtime_tamper_count=$((runtime_tamper_count + 1))
  if python3 scripts/dev/madaros_v2_e2h_enir_memory_move_poison_verify.py \
      --driver "$wrapper" --source-dir tools/eisa --corpus tools/eisa/eisa_evm_run.sio \
      --oracle "$ORACLE_OUT" --out-dir "$TMP_DIR/tampered-memory-$field" \
      --receipt "$TMP_DIR/tampered-memory-$field.json" --root "$ROOT_DIR" >"$TMP_DIR/tampered-memory-$field.log" 2>&1; then
    fail "independent verifier accepted memory receipt $field tamper"
  fi
done

wrapper="$TMP_DIR/tamper-emov-second-observation"
cat >"$wrapper" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" == "run" ]]; then
  "$DRIVER" "\$@" | sed "/ordinal=1/s/value_bits=[^|]*/value_bits=777/"
else
  exec "$DRIVER" "\$@"
fi
EOF
chmod +x "$wrapper"
runtime_tamper_count=$((runtime_tamper_count + 1))
if python3 scripts/dev/madaros_v2_e2h_enir_memory_move_poison_verify.py \
    --driver "$wrapper" --source-dir tools/eisa --corpus tools/eisa/eisa_evm_run.sio \
    --oracle "$ORACLE_OUT" --out-dir "$TMP_DIR/tampered-emov" \
    --receipt "$TMP_DIR/tampered-emov.json" --root "$ROOT_DIR" >"$TMP_DIR/tampered-emov.log" 2>&1; then
  fail "independent verifier accepted second emov observation tamper"
fi
[[ "$runtime_tamper_count" == "9" ]] || fail "runtime tamper count drifted: $runtime_tamper_count"

mkdir -p "$TMP_DIR/source-tamper"
cp tools/eisa/eisa_enir_v2_{mem,emov,mem_poison}.eisa "$TMP_DIR/source-tamper/"
sed -i 's/let x=7.25/let x=7.5/' "$TMP_DIR/source-tamper/eisa_enir_v2_mem.eisa"
if python3 scripts/dev/madaros_v2_e2h_enir_memory_move_poison_verify.py \
    --driver "$DRIVER" --source-dir "$TMP_DIR/source-tamper" --corpus tools/eisa/eisa_evm_run.sio \
    --oracle "$ORACLE_OUT" --out-dir "$TMP_DIR/source-tamper-cases" \
    --receipt "$TMP_DIR/source-tamper.json" --root "$ROOT_DIR" >"$TMP_DIR/source-tamper.log" 2>&1; then
  fail "causal source tamper passed frozen graph/METRON validation"
fi
grep -Fq 'source/frozen-image graph mismatch' "$TMP_DIR/source-tamper.log" || fail "causal source tamper lacked graph-level rejection"

E2G_BASE_REF=HEAD bash scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_gate.sh >"$TMP_DIR/e2g-regression.log"
grep -Fq 'E2G_ENIR_V2_FUEL_CONTROL_FRAIL_FULL_GATE_PASS' "$TMP_DIR/e2g-regression.log" || fail "E2G-to-E1 regression chain failed"

if [[ -n "$KEEP_DIR" ]]; then
  mkdir -p "$KEEP_DIR"
  cp "$RECEIPT" "$KEEP_DIR/e2h-enir-memory-move-poison.receipt.json"
  cp "$TMP_DIR"/cases/*.enir "$KEEP_DIR/"
fi

RECEIPT_SHA="$(sha256sum "$RECEIPT" | cut -d' ' -f1)"
echo "E2H_ENIR_MEMORY_MOVE_POISON_FULL_GATE_PASS receipt_sha256=$RECEIPT_SHA programs=3 observations=4 cumulative=30/30,39/39 source_negatives=$source_negative_count artifact_tampers=$artifact_tamper_count runtime_tampers=$runtime_tamper_count two_store_dominance=pass causal_source_tamper=pass qd128_memory=atomic-full-product emov=word-identical negzero=arithmetic-and-literal-canonical poison=store-load-move-preserved metron_receipts=exact e2g_regression=pass e2f_regression=pass e2e_regression=pass e2d_regression=pass e2c_regression=pass e2b_regression=pass e2a_regression=pass e1_regression=pass e2_scope_enir_to_mir=separate-stage codegen_diff=0"
