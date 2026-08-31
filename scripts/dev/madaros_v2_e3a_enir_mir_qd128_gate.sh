#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${E3A_BASE_REF:-}"
if [[ -z "$BASE_REF" ]]; then
  BASE_REF="$(git merge-base HEAD "${ENIR_GATE_BASE:-origin/main}")"
fi
SEED="${SOUC_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
KEEP_DIR="${E3A_RECEIPT_DIR:-}"
TMP_DIR="$(mktemp -d /tmp/madaros-e3a-enir-mir-qd128.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

DRIVER="$TMP_DIR/madaros-enir"
ORACLE="$TMP_DIR/eisa-metron-corpus"
ORACLE_OUT="$TMP_DIR/eisa-metron-corpus.out"
RECEIPT="$TMP_DIR/e3a-enir-mir-qd128.receipt.json"

fail() {
  echo "E3A_ENIR_MIR_QD128_GATE_FAIL: $*" >&2
  exit 1
}

. scripts/dev/madaros_v2_enir_gate_scope.sh

[[ -x "$SEED" ]] || fail "missing executable Stage0 seed: $SEED"
git rev-parse --verify "$BASE_REF" >/dev/null 2>&1 || fail "base ref not found: $BASE_REF"
E3A_PROTECTED=(self-hosted/compiler/main.sio self-hosted/ir self-hosted/native self-hosted/wasm self-hosted/gpu stdlib/runtime stdlib/eisa stdlib/math/dd64.sio stdlib/math/qd128.sio self-hosted/enir/qd.sio tools/eisa/eisa_evm_run.sio)
if [[ "${E3A_ALLOW_DOWNSTREAM_ENIR_EXTENSION:-0}" != "1" ]]; then E3A_PROTECTED+=(self-hosted/enir/ir.sio self-hosted/enir/interpreter.sio self-hosted/enir/source_lower.sio); fi
madaros_v2_enir_gate_scope_or_skip "$BASE_REF" "E3A_ENIR_MIR_QD128_GATE" \
  "E3A changed protected production, ENIR, qd, or oracle surfaces" "${E3A_PROTECTED[@]}"

scripts/dev/souc-build-lock.sh "$SEED" self-hosted/enir/driver.sio "$DRIVER" >"$TMP_DIR/driver-build.log" 2>&1
[[ -s "$DRIVER" ]] || fail "ENIR/MIR driver build produced no ELF"
if grep -Eq '^error(\[E[0-9]+\])?:|unknown identifier|typecheck: failed|assignment type mismatch|private struct field' "$TMP_DIR/driver-build.log"; then
  tail -100 "$TMP_DIR/driver-build.log" >&2
  fail "Stage0 reported diagnostics while building ENIR/MIR driver"
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

python3 scripts/dev/madaros_v2_e3a_enir_mir_qd128_verify.py \
  --driver "$DRIVER" --source-dir tools/eisa --corpus tools/eisa/eisa_evm_run.sio \
  --oracle "$ORACLE_OUT" --out-dir "$TMP_DIR/cases" --receipt "$RECEIPT" --root "$ROOT_DIR"

# E3A remains the arithmetic regression slice; E3B owns memory and move.
mkdir -p "$TMP_DIR/unsupported"
for name in v2_loop v2_frail v2_fuel; do
  "$DRIVER" lower-v2 "tools/eisa/eisa_enir_${name}.eisa" >"$TMP_DIR/unsupported/${name}.enir"
done
"$DRIVER" lower-v1 tools/eisa/eisa_enir_v1_rump_dd.eisa >"$TMP_DIR/unsupported/v1_rump_dd.enir"
cat >"$TMP_DIR/v0.eisa" <<'EOF'
epistemic fn e3a_v0() {
let x = 1.0
gate x
}
EOF
"$DRIVER" lower "$TMP_DIR/v0.eisa" >"$TMP_DIR/unsupported/v0.enir"
unsupported_count=0
for artifact in "$TMP_DIR"/unsupported/*.enir; do
  unsupported_count=$((unsupported_count + 1))
  if "$DRIVER" lower-mir "$artifact" >"$artifact.out" 2>&1; then
    fail "E3A lowering accepted unsupported ENIR: $(basename "$artifact")"
  fi
  grep -Fq 'mir-lower-error|' "$artifact.out" || fail "unsupported ENIR lacks classified MIR lowering error: $(basename "$artifact")"
done
[[ "$unsupported_count" == "5" ]] || fail "unsupported ENIR count drifted: $unsupported_count"

# Poison is an explicit arithmetic trap outcome, not a crash or omitted row.
cat >"$TMP_DIR/poison.eisa" <<'EOF'
epistemic fn e3a_poison() {
fuel 12
let one=1
let zero=0
let bad=one/zero
gate bad
}
EOF
"$DRIVER" lower-v2 "$TMP_DIR/poison.eisa" >"$TMP_DIR/poison.enir"
"$DRIVER" lower-mir "$TMP_DIR/poison.enir" >"$TMP_DIR/poison.emir"
"$DRIVER" validate-mir "$TMP_DIR/poison.enir" "$TMP_DIR/poison.emir" >"$TMP_DIR/poison.relation"
"$DRIVER" run "$TMP_DIR/poison.enir" >"$TMP_DIR/poison.enir.out"
"$DRIVER" run-mir "$TMP_DIR/poison.emir" >"$TMP_DIR/poison.mir.out"
grep -Fq 'value_bits=9218868437227405313' "$TMP_DIR/poison.mir.out" || fail "MIR poison witness lacks canonical payload"
grep -Fq 'uncertainty_bits=9218868437227405312' "$TMP_DIR/poison.mir.out" || fail "MIR poison witness lacks infinite uncertainty"
grep -Fq 'status=1' "$TMP_DIR/poison.mir.out" || fail "MIR poison witness lacks poisoned status"

python3 - "$TMP_DIR/cases/v2_add.emir" "$TMP_DIR/cases/v2_add.enir" "$TMP_DIR/tampers" <<'PY'
from pathlib import Path
import sys

mir = Path(sys.argv[1]).read_text(encoding="ascii")
enir = Path(sys.argv[2]).read_text(encoding="ascii")
out = Path(sys.argv[3]); out.mkdir()

def write(name, text):
    (out / f"{name}.emir").write_text(text, encoding="ascii")

write("schema", mir.replace("emir|2|3|", "emir|1|3|", 1))
write("stage", mir.replace("emir|2|3|", "emir|2|2|", 1))
write("profile", mir.replace("|v2_add|2|12|", "|v2_add|1|12|", 1))
write("fuel", mir.replace("|v2_add|2|12|", "|v2_add|2|4|", 1))
write("source_hash", mir.replace(mir.splitlines()[0].rsplit("|", 1)[1], "1", 1))
write("type_error", mir.replace("mtype|0|4|2|1|1|1|2", "mtype|0|4|1|1|1|1|2", 1))
write("value_source_id", mir.replace("mvalue|2|0|0|0|-1|0|0|0|0|0|0|-1|2|2", "mvalue|2|0|0|0|-1|0|0|0|0|0|0|-1|2|1", 1))
write("value_bits", mir.replace("mvalue|0|0|1|4591870180066957722|", "mvalue|0|0|1|4591870180066957723|", 1))
write("nonfinite_const", mir.replace("mvalue|0|0|1|4591870180066957722|3|", "mvalue|0|0|1|9218868437227405313|6|", 1))
write("value_provenance", mir.replace("mvalue|2|0|0|0|-1|0|0|0|0|0|0|-1|2|2", "mvalue|2|0|0|0|-1|0|0|0|0|0|0|-1|1|2", 1))
write("prov_source_id", mir.replace("mprov|2|5|-1|2|-1|-1|2", "mprov|2|5|-1|2|-1|-1|1", 1))
write("prov_span", mir.replace("mprov|2|5|-1|2|-1|-1|2", "mprov|2|6|-1|2|-1|-1|2", 1))
write("prov_transform", mir.replace("mprov|2|5|-1|2|-1|-1|2", "mprov|2|5|-1|3|-1|-1|2", 1))
write("block_first", mir.replace("mblock|0|0|4|0|1", "mblock|0|1|4|0|1", 1))
write("block_count", mir.replace("mblock|0|0|4|0|1", "mblock|0|0|3|0|1", 1))
write("terminator", mir.replace("mblock|0|0|4|0|1", "mblock|0|0|4|1|1", 1))
write("terminator_tick", mir.replace("mblock|0|0|4|0|1", "mblock|0|0|4|0|0", 1))
write("const_result_oob", mir.replace("minstr|0|20|0|0|", "minstr|0|20|99|0|", 1))
write("opcode", mir.replace("minstr|2|21|", "minstr|2|99|", 1))
write("operand", mir.replace("minstr|2|21|2|0|0|1|", "minstr|2|21|2|0|0|99|", 1))
write("effect", mir.replace("minstr|2|21|2|0|0|1|0|1|", "minstr|2|21|2|0|0|1|1|1|", 1))
write("trap", mir.replace("minstr|2|21|2|0|0|1|0|1|", "minstr|2|21|2|0|0|1|0|0|", 1))
write("policy", mir.replace("minstr|3|26|-1|-1|2|-1|1|0|0|", "minstr|3|26|-1|-1|2|-1|1|0|1|", 1))
write("source_op", mir.replace("minstr|2|21|2|0|0|1|0|1|-1|2|1", "minstr|2|21|2|0|0|1|0|1|-1|1|1", 1))
write("semantic_tick", mir.replace("minstr|2|21|2|0|0|1|0|1|-1|2|1", "minstr|2|21|2|0|0|1|0|1|-1|2|2", 1))
write("observation_source", mir.replace("mobs|0|v2_add|0|0|0", "mobs|0|v2_add|0|0|1", 1))
write("footer", mir.replace("mend|1|3|3|1|4|1", "mend|1|3|3|1|3|1", 1))
write("reordered", mir.replace("mblock|0|0|4|0|1\n", "").replace("mend|", "mblock|0|0|4|0|1\nmend|", 1))
write("noncanonical", mir.replace("|2|12|", "|2|012|", 1))
write("crlf", mir.replace("\n", "\r\n"))

# Keep the ENIR beside the tamper set for direct relation checks.
(out / "source.enir").write_text(enir, encoding="ascii")
PY

artifact_tamper_count=0
for artifact in "$TMP_DIR"/tampers/*.emir; do
  artifact_tamper_count=$((artifact_tamper_count + 1))
  verify_ok=0
  relation_ok=0
  if "$DRIVER" verify-mir "$artifact" >"$artifact.verify.out" 2>&1; then verify_ok=1; fi
  if "$DRIVER" validate-mir "$TMP_DIR/tampers/source.enir" "$artifact" >"$artifact.relation.out" 2>&1; then relation_ok=1; fi
  if [[ "$verify_ok" == "1" && "$relation_ok" == "1" ]]; then
    fail "both MIR verifier and relational validator accepted tamper: $(basename "$artifact")"
  fi
done
[[ "$artifact_tamper_count" == "30" ]] || fail "artifact tamper count drifted: $artifact_tamper_count"

# A valid MIR for a different ENIR graph must fail the source-hash relation.
if "$DRIVER" validate-mir "$TMP_DIR/cases/v2_sub.enir" "$TMP_DIR/cases/v2_add.emir" >"$TMP_DIR/cross-source.out" 2>&1; then
  fail "relational validator accepted a MIR bound to a different ENIR graph"
fi
grep -Fq 'mir-relation-error|' "$TMP_DIR/cross-source.out" || fail "cross-source rejection lacks relational error"

sed 's/let x = 0.1/let x = 0.2/' tools/eisa/eisa_enir_v2_add.eisa >"$TMP_DIR/v2_add_same_name_tampered.eisa"
cmp -s tools/eisa/eisa_enir_v2_add.eisa "$TMP_DIR/v2_add_same_name_tampered.eisa" && fail "same-name source tamper changed no bytes"
"$DRIVER" lower-v2 "$TMP_DIR/v2_add_same_name_tampered.eisa" >"$TMP_DIR/v2_add_same_name_tampered.enir"
if "$DRIVER" validate-mir "$TMP_DIR/v2_add_same_name_tampered.enir" "$TMP_DIR/cases/v2_add.emir" >"$TMP_DIR/same-name-cross-hash.out" 2>&1; then
  fail "relational validator accepted same-name ENIR content/hash drift"
fi
grep -Fq 'mir-relation-error|' "$TMP_DIR/same-name-cross-hash.out" || fail "same-name hash rejection lacks relational error"

runtime_tamper_count=0
for field in source_enir_hash value_bits error0_bits error1_bits error2_bits error3_bits uncertainty_bits status; do
  wrapper="$TMP_DIR/tamper-$field"
  cat >"$wrapper" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" == "run-mir" ]]; then
  "$DRIVER" "\$@" | sed "/^mir-exec|/s/$field=[^|]*/$field=777/"
else
  exec "$DRIVER" "\$@"
fi
EOF
  chmod +x "$wrapper"
  runtime_tamper_count=$((runtime_tamper_count + 1))
  if python3 scripts/dev/madaros_v2_e3a_enir_mir_qd128_verify.py \
      --driver "$wrapper" --source-dir tools/eisa --corpus tools/eisa/eisa_evm_run.sio \
      --oracle "$ORACLE_OUT" --out-dir "$TMP_DIR/runtime-$field" \
      --receipt "$TMP_DIR/runtime-$field.json" --root "$ROOT_DIR" >"$TMP_DIR/runtime-$field.log" 2>&1; then
    fail "independent checker accepted MIR runtime $field tamper"
  fi
done
for field in mir_hash fuel_left executed_instrs; do
  wrapper="$TMP_DIR/tamper-final-$field"
  cat >"$wrapper" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" == "run-mir" ]]; then
  "$DRIVER" "\$@" | sed "/^mir-exec-ok|/s/$field=[^|]*/$field=777/"
else
  exec "$DRIVER" "\$@"
fi
EOF
  chmod +x "$wrapper"
  runtime_tamper_count=$((runtime_tamper_count + 1))
  if python3 scripts/dev/madaros_v2_e3a_enir_mir_qd128_verify.py \
      --driver "$wrapper" --source-dir tools/eisa --corpus tools/eisa/eisa_evm_run.sio \
      --oracle "$ORACLE_OUT" --out-dir "$TMP_DIR/runtime-final-$field" \
      --receipt "$TMP_DIR/runtime-final-$field.json" --root "$ROOT_DIR" >"$TMP_DIR/runtime-final-$field.log" 2>&1; then
    fail "independent checker accepted MIR final runtime $field tamper"
  fi
done
[[ "$runtime_tamper_count" == "11" ]] || fail "runtime tamper count drifted: $runtime_tamper_count"

mkdir -p "$TMP_DIR/source-tamper"
cp tools/eisa/eisa_enir_v2_{const_gate,add,sub,mul,div,sqrt}.eisa "$TMP_DIR/source-tamper/"
sed -i 's/let x = 0.1/let x = 0.2/' "$TMP_DIR/source-tamper/eisa_enir_v2_add.eisa"
cmp -s tools/eisa/eisa_enir_v2_add.eisa "$TMP_DIR/source-tamper/eisa_enir_v2_add.eisa" && fail "causal source tamper transform changed no bytes"
if python3 scripts/dev/madaros_v2_e3a_enir_mir_qd128_verify.py \
    --driver "$DRIVER" --source-dir "$TMP_DIR/source-tamper" --corpus tools/eisa/eisa_evm_run.sio \
    --oracle "$ORACLE_OUT" --out-dir "$TMP_DIR/source-tamper-cases" \
    --receipt "$TMP_DIR/source-tamper.json" --root "$ROOT_DIR" >"$TMP_DIR/source-tamper.log" 2>&1; then
  fail "causal source tamper passed MIR/METRON validation"
fi

E2H_BASE_REF=HEAD bash scripts/dev/madaros_v2_e2h_enir_memory_move_poison_gate.sh >"$TMP_DIR/e2h-regression.log"
grep -Fq 'E2H_ENIR_MEMORY_MOVE_POISON_FULL_GATE_PASS' "$TMP_DIR/e2h-regression.log" || fail "E2H-through-E1 regression chain failed"

if [[ -n "$KEEP_DIR" ]]; then
  mkdir -p "$KEEP_DIR"
  cp "$RECEIPT" "$KEEP_DIR/e3a-enir-mir-qd128.receipt.json"
  cp "$TMP_DIR"/cases/*.emir "$KEEP_DIR/"
fi

RECEIPT_SHA="$(sha256sum "$RECEIPT" | cut -d' ' -f1)"
echo "E3A_ENIR_MIR_QD128_ARITHMETIC_FULL_GATE_PASS receipt_sha256=$RECEIPT_SHA programs=6 observations=6 unsupported_enir=$unsupported_count artifact_tampers=$artifact_tamper_count runtime_tampers=$runtime_tamper_count poison=explicit source_hash_binding=cross-name+same-name relation=native+independent execution=enir==mir==metron semantic_ticks=exact abi=independent machine_ir=unused memory=e3b-separate cfg=single-block source_tamper=pass e2h_regression=pass e2g_regression=pass e2f_regression=pass e2e_regression=pass e2d_regression=pass e2c_regression=pass e2b_regression=pass e2a_regression=pass e1_regression=pass codegen_diff=0"
