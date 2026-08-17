#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${E3B_BASE_REF:-}"
if [[ -z "$BASE_REF" ]]; then
  BASE_REF="$(git merge-base HEAD "${ENIR_GATE_BASE:-origin/main}")"
fi
SEED="${SOUC_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
KEEP_DIR="${E3B_RECEIPT_DIR:-}"
TMP_DIR="$(mktemp -d /tmp/madaros-e3b-enir-mir-memory.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

DRIVER="$TMP_DIR/madaros-enir"
ORACLE="$TMP_DIR/eisa-metron-corpus"
ORACLE_OUT="$TMP_DIR/eisa-metron-corpus.out"
RECEIPT="$TMP_DIR/e3b-enir-mir-memory.receipt.json"

fail() {
  echo "E3B_ENIR_MIR_MEMORY_GATE_FAIL: $*" >&2
  exit 1
}

. scripts/dev/madaros_v2_enir_gate_scope.sh

[[ -x "$SEED" ]] || fail "missing executable Stage0 seed: $SEED"
git rev-parse --verify "$BASE_REF" >/dev/null 2>&1 || fail "base ref not found: $BASE_REF"
E3B_PROTECTED=(self-hosted/compiler/main.sio self-hosted/ir self-hosted/native self-hosted/wasm self-hosted/gpu stdlib/runtime stdlib/eisa stdlib/math/dd64.sio stdlib/math/qd128.sio self-hosted/enir/qd.sio tools/eisa/eisa_evm_run.sio)
if [[ "${E3B_ALLOW_DOWNSTREAM_ENIR_EXTENSION:-0}" != "1" ]]; then E3B_PROTECTED+=(self-hosted/enir/ir.sio self-hosted/enir/interpreter.sio self-hosted/enir/source_lower.sio); fi
madaros_v2_enir_gate_scope_or_skip "$BASE_REF" "E3B_ENIR_MIR_MEMORY_GATE" \
  "E3B changed protected production, ENIR, qd, or oracle surfaces" "${E3B_PROTECTED[@]}"

scripts/dev/souc-build-lock.sh "$SEED" self-hosted/enir/driver.sio "$DRIVER" >"$TMP_DIR/driver-build.log" 2>&1
[[ -s "$DRIVER" ]] || fail "ENIR/MIR driver build produced no ELF"
if grep -Eq '^error:|unknown identifier|typecheck: failed|assignment type mismatch|private struct field' "$TMP_DIR/driver-build.log"; then
  tail -100 "$TMP_DIR/driver-build.log" >&2
  fail "Stage0 reported diagnostics while building ENIR/MIR driver"
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

python3 scripts/dev/madaros_v2_e3b_enir_mir_memory_verify.py \
  --driver "$DRIVER" --source-dir tools/eisa --corpus tools/eisa/eisa_evm_run.sio \
  --oracle "$ORACLE_OUT" --out-dir "$TMP_DIR/cases" --receipt "$RECEIPT" --root "$ROOT_DIR"

# The load must name and consume the latest store, not merely any store to the slot.
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
"$DRIVER" lower-mir "$TMP_DIR/two-store.enir" >"$TMP_DIR/two-store.emir"
"$DRIVER" validate-mir "$TMP_DIR/two-store.enir" "$TMP_DIR/two-store.emir" >"$TMP_DIR/two-store.relation"
grep -Fq 'minstr|4|27|2|0|-1|-1|2|0|-1|4|1|0|3' "$TMP_DIR/two-store.emir" || fail "MIR load does not name latest dominating store site"
"$DRIVER" run-mir "$TMP_DIR/two-store.emir" >"$TMP_DIR/two-store.out"
grep -Fq 'value_bits=4620974692658839552' "$TMP_DIR/two-store.out" || fail "latest store did not determine MIR loaded value"
grep -Fq 'mir-memory|schema=2|stage=3|module=two_store|' "$TMP_DIR/two-store.out" || fail "MIR latest-store receipt missing"
grep -Fq '|slot=0|site=3|source_op=3|value_bits=4620974692658839552|' "$TMP_DIR/two-store.out" || fail "MIR latest-store receipt has wrong site/product"

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
"$DRIVER" lower-mir "$TMP_DIR/negative-zero-memory.enir" >"$TMP_DIR/negative-zero-memory.emir"
"$DRIVER" validate-mir "$TMP_DIR/negative-zero-memory.enir" "$TMP_DIR/negative-zero-memory.emir" >/dev/null
"$DRIVER" run-mir "$TMP_DIR/negative-zero-memory.emir" >"$TMP_DIR/negative-zero-memory.out"
[[ "$(grep -c '^mir-exec|.*value_bits=0|' "$TMP_DIR/negative-zero-memory.out")" == "2" ]] || fail "negative zero was not canonical at both MIR gates"
grep -Fq 'mir-memory|schema=2|stage=3|module=negative_zero_memory|' "$TMP_DIR/negative-zero-memory.out" || fail "negative-zero MIR memory receipt missing"

python3 - "$TMP_DIR/cases/v2_mem.emir" "$TMP_DIR/cases/v2_mem.enir" \
  "$TMP_DIR/cases/v2_emov.emir" "$TMP_DIR/cases/v2_emov.enir" \
  "$TMP_DIR/cases/v2_mem_poison.emir" "$TMP_DIR/cases/v2_mem_poison.enir" "$TMP_DIR/tampers" <<'PY'
from pathlib import Path
import sys

mem, mem_enir, move, move_enir, poison, poison_enir = [Path(p).read_text(encoding="ascii") for p in sys.argv[1:7]]
out = Path(sys.argv[7]); out.mkdir()

def rows(text):
    return [line.split("|") for line in text.splitlines()]

def changed(text, tag, occurrence, field, value):
    data = rows(text)
    selected = [row for row in data if row[0] == tag][occurrence]
    selected[field] = str(value)
    return "\n".join("|".join(row) for row in data) + "\n"

def write(name, text, source):
    (out / f"{name}.emir").write_text(text, encoding="ascii")
    (out / f"{name}.enir").write_text(source, encoding="ascii")

write("schema", changed(mem, "emir", 0, 1, 1), mem_enir)
write("stage", changed(mem, "emir", 0, 2, 2), mem_enir)
write("profile", changed(mem, "emir", 0, 4, 1), mem_enir)
write("fuel", changed(mem, "emir", 0, 5, 3), mem_enir)
write("source_hash", changed(mem, "emir", 0, 6, 1), mem_enir)
write("type", changed(mem, "mtype", 0, 3, 1), mem_enir)
write("value_bits", changed(mem, "mvalue", 0, 4, 1), mem_enir)
for index, name in enumerate(("error0", "error1", "error2", "error3"), 6):
    write(name, changed(poison, "mvalue", 4, index, 1), poison_enir)
write("uncertainty", changed(poison, "mvalue", 4, 10, 1), poison_enir)
write("status", changed(poison, "mvalue", 4, 11, 1), poison_enir)
write("value_source_id", changed(mem, "mvalue", 1, 14, 0), mem_enir)
write("load_provenance_origin", changed(mem, "mprov", 1, 3, 0), mem_enir)
write("load_provenance_transform", changed(mem, "mprov", 1, 4, 9), mem_enir)
write("block_first", changed(mem, "mblock", 0, 2, 1), mem_enir)
write("block_count", changed(mem, "mblock", 0, 3, 3), mem_enir)
write("store_result", changed(mem, "minstr", 1, 3, 1), mem_enir)
write("store_operand", changed(mem, "minstr", 1, 5, 99), mem_enir)
write("store_effect", changed(mem, "minstr", 1, 7, 0), mem_enir)
write("store_slot", changed(mem, "minstr", 1, 12, 16), mem_enir)
write("store_origin", changed(mem, "minstr", 1, 13, 0), mem_enir)
write("load_result", changed(mem, "minstr", 2, 3, 0), mem_enir)
write("load_effect", changed(mem, "minstr", 2, 7, 0), mem_enir)
write("load_slot", changed(mem, "minstr", 2, 12, 1), mem_enir)
write("load_origin_stale", changed(mem, "minstr", 2, 13, 0), mem_enir)
write("load_origin_future", changed(mem, "minstr", 2, 13, 3), mem_enir)
write("load_source_op", changed(mem, "minstr", 2, 10, 1), mem_enir)
write("move_operand", changed(move, "minstr", 4, 5, 99), move_enir)
write("move_effect", changed(move, "minstr", 4, 7, 2), move_enir)
write("move_slot", changed(move, "minstr", 4, 12, 0), move_enir)
write("move_origin", changed(move, "minstr", 4, 13, 3), move_enir)
write("observation", changed(mem, "mobs", 0, 5, 1), mem_enir)
write("footer", changed(mem, "mend", 0, 5, 3), mem_enir)
write("noncanonical", mem.replace("|2|12|", "|2|012|", 1), mem_enir)
write("reordered", mem.replace("mblock|0|0|4|0|1\n", "").replace("mend|", "mblock|0|0|4|0|1\nmend|", 1), mem_enir)
write("crlf", mem.replace("\n", "\r\n"), mem_enir)
PY

artifact_tamper_count=0
for artifact in "$TMP_DIR"/tampers/*.emir; do
  artifact_tamper_count=$((artifact_tamper_count + 1))
  source="${artifact%.emir}.enir"
  verify_ok=0
  relation_ok=0
  if "$DRIVER" verify-mir "$artifact" >"$artifact.verify.out" 2>&1; then verify_ok=1; fi
  if "$DRIVER" validate-mir "$source" "$artifact" >"$artifact.relation.out" 2>&1; then relation_ok=1; fi
  if [[ "$verify_ok" == "1" && "$relation_ok" == "1" ]]; then
    fail "both MIR verifier and relational validator accepted tamper: $(basename "$artifact")"
  fi
done
[[ "$artifact_tamper_count" == "38" ]] || fail "artifact tamper count drifted: $artifact_tamper_count"

if "$DRIVER" validate-mir "$TMP_DIR/cases/v2_emov.enir" "$TMP_DIR/cases/v2_mem.emir" >"$TMP_DIR/cross-source.out" 2>&1; then
  fail "relational validator accepted MIR bound to a different ENIR graph"
fi
sed 's/let x=7.25/let x=7.5/' tools/eisa/eisa_enir_v2_mem.eisa >"$TMP_DIR/v2_mem_same_name_tampered.eisa"
"$DRIVER" lower-v2 "$TMP_DIR/v2_mem_same_name_tampered.eisa" >"$TMP_DIR/v2_mem_same_name_tampered.enir"
if "$DRIVER" validate-mir "$TMP_DIR/v2_mem_same_name_tampered.enir" "$TMP_DIR/cases/v2_mem.emir" >"$TMP_DIR/same-name.out" 2>&1; then
  fail "relational validator accepted same-name ENIR hash drift"
fi

runtime_tamper_count=0
for field in source_enir_hash slot site source_op value_bits error0_bits error1_bits error2_bits error3_bits uncertainty_bits status; do
  wrapper="$TMP_DIR/tamper-memory-$field"
  cat >"$wrapper" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" == "run-mir" ]]; then
  "$DRIVER" "\$@" | sed "/^mir-memory|/s/$field=[^|]*/$field=777/"
else
  exec "$DRIVER" "\$@"
fi
EOF
  chmod +x "$wrapper"
  runtime_tamper_count=$((runtime_tamper_count + 1))
  if python3 scripts/dev/madaros_v2_e3b_enir_mir_memory_verify.py \
      --driver "$wrapper" --source-dir tools/eisa --corpus tools/eisa/eisa_evm_run.sio \
      --oracle "$ORACLE_OUT" --out-dir "$TMP_DIR/runtime-memory-$field" \
      --receipt "$TMP_DIR/runtime-memory-$field.json" --root "$ROOT_DIR" >"$TMP_DIR/runtime-memory-$field.log" 2>&1; then
    fail "independent checker accepted MIR memory receipt $field tamper"
  fi
done
for field in source_enir_hash ordinal site source_op value_id value_bits error0_bits error1_bits error2_bits error3_bits uncertainty_bits status gate_class source_span; do
  wrapper="$TMP_DIR/tamper-observation-$field"
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
  if python3 scripts/dev/madaros_v2_e3b_enir_mir_memory_verify.py \
      --driver "$wrapper" --source-dir tools/eisa --corpus tools/eisa/eisa_evm_run.sio \
      --oracle "$ORACLE_OUT" --out-dir "$TMP_DIR/runtime-observation-$field" \
      --receipt "$TMP_DIR/runtime-observation-$field.json" --root "$ROOT_DIR" >"$TMP_DIR/runtime-observation-$field.log" 2>&1; then
    fail "independent checker accepted MIR observation $field tamper"
  fi
done
for field in mir_hash source_enir_hash executed_instrs observations fuel_initial fuel_left last_write; do
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
  if python3 scripts/dev/madaros_v2_e3b_enir_mir_memory_verify.py \
      --driver "$wrapper" --source-dir tools/eisa --corpus tools/eisa/eisa_evm_run.sio \
      --oracle "$ORACLE_OUT" --out-dir "$TMP_DIR/runtime-final-$field" \
      --receipt "$TMP_DIR/runtime-final-$field.json" --root "$ROOT_DIR" >"$TMP_DIR/runtime-final-$field.log" 2>&1; then
    fail "independent checker accepted MIR final receipt $field tamper"
  fi
done
[[ "$runtime_tamper_count" == "32" ]] || fail "runtime tamper count drifted: $runtime_tamper_count"

mkdir -p "$TMP_DIR/source-tamper"
cp tools/eisa/eisa_enir_v2_{mem,emov,mem_poison}.eisa "$TMP_DIR/source-tamper/"
sed -i 's/let x=7.25/let x=7.5/' "$TMP_DIR/source-tamper/eisa_enir_v2_mem.eisa"
if python3 scripts/dev/madaros_v2_e3b_enir_mir_memory_verify.py \
    --driver "$DRIVER" --source-dir "$TMP_DIR/source-tamper" --corpus tools/eisa/eisa_evm_run.sio \
    --oracle "$ORACLE_OUT" --out-dir "$TMP_DIR/source-tamper-cases" \
    --receipt "$TMP_DIR/source-tamper.json" --root "$ROOT_DIR" >"$TMP_DIR/source-tamper.log" 2>&1; then
  fail "causal source tamper passed MIR/METRON validation"
fi

E3A_BASE_REF=HEAD E3A_ALLOW_DOWNSTREAM_ENIR_EXTENSION="${E3B_ALLOW_DOWNSTREAM_ENIR_EXTENSION:-0}" bash scripts/dev/madaros_v2_e3a_enir_mir_qd128_gate.sh >"$TMP_DIR/e3a-regression.log"
grep -Fq 'E3A_ENIR_MIR_QD128_ARITHMETIC_FULL_GATE_PASS' "$TMP_DIR/e3a-regression.log" || fail "E3A-through-E1 regression chain failed"

if [[ -n "$KEEP_DIR" ]]; then
  mkdir -p "$KEEP_DIR"
  cp "$RECEIPT" "$KEEP_DIR/e3b-enir-mir-memory.receipt.json"
  cp "$TMP_DIR"/cases/*.emir "$KEEP_DIR/"
fi

RECEIPT_SHA="$(sha256sum "$RECEIPT" | cut -d' ' -f1)"
echo "E3B_ENIR_MIR_MEMORY_FULL_GATE_PASS receipt_sha256=$RECEIPT_SHA programs=3 observations=4 artifact_tampers=$artifact_tamper_count runtime_tampers=$runtime_tamper_count two_store_dominance=pass source_hash_binding=cross-name+same-name causal_source_tamper=pass relation=native+independent execution=enir==mir==metron memory=atomic-full-product provenance=latest-store move=word-identical negzero=canonical poison=preserved abi=independent machine_ir=unused memory_ssa=deferred cfg=single-block e3a_regression=pass e2h_regression=pass e1_regression=pass codegen_diff=0"
