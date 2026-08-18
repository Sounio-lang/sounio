#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${E3C_BASE_REF:-}"
if [[ -z "$BASE_REF" ]]; then
  BASE_REF="$(git merge-base HEAD "${ENIR_GATE_BASE:-origin/main}")"
fi
SEED="${SOUC_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
KEEP_DIR="${E3C_RECEIPT_DIR:-}"
TMP_DIR="$(mktemp -d /tmp/madaros-e3c-cfg-memory-ssa.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT
DRIVER="$TMP_DIR/madaros-enir"
ORACLE="$TMP_DIR/eisa-metron-corpus"
ORACLE_OUT="$TMP_DIR/eisa-metron-corpus.out"
RECEIPT="$TMP_DIR/e3c-cfg-memory-ssa.receipt.json"

fail(){ echo "E3C_CFG_MEMORY_SSA_GATE_FAIL: $*" >&2; exit 1; }

. scripts/dev/madaros_v2_enir_gate_scope.sh

[[ -x "$SEED" ]] || fail "missing executable Stage0 seed: $SEED"
git rev-parse --verify "$BASE_REF" >/dev/null 2>&1 || fail "base ref not found: $BASE_REF"
E3C_PROTECTED=(self-hosted/compiler/main.sio self-hosted/ir self-hosted/native self-hosted/wasm self-hosted/gpu stdlib/runtime stdlib/eisa stdlib/math/dd64.sio stdlib/math/qd128.sio self-hosted/enir/qd.sio tools/eisa/eisa_evm_run.sio)
if [[ "${E3C_ALLOW_DOWNSTREAM_ENIR_EXTENSION:-0}" != 1 ]]; then E3C_PROTECTED+=(self-hosted/enir/ir.sio self-hosted/enir/interpreter.sio); fi
madaros_v2_enir_gate_scope_or_skip "$BASE_REF" "E3C_CFG_MEMORY_SSA_GATE" \
  "E3C changed production codegen/ABI/runtime, protected ENIR surfaces, pinned qd semantics, or frozen METRON oracle" "${E3C_PROTECTED[@]}"

scripts/dev/souc-build-lock.sh "$SEED" self-hosted/enir/driver.sio "$DRIVER" >"$TMP_DIR/driver-build.log" 2>&1
[[ -s "$DRIVER" ]] || fail "ENIR/CFG-MIR driver build produced no ELF"
if grep -Eq '^error:|unknown identifier|typecheck: failed|assignment type mismatch|private struct field|string literal too long' "$TMP_DIR/driver-build.log"; then tail -120 "$TMP_DIR/driver-build.log" >&2; fail "Stage0 reported diagnostics while building ENIR/CFG-MIR driver"; fi
chmod +x "$DRIVER"
scripts/dev/souc-build-lock.sh "$SEED" tools/eisa/eisa_evm_run.sio "$ORACLE" >"$TMP_DIR/oracle-build.log" 2>&1
[[ -s "$ORACLE" ]] || fail "source-fresh METRON corpus build produced no ELF"
chmod +x "$ORACLE"; "$ORACLE" >"$ORACLE_OUT"
[[ "$(grep -c '^eisa-receipt:' "$ORACLE_OUT")" == "39" ]] || fail "METRON observation manifest drifted"

python3 scripts/dev/madaros_v2_e3c_cfg_memory_ssa_verify.py --driver "$DRIVER" --corpus tools/eisa/eisa_evm_run.sio --oracle "$ORACLE_OUT" --out-dir "$TMP_DIR/cases" --receipt "$RECEIPT" --root "$ROOT_DIR"

# Source-level path negatives: definite initialization is an intersection over
# predecessors, while the declared E3C slice remains one slot and one store per arm.
mkdir -p "$TMP_DIR/source-negative"
cat >"$TMP_DIR/source-negative/body_only_store.eisa" <<'EOF'
epistemic fn body_only_store() {
fuel 30
let count=0
let x=7.25
while count != 0.0 {
store [m] <- x
set count=0
}
let loaded=load [m]
gate loaded
}
EOF
cat >"$TMP_DIR/source-negative/load_before_store.eisa" <<'EOF'
epistemic fn load_before_store() {
fuel 30
let count=0
let loaded=load [m]
store [m] <- count
while count != 0.0 {
set count=0
}
gate loaded
}
EOF
cat >"$TMP_DIR/source-negative/store_after_loop.eisa" <<'EOF'
epistemic fn store_after_loop() {
fuel 30
let count=0
while count != 0.0 {
set count=0
}
store [m] <- count
gate count
}
EOF
source_negative_count=0
for source in "$TMP_DIR"/source-negative/*.eisa; do
  source_negative_count=$((source_negative_count+1))
  if "$DRIVER" lower-v2 "$source" >"$source.out" 2>&1; then fail "source lowerer accepted E3C path negative: $(basename "$source")"; fi
  grep -Fq 'enir-lower-error|' "$source.out" || fail "source negative lacks classified lowering error"
done
[[ "$source_negative_count" == "3" ]] || fail "source negative count drifted"

# Valid ENIR shapes outside the declared Memory-SSA profile must fail closed at E3C.
cat >"$TMP_DIR/entry-only.eisa" <<'EOF'
epistemic fn entry_only() {
fuel 30
let count=0
let x=7.25
store [m] <- x
while count != 0.0 {
set count=0
}
let loaded=load [m]
gate loaded
}
EOF
"$DRIVER" lower-v2 "$TMP_DIR/entry-only.eisa" >"$TMP_DIR/entry-only.enir"
if "$DRIVER" lower-cfg-mir "$TMP_DIR/entry-only.enir" >"$TMP_DIR/entry-only.out" 2>&1; then fail "E3C accepted memory CFG without a backedge store version"; fi
grep -Fq 'cmir-lower-error|' "$TMP_DIR/entry-only.out" || fail "out-of-profile CFG lacks classified lowering error"

python3 - "$TMP_DIR/cases/v2_mem_phi_once.cmir" "$TMP_DIR/cases/v2_mem_phi_once.enir" "$TMP_DIR/tampers" <<'PY'
from pathlib import Path
import sys

cmir=Path(sys.argv[1]).read_text(encoding="ascii");enir=Path(sys.argv[2]).read_text(encoding="ascii");out=Path(sys.argv[3]);out.mkdir()
def rows(text): return [line.split("|") for line in text.splitlines()]
def changed(tag, occurrence, field, value):
    data=rows(cmir); selected=[row for row in data if row[0]==tag][occurrence]; selected[field]=str(value); return "\n".join("|".join(row) for row in data)+"\n"
def write(name,text):
    (out/f"{name}.cmir").write_text(text,encoding="ascii");(out/f"{name}.enir").write_text(enir,encoding="ascii")

cases=[
 ("schema","cmir",0,1,2),("stage","cmir",0,2,3),("profile","cmir",0,4,1),("fuel","cmir",0,5,3),("source_hash","cmir",0,6,1),
 ("type","ctype",0,3,1),("value_bits","cvalue",0,4,1),("value_status","cvalue",0,11,1),("value_source","cvalue",0,14,1),("prov_origin","cprov",9,3,2),("prov_transform","cprov",9,4,9),
 ("block_id","cblock",1,1,2),("block_first","cblock",2,2,4),("block_count","cblock",2,3,3),("block_arg_start","cblock",1,4,1),("block_arg_count","cblock",1,5,1),("block_term","cblock",1,6,0),("block_condition","cblock",1,7,0),("block_edge","cblock",1,8,2),("block_tick","cblock",1,10,0),("block_source","cblock",1,11,0),
 ("barg_value","cbarg",0,4,99),("barg_source","cbarg",0,7,1),
 ("edge_from","cedge",3,2,1),("edge_to","cedge",3,3,3),("edge_arg_count","cedge",3,4,1),("edge_arg","cedge",3,5,99),("edge_source","cedge",3,9,2),
 ("version_id","cmver",0,1,2),("version_slot","cmver",0,2,1),("version_kind","cmver",0,3,1),("version_block","cmver",1,4,1),("version_def","cmver",1,5,3),("version_source","cmver",1,6,3),
 ("phi_block","cmphi",0,2,2),("phi_slot","cmphi",0,3,1),("phi_result","cmphi",0,4,1),("phi_entry_edge","cmphi",0,5,2),("phi_entry_version","cmphi",0,6,1),("phi_back_edge","cmphi",0,7,2),("phi_back_version","cmphi",0,8,0),("phi_source","cmphi",0,9,2),
 ("entry_store_effect","cinstr",2,7,0),("entry_store_trap","cinstr",2,8,1),("entry_store_policy","cinstr",2,9,0),("entry_store_in","cinstr",2,13,2),("entry_store_out","cinstr",2,14,1),("body_store_in","cinstr",4,13,0),("body_store_out","cinstr",4,14,0),("load_in","cinstr",7,13,1),("load_out","cinstr",7,14,2),("source_op","cinstr",7,10,6),("tick","cinstr",7,11,2),
 ("observation","cobs",0,5,1),("footer","cend",0,8,0),
]
for name,tag,occurrence,field,value in cases: write(name,changed(tag,occurrence,field,value))
data=rows(cmir);const=[row for row in data if row[0]=="cvalue"][0];const[4]="9218868437227405312";const[5]="4";write("nonfinite_const","\n".join("|".join(row) for row in data)+"\n")
write("noncanonical",cmir.replace("|2|30|","|2|030|",1));write("reordered",cmir.replace("cblock|0|", "TMP|0|",1).replace("cend|","cblock|0|",1).replace("TMP|0|","cend|",1));write("crlf",cmir.replace("\n","\r\n"))
PY

artifact_tamper_count=0
for artifact in "$TMP_DIR"/tampers/*.cmir; do
  artifact_tamper_count=$((artifact_tamper_count+1)); source="${artifact%.cmir}.enir"; verify_ok=0; relation_ok=0
  if "$DRIVER" verify-cfg-mir "$artifact" >"$artifact.verify" 2>&1; then verify_ok=1; fi
  if "$DRIVER" validate-cfg-mir "$source" "$artifact" >"$artifact.relation" 2>&1; then relation_ok=1; fi
  if [[ "$verify_ok" == 1 && "$relation_ok" == 1 ]]; then fail "both CFG verifier and relation accepted tamper: $(basename "$artifact")"; fi
done
[[ "$artifact_tamper_count" == "59" ]] || fail "artifact tamper count drifted: $artifact_tamper_count"

if "$DRIVER" validate-cfg-mir "$TMP_DIR/cases/v2_mem_phi_zero.enir" "$TMP_DIR/cases/v2_mem_phi_once.cmir" >"$TMP_DIR/cross-source.out" 2>&1; then fail "relation accepted a different same-shape source"; fi

runtime_tamper_count=0
run_runtime_tamper(){
  local family="$1" field="$2" selector="$3"
  local wrapper="$TMP_DIR/tamper-$family-$field"
  cat >"$wrapper" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" == "run-cfg-mir" ]]; then "$DRIVER" "\$@" | sed "/^$selector|/s/$field=[^|]*/$field=777/"; else exec "$DRIVER" "\$@"; fi
EOF
  chmod +x "$wrapper"; runtime_tamper_count=$((runtime_tamper_count+1))
  if python3 scripts/dev/madaros_v2_e3c_cfg_memory_ssa_verify.py --driver "$wrapper" --corpus tools/eisa/eisa_evm_run.sio --oracle "$ORACLE_OUT" --out-dir "$TMP_DIR/runtime-$family-$field" --receipt "$TMP_DIR/runtime-$family-$field.json" --root "$ROOT_DIR" >"$TMP_DIR/runtime-$family-$field.log" 2>&1; then fail "independent checker accepted $family $field tamper"; fi
}
for field in block source_block term condition edge source_edge to taken poisoned frail; do run_runtime_tamper control "$field" cmir-control; done
for field in source_enir_hash phi block slot edge incoming_version result_version; do run_runtime_tamper phi "$field" cmir-memory-phi; done
for field in source_enir_hash ordinal site source_op value_id value_bits error0_bits error1_bits error2_bits error3_bits uncertainty_bits status branch_poisoned frail_branches source_span; do run_runtime_tamper observation "$field" cmir-exec; done
for field in source_enir_hash slot version site source_op value_bits error0_bits error1_bits error2_bits error3_bits uncertainty_bits status; do run_runtime_tamper memory "$field" cmir-memory; done
for field in cmir_hash source_enir_hash executed_instrs observations fuel_initial fuel_left stop_kind last_write branch_poisoned frail_branches; do run_runtime_tamper final "$field" cmir-exec-ok; done
[[ "$runtime_tamper_count" == "54" ]] || fail "runtime tamper count drifted: $runtime_tamper_count"

E3B_BASE_REF=HEAD E3B_ALLOW_DOWNSTREAM_ENIR_EXTENSION=1 bash scripts/dev/madaros_v2_e3b_enir_mir_memory_gate.sh >"$TMP_DIR/e3b-regression.log"
grep -Fq 'E3B_ENIR_MIR_MEMORY_FULL_GATE_PASS' "$TMP_DIR/e3b-regression.log" || fail "E3B-through-E1 regression failed"

if [[ -n "$KEEP_DIR" ]]; then mkdir -p "$KEEP_DIR"; cp "$RECEIPT" "$KEEP_DIR/e3c-cfg-memory-ssa.receipt.json"; cp "$TMP_DIR"/cases/*.cmir "$KEEP_DIR/"; fi
RECEIPT_SHA="$(sha256sum "$RECEIPT"|cut -d' ' -f1)"
echo "E3C_CFG_MEMORY_SSA_FULL_GATE_PASS receipt_sha256=$RECEIPT_SHA programs=5 observations=5 source_negatives=$source_negative_count artifact_tampers=$artifact_tamper_count runtime_tampers=$runtime_tamper_count cfg=explicit-4-block block_args=explicit edges=explicit memory_ssa=store-versions+loop-header-phi zero_trip=pass backedge=pass relation=native+independent execution=enir==cmir==independent metron=e2g-exact abi=independent machine_ir=unused irreducible_cfg=deferred alias_analysis=deferred e3b_regression=pass codegen_diff=0"
