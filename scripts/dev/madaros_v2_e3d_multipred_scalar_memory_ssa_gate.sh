#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${E3D_BASE_REF:-}"
if [[ -z "$BASE_REF" ]]; then
  BASE_REF="$(git merge-base HEAD "${ENIR_GATE_BASE:-origin/main}")"
fi
SEED="${SOUC_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
KEEP_DIR="${E3D_RECEIPT_DIR:-}"
TMP_DIR="$(mktemp -d /tmp/madaros-e3d-multipred-ssa.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT
DRIVER="$TMP_DIR/madaros-enir"
RECEIPT="$TMP_DIR/e3d-multipred-scalar-memory-ssa.receipt.json"

fail(){ echo "E3D_MULTIPRED_SCALAR_MEMORY_SSA_GATE_FAIL: $*" >&2; exit 1; }

. scripts/dev/madaros_v2_enir_gate_scope.sh

[[ -x "$SEED" ]] || fail "missing executable Stage0 seed: $SEED"
git rev-parse --verify "$BASE_REF" >/dev/null 2>&1 || fail "base ref not found: $BASE_REF"
E3D_PROTECTED=(self-hosted/compiler/main.sio self-hosted/ir self-hosted/native self-hosted/wasm self-hosted/gpu stdlib/runtime stdlib/eisa stdlib/math/dd64.sio stdlib/math/qd128.sio self-hosted/enir/qd.sio self-hosted/enir/mir_cfg.sio tools/eisa/eisa_evm_run.sio)
madaros_v2_enir_gate_scope_or_skip "$BASE_REF" "E3D_MULTIPRED_SCALAR_MEMORY_SSA_GATE" \
  "E3D changed production codegen/ABI/runtime, frozen E3C MIR, pinned qd semantics, or METRON oracle" "${E3D_PROTECTED[@]}"

scripts/dev/souc-build-lock.sh "$SEED" self-hosted/enir/driver.sio "$DRIVER" >"$TMP_DIR/driver-build.log" 2>&1
[[ -s "$DRIVER" ]] || fail "source-fresh ENIR/Join-MIR driver build produced no ELF"
if grep -Eq '^error:|unknown identifier|typecheck: failed|assignment type mismatch|private struct field|string literal too long' "$TMP_DIR/driver-build.log"; then tail -120 "$TMP_DIR/driver-build.log" >&2; fail "Stage0 reported diagnostics while building Join-MIR driver"; fi
chmod +x "$DRIVER"

python3 scripts/dev/madaros_v2_e3d_multipred_scalar_memory_ssa_verify.py \
  --driver "$DRIVER" --source-dir tools/eisa --out-dir "$TMP_DIR/cases" --receipt "$RECEIPT" --root "$ROOT_DIR"

mkdir -p "$TMP_DIR/source-negative"
python3 - "$TMP_DIR/source-negative" <<'PY'
from pathlib import Path
import sys

out = Path(sys.argv[1])
base = '''epistemic fn bad() {
fuel 60
let cond=0
let x=2
let y=3
let a=10
let b=100
if_zero cond == 0.0 {
store [m0] <- a
store [m1] <- b
yield x
} else {
store [m0] <- b
store [m1] <- a
yield y
}
join selected
let v0=load [m0]
let v1=load [m1]
let out=selected+v0
gate out
}'''
cases = {
    "missing_else": base.replace("} else {", "}"),
    "missing_then_yield": base.replace("yield x\n", "", 1),
    "missing_else_yield": base.replace("yield y\n", "", 1),
    "one_slot": base.replace("store [m1] <- b\n", "", 1),
    "third_slot": base.replace("store [m1] <- b", "store [m1] <- b\nstore [m2] <- a", 1),
    "duplicate_then_store": base.replace("store [m1] <- b", "store [m0] <- b", 1),
    "asymmetric_slot": base.replace("store [m1] <- a", "store [m0] <- a", 1),
    "load_in_arm": base.replace("store [m0] <- a", "let early=load [m0]\nstore [m0] <- a", 1),
    "store_after_join": base.replace("join selected", "join selected\nstore [late] <- a", 1),
    "low_fuel": base.replace("fuel 60", "fuel 5", 1),
    "unknown_condition": base.replace("if_zero cond", "if_zero absent", 1),
    "let_in_arm": base.replace("store [m0] <- a", "let local=1\nstore [m0] <- a", 1),
}
for name, source in cases.items():
    if source == base:
        raise SystemExit(f"negative fixture did not change: {name}")
    (out / f"{name}.eisa").write_text(source + "\n", encoding="ascii")
PY

source_negative_count=0
for source in "$TMP_DIR"/source-negative/*.eisa; do
  source_negative_count=$((source_negative_count+1))
  if "$DRIVER" lower-join-v2 "$source" >"$source.out" 2>&1; then fail "source lowerer accepted E3D negative: $(basename "$source")"; fi
  grep -Fq 'enir-lower-error|' "$source.out" || fail "source negative lacks classified lowering error"
done
[[ "$source_negative_count" == 12 ]] || fail "source negative count drifted: $source_negative_count"

python3 - "$TMP_DIR/cases/v2_join_then.jmir" "$TMP_DIR/cases/v2_join_then.enir" "$TMP_DIR/tampers" <<'PY'
from pathlib import Path
import sys

jmir=Path(sys.argv[1]).read_text(encoding="ascii");enir=Path(sys.argv[2]).read_text(encoding="ascii");out=Path(sys.argv[3]);out.mkdir()
def rows(text): return [line.split("|") for line in text.splitlines()]
def changed(tag, occurrence, field, value):
    data=rows(jmir); selected=[row for row in data if row[0]==tag][occurrence]; selected[field]=str(value); return "\n".join("|".join(row) for row in data)+"\n"
def write(name,text):
    (out/f"{name}.jmir").write_text(text,encoding="ascii");(out/f"{name}.enir").write_text(enir,encoding="ascii")

cases=[
 ("schema","jmir",0,1,3),("stage","jmir",0,2,3),("profile","jmir",0,4,1),("fuel","jmir",0,5,2),("source_hash","jmir",0,6,1),
 ("type","jtype",0,3,1),("value_bits","jvalue",1,4,1),("value_status","jvalue",1,11,1),("value_source","jvalue",1,14,2),("prov_origin","jprov",7,3,2),("prov_transform","jprov",7,4,9),
 ("block_id","jblock",1,1,2),("block_first","jblock",2,2,8),("block_count","jblock",2,3,3),("block_arg_start","jblock",3,4,1),("block_arg_count","jblock",3,5,0),("block_term","jblock",0,6,1),("block_condition","jblock",0,7,2),("block_edge0","jblock",0,8,2),("block_edge1","jblock",0,9,3),("block_tick","jblock",0,10,0),("block_source","jblock",1,11,0),
 ("barg_block","jbarg",0,2,2),("barg_value","jbarg",0,4,8),("barg_source","jbarg",0,7,1),
 ("edge_from","jedge",2,2,0),("edge_to","jedge",2,3,2),("edge_arg_count","jedge",2,4,0),("edge_arg","jedge",2,5,2),("edge_source","jedge",2,9,1),
 ("sphi_block","jsphi",0,2,2),("sphi_result","jsphi",0,3,8),("sphi_edge0","jsphi",0,4,0),("sphi_value0","jsphi",0,5,2),("sphi_edge1","jsphi",0,6,1),("sphi_value1","jsphi",0,7,1),("sphi_source","jsphi",0,8,1),
 ("version_id","jmver",0,1,3),("version_slot","jmver",1,2,0),("version_kind","jmver",0,3,1),("version_block","jmver",2,4,1),("version_def","jmver",2,5,8),("version_source","jmver",2,6,8),
 ("mphi0_block","jmphi",0,2,2),("mphi0_slot","jmphi",0,3,1),("mphi0_result","jmphi",0,4,5),("mphi0_edge0","jmphi",0,5,0),("mphi0_version0","jmphi",0,6,1),("mphi0_edge1","jmphi",0,7,1),("mphi0_version1","jmphi",0,8,3),("mphi0_source","jmphi",0,9,2),
 ("mphi1_slot","jmphi",1,3,0),("mphi1_result","jmphi",1,4,4),("mphi1_version0","jmphi",1,6,0),("mphi1_version1","jmphi",1,8,2),
 ("store0_effect","jinstr",7,7,0),("store0_in","jinstr",7,13,4),("store0_out","jinstr",7,14,1),("store1_slot","jinstr",8,12,0),("store2_out","jinstr",9,14,3),("load0_in","jinstr",11,13,5),("load1_slot","jinstr",12,12,0),("source_op","jinstr",17,10,16),("tick","jinstr",17,11,2),
 ("observation","jobs",0,5,1),("footer_scalar","jend",0,7,0),("footer_versions","jend",0,8,5),("footer_phis","jend",0,9,1),
]
for name,tag,occurrence,field,value in cases: write(name,changed(tag,occurrence,field,value))
data=rows(jmir);const=[row for row in data if row[0]=="jvalue"][1];const[4]="9218868437227405312";const[5]="4";write("nonfinite_const","\n".join("|".join(row) for row in data)+"\n")
write("noncanonical",jmir.replace("|4|4|","|04|4|",1));write("reordered",jmir.replace("jblock|0|","TMP|0|",1).replace("jend|","jblock|0|",1).replace("TMP|0|","jend|",1));write("crlf",jmir.replace("\n","\r\n"))
PY

artifact_tamper_count=0
for artifact in "$TMP_DIR"/tampers/*.jmir; do
  artifact_tamper_count=$((artifact_tamper_count+1)); source="${artifact%.jmir}.enir"; verify_ok=0; relation_ok=0
  if "$DRIVER" verify-join-mir "$artifact" >"$artifact.verify" 2>&1; then verify_ok=1; fi
  if "$DRIVER" validate-join-mir "$source" "$artifact" >"$artifact.relation" 2>&1; then relation_ok=1; fi
  if [[ "$verify_ok" == 1 && "$relation_ok" == 1 ]]; then fail "both Join-MIR verifier and relation accepted tamper: $(basename "$artifact")"; fi
done
[[ "$artifact_tamper_count" == 72 ]] || fail "artifact tamper count drifted: $artifact_tamper_count"

if "$DRIVER" validate-join-mir "$TMP_DIR/cases/v2_join_else.enir" "$TMP_DIR/cases/v2_join_then.jmir" >"$TMP_DIR/cross-source.out" 2>&1; then fail "relation accepted a different same-shape source"; fi

runtime_tamper_count=0
run_runtime_tamper(){
  local family="$1" field="$2" selector="$3"
  local wrapper="$TMP_DIR/tamper-$family-$field"
  cat >"$wrapper" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" == "run-join-mir" ]]; then "$DRIVER" "\$@" | sed "/^$selector|/s/$field=[^|]*/$field=777/"; else exec "$DRIVER" "\$@"; fi
EOF
  chmod +x "$wrapper"; runtime_tamper_count=$((runtime_tamper_count+1))
  if python3 scripts/dev/madaros_v2_e3d_multipred_scalar_memory_ssa_verify.py --driver "$wrapper" --source-dir tools/eisa --out-dir "$TMP_DIR/runtime-$family-$field" --receipt "$TMP_DIR/runtime-$family-$field.json" --root "$ROOT_DIR" >"$TMP_DIR/runtime-$family-$field.log" 2>&1; then fail "independent checker accepted $family $field tamper"; fi
}
for field in block source_block term condition edge source_edge to taken poisoned frail; do run_runtime_tamper control "$field" jmir-control; done
for field in source_enir_hash phi block edge incoming_value result_value; do run_runtime_tamper scalar "$field" jmir-scalar-phi; done
for field in source_enir_hash phi block slot edge incoming_version result_version; do run_runtime_tamper memory_phi "$field" jmir-memory-phi; done
for field in source_enir_hash ordinal site source_op value_id value_bits error0_bits error1_bits error2_bits error3_bits uncertainty_bits status branch_poisoned frail_branches source_span; do run_runtime_tamper observation "$field" jmir-exec; done
for field in source_enir_hash slot version site source_op value_bits error0_bits error1_bits error2_bits error3_bits uncertainty_bits status; do run_runtime_tamper memory "$field" jmir-memory; done
for field in jmir_hash source_enir_hash executed_instrs observations fuel_initial fuel_left stop_kind last_write branch_poisoned frail_branches; do run_runtime_tamper final "$field" jmir-exec-ok; done
[[ "$runtime_tamper_count" == 60 ]] || fail "runtime tamper count drifted: $runtime_tamper_count"

mkdir -p "$TMP_DIR/source-tamper"
cp tools/eisa/eisa_enir_v2_join_then.eisa "$TMP_DIR/source-tamper/"
cp tools/eisa/eisa_enir_v2_join_else.eisa "$TMP_DIR/source-tamper/"
sed -i 's/let a_then=10/let a_then=11/' "$TMP_DIR/source-tamper/eisa_enir_v2_join_then.eisa"
python3 scripts/dev/madaros_v2_e3d_multipred_scalar_memory_ssa_verify.py --driver "$DRIVER" --source-dir "$TMP_DIR/source-tamper" --out-dir "$TMP_DIR/source-tamper-out" --receipt "$TMP_DIR/source-tamper.json" --root "$ROOT_DIR" >"$TMP_DIR/source-tamper.log"
[[ "$(sha256sum "$RECEIPT" | cut -d' ' -f1)" != "$(sha256sum "$TMP_DIR/source-tamper.json" | cut -d' ' -f1)" ]] || fail "causal source tamper left receipt unchanged"

E3C_BASE_REF=HEAD E3C_ALLOW_DOWNSTREAM_ENIR_EXTENSION=1 bash scripts/dev/madaros_v2_e3c_cfg_memory_ssa_gate.sh >"$TMP_DIR/e3c-regression.log"
grep -Fq 'E3C_CFG_MEMORY_SSA_FULL_GATE_PASS' "$TMP_DIR/e3c-regression.log" || fail "E3C-through-E1 regression failed"

if [[ -n "$KEEP_DIR" ]]; then mkdir -p "$KEEP_DIR"; cp "$RECEIPT" "$KEEP_DIR/e3d-multipred-scalar-memory-ssa.receipt.json"; cp "$TMP_DIR"/cases/*.jmir "$KEEP_DIR/"; fi
RECEIPT_SHA="$(sha256sum "$RECEIPT" | cut -d' ' -f1)"
echo "E3D_MULTIPRED_SCALAR_MEMORY_SSA_FULL_GATE_PASS receipt_sha256=$RECEIPT_SHA programs=2 observations=2 source_negatives=$source_negative_count artifact_tampers=$artifact_tamper_count runtime_tampers=$runtime_tamper_count cfg=acyclic-diamond predecessors=2 scalar_phi=explicit memory_slots=2 memory_versions=6 memory_phis=2 paths=then+else relation=native+independent execution=source==enir==jmir==independent source_tamper=pass alias_analysis=deferred abi=independent machine_ir=unused e3c_regression=pass codegen_diff=0"
