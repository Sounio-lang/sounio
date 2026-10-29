#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=$1
RAW=$2
mkdir "$OUT"
export SOUNIO_STDLIB_PATH=${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}
NODE=${NODE_BIN:-node}
CXX=${CXX:-g++}
finish(){ r=$?;trap - EXIT;for manifest in INPUTS COMPILER EXECUTABLES;do if [[ -s "$OUT/$manifest.sha256" ]];then sha256sum -c "$OUT/$manifest.sha256" >"$OUT/$manifest.final.verify" 2>&1||r=1;fi;done;echo "$r" >"$OUT/STATUS";exit "$r"; }
trap finish EXIT
sha256sum "$ROOT/tests/fixtures/wave4_qpu_dyadic/counts.sio.in" "$ROOT/tests/reference/wave4_qpu_dyadic_reference.cpp" "$0" "$SOUNIO_STDLIB_PATH/qpu/chsh.sio" >"$OUT/INPUTS.sha256"
sha256sum "$RAW" >"$OUT/COMPILER.sha256"
"$CXX" -std=c++23 -O2 -fno-fast-math -frounding-math -Wall -Wextra -Werror "$ROOT/tests/reference/wave4_qpu_dyadic_reference.cpp" -o "$OUT/reference"
sha256sum "$OUT/reference" >"$OUT/EXECUTABLES.sha256"
timeout 300 "$OUT/reference" --exhaustive-scale >"$OUT/exhaustive.stdout"
timeout 30 "$OUT/reference" --self-test >"$OUT/oracle-controls.stdout"
"$OUT/reference" --generate "$OUT/generated"
"$NODE" - "$ROOT" "$OUT" "$SOUNIO_STDLIB_PATH" <<'JS'
const fs=require('fs'),r=process.argv[2],o=process.argv[3],lib=process.argv[4];
const t=fs.readFileSync(r+'/tests/fixtures/wave4_qpu_dyadic/counts.sio.in','utf8');
const source=fs.readFileSync(lib+'/qpu/chsh.sio','utf8');
let original=source.slice(source.indexOf('fn wide_counts('),source.indexOf('\nfn mix_wide(')).trim();
original=original.replace('fn wide_counts(t: Tape,','fn wide_core_literal(p: BasisProb,').replace('    let p = tape_replay_2(t)\n','');
const literal=t.slice(t.indexOf('fn wide_core_literal('),t.indexOf('\nfn wide_core_dyadic(')).trim();
const dyadic=t.slice(t.indexOf('fn wide_core_dyadic('),t.indexOf('\nfn emit_counts(')).trim();
const candidate=literal.replace('fn wide_core_literal','fn wide_core_dyadic').replace('let scale = 4294967296.0','let inverse_scale = 0.00000000023283064365386962890625').replace(' / scale',' * inverse_scale');
if(candidate!==dyadic)throw Error('candidate changed beyond exact reciprocal');
function exactOnce(s,a,b){if(s.split(a).length!==2)throw Error('nonunique hoist normalization');return s.replace(a,b);}
function sourceKind(body){
 if(body===literal)return 'literal';
 if(body.replace('fn wide_core_literal','fn wide_core_dyadic')===dyadic)return 'dyadic';
 let normalized=exactOnce(body,'    let p00 = p.p00\n','');
 normalized=exactOnce(normalized,'    let b01 = p00 + p.p01\n','    let b01 = p.p00 + p.p01\n');
 normalized=exactOnce(normalized,'if u < p00 {','if u < p.p00 {');
 if(normalized.replace('fn wide_core_literal','fn wide_core_dyadic')!==dyadic)throw Error('source body mismatch after exact hoist');
 if(body!==hoisted)throw Error('noncanonical hoist placement');
 return 'dyadic-hoist';
}
const dyadicOriginal=dyadic.replace('fn wide_core_dyadic','fn wide_core_literal');
const hoisted=dyadicOriginal.replace('    let b01 = p.p00 + p.p01\n','    let p00 = p.p00\n    let b01 = p00 + p.p01\n').replace('if u < p.p00 {','if u < p00 {');
if(sourceKind(literal)!=='literal'||sourceKind(dyadicOriginal)!=='dyadic'||sourceKind(hoisted)!=='dyadic-hoist')throw Error('source validator positive control');
for(const [name,bad] of [
 ['extra-operation',hoisted.replace('s = s *','s = s +')],
 ['wrong-p00',hoisted.replace('let p00 = p.p00','let p00 = p.p01')],
 ['missing-declaration',hoisted.replace('    let p00 = p.p00\n','')],
 ['duplicate-declaration',hoisted.replace('    let p00 = p.p00\n','    let p00 = p.p00\n    let p00 = p.p00\n')],
 ['literal-hoist',hoisted.replace(' * inverse_scale',' / scale').replace('let inverse_scale = 0.00000000023283064365386962890625','let scale = 4294967296.0')],
 ['changed-threshold',hoisted.replace('if u < b01 {','if u <= b01 {')],
 ['moved-declaration',hoisted.replace('    let p00 = p.p00\n','').replace('    while k < shots {\n','    while k < shots {\n    let p00 = p.p00\n')]
]){
 let rejected=false;try{sourceKind(bad);}catch{rejected=true;}
 if(!rejected)throw Error('source validator accepted '+name);
}
console.log('SOURCE_VALIDATOR_CONTROLS_OK positives=3 negatives=7');
console.log('EXPLICIT_SOURCE_KIND '+sourceKind(original));
if(t.split('@CALLS@').length!==2)throw Error('marker');
const driver=t.replace('@CALLS@',fs.readFileSync(o+'/generated/CALLS.sio','utf8'));
fs.writeFileSync(o+'/driver.sio',driver);
fs.writeFileSync(o+'/wrong.sio',driver.replace('0.00000000023283064365386962890625','0.000000000232830643653869680605038284564229679463043254372678347863256931304931640625'));
console.log('EXACT_SOURCE_BODY_AND_TWO_REPLACEMENTS_OK');
JS
find "$OUT/generated" -type f -print0|sort -z|xargs -0 sha256sum >>"$OUT/INPUTS.sha256"
sha256sum "$OUT/driver.sio" "$OUT/wrong.sio" >>"$OUT/INPUTS.sha256"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288;ulimit -c 0
for mode in 0 1;do
 export SOUNIO_LOOP_REGIONS=$mode
 timeout 300 "$RAW" check "$OUT/driver.sio" >"$OUT/check-$mode.stdout" 2>&1
 timeout 600 "$RAW" compile "$OUT/driver.sio" -o "$OUT/driver-$mode.elf" >"$OUT/compile-$mode.stdout" 2>&1
 chmod +x "$OUT/driver-$mode.elf"
 sha256sum "$OUT/driver-$mode.elf" >>"$OUT/EXECUTABLES.sha256"
 timeout 300 "$OUT/driver-$mode.elf" >"$OUT/native-$mode.stdout" 2>&1
 "$OUT/reference" "$OUT/native-$mode.stdout" >"$OUT/oracle-$mode.stdout"
done
export SOUNIO_LOOP_REGIONS=1
timeout 600 "$RAW" compile "$OUT/wrong.sio" -o "$OUT/wrong.elf" >"$OUT/wrong.compile.stdout" 2>&1
chmod +x "$OUT/wrong.elf"
sha256sum "$OUT/wrong.elf" >>"$OUT/EXECUTABLES.sha256"
timeout 300 "$OUT/wrong.elf" >"$OUT/wrong.native.stdout" 2>&1
wrong_rc=0
"$OUT/reference" "$OUT/wrong.native.stdout" >"$OUT/wrong.oracle.stdout" 2>&1||wrong_rc=$?
[[ "$wrong_rc" == 1 ]]
echo QPU_DYADIC_COUNTS_AND_WRONG_RECIPROCAL_CONTROL_OK
