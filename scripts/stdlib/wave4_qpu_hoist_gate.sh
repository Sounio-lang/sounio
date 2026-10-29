#!/usr/bin/env bash
# Exact dyadic wide-counts baseline versus immutable p00 scalar hoist.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[[ $# == 2 && "$1" == /* && "$2" == /* ]] || exit 64
OUT=$1;RAW=$2
[[ ! -e "$OUT" && -x "$RAW" && ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ ]] || exit 64
: "${CXX:?qualified private C++23 driver}"
NODE=${NODE_BIN:-node}
mkdir "$OUT";mkdir "$OUT/native"
finish(){ r=$?;trap - EXIT;for m in SOURCES INPUTS TOOLS EXECUTABLES;do if [[ -s "$OUT/$m.sha256" ]];then sha256sum -c "$OUT/$m.sha256" >"$OUT/$m.final.verify" 2>&1||r=1;fi;done;echo "$r" >"$OUT/STATUS";exit "$r";};trap finish EXIT
printf '%s\n' "slurm:$SLURM_JOB_ID" >"$OUT/AUTHORITY"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export LD_LIBRARY_PATH="$OUT/native${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ulimit -s 524288;ulimit -c 0
find "$ROOT/stdlib" -type f -print0|sort -z|xargs -0 sha256sum >"$OUT/SOURCES.sha256"
sha256sum "$ROOT/runtime/stdlib_platform.cpp" "$ROOT/tests/fixtures/wave4_qpu_hoist/counts.sio.in" "$ROOT/tests/fixtures/wave4_qpu_hoist/SOURCE_LEDGER.json" "$ROOT/tests/reference/wave4_qpu_hoist_reference.cpp" "$0" >"$OUT/INPUTS.sha256"
sha256sum "$RAW" "$CXX" "$(command -v "$NODE")" >"$OUT/TOOLS.sha256"
"$CXX" --version >"$OUT/CXX_VERSION"
printf 'case\tphase\tstatus\n' >"$OUT/PHASES.tsv"
phase(){
 local name=$1 kind=$2 seconds=$3 rc=0;shift 3
 timeout "$seconds" "$@" >"$OUT/$name.$kind.stdout" 2>"$OUT/$name.$kind.stderr"||rc=$?
 printf '%s\t%s\t%s\n' "$name" "$kind" "$rc" >>"$OUT/PHASES.tsv"
 return "$rc"
}
phase reference build 180 "$CXX" -std=c++23 -O2 -fno-fast-math -frounding-math -Wall -Wextra -Werror "$ROOT/tests/reference/wave4_qpu_hoist_reference.cpp" -o "$OUT/reference"
sha256sum "$OUT/reference" >"$OUT/EXECUTABLES.sha256"
phase oracle-controls verify 30 "$OUT/reference" --self-test
phase generate fixtures 30 "$OUT/reference" --generate "$OUT/generated"
"$NODE" - "$ROOT" "$OUT" <<'JS'
const fs=require('fs'),crypto=require('crypto'),r=process.argv[2],o=process.argv[3],hash=s=>crypto.createHash('sha256').update(s).digest('hex');
const ledger=JSON.parse(fs.readFileSync(r+'/tests/fixtures/wave4_qpu_hoist/SOURCE_LEDGER.json')),lib=fs.readFileSync(r+'/stdlib/qpu/chsh.sio','utf8'),t=fs.readFileSync(r+'/tests/fixtures/wave4_qpu_hoist/counts.sio.in','utf8'),oracle=fs.readFileSync(r+'/tests/reference/wave4_qpu_hoist_reference.cpp');
if(hash(lib)!==ledger.library_after_sha256||hash(t)!==ledger.template_sha256||hash(oracle)!==ledger.oracle_sha256)throw Error('source pin');
function fn(s,name){const p=s.indexOf('fn '+name+'('),e=s.indexOf('\n}\n',p);if(p<0||e<0)throw Error('function absent');return s.slice(p,e+3);}
const candidate=fn(lib,'wide_counts'),a=fn(t,'wide_core_literal'),b=fn(t,'wide_core_hoist');
const adapt=s=>s.replace('fn wide_counts(t: Tape,','fn wide_core_hoist(p: BasisProb,').replace('    let p = tape_replay_2(t)\n','');
if(adapt(candidate)!==b)throw Error('candidate kernel is not exact library body');
const undo=b.replace('fn wide_core_hoist','fn wide_core_literal').replace('    let p00 = p.p00\n    let b01 = p00 + p.p01\n','    let b01 = p.p00 + p.p01\n').replace('if u < p00 {','if u < p.p00 {');
if(undo!==a)throw Error('extra candidate change');
if(t.split('@CALLS@').length!==2)throw Error('single fixture marker');
const driver=t.replace('@CALLS@',fs.readFileSync(o+'/generated/CALLS.sio','utf8'));
fs.writeFileSync(o+'/driver.sio',driver);
if(driver.split('let p00 = p.p00').length!==2)throw Error('single hoist mutation site');
fs.writeFileSync(o+'/wrong-hoist.sio',driver.replace('let p00 = p.p00','let p00 = p.p01'));
console.log('EXACT_CURRENT_DYADIC_AND_ONLY_P00_HOIST_OK');
JS
find "$OUT/generated" -type f -print0|sort -z|xargs -0 sha256sum >>"$OUT/INPUTS.sha256"
sha256sum "$OUT/driver.sio" "$OUT/wrong-hoist.sio" >>"$OUT/INPUTS.sha256"
phase runtime build 180 "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -shared -fPIC -pthread "$ROOT/runtime/stdlib_platform.cpp" -o "$OUT/native/libstdlib_platform.so"
sha256sum "$OUT/native/libstdlib_platform.so" >>"$OUT/EXECUTABLES.sha256"
for mode in 0 1;do
 export SOUNIO_LOOP_REGIONS=$mode
 phase "counts-$mode" check 1200 "$RAW" check "$OUT/driver.sio"
 phase "counts-$mode" compile 1200 "$RAW" compile "$OUT/driver.sio" -o "$OUT/counts-$mode.elf"
 chmod +x "$OUT/counts-$mode.elf";sha256sum "$OUT/counts-$mode.elf" >>"$OUT/EXECUTABLES.sha256"
 phase "counts-$mode" native 300 "$OUT/counts-$mode.elf"
 phase "counts-$mode" oracle 300 "$OUT/reference" "$OUT/counts-$mode.native.stdout"
 for source in tests/stdlib/qpu/test_qpu_chsh.sio tests/stdlib/quantum/test_born_shot_law.sio tests/stdlib/quantum/test_qasm_tape.sio;do
  name=$(basename "$source" .sio)-$mode
  sha256sum "$ROOT/$source" >>"$OUT/INPUTS.sha256"
  phase "$name" check 1200 "$RAW" check "$ROOT/$source"
  phase "$name" compile 1200 "$RAW" compile "$ROOT/$source" -o "$OUT/$name.elf"
  chmod +x "$OUT/$name.elf";sha256sum "$OUT/$name.elf" >>"$OUT/EXECUTABLES.sha256"
  phase "$name" native 30 "$OUT/$name.elf"
  grep -Fxq ALL_PASS "$OUT/$name.native.stdout"
 done
done
export SOUNIO_LOOP_REGIONS=1
phase wrong-hoist compile 1200 "$RAW" compile "$OUT/wrong-hoist.sio" -o "$OUT/wrong-hoist.elf"
chmod +x "$OUT/wrong-hoist.elf";sha256sum "$OUT/wrong-hoist.elf" >>"$OUT/EXECUTABLES.sha256"
phase wrong-hoist native 300 "$OUT/wrong-hoist.elf"
rc=0;phase wrong-hoist oracle-reject 300 "$OUT/reference" "$OUT/wrong-hoist.native.stdout"||rc=$?
[[ $rc == 1 ]]
echo DYADIC_P00_HOIST_COUNTS14_AND_ORIGINALS_OK
