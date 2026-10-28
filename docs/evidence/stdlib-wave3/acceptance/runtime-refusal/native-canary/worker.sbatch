#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027/runtime-refusal-native01
D=$B/results;S=$B/source
NODE=/scratch/sounio-stdlib-wave3-20261027/services-tools-recovered/tools/node
RAW=/scratch/sounio-stdlib-wave3-20261027/borrow-paged-integration03/results/madaros
export PATH="$(dirname "$NODE"):/usr/bin:/bin"
mkdir "$D"
cp -a /scratch/sounio-stdlib-wave3-20261027/root-integrated01/source "$S"
mkdir -p "$S/tests/stdlib/runtime-refusal"
trap 'r=$?;echo "$r" >"$D/STATUS";exit "$r"' EXIT
ulimit -s 524288;ulimit -c 0
export SOUC_BIN="$B/retain-souc.sh" STDLIB_BASELINE_ROOT="$S" SOUC_NATIVE_BIN="$RAW" SOUNIO_SOUC_BIN="$RAW" SOUNIO_SOUC_RAW_MODE=modular
export SOUNIO_STDLIB_PATH="$S/stdlib" SOUNIO_LOOP_REGIONS=0 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_REPO_HARD_NO_RUST=1 SKIP_BUILD=1
export SOUNIO_MADAROS_CACHE="$B/cache" SOUNIO_BUILD_LOCK="$B/build.lock"
sha256sum "$RAW" "$B/canary.sio" "$B/retain-souc.sh" >"$D/INPUTS.sha256"
for kind in good wrong-diagnostic unexpected-success;do
 cp "$B/canary.sio" "$S/tests/stdlib/runtime-refusal/canary.sio"
 if [[ "$kind" == wrong-diagnostic ]];then sed -i 's/expect-diagnostic: STDLIB_RUNTIME_REFUSAL_CANARY/expect-diagnostic: WRONG_DIAGNOSTIC/' "$S/tests/stdlib/runtime-refusal/canary.sio";fi
 if [[ "$kind" == unexpected-success ]];then sed -i 's/return 1/return 0/' "$S/tests/stdlib/runtime-refusal/canary.sio";fi
 export STDLIB_BASELINE_RETAIN="$D/retained/$kind"
 rc=0
 bash "$S/scripts/stdlib/run_stdlib_e2e.sh" --filter runtime-refusal/canary.sio --json-out "$D/$kind.json" --artifacts-dir "$D/$kind.capture" >"$D/$kind.stdout" 2>&1 ||rc=$?
 if [[ "$kind" == good ]];then [[ "$rc" == 0 ]];else [[ "$rc" == 1 ]];fi
 printf '%s\t%s\n' "$kind" "$rc" >>"$D/RESULTS.tsv"
done
node - "$D" >"$D/VERIFY.stdout" <<'NODE'
const fs=require('fs'),assert=require('assert/strict'),d=process.argv[2];
for(const name of ['good','wrong-diagnostic','unexpected-success']){
 const j=JSON.parse(fs.readFileSync(d+'/'+name+'.json'));assert.equal(j.results.length,1);assert.equal(j.totals.skip,0);assert.equal(j.results[0].status,name==='good'?'pass':'fail');
 const p=d+'/'+name+'.capture/tests/stdlib/runtime-refusal/canary.sio/';
 for(const phase of ['check','compile'])assert.equal(JSON.parse(fs.readFileSync(p+phase+'/result.json')).exit_code,0);
 assert.equal(JSON.parse(fs.readFileSync(p+'native/result.json')).exit_code,name==='unexpected-success'?0:1);
}
console.log('RUNTIME_REFUSAL_NATIVE_CANARY_OK cases=3 numerical_scope=none');
NODE
sha256sum -c "$D/INPUTS.sha256" >"$D/INPUTS.verify"
