#!/usr/bin/env node
// Pin the mandatory partition inventory; this does not validate scientific values.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import cp from 'node:child_process';
import assert from 'node:assert/strict';
const root=process.argv[2];assert(root,'usage: deep_science_partition_contract.mjs REPO [--selftest]');
const workflow=fs.readFileSync(path.join(root,'.github/workflows/ci.yml'),'utf8');
const script=fs.readFileSync(path.join(root,'scripts/stdlib/deep_science_gate.sh'),'utf8');
const full=fs.readFileSync(path.join(root,'scripts/ci/madaros_full_gate.sh'),'utf8');
function check(w,s,f){
 const block=w.split('  stdlib-deep-science:\n')[1]?.split('  gate-wave-0:\n')[0];assert(block,'missing science job');
 assert(block.includes('      fail-fast: false\n      matrix:\n        group: [core, integration]\n'),'both partitions must run even if one fails');
 assert(block.includes('          STDLIB_DEEP_GROUP: ${{ matrix.group }}\n'),'partition not passed');
 assert(block.includes('name: stdlib-deep-science-${{ matrix.group }}-${{ github.sha }}-${{ github.run_attempt }}'),'partition artifact collision');
 assert(s.includes('case "$group" in all|core|integration) ;; *) echo "invalid STDLIB_DEEP_GROUP" >&2; exit 64 ;; esac'),'unknown groups must fail');
 assert(s.includes('if [[ "$group" == all || "$group" == core ]]; then run_core; fi'),'core disconnected');
 assert(s.includes('if [[ "$group" == all || "$group" == integration ]]; then run_integration; fi'),'integration disconnected');
 assert(s.includes('run_cpp_batch\nfor src in tests/compile-fail/stdlib_deep_'),'CPP preparation disconnected');
 assert(s.includes('(( jobs > memory_jobs )) && jobs="$memory_jobs"'),'worker memory bound removed');
 assert(s.indexOf('run_core() {')<s.indexOf('run_native_batch\n'),'native coverage outside core');
 assert(s.indexOf('run_integration() {')>s.indexOf('run_oracle_batch\n'),'oracle pairing split');
 assert(s.includes('record arrow-gate scripts/arrow_ipc_gate.sh "$arrow_rc"'),'Arrow gate missing');
 assert(s.includes('record concordance math-pure-ln-boundary "$rc"'),'logarithm gate missing');
 assert(w.includes('      - name: Madaros full gate against the fresh build\n        env:\n          MADAROS_RAW_BIN: /tmp/madaros-ci.elf\n          SOUNIO_MADAROS_DEREF_F64_GATE_BIN: /tmp/madaros-ci.elf\n        run: bash scripts/ci/madaros_full_gate.sh'),'full gate CI must bind both compiler entrypoints');
 assert(f.includes('bash "$ROOT_DIR/scripts/ci/madaros_imported_deref_f64_array_gate.sh"'),'full gate lost dereference witness');
}
check(workflow,script,full);
if(process.argv[3]==='--selftest'){
 const cases=[
 [workflow.replace('group: [core, integration]','group: [core]'),script,full],
 [workflow.replace('fail-fast: false','fail-fast: true'),script,full],
 [workflow.replace('STDLIB_DEEP_GROUP: ${{ matrix.group }}','STDLIB_DEEP_GROUP: core'),script,full],
 [workflow.replace('stdlib-deep-science-${{ matrix.group }}-','stdlib-deep-science-'),script,full],
 [workflow,script.replace('then run_integration; fi','then :; fi'),full],
 [workflow,script.replace('run_cpp_batch\nfor src in tests/compile-fail/stdlib_deep_',':\nfor src in tests/compile-fail/stdlib_deep_'),full],
 [workflow,script.replace('(( jobs > memory_jobs )) && jobs="$memory_jobs"',':'),full],
 [workflow.replace('SOUNIO_MADAROS_DEREF_F64_GATE_BIN: /tmp/madaros-ci.elf','IGNORED_BIN: /tmp/madaros-ci.elf'),script,full],
 ];
 for(const [i,args] of cases.entries())assert.throws(()=>check(...args),'accepted mutation '+i);
 console.log('DEEP_SCIENCE_PARTITION_MUTATIONS_PASS controls='+cases.length);
}
console.log('DEEP_SCIENCE_PARTITION_CONTRACT_PASS');

// Fixtures exercise process orchestration only, without a compiler or numerical oracle.
if(process.argv[3]==='--selftest'){
 const begin=script.indexOf('# CPP_POOL_BEGIN:');
 const end=script.indexOf('# CPP_POOL_END',begin);
 assert(begin>=0&&end>begin,'missing CPP pool fixture boundary');
 const pool=script.slice(begin,end);
 const fixture=String.raw`set -uo pipefail
out="$TEST_OUT"
jobs=2
fail=0
record() {
 printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$out/fixture.rows"
 [[ "$3" == 0 ]] || fail=1
 return 0
}
compile_cpp() {
 case "$1" in
 bad) record cpp-compile "$1" 1 ;;
 early_exit) exit 77 ;;
 *) record cpp-compile "$1" 0 ;;
 esac
}
`;
 const driver=String.raw`
queue_cpp good_a good_a ''
case "$CPP_FIXTURE_CASE" in
 success) queue_cpp good_b good_b '' ;;
 compile_error) queue_cpp bad bad '' ;;
 early_exit) queue_cpp early_exit early_exit '' ;;
 *) exit 64 ;;
esac
run_cpp_batch
grep -Fq good_a "$out/fixture.rows" || exit 1
if [[ "$CPP_FIXTURE_CASE" == success ]]; then
 [[ "$fail" == 0 ]] || exit 1
else
 [[ "$fail" == 1 ]] || exit 1
 grep -Fq failed-or-missing-status "$out/fixture.rows" || exit 1
fi
`;
 function runPool(body,kind){
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'deep-cpp-pool-'));
  try{return cp.spawnSync('bash',['-s'],{input:fixture+body+driver,env:{...process.env,TEST_OUT:dir,CPP_FIXTURE_CASE:kind},encoding:'utf8'});}
  finally{fs.rmSync(dir,{recursive:true,force:true});}
 }
 for(const kind of ['success','compile_error','early_exit']){
  const r=runPool(pool,kind);assert.equal(r.status,0,'CPP pool '+kind+' failed: '+r.stdout+r.stderr);
 }
 const mutant=pool.replace('record cpp-worker "${CPP_SOURCES[$idx]}:failed-or-missing-status" 1',':');
 assert.notEqual(mutant,pool,'CPP failure mutation did not match');
 assert.notEqual(runPool(mutant,'compile_error').status,0,'lost child failure accepted');
 assert.notEqual(runPool(mutant,'early_exit').status,0,'early child exit accepted without a status');
 console.log('DEEP_CPP_POOL_FAILURE_CONTROLS_PASS');
}
