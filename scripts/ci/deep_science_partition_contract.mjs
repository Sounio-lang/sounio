#!/usr/bin/env node
// Pin the mandatory partition inventory; this does not validate scientific values.
import fs from 'node:fs';
import path from 'node:path';
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
 assert(s.indexOf('run_core() {')<s.indexOf('run_native_batch\n'),'native coverage outside core');
 assert(s.indexOf('run_integration() {')>s.indexOf('run_oracle_batch\n'),'oracle pairing split');
 assert(s.includes('record arrow-gate scripts/arrow_ipc_gate.sh "$arrow_rc"'),'Arrow gate missing');
 assert(s.includes('record concordance math-pure-ln-boundary "$rc"'),'logarithm gate missing');
 assert(f.includes('SOUNIO_MADAROS_DEREF_F64_GATE_BIN="$RAW_MADAROS" \\\n  bash "$ROOT_DIR/scripts/ci/madaros_imported_deref_f64_array_gate.sh"'),'full gate rebuilds rather than forwarding its compiler');
}
check(workflow,script,full);
if(process.argv[3]==='--selftest'){
 const cases=[
 [workflow.replace('group: [core, integration]','group: [core]'),script,full],
 [workflow.replace('fail-fast: false','fail-fast: true'),script,full],
 [workflow.replace('STDLIB_DEEP_GROUP: ${{ matrix.group }}','STDLIB_DEEP_GROUP: core'),script,full],
 [workflow.replace('stdlib-deep-science-${{ matrix.group }}-','stdlib-deep-science-'),script,full],
 [workflow,script.replace('then run_integration; fi','then :; fi'),full],
 [workflow,script,full.replace('SOUNIO_MADAROS_DEREF_F64_GATE_BIN="$RAW_MADAROS"','IGNORED_BIN="$RAW_MADAROS"')],
 ];
 for(const [i,args] of cases.entries())assert.throws(()=>check(...args),'accepted mutation '+i);
 console.log('DEEP_SCIENCE_PARTITION_MUTATIONS_PASS controls='+cases.length);
}
console.log('DEEP_SCIENCE_PARTITION_CONTRACT_PASS');
