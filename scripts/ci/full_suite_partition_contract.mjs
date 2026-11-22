#!/usr/bin/env node
// Selection/worker-budget checks only. Never execute a real compiler in fixtures.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import cp from 'node:child_process';
import assert from 'node:assert/strict';
const root=process.argv[2];assert(root,'usage: full_suite_partition_contract.mjs REPO [--selftest]');
const workflow=fs.readFileSync(path.join(root,'.github/workflows/ci.yml'),'utf8');
const helper=path.join(root,'scripts/ci/full_suite_partition.sh');
function check(w){
 const block=w.split('  full-test-suite:\n')[1]?.split('  slow-lane:\n')[0];assert(block,'full suite missing');
 assert(block.includes('name: Full Test Suite (${{ matrix.shard }})'),'shard identity missing');
 assert(block.includes('needs: [impact, fast-checks, native-selfhost-linux-x86_64]'),'fresh producer dependency missing');
 assert(block.includes('fail-fast: false\n      matrix:\n        shard: [0, 1, 2, 3]'),'all four shards required');
 assert(!block.includes('continue-on-error:'),'suite failure must block integration');
 assert(block.includes('SOUNIO_TEST_JOBS: 2\n          SOUNIO_TEST_JOBS_DYNAMIC: "0"\n'),'requested jobs must be an effective bound');
 assert(block.includes('FULL_SUITE_SHARD: ${{ matrix.shard }}'),'matrix shard not passed');
 assert(block.includes('bash scripts/ci/full_suite_partition.sh "$FULL_SUITE_SHARD" "$RUNNER_TEMP/full-suite"'),'canonical selector missing');
 assert(block.includes('bash scripts/run_sio_test_suite.sh --test-list "$RUNNER_TEMP/full-suite/selected.list" --format junit --jobs 2'),'canonical runner/list/budget lost');
 assert(block.includes('SOUNIO_TEST_SOUC_BIN: /tmp/souc-stage2'),'suite compiler routing changed');
 assert(block.includes('set -euo pipefail'),'suite pipeline must propagate failures');
 assert(block.includes('grep -Fxq "Parallel jobs: 2" "$RUNNER_TEMP/full-suite/console.log"'),'effective workers must be audited');
 assert(block.includes('name: native-compiler-linux-x86_64'),'suite must download producer artifact');
 assert(block.includes('name: qd128 transcendental accuracy (mpmath corpus replay)\n        if: matrix.shard == 0\n'),'accuracy replay must run exactly once');
 assert(block.includes('run: bash scripts/ci/qd128_transcendental_accuracy_gate.sh'),'accuracy replay disconnected');
 assert(block.includes('name: test-results-${{ matrix.shard }}-${{ github.run_attempt }}'),'results collide across shards or retries');
 assert(block.includes('path: ${{ runner.temp }}/full-suite/'),'inventory receipts not retained');
}
check(workflow);
assert(fs.readFileSync(helper,'utf8').includes('bash "$ROOT/scripts/run_sio_test_suite.sh" --list-tests'),'inventory must come from the public canonical runner');
if(process.argv[3]==='--selftest'){
 const mutations=[
  workflow.replace('          set -euo pipefail\n          chmod +x /tmp/souc-stage2','          set -eu\n          chmod +x /tmp/souc-stage2'),
  workflow.replace('          grep -Fxq "Parallel jobs: 2" "$RUNNER_TEMP/full-suite/console.log"','          true'),
  workflow.replace('shard: [0, 1, 2, 3]','shard: [0, 1, 2]'),
  workflow.replace('SOUNIO_TEST_JOBS_DYNAMIC: "0"','SOUNIO_TEST_JOBS_DYNAMIC: "1"'),
  workflow.replace('FULL_SUITE_SHARD: ${{ matrix.shard }}','FULL_SUITE_SHARD: 0'),
  workflow.replace('"$RUNNER_TEMP/full-suite/selected.list" --format junit --jobs 2','"$RUNNER_TEMP/full-suite/selected.list" --format junit --jobs 7'),
  workflow.replace('if: matrix.shard == 0','if: matrix.shard == 9'),
  workflow.replace('test-results-${{ matrix.shard }}-${{ github.run_attempt }}','test-results'),
 ];
 for(const [i,w] of mutations.entries()){assert.notEqual(w,workflow,'ineffective mutation '+i);assert.throws(()=>check(w),'accepted mutation '+i);}
 const temp=fs.mkdtempSync(path.join(os.tmpdir(),'full-suite-partition-fixture-'));
 try{
  const marker=path.join(temp,'compiler-called'),compiler=path.join(temp,'mock-souc');
  fs.writeFileSync(compiler,'#!/usr/bin/env bash\nprintf called > "$COMPILER_CALLED"\nexit 99\n',{mode:0o755});
  const fixture=path.join(temp,'repo');fs.mkdirSync(path.join(fixture,'scripts/ci'),{recursive:true});fs.mkdirSync(path.join(fixture,'tests/run-pass'),{recursive:true});
  const fixtureHelper=path.join(fixture,'scripts/ci/full_suite_partition.sh');fs.copyFileSync(helper,fixtureHelper);
  const cases=['a','b','c','d','e','f','g','h'].map(name=>'tests/run-pass/'+name+'.sio');
  for(const name of cases)fs.writeFileSync(path.join(fixture,name),'fixture');
  fs.writeFileSync(path.join(fixture,'inventory.list'),cases.join('\n')+'\n');
  fs.writeFileSync(path.join(fixture,'scripts/run_sio_test_suite.sh'),`#!/usr/bin/env bash
set -euo pipefail
if [[ $1 == --list-tests ]]; then cat "$(dirname "$0")/../inventory.list"; exit 0; fi
[[ $1 == --test-list ]] || exit 99
printf 'Parallel jobs: %s\n' "\${FAKE_WORKERS:-2}"
printf 'fixture diagnostic\n' >&2
printf '<testsuite/>\n' > test-results.xml
exit "\${FAKE_SUITE_RC:-0}"
`);

  cp.execFileSync('git',['init','-q'],{cwd:fixture});cp.execFileSync('git',['add','.'],{cwd:fixture});cp.execFileSync('git',['-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture'],{cwd:fixture});
  const env={...process.env,SOUNIO_TEST_SOUC_BIN:compiler,COMPILER_CALLED:marker,SOUNIO_TEST_JOBS_DYNAMIC:'0'};
  const invoke=(shard,out,extra={})=>cp.spawnSync('bash',[fixtureHelper,shard,out],{env:{...env,...extra},encoding:'utf8',timeout:60000});
  let canonical;const union=[];
  for(const shard of ['0','1','2','3']){
   const out=path.join(temp,'out-'+shard),result=invoke(shard,out);assert.equal(result.status,0,result.stderr);
   const all=fs.readFileSync(path.join(out,'canonical.list'),'utf8');if(canonical===undefined)canonical=all;else assert.equal(all,canonical);
   const selected=fs.readFileSync(path.join(out,'selected.list'),'utf8').trim().split('\n');assert(selected.every(Boolean));union.push(...selected);
   assert.equal(fs.readFileSync(path.join(out,'GROUP'),'utf8').trim(),shard);
  }
  assert.deepEqual(union.sort(),canonical.trim().split('\n').sort(),'canonical cases omitted or duplicated');
  assert(!fs.existsSync(marker),'list selection executed a compiler');
  assert.equal(invoke('4',path.join(temp,'bad')).status,64,'unknown shard accepted');
  const mockbin=path.join(temp,'mockbin');fs.mkdirSync(mockbin);
  const realSplit=cp.execFileSync('bash',['-c','command -v split'],{encoding:'utf8'}).trim();
  fs.writeFileSync(path.join(mockbin,'split'),`#!/usr/bin/env bash
set -euo pipefail
"$REAL_SPLIT" "$@"
prefix="\${@: -1}"
case "$PARTITION_MUTATION" in
 omission) sed -i '1d' "\${prefix}00" ;;
 duplicate) head -n 1 "\${prefix}00" >> "\${prefix}01" ;;
esac
`,{mode:0o755});
  for(const mode of ['omission','duplicate']){
   const result=invoke('0',path.join(temp,mode),{PATH:mockbin+':'+env.PATH,REAL_SPLIT:realSplit,PARTITION_MUTATION:mode});
   assert.notEqual(result.status,0,'accepted '+mode);assert(result.stderr.includes('omitted or duplicated'),'unrelated failure masked '+mode);
  }
  const block=workflow.split('  full-test-suite:\n')[1].split('  slow-lane:\n')[0];
  const runBlock=block.split('      - name: Run full test suite\n')[1].split('        run: |\n')[1];
  const lines=[];for(const line of runBlock.split('\n')){if(!line.startsWith('          '))break;lines.push(line.slice(10));}
  const command=lines.join('\n').replace('chmod +x /tmp/souc-stage2','chmod +x "$SOUNIO_TEST_SOUC_BIN"');
  const runCase=(rc,workers)=>cp.spawnSync('bash',['-c',command],{cwd:fixture,env:{...env,RUNNER_TEMP:path.join(temp,'workflow'),FULL_SUITE_SHARD:'0',FAKE_SUITE_RC:rc,FAKE_WORKERS:workers},encoding:'utf8'});
  assert.equal(runCase('0','2').status,0,'positive suite fragment failed');
  assert.equal(runCase('23','2').status,23,'tee laundered suite failure');
  assert.notEqual(runCase('0','7').status,0,'effective worker mismatch accepted');
  assert(!fs.existsSync(marker),'workflow fixture executed a compiler');
  console.log('FULL_SUITE_FAILURE_PROPAGATION_AND_WORKER_AUDIT_PASS');
  console.log('FULL_SUITE_CANONICAL_UNION_AND_NEGATIVE_FIXTURES_PASS');
 }finally{fs.rmSync(temp,{recursive:true,force:true});}
 console.log('FULL_SUITE_WORKFLOW_MUTATIONS_PASS');
}
console.log('FULL_SUITE_PARTITION_CONTRACT_PASS');
