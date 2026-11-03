#!/usr/bin/env node
// Every prior Contracts gate must survive on exactly one required matrix shard.
// These checks inspect orchestration only; they execute no Sounio or math code.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
const root=process.argv[2];
assert(root,'usage: contracts_parallel_contract.mjs REPO [--selftest]');
const baseline=JSON.parse(fs.readFileSync(path.join(root,'scripts/ci/contracts_steps_baseline.json'),'utf8'));
const partition=fs.readFileSync(path.join(root,'scripts/ci/contracts_partition.tsv'),'utf8').trim().split('\n').filter(l=>!l.startsWith('#')).map(l=>l.split('\t'));
const shards=['core-a','core-b','core-c','research'];
const assignments=new Map();
for(const row of partition)assert.equal(row.length,2,'malformed partition row');
for(const [owner,name] of partition){assert(shards.includes(owner),'unknown required shard');assert(name&&!assignments.has(name),'missing/duplicate assignment');assignments.set(name,owner);}
const workflow=fs.readFileSync(path.join(root,'.github/workflows/ci.yml'),'utf8');
const canonical=s=>s.split('\n').filter(l=>l.trim()&&!/^\s*#/.test(l)).map(l=>l.trimEnd()).join('\n');
const common=new Set(['      - uses: actions/checkout@v4','      - name: Materialize the committed Madaros prebuilt','      - uses: actions/setup-node@v4','      - name: Contracts evaluation summary']);
function job(w,id){const a=w.indexOf('  '+id+':\n');assert(a>=0,'missing '+id);const rest=w.slice(a+1),e=rest.search(/^  [A-Za-z0-9_-]+:/m);return e<0?w.slice(a):w.slice(a,a+1+e);}
function check(w){
 const core=job(w,'contracts-core');
 assert(core.includes('    needs: [impact, fast-checks]\n'),'Contracts early gates lost');
 assert(core.includes('    name: Contracts Core (${{ matrix.shard }})\n'),'matrix check identities missing');
 assert(core.includes('    strategy:\n      fail-fast: false\n      matrix:\n        shard: [core-a, core-b, core-c, research]\n'),'matrix incomplete or fail-fast enabled');
 const parts=core.split(/(?=^      - )/m).slice(1);
 const restored=[];
 for(const p of parts){
  let text=canonical(p);const name=text.split('\n')[0];
  if(common.has(name)){assert(!text.includes('matrix.shard =='),'common setup conditional');if(name==='      - uses: actions/checkout@v4'){assert(text.includes('          fetch-depth: 0\n          filter: blob:none'),'checkout must preserve history without historical blobs');assert(!text.includes('sparse-checkout:'),'working tree must remain complete');text=text.replace('\n          filter: blob:none','');}restored.push(text);continue;}
  const selector=/^        if: matrix.shard == '(core-a|core-b|core-c|research)'(?: && \((.*)\))?$/m.exec(text);
  assert(selector,'gate must select exactly one known shard: '+name);
  assert.equal(selector[1],assignments.get(name.replace('      - name: ','')),'gate moved outside explicit partition: '+name);
  text=selector[2]?text.replace(selector[0],'        if: '+selector[2]):text.replace(selector[0]+'\n','');
  restored.push(text);
 }
 assert.deepEqual(restored,baseline.steps,'gate inventory, original conditions, environment, command or ordering changed');
 assert.deepEqual([...assignments.keys()],parts.filter(p=>!common.has(p.split('\n')[0])).map(p=>p.split('\n')[0].replace('      - name: ','')),'partition inventory differs from workflow');
 for(const shard of shards)assert([...assignments.values()].includes(shard),'empty required partition');
 // These related steps stay together; no setup or wrapper depends on another runner.
 const owner=name=>{const p=parts.find(p=>p.startsWith('      - name: '+name+'\n'));assert(p,'missing '+name);return /matrix.shard == '(core-a|core-b|core-c|research)'/.exec(p)?.[1];};
 for(const name of ['Measured claims match the tree','Measured-claim gate fails when it should','Self-falsifying compilation line — numpy','Self-falsifying compilation line','Ontology frontiers hold','Ontology validation','Ontology CLI smoke gate'])assert.equal(owner(name),'research','dependent research setup separated');
 for(const group of [["PIREUS admission engine compiles and answers","Pireus Walsh twist channel spectrum gate","Pireus typed XOR GPU lowering gate","Pireus argv-driven material tests run with their arguments"],["PBPK28 driver refusals (Madaros)","PBPK28 driver refusals (lean_single)"],["One name, one definition","Duplicate-definition gate fails when it should"],["Doc snippet checker self-test","Selected documentation sample check"],["Canonical lean_single fixed point","lean_single nested private-store regression (#2363)"],["TypeKind archaeology ladder","TypeKind archaeology family C","TypeKind archaeology family E","TypeKind archaeology family B"],["Fano kernel bytes come from the compiler's lowering","Fano kernel provenance gate rejects a one-bit mutation"]])for(const name of group)assert.equal(owner(name),owner(group[0]),'dependent core group separated: '+name);
 const join=job(w,'contracts');assert(join.includes('needs: [contracts-core, contracts-ontology]'),'matrix verdict disconnected');
 assert(join.includes('needs.contracts-core.result'),'matrix result not evaluated');
 const build=job(w,'current-source-madaros');
 assert(build.includes('SOUNIO_MADAROS_CACHE: /tmp/madaros-build-cache'),'cache path not explicitly isolated');
 assert(build.includes('uses: actions/cache@v4'),'cache transport disconnected');
 assert(!build.includes('restore-keys:'),'cross-recipe prefix restore is forbidden');
 assert(build.includes('path: /tmp/madaros-build-cache'),'restore/build cache mismatch');
 assert(build.includes("hashFiles('self-hosted/**', 'stdlib/**', 'bin/souc-linux-x86_64', 'bin/souc-lean-single-x86_64', 'scripts/ci/build_modular_madaros.sh', 'scripts/dev/madaros-cache.sh', 'scripts/dev/souc-build-lock.sh')"),'cache invalidators lost');
 const buildStep=build.split(/(?=^      - )/m).find(p=>p.startsWith("      - name: Build Madaros from this PR's own source\n"));
 assert(buildStep&&!/^        if:/m.test(buildStep),'cache hit must still enter verified builder');
 assert(buildStep.includes('bash scripts/ci/build_modular_madaros.sh /tmp/madaros-ci.elf'),'verified builder bypassed');
 assert(build.includes('stdlib_deep_compiler_receipt.sh prepare'),'fresh checkout receipt missing');
 return {gates:'preserved',matrix:shards.join(','),cache:'verified builder unconditional'};
}
const result=check(workflow);
function partsForMutation(w){return job(w,'contracts-core').split(/(?=^      - )/m).filter(p=>p.startsWith('      - name: Contracts shard decision self-test\n'));}
if(process.argv[3]==='--selftest'){
 const mutations=[
  workflow.replace(partsForMutation(workflow)[0],''),
  workflow.replace(partsForMutation(workflow)[0],partsForMutation(workflow)[0]+partsForMutation(workflow)[0]),
  workflow.replace('          path: /tmp/madaros-build-cache','          path: /tmp/madaros-build-cache\n          restore-keys: madaros-build-v1-'),
  workflow.replace('run: bash scripts/ci/ontology_frontiers_gate.sh','run: true'),
  workflow.replace("if: matrix.shard == 'core-a'","if: matrix.shard == 'missing'"),
  workflow.replace("matrix.shard == 'research' && (needs.impact.outputs.ontology == 'true' || needs.impact.outputs.full == 'true')","matrix.shard == 'research' && (false)"),
  workflow.replace('shard: [core-a, core-b, core-c, research]','shard: [core-a, core-b, research]'),
  workflow.replace('fail-fast: false\n      matrix:\n        shard: [core-a, core-b, core-c, research]','fail-fast: true\n      matrix:\n        shard: [core-a, core-b, core-c, research]'),
  workflow.replace("      - name: Build Madaros from this PR's own source\n","      - name: Build Madaros from this PR's own source\n        if: false\n"),
  workflow.replace('bash scripts/ci/build_modular_madaros.sh /tmp/madaros-ci.elf','cp /tmp/old-madaros /tmp/madaros-ci.elf'),
  workflow.replace('path: /tmp/madaros-build-cache','path: /tmp/other-cache'),
  workflow.replace('needs: [contracts-core, contracts-ontology]','needs: [contracts-ontology]'),
 ];
 for(const [i,w]of mutations.entries()){assert.notEqual(w,workflow,'ineffective negative fixture '+i);assert.throws(()=>check(w),'accepted mutant '+i);}
 console.log('CONTRACTS_PARALLEL_NEGATIVE_CONTROLS_PASS');
}
console.log('CONTRACTS_PARALLEL_CONTRACT_PASS '+JSON.stringify(result));
