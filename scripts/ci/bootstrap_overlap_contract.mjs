#!/usr/bin/env node
// Structural and real failure controls for orchestration, without compiler execution.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import assert from 'node:assert/strict';
import {execFileSync,spawnSync} from 'node:child_process';
const root=process.argv[2]; assert(root,'usage: bootstrap_overlap_contract.mjs REPO [--selftest]');
const workflow=fs.readFileSync(path.join(root,'.github/workflows/ci.yml'),'utf8');
const decision=fs.readFileSync(path.join(root,'scripts/ci/evaluate_ci_decision.py'),'utf8');
const baseline=JSON.parse(fs.readFileSync(path.join(root,'scripts/ci/bootstrap_steps_baseline.json'),'utf8'));
function job(w,id){const start=w.indexOf('  '+id+':\n');assert(start>=0,'missing '+id);const r=w.slice(start+1),end=r.search(/^  [A-Za-z0-9_-]+:/m);return end<0?w.slice(start):w.slice(start,start+1+end);}
const canonical=s=>s.split('\n').filter(l=>l.trim()&&!/^\s*#/.test(l)).join('\n');
function check(w,d) {
 const p=job(w,'source-bootstrap-selfhost-linux-x86_64'),r=job(w,'source-bootstrap-regressions'),m=job(w,'native-selfhost-macos-arm64'),join=job(w,'ci-decision');
 assert(p.includes("    if: needs.impact.outputs.compiler == 'true' || needs.impact.outputs.full == 'true'"),'producer selection changed');
 assert(p.includes('    needs: [impact, fast-checks]\n'),'producer serializes behind regressions');
 assert(r.includes('    needs: [impact, fast-checks, source-bootstrap-selfhost-linux-x86_64]\n'),'regressions need fresh producer');
 assert(r.includes("    if: ${{ !cancelled() && (needs.impact.outputs.compiler == 'true' || needs.impact.outputs.full == 'true') && needs.source-bootstrap-selfhost-linux-x86_64.result == 'success' }}"),'regression selection/producer success guard changed');
 assert(m.includes('    needs: [impact, fast-checks, source-bootstrap-selfhost-linux-x86_64]\n'),'macOS must overlap regressions');
 for(const j of [p,r])assert.equal(j.split('run: bash scripts/lib/materialize_madaros_prebuilt.sh').length,2,'each job needs one materialization');
 const found=[...p.split(/(?=^      - )/m),...r.split(/(?=^      - )/m)].map(canonical);
 for(const old of baseline)assert.equal(found.filter(s=>s===canonical(old)).length,1,'omitted, altered or duplicated original proof: '+old.split('\n')[0]);
 assert(!p.includes('Source-bootstrap modular typecheck proof'),'producer still waits for modular proof');
 assert(p.includes('name: source-bootstrap-compiler-linux-x86_64')&&r.includes('name: source-bootstrap-compiler-linux-x86_64'),'same-run artifact not connected');
 assert(p.includes('git rev-parse HEAD > "$ARTIFACT/SOURCE_GIT_SHA"'),'source binding lost');
 assert(p.includes('sha256sum souc-stage2 > souc-stage2.sha256'),'binary binding lost');
 for(const cmd of ['test "$(cat "$ARTIFACT/SOURCE_GIT_SHA")" = "$(git rev-parse HEAD)"','sha256sum -c souc-stage2.sha256','chmod +x "$ARTIFACT/souc-stage2"'])assert(r.includes(cmd),'verification lost: '+cmd);
 assert(join.includes('      - source-bootstrap-regressions\n'),'regressions disconnected from final verdict');
 assert(d.includes('"source-bootstrap-regressions": truthy(impact.get("compiler")) or truthy(impact.get("full")),'),'selected regressions must succeed');
 assert(job(w,'fast-checks').includes('run: node scripts/ci/bootstrap_overlap_contract.mjs "$GITHUB_WORKSPACE" --selftest'),'contract disconnected');
}
check(workflow,decision);
if(process.argv.includes('--selftest')){
 const bad=[
  [workflow.replace('      - source-bootstrap-regressions\n',''),decision],
  [workflow,decision.replace('"source-bootstrap-regressions": truthy(impact.get("compiler")) or truthy(impact.get("full")),','"source-bootstrap-regressions": False,')],
  [workflow.replace('run: bash scripts/ci/bootstrap_generic_array_layout_gate.sh','run: true # bash scripts/ci/bootstrap_generic_array_layout_gate.sh'),decision],
  [workflow.replace('sha256sum -c souc-stage2.sha256','true # no digest'),decision],
  [workflow.replace('test "$(cat "$ARTIFACT/SOURCE_GIT_SHA")" = "$(git rev-parse HEAD)"','true # no source check'),decision],
  [workflow.replace('    needs: [impact, fast-checks, source-bootstrap-selfhost-linux-x86_64]\n','    needs: [impact, fast-checks]\n'),decision],
 ];
 for(const [w,d]of bad){assert(w!==workflow||d!==decision,'mutation missed target');assert.throws(()=>check(w,d),'unsafe mutation accepted');}
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'bootstrap-receipt-'));
 function run(fragment){return spawnSync('bash',['-euo','pipefail','-c',fragment.replaceAll('/tmp/sounio-source-bootstrap-selfhost-linux-x86_64/artifacts',dir)],{cwd:root,encoding:'utf8'});}
 function script(w,name){const j=w.split('      - name: '+name+'\n')[1];assert(j,'missing fixture command');return j.split(/(?=^      - |^  [A-Za-z])/m)[0].split('        run: |\n')[1].split('\n').filter(l=>l.startsWith('          ')).map(l=>l.slice(10)).join('\n');}
 try{
  fs.writeFileSync(path.join(dir,'souc-stage2'),'fixture only: never executed\n');
  const bind=script(workflow,'Bind the source-bootstrap stage2 to this checkout'),verify=script(workflow,'Verify source-bootstrap provenance and executable permission');
  assert.equal(run(bind).status,0); assert.equal(run(verify).status,0,'good receipt rejected');
  fs.appendFileSync(path.join(dir,'souc-stage2'),'corruption\n');assert.notEqual(run(verify).status,0,'corrupt binary accepted');
  assert.equal(run(bind).status,0);fs.writeFileSync(path.join(dir,'SOURCE_GIT_SHA'),'wrong-source\n');assert.notEqual(run(verify).status,0,'wrong source accepted');
  assert.equal(run(bind).status,0);fs.rmSync(path.join(dir,'souc-stage2.sha256'));assert.notEqual(run(verify).status,0,'missing digest accepted');
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
 console.log('BOOTSTRAP_REAL_RECEIPT_FAILURE_CONTROLS_PASS');
 console.log('BOOTSTRAP_OVERLAP_MUTATIONS_PASS');
}
console.log('BOOTSTRAP_ORIGINAL_PROOFS_AND_REQUIRED_JOIN_PASS');
