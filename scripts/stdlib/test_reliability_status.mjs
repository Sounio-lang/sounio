import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {aggregate} from './reliability_status.mjs';
import {inventory,maskSource} from './inventory.mjs';
const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'stdlib-reliability-controls-'));
const write=(p,v)=>{fs.mkdirSync(path.dirname(p),{recursive:true});fs.writeFileSync(p,typeof v==='string'?v:JSON.stringify(v));};
let checks=0;
const test=(name,f)=>{f();checks++;console.log('PASS '+name);};
const totals={pass:1,fail:0,skip:0,total:1};
const e2e={schema:'sounio.stdlib.e2e.result.v1',totals,results:[{status:'pass',test_path:'tests/stdlib/a.sio'}]};
const inv={schema:'sounio.stdlib.inventory.v1',counts:{sio_files:1,disabled_files:0,stub_mod_files:0,active_module_entrypoints:1}};
const hyper={schema:'sounio.stdlib.hyper_execution_status.v1',status_summary:'pass',totals};
const science={schema:'sounio.stdlib.science_pipeline_status.v1',status_summary:'pass',runtime_regression_enforcement:'strict',
  runtime_regression_summary:{status:'pass',fail:0},runtime_provenance:{souc_bin:'fixture',souc_version:'fixture',pinned_version_expected:'fixture'}};
const paths=Object.fromEntries(['e2e','inventory','hyper','science'].map(k=>[k,path.join(tmp,k+'.json')]));
const reset=()=>{for(const[k,v]of Object.entries({e2e,inventory:inv,hyper,science}))write(paths[k],v);};
const goodCodes={inventory:0,hyper:0,e2e:0};
try {
  test('valid complete receipts pass',()=>{reset();assert.equal(aggregate(tmp,paths,goodCodes,true).status_summary,'pass');});
  test('contract adjustment annotations survive aggregation',()=>{
    reset();write(path.join(tmp,'tests/stdlib/contracts/test.sio'),'//@ contract-adjustment: preserved checked contract\nfn main() {}');
    assert.deepEqual(aggregate(tmp,paths,goodCodes).contract_adjustments,[{test_path:'tests/stdlib/contracts/test.sio',adjustment:'preserved checked contract'}]);
  });
  test('producer failure cannot reuse a passing artifact',()=>{reset();assert.equal(aggregate(tmp,paths,{...goodCodes,hyper:1}).status_summary,'fail');});
  test('missing hyper retains E2E results',()=>{reset();fs.unlinkSync(paths.hyper);const r=aggregate(tmp,paths,goodCodes);assert.equal(r.totals.pass,1);assert.equal(r.status_summary,'not_run');});
  test('empty selection fails closed',()=>{reset();write(paths.e2e,{...e2e,totals:{pass:0,fail:0,skip:0,total:0},results:[]});assert.notEqual(aggregate(tmp,paths,goodCodes).status_summary,'pass');});
  test('totals and duplicate cases must reconcile',()=>{reset();write(paths.e2e,{...e2e,totals:{pass:2,fail:0,skip:0,total:2},results:[e2e.results[0],e2e.results[0]]});assert.notEqual(aggregate(tmp,paths,goodCodes).status_summary,'pass');});
  test('strict runtime failures remain failures',()=>{reset();write(paths.science,{...science,runtime_regression_summary:{status:'fail',fail:1}});assert.equal(aggregate(tmp,paths,goodCodes,true).status_summary,'fail');});
  test('source inventory ignores commented and quoted declarations',()=>{
    const root=path.join(tmp,'source');
    write(path.join(root,'stdlib/demo/lib.sio'),'// pub fn fake() {}\n/* outer /* pub fn fake2() {} */ */\npub fn real() { let s = "pub fn fake3() {}" }\n  pub struct Value {}\n');
    write(path.join(root,'tests/stdlib/demo/test.sio'),'use demo::lib::{real}\nfn main() {}');
    const j=inventory(path.join(root,'stdlib'),root);
    assert.equal(j.counts.sio_files,1);assert.deepEqual(j.source_files[0].public_declarations.map(x=>x.name),['real','Value']);
    assert.equal(j.source_files[0].import_witness_candidates.length,1);assert.equal(j.source_files[0].evidence_status,'source_only');
    assert.equal(maskSource('/*\npub fn nope(){}\n*/').split('\n').length,3);
  });
  test('full shell flow still executes E2E after failed hyper',()=>{
    const root=path.join(tmp,'flow');
    write(path.join(root,'scripts/dev/stdlib_reliability_gate.sh'),fs.readFileSync(path.join(repo,'scripts/dev/stdlib_reliability_gate.sh'),'utf8'));
    write(path.join(root,'scripts/stdlib/reliability_status.mjs'),fs.readFileSync(path.join(repo,'scripts/stdlib/reliability_status.mjs'),'utf8'));
    const producer=(json,rc,marker='')=>'#!/bin/bash\nwhile [[ $# -gt 0 ]];do case "$1" in --json-out|--out-json) out=$2;shift 2;;*)shift;;esac;done\nprintf \'%s\\n\' \''+JSON.stringify(json)+'\' > "$out"\n'+marker+'\nexit '+rc+'\n';
    write(path.join(root,'scripts/stdlib/scan_stdlib.sh'),producer(inv,0));
    write(path.join(root,'scripts/stdlib/stdlib_hyper_execution_gate.sh'),producer({...hyper,status_summary:'fail'},1));
    write(path.join(root,'scripts/stdlib/run_stdlib_e2e.sh'),producer(e2e,0,'printf executed > "'+root+'/E2E_EXECUTED"'));
    const sci=path.join(root,'science.json');write(sci,science);
    const r=spawnSync('bash',[path.join(root,'scripts/dev/stdlib_reliability_gate.sh'),'--science-json',sci],{encoding:'utf8'});
    assert.equal(r.status,1,r.stderr);
    assert.ok(fs.existsSync(path.join(root,'E2E_EXECUTED')));
    const result=JSON.parse(fs.readFileSync(path.join(root,'artifacts/stdlib/stdlib_reliability_status.v1.json'),'utf8'));
    assert.equal(result.totals.pass,1);assert.equal(result.status_summary,'fail');
  });
  console.log('STDLIB_RELIABILITY_CONTROLS_OK cases='+checks);
} finally {fs.rmSync(tmp,{recursive:true,force:true});}
