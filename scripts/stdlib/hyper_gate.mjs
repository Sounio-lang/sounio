#!/usr/bin/env node
// Operational scheduling/receipts only. Metric arithmetic belongs to C++23.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
let out=process.env.STDLIB_HYPER_STATUS_OUT||path.join(root,'artifacts/stdlib/stdlib_hyper_execution_status.v1.json');
let golden=process.env.STDLIB_HYPER_GOLDEN||path.join(root,'tests/fixtures/hyper/pipeline_golden.v1.json');
let matrix=process.env.STDLIB_NATIVE_LANE_MATRIX_OUT||path.join(root,'artifacts/stdlib/native_lane_matrix.v1.json');
const mode=process.env.STDLIB_HYPER_GATE_MODE||'required';
for(let i=2;i<process.argv.length;i++){
 const a=process.argv[i];
 if(a==='--help'||a==='-h'){console.log('stdlib_hyper_execution_gate.sh [--out-json PATH] [--golden PATH] [--lane-matrix-json PATH]');process.exit(0);}
 if(['--out-json','--golden','--lane-matrix-json'].includes(a)){
  const v=process.argv[++i];if(!v||v.startsWith('--'))throw Error(a+' requires a path');
  if(a==='--out-json')out=v;else if(a==='--golden')golden=v;else matrix=v;
 }else throw Error('unknown option '+a);
}
if(!['required','auto','off'].includes(mode))throw Error('invalid STDLIB_HYPER_GATE_MODE');
out=path.resolve(out);matrix=path.resolve(matrix);golden=path.resolve(golden);
if(new Set([out,matrix,golden]).size!==3)throw Error('output paths must be distinct from each other and the golden');
const tests={
 nn:'tests/stdlib/nn/test_hyper_quaternion_e2e.sio',onn:'tests/stdlib/onn/test_hyper_onn_e2e.sio',
 qnn:'tests/stdlib/qnn/test_hyper_qnn_e2e.sio',snn:'tests/stdlib/snn/test_snn_e2e.sio',
 spnn:'tests/stdlib/spnn/test_spiking_e2e.sio',quantnn:'tests/stdlib/quantnn/test_quantum_e2e.sio',
 math:'tests/stdlib/math/test_hyper_math_e2e.sio'};
const work=fs.mkdtempSync(path.join(process.env.STDLIB_HYPER_WORK_ROOT||os.tmpdir(),'stdlib-hyper-native-'));
const rows=[],failures=[],ignoreHits=[],notes=[];
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const execute=(program,args,stem,seconds=180)=>{
 const p=spawnSync('timeout',['--kill-after=5s',String(seconds),program,...args],{cwd:root,env:process.env,maxBuffer:64*1024*1024});
 fs.writeFileSync(path.join(work,stem+'.stdout'),p.stdout||Buffer.alloc(0));
 fs.writeFileSync(path.join(work,stem+'.stderr'),p.stderr||Buffer.alloc(0));
 const result={exit_code:p.status??125,signal:p.signal||null,error:p.error?.message||null};
 fs.writeFileSync(path.join(work,stem+'.status.json'),JSON.stringify({program,args,...result})+'\n');
 return result;
};
const compiler=process.env.SOUC_BIN||path.join(root,'bin/souc');
let oracleBuild={exit_code:125},oracleRun={exit_code:125},laneMetric=new Map();
if(mode!=='off'){
 oracleBuild=execute(process.env.CXX||'g++',['-std=c++23','-O2','-Wall','-Wextra','-Werror',path.join(root,'tests/reference/stdlib_hyper_metrics.cpp'),'-o',path.join(work,'metric-oracle')],'oracle-build',180);
 for(const[lane,source]of Object.entries(tests)){
  const full=path.join(root,source);
  const r={lane,test_path:source,required:true,status:'fail',reason:'not_run',metrics_status:'not_run',source_sha256:'',phases:{}};
  if(!fs.existsSync(full)){r.reason='required_test_missing';rows.push(r);continue;}
  const text=fs.readFileSync(full,'utf8');r.source_sha256=sha(full);
  if(/^\s*\/\/@\s*ignore(?:\s|$)/m.test(text))ignoreHits.push(source);
  r.phases.check=execute(compiler,['check',full],lane+'.check');
  if(r.phases.check.exit_code!==0){r.reason='check-failed';rows.push(r);continue;}
  const elf=path.join(work,lane+'.elf');
  r.phases.compile=execute(compiler,['compile',full,'-o',elf],lane+'.compile');
  if(r.phases.compile.exit_code!==0||!fs.existsSync(elf)){r.reason='compile-failed';rows.push(r);continue;}
  r.executable_sha256=sha(elf);
  r.phases.native=execute(elf,[],lane+'.native',300);
  r.status=r.phases.native.exit_code===0?'pass':'fail';r.reason=r.status==='pass'?'':'native-failed';
  if(sha(full)!==r.source_sha256){r.status='fail';r.reason='input_changed';}
  rows.push(r);
 }
 if(oracleBuild.exit_code===0&&fs.existsSync(golden)){
  const args=[golden];
  for(const lane of Object.keys(tests)){
   const p=path.join(work,lane+'.native.stdout');
   if(!fs.existsSync(p))fs.writeFileSync(p,'');
   args.push(lane,p);
  }
  oracleRun=execute(path.join(work,'metric-oracle'),args,'oracle-run',30);
  const output=fs.readFileSync(path.join(work,'oracle-run.stdout'),'utf8');
  for(const line of output.split('\n')){const m=/^LANE\t([^\t]+)\t(pass|fail)$/.exec(line);if(m)laneMetric.set(m[1],m[2]);}
 }
}
for(const r of rows){
 r.metrics_status=laneMetric.get(r.lane)||'not_run';
 if(r.status!=='pass')failures.push({test_path:r.test_path,category:r.reason,error_excerpt:JSON.stringify(r.phases)});
 if(r.metrics_status!=='pass')failures.push({test_path:r.test_path,category:'golden_mismatch_or_unavailable',error_excerpt:'see '+work+'/oracle-run.stdout'});
}
const totals={pass:rows.filter(r=>r.status==='pass').length,fail:rows.filter(r=>r.status==='fail').length,skip:0,total:rows.length};
const status=mode==='off'?'not_run':rows.length===7&&totals.fail===0&&oracleBuild.exit_code===0&&oracleRun.exit_code===0&&ignoreHits.length===0?'pass':'fail';
let counts={};try{counts=JSON.parse(fs.readFileSync(process.env.STDLIB_HYPER_INVENTORY_JSON||path.join(root,'artifacts/stdlib/stdlib_inventory.v1.json'),'utf8')).counts||{};}catch{notes.push('inventory unavailable');}
const lanes=Object.entries(tests).map(([lane,test_path])=>{
 const r=rows.find(r=>r.lane===lane);
 return {lane,test_path,required:true,status:r?.status==='pass'&&r.metrics_status==='pass'?'pass':mode==='off'?'not_run':'fail',
 reason:r?.reason|| (mode==='off'?'gate_disabled':''),metrics_status:r?.metrics_status||'not_run',
 blockers:r?.status==='pass'&&r.metrics_status==='pass'?[]:['native_lane_missing_or_failed']};
});
const blockers=lanes.every(r=>r.status==='pass')?[]:['native_lane_missing_or_failed'];
const doc={schema:'sounio.stdlib.hyper_execution_status.v1',generated_at_utc:new Date().toISOString(),command:'bash scripts/stdlib/stdlib_hyper_execution_gate.sh',
 mode,status_summary:status,totals,inventory:{hyper_active_files:counts.hyper_active_files||0,hyper_disabled_files:counts.hyper_disabled_files||0,hyper_stub_mod_files:counts.hyper_stub_mod_files||0},
 failures,golden_comparison:{status:oracleRun.exit_code===0?'pass':mode==='off'?'not_run':'fail',mismatches:failures.filter(r=>r.category==='golden_mismatch_or_unavailable'),oracle:'C++23',golden_sha256:fs.existsSync(golden)?sha(golden):'',artifact_directory:work},
 ignore_policy:{status:ignoreHits.length?'fail':'pass',violations:ignoreHits},native_lane_matrix_artifact:matrix,lane_statuses:lanes,blockers,notes,
 execution_rows:rows,oracle_build:oracleBuild,oracle_execution:oracleRun};
for(const[p,d]of [[out,doc],[matrix,{schema:'sounio.stdlib.native_lane_matrix.v1',generated_at_utc:doc.generated_at_utc,mode,status_summary:status,lanes,blockers,notes}]]){
 fs.mkdirSync(path.dirname(p),{recursive:true});fs.writeFileSync(p,JSON.stringify(d,null,2)+'\n');
}
console.log('STDLIB_HYPER_GATE_'+status.toUpperCase()+' artifacts='+work);
process.exitCode=status==='pass'||mode==='off'?0:1;
