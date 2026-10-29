#!/usr/bin/env node
import fs from 'node:fs';import os from 'node:os';import path from 'node:path';
import assert from 'node:assert/strict';import {spawnSync} from 'node:child_process';import {fileURLToPath} from 'node:url';
import {argvSnapshot,verifyArgvSnapshot} from './argv_inputs.mjs';
const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'stdlib-argv-contract-'));
const root=path.join(tmp,'archive'),fixture='tests/stdlib/group/probe.sio';
const write=(p,s)=>{fs.mkdirSync(path.dirname(p),{recursive:true});fs.writeFileSync(p,s);};
const run=(cmd,args,env={})=>{const r=spawnSync(cmd,args,{env:{...process.env,...env},encoding:'utf8',timeout:30000});assert.ifError(r.error);return r;};
let count=0;const ok=name=>{count++;console.log('PASS '+name);};
try {
 for(const file of ['run_stdlib_e2e.sh','argv_inputs.mjs','bounded_command.mjs','stdlib_e2e_diagnostics.mjs','emit_stdlib_e2e_result.mjs','run_full_suite.mjs','inventory.mjs','compound_gates.mjs','compiler_binding.mjs'])write(path.join(root,'scripts/stdlib',file),fs.readFileSync(path.join(repo,'scripts/stdlib',file)));
 write(path.join(root,'scripts/lib/resolve_souc.sh'),fs.readFileSync(path.join(repo,'scripts/lib/resolve_souc.sh')));
 fs.mkdirSync(path.join(root,'stdlib'),{recursive:true});
 const compiler=path.join(tmp,'compiler'),payload=path.join(tmp,'payload');
 write(compiler,'#!/usr/bin/env node\nconst fs=require("node:fs");const a=process.argv.slice(2);if(a[0]==="check")process.exit(0);if(a[0]!=="compile")process.exit(65);fs.copyFileSync(process.env.PAYLOAD,a[a.indexOf("-o")+1]);\n');fs.chmodSync(compiler,0o700);
 write(payload,'#!/usr/bin/env node\nconst fs=require("node:fs");fs.writeFileSync(process.env.OBSERVED,JSON.stringify(process.argv.slice(2)));if(process.env.MUTATE)fs.appendFileSync(process.env.MUTATE,"changed");\n');fs.chmodSync(payload,0o700);
 const args=['docs/a b;$(touch INJECTED)*?.txt',' leading file ','tools/quote\'and"double.txt',path.join(tmp,'external file')];
 for(const arg of args)write(path.resolve(root,arg),'immutable argv fixture\n');
 const test=path.join(root,fixture),sidecar=test+'.args';
 let n=0;
 const e2e=(name,extra={})=>{
  const output=path.join(tmp,'run'+(++n)),capture=output+'.capture',observed=output+'.argv';
  const r=run('bash',[path.join(root,'scripts/stdlib/run_stdlib_e2e.sh'),'--filter',fixture,'--json-out',output+'.json','--artifacts-dir',capture],{SOUC_BIN:compiler,PAYLOAD:payload,OBSERVED:observed,SOUNIO_STDLIB_PATH:path.join(root,'stdlib'),...extra});
  return {r,observed,capture,doc:fs.existsSync(output+'.json')?JSON.parse(fs.readFileSync(output+'.json')):null};
 };
 write(test,'//@ run-pass\nfn main() {}\n');write(sidecar,args.join('\n\n'));
 let p=e2e('literal');assert.equal(p.r.status,0,p.r.stderr);assert.deepEqual(JSON.parse(fs.readFileSync(p.observed)),args);assert(!fs.existsSync(path.join(root,'INJECTED')));
 const before=JSON.parse(fs.readFileSync(path.join(p.capture,'capture.json'))).argv_inputs;
 const after=JSON.parse(fs.readFileSync(path.join(p.capture,'complete.json'))).argv_inputs_after;
 assert.deepEqual(before,after);assert.equal(before[0].dependencies.length,4);ok('literal spaces/metacharacters/absolute/unterminated/blank lines and capture manifests');
 fs.unlinkSync(sidecar);p=e2e('absent');assert.equal(p.r.status,0,p.r.stderr);assert.deepEqual(JSON.parse(fs.readFileSync(p.observed)),[]);ok('absent sidecar keeps empty argv');
 fs.symlinkSync(path.join(tmp,'absent-target'),sidecar);p=e2e('dangling');assert.equal(p.r.status,1,p.r.stderr);assert(!fs.existsSync(p.observed));fs.unlinkSync(sidecar);ok('reject dangling sidecar');
 write(sidecar,'\n\n');p=e2e('empty');assert.equal(p.r.status,0,p.r.stderr);assert.deepEqual(JSON.parse(fs.readFileSync(p.observed)),[]);ok('empty sidecar keeps empty argv');
 for(const [name,content] of [['NUL',Buffer.from([0])],['UTF8',Buffer.from([255])],['missing','docs/missing.txt\n'],['directory','docs\n']]){
  write(sidecar,content);p=e2e(name);assert.equal(p.r.status,1,p.r.stderr);assert.equal(p.doc.results[0].status,'fail');assert.equal(p.doc.results[0].exit_code,65);assert(!fs.existsSync(p.observed));ok('reject '+name);
 }
 write(sidecar,'docs/missing.txt\n');
 for(const header of ['//@ check-only','//@ ignore vendor input missing']){
  write(test,header+'\nfn main() {}\n');p=e2e(header);assert.equal(p.r.status,0,p.r.stderr);assert(!fs.existsSync(p.observed));ok('preserve '+header);
 }
 write(test,'//@ run-pass\nfn main() {}\n');write(sidecar,args.join('\n'));
 for(const target of [sidecar,path.resolve(root,args[0]),path.resolve(root,args[2]),args[3]]){
  const saved=fs.readFileSync(target);p=e2e('mutation',{MUTATE:target});assert.notEqual(p.r.status,0);assert.match(p.r.stdout+p.r.stderr,/argv input changed/);const afterPath=path.join(p.capture,'argv-inputs.after.json');assert(fs.existsSync(afterPath));assert.equal(JSON.parse(fs.readFileSync(afterPath)).length,1);fs.writeFileSync(target,saved);ok('detect runtime mutation '+path.basename(target));
 }
 const snap=argvSnapshot(root,fixture);fs.unlinkSync(sidecar);assert.throws(()=>verifyArgvSnapshot(root,snap),/argv input changed/);write(sidecar,args.join('\n'));ok('detect sidecar deletion');
 // Full suite tests use a producer fixture: compiler success is not claimed here.
 write(path.join(root,'scripts/stdlib/run_stdlib_e2e.sh'),'#!/bin/bash\nexec node "$(dirname "$0")/producer.mjs" "$@"\n');
 write(path.join(root,'scripts/stdlib/producer.mjs'),`import fs from 'node:fs';import path from 'node:path';import {fileURLToPath} from 'node:url';const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..'),a=process.argv.slice(2),p=a[a.indexOf('--filter')+1],out=a[a.indexOf('--json-out')+1];if(process.env.FULL_MUTATE)fs.appendFileSync(path.resolve(root,process.env.FULL_MUTATE),'changed');const results=[{test_path:p,status:'pass',mode:'run-pass',reason:'',exit_code:0}];fs.writeFileSync(out,JSON.stringify({schema:'sounio.stdlib.e2e.result.v1',results,totals:{pass:1,fail:0,skip:0,total:1}}));`);
 const full=(name,env={})=>{
  const out=path.join(tmp,'full-'+name);
  const r=run(process.execPath,[path.join(root,'scripts/stdlib/run_full_suite.mjs'),'--out-dir',out,'--partition','file','--jobs','2'],env);
  return {r,out,doc:JSON.parse(fs.readFileSync(path.join(out,'RESULT.json')))};
 };
 p=full('positive');assert.equal(p.r.status,0,p.r.stderr);assert.equal(p.doc.totals.total,1);assert.equal(p.doc.totals.pass,1);
 assert.equal(JSON.parse(fs.readFileSync(path.join(p.out,'ARGV_WORKERS.before.json')))[0].inputs[0].dependencies.length,4);ok('full suite copies and guards docs/tools/external dependencies without changing selection');
 for(const [name,target] of [['docs',args[0]],['tools',args[2]],['sidecar',fixture+'.args']]){
  p=full(name,{FULL_MUTATE:target});assert.equal(p.r.status,1,p.r.stderr);assert.equal(p.doc.status_summary,'not_run');assert.equal(p.doc.totals.total,1);assert(p.doc.selection_errors.some(e=>e.worker_root&&/argv input changed/.test(e.error)));ok('full suite detects worker '+name+' mutation');
 }
 const rootTarget=path.resolve(root,args[0]),savedRoot=fs.readFileSync(rootTarget);
 p=full('original-root',{FULL_MUTATE:rootTarget});assert.equal(p.r.status,1);assert(p.doc.selection_errors.some(e=>!e.worker_root&&/argv input changed/.test(e.error)));fs.writeFileSync(rootTarget,savedRoot);ok('full suite detects original dependency mutation');
 console.log('STDLIB_ARGV_CONTRACT_OK controls='+count);
} finally {fs.rmSync(tmp,{recursive:true,force:true});}
