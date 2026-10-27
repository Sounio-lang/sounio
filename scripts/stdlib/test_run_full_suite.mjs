import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import test from 'node:test';
const repository=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
test('file partitions isolate relative outputs and refuse omitted cases',()=>{
 const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'stdlib-full-file-'));
 try{
  const root=path.join(tmp,'archive');
  for(const p of ['scripts/stdlib','tests/stdlib/group','stdlib'])fs.mkdirSync(path.join(root,p),{recursive:true});
  for(const name of ['run_full_suite.mjs','inventory.mjs'])fs.copyFileSync(path.join(repository,'scripts/stdlib',name),path.join(root,'scripts/stdlib',name));
  for(const name of ['a','b'])fs.writeFileSync(path.join(root,'tests/stdlib/group',name+'.sio'),'// metadata fixture\n');
  fs.writeFileSync(path.join(root,'stdlib','module.sio'),'// immutable fixture\n');
  fs.writeFileSync(path.join(root,'scripts/stdlib/run_stdlib_e2e.sh'),'#!/bin/bash\nexec node "$(dirname "$0")/fixture_runner.mjs" "$@"\n');
  fs.writeFileSync(path.join(root,'scripts/stdlib/fixture_runner.mjs'),"import fs from 'node:fs';import path from 'node:path';import {fileURLToPath} from 'node:url';\nimport {setTimeout} from 'node:timers/promises';\nconst root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..'),args=process.argv.slice(2);\nconst filter=args[args.indexOf('--filter')+1],output=args[args.indexOf('--json-out')+1];\nconst collision=path.join(root,'case-output.txt');fs.writeFileSync(collision,filter);\nawait setTimeout(100);\nif(fs.readFileSync(collision,'utf8')!==filter)throw Error('relative output collision');\nconst paths=['tests/stdlib/group/a.sio','tests/stdlib/group/b.sio'].filter(p=>p.startsWith(filter));\nconst results=paths.filter(p=>!(process.env.OMIT_A==='1'&&p.endsWith('/a.sio'))).map(test_path=>({test_path,status:'pass',mode:'check',reason:'',exit_code:0}));\nfs.writeFileSync(output,JSON.stringify({schema:'sounio.stdlib.e2e.result.v1',results,totals:{pass:results.length,fail:0,skip:0,total:results.length}}));\n");
  const run=(name,extra={})=>{
   const env={...process.env,STDLIB_FULL_CAPTURE:'0'};delete env.STDLIB_FULL_WORKER_DIR;Object.assign(env,extra);
   const out=path.join(tmp,name),result=spawnSync(process.execPath,[path.join(root,'scripts/stdlib/run_full_suite.mjs'),'--out-dir',out,'--jobs','2','--partition','file'],{env,encoding:'utf8',timeout:30000});
   assert.ifError(result.error);
   return {result,report:JSON.parse(fs.readFileSync(path.join(out,'RESULT.json'),'utf8'))};
  };
  const external=path.join(tmp,'private-workers');
  const positive=run('positive',{STDLIB_FULL_WORKER_DIR:external});
  assert.equal(positive.result.status,0,positive.result.stderr);
  assert.deepEqual(positive.report.totals,{pass:2,fail:0,skip:0,total:2});
  assert.deepEqual(positive.report.selection_errors,[]);
  assert.equal(new Set(positive.report.context.worker_roots).size,2);
  assert(positive.report.context.worker_roots.every(p=>p.startsWith(external+path.sep)));
  assert.equal(fs.existsSync(path.join(root,'case-output.txt')),false);
  for(const worker of positive.report.context.worker_roots)assert(fs.readFileSync(path.join(worker,'case-output.txt'),'utf8').endsWith('.sio'));
  const omitted=run('omitted',{OMIT_A:'1'});
  assert.equal(omitted.result.status,1);
  assert.equal(omitted.report.status_summary,'not_run');
  assert(omitted.report.selection_errors.some(e=>e.path==='tests/stdlib/group/a.sio'&&e.error==='missing test result'));
 }finally{fs.rmSync(tmp,{recursive:true,force:true});}
});
