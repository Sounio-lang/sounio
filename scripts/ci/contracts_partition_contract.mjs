#!/usr/bin/env node
// History/working-tree conformance for filtered full-history checkouts.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import assert from 'node:assert/strict';
import {execFileSync,spawnSync} from 'node:child_process';
const root=process.argv[2];assert(root,'usage: contracts_partition_contract.mjs REPO [--selftest]');
const workflow=fs.readFileSync(path.join(root,'.github/workflows/ci.yml'),'utf8');
function job(w,id){const a=w.indexOf('  '+id+':\n');assert(a>=0,'missing '+id);const r=w.slice(a+1),e=r.search(/^  [A-Za-z0-9_-]+:/m);return e<0?w.slice(a):w.slice(a,a+1+e);}
function check(w){
 for(const id of ['impact','contracts-core','contracts-ontology']){
  const b=job(w,id),checkout=b.split(/(?=^      - )/m).find(s=>s.startsWith('      - uses: actions/checkout@v4\n'));
  assert(checkout?.includes('          fetch-depth: 0\n          filter: blob:none\n'),'full metadata history and lazy blobs required: '+id);
  assert(!checkout.includes('sparse-checkout:'),'current working tree must be complete: '+id);
 }
 assert(job(w,'contracts-core').includes('shard: [core-a, core-b, core-c, research]'),'required matrix incomplete');
 assert(job(w,'fast-checks').includes('run: node scripts/ci/contracts_partition_contract.mjs "$GITHUB_WORKSPACE" --selftest'),'history fixture disconnected');
}
check(workflow);
if(process.argv.includes('--selftest')){
 for(const mutate of [
  s=>s.replace('fetch-depth: 0','fetch-depth: 1'),
  s=>s.replace('filter: blob:none','filter: tree:0'),
  s=>s.replace('filter: blob:none','filter: blob:none\n          sparse-checkout: scripts'),
  s=>s.replace('shard: [core-a, core-b, core-c, research]','shard: [core-a, core-c, research]'),
 ]){const bad=mutate(workflow);assert.notEqual(bad,workflow,'ineffective control');assert.throws(()=>check(bad),'unsafe checkout accepted');}
 const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'contracts-history-'));
 const src=path.join(tmp,'source'),remote=path.join(tmp,'origin.git'),partial=path.join(tmp,'partial'),shallow=path.join(tmp,'shallow');
 const git=(dir,args)=>execFileSync('git',['-C',dir,...args],{encoding:'utf8',stdio:['ignore','pipe','pipe']}).trim();
 try{
  fs.mkdirSync(src);git(src,['init','-q','-b','main']);git(src,['config','user.name','CI fixture']);git(src,['config','user.email','fixture@example.invalid']);git(src,['config','commit.gpgsign','false']);
  fs.writeFileSync(path.join(src,'source.sio'),'historical source fixture\n');git(src,['add','.']);git(src,['commit','-qm','historical fixture']);
  const old=git(src,['rev-parse','HEAD']),oldBlob=git(src,['rev-parse','HEAD:source.sio']);
  git(src,['tag','historical-fixture',old]);git(src,['branch','historical-branch',old]);
  fs.writeFileSync(path.join(src,'source.sio'),'current source fixture\n');fs.writeFileSync(path.join(src,'current.txt'),'all current files must exist\n');git(src,['add','.']);git(src,['commit','-qm','current fixture']);
  execFileSync('git',['clone','-q','--bare','--no-hardlinks',src,remote],{stdio:'pipe'});git(remote,['config','uploadpack.allowFilter','true']);
  execFileSync('git',['clone','-q','--filter=blob:none','--no-checkout','file://'+remote,partial],{stdio:'pipe'});
  git(partial,['config','protocol.file.allow','always']);git(partial,['checkout','-q','main']);
  assert.equal(git(partial,['config','--get','remote.origin.promisor']),'true');
  assert.equal(git(partial,['config','--get','remote.origin.partialclonefilter']),'blob:none');
  assert.equal(git(partial,['log','--format=%H']),git(src,['log','--format=%H']),'history changed');
  assert.equal(git(partial,['rev-parse','refs/tags/historical-fixture']),old,'historical tag lost');
  assert.equal(git(partial,['rev-parse','refs/remotes/origin/historical-branch']),old,'historical branch lost');
  assert.equal(git(partial,['ls-files']),git(src,['ls-files']),'current file inventory changed');
  for(const file of ['source.sio','current.txt'])assert.equal(fs.readFileSync(path.join(partial,file),'utf8'),fs.readFileSync(path.join(src,file),'utf8'),'current file missing');
  assert.equal(git(partial,['diff','--name-only',old,'HEAD']),git(src,['diff','--name-only',old,'HEAD']),'historical diff changed');
  assert(git(partial,['rev-list','--objects','--missing=print','HEAD']).includes('?'+oldBlob),'fixture did not defer the old blob');
  assert.equal(git(partial,['show',old+':source.sio']),'historical source fixture','lazy historical content unavailable');
  assert(!git(partial,['rev-list','--objects','--missing=print','HEAD']).includes('?'+oldBlob),'historical blob did not materialize');
  execFileSync('git',['clone','-q','--depth','1','file://'+remote,shallow],{stdio:'pipe'});
  const missing=spawnSync('git',['-C',shallow,'cat-file','-e',old+'^{commit}'],{encoding:'utf8'});
  assert.notEqual(missing.status,0,'shallow negative control unexpectedly has complete history');
  console.log('CONTRACTS_REAL_FILTERED_HISTORY_AND_LAZY_CONTENT_PASS');
  console.log('CONTRACTS_SHALLOW_HISTORY_NEGATIVE_CONTROL_PASS');
 }finally{fs.rmSync(tmp,{recursive:true,force:true});}
}
console.log('CONTRACTS_FULL_HISTORY_CHECKOUT_CONTRACT_PASS');
