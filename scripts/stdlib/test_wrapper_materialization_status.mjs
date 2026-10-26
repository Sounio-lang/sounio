#!/usr/bin/env node
// Operational launcher regression. /bin/true stands in for an explicit raw ELF;
// no Sounio compiler, build, native scientific code, or real Python is invoked.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'sounio-materialization-status-'));
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const guards=path.join(tmp,'guards');fs.mkdirSync(guards);const denylog=path.join(tmp,'python-attempts');
for(const name of ['python','python3','python3.12']){const p=path.join(guards,name);fs.writeFileSync(p,'#!/bin/bash\nprintf "%s\\n" "$0 $*" >> "$PYTHON_DENY_LOG"\nexit 125\n');fs.chmodSync(p,0o700);assert.equal(fs.realpathSync(p),p);assert(fs.statSync(p).mode&0o111);}
const clean={PATH:guards+':/usr/bin:/bin',HOME:tmp,PYTHON_DENY_LOG:denylog,LANG:'C',MADAROS_STACK_KB:'0'};
for(const name of ['python','python3','python3.12'])assert.equal(spawnSync(name,['--controlled-deny-probe'],{env:clean}).status,125);
fs.renameSync(denylog,path.join(tmp,'controlled-deny-probes.log'));fs.writeFileSync(denylog,'');
const variants={};
for(const variant of ['before','after']){
 const root=path.join(tmp,variant);fs.mkdirSync(path.join(root,'bin'),{recursive:true});fs.mkdirSync(path.join(root,'scripts/lib'),{recursive:true});
 fs.copyFileSync(path.join(repo,'scripts/lib/resolve_madaros.sh'),path.join(root,'scripts/lib/resolve_madaros.sh'));
 fs.writeFileSync(path.join(root,'scripts/lib/materialize_madaros_prebuilt.sh'),'#!/usr/bin/env bash\nprintf "%s\\n" "$MATERIALIZE_RC" >> "$MATERIALIZE_LOG"\nexit "$MATERIALIZE_RC"\n');
 for(const name of ['souc','madaros']){
  let source=fs.readFileSync(path.join(repo,'bin',name),'utf8');
  if(variant==='before'){
   const fixed='  if bash "$ROOT_DIR/scripts/lib/materialize_madaros_prebuilt.sh"; then\n    :\n  else\n    _mat_rc=$?';
   assert(source.includes(fixed));source=source.replace(fixed,'  if ! bash "$ROOT_DIR/scripts/lib/materialize_madaros_prebuilt.sh"; then\n    _mat_rc=$?');
  }
  fs.writeFileSync(path.join(root,'bin',name),source,{mode:0o700});
 }
 variants[variant]=root;
}
const records=[];
const beforeHashes=Object.fromEntries(Object.values(variants).flatMap(r=>['bin/souc','bin/madaros','scripts/lib/resolve_madaros.sh','scripts/lib/materialize_madaros_prebuilt.sh'].map(p=>[path.join(r,p),sha(path.join(r,p))])));
let n=0;function check(label,f){f();console.log(`PASS ${++n} ${label}`);}
function run(variant,name,rc,extra={}){const result=spawnSync('bash',[path.join(variants[variant],'bin',name),'--version'],{env:{...clean,MATERIALIZE_RC:String(rc),MATERIALIZE_LOG:path.join(tmp,'materializer-calls'),...extra},encoding:'utf8'});records.push({variant,launcher:name,materializer_rc:rc,explicit_env:extra,status:result.status,signal:result.signal,stdout:result.stdout,stderr:result.stderr});return result;}
for(const name of ['souc','madaros'])for(const rc of [78,125]){
 check(`${name} before false-success0 versus after exact materializer${rc}`,()=>{assert.equal(run('before',name,rc).status,0);assert.equal(run('after',name,rc).status,rc);});
 check(`${name} explicit valid ELF override survives materializer${rc}`,()=>{assert.equal(run('before',name,rc,{MADAROS_RAW_BIN:'/bin/true'}).status,0);assert.equal(run('after',name,rc,{MADAROS_RAW_BIN:'/bin/true'}).status,0);});
 check(`${name} strict committed policy still refuses override after materializer${rc}`,()=>{const a=run('before',name,rc,{MADAROS_RAW_BIN:'/bin/true',SOUNIO_REQUIRE_COMMITTED_MADAROS:'1'}),b=run('after',name,rc,{MADAROS_RAW_BIN:'/bin/true',SOUNIO_REQUIRE_COMMITTED_MADAROS:'1'});assert.equal(a.status,78);assert.equal(b.status,78);});
}
for(const name of ['souc','madaros']){
 check(`${name} unusable explicit override remains refusal78`,()=>{assert.equal(run('before',name,78,{MADAROS_RAW_BIN:'/no/such/raw'}).status,78);assert.equal(run('after',name,78,{MADAROS_RAW_BIN:'/no/such/raw'}).status,78);});
 check(`${name} successful materialization explicit route unchanged`,()=>{assert.equal(run('before',name,0,{MADAROS_RAW_BIN:'/bin/true'}).status,0);assert.equal(run('after',name,0,{MADAROS_RAW_BIN:'/bin/true'}).status,0);});
}
check('runtime Python deny log empty',()=>assert.equal(fs.readFileSync(denylog).length,0));
for(const [p,digest]of Object.entries(beforeHashes))assert.equal(sha(p),digest,p);
fs.writeFileSync(path.join(tmp,'ACTUAL_ROWS.json'),JSON.stringify(records,null,2)+'\n');
const identities={scope:'Operational Bash launchers and fake materializer; /bin/true is not a compiler',true:sha('/bin/true'),node:{path:process.execPath,sha256:sha(process.execPath)},variants:Object.fromEntries(Object.entries(variants).map(([v,r])=>[v,Object.fromEntries(['bin/souc','bin/madaros','scripts/lib/resolve_madaros.sh','scripts/lib/materialize_madaros_prebuilt.sh'].map(p=>[p,sha(path.join(r,p))]))])),guards:Object.fromEntries(['python','python3','python3.12'].map(n=>[n,sha(path.join(guards,n))]))};
fs.writeFileSync(path.join(tmp,'IDENTITIES.json'),JSON.stringify(identities,null,2)+'\n');
console.log(`PASS ${n} operational controls; retained source variants and identities at ${tmp}`);
