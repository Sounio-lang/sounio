// Synthetic filesystem metadata controls only: no compiler or ELF is executed.
import fs from 'node:fs';
import path from 'node:path';
import {admit,captureAdmission,digest,successfulCompilerResult,elfIdentity} from './admission.mjs';
if(!process.env.SLURM_JOB_ID)throw Error('metadata controls require Slurm receipt');
const dest=path.resolve(process.argv[2]??'');fs.mkdirSync(dest,{recursive:false});
const PATCH='127b070e8a0e3e9fae8891383797a6cd25cd13094429f5f19b42d8dbb73f74a6';
const results=[];
function write(p,s){fs.mkdirSync(path.dirname(p),{recursive:true});fs.writeFileSync(p,s)}
function json(p,v){write(p,JSON.stringify(v,null,2)+'\n');return {file:p,sha256:digest(p)}}
function pin(p){return {file:p,sha256:digest(p)}}
function tree(root,out){
 const files=[];
 function visit(d,base=''){for(const e of fs.readdirSync(d,{withFileTypes:true})){const file=base?base+'/'+e.name:e.name,p=path.join(d,e.name);if(e.isDirectory())visit(p,file);else files.push({file,bytes:fs.statSync(p).size,sha256:digest(p)})}}
 visit(root);files.sort((a,b)=>a.file<b.file?-1:a.file>b.file?1:0);
 return json(out,{schema:'sounio.sealed-tree.v1',files});
}
function setup(name){
 const d=path.join(dest,name),source=path.join(d,'source'),sdk=path.join(d,'sdk'),raw=path.join(d,'raw');
 write(path.join(source,'self-hosted/ir/lower.sio'),'SYNTHETIC metadata lower; never compiled\n');
 write(path.join(source,'stdlib/mod.sio'),'SYNTHETIC source graph dependency\n');
 write(path.join(sdk,'runtime.cpp'),'SYNTHETIC SDK dependency\n');
 write(raw,Buffer.from([0x7f,0x45,0x4c,0x46,0])); // metadata stub, never executable
 const auth={authorized_by:'root',patch_sha256:PATCH,raw,raw_sha256:digest(raw),source,sdk,
 lower_sha256:digest(path.join(source,'self-hosted/ir/lower.sio')),
 source_manifest:tree(source,path.join(d,'SOURCE.json')),sdk_manifest:tree(sdk,path.join(d,'SDK.json'))};
 write(path.join(d,'RAW_READY'),auth.raw_sha256+'  '+raw+'\n');
 auth.raw_ready=path.join(d,'RAW_READY');auth.raw_ready_sha256=digest(auth.raw_ready);
 write(path.join(d,'BUILD_STATUS'),'0\n');write(path.join(d,'ENDED_UTC'),'2026-10-29T14:00:00Z\n');
 write(path.join(d,'FINAL_GUARD'),'0\n');
 const proof={schema:'sounio.compiled-graph.root-proof.v1',finalized:true,raw,raw_sha256:auth.raw_sha256,
 source_manifest_sha256:auth.source_manifest.sha256,sdk_manifest_sha256:auth.sdk_manifest.sha256,
 build_status:pin(path.join(d,'BUILD_STATUS')),ended_utc:pin(path.join(d,'ENDED_UTC')),
 final_guards:[pin(path.join(d,'FINAL_GUARD'))]};
 auth.root_proof=json(path.join(d,'ROOT_PROOF.json'),proof);
 return {d,auth,proof,manifest:{patch_sha256:PATCH,candidate_lower_sha256:auth.lower_sha256}};
}
function reseal(x){x.auth.root_proof=json(x.auth.root_proof.file,x.proof)}
function ready(x,text){write(x.auth.raw_ready,text);x.auth.raw_ready_sha256=digest(x.auth.raw_ready)}
function check(name,mutate,accept=false){
 const x=setup(name);mutate(x);
 let caught=null;try{admit(x.auth,x.manifest)}catch(e){caught=e.message}
 if(accept?caught:!caught)throw Error(name+': unexpected '+(caught??'acceptance'));
 results.push({name,expected:accept?'accept':'refuse',observed:caught?'refuse':'accept',reason:caught});
}
check('complete',()=>{},true);
check('ready-wrong-hash',x=>ready(x,'0'.repeat(64)+'  '+x.auth.raw+'\n'));
check('ready-wrong-path',x=>ready(x,x.auth.raw_sha256+'  '+x.auth.raw+'.other\n'));
check('ready-malformed',x=>ready(x,x.auth.raw_sha256+' '+x.auth.raw+'\n'));
check('build-failed',x=>{write(x.proof.build_status.file,'1\n');x.proof.build_status=pin(x.proof.build_status.file);reseal(x)});
check('build-missing',x=>fs.unlinkSync(x.proof.build_status.file));
check('terminal-not-final',x=>{x.proof.finalized=false;reseal(x)});
check('terminal-missing',x=>{delete x.proof.ended_utc;reseal(x)});
check('root-graph-binding',x=>{x.proof.source_manifest_sha256='0'.repeat(64);reseal(x)});
check('source-mutated',x=>write(path.join(x.auth.source,'stdlib/mod.sio'),'changed\n'));
check('source-extra',x=>write(path.join(x.auth.source,'extra.sio'),'extra\n'));
check('source-missing',x=>fs.unlinkSync(path.join(x.auth.source,'stdlib/mod.sio')));
check('sdk-mutated',x=>write(path.join(x.auth.sdk,'runtime.cpp'),'changed\n'));
check('sdk-extra',x=>write(path.join(x.auth.sdk,'extra.hpp'),'extra\n'));
check('source-symlink',x=>{const f=path.join(x.auth.source,'stdlib/mod.sio');fs.unlinkSync(f);fs.symlinkSync(x.auth.raw,f)});
check('sdk-manifest-duplicate',x=>{
 const j=JSON.parse(fs.readFileSync(x.auth.sdk_manifest.file));j.files.push(j.files[0]);
 x.auth.sdk_manifest=json(x.auth.sdk_manifest.file,j);x.proof.sdk_manifest_sha256=x.auth.sdk_manifest.sha256;reseal(x)});
check('guard-nonzero',x=>{write(x.proof.final_guards[0].file,'1\n');x.proof.final_guards[0]=pin(x.proof.final_guards[0].file);reseal(x)});
check('guard-empty',x=>{x.proof.final_guards=[];reseal(x)});
check('lower-mismatch',x=>{x.auth.lower_sha256='0'.repeat(64)});
check('raw-mutated',x=>write(x.auth.raw,'changed'));
check('raw-ready-symlink',x=>{fs.renameSync(x.auth.raw_ready,x.auth.raw_ready+'.target');fs.symlinkSync(x.auth.raw_ready+'.target',x.auth.raw_ready)});
const x=setup('before-after');
fs.mkdirSync(path.join(x.d,'receipts'));
captureAdmission(x.auth,x.manifest,path.join(x.d,'receipts'),'before');
write(path.join(x.auth.sdk,'runtime.cpp'),'after mutation\n');
let refused=false;try{captureAdmission(x.auth,x.manifest,path.join(x.d,'receipts'),'after')}catch{refused=true}
if(!refused)throw Error('after mutation accepted');
results.push({name:'before-after-sdk-mutation',expected:'refuse',observed:'refuse'});
for(const[name,text,accept]of [
 ['clean','warning: operational note\n',true],
 ['error-zero','error: lower failed\n',false],
 ['coded-error-zero','error[E038]: conflict\n',false],
 ['ansi-error-zero','\u001b[31merror: failed\u001b[0m\n',false],
 ['diagnostic-code-zero','E137 unresolved\n',false],
 ['fatal-zero','fatal: failed\n',false]
]){
 const p=path.join(dest,'diagnostic-'+name);write(p+'.stdout',text);write(p+'.stderr','');
 let caught=false;try{successfulCompilerResult({code:0,signal:null,timedout:false},p)}catch{caught=true}
 if(caught===accept)throw Error('diagnostic control '+name);
 results.push({name,expected:accept?'accept':'refuse',observed:caught?'refuse':'accept'});
}
const elf=path.join(dest,'metadata-only.elf');write(elf,Buffer.from([0x7f,0x45,0x4c,0x46,0]));
const before=elfIdentity(elf);write(elf,Buffer.from([0x7f,0x45,0x4c,0x46,1]));
if(elfIdentity(elf).sha256===before.sha256)throw Error('ELF mutation not detected');
results.push({name:'ELF-before-after-hash',expected:'different',observed:'different'});
write(elf,'not an ELF');refused=false;try{elfIdentity(elf)}catch{refused=true}
if(!refused)throw Error('non-ELF accepted');
results.push({name:'ELF-magic',expected:'refuse',observed:'refuse'});
fs.writeFileSync(path.join(dest,'RESULT.json'),JSON.stringify({status:'PASS',scope:'synthetic-metadata-only',compiler_invocations:0,elf_executions:0,controls:results},null,2)+'\n');
console.log('SHADOW_ADMISSION_METADATA_CONTROLS_OK '+results.length);
