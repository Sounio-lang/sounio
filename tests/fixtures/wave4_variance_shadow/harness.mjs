// Orchestration and byte identity only. Numerical agreement belongs to C++23.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../..');
const fixture='tests/fixtures/wave4_variance_shadow/';
const manifestPath=path.join(root,fixture,'manifest.json');
const manifest=JSON.parse(fs.readFileSync(manifestPath,'utf8'));
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const fail=s=>{throw Error(s)};
function guard(){
 for(const r of [...manifest.files,...manifest.canonical.flatMap(r=>[
  {file:r.file,sha256:r.sha256},...(r.oracle?[{file:r.oracle,sha256:r.oracle_sha256}]:[])])])
  if(sha(path.join(root,r.file))!==r.sha256)fail('source identity changed: '+r.file);
}
async function run(exe,args,prefix,seconds,env={}){
 const out=fs.openSync(prefix+'.stdout','wx'),err=fs.openSync(prefix+'.stderr','wx');
 let p,killTimer,timer;
 try{
  p=spawn(exe,args,{cwd:root,env:{...process.env,...env},stdio:['ignore',out,err],detached:true});
  let timedout=false;
  const result=await new Promise((resolve,reject)=>{
   p.once('error',reject);
   p.once('close',(code,signal)=>resolve({code,signal,timedout}));
   timer=setTimeout(()=>{
    timedout=true;
    try{process.kill(-p.pid,'SIGTERM')}catch{}
    killTimer=setTimeout(()=>{try{process.kill(-p.pid,'SIGKILL')}catch{}},2000);
   },seconds*1000);
  });
  fs.writeFileSync(prefix+'.result.json',JSON.stringify({exe,args,seconds,...result},null,2)+'\n',{flag:'wx'});
  return result;
 }finally{clearTimeout(timer);clearTimeout(killTimer);fs.closeSync(out);fs.closeSync(err)}
}
function ok(r,where){if(r.code!==0||r.signal||r.timedout)fail(where+' failed')}
function refused(r,where){if(r.code===0||r.signal||r.timedout)fail(where+' needs ordinary nonzero status')}
const mode=process.argv[2]??'--inventory';
guard();
if(mode==='--inventory'){
 console.log(JSON.stringify({status:'DRAFT_NOT_QUALIFIED',manifest,compiler_invocations:0},null,2));
}else{
 if(!process.env.SLURM_JOB_ID)fail('build/native stages require Slurm');
 const dest=path.resolve(process.argv[3]??fail('provide fresh results directory'));
 fs.mkdirSync(dest,{recursive:false});
 const cxx=process.env.CXX??fail('explicit qualified CXX required');
 const cxxSha=process.env.CXX_SHA256??fail('explicit CXX_SHA256 required');
 if(sha(cxx)!==cxxSha)fail('CXX pin mismatch');
 const oracle=path.join(dest,'reference');
 try{
  ok(await run(cxx,['-std=c++23','-O2','-Wall','-Wextra','-Werror',
   path.join(root,'tests/reference/wave4_variance_shadow_reference.cpp'),'-o',oracle],
   path.join(dest,'oracle.build'),120),'C++23 oracle build');
  ok(await run(oracle,['--emit'],path.join(dest,'oracle.emit'),10),'oracle emit');
  const expected=fs.readFileSync(path.join(dest,'oracle.emit.stdout'),'utf8');
  for(const[kind,text]of [
   ['complete',expected],['wrong',expected.replace('0 4','0 5')],
   ['drop',expected.split('\n').slice(1).join('\n')],
   ['extra',expected+expected.split('\n')[0]+'\n'],
   ['truncated','0 4']]){
   const f=path.join(dest,kind+'.txt');fs.writeFileSync(f,text);
   const r=await run(oracle,['--check',f],path.join(dest,'control.'+kind),10);
   kind==='complete'?ok(r,kind):refused(r,kind);
  }
  if(mode==='--oracle-controls'){
   fs.writeFileSync(path.join(dest,'ORACLE_COMPONENT_ONLY'),'PASS; no Sounio compiler invoked\n');
  }else if(mode==='--language-components'){
   // This envelope records explicit root authorization. It does not mint permission.
   const authPath=process.env.SHADOW_AUTHORIZATION??fail('root authorization envelope required');
   const auth=JSON.parse(fs.readFileSync(authPath,'utf8'));
   const raw=path.resolve(auth.raw),source=path.resolve(auth.source);
   if(auth.authorized_by!=='root'||auth.patch_sha256!==manifest.patch_sha256)
    fail('authorization/patch identity mismatch');
   if(sha(auth.raw_ready)!==auth.raw_ready_sha256||sha(raw)!==auth.raw_sha256)
    fail('RAW_READY/compiler identity mismatch');
   if(auth.lower_sha256!==manifest.candidate_lower_sha256||
      sha(path.join(source,'self-hosted/ir/lower.sio'))!==auth.lower_sha256)
    fail('candidate lower identity mismatch; never apply patch in this harness');
   fs.copyFileSync(authPath,path.join(dest,'AUTHORIZATION.json'));
   const env={SOUNIO_STDLIB_PATH:path.join(source,'stdlib'),
    SOUNIO_LOOP_CHECKPOINT_ELISION:'0',SOUNIO_NV2_IR_TRACE_FULL:'1',
    SOUNIO_NV2_IR_TRACE_FULL_STAGE:'source'};
   for(const loop of ['0','1'])for(const optimize of ['0','1']){
    const tag='loop'+loop+'.opt'+optimize;
    const flags=optimize==='1'?['-O']:[];
    const e={...env,SOUNIO_LOOP_REGIONS:loop};
    for(const rel of [manifest.native,...manifest.canonical.map(r=>r.file)]){
     const base=path.basename(rel,'.sio'),prefix=path.join(dest,tag+'.'+base);
     const elf=prefix+'.elf';
     ok(await run(raw,['check',path.join(root,rel)],prefix+'.check',1200,e),'check '+rel);
     ok(await run(raw,['compile',...flags,path.join(root,rel),'-o',elf],prefix+'.compile',1200,e),'compile '+rel);
     if(!fs.existsSync(elf))fail('successful compile missing ELF');
     ok(await run(elf,[],prefix+'.native',120,e),'native '+rel);
     if(rel===manifest.native)
      ok(await run(oracle,['--check',prefix+'.native.stdout'],prefix+'.oracle',10),'numeric agreement');
     const row=manifest.canonical.find(r=>r.file===rel);
     if(row?.marker&&!fs.readFileSync(prefix+'.native.stdout','utf8').includes(row.marker))
      fail('missing canonical marker '+rel);
    }
    const m=path.join(dest,tag+'.assertion-mutant'),mutant=m+'.elf';
    ok(await run(raw,['check',path.join(root,manifest.assertion_mutant)],m+'.check',1200,e),'mutant check');
    ok(await run(raw,['compile',...flags,path.join(root,manifest.assertion_mutant),'-o',mutant],m+'.compile',1200,e),'mutant compile');
    refused(await run(mutant,[],m+'.native',30,e),'assertion mutant');
    for(const row of manifest.semantic_refusals){
     const n=path.join(dest,tag+'.'+path.basename(row.file,'.sio')),elf=n+'.elf';
     ok(await run(raw,['check',path.join(root,row.file)],n+'.check',1200,e),'semantic input check');
     refused(await run(raw,['compile',...flags,path.join(root,row.file),'-o',elf],n+'.compile',1200,e),'semantic refusal');
     if(fs.existsSync(elf))fail('semantic refusal published ELF');
     const output=fs.readFileSync(n+'.compile.stdout','utf8')+fs.readFileSync(n+'.compile.stderr','utf8');
     if(!output.includes(row.diagnostic))fail('wrong refusal cause; parse/check/other error is not acceptance');
    }
   }
   if(sha(raw)!==auth.raw_sha256||
    sha(path.join(source,'self-hosted/ir/lower.sio'))!==auth.lower_sha256)fail('compiler/source changed');
   fs.writeFileSync(path.join(dest,'LANGUAGE_COMPONENTS_ONLY'),
    'No full qualification: CFG, private Lowerer, wide profile and existing regression receipts remain mandatory.\n');
  }else fail('unknown mode');
 }finally{guard()}
}
