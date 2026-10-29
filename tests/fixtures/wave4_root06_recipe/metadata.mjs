// ROOT06 metadata/recipe contract only. No compiler is launched by this module.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {verifyTree} from '../wave4_variance_shadow/admission.mjs';
const bad=s=>{throw Error(s)};
export const CAPACITY_SHA='eaa61d90e8d2d1897f545636c06496f8b0791fbbf206dca5bd4fa4151926ac60';
const hash=s=>typeof s==='string'&&/^[0-9a-f]{64}$/.test(s);
export function digest(p){
 const h=crypto.createHash('sha256'),fd=fs.openSync(p,'r'),b=Buffer.alloc(262144);
 try{for(let n;(n=fs.readSync(fd,b,0,b.length,null))>0;)h.update(b.subarray(0,n))}
 finally{fs.closeSync(fd)}
 return h.digest('hex');
}
export function pinned(r){
 if(!r||!path.isAbsolute(r.file)||!hash(r.sha256)||!fs.lstatSync(r.file).isFile()||fs.lstatSync(r.file).isSymbolicLink()||!Number.isSafeInteger(r.mode)||(fs.lstatSync(r.file).mode&0o7777)!==r.mode||digest(r.file)!==r.sha256)bad('pinned input identity mismatch');
 return r.file;
}
const json=r=>JSON.parse(fs.readFileSync(pinned(r),'utf8'));
function run(exe,args,options={}){
 const r=spawnSync(exe,args,{encoding:'buffer',maxBuffer:33554432,...options});
 if(r.error||r.status!==0)bad('metadata command refused: '+exe);
 return r.stdout;
}
export function qualification(role,item,producer){
 const fixed={'producer-flow-base104':104,'producer-bounds-base34':34,'producer-metadata-base211':211};
 const m=json(item.manifest),r=json(item.receipt);
 if(m.schema!=='sounio.qualification.cases.v1'||m.role!==role||!Array.isArray(m.ids)||m.ids.length===0||
  m.ids.some((id,i)=>typeof id!=='string'||!id||(i>0&&m.ids[i-1]>=id)))bad('exact original case-ID manifest required');
 if(fixed[role]!==undefined&&m.ids.length!==fixed[role])bad('base case manifest count mismatch');
 if(r.schema!=='sounio.producer.qualification-adapter.v1'||r.role!==role||r.completed!==true||
  r.manifest_sha256!==item.manifest.sha256||r.producer_source_sha256!==producer.source.sha256||
  r.producer_elf_sha256!==producer.elf.sha256||r.failed!==0||r.skipped!==0||
  r.required!==m.ids.length||r.passed!==m.ids.length||!Array.isArray(r.cases)||
  JSON.stringify(r.cases.map(c=>c.id))!==JSON.stringify(m.ids)||r.cases.some(c=>c.status!=='pass'))bad('qualification does not bind actual final producer/all exact cases');
 pinned(r.adapter);
 if(!Array.isArray(r.originals)||r.originals.length===0||typeof r.job_id!=='string'||!r.job_id)bad('original actual-run receipts required');
 for(const original of r.originals)pinned(original);
 return {role,cases:m.ids.length,receipt_sha256:item.receipt.sha256,manifest_sha256:item.manifest.sha256};
}
export function layout(p){
 const R=p.run_root;
 if(typeof R!=='string'||!path.isAbsolute(R)||path.resolve(R)!==R||R.includes('root-integrated05'))bad('fresh dedicated ROOT06 run root required');
 const paths={R,S_GIT:path.join(R,'source-git'),S:path.join(R,'source-build'),D:path.join(R,'results'),
  PREFLIGHT:path.join(R,'preflight'),CACHE:path.join(R,'cache-private'),LOCK:path.join(R,'build.lock'),
  SEED:path.join(R,'seed-private/producer.elf'),OLOCK:path.join(R,'recipe.lock')};
 for(const x of p.sdk??[])if(R===x.root||R.startsWith(x.root+'/')||x.root.startsWith(R+'/'))bad('run/SDK roots overlap');
 if(R===p.repository||R.startsWith(p.repository+'/'))bad('run outputs must be outside Git source worktree');
 return paths;
}
export function verifyPlan(p,phase='review'){
 if(p.schema!=='sounio.root06.recipe.v1'||!/^[0-9a-f]{40}$/.test(p.head)||!/^[0-9a-f]{40}$/.test(p.tree))bad('supplied actual HEAD/tree required');
 const L=layout(p);
 for(const key of ['node','bash','git','tar','time','timeout','cxx'])pinned(p.tools[key]);
 for(const row of Object.values(p.tools))pinned(row);
 if(!Number.isSafeInteger(p.build_seconds)||p.build_seconds<1||p.build_seconds>10800||!Number.isSafeInteger(p.preflight_seconds)||p.preflight_seconds<1||p.preflight_seconds>1200)bad('explicit bounded deadlines required');
 pinned(p.tool_environment.file); // nested pin descriptor, with separate context_R
 if(!path.isAbsolute(p.tool_environment.context_R))bad('qualified private environment context required');
 for(const key of ['archive','git_census','capacity','worker_optin','preflight_recipe','preflight_cases','metadata_helper','graph_helper','build_template','magic'])pinned(p.inputs[key]);
 if(p.inputs.graph_helper.file!==new URL('../wave4_variance_shadow/admission.mjs',import.meta.url).pathname)bad('actual graph helper not bound');
 if(p.inputs.metadata_helper.file!==new URL(import.meta.url).pathname||p.inputs.build_template.file!==new URL('../../../scripts/stdlib/root06_build_template.sh',import.meta.url).pathname)bad('actual metadata/template helpers not bound to plan');
 if(p.inputs.capacity.sha256!==CAPACITY_SHA)bad('exact capacity eaa patch required');
 pinned(p.pristine_manifest);pinned(p.build_manifest);
 const git=p.tools.git.file;
 if(run(git,['-C',p.repository,'rev-parse',p.head+'^{tree}']).toString().trim()!==p.tree)bad('HEAD/tree binding mismatch');
 if(phase==='review'&&run(git,['-C',p.repository,'rev-parse','HEAD']).toString().trim()!==p.head)bad('source HEAD is not actual current reviewed HEAD');
 const census=run(git,['-C',p.repository,'ls-tree','-rz','--full-tree',p.head]);
 if(!census.equals(fs.readFileSync(p.inputs.git_census.file)))bad('full Git census differs; selected subtrees are forbidden');
 const archiveFD=fs.openSync(p.inputs.archive.file,'r');
 try{if(run(git,['get-tar-commit-id'],{stdio:[archiveFD,'pipe','pipe']}).toString().trim()!==p.head)bad('archive origin commit mismatch')}
 finally{fs.closeSync(archiveFD)}
 const source=run(git,['-C',p.repository,'show',p.head+':self-hosted/compiler/lean_single.sio']);
 pinned(p.producer.source);pinned(p.producer.elf);
 if((fs.statSync(p.producer.elf.file).mode&0o111)===0)bad('qualified producer ELF must already be executable');
 if(crypto.createHash('sha256').update(source).digest('hex')!==p.producer.source.sha256)bad('final producer source differs from source Git graph');
 const magic=Buffer.alloc(4),fd=fs.openSync(p.producer.elf.file,'r');
 try{if(fs.readSync(fd,magic,0,4,0)!==4||magic.toString('hex')!=='7f454c46')bad('producer ELF magic required')}finally{fs.closeSync(fd)}
 const roles=['producer-flow-base104','producer-bounds-base34','producer-metadata-base211','producer-frontiers'];
 if(Object.keys(p.qualifications).sort().join()!==roles.slice().sort().join())bad('four exact qualification roles required');
 const qualified=roles.map(role=>qualification(role,p.qualifications[role],p.producer));
 if(!Array.isArray(p.sdk)||p.sdk.length===0)bad('immutable complete private SDK required');
 for(const sdk of p.sdk){pinned(sdk.manifest);verifyTree(sdk.root,sdk.manifest)}
 if(phase!=='review'){
  verifyTree(L.S_GIT,p.pristine_manifest);
  verifyGitTree(p,L.S_GIT,census);
 }
 if(phase==='copy'){verifyTree(L.S,p.pristine_manifest);verifyGitTree(p,L.S,census)}
 if(phase==='build'||phase==='overlay'||phase==='final'||phase==='preflight'){
  verifyTree(L.S,p.build_manifest);verifyOverlay(p,L);
 }
 if(phase==='build'){
  verifyPreflight(p,L);
  const ready=JSON.parse(fs.readFileSync(path.join(L.R,'PREFLIGHT_READY'),'utf8'));
  if(ready.phase!=='preflight'||ready.head!==p.head||ready.tree!==p.tree||ready.run_root!==L.R||
   ready.producer_elf_sha256!==p.producer.elf.sha256||ready.build_manifest_sha256!==p.build_manifest.sha256||ready.guards_complete!==true)bad('preflight readiness graph binding mismatch');
 }
 if(phase==='final'){
  pinned({...p.producer.elf,file:L.SEED});
 }
 return {schema:'sounio.root06.validated-plan.v1',phase,head:p.head,tree:p.tree,producer_elf_sha256:p.producer.elf.sha256,build_manifest_sha256:p.build_manifest.sha256,layout:L,qualified,status:0};
}
function verifyGitTree(p,root,census){
 const rows=census.toString('utf8').split('\0').filter(Boolean);
 const expected=new Map(rows.map(line=>{
  const m=/^(100644|100755|120000) blob ([0-9a-f]{40})\t(.+)$/.exec(line);
  if(!m)bad('unsupported Git census object/path');
  return [m[3],{mode:m[1],blob:m[2]}];
 }));
 const actual=[];
 function walk(base=''){
  for(const name of fs.readdirSync(path.join(root,base))){
   const rel=base?base+'/'+name:name,q=path.join(root,rel),st=fs.lstatSync(q);
   if(st.isDirectory())walk(rel);
   else{
    const want=expected.get(rel);if(!want)bad('extra path outside complete Git census');
    const symlink=st.isSymbolicLink(),mode=symlink?'120000':(st.mode&0o111)?'100755':'100644';
    if(mode!==want.mode)bad('Git object mode/type mismatch');
    const bytes=symlink?fs.readlinkSync(q,{encoding:'buffer'}):fs.readFileSync(q);
    const blob=crypto.createHash('sha1').update(Buffer.from('blob '+bytes.length+'\0')).update(bytes).digest('hex');
    if(blob!==want.blob)bad('Git blob mismatch: '+rel);
    actual.push(rel);
   }
  }
 }
 walk();if(actual.length!==expected.size)bad('missing Git census input');
}
function patchPaths(text){
 return [...new Set([...text.matchAll(/^\+\+\+ b\/([^\r\n]+)$/gm)].map(m=>m[1]))].sort();
}
function verifyOverlay(p,L){
 const before=json(p.pristine_manifest),after=json(p.build_manifest);
 const bm=new Map(before.nodes.map(n=>[n.file,n])),am=new Map(after.nodes.map(n=>[n.file,n]));
 if(before.root_mode!==after.root_mode||JSON.stringify([...bm.keys()])!==JSON.stringify([...am.keys()]))bad('overlay changed full path census/root mode');
 const allowed=patchPaths(fs.readFileSync(p.inputs.capacity.file,'utf8'));
 if(JSON.stringify(patchPaths(fs.readFileSync(p.inputs.worker_optin.file,'utf8')))!==JSON.stringify(['self-hosted/compiler/main.sio']))bad('Worker opt-in patch scope');
 allowed.push('self-hosted/compiler/main.sio');allowed.sort();
 const changed=[];
 for(const [file,b]of bm){
  const a=am.get(file);
  if(JSON.stringify(b)!==JSON.stringify(a)){
   if(a.type!=='file'||b.type!=='file'||a.mode!==b.mode||!allowed.includes(file))bad('undeclared overlay path/type/mode change');
   changed.push(file);
  }
 }
 if(JSON.stringify(changed)!==JSON.stringify(allowed))bad('capacity+Worker overlay changed-path census mismatch');
 const main='self-hosted/compiler/main.sio',s=fs.readFileSync(path.join(L.S_GIT,main),'utf8');
 const importLine='use parser::ast::*',entry='fn main() with IO, Mut, Panic, Div, Alloc {';
 if(s.split(importLine).length!==2||s.split(entry).length!==2)bad('Worker opt-in preimage not unique');
 const canonical=s.replace(importLine,importLine+'\nuse native::worker_proof::{worker_lower_backend_opt_in}')
  .replace(entry,entry+'\n    worker_lower_backend_opt_in()');
 if(fs.readFileSync(path.join(L.S,main),'utf8')!==canonical)bad('Worker opt-in differs from exact two additions');
}
export function verifyPreflight(p,L=layout(p)){
 const m=json(p.inputs.preflight_cases);
 const r=JSON.parse(fs.readFileSync(path.join(L.PREFLIGHT,'PREFLIGHT_RESULT.json'),'utf8'));
 if(m.schema!=='sounio.root06.preflight-cases.v1'||!Array.isArray(m.cases)||m.cases.length===0)bad('exact supplied preflight cases required');
 const ids=m.cases.map(c=>c.id),roles=new Set(m.cases.map(c=>c.role));
 if(ids.some((id,i)=>typeof id!=='string'||!id||(i>0&&ids[i-1]>=id)))bad('ordered unique preflight IDs required');
 for(const role of ['runtime-pthread','seed-cli-scalar-oracle','binding-stamp'])if(!roles.has(role))bad('required short preflight role absent');
 if(r.schema!=='sounio.root06.preflight-result.v1'||r.head!==p.head||r.tree!==p.tree||
  r.producer_source_sha256!==p.producer.source.sha256||r.producer_elf_sha256!==p.producer.elf.sha256||
  r.build_manifest_sha256!==p.build_manifest.sha256||r.case_manifest_sha256!==p.inputs.preflight_cases.sha256||
  JSON.stringify(r.sdk_manifest_sha256)!==JSON.stringify(p.sdk.map(s=>s.manifest.sha256))||
  !Array.isArray(r.cases)||JSON.stringify(r.cases.map(c=>c.id))!==JSON.stringify(ids)||
  r.cases.some(c=>c.status!=='pass')||r.failed!==0||r.skipped!==0)bad('preflight result/graph/exact cases mismatch');
 pinned(r.adapter);
 if(!Array.isArray(r.originals)||r.originals.length===0)bad('preflight actual outputs/oracles required');
 for(const original of r.originals)pinned(original);
 return r;
}
export function explicitError(text){
 return /error\[E|(?:^|\s)error:|parse error|undefined identifier|unknown variable|field initializer mismatch|would have dropped|cannot compile an assignment|native.*refus|emission.*refus|closure.*refus|AST closure incomplete|IR_INVALID_REG|missing_lowered_body/im.test(text);
}
export function publish(root,phase,guard,status,elf=null){
 if(status!==0||guard.status!==0||guard.phase!=='final'||guard.schema!=='sounio.root06.validated-plan.v1'||guard.layout?.R!==root||!/^[0-9a-f]{40}$/.test(guard.head)||!/^[0-9a-f]{40}$/.test(guard.tree)||!hash(guard.producer_elf_sha256)||!hash(guard.build_manifest_sha256))bad('publication before terminal successful EXIT guards');
 const target=path.join(root,phase==='build'?'RAW_READY':'PREFLIGHT_READY');
 if(fs.existsSync(target))bad('readiness already exists; fresh output required');
 let body=JSON.stringify({phase,head:guard.head,tree:guard.tree,run_root:root,producer_elf_sha256:guard.producer_elf_sha256,build_manifest_sha256:guard.build_manifest_sha256,guards_complete:true})+'\n';
 if(phase==='build'){
  const st=fs.lstatSync(elf);if(!st.isFile()||st.isSymbolicLink()||(st.mode&0o111)===0)bad('regular executable built ELF required');
  const b=fs.readFileSync(elf);if(b.subarray(0,4).toString('hex')!=='7f454c46')bad('built ELF magic required');
  body=digest(elf)+'  '+elf+'\n';
 }
 const temporary=target+'.tmp';
 fs.writeFileSync(temporary,body,{flag:'wx'});fs.renameSync(temporary,target);
}
export function authorize(planFile,record){
 const a=json(record);if(a.authorized_by!=='root'||a.plan_sha256!==digest(planFile)||a.scope!=='ROOT06_STAGING_PREFLIGHT_BUILD')bad('manual root plan authorization required');
 return a;
}
export function loadPlan(file){return JSON.parse(fs.readFileSync(file,'utf8'))}
if(process.argv[1]===new URL(import.meta.url).pathname){
 const [action,file,phase='review']=process.argv.slice(2),p=loadPlan(file);
 if(action==='preflight-result')verifyPreflight(p);
 else if(action==='authorize')authorize(file,{file:process.env.ROOT06_AUTHORIZATION,sha256:process.env.ROOT06_AUTHORIZATION_SHA256,mode:Number(process.env.ROOT06_AUTHORIZATION_MODE)});
 else if(action==='verify')console.log(JSON.stringify(verifyPlan(p,phase),null,2));
 else if(action==='env'){
  layout(p);
  const L=layout(p),rows={...L,NODE:p.tools.node.file,BASH_BIN:p.tools.bash.file,GIT:p.tools.git.file,TAR:p.tools.tar.file,
   TIME:p.tools.time.file,TIMEOUT:p.tools.timeout.file,CXX_PIN:p.tools.cxx.file,MAGIC_DB:p.inputs.magic.file,SDK_ENV:p.tool_environment.file.file,SDK_ENV_R:p.tool_environment.context_R,
   ARCHIVE:p.inputs.archive.file,CAPACITY:p.inputs.capacity.file,WORKER_PATCH:p.inputs.worker_optin.file,PREFLIGHT_RECIPE:p.inputs.preflight_recipe.file,
   PRODUCER:p.producer.elf.file,BUILD_SECONDS:String(p.build_seconds),PREFLIGHT_SECONDS:String(p.preflight_seconds)};
  for(const[k,v]of Object.entries(rows))process.stdout.write(k+'\0'+v+'\0');
 }else if(action==='publish'){
  authorize(file,{file:process.env.ROOT06_AUTHORIZATION,sha256:process.env.ROOT06_AUTHORIZATION_SHA256,mode:Number(process.env.ROOT06_AUTHORIZATION_MODE)});
  const current=verifyPlan(p,'final');
  if(phase==='preflight')verifyPreflight(p);
  const guard=JSON.parse(fs.readFileSync(path.join(layout(p).R,'final-'+phase+'.json'),'utf8'));
  const text=fs.readFileSync(path.join(layout(p).R,phase==='build'?'results/BUILD_STATUS':'preflight/STATUS'),'utf8');
  if(!/^0\n?$/.test(text)||guard.head!==p.head||guard.tree!==p.tree)bad('terminal status/graph binding mismatch');
  if(JSON.stringify(guard)!==JSON.stringify(current))bad('recorded EXIT guard is not current full snapshot proof');
  const status=0;
  publish(layout(p).R,phase,guard,status,path.join(layout(p).D,'madaros'));
 }else bad('unknown metadata action');
}
