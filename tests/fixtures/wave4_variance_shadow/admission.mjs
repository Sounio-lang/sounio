// Metadata admission only. Root authorization is a manual decision outside this module.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
export const digest=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const bad=s=>{throw Error(s)};
const hash=s=>typeof s==='string'&&/^[0-9a-f]{64}$/.test(s);
function regular(p){if(!fs.lstatSync(p).isFile()||fs.lstatSync(p).isSymbolicLink())bad('regular file required: '+p)}
function pinned(row){if(!row||!path.isAbsolute(row.file)||!hash(row.sha256))bad('invalid pinned input');regular(row.file);if(digest(row.file)!==row.sha256)bad('pinned input changed: '+row.file);return fs.readFileSync(row.file,'utf8')}
const modeOf=p=>fs.lstatSync(p).mode&0o7777;
const hashBytes=b=>crypto.createHash('sha256').update(b).digest('hex');
export function treeManifest(root){
 if(!path.isAbsolute(root)||!fs.lstatSync(root).isDirectory()||fs.lstatSync(root).isSymbolicLink())bad('sealed directory root required');
 const nodes=[];
 function visit(base=''){
  for(const e of fs.readdirSync(path.join(root,base),{withFileTypes:true})){
   const file=base?base+'/'+e.name:e.name,p=path.join(root,file),st=fs.lstatSync(p),mode=modeOf(p);
   if(st.isSymbolicLink()){
    const target=fs.readlinkSync(p,{encoding:'buffer'});
    nodes.push({file,type:'symlink',mode,bytes:target.length,sha256:hashBytes(target),target_base64:target.toString('base64')});
   }else if(st.isDirectory()){nodes.push({file,type:'directory',mode});visit(file)}
   else if(st.isFile())nodes.push({file,type:'file',mode,bytes:st.size,sha256:digest(p)});
   else bad('unsupported sealed tree node: '+file);
  }
 }
 visit();nodes.sort((x,y)=>x.file<y.file?-1:x.file>y.file?1:0);
 return {schema:'sounio.sealed-tree.v2',root_mode:modeOf(root),nodes};
}
export function verifyTree(root,manifest){
 const parsed=JSON.parse(pinned(manifest));
 if(parsed.schema!=='sounio.sealed-tree.v2'||!Array.isArray(parsed.nodes)||parsed.nodes.length===0)bad('nonempty full tree manifest v2 required');
 const names=parsed.nodes.map(r=>r.file);
 if(names.some((s,i)=>typeof s!=='string'||!s||s.startsWith('/')||s.split('/').some(x=>!x||x==='.'||x==='..')||s.includes('\\')||(i>0&&names[i-1]>=s)))bad('manifest order/path/duplicate');
 const actual=treeManifest(root);
 if(parsed.root_mode!==actual.root_mode||parsed.nodes.length!==actual.nodes.length)bad('sealed tree root/census mismatch');
 for(let i=0;i<parsed.nodes.length;i++){
  const want=parsed.nodes[i],got=actual.nodes[i];
  if(want.file!==got.file||want.type!==got.type||want.mode!==got.mode)bad('sealed tree node/type/mode mismatch: '+got.file);
  if(got.type==='file'&&(want.bytes!==got.bytes||want.sha256!==got.sha256))bad('sealed file content mismatch: '+got.file);
  if(got.type==='symlink'&&(want.bytes!==got.bytes||want.sha256!==got.sha256||want.target_base64!==got.target_base64))bad('sealed symlink literal mismatch: '+got.file);
  const keys=Object.keys(want).sort().join(',');
  if(keys!==Object.keys(got).sort().join(','))bad('sealed manifest unknown/missing fields: '+got.file);
 }
 return {root,manifest_sha256:manifest.sha256,nodes:actual.nodes.length};
}
export function explicitDiagnostics(text){
 const clean=text.replace(/\x1b\[[0-9;]*m/g,'');
 return /(?:^|\n)\s*(?:error\b|fatal\b|\[?E[0-9]{3,}\]?(?:\s|:))/im.test(clean)||
  /\berror(?:\[[^\]\n]+\])?:/i.test(clean);
}
export function successfulCompilerResult(result,prefix){
 if(result.code!==0||result.signal||result.timedout)bad('compiler stage failed');
 const text=fs.readFileSync(prefix+'.stdout','utf8')+'\n'+fs.readFileSync(prefix+'.stderr','utf8');
 if(explicitDiagnostics(text))bad('explicit diagnostic despite successful exit');
}
export function elfIdentity(file){
 regular(file);const fd=fs.openSync(file,'r');const b=Buffer.alloc(4);
 try{if(fs.readSync(fd,b,0,4,0)!==4||!b.equals(Buffer.from([0x7f,0x45,0x4c,0x46])))bad('ELF magic required')}
 finally{fs.closeSync(fd)}
 return {file,bytes:fs.statSync(file).size,sha256:digest(file)};
}
export function admit(auth,manifest){
 if(auth.authorized_by!=='root'||auth.patch_sha256!==manifest.patch_sha256)bad('manual root authorization/patch mismatch');
 for(const p of ['raw','source','sdk'])if(typeof auth[p]!=='string'||!path.isAbsolute(auth[p])||path.resolve(auth[p])!==auth[p])bad('absolute canonical path required');
 if(!hash(auth.raw_sha256))bad('raw hash required');
 const ready=pinned({file:auth.raw_ready,sha256:auth.raw_ready_sha256});
 const m=/^([0-9a-f]{64})  ([^\r\n]+)\n?$/.exec(ready);
 if(!m||m[1]!==auth.raw_sha256||m[2]!==auth.raw)bad('RAW_READY sha256sum identity mismatch');
 regular(auth.raw);if(digest(auth.raw)!==auth.raw_sha256)bad('raw compiler content mismatch');
 if(auth.lower_sha256!==manifest.candidate_lower_sha256||digest(path.join(auth.source,'self-hosted/ir/lower.sio'))!==auth.lower_sha256)bad('candidate lower mismatch');
 const source=verifyTree(auth.source,auth.source_manifest),sdk=verifyTree(auth.sdk,auth.sdk_manifest);
 const proof=JSON.parse(pinned(auth.root_proof));
 if(proof.schema!=='sounio.compiled-graph.root-proof.v1'||proof.finalized!==true||
    proof.raw!==auth.raw||proof.raw_sha256!==auth.raw_sha256||
    proof.source_manifest_sha256!==auth.source_manifest.sha256||
    proof.sdk_manifest_sha256!==auth.sdk_manifest.sha256)bad('terminal root graph proof mismatch');
 pinned(proof.adapter);
 if(!Array.isArray(proof.original_inputs)||proof.original_inputs.length===0)bad('pinned original build inputs required');
 const roles=new Set();
 for(const input of proof.original_inputs){
  if(typeof input.role!=='string'||!/^[A-Z][A-Z0-9_]{0,63}$/.test(input.role)||roles.has(input.role))bad('invalid/duplicate original input role');
  roles.add(input.role);pinned(input);
 }
 for(const role of ['BUILD_STATUS_SOURCE','FINAL_SOURCE_VERIFY','FINAL_SDK_VERIFY','ENDED_SOURCE'])
  if(!roles.has(role))bad('missing original input role: '+role);
 const build=pinned(proof.build_status);
 if(!/^0\n?$/.test(build))bad('terminal BUILD_STATUS0 required');
 const ended=pinned(proof.ended_utc);
 if(!/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\n?$/.test(ended))bad('terminal ENDED_UTC required');
 if(!Array.isArray(proof.final_guards)||proof.final_guards.length===0)bad('final guards required');
 for(const g of proof.final_guards)if(!/^0\n?$/.test(pinned(g)))bad('final source/SDK/build guard not zero');
 return {raw:auth.raw,raw_sha256:auth.raw_sha256,source,sdk,root_proof_sha256:auth.root_proof.sha256};
}
export function captureAdmission(auth,manifest,dest,label){
 const receipt=admit(auth,manifest);
 const target=path.join(dest,'admission-'+label);fs.mkdirSync(target);
 for(const[name,row]of Object.entries({
  RAW_READY:{file:auth.raw_ready,sha256:auth.raw_ready_sha256},
  SOURCE_MANIFEST:auth.source_manifest,SDK_MANIFEST:auth.sdk_manifest,ROOT_PROOF:auth.root_proof
 }))fs.copyFileSync(row.file,path.join(target,name));
 const proof=JSON.parse(fs.readFileSync(auth.root_proof.file,'utf8'));
 fs.copyFileSync(proof.adapter.file,path.join(target,'ROOT_ADAPTER'));
 for(const input of proof.original_inputs)fs.copyFileSync(input.file,path.join(target,'ORIGINAL_'+input.role));
 for(const[name,row]of [['BUILD_STATUS',proof.build_status],['ENDED_UTC',proof.ended_utc],...proof.final_guards.map((g,i)=>['FINAL_GUARD_'+i,g])])
  fs.copyFileSync(row.file,path.join(target,name));
 fs.writeFileSync(path.join(target,'RECEIPT.json'),JSON.stringify(receipt,null,2)+'\n');
 return receipt;
}
