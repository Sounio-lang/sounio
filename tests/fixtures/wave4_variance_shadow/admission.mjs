// Metadata admission only. Root authorization is a manual decision outside this module.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
export const digest=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const bad=s=>{throw Error(s)};
const hash=s=>typeof s==='string'&&/^[0-9a-f]{64}$/.test(s);
function regular(p){if(!fs.lstatSync(p).isFile()||fs.lstatSync(p).isSymbolicLink())bad('regular file required: '+p)}
function pinned(row){if(!row||!path.isAbsolute(row.file)||!hash(row.sha256))bad('invalid pinned input');regular(row.file);if(digest(row.file)!==row.sha256)bad('pinned input changed: '+row.file);return fs.readFileSync(row.file,'utf8')}
function census(root,base=''){
 const out=[];
 for(const e of fs.readdirSync(path.join(root,base),{withFileTypes:true})){
  const rel=base?base+'/'+e.name:e.name,full=path.join(root,rel);
  if(e.isSymbolicLink())bad('sealed tree symlink: '+rel);
  if(e.isDirectory())out.push(...census(root,rel));
  else if(e.isFile())out.push(rel);
  else bad('unsupported sealed tree node: '+rel);
 }
 return out.sort();
}
export function verifyTree(root,manifest){
 if(!path.isAbsolute(root)||!fs.lstatSync(root).isDirectory()||fs.lstatSync(root).isSymbolicLink())bad('sealed root required');
 const sealed=pinned(manifest);
 const parsed=JSON.parse(sealed);
 if(parsed.schema!=='sounio.sealed-tree.v1'||!Array.isArray(parsed.files)||parsed.files.length===0)bad('nonempty full tree manifest required');
 const names=parsed.files.map(r=>r.file);
 if(names.some((s,i)=>typeof s!=='string'||!s||s.startsWith('/')||s.split('/').some(x=>!x||x==='.'||x==='..')||s.includes('\\')||(i>0&&names[i-1]>=s)))bad('manifest order/path/duplicate');
 const actual=census(root);
 if(JSON.stringify(actual)!==JSON.stringify(names))bad('sealed tree census mismatch');
 for(const row of parsed.files){
  const p=path.join(root,row.file);regular(p);
  if(!Number.isSafeInteger(row.bytes)||row.bytes<0||fs.statSync(p).size!==row.bytes||!hash(row.sha256)||digest(p)!==row.sha256)bad('sealed tree content mismatch: '+row.file);
 }
 return {root,manifest_sha256:manifest.sha256,files:names.length};
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
 for(const[name,row]of [['BUILD_STATUS',proof.build_status],['ENDED_UTC',proof.ended_utc],...proof.final_guards.map((g,i)=>['FINAL_GUARD_'+i,g])])
  fs.copyFileSync(row.file,path.join(target,name));
 fs.writeFileSync(path.join(target,'RECEIPT.json'),JSON.stringify(receipt,null,2)+'\n');
 return receipt;
}
