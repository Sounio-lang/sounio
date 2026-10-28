// Execute canonical native/C++ gates for source fragments that have no main.
// Historical standalone skips remain in each row as standalone_result.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawn} from 'node:child_process';
import {compilerEnvironment,verifyCompilerBinding} from './compiler_binding.mjs';
export const compoundGates = [
 {name:'bdf',script:'ode_bdf_checked_gate.sh',marker:'BDF_GATE_PASS',members:['bdf/legacy_regressions_main.sio']},
 {name:'fem',script:'fem_poisson_gate.sh',marker:'FEM_POISSON_GATE_PASS',members:['fem_poisson/legacy_cg_main.sio','fem_poisson/singular_legacy_main.sio']},
 {name:'fft',script:'math_fft_checked_gate.sh',marker:'FFT_GATE_PASS',members:['fft/bench_legacy_main.sio','fft/legacy_bounds_main.sio','fft/legacy_valid_main.sio']},
 {name:'numdiff',script:'math_numdiff_checked_gate.sh',marker:'NUMDIFF_GATE_PASS',members:['numdiff/legacy_negative_count.sio','numdiff/legacy_negative_h.sio','numdiff/legacy_roots_main.sio']},
 {name:'token',script:'genomics_token_bench_gate.sh',marker:'TOKEN_BENCH_GATE_PASS',members:['genomics_counts/token_bench.sio']}
].map(g=>({...g,members:g.members.map(p=>'tests/stdlib/'+p)}));
const digest=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
export function validateCompoundReceipt(gate,receipt,inputs,compiler) {
 if(receipt.code!==0||receipt.signal||receipt.error)throw Error('compound producer failed');
 if(receipt.status.trim()!=='0')throw Error('compound status missing or nonzero');
 if(receipt.stdout.split(/\r?\n/).filter(s=>s===gate.marker).length!==1)throw Error('compound completion marker missing or duplicate');
 if(!compiler)throw Error('missing bound compiler');
 const compilerLines=receipt.compilerManifest.trim().split(/\r?\n/);
 const c=compilerLines.length===1&&/^([a-f0-9]{64}) [ *](.+)$/.exec(compilerLines[0]);
 if(!c||c[1]!==compiler.sha256||c[2]!==compiler.path)throw Error('compound compiler identity mismatch');
 const lines=receipt.manifest.split(/\r?\n/).filter(Boolean).map(line=>{
  const m=/^([a-f0-9]{64}) [ *](.+)$/.exec(line);if(!m)throw Error('malformed compound input manifest');
  return {sha256:m[1],path:m[2]};
 });
 for(const member of gate.members){
  const initial=inputs.find(p=>p.path===member);if(!initial)throw Error('compound member absent from selection: '+member);
  const bound=lines.filter(p=>p.path===member);
  if(bound.length===0||bound.some(p=>p.sha256!==initial.sha256))throw Error('compound member digest missing or changed: '+member);
 }
}
export async function runCompoundGates({root,output,results,inputs,onChild=()=>{},onDone=()=>{},shouldStop=()=>false,compiler}) {
 verifyCompilerBinding(compiler);
 fs.mkdirSync(output);
 const receipts=[];
 for(const gate of compoundGates){
  if(shouldStop())throw Error('compound execution interrupted');
  const rows=gate.members.map(member=>results.find(r=>r.test_path===member));
  // Never turn an existing check, compile or runtime failure into coverage.
  if(rows.some(r=>!r||r.status!=='skip'||r.reason!=='no-annotation'||r.mode!=='no-annotation'))
   throw Error('compound member is not a standalone fragment: '+gate.name);
  const script=path.join(root,'scripts/stdlib',gate.script),dir=path.join(output,gate.name);
  const log=path.join(output,gate.name+'.stdout'),err=path.join(output,gate.name+'.stderr');
  const outfd=fs.openSync(log,'wx'),errfd=fs.openSync(err,'wx');
  const receipt=await new Promise(resolve=>{
   const child=spawn('timeout',['1200','bash',script,dir],{cwd:root,env:compilerEnvironment(compiler),detached:true,stdio:['ignore',outfd,errfd]});
   onChild(child);let settled=false;
   const done=(code,signal,error)=>{if(settled)return;settled=true;onDone(child);fs.closeSync(outfd);fs.closeSync(errfd);resolve({code,signal,error});};
   child.once('error',e=>done(null,null,e.message));child.once('close',(code,signal)=>done(code,signal,null));
  });
  let validation='';
  try{
   validateCompoundReceipt(gate,{...receipt,status:fs.readFileSync(path.join(dir,'STATUS'),'utf8'),stdout:fs.readFileSync(log,'utf8'),manifest:fs.readFileSync(path.join(dir,'INPUTS.sha256'),'utf8'),compilerManifest:fs.readFileSync(path.join(dir,'COMPILER.sha256'),'utf8')},inputs,compiler);
   // The original file also has to remain unchanged after the gate.
   for(const member of gate.members)if(digest(path.join(root,member))!==inputs.find(p=>p.path===member).sha256)throw Error('compound source changed: '+member);
  }catch(e){validation=e.message;}
  verifyCompilerBinding(compiler);
  const record={name:gate.name,compiler_sha256:compiler.sha256,script:path.relative(root,script),script_sha256:digest(script),members:gate.members,output_dir:dir,...receipt,validation};
  receipts.push(record);
  fs.writeFileSync(path.join(output,'RECEIPTS.json'),JSON.stringify(receipts,null,2)+'\n');
  for(const row of rows){
   const standalone={...row};
   Object.assign(row,{status:validation?'fail':'pass',mode:'compound-native',reason:validation?'compound-gate-failed':'',exit_code:validation?(receipt.code||1):0,detail_excerpt:validation,compound_gate:gate.name,standalone_result:standalone});
  }
 }
 fs.writeFileSync(path.join(output,'RECEIPTS.json'),JSON.stringify(receipts,null,2)+'\n');
 return receipts;
}
