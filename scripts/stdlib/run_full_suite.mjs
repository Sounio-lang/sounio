#!/usr/bin/env node
// Parallelism is by disjoint canonical-runner selections, never by a changed
// test contract. Every original test path must appear exactly once in receipts.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {walk} from './inventory.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
let output='',jobs=2,slow=false,partition='directory';
for(let i=2;i<process.argv.length;i++) {
  const a=process.argv[i];
  if(a==='--help'){console.log('run_full_suite.mjs --out-dir ABS_FRESH_PATH [--jobs N] [--slow] [--partition directory|file]');process.exit(0);}
  if(a==='--partition'){partition=process.argv[++i];if(!['directory','file'].includes(partition))throw Error('invalid partition');continue;}
  if(a==='--slow'){slow=true;continue;}
  if(a==='--out-dir'){output=process.argv[++i]||'';continue;}
  if(a==='--jobs'){const v=process.argv[++i];if(!/^[1-9][0-9]*$/.test(v||''))throw Error('jobs must be positive');jobs=Number(v);continue;}
  throw Error('unknown argument: '+a);
}
if(!path.isAbsolute(output)||!Number.isSafeInteger(jobs)||jobs>64)throw Error('require absolute fresh output and jobs<=64');
fs.mkdirSync(output);
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const paths=walk(path.join(root,'tests/stdlib'),false).filter(p=>p.endsWith('.sio')).map(p=>path.relative(root,p)).sort();
if(paths.length===0)throw Error('empty test selection');
const initial=paths.map(p=>({path:p,sha256:hash(path.join(root,p))}));
fs.writeFileSync(path.join(output,'INPUTS.json'),JSON.stringify(initial,null,2)+'\n');
const directoryPrefixes=[...new Set(paths.map(p=>{const rest=p.slice('tests/stdlib/'.length);return rest.includes('/')?'tests/stdlib/'+rest.split('/')[0]+'/':p;}))].sort();
if(partition==='file' && paths.some(p=>paths.some(q=>p!==q&&q.startsWith(p))))throw Error('file selections overlap');
const prefixes=partition==='file'?paths:directoryPrefixes;
// Per-file execution gets one private archived tree per worker: relative output
// files cannot collide, and source mutation is checked in every copy.
if(partition==='file'&&fs.existsSync(path.join(root,'.git')))throw Error('file partition requires a fresh source archive, not a Git worktree');
const workerCount=Math.min(jobs,prefixes.length),workerRoots=[];
const copyInvariants=partition==='file'?[...initial,...walk(path.join(root,'stdlib'),false).filter(p=>p.endsWith('.sio')).map(p=>({path:path.relative(root,p),sha256:hash(p)}))]:[];
for(let i=0;i<workerCount;i++){
 if(partition!=='file'){workerRoots.push(root);continue;}
 const copy=path.join(output,'workers',String(i));fs.mkdirSync(path.dirname(copy),{recursive:true});
 fs.cpSync(root,copy,{recursive:true,mode:fs.constants.COPYFILE_FICLONE,filter:p=>p!==output&&!p.startsWith(output+path.sep)});
 for(const input of copyInvariants)if(hash(path.join(copy,input.path))!==input.sha256)throw Error('worker copy differs: '+input.path);
 workerRoots.push(copy);
}
const active=new Set(),receipts=[];
let cursor=0,interrupted=false;
process.on('SIGTERM',()=>{interrupted=true;for(const p of active){try{process.kill(-p.pid,'SIGTERM');}catch{}}});
async function worker(workerIndex) {
  const workerRoot=workerRoots[workerIndex];
  while(cursor<prefixes.length&&!interrupted) {
    const index=cursor++,prefix=prefixes[index],stem=String(index).padStart(4,'0');
    const log=fs.openSync(path.join(output,stem+'.log'),'wx');
    const args=[path.join(workerRoot,'scripts/stdlib/run_stdlib_e2e.sh'),'--filter',prefix,'--json-out',path.join(output,stem+'.json'),'--verbose'];
    if(process.env.STDLIB_FULL_CAPTURE==='1')args.push('--artifacts-dir',path.join(output,stem+'.capture'));
    const env={...process.env};if(partition==='file'&&(!env.SOUNIO_STDLIB_PATH||path.resolve(env.SOUNIO_STDLIB_PATH)===path.join(root,'stdlib')))env.SOUNIO_STDLIB_PATH=path.join(workerRoot,'stdlib');if(slow)env.SOUNIO_SLOW_TESTS_AVAILABLE='1';
    const result=await new Promise(resolve=>{
      const child=spawn('bash',args,{cwd:workerRoot,env,detached:true,stdio:['ignore',log,log]});active.add(child);
      let settled=false;
      const done=(code,signal,error)=>{if(settled)return;settled=true;active.delete(child);fs.closeSync(log);resolve({code,signal,error});};
      child.once('error',e=>done(null,null,e.message));child.once('close',(code,signal)=>done(code,signal,null));
    });
    receipts.push({prefix,stem,...result});
  }
}
await Promise.all(Array.from({length:workerCount},(_,i)=>worker(i)));
const seen=new Set(),results=[],errors=[];
for(const r of receipts.sort((a,b)=>a.stem.localeCompare(b.stem))) {
  try {
    const doc=JSON.parse(fs.readFileSync(path.join(output,r.stem+'.json'),'utf8'));
    if(doc.schema!=='sounio.stdlib.e2e.result.v1'||!Array.isArray(doc.results))throw Error('invalid result schema');
    if(![0,1].includes(r.code))throw Error('producer aborted '+JSON.stringify(r));
    const totals={pass:0,fail:0,skip:0,total:doc.results.length};
    for(const row of doc.results) {
      if(!Object.hasOwn(totals,row.status)||row.status==='total'||seen.has(row.test_path)||!paths.includes(row.test_path)||!row.test_path.startsWith(r.prefix))throw Error('invalid/duplicate/out-of-selection row '+row.test_path);
      seen.add(row.test_path);results.push(row);totals[row.status]++;
    }
    for(const k of Object.keys(totals))if(doc.totals?.[k]!==totals[k])throw Error('producer totals mismatch');
    if((r.code===0)!==(totals.fail===0))throw Error('producer exit and result disagree');
  }catch(e){errors.push({prefix:r.prefix,error:e.message});}
}
for(const p of paths)if(!seen.has(p))errors.push({path:p,error:'missing test result'});
for(const p of initial)if(hash(path.join(root,p.path))!==p.sha256)errors.push({path:p.path,error:'test input changed during execution'});
if(partition==='file')for(const copy of workerRoots)for(const input of copyInvariants){
 try{if(hash(path.join(copy,input.path))!==input.sha256)errors.push({path:input.path,worker_root:copy,error:'worker source changed during execution'});}
 catch(error){errors.push({path:input.path,worker_root:copy,error:'worker source missing: '+error.message});}
}
const totals={pass:0,fail:0,skip:0,total:results.length};
for(const r of results)totals[r.status]++;
const doc={schema:'sounio.stdlib.e2e.result.v1',generated_at_utc:new Date().toISOString(),
 command:'node scripts/stdlib/run_full_suite.mjs',context:{root_dir:root,slow_enabled:slow,jobs,partition,worker_roots:workerRoots},
 status_summary:errors.length||interrupted?'not_run':totals.fail?'fail':'pass',totals,
 results:results.sort((a,b)=>a.test_path.localeCompare(b.test_path)),failures:results.filter(r=>r.status==='fail'),
 ignored:results.filter(r=>r.mode==='ignore'),selection_errors:errors,producers:receipts};
fs.writeFileSync(path.join(output,'RESULT.json'),JSON.stringify(doc,null,2)+'\n');
console.log(JSON.stringify({status:doc.status_summary,totals,selection_errors:errors.length}));
process.exitCode=doc.status_summary==='pass'?0:1;
