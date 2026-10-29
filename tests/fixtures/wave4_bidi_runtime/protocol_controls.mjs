#!/usr/bin/env node
// Orchestration and file-format metadata only. No Unicode resolution is implemented here.
// The corpus and all expected levels come from the sealed C++23 producer.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const options={};
for(let i=2;i<process.argv.length;i+=2){
 const key=process.argv[i],value=process.argv[i+1];
 if(!['--module','--golden','--corpus','--original-source','--out'].includes(key)||!value||Object.hasOwn(options,key))throw Error('require unique --module --golden --corpus --original-source --out paths');
 options[key]=path.resolve(value);
}
for(const key of ['--module','--golden','--corpus','--original-source','--out'])assert(options[key],key);
const subject=options['--module'],goldenFile=options['--golden'],corpus=options['--corpus'],originalSource=options['--original-source'],out=options['--out'];
const inside=(a,b)=>a===b||a.startsWith(b+path.sep);
for(const input of [subject,goldenFile,corpus,originalSource])assert(!inside(out,input)&&!inside(input,out),'output/input overlap');
assert(!fs.existsSync(out),'fresh output required');
fs.mkdirSync(out);fs.mkdirSync(path.join(out,'cases'));
const sha=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const fileIdentity=file=>{
 const st=fs.lstatSync(file,{bigint:true});assert(st.isFile()&&!st.isSymbolicLink(),'regular immutable input required');
 const data=fs.readFileSync(file);
 return {path:file,bytes:data.length,sha256:sha(data),device:st.dev.toString(),inode:st.ino.toString()};
};
const golden=JSON.parse(fs.readFileSync(goldenFile,'utf8'));
assert.equal(golden.schema,'sounio.unicode18.bidi.official-corpus.v1');
assert.equal(golden.files.length,214);
const expectedNames=golden.files.map(r=>r.file).sort();
assert.equal(new Set(expectedNames).size,214);
for(const name of expectedNames)assert(/^(?:[0-9]{6}\.bin|CHUNKS\.tsv|COUNTS\.json|SMOKE\.bin)$/.test(name),'safe fixture basename');
const originalSnapshot=()=>{
 assert(fs.lstatSync(corpus).isDirectory()&&!fs.lstatSync(corpus).isSymbolicLink(),'original corpus must be a real directory');
 assert.deepEqual(fs.readdirSync(corpus).sort(),expectedNames);
 return expectedNames.map(file=>({file,...fileIdentity(path.join(corpus,file))}));
};
const before=originalSnapshot(),sourceBefore=fileIdentity(originalSource),subjectBefore=fileIdentity(subject),goldenBefore=fileIdentity(goldenFile);
assert.equal(sourceBefore.sha256,golden.provenance.program_source.sha256,'qualified original source required');
for(const row of before){const pin=golden.files.find(p=>p.file===row.file);assert.equal(row.bytes,pin.bytes);assert.equal(row.sha256,pin.sha256);}
const report={schema:'sounio.bidi.runtime.protocol-controls.v1',scope:'Metadata-only validator controls over unchanged official214-file corpus; no scientific resolver replacement',subject:subjectBefore,golden:goldenBefore,original_source:sourceBefore,original_corpus:before,mutation_policy:'All replacement writes first unlink the destination hardlink; original inodes are never written.',cases:[]};
const writeReport=()=>fs.writeFileSync(path.join(out,'RESULTS.json'),JSON.stringify(report,null,2)+'\n');
writeReport();fs.writeFileSync(path.join(out,'STATUS'),'1\n');
let passed=0,failed=0,completed=false;
try{
 const {validateBidiCorpus}=await import(pathToFileURL(subject).href+'?sha256='+subjectBefore.sha256);
 assert.equal(typeof validateBidiCorpus,'function');
 const chunks=fs.readFileSync(path.join(corpus,'CHUNKS.tsv'),'utf8').trimEnd().split('\n').map(line=>{const [file,cases,bytes]=line.split('\t');return {file,cases:Number(cases),bytes:Number(bytes)};});
 assert.equal(chunks.length,211);
 const first=chunks[0].file,second=chunks[1].file,last=chunks.at(-1).file;
 assert.equal(chunks[0].cases,chunks[1].cases,'duplicate-content control requires equal case headers');
 let index=0;
 function clone(name){
  const dir=path.join(out,'cases',String(++index).padStart(2,'0')+'-'+name);fs.mkdirSync(dir);
  for(const file of expectedNames){
   const from=path.join(corpus,file),to=path.join(dir,file);
   try{fs.linkSync(from,to);}catch(e){if(e.code!=='EXDEV')throw e;fs.copyFileSync(from,to);}
  }
  return dir;
 }
 function replace(dir,file,bytes){
  assert(expectedNames.includes(file),'replacement outside known fixture list');
  const target=path.join(dir,file);
  assert(fs.lstatSync(target).isFile()&&!fs.lstatSync(target).isSymbolicLink(),'replacement starts at regular clone');
  fs.unlinkSync(target); // Mandatory: never write an inode shared with the original corpus.
  fs.writeFileSync(target,bytes,{flag:'wx',mode:0o600});
 }
 function updateManifest(dir,edit){
  const rows=fs.readFileSync(path.join(dir,'CHUNKS.tsv'),'utf8').trimEnd().split('\n').map(line=>line.split('\t'));
  edit(rows);replace(dir,'CHUNKS.tsv',rows.map(r=>r.join('\t')).join('\n')+'\n');
 }
 async function check(name,expected,mutate,alias=false){
  const dir=alias?path.join(out,'cases',String(++index).padStart(2,'0')+'-'+name):clone(name);
  if(alias)fs.symlinkSync(corpus,dir,'dir');
  else if(mutate)mutate(dir);
  let observed='accepted',error=null,rows=null;
  try{rows=await validateBidiCorpus(dir);}catch(e){observed='rejected';error=e.message;}
  if(observed==='accepted'&&expected==='accepted'){
   try{assert(Array.isArray(rows));assert.equal(rows.length,211);assert.deepEqual(rows.map(r=>r.file).sort(),chunks.map(r=>r.file).sort());}
   catch(e){observed='invalid-positive';error=e.message;}
  }
  const okay=observed===expected;
  const record={name,expected,observed,okay,error,fixture_directory:dir};
  report.cases.push(record);writeReport();
  assert.deepEqual(originalSnapshot(),before,'original corpus changed');
  assert.deepEqual(fileIdentity(originalSource),sourceBefore,'original source changed');
  if(okay){passed++;console.log('PASS '+name);}else{failed++;console.log('FAIL '+name+' expected='+expected+' observed='+observed);}
 }
 await check('complete','accepted');
 await check('drop-file','rejected',d=>fs.unlinkSync(path.join(d,last)));
 await check('drop-row-and-file','rejected',d=>{fs.unlinkSync(path.join(d,last));updateManifest(d,r=>r.pop());});
 await check('duplicate-manifest-row','rejected',d=>updateManifest(d,r=>r.push([...r[0]])));
 await check('duplicate-payload-adjusted-header','rejected',d=>{
  const bytes=fs.readFileSync(path.join(corpus,first));replace(d,second,bytes);
  updateManifest(d,r=>{const row=r.find(x=>x[0]===second);row[1]=String(bytes.readUInt32LE(12));row[2]=String(bytes.length);});
 });
 await check('extra-unlisted-file','rejected',d=>fs.copyFileSync(path.join(corpus,first),path.join(d,'999999.bin')));
 await check('extra-listed-file','rejected',d=>{fs.copyFileSync(path.join(corpus,first),path.join(d,'999999.bin'));updateManifest(d,r=>r.push(['999999.bin',String(chunks[0].cases),String(chunks[0].bytes)]));});
 await check('wrong-header-magic','rejected',d=>{const b=fs.readFileSync(path.join(corpus,first));b[0]^=1;replace(d,first,b);});
 await check('wrong-header-count','rejected',d=>{const b=fs.readFileSync(path.join(corpus,first));b.writeUInt32LE(b.readUInt32LE(12)-1,12);replace(d,first,b);});
 await check('truncated','rejected',d=>{const b=fs.readFileSync(path.join(corpus,first));replace(d,first,b.subarray(0,-1));});
 await check('truncated-adjusted-manifest','rejected',d=>{const b=fs.readFileSync(path.join(corpus,first));replace(d,first,b.subarray(0,-1));updateManifest(d,r=>{r[0][2]=String(b.length-1);});});
 await check('payload-mutated-header-intact','rejected',d=>{const b=fs.readFileSync(path.join(corpus,first));b[b.length-1]^=1;replace(d,first,b);});
 await check('counts-incorrect','rejected',d=>{const c=JSON.parse(fs.readFileSync(path.join(corpus,'COUNTS.json')));c.cases--;replace(d,'COUNTS.json',JSON.stringify(c)+'\n');});
 await check('counts-malformed','rejected',d=>replace(d,'COUNTS.json','{\n'));
 await check('smoke-header-incorrect','rejected',d=>{const b=fs.readFileSync(path.join(corpus,'SMOKE.bin'));b.writeUInt32LE(63,12);replace(d,'SMOKE.bin',b);});
 await check('smoke-level-incorrect','rejected',d=>{
  const b=fs.readFileSync(path.join(corpus,'SMOKE.bin'));
  let at=16,changed=false;
  for(let k=0;k<b.readUInt32LE(12)&&!changed;k++){
   const n=b.readUInt32LE(at),order=b.readUInt32LE(at+12),levels=at+24+n*4;
   assert(levels+n*4+order*4<=b.length,'serialized smoke record bounds');
   for(let j=0;j<n;j++){const offset=levels+j*4;if(b.readUInt32LE(offset)!==0xffffffff){b[offset]^=1;changed=true;break;}}
   at+=24+n*8+order*4;
  }
  assert(changed,'non-omitted official smoke level');replace(d,'SMOKE.bin',b);
 });
 await check('smoke-truncated-body','rejected',d=>{const b=fs.readFileSync(path.join(corpus,'SMOKE.bin'));replace(d,'SMOKE.bin',b.subarray(0,-1));});
 for(const file of [first,'CHUNKS.tsv','COUNTS.json','SMOKE.bin']){
  await check('symlink-'+file.replaceAll('.','-'),'rejected',d=>{fs.unlinkSync(path.join(d,file));fs.symlinkSync(path.join(corpus,file),path.join(d,file));});
 }
 await check('symlink-corpus-directory','rejected',null,true);
 assert.equal(report.cases.length,22,'nonempty full protocol selection');completed=true;
}catch(error){report.fatal_error=error.stack||error.message;}finally{
 report.original_corpus_after=originalSnapshot();
 report.original_source_after=fileIdentity(originalSource);
 report.subject_after=fileIdentity(subject);report.golden_after=fileIdentity(goldenFile);
 report.original_corpus_unchanged=JSON.stringify(report.original_corpus_after)===JSON.stringify(before);
 report.original_source_unchanged=JSON.stringify(report.original_source_after)===JSON.stringify(sourceBefore);
 report.subject_unchanged=JSON.stringify(report.subject_after)===JSON.stringify(subjectBefore);
 report.golden_unchanged=JSON.stringify(report.golden_after)===JSON.stringify(goldenBefore);
 report.passed=passed;report.failed=failed;report.completed=completed;writeReport();
 assert(report.original_corpus_unchanged&&report.original_source_unchanged&&report.subject_unchanged&&report.golden_unchanged,'immutable inputs changed');
 fs.writeFileSync(path.join(out,'STATUS'),completed&&failed===0?'0\n':'1\n');
}
console.log('BIDI_RUNTIME_PROTOCOL_CONTROLS '+JSON.stringify({passed,failed,total:report.cases.length}));
process.exitCode=completed&&failed===0?0:1;
