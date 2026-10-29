#!/usr/bin/env node
// Runtime fixture orchestration only. The pinned existing C++23 producer and
// original Sounio resolver perform all scientific computation.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawn,spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const TEST='tests/stdlib/wave3-bidi/conformance.sio';
const FIXTURE='unicode18-bidi';
const GOLDEN='tests/fixtures/wave4_bidi_runtime/official-corpus.json';
const SCRIPT_ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const COUNT={unicode:'18.0.0',cases:861948,class_cases:770241,character_cases:91707,scalars:3767120,max_case_scalars:130,chunks:211};
const SOURCE_FILES=['stdlib/text/data/UCD18-SOURCES.sha256','stdlib/text/data/ucd18-source.tar.gz','tests/reference/wave3_bidi_cases.hpp','tests/reference/wave3_bidi_generator.cpp'].sort();
function json(p){regular(p);return JSON.parse(fs.readFileSync(p,'utf8'));}
function regular(p){const s=fs.lstatSync(p);if(!s.isFile()||s.isSymbolicLink())throw Error('not regular: '+p);return s;}
function safeDirectory(dir){for(let p=path.resolve(dir);;p=path.dirname(p)){const s=fs.lstatSync(p);if(!s.isDirectory()||s.isSymbolicLink())throw Error('unsafe directory: '+p);if(p===path.dirname(p))break;}}
function relative(s){return typeof s==='string'&&s!==''&&!s.includes('\\')&&!path.posix.isAbsolute(s)&&path.posix.normalize(s)===s&&s.split('/').every(p=>p!==''&&p!=='.'&&p!=='..');}
function fileIdentity(p){const s=regular(p);return {bytes:s.size,sha256:hash(p)};}
function inputIdentity(p){p=path.resolve(p);return {path:p,...fileIdentity(p)};}
function sameIdentity(p,expected){if(JSON.stringify(fileIdentity(p))!==JSON.stringify({bytes:expected.bytes,sha256:expected.sha256}))throw Error('identity changed: '+p);}
function manifestRows(rows,key){
 if(!Array.isArray(rows)||rows.length===0)throw Error('missing manifest rows');
 let previous='';for(const r of rows){if(!r||Object.keys(r).sort().join(',')!==[key,'bytes','sha256'].sort().join(',')||!relative(r[key])||r[key]<=previous||!Number.isSafeInteger(r.bytes)||r.bytes<0||!/^[0-9a-f]{64}$/.test(r.sha256))throw Error('unsafe or duplicate manifest entry');previous=r[key];}
}
export function readOfficialGolden(root=SCRIPT_ROOT){
 const golden=json(path.join(root,GOLDEN));
 if(golden.schema!=='sounio.unicode18.bidi.official-corpus.v1')throw Error('unknown official corpus schema');
 manifestRows(golden.files,'file');manifestRows(golden.source_files,'path');manifestRows(golden.upstream_files,'path');
 if(golden.files.length!==214||JSON.stringify(golden.counts)!==JSON.stringify(COUNT)||JSON.stringify(golden.source_files.map(r=>r.path))!==JSON.stringify(SOURCE_FILES))throw Error('incomplete official corpus pins');
 if(!golden.provenance||golden.provenance.qualification_job!=='19618'||golden.provenance.files_byte_identical!==true)throw Error('missing qualified golden provenance');
 for(const row of golden.source_files){safeDirectory(path.dirname(path.join(root,row.path)));sameIdentity(path.join(root,row.path),row);}
 return golden;
}
function tool(command){
 if(command.includes(path.sep))return fs.realpathSync(command);
 for(const p of (process.env.PATH||'').split(path.delimiter)){
  const f=path.join(p,command);try{fs.accessSync(f,fs.constants.X_OK);return fs.realpathSync(f);}catch{}
 }throw Error('tool missing: '+command);
}
function run(command,args,cwd,log){
 const r=spawnSync(command,args,{cwd,encoding:'utf8',maxBuffer:16*1024*1024});
 fs.writeFileSync(log,(r.stdout||'')+(r.stderr||''));
 if(r.error||r.status!==0)throw Error('producer failed '+command+': '+(r.error?.message||r.signal||r.status));
 return r.stdout;
}
export function validateBidiCorpus(corpus,golden=readOfficialGolden(SCRIPT_ROOT)){
 safeDirectory(corpus);
 const text=fs.readFileSync(path.join(corpus,'CHUNKS.tsv'),'utf8');
 if(!text.endsWith('\n'))throw Error('unterminated chunk manifest');
 const rows=[],seen=new Set();let total=0;
 for(const line of text.slice(0,-1).split('\n')){
  const m=/^([0-9]{6}\.bin)\t([1-9][0-9]*)\t([1-9][0-9]*)$/.exec(line);
  if(!m||seen.has(m[1]))throw Error('invalid or duplicate chunk');
  const count=Number(m[2]),bytes=Number(m[3]),file=path.join(corpus,m[1]);
  if(!Number.isSafeInteger(count)||count>4096||!Number.isSafeInteger(bytes)||bytes<16||bytes>1048576)throw Error('chunk bounds');
  if(!fs.lstatSync(file).isFile())throw Error('fixture is not regular');
  const data=fs.readFileSync(file);
  if(data.length!==bytes||data.readUInt32LE(0)!==0x44494253||data.readUInt32LE(4)!==0x00383149||data.readUInt32LE(8)!==1||data.readUInt32LE(12)!==count)throw Error('chunk header/size');
  seen.add(m[1]);total+=count;rows.push({file:m[1],cases:count,bytes,sha256:hash(file)});
 }
 if(rows.length!==211||total!==861948)throw Error('complete official corpus required');
 const actual=fs.readdirSync(corpus).sort();
 const expected=[...seen,'CHUNKS.tsv','COUNTS.json','SMOKE.bin'].sort();
 if(JSON.stringify(actual)!==JSON.stringify(expected)||actual.some(x=>!fs.lstatSync(path.join(corpus,x)).isFile()))throw Error('unlisted fixture');
 const smoke=fs.readFileSync(path.join(corpus,'SMOKE.bin'));
 if(smoke.length<16||smoke.length>1048576||smoke.readUInt32LE(0)!==0x44494253||smoke.readUInt32LE(4)!==0x00383149||smoke.readUInt32LE(8)!==1||smoke.readUInt32LE(12)!==64)throw Error('smoke header');

 const entries=fs.readdirSync(corpus).sort().map(file=>({file,...fileIdentity(path.join(corpus,file))}));
 if(JSON.stringify(entries)!==JSON.stringify(golden.files))throw Error('official corpus golden bytes differ');
 if(JSON.stringify(JSON.parse(fs.readFileSync(path.join(corpus,'COUNTS.json'),'utf8')))!==JSON.stringify(golden.counts))throw Error('official counts differ');
 return rows;
}

function native(elf,cwd,log){
 return new Promise((resolve,reject)=>{
  const fd=fs.openSync(log,'wx'),child=spawn(elf,[],{cwd,stdio:['ignore','pipe','pipe']});
  let bytes=0,stored=0,limited=false,error=null;
  const receive=b=>{bytes+=b.length;if(bytes>1048576){limited=true;child.kill('SIGKILL');return;}let offset=0;while(offset<b.length){const n=fs.writeSync(fd,b,offset,b.length-offset);if(n<=0)throw Error('short runtime output write');offset+=n;stored+=n;}};
  child.stdout.on('data',receive);child.stderr.on('data',receive);
  child.once('error',e=>{error=e;});
  child.once('close',(status,signal)=>{fs.closeSync(fd);if(error){reject(error);return;}resolve({status:limited?122:status,signal,output_bytes:stored,observed_output_bytes:bytes});});
 });
}
function observed(log,count){
 const lines=fs.readFileSync(log,'utf8').split('\n').filter(x=>x.length!==0);
 return lines.length===2&&lines.filter(x=>x==='BIDI_NATIVE_CONFORMANCE_OK').length===1&&lines.filter(x=>x==='BIDI_CASES_OK '+count).length===1;
}

function tree(out){
 safeDirectory(out);const files=[],directories=[];
 function visit(dir){
  for(const name of fs.readdirSync(dir).sort()){
   const p=path.join(dir,name),r=path.relative(out,p).split(path.sep).join('/'),s=fs.lstatSync(p);
   if(!relative(r)||s.isSymbolicLink())throw Error('unsafe fixture artifact: '+p);
   if(s.isDirectory()){directories.push(r);visit(p);}
   else if(s.isFile()){if(r!=='ARTIFACTS.json')files.push({file:r,...fileIdentity(p)});}
   else throw Error('nonregular fixture artifact: '+p);
  }
 }visit(out);files.sort((a,b)=>a.file<b.file?-1:a.file>b.file?1:0);directories.sort();return {files,directories};
}
function writeJson(p,data){fs.writeFileSync(p,JSON.stringify(data,null,2)+'\n');}
function exact(a,b,message){if(JSON.stringify(a)!==JSON.stringify(b))throw Error(message);}
export function verifyBidiArtifacts(root,test,out,binding,expectedStatus){
 root=fs.realpathSync(root);out=path.resolve(out);safeDirectory(out);
 if(test!==TEST||binding.fixture!==FIXTURE||binding.test_path!==test||binding.output!==out||binding.executedElfPath!==path.join(out,'program.elf'))throw Error('runtime fixture binding differs');
 sameIdentity(path.join(root,'scripts/stdlib/runtime_fixtures.mjs'),binding.helper);
 sameIdentity(path.join(root,GOLDEN),binding.golden);
 const manifest=json(path.join(out,'ARTIFACTS.json'));
 if(Object.keys(manifest).sort().join(',')!==['schema','fixture','test_path','status','executedElf','self','files','directories'].sort().join(',')||manifest.schema!=='sounio.stdlib.runtime-fixture.artifacts.v1'||manifest.fixture!==FIXTURE||manifest.test_path!==TEST||manifest.status!==expectedStatus||![0,1].includes(manifest.status)||manifest.self!=='ARTIFACTS.json')throw Error('runtime artifact manifest identity/status');
 manifestRows(manifest.files,'file');
 exact(tree(out),{files:manifest.files,directories:manifest.directories},'runtime artifact census/hash differs');
 exact(manifest.executedElf,{file:'program.elf',bytes:binding.originalElf.bytes,sha256:binding.originalElf.sha256},'executed ELF differs from native start');
 const retained=path.join(out,'program.elf');
 const present=manifest.files.some(r=>r.file==='program.elf');
 if(present){sameIdentity(retained,binding.originalElf);if(fs.readFileSync(retained).subarray(0,4).toString('hex')!=='7f454c46')throw Error('retained ELF header');}
 else if(manifest.status===0)throw Error('missing retained ELF');
 if(fs.readFileSync(path.join(out,'STATUS'),'utf8')!==manifest.status+'\n')throw Error('runtime status file differs');
 const initial=json(path.join(out,'INPUTS.json')),final=json(path.join(out,'INPUTS.final.json'));
 exact(initial,final,'runtime inputs changed');
 if(!initial.some(r=>r.path===binding.originalElf.path&&r.bytes===binding.originalElf.bytes&&r.sha256===binding.originalElf.sha256))throw Error('original temporal ELF not bound');
 // The temporary original ELF has intentionally been deleted by finalization.
 // Its initial/final observations remain provenance, not a live dependency.
 for(const r of initial){if(r.path!==binding.originalElf.path)sameIdentity(r.path,r);}
 const results=fs.existsSync(path.join(out,'RESULTS.json'))?json(path.join(out,'RESULTS.json')):[];
 if(!Array.isArray(results)||results.length>211||(!present&&results.length))throw Error('runtime result census');
 const golden=readOfficialGolden(root);
 const rows=manifest.status===0?validateBidiCorpus(path.join(out,'corpus'),golden):null;
 for(let i=0;i<results.length;i++){
  const r=results[i];if(!r||r.file!==String(i).padStart(6,'0')+'.bin'||r.executedElf!=='program.elf'||r.elf_sha256!==binding.originalElf.sha256)throw Error('runtime result ELF/order');
  const pinned=golden.files.find(x=>x.file===r.file);if(!pinned||r.bytes!==pinned.bytes||r.sha256!==pinned.sha256)throw Error('runtime result fixture hash');
  const log=path.join(out,'runs',r.file.slice(0,-4),'native.stdout');
  if(!Number.isSafeInteger(r.observed_output_bytes)||r.observed_output_bytes<r.output_bytes||r.marker!==observed(log,r.cases)||r.output_bytes!==regular(log).size)throw Error('runtime result marker/byte count differs');
  sameIdentity(path.join(out,'runs',r.file.slice(0,-4),'data/current.bin'),pinned);
  if(manifest.status===0&&(r.status!==0||r.signal!==null||!r.marker||r.cases!==rows[i].cases))throw Error('incomplete passing runtime result');
 }
 if(manifest.status===0){
  exact(json(path.join(out,'DERIVED.json')),golden.files,'derived corpus pins differ');
  exact(json(path.join(out,'FIXTURES.json')),rows,'fixture inventory differs');
  for(const row of golden.upstream_files){safeDirectory(path.dirname(path.join(out,'upstream',row.path)));sameIdentity(path.join(out,'upstream',row.path),row);}
  const elfs=json(path.join(out,'ELFS.json'));
  if(!Array.isArray(elfs)||elfs.length!==3||elfs[0].role!=='temporal-compile-output'||elfs[0].path!==binding.originalElf.path||elfs[0].bytes!==binding.originalElf.bytes||elfs[0].sha256!==binding.originalElf.sha256||elfs[1].role!=='producer'||elfs[1].path!==path.join(out,'generator')||elfs[2].role!=='executed-program'||elfs[2].path!==retained||elfs[2].bytes!==binding.originalElf.bytes||elfs[2].sha256!==binding.originalElf.sha256)throw Error('ELF inventory binding differs');
  sameIdentity(elfs[1].path,elfs[1]);sameIdentity(elfs[2].path,elfs[2]);
  if(results.length!==211)throw Error('missing official chunk result');
  const controls=json(path.join(out,'CONTROLS.json'));
  if(controls.length!==2||controls[0].kind!=='absent'||controls[1].kind!=='wrong-level'||controls.some(r=>!Number.isSafeInteger(r.observed_output_bytes)||r.observed_output_bytes<r.output_bytes||r.executedElf!=='program.elf'||r.elf_sha256!==binding.originalElf.sha256||r.status!==1||r.signal!==null))throw Error('runtime controls/ELF differ');
  const absent=fs.readFileSync(path.join(out,'absent.stdout'),'utf8'),wrong=fs.readFileSync(path.join(out,'wrong-level.stdout'),'utf8');
  if(controls[0].output_bytes!==regular(path.join(out,'absent.stdout')).size||controls[1].output_bytes!==regular(path.join(out,'wrong-level.stdout')).size)throw Error('runtime control byte count differs');
  if(absent.includes('BIDI_NATIVE_CONFORMANCE_OK')||!wrong.includes('BIDI_LEVEL_MISMATCH')||wrong.includes('BIDI_NATIVE_CONFORMANCE_OK'))throw Error('runtime negative output differs');
 }
 return {manifest:inputIdentity(path.join(out,'ARTIFACTS.json')),status:manifest.status,files:manifest.files.length,executedElf:manifest.executedElf};
}
export async function executeBidiFixture(root,test,elf,out){
 root=fs.realpathSync(root);elf=path.resolve(elf);out=path.resolve(out);
 if(test!==TEST)throw Error('unregistered runtime fixture test');
 safeDirectory(path.dirname(elf));regular(elf);safeDirectory(path.dirname(out));
 if(fs.readFileSync(elf).subarray(0,4).toString('hex')!=='7f454c46')throw Error('native ELF required');
 const original=inputIdentity(elf),retained=path.join(out,'program.elf');
 fs.mkdirSync(out);fs.writeFileSync(path.join(out,'STATUS'),'1\n');
 let initial=[original],okay=false,failure=null;
 writeJson(path.join(out,'INPUTS.json'),initial);
 try{
  fs.copyFileSync(elf,retained);fs.chmodSync(retained,0o700);sameIdentity(retained,original);
  const cxx=tool(process.env.CXX||'g++'),golden=readOfficialGolden(root);
  const inputs=[TEST,'tests/reference/wave3_bidi_generator.cpp','tests/reference/wave3_bidi_cases.hpp','stdlib/text/data/ucd18-source.tar.gz','stdlib/text/data/UCD18-SOURCES.sha256','scripts/stdlib/runtime_fixtures.mjs',GOLDEN].map(p=>path.join(root,p)).concat([elf,process.execPath,cxx]);
  initial=inputs.map(inputIdentity);writeJson(path.join(out,'INPUTS.json'),initial);
  fs.mkdirSync(path.join(out,'upstream'));
  run('tar',['-xzf',path.join(root,'stdlib/text/data/ucd18-source.tar.gz'),'-C',path.join(out,'upstream')],root,path.join(out,'extract.stdout'));
  run('sha256sum',['-c',path.join(root,'stdlib/text/data/UCD18-SOURCES.sha256')],path.join(out,'upstream'),path.join(out,'upstream.verify'));
  for(const row of golden.upstream_files){safeDirectory(path.dirname(path.join(out,'upstream',row.path)));sameIdentity(path.join(out,'upstream',row.path),row);}
  const generator=path.join(out,'generator');
  run(cxx,['-std=c++23','-O2','-Wall','-Wextra','-Werror',path.join(root,'tests/reference/wave3_bidi_generator.cpp'),'-o',generator],root,path.join(out,'generator.compile.stdout'));
  const corpus=path.join(out,'corpus'),ucd=path.join(out,'upstream/ucd-18.0.0');
  const generated=run(generator,[ucd,corpus],root,path.join(out,'generate.stdout'));
  if(generated.trim()!=='BIDI18_CORPUS_GENERATED cases=861948 class_cases=770241 character_cases=91707 chunks=211')throw Error('generator inventory marker');
  const rows=validateBidiCorpus(corpus,golden),derived=golden.files;
  writeJson(path.join(out,'DERIVED.json'),derived);writeJson(path.join(out,'FIXTURES.json'),rows);
  writeJson(path.join(out,'ELFS.json'),[{...original,role:'temporal-compile-output'},{...inputIdentity(generator),role:'producer'},{...inputIdentity(retained),role:'executed-program'}]);
  const controls=[];writeJson(path.join(out,'CONTROLS.json'),controls);
  const absent=path.join(out,'absent');fs.mkdirSync(absent);
  const missing=await native(retained,absent,path.join(out,'absent.stdout'));
  controls.push({kind:'absent',...missing,executedElf:'program.elf',elf_sha256:original.sha256});writeJson(path.join(out,'CONTROLS.json'),controls);
  if(missing.status!==1||fs.readFileSync(path.join(out,'absent.stdout'),'utf8').includes('BIDI_NATIVE_CONFORMANCE_OK'))throw Error('absent fixture control');
  fs.mkdirSync(path.join(out,'runs'));const receipts=[];writeJson(path.join(out,'RESULTS.json'),receipts);
  for(const row of rows){
   const cwd=path.join(out,'runs',row.file.slice(0,-4));fs.mkdirSync(cwd);fs.mkdirSync(path.join(cwd,'data'));
   const source=path.join(corpus,row.file),copy=path.join(cwd,'data/current.bin');fs.copyFileSync(source,copy);sameIdentity(copy,row);
   const log=path.join(cwd,'native.stdout'),r=await native(retained,cwd,log);
   const result={...row,...r,marker:observed(log,row.cases),executedElf:'program.elf',elf_sha256:original.sha256};
   receipts.push(result);writeJson(path.join(out,'RESULTS.json'),receipts);
   if(r.status!==0||!result.marker)throw Error('native BIDI fixture failed: '+row.file);
   sameIdentity(copy,row);sameIdentity(source,row);sameIdentity(retained,original);
  }
  const mutant=path.join(out,'mutant');run(generator,['--wrong-level-control',ucd,mutant],root,path.join(out,'mutant.generate.stdout'));
  const bad=path.join(out,'wrong-level');fs.mkdirSync(bad);fs.mkdirSync(path.join(bad,'data'));
  fs.copyFileSync(path.join(mutant,'wrong-level.bin'),path.join(bad,'data/current.bin'));
  const wrong=await native(retained,bad,path.join(out,'wrong-level.stdout'));
  controls.push({kind:'wrong-level',...wrong,executedElf:'program.elf',elf_sha256:original.sha256});writeJson(path.join(out,'CONTROLS.json'),controls);
  const badText=fs.readFileSync(path.join(out,'wrong-level.stdout'),'utf8');
  if(wrong.status!==1||!badText.includes('BIDI_LEVEL_MISMATCH')||badText.includes('BIDI_NATIVE_CONFORMANCE_OK'))throw Error('wrong-level fixture accepted');
  exact(validateBidiCorpus(corpus,golden),rows,'corpus changed');
  for(const row of golden.upstream_files){sameIdentity(path.join(out,'upstream',row.path),row);}
  sameIdentity(path.join(root,GOLDEN),initial.find(r=>r.path===path.join(root,GOLDEN)));
  for(const r of json(path.join(out,'ELFS.json'))){sameIdentity(r.path,r);}
  okay=true;
 }catch(e){failure=e;}
 finally{
  let final=[];
  try{final=initial.map(r=>inputIdentity(r.path));if(JSON.stringify(final)!==JSON.stringify(initial)){okay=false;failure??=Error('runtime fixture inputs changed');}}
  catch(e){okay=false;failure??=e;}
  writeJson(path.join(out,'INPUTS.final.json'),final);
  fs.writeFileSync(path.join(out,'STATUS'),okay?'0\n':'1\n');
  if(failure)writeJson(path.join(out,'FAILURE.json'),{message:failure.message});
  const snapshot=tree(out);
  writeJson(path.join(out,'ARTIFACTS.json'),{schema:'sounio.stdlib.runtime-fixture.artifacts.v1',fixture:FIXTURE,test_path:TEST,status:okay?0:1,executedElf:{file:'program.elf',bytes:original.bytes,sha256:original.sha256},self:'ARTIFACTS.json',...snapshot});
 }
 if(failure)throw failure;
 console.log('BIDI_OFFICIAL_RUNTIME_FIXTURES_OK chunks=211 cases=861948');
 console.log('BIDI_NATIVE_CONFORMANCE_OK');
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const [op,root,test,elf,out,...extra]=process.argv.slice(2);
 if(op!=='run'||!out||extra.length){console.error('usage: runtime_fixtures.mjs run ROOT TEST ELF FRESH_OUT');process.exitCode=65;}
 else{try{await executeBidiFixture(root,test,elf,out);}catch(e){console.error(e.stack||e.message);process.exitCode=1;}}
}
