#!/usr/bin/env node
// Operational metadata controls only. This never compiles or executes Sounio,
// never computes BIDI answers, and never represents a scientific conformance run.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {readOfficialGolden,verifyBidiArtifacts} from './runtime_fixtures.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const TEST='tests/stdlib/wave3-bidi/conformance.sio',GOLDEN='tests/fixtures/wave4_bidi_runtime/official-corpus.json';
const helper=path.join(root,'scripts/stdlib/stdlib_e2e_diagnostics.mjs');
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'bidi-artifacts-controls-'));
const wrapper=path.join(tmp,'compiler-envelope-test-double');fs.copyFileSync('/bin/true',wrapper);
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const id=p=>({path:p,real_path:fs.realpathSync(p),sha256:hash(p),bytes:fs.statSync(p).size});
const j=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const put=(p,x)=>fs.writeFileSync(p,JSON.stringify(x,null,2)+'\n');
let index=0,checks=0;
function call(...args){return spawnSync(process.execPath,[helper,...args],{encoding:'utf8',maxBuffer:16*1024*1024});}
function okay(...args){const r=call(...args);assert.equal(r.status,0,r.stderr);return r.stdout;}
function check(name,fn){fn();console.log('PASS '+(++checks)+' '+name);}
function tree(out){
 const files=[],directories=[];
 function visit(dir){for(const n of fs.readdirSync(dir).sort()){const p=path.join(dir,n),file=path.relative(out,p).split(path.sep).join('/'),s=fs.lstatSync(p);if(s.isDirectory()){directories.push(file);visit(p);}else if(file!=='ARTIFACTS.json')files.push({file,bytes:s.size,sha256:hash(p)});}}
 visit(out);files.sort((a,b)=>a.file<b.file?-1:a.file>b.file?1:0);directories.sort();return {files,directories};
}
function artifact(out,binding){
 fs.mkdirSync(out);const original=binding.originalElf;
 fs.copyFileSync(original.path,path.join(out,'program.elf'));
 const inputs=[original,binding.helper,binding.golden].map(r=>({path:r.path,bytes:r.bytes,sha256:r.sha256}));
 put(path.join(out,'INPUTS.json'),inputs);put(path.join(out,'INPUTS.final.json'),inputs);
 fs.writeFileSync(path.join(out,'STATUS'),'1\n');
 put(path.join(out,'FAILURE.json'),{message:'operational producer failure test double; no Sounio execution'});
 put(path.join(out,'ARTIFACTS.json'),{schema:'sounio.stdlib.runtime-fixture.artifacts.v1',fixture:'unicode18-bidi',test_path:TEST,status:1,executedElf:{file:'program.elf',bytes:original.bytes,sha256:original.sha256},self:'ARTIFACTS.json',...tree(out)});
}
function prepare(){
 const out=path.join(tmp,'capture-'+(++index)),original=path.join(tmp,'original-'+index+'.elf');
 fs.copyFileSync('/bin/true',original);
 okay('init',out,root,wrapper,path.join(root,'stdlib'));
 const compile=['node',path.join(root,'scripts/stdlib/bounded_command.mjs'),'300','16777216','--',wrapper,'compile',path.join(root,TEST),'-o',original];
 const raw=okay('begin',out,TEST,'compile',...compile);
 fs.writeFileSync(raw,'operational compile-envelope fixture; command not executed\n');
 okay('finish',out,TEST,'compile','0');
 const fixture=path.join(out,TEST,'runtime-fixture');
 const native=['node',path.join(root,'scripts/stdlib/bounded_command.mjs'),'600','16777216','--','node',path.join(root,'scripts/stdlib/runtime_fixtures.mjs'),'run',root,TEST,original,fixture];
 const log=okay('begin',out,TEST,'native',...native);
 const start=j(path.join(out,TEST,'native/start.json'));
 artifact(fixture,start.runtime_fixture);
 fs.writeFileSync(log,'operational producer failure fixture; no Sounio execution\n');
 okay('finish',out,TEST,'native','1');
 fs.unlinkSync(original);
 return {out,original,fixture,native:path.join(out,TEST,'native'),start};
}
function refused(x){const r=call('verify',x.out,TEST,'native','1');assert.equal(r.status,65,r.stdout);assert.match(r.stderr,/E2E artifact capture/);}
function mutate(name,fn){check(name,()=>{const x=prepare();fn(x);refused(x);});}
function rehashArtifact(x){
 const p=path.join(x.fixture,'ARTIFACTS.json'),m=j(p);Object.assign(m,tree(x.fixture));put(p,m);
 const r=path.join(x.native,'result.json'),data=j(r);data.runtime_fixture_artifacts.manifest=id(p);data.runtime_fixture_artifacts.files=m.files.length;put(r,data);
}
readOfficialGolden(root);
check('recoverable failure artifact finalizes after temporary ELF unlink',()=>{
 const x=prepare();assert(!fs.existsSync(x.original));
 okay('verify',x.out,TEST,'native','1');okay('finalize',x.out);
 const complete=j(path.join(x.out,'complete.json'));assert.equal(complete.captured_phases,2);
 assert.equal(j(path.join(x.fixture,'ARTIFACTS.json')).status,1);
});
mutate('retained ELF bytes changed',x=>fs.appendFileSync(path.join(x.fixture,'program.elf'),'changed'));
mutate('retained ELF changed even after manifest and footer reseal',x=>{fs.appendFileSync(path.join(x.fixture,'program.elf'),'changed');rehashArtifact(x);});
mutate('unlisted artifact file',x=>fs.writeFileSync(path.join(x.fixture,'extra'),'unexpected'));
mutate('missing artifact file',x=>fs.unlinkSync(path.join(x.fixture,'FAILURE.json')));
mutate('symlink artifact file',x=>{fs.unlinkSync(path.join(x.fixture,'FAILURE.json'));fs.symlinkSync(path.join(root,TEST),path.join(x.fixture,'FAILURE.json'));});
mutate('symlink artifact directory',x=>fs.symlinkSync(root,path.join(x.fixture,'linked-directory')));
mutate('manifest escape path',x=>{const p=path.join(x.fixture,'ARTIFACTS.json'),m=j(p);m.files[0].file='../outside';put(p,m);});
mutate('duplicate manifest entry',x=>{const p=path.join(x.fixture,'ARTIFACTS.json'),m=j(p);m.files.push(m.files[0]);put(p,m);});
mutate('manifest wrong byte count',x=>{const p=path.join(x.fixture,'ARTIFACTS.json'),m=j(p);m.files[0].bytes++;put(p,m);});
mutate('manifest missing file row',x=>{const p=path.join(x.fixture,'ARTIFACTS.json'),m=j(p);m.files.pop();put(p,m);});
mutate('extra empty directory',x=>fs.mkdirSync(path.join(x.fixture,'unlisted-directory')));
mutate('manifest substitution is bound by native result',x=>{const p=path.join(x.fixture,'ARTIFACTS.json'),m=j(p);m.extra='tamper';put(p,m);});
mutate('fixture executed ELF differs from native-start hash',x=>{const p=path.join(x.fixture,'ARTIFACTS.json'),m=j(p);m.executedElf.sha256='0'.repeat(64);put(p,m);rehashArtifact(x);});
mutate('wrong successful status cannot hide missing official chunks',x=>{const p=path.join(x.fixture,'ARTIFACTS.json'),m=j(p);m.status=0;put(p,m);fs.writeFileSync(path.join(x.fixture,'STATUS'),'0\n');rehashArtifact(x);});
mutate('wrong helper in both phase envelopes',x=>{for(const n of ['start.json','result.json']){const p=path.join(x.native,n),r=j(p);r.command[6]=path.join(root,'scripts/stdlib/argv_inputs.mjs');put(p,r);}});
mutate('wrong output in both phase envelopes',x=>{for(const n of ['start.json','result.json']){const p=path.join(x.native,n),r=j(p);r.command[11]=path.join(tmp,'wrong-output');put(p,r);}});
mutate('wrong compile-output binding',x=>{const p=path.join(x.out,TEST,'compile/result.json'),r=j(p);r.command[9]=path.join(tmp,'other.elf');put(p,r);});
mutate('runtime artifact tree moved away from exact binding',x=>fs.renameSync(x.fixture,path.join(x.fixture,'..','other-tree')));
mutate('changed final input provenance',x=>{const p=path.join(x.fixture,'INPUTS.final.json'),r=j(p);r[0].sha256='f'.repeat(64);put(p,r);rehashArtifact(x);});
check('runtime-fixture directory at another path refused by census',()=>{
 const x=prepare(),dir=path.join(x.out,'tests/stdlib/unregistered.sio/runtime-fixture');fs.mkdirSync(dir,{recursive:true});
 const r=call('finalize',x.out);assert.equal(r.status,65);assert.match(r.stderr,/registered path/);
});
check('actual helper seals recoverable producer failure without native execution',()=>{
 const original=path.join(tmp,'actual-failure-input.elf'),out=path.join(tmp,'actual-producer-failure');fs.copyFileSync('/bin/true',original);
 const r=spawnSync(process.execPath,[path.join(root,'scripts/stdlib/runtime_fixtures.mjs'),'run',root,TEST,original,out],{env:{...process.env,CXX:'/bin/false'},encoding:'utf8',maxBuffer:16*1024*1024});
 assert.equal(r.status,1,r.stderr);assert.match(r.stderr,/producer failed/);
 const binding={fixture:'unicode18-bidi',test_path:TEST,output:out,executedElfPath:path.join(out,'program.elf'),originalElf:id(original),helper:id(path.join(root,'scripts/stdlib/runtime_fixtures.mjs')),golden:id(path.join(root,GOLDEN))};
 fs.unlinkSync(original);const result=verifyBidiArtifacts(root,TEST,out,binding,1);
 assert.equal(result.status,1);assert(result.files>10);assert(!fs.existsSync(path.join(out,'RESULTS.json')));
});

let retainedOperationalElf=null;
check('actual partial helper keeps fixture bytes distinct from native output bytes',()=>{
 const source=path.join(tmp,'metadata-only-elf.cpp'),original=path.join(tmp,'metadata-only-elf'),out=path.join(tmp,'actual-partial-failure');
 fs.writeFileSync(source,"#include <cstdint>\n#include <cstdlib>\n#include <filesystem>\n#include <fstream>\n#include <iostream>\n#include <string>\n// Operational ELF test double: reads only protocol headers, never BIDI levels.\nint main(){\n std::ifstream in(\"data/current.bin\",std::ios::binary);\n if(!in)return 1;\n unsigned char header[16]{};in.read(reinterpret_cast<char*>(header),16);\n if(in.gcount()!=16)return 1;\n if(std::filesystem::current_path().filename()==\"000001\"){\n  if(std::getenv(\"OVERFLOW_METADATA_DOUBLE\"))std::cout<<std::string(2*1024*1024,'x');\n  else std::cout<<\"INTENTIONAL_METADATA_DOUBLE_REFUSAL\\n\";\n  return 1;\n }\n const auto count=std::uint32_t(header[12])|(std::uint32_t(header[13])<<8)|(std::uint32_t(header[14])<<16)|(std::uint32_t(header[15])<<24);\n std::cout<<\"BIDI_NATIVE_CONFORMANCE_OK\\nBIDI_CASES_OK \"<<count<<\"\\n\";\n return 0;\n}\n");
 const built=spawnSync('g++',['-std=c++23','-O2','-Wall','-Wextra','-Werror',source,'-o',original],{encoding:'utf8'});
 assert.equal(built.status,0,built.stderr);
 const r=spawnSync(process.execPath,[path.join(root,'scripts/stdlib/runtime_fixtures.mjs'),'run',root,TEST,original,out],{env:{...process.env,CXX:'g++'},encoding:'utf8',maxBuffer:16*1024*1024});
 assert.equal(r.status,1,r.stderr);assert.match(r.stderr,/native BIDI fixture failed: 000001.bin/);
 const records=j(path.join(out,'RESULTS.json'));assert.equal(records.length,2);
 for(const record of records){
  const fixture=path.join(out,'runs',record.file.slice(0,-4),'data/current.bin'),log=path.join(out,'runs',record.file.slice(0,-4),'native.stdout');
  assert.equal(record.bytes,fs.statSync(fixture).size);assert.equal(record.output_bytes,fs.statSync(log).size);
  assert.notEqual(record.bytes,record.output_bytes);
 }
 console.log('BIDI_PARTIAL_METADATA_BYTES '+JSON.stringify(records.map(r=>({file:r.file,fixture_bytes:r.bytes,output_bytes:r.output_bytes,status:r.status,marker:r.marker}))));
 assert.equal(records[0].status,0);assert.equal(records[0].marker,true);assert.equal(records[1].status,1);assert.equal(records[1].marker,false);
 const binding={fixture:'unicode18-bidi',test_path:TEST,output:out,executedElfPath:path.join(out,'program.elf'),originalElf:id(original),helper:id(path.join(root,'scripts/stdlib/runtime_fixtures.mjs')),golden:id(path.join(root,GOLDEN))};
 fs.unlinkSync(original);assert.equal(verifyBidiArtifacts(root,TEST,out,binding,1).status,1);
 retainedOperationalElf=path.join(out,'program.elf');
});


check('actual partial output limit seals stored bytes separately from observed bytes',()=>{
 assert(retainedOperationalElf);
 const original=path.join(tmp,'metadata-overflow-elf'),out=path.join(tmp,'actual-partial-output-limit');
 fs.copyFileSync(retainedOperationalElf,original);
 const r=spawnSync(process.execPath,[path.join(root,'scripts/stdlib/runtime_fixtures.mjs'),'run',root,TEST,original,out],{env:{...process.env,CXX:'g++',OVERFLOW_METADATA_DOUBLE:'1'},encoding:'utf8',maxBuffer:16*1024*1024});
 assert.equal(r.status,1,r.stderr);assert.match(r.stderr,/native BIDI fixture failed: 000001.bin/);
 const records=j(path.join(out,'RESULTS.json'));assert.equal(records.length,2);assert.equal(records[1].status,122);
 const log=path.join(out,'runs/000001/native.stdout'),limited=records[1];
 assert.equal(limited.output_bytes,fs.statSync(log).size);assert(limited.output_bytes<=1048576);
 assert(limited.observed_output_bytes>1048576);assert(limited.observed_output_bytes>limited.output_bytes);
 assert.equal(limited.bytes,fs.statSync(path.join(out,'runs/000001/data/current.bin')).size);
 console.log('BIDI_OUTPUT_LIMIT_METADATA_BYTES '+JSON.stringify({fixture_bytes:limited.bytes,stored_output_bytes:limited.output_bytes,observed_output_bytes:limited.observed_output_bytes,status:limited.status}));
 const binding={fixture:'unicode18-bidi',test_path:TEST,output:out,executedElfPath:path.join(out,'program.elf'),originalElf:id(original),helper:id(path.join(root,'scripts/stdlib/runtime_fixtures.mjs')),golden:id(path.join(root,GOLDEN))};
 fs.unlinkSync(original);assert.equal(verifyBidiArtifacts(root,TEST,out,binding,1).status,1);
});


check('explicit fixture never falls back and retains artifacts without capture option',()=>{
 const project=path.join(tmp,'runner-project');
 for(const file of ['scripts/stdlib/run_stdlib_e2e.sh','scripts/stdlib/runtime_fixtures.mjs','scripts/stdlib/argv_inputs.mjs','scripts/stdlib/bounded_command.mjs','scripts/stdlib/emit_stdlib_e2e_result.mjs','scripts/lib/resolve_souc.sh',GOLDEN,...readOfficialGolden(root).source_files.map(r=>r.path)]){
  const target=path.join(project,file);fs.mkdirSync(path.dirname(target),{recursive:true});fs.copyFileSync(path.join(root,file),target);
 }
 const source=path.join(project,TEST);fs.mkdirSync(path.dirname(source),{recursive:true});
 fs.writeFileSync(source,'//@ run-pass\n//@ runtime-fixture: unicode18-bidi\nOperational source text; deliberately never parsed by a Sounio compiler.\n');
 const invocations=path.join(tmp,'runner-invocations'),compiler=path.join(tmp,'runner-compiler-test-double');
 fs.writeFileSync(compiler,'#!/bin/bash\nprintf "%s\\n" "$1" >> "$INVOCATIONS"\nif test "$1" = check;then exit 0;fi\nif test "$1" = compile && test "$3" = -o;then cp /bin/true "$4";exit 0;fi\nexit 64\n');fs.chmodSync(compiler,0o700);
 const env={...process.env,SOUC_BIN:compiler,SOUNIO_STDLIB_PATH:path.join(project,'stdlib'),INVOCATIONS:invocations,CXX:'/bin/false',STDLIB_E2E_RUN_ALL_RUN_PASS:'0'};
 const result=spawnSync('bash',[path.join(project,'scripts/stdlib/run_stdlib_e2e.sh'),'--filter',TEST],{env,encoding:'utf8',maxBuffer:16*1024*1024});
 assert.equal(result.status,1,result.stdout+result.stderr);
 assert.match(fs.readFileSync(invocations,'utf8'),/^check\ncompile\n$/);
 assert(!result.stdout.includes('run-pass-check-only'));
 const match=/Runtime fixture artifacts retained at: ([^\n]+)/.exec(result.stderr);assert(match,result.stderr);
 assert(fs.statSync(path.join(match[1],'program.elf')).isFile());assert.equal(j(path.join(match[1],'ARTIFACTS.json')).status,1);
 put(path.join(tmp,'RUNNER_RETENTION.json'),{scope:'Operational compile-envelope test double and actual producer failure; no Sounio execution',status:result.status,retained:match[1],program:id(path.join(match[1],'program.elf')),manifest:id(path.join(match[1],'ARTIFACTS.json')),stdout:result.stdout,stderr:result.stderr});
 fs.symlinkSync(path.join(tmp,'absent-argv-target'),source+'.args');
 const before=fs.readFileSync(invocations,'utf8');
 const badJson=path.join(tmp,'dangling-args-result.json');
 const bad=spawnSync('bash',[path.join(project,'scripts/stdlib/run_stdlib_e2e.sh'),'--filter',TEST,'--json-out',badJson],{env,encoding:'utf8'});
 assert.equal(bad.status,1);assert.equal(fs.readFileSync(invocations,'utf8'),before);
 assert.equal(j(badJson).results[0].reason,'invalid-runtime-fixture-contract');
});

console.log('BIDI_RUNTIME_ARTIFACT_METADATA_CONTROLS_OK count='+checks+' temporary='+tmp);
