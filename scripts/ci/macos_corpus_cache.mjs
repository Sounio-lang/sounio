#!/usr/bin/env node
// Compilation may be reused; status/stdout/Metal receipts must always be new.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {snapshot as nativeSnapshot,environment} from './macos_native_cache.mjs';
const hash=data=>crypto.createHash('sha256').update(data).digest('hex');
const manifest='tests/manifests/madaros_macos_200.txt';
const recipe=['scripts/ci/madaros_macos_corpus.cpp','scripts/ci/madaros_macos_corpus_gate.sh',
 'scripts/ci/madaros_macos_corpus_diagnostics_gate.sh','scripts/ci/macos_corpus_cache.mjs',
 'scripts/ci/macos_corpus_cache_contract.mjs','scripts/ci/metal_binary64_conformance.cpp',
 'scripts/dev/build_native_metal_runtime.sh','scripts/dev/embed_metal_softfloat.cpp',
 'vendor/metal-softfloat/softfloat64.metal',manifest];
export function snapshot(root,compiler,env=environment()){
 const normalized=structuredClone(env);
 // The gate always sets these values, including when the key step precedes it.
 if(normalized.flags){delete normalized.flags.SOUNIO_METAL_RUNTIME;delete normalized.flags.SOUNIO_METAL_TRACE;}
 const native=nativeSnapshot(root,compiler,normalized);
 const members=fs.readFileSync(path.join(root,manifest),'utf8').split('\n').filter(Boolean);
 assert.equal(members.length,200,'corpus denominator');
 assert.equal(new Set(members).size,200,'duplicate corpus member');
 for(const name of members)assert(path.basename(name)===name&&name.endsWith('.sio'),'unsafe corpus member');
 // Includes tracked, untracked AND ignored source modules, so transitive
 // imports cannot be hidden by .gitignore or a change outside the manifest.
 const sources=execFileSync('git',['ls-files','--cached','--others','-z','--','*.sio','*.sio.in'],{cwd:root,encoding:'utf8'}).split('\0').filter(Boolean);
 const files=[...new Set([...recipe,...sources,...members.map(n=>'tests/run-pass/'+n)])].sort();
 const inputs={schema:'macos-corpus-inputs-v1',native:native.inputs,
  cwd:fs.realpathSync(root),runtimePath:path.resolve(root,'artifacts/self-hosted/libsounio_metal.dylib'),
  target:'aarch64-macos',requireMetal:true,members,
  files:files.map(name=>[name,hash(fs.readFileSync(path.join(root,name)))])};
 return {key:'macos-corpus-v1-'+hash(JSON.stringify(inputs)),inputs};
}
export function seal(expected,cache,results){
 const rows=fs.readFileSync(path.join(results,'results.tsv'),'utf8').trimEnd().split('\n').slice(1).map(l=>l.split('\t'));
 assert.equal(rows.length,200,'incomplete result receipts');
 assert.equal(new Set(rows.map(r=>r[0])).size,200,'duplicate result receipt');
 for(const name of expected.inputs.members){
  const row=rows.find(r=>r[0]===name);assert(row&&row[1]==='pass'&&row[2]==='0','unaccepted member '+name);
 }
 fs.mkdirSync(cache,{recursive:true});
 const programs=expected.inputs.members.map(name=>{
  const src=path.join(results,name),dst=path.join(cache,name);
  assert.equal(hash(fs.readFileSync(path.join(src,'source.sio'))),
   expected.inputs.files.find(([p])=>p==='tests/run-pass/'+name)[1],'changed fixture '+name);
  const payload=fs.readFileSync(path.join(src,'program')),sha256=hash(payload);
  assert.equal(fs.readFileSync(path.join(src,'program.sha256'),'utf8').split(/\s/)[0],sha256,'bad build receipt '+name);
  fs.mkdirSync(dst,{recursive:true});fs.writeFileSync(path.join(dst,'program'),payload,{mode:0o755});
  fs.writeFileSync(path.join(dst,'program.sha256'),sha256+'  program\n');
  return {name,sha256};
 });
 fs.writeFileSync(path.join(cache,'receipt.json'),JSON.stringify({schema:'macos-corpus-cache-v1',key:expected.key,inputs:expected.inputs,programs},null,2)+'\n');
}
export function verify(expected,cache){
 const r=JSON.parse(fs.readFileSync(path.join(cache,'receipt.json'),'utf8'));
 assert.equal(r.schema,'macos-corpus-cache-v1','cache schema');
 assert.equal(r.key,expected.key,'corpus cache key differs from current inputs');
 assert.deepEqual(r.inputs,expected.inputs,'corpus cache recipe differs');
 assert.deepEqual(r.programs.map(p=>p.name),expected.inputs.members,'corpus cache member identity');
 for(const {name,sha256}of r.programs){
  assert.equal(hash(fs.readFileSync(path.join(cache,name,'program'))),sha256,'corpus payload digest '+name);
  assert.equal(fs.readFileSync(path.join(cache,name,'program.sha256'),'utf8'),sha256+'  program\n','corpus digest receipt '+name);
 }
 console.log('MACOS_CORPUS_EXACT_CACHE_VERIFIED key='+expected.key);
 return r.programs;
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href){
 const [mode,root,compiler,cache,results,key]=process.argv.slice(2);
 const expected=snapshot(root,compiler);
 if(mode==='key'){
  if(process.env.GITHUB_OUTPUT)fs.appendFileSync(process.env.GITHUB_OUTPUT,'key='+expected.key+'\n');
  console.log(expected.key);
 }else if(mode==='verify'){
  for(const p of verify(expected,cache))console.log('MACOS_CORPUS_PROGRAM '+p.name+' '+p.sha256);
 }
 else if(mode==='seal'){assert.equal(expected.key,key,'corpus inputs changed during execution');seal(expected,cache,results);}
 else throw Error('expected key, verify or seal');
}
