#!/usr/bin/env node
// Transport is untrusted until exact current inputs and payload digest agree.
// This caches compilation only; the builder still runs its seed/ABI probes.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
const digest=data=>crypto.createHash('sha256').update(data).digest('hex');
const scopes=['self-hosted','stdlib','tests/run-pass/a64_macos_mapped_stack.sio','scripts/ci/build_madaros_macos_arm64.sh','scripts/ci/native_build_profile.cpp','scripts/ci/macos_native_cache.mjs','scripts/ci/macos_native_cache_contract.mjs','.github/workflows/ci.yml'];
export function environment(){
 assert.equal(process.platform,'darwin','native cache requires Darwin');
 assert.equal(process.arch,'arm64','native cache requires arm64');
 const command=(exe,args)=>execFileSync(exe,args,{encoding:'utf8'}).trim();
 return {
  platform:process.platform,arch:process.arch,
  // Bind compiler flags without recording raw environment values. The builder
  // forces STDLIB_PATH to this checkout; RESTORE is transport, not compilation.
  flags:Object.fromEntries(Object.keys(process.env).sort().filter(k=>
   (/^(SOUNIO_|SOUC_|MADAROS_)/.test(k)||['LANG','LC_ALL','LC_CTYPE','TZ','SDKROOT','MACOSX_DEPLOYMENT_TARGET','DEVELOPER_DIR'].includes(k))
   &&!['SOUNIO_MACOS_NATIVE_CACHE_RESTORE','SOUNIO_STDLIB_PATH'].includes(k)
  ).map(k=>[k,digest(process.env[k])])),
  os:command('/usr/bin/sw_vers',['-buildVersion']),
  clang:command('/usr/bin/xcrun',['clang','--version']),
  sdk:command('/usr/bin/xcrun',['--sdk','macosx','--show-sdk-version']),
  sdkBuild:command('/usr/bin/xcrun',['--sdk','macosx','--show-sdk-build-version']),
  developer:command('/usr/bin/xcode-select',['-p'])
 };
}
export function snapshot(root,seed,env=environment()){
 const files=execFileSync('git',['ls-files','--cached','--others','--exclude-standard','-z','--',...scopes],{cwd:root,encoding:'utf8'}).split('\0').filter(Boolean);
 // Include ignored source modules too: Git visibility must not hide a compiler input.
 function sourceFiles(dir){
  for(const entry of fs.readdirSync(path.join(root,dir),{withFileTypes:true})){
   const name=path.posix.join(dir,entry.name);
   if(entry.isDirectory())sourceFiles(name);
   else if(entry.isSymbolicLink()&&fs.statSync(path.join(root,name)).isDirectory())throw Error('source directory symlink unsupported: '+name);
   else if(name.endsWith('.sio'))files.push(name);
  }
 }
 for(const dir of ['self-hosted','stdlib'])sourceFiles(dir);
 const entries=[...new Set(files)].sort().map(name=>{
  const file=path.join(root,name),stat=fs.lstatSync(file);
  assert(stat.isFile()||stat.isSymbolicLink(),'unsupported input '+name);
  return [name,stat.isSymbolicLink()?fs.readlinkSync(file):'file',digest(fs.readFileSync(file))];
 });
 for(const scope of ['self-hosted/','stdlib/'])assert(entries.some(([n])=>n.startsWith(scope)),'missing input tree '+scope);
 for(const name of scopes.filter(n=>n!=='self-hosted'&&n!=='stdlib'))assert(entries.some(([n])=>n===name),'missing recipe input '+name);
 const inputs={schema:'macos-native-inputs-v1',environment:env,seed:digest(fs.readFileSync(seed)),entries};
 return {key:'macos-native-v1-'+digest(JSON.stringify(inputs)),inputs};
}
export function seal(expected,cache,binary){
 fs.mkdirSync(cache,{recursive:true});
 fs.copyFileSync(binary,path.join(cache,'madaros'));
 fs.writeFileSync(path.join(cache,'receipt.json'),JSON.stringify({schema:'macos-native-cache-v1',key:expected.key,inputs:expected.inputs,sha256:digest(fs.readFileSync(path.join(cache,'madaros')))},null,2)+'\n');
}
export function restore(expected,cache,out){
 const receipt=JSON.parse(fs.readFileSync(path.join(cache,'receipt.json'),'utf8'));
 assert.equal(receipt.schema,'macos-native-cache-v1','cache schema');
 assert.equal(receipt.key,expected.key,'cache key differs from current inputs');
 assert.deepEqual(receipt.inputs,expected.inputs,'cache recipe differs from current inputs');
 const payload=fs.readFileSync(path.join(cache,'madaros'));
 assert.equal(digest(payload),receipt.sha256,'cached binary digest mismatch');
 fs.writeFileSync(out,payload,{mode:0o755});
 console.log('MACOS_NATIVE_EXACT_CACHE_VERIFIED key='+expected.key);
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href){
 const [mode,root,seed,cache,out,key]=process.argv.slice(2);
 const expected=snapshot(root,seed);
 if(mode==='key'){
  if(process.env.GITHUB_OUTPUT)fs.appendFileSync(process.env.GITHUB_OUTPUT,'key='+expected.key+'\n');
  console.log(expected.key);
 }else if(mode==='seal'){assert.equal(expected.key,key,'inputs changed during acceptance');seal(expected,cache,out);}
 else if(mode==='restore')restore(expected,cache,out);
 else throw Error('expected key, seal or restore');
}
