#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import {snapshot,seal,verify} from './macos_corpus_cache.mjs';
const root=path.resolve(process.argv[2]||'.'),get=p=>fs.readFileSync(path.join(root,p),'utf8');
const cpp=get('scripts/ci/madaros_macos_corpus.cpp'),gate=get('scripts/ci/madaros_macos_corpus_gate.sh'),workflow=get('.github/workflows/ci.yml');
const digest=s=>crypto.createHash('sha256').update(s).digest('hex');
const oracleFingerprint='e0c4006a7deb3bbed718df517e0e7a96da9ccf7c854aadd361f73d756241486d';
function check(w,c,g){
 const mac=w.split('  native-selfhost-macos-arm64:\n')[1].split(/\n  [a-z][a-z0-9-]+:/)[0];
 assert(!mac.includes('restore-keys:'),'prefix restore forbidden');
 assert(mac.includes('Upload native corpus execution receipts')&&mac.includes('/tmp/madaros-corrected-corpus/**/*.tsv')&&mac.includes('/tmp/madaros-corrected-corpus/**/*.sha256'),'execution receipts must be published');
 const step=mac.split('      - name: Corrected pinned 200 corpus with mandatory real Metal\n')[1].split('      - name: ')[0];
 assert(!/^        if:/m.test(step),'corpus cannot be skipped');
 assert(step.includes('run: bash scripts/ci/madaros_macos_corpus_gate.sh /tmp/madaros-current-macos /tmp/madaros-corrected-corpus'),'corpus command changed');
 assert(step.includes("steps.corpus-cache.outputs.cache-hit == 'true'"),'reuse only exact HIT');
 for(const x of ['id: corpus-cache-inputs','id: corpus-cache','key: ${{ steps.corpus-cache-inputs.outputs.key }}','path: ${{ runner.temp }}/madaros-corpus-cache','macos_corpus_cache.mjs seal'])assert(mac.includes(x),'missing '+x);
 assert(mac.indexOf('      - name: Seal accepted native corpus compilations\n')>mac.indexOf('      - name: Corrected pinned 200 corpus with mandatory real Metal\n'),'cache saved before runtime acceptance');
 assert(mac.includes('/tmp/madaros-corrected-corpus/results "$EXPECTED_KEY"'),'seal must recheck original inputs');
 assert(w.includes('node scripts/ci/macos_corpus_cache_contract.mjs "$GITHUB_WORKSPACE" --selftest'),'failure controls unwired');
 assert(g.includes('export CORPUS_REQUIRE_METAL=1')&&g.includes('export CORPUS_REQUIRE_200=1'),'full denominator/Metal required');
 assert(g.includes('CORPUS_COMPILE_CACHE_RESTORE= bash scripts/ci/madaros_macos_corpus_diagnostics_gate.sh'),'invalid compiler control must stay fresh');
 assert(c.includes('if(cached && replay)'),'execution replay forbidden with cache');
 assert(c.includes('node scripts/ci/macos_corpus_cache.mjs verify '),'cache inputs must be verified');
 assert(c.indexOf('compiled corpus cache verification failed')<c.indexOf('std::ofstream results(out/"results.tsv")'),'verify before writing result receipts');
 assert(c.includes('shasum -a 256 -c '),'copied payload must be reverified');
 assert(c.includes('else cr=replay?prior.at(name).first:cmd("timeout "+timeout+" "+q(compiler)+" build "+q(src.string())+" --target aarch64-macos -o "+q(bin)+redir("build"));'),'fresh compile removed');
 const oracles=c.slice(c.indexOf('            else if(cmd("file '),c.indexOf('            const std::string mode='));
 assert.equal(digest(oracles),oracleFingerprint,'predecessor format/signature/runtime/exit/stdout/Metal oracles changed');
}
check(workflow,cpp,gate);
if(process.argv.includes('--selftest')){
 for(const [w,c,g]of [
  [workflow.replace('      - name: Corrected pinned 200 corpus with mandatory real Metal\n','      - name: Corrected pinned 200 corpus with mandatory real Metal\n        if: false\n'),cpp,gate],
  [workflow,cpp.replace('if(cached && replay)','if(false)'),gate],
  [workflow,cpp.replace('shasum -a 256 -c ','true # removed '),gate],
  [workflow,cpp.replace('rr!=expected_rc','false'),gate],
  [workflow,cpp.replace('metal_not_executed','pass'),gate],
  [workflow,cpp,gate.replace('export CORPUS_REQUIRE_200=1','export CORPUS_REQUIRE_200=')],
  [workflow,cpp,gate.replace('CORPUS_COMPILE_CACHE_RESTORE= bash','bash')]
 ]){assert(w!==workflow||c!==cpp||g!==gate);assert.throws(()=>check(w,c,g));}
 const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'corpus-cache-contract-'));
 try{
  const fixture=path.join(tmp,'repo'),tools=path.join(tmp,'tools');fs.mkdirSync(fixture);fs.mkdirSync(tools);
  const put=(name,text,mode=0o644)=>{const p=path.join(fixture,name);fs.mkdirSync(path.dirname(p),{recursive:true});fs.writeFileSync(p,text,{mode});};
  const files=['self-hosted/compiler/main.sio','stdlib/core.sio','tests/run-pass/a64_macos_mapped_stack.sio','scripts/ci/build_madaros_macos_arm64.sh','scripts/ci/native_build_profile.cpp','scripts/ci/macos_native_cache.mjs','scripts/ci/macos_native_cache_contract.mjs','.github/workflows/ci.yml',
   'scripts/ci/madaros_macos_corpus.cpp','scripts/ci/madaros_macos_corpus_gate.sh','scripts/ci/madaros_macos_corpus_diagnostics_gate.sh','scripts/ci/macos_corpus_cache.mjs','scripts/ci/macos_corpus_cache_contract.mjs','scripts/ci/metal_binary64_conformance.cpp','scripts/dev/build_native_metal_runtime.sh','scripts/dev/embed_metal_softfloat.cpp','vendor/metal-softfloat/softfloat64.metal'];
  for(const f of files)put(f,'fixture recipe '+f+'\n');
  const names=['gpu_vec_add.sio',...Array.from({length:199},(_,i)=>'fixture_'+i+'.sio')];
  for(const name of names)put('tests/run-pass/'+name,'//@ expect-stdout-contains: sentinel\n');
  put('tests/manifests/madaros_macos_200.txt',names.join('\n')+'\n');
  assert.equal(spawnSync('git',['init','-q',fixture]).status,0);
  const harness=path.join(tmp,'harness'),compiler=path.join(tmp,'compiler');
  const build=spawnSync(process.env.CXX||'c++',['-std=c++23','-O1','-pthread','-Wall','-Wextra','-Werror',path.join(root,'scripts/ci/madaros_macos_corpus.cpp'),'-o',harness],{encoding:'utf8'});
  assert.equal(build.status,0,build.stderr);
  const script=[
   '#!/bin/sh','set -eu','printf "%s\\n" "$2" >> "$COMPILE_TRACE"',
   'if [ "$COMPILE_FAIL" = 1 ]; then echo "fixture compile refusal" >&2; exit 23; fi',
   'out=""','while [ "$#" -gt 0 ]; do if [ "$1" = -o ]; then out="$2"; shift 2; else shift; fi; done',
   'cat > "$out" <<\'PROGRAM\'',
   '#!/bin/sh','set -eu','name=$(basename "$(dirname "$0")")','printf "%s\\n" "$name" >> "$EXEC_TRACE"',
   'if [ "$name" = gpu_vec_add.sio ] && [ "$NO_METAL_TRACE" != 1 ]; then echo "SOUNIO_METAL_EXECUTED device=fixture" >&2; fi',
   'if [ "$name" = fixture_0.sio ] && [ "$FAIL_RUNTIME" = 1 ]; then exit 7; fi',
   'echo sentinel','PROGRAM','chmod +x "$out"',''].join('\n');
  fs.writeFileSync(compiler,script,{mode:0o755});
  fs.writeFileSync(path.join(tools,'file'),'#!/bin/sh\necho "Mach-O 64-bit arm64 executable"\n',{mode:0o755});
  fs.writeFileSync(path.join(tools,'codesign'),'#!/bin/sh\nexit 0\n',{mode:0o755});
  // Keep fixture hashing fast while using an independent SHA256 implementation.
  if(spawnSync('sha256sum',['--version']).status===0)fs.writeFileSync(path.join(tools,'shasum'),'#!/bin/sh\nif [ "$1" = -a ]; then shift 2; fi\nexec sha256sum "$@"\n',{mode:0o755});
  const fixtureEnv={platform:'darwin',arch:'arm64',flags:{},os:'fixture',clang:'fixture',sdk:'fixture'};
  const bridge=path.join(tmp,'verify.mjs');
  fs.writeFileSync(bridge,[
   'import fs from "node:fs";',
   'import crypto from "node:crypto";',
   'import {snapshot,verify} from '+JSON.stringify(pathToFileURL(path.join(root,'scripts/ci/macos_corpus_cache.mjs')).href)+';',
   'const [,,script,mode,root,compiler,cache]=process.argv;',
   'if(mode!=="verify")throw Error("unexpected verifier mode");',
   'const programs=verify(snapshot(root,compiler,JSON.parse(process.env.CORPUS_FIXTURE_ENV)),cache);',
   'for(const p of programs)console.log("MACOS_CORPUS_PROGRAM "+p.name+" "+p.sha256);',
   'if(process.env.TAMPER_AFTER_VERIFY==="1"){const p=cache+"/fixture_0.sio/program";fs.appendFileSync(p,"\\n# changed after verification\\n");fs.writeFileSync(p+".sha256",crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex")+"  program\\n");}',''].join('\n'));
  fs.writeFileSync(path.join(tools,'node'),'#!/bin/sh\nexec "'+process.execPath+'" "'+bridge+'" "$@"\n',{mode:0o755});
  const cache=path.join(tmp,'cache'),trace=path.join(tmp,'exec.trace'),compiles=path.join(tmp,'compile.trace');
  const env={...process.env,PATH:tools+':'+process.env.PATH,CORPUS_REQUIRE_200:'1',CORPUS_REQUIRE_METAL:'1',EXEC_TRACE:trace,COMPILE_TRACE:compiles,FAIL_RUNTIME:'0',NO_METAL_TRACE:'0',COMPILE_FAIL:'0',CORPUS_REPLAY:undefined,CORPUS_FIXTURE_ENV:JSON.stringify(fixtureEnv)};
  function run(label,extra={}){
   fs.writeFileSync(trace,'');fs.writeFileSync(compiles,'');
   const out=path.join(tmp,label),r=spawnSync(harness,[compiler,'tests/manifests/madaros_macos_200.txt',out,'4','10'],{cwd:fixture,env:{...env,...extra},encoding:'utf8',maxBuffer:8*1024*1024});
   return {r,out,executed:fs.readFileSync(trace,'utf8').trim().split('\n').filter(Boolean),compiled:fs.readFileSync(compiles,'utf8').trim().split('\n').filter(Boolean)};
  }
  const fresh=run('fresh');assert.equal(fresh.r.status,0,fresh.r.stderr);assert.equal(fresh.compiled.length,200);assert.deepEqual(new Set(fresh.executed),new Set(names));
  const expected=snapshot(fixture,compiler,fixtureEnv);seal(expected,cache,fresh.out);verify(expected,cache);
  const hotEnv={CORPUS_COMPILE_CACHE_RESTORE:cache};
  const hot=run('hot',hotEnv);assert.equal(hot.r.status,0,hot.r.stderr);assert.equal(hot.compiled.length,0);assert.deepEqual(new Set(hot.executed),new Set(names));
  assert(fs.readFileSync(path.join(hot.out,'summary.txt'),'utf8').includes('compile_verified-cache=200'));
  assert(fs.readFileSync(path.join(fresh.out,'summary.txt'),'utf8').includes('compile_fresh=200'));
  fs.appendFileSync(path.join(cache,'fixture_0.sio/program'),'bad');
  let bad=run('corrupt',hotEnv);assert.equal(bad.r.status,2);assert.equal(bad.executed.length,0);assert.equal(bad.compiled.length,0);assert(!fs.existsSync(path.join(bad.out,'results.tsv')));seal(expected,cache,fresh.out);
  for(const f of ['tests/run-pass/fixture_0.sio','stdlib/core.sio','scripts/ci/madaros_macos_corpus.cpp']){
   const p=path.join(fixture,f),old=fs.readFileSync(p);fs.appendFileSync(p,'changed');
   bad=run('changed-'+path.basename(f),hotEnv);assert.equal(bad.r.status,2);assert.equal(bad.executed.length,0);fs.writeFileSync(p,old);
  }
  put('.gitignore','ignored_module.sio\n');put('tests/ignored_module.sio','ignored transitive source');
  bad=run('ignored-transitive-source',hotEnv);assert.equal(bad.r.status,2);assert.equal(bad.executed.length,0);fs.unlinkSync(path.join(fixture,'tests/ignored_module.sio'));
  const oldCompiler=fs.readFileSync(compiler);fs.appendFileSync(compiler,'\n# changed compiler\n');bad=run('compiler-changed',hotEnv);assert.equal(bad.r.status,2);assert.equal(bad.executed.length,0);fs.writeFileSync(compiler,oldCompiler);
  assert.throws(()=>verify(snapshot(fixture,compiler,{...fixtureEnv,sdk:'changed'}),cache),/key differs/);
  fs.renameSync(path.join(cache,'receipt.json'),path.join(cache,'receipt.saved'));bad=run('missing-receipt',hotEnv);assert.equal(bad.r.status,2);assert.equal(bad.executed.length,0);fs.renameSync(path.join(cache,'receipt.saved'),path.join(cache,'receipt.json'));
  bad=run('runtime-fails',{...hotEnv,FAIL_RUNTIME:'1'});assert.equal(bad.r.status,1);assert.equal(bad.compiled.length,0);assert.deepEqual(new Set(bad.executed),new Set(names));assert(fs.readFileSync(path.join(bad.out,'results.tsv'),'utf8').includes('fixture_0.sio\texit_mismatch\t0\t7'));
  bad=run('metal-missing',{...hotEnv,NO_METAL_TRACE:'1'});assert.equal(bad.r.status,1);assert.equal(bad.compiled.length,0);assert(fs.readFileSync(path.join(bad.out,'results.tsv'),'utf8').includes('gpu_vec_add.sio\tmetal_not_executed'));
  bad=run('replay-forbidden',{...hotEnv,CORPUS_REPLAY:'1'});assert.equal(bad.r.status,2);assert.equal(bad.executed.length,0);
  bad=run('copy-tamper',{...hotEnv,TAMPER_AFTER_VERIFY:'1'});assert.equal(bad.r.status,1);assert.equal(bad.compiled.length,0);assert(!bad.executed.includes('fixture_0.sio'));assert(fs.readFileSync(path.join(bad.out,'results.tsv'),'utf8').includes('fixture_0.sio\tcompile_cache_invalid'));seal(expected,cache,fresh.out);
  bad=run('compile-refusal',{COMPILE_FAIL:'1'});assert.equal(bad.r.status,1);assert.equal(bad.executed.length,0);assert(bad.r.stderr.includes('fixture compile refusal'));
  console.log('CORPUS_REAL_HARNESS_FRESH200_AND_CACHE_EXECUTION200_PASS');
  console.log('CORPUS_REAL_DIGEST_INPUT_REFUSAL_RUNTIME_AND_METAL_FAILURE_CONTROLS_PASS');
 }finally{fs.rmSync(tmp,{recursive:true,force:true});}
}
console.log('CORPUS_COMPILE_CACHE_FRESH_EXECUTION_CONTRACT_PASS');
