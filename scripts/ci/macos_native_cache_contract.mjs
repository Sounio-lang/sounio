#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {snapshot,seal,restore} from './macos_native_cache.mjs';
const root=path.resolve(process.argv[2]||'.');
const workflow=fs.readFileSync(path.join(root,'.github/workflows/ci.yml'),'utf8');
const builder=fs.readFileSync(path.join(root,'scripts/ci/build_madaros_macos_arm64.sh'),'utf8');
function check(w,b){
 const mac=w.split('  native-selfhost-macos-arm64:\n')[1].split(/\n  [a-z][a-z0-9-]+:/)[0];
 assert(!mac.includes('restore-keys:'),'prefix restore forbidden');
 assert(mac.includes("uses: actions/setup-node@v4\n        with:\n          node-version: '22'"),'verifier runtime unavailable');
 for(const part of ['uses: actions/cache/restore@v4','uses: actions/cache/save@v4','key: ${{ steps.native-cache-inputs.outputs.key }}','path: ${{ runner.temp }}/madaros-native-cache','run: node scripts/ci/macos_native_cache.mjs key','run: node scripts/ci/macos_native_cache.mjs seal'])assert(mac.includes(part),'missing '+part);
 const build=mac.split('      - name: Build and execute current-source native Madaros\n')[1].split('      - name: ')[0];
 assert(!/^        if:/m.test(build),'verified builder must always run');
 assert(build.includes("steps.native-cache.outputs.cache-hit == 'true'"),'reuse only exact hit');
 const gates=[
 ['Native Madaros ABI acceptance','bash scripts/ci/madaros_macos_arm64_acceptance_gate.sh /tmp/madaros-current-macos'],
 ['Native module registry capacity and scalar bool globals','bash scripts/ci/module_registry_capacity_gate.sh /tmp/madaros-current-macos /tmp/madaros-module-registry'],
 ['Native type environment capacity and binding refusal','bash scripts/ci/type_env_capacity_gate.sh /tmp/madaros-current-macos /tmp/madaros-type-env'],
 ['Native IR tail growth and ARM64 address-window contract','bash scripts/ci/ir_tail_growth_gate.sh /tmp/madaros-current-macos /tmp/madaros-ir-tail'],
 ['Native Madaros positive and negative regressions','bash scripts/ci/madaros_macos_port_regressions_gate.sh /tmp/madaros-current-macos /tmp/madaros-port-regressions'],
 ['Native wide arithmetic with exact C++23 oracle','bash scripts/ci/madaros_macos_wide_ir_gate.sh "$PWD/artifacts/self-hosted/souc-self-hosted-arm64-macos" /tmp/madaros-wide-ir'],
 ['Native CPU real-IR runtime gate','bash scripts/ci/madaros_macos_corpus_ir_gate.sh "$PWD/artifacts/self-hosted/souc-self-hosted-arm64-macos"'],
 ['Native Metal real-IR execution gate','bash scripts/ci/madaros_macos_metal_ir_gate.sh "$PWD/artifacts/self-hosted/souc-self-hosted-arm64-macos" /tmp/madaros-metal-ir'],
 ['Corrected pinned 200 corpus with mandatory real Metal','bash scripts/ci/madaros_macos_corpus_gate.sh /tmp/madaros-current-macos /tmp/madaros-corrected-corpus']
 ];
 let previous=mac.indexOf('      - name: Build and execute current-source native Madaros\n');
 for(const [name,command]of gates){
  const marker='      - name: '+name+'\n';
  const at=mac.indexOf(marker);assert(at>previous,'acceptance order '+name);previous=at;
  const step=mac.slice(at+marker.length).split('      - name: ')[0];
  assert(!/^        if:/m.test(step),'acceptance skipped '+name);
  assert(step.includes('        run: '+command+'\n'),'acceptance changed '+name);
  if(name==='Native wide arithmetic with exact C++23 oracle')assert(step.includes('MADAROS_RAW_BIN: /tmp/madaros-current-macos'),'wide gate must reuse accepted candidate');
 }
 assert(mac.indexOf('      - name: Seal accepted native macOS compilation\n')>previous,'cache saved before all acceptance');
 assert(mac.includes('/tmp/madaros-current-macos "$EXPECTED_KEY"'),'seal must recheck original input key');
 assert(w.includes('run: node scripts/ci/macos_native_cache_contract.mjs "$GITHUB_WORKSPACE" --selftest'),'contract unwired');
 const reuse=b.indexOf('if [[ -n "${SOUNIO_MACOS_NATIVE_CACHE_RESTORE:-}" ]]');
 assert(reuse>b.indexOf('test "$(<"$BUILD_DIR/stack.stdout.log")" = 20000'),'seed execution bypassed');
 assert(b.includes('node "$ROOT_DIR/scripts/ci/macos_native_cache.mjs" restore "$ROOT_DIR" "$SEED_BIN" "$SOUNIO_MACOS_NATIVE_CACHE_RESTORE" "$BUILD_DIR/madaros"'),'unchecked cache');
 assert(b.includes('"$BUILD_DIR/profile-build" "$BUILD_DIR/profile" "$SEED_BIN" self-hosted/compiler/main.sio "$BUILD_DIR/madaros" --target aarch64-macos >"$BUILD_DIR/build.log" 2>&1\nfi'),'cold build bypassed');
 assert(b.indexOf('timeout 120 bash "$ROOT_DIR/scripts/ci/madaros_macos_arm64_acceptance_gate.sh"')>reuse,'ABI acceptance bypassed');
 assert(b.includes('codesign --verify "$BUILD_DIR/madaros"'),'signature verification missing');
}
check(workflow,builder);
if(process.argv.includes('--selftest')){
 for(const w of [
  workflow.replace('        id: native-cache\n','        id: native-cache\n        restore-keys: unsafe\n'),
  workflow.replace('      - name: Native Madaros ABI acceptance\n','      - name: Native Madaros ABI acceptance\n        if: false\n'),
  workflow.replace('run: bash scripts/ci/madaros_macos_corpus_gate.sh','run: true # bash scripts/ci/madaros_macos_corpus_gate.sh'),
  workflow.replace('      - name: Build and execute current-source native Madaros\n','      - name: Build and execute current-source native Madaros\n        if: false\n'),
  workflow.replace("steps.native-cache.outputs.cache-hit == 'true'","steps.native-cache.outputs.cache-hit != 'true'")
 ]){assert.notEqual(w,workflow);assert.throws(()=>check(w,builder),'accepted mutation: '+w.split('\n').filter((line,i)=>line!==workflow.split('\n')[i]).slice(0,3).join(' | '));}
 const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'macos-native-cache-'));
 try{
  const fixture=path.join(tmp,'repo');fs.mkdirSync(fixture);
  execFileSync('git',['init','-q',fixture]);
  for(const name of ['self-hosted/compiler/main.sio','stdlib/core.sio','tests/run-pass/a64_macos_mapped_stack.sio','scripts/ci/build_madaros_macos_arm64.sh','scripts/ci/native_build_profile.cpp','scripts/ci/macos_native_cache.mjs','scripts/ci/macos_native_cache_contract.mjs','.github/workflows/ci.yml']){
   const p=path.join(fixture,name);fs.mkdirSync(path.dirname(p),{recursive:true});fs.writeFileSync(p,'fixture '+name+'\n');
  }
  const seed=path.join(tmp,'seed'),binary=path.join(tmp,'binary'),cache=path.join(tmp,'cache'),out=path.join(tmp,'out');
  fs.writeFileSync(seed,'seed fixture');fs.writeFileSync(binary,'payload fixture: never executed');
  const env={platform:'darwin',arch:'arm64',os:'fixture-build',clang:'fixture-clang',sdk:'fixture-sdk'};
  const expected=snapshot(fixture,seed,env);
  seal(expected,cache,binary);restore(expected,cache,out);assert.equal(fs.readFileSync(out,'utf8'),fs.readFileSync(binary,'utf8'));
  fs.appendFileSync(path.join(cache,'madaros'),'corrupt');assert.throws(()=>restore(expected,cache,out),/digest mismatch/);seal(expected,cache,binary);
  for(const name of ['self-hosted/compiler/main.sio','stdlib/core.sio','.github/workflows/ci.yml','scripts/ci/build_madaros_macos_arm64.sh','scripts/ci/native_build_profile.cpp','scripts/ci/macos_native_cache.mjs','tests/run-pass/a64_macos_mapped_stack.sio']){
   const p=path.join(fixture,name),old=fs.readFileSync(p);fs.appendFileSync(p,'changed');
   const changed=snapshot(fixture,seed,env);assert.notEqual(changed.key,expected.key);assert.throws(()=>restore(changed,cache,out),/key differs/);fs.writeFileSync(p,old);
  }
  fs.appendFileSync(seed,'changed');assert.throws(()=>restore(snapshot(fixture,seed,env),cache,out),/key differs/);fs.writeFileSync(seed,'seed fixture');
  assert.throws(()=>restore(snapshot(fixture,seed,{...env,sdk:'changed SDK'}),cache,out),/key differs/);
  fs.writeFileSync(path.join(fixture,'.gitignore'),'new.sio\n');
  const extra=path.join(fixture,'self-hosted/compiler/new.sio');fs.writeFileSync(extra,'new module');assert.notEqual(snapshot(fixture,seed,env).key,expected.key);fs.unlinkSync(extra);
  const receipt=path.join(cache,'receipt.json');fs.unlinkSync(receipt);assert.throws(()=>restore(expected,cache,out));seal(expected,cache,binary);
  const r=JSON.parse(fs.readFileSync(receipt));r.inputs.environment.os='wrong';fs.writeFileSync(receipt,JSON.stringify(r));assert.throws(()=>restore(expected,cache,out),/recipe differs/);
  const profiler=path.join(tmp,'profile-build');
  execFileSync(process.env.CXX||'c++',['-std=c++23','-O2','-Wall','-Wextra','-Werror','-pthread',path.join(root,'scripts/ci/native_build_profile.cpp'),'-o',profiler]);
  const logs=path.join(tmp,'profile');
  const program=path.join(tmp,'program');
  fs.writeFileSync(program,'#!/bin/sh\nprintf "source: test\\n"\nprintf "total source: test\\n"\nprintf "tokens: 1\\n"\nprintf "arm64_compile: test\\n"\nprintf "macho_arm64: test\\n"\nprintf "stderr preserved\\n" >&2\nexit 7\n',{mode:0o755});
  let failure;try{execFileSync(profiler,[logs,program],{encoding:'utf8',stdio:'pipe'});}catch(e){failure=e;}
  assert(failure);assert.equal(failure.status,7);assert(failure.stdout.includes('stderr preserved'));
  for(const event of ['start','source_read','imports_and_prelude_done','lex_and_resolution_done','arm64_compile_done','macho_writer','exit'])assert(fs.readFileSync(path.join(logs,'events.tsv'),'utf8').includes(event+'\t'));
  assert.throws(()=>execFileSync(profiler,[path.join(tmp,'missing'),path.join(tmp,'absent')],{stdio:'pipe'}),e=>e.status===127);
  const success=execFileSync(profiler,[path.join(tmp,'ok'),'/bin/echo','fresh execution'],{encoding:'utf8'});assert.equal(success,'fresh execution\n');
  console.log('NATIVE_PROFILE_REAL_EXECUTION_AND_FAILURE_PROPAGATION_PASS');
  console.log('MACOS_NATIVE_REAL_DIGEST_AND_INPUT_REFUSALS_PASS');
 }finally{fs.rmSync(tmp,{recursive:true,force:true});}
 console.log('MACOS_NATIVE_ACCEPTANCE_BYPASS_MUTATIONS_REJECTED');
}
console.log('MACOS_NATIVE_EXACT_CACHE_CONTRACT_PASS');
