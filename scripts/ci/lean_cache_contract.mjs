#!/usr/bin/env node
// Orchestration checks only: no proof statement or numerical implementation.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
const root=process.argv[2];
assert(root,'usage: lean_cache_contract.mjs REPO [--selftest]');
const w=fs.readFileSync(path.join(root,'.github/workflows/ci.yml'),'utf8');
function check(w) {
 const start=w.indexOf('  lean-proofs:\n'),end=w.indexOf('\n  website:',start);
 assert(start>=0&&end>start,'Lean job missing');
 const job=w.slice(start,end);
 const a=job.indexOf('      # Toolchains and build products'),b=job.indexOf('      - name: Install elan',a);
 assert(a>=0&&b>a,'cache boundary missing');
 const cache=job.slice(a,b);
 assert(!cache.includes('restore-keys:'),'cross-source fallback forbidden');
 assert(!cache.includes('if:'),'cache restore must not condition proof execution');
 assert(cache.includes('key: lean-elan-v2-')&&cache.includes("hashFiles('formal/lean-toolchain', 'formal/lean4/lean-toolchain')"),'both toolchain identities required');
 assert(cache.includes('path: ~/.elan\n'),'toolchain cache must be separate');
 assert(cache.includes('            formal/.lake\n            formal/lean4/.lake\n'),'both build trees required');
 const key=cache.split('\n').find(l=>l.includes('key: lean-build-v1-'));
 for(const input of ['formal/**','!formal/**/.lake/**','scripts/ci/*lean*.sh','scripts/ci/epistemic_measure_correspondence_gate.sh','.github/workflows/ci.yml']) assert(key?.includes("'"+input+"'"),'missing key input: '+input);
 assert(key.includes('runner.os')&&key.includes('runner.arch'),'platform identity required');
 const baseline=job.slice(0,a)+job.slice(b);
 assert.equal(crypto.createHash('sha256').update(baseline).digest('hex'),'732a5c0426b984fa5ed71a2003fb9bc88957646d588ae7cb259d598ec8b35e04','mandatory proof gates, builds, conditions, or audit commands changed');
 const fast=w.slice(w.indexOf('  fast-checks:\n'),w.indexOf('\n  contracts-core:'));
 assert(fast.includes('run: node scripts/ci/lean_cache_contract.mjs "$GITHUB_WORKSPACE" --selftest'),'contract must run in Fast Checks');
}
check(w);
if(process.argv.includes('--selftest')) {
 const mutations=[
  s=>s.replace("'formal/lean-toolchain', 'formal/lean4/lean-toolchain'","'formal/lean4/lean-toolchain'"),
  s=>s.replace("'formal/**', ",''),
  s=>s.replace('            formal/.lake\n',''),
  s=>s.replace('key: lean-build-v1-','restore-keys: lean-build-v1-\n          key: lean-build-v1-'),
  s=>s.replace('      - name: Build all default-target proof modules\n','      - name: Build all default-target proof modules\n        if: false\n'),
  s=>s.replace('run: bash scripts/ci/antigarbling_fusion_lean_gate.sh','run: echo skipped'),
  s=>s.replace('          lake build\n','          true\n'),
 ];
 for(const mutate of mutations) { const bad=mutate(w);assert.notEqual(bad,w,'mutation missed target');assert.throws(()=>check(bad),'unsafe mutation accepted'); }
 console.log('PASS: negative controls reject stale keys, omitted builds/audits and conditional proof skips');
}
console.log('PASS: exact-source Lean cache and unchanged mandatory proof gates');
