#!/usr/bin/env node
// Fake-compiler metadata regressions; never invokes a Sounio compiler.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'sounio-e2e-diagnostics-'));
const root=path.join(tmp,'project');
for(const d of ['scripts/stdlib','scripts/lib','tests/stdlib/nested','stdlib']) fs.mkdirSync(path.join(root,d),{recursive:true});
for(const p of ['scripts/stdlib/run_stdlib_e2e.sh','scripts/stdlib/stdlib_e2e_diagnostics.mjs','scripts/stdlib/emit_stdlib_e2e_result.mjs','scripts/lib/resolve_souc.sh']) fs.copyFileSync(path.join(repo,p),path.join(root,p));
const fake=path.join(tmp,'fake-compiler');
fs.writeFileSync(fake,`#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$INVOCATIONS"
name="\${2##*/}"
if [[ "$1" == check ]]; then
  if [[ "$name" == b_late.sio && -n "$LATE_MUTATE_LOG" ]]; then printf "later-case-mutation\\n" >> "$LATE_MUTATE_LOG"; fi
  if [[ -n "$SABOTAGE_CAPTURE_LOG" ]]; then rm -- "$SABOTAGE_CAPTURE_LOG"; if [[ -n "$SABOTAGE_SYMLINK_TARGET" ]]; then ln -s -- "$SABOTAGE_SYMLINK_TARGET" "$SABOTAGE_CAPTURE_LOG"; fi; fi
  if [[ "$name" == banner.sio ]]; then
    for ((i=1;i<=12;i++)); do echo "banner $i"; done
    echo 'error[E009]: genuine diagnostic below excerpt' >&2
    exit 1
  fi
  if [[ "$name" == exit65.sio ]]; then echo genuine65; exit 65; fi
  exit 0
fi
case "$name" in
  rc1.sio) echo actual-run-error; exit 1;;
  rc124.sio) echo partial-before-timeout; exit 124;;
  rc139.sio) echo signal-compatible-status; exit 139;;
  missing.sio) echo other;;
  bytes.sio) printf 'before\\0after\\n\\n\\n';;
  *) echo expected;;
esac
`);fs.chmodSync(fake,0o700);
const cases={banner:'//@ check-only\n',exit65:'//@ check-only\n',ok:'//@ run-pass\n//@ expect-stdout: expected\n',rc1:'//@ run-pass\n',rc124:'//@ run-pass\n',rc139:'//@ run-pass\n',missing:'//@ run-pass\n//@ expect-stdout: absent\n',bytes:'//@ run-pass\n',ignored:'//@ ignore known\n//@ ignore-owner: test\n//@ ignore-unblock: fixed\n',plain:''};
for(const [n,h] of Object.entries(cases))fs.writeFileSync(path.join(root,'tests/stdlib/nested',n+'.sio'),h+'fn main() {}\n');
const inv=path.join(tmp,'invocations');
const env={...process.env,SOUC_BIN:fake,INVOCATIONS:inv,SOUNIO_STDLIB_PATH:path.join(root,'stdlib'),STDLIB_E2E_RUN_TIMEOUT_SEC:'30',SABOTAGE_CAPTURE_LOG:'',SABOTAGE_SYMLINK_TARGET:'',LATE_MUTATE_LOG:''};
let n=0;function check(name,fn){fn();console.log(`PASS ${++n} ${name}`);}
function run(extra=[],envExtra={}){return spawnSync('bash',[path.join(root,'scripts/stdlib/run_stdlib_e2e.sh'),'--verbose',...extra],{env:{...env,...envExtra},encoding:'utf8'});}
function json(p){return JSON.parse(fs.readFileSync(p,'utf8'));}
function canonical(x){delete x.generated_at_utc;return x;}
const baseJson=path.join(tmp,'default.json'),capJson=path.join(tmp,'capture.json'),out=path.join(tmp,'capture');
const base=run(['--json-out',baseJson]),cap=run(['--json-out',capJson,'--artifacts-dir',out]);
check('default versus capture stdout and exit identical',()=>{assert.equal(base.status,1);assert.equal(cap.status,base.status);assert.equal(cap.stdout,base.stdout);assert.equal(cap.stderr.replace(/: line [0-9]+:/g,': line <location>:'),base.stderr.replace(/: line [0-9]+:/g,': line <location>:')); });
check('schema v1 rows and totals unchanged except UTC',()=>assert.deepEqual(canonical(json(capJson)),canonical(json(baseJson))));
const log=(name,phase='run')=>path.join(out,'tests/stdlib/nested',name+'.sio',phase);
check('full diagnostic beyond six lines retained',()=>{assert.match(fs.readFileSync(path.join(log('banner','check'),'combined.output'),'utf8'),/error\[E009\]/);assert(!JSON.stringify(json(capJson)).includes('genuine diagnostic'));});
check('raw NUL and all final newlines retained',()=>assert.deepEqual(fs.readFileSync(path.join(log('bytes'),'combined.output')),Buffer.from('before\0after\n\n\n')));
check('check65 is genuine science status not metadata refusal',()=>assert.equal(json(path.join(log('exit65','check'),'result.json')).exit_code,65));
check('run statuses 0 1 124 139 independently retained',()=>{for(const [name,rc]of [['ok',0],['rc1',1],['rc124',124],['rc139',139]])assert.equal(json(path.join(log(name),'result.json')).exit_code,rc);});
check('combined command includes unchanged timeout and wrapper identity',()=>{const r=json(path.join(log('ok'),'result.json'));assert.deepEqual(r.command,['timeout','30s',fake,'run',path.join(root,'tests/stdlib/nested/ok.sio')]);assert.match(json(path.join(out,'capture.json')).identity_scope,/not an execve witness/);});
check('skip and unannotated cases never execute',()=>{const commands=fs.readFileSync(inv,'utf8');assert(!commands.includes('ignored.sio'));assert(!commands.includes('plain.sio'));assert(!fs.existsSync(log('ignored')));});
check('all identities verified at finalization',()=>assert.equal(json(path.join(out,'complete.json')).source_and_caller_identity_unchanged,true));
function refuse(p,label){const before=fs.readFileSync(inv,'utf8');const r=run(['--artifacts-dir',p]);assert.equal(r.status,65,label);assert.match(r.stderr,/E2E artifact capture/);assert.equal(fs.readFileSync(inv,'utf8'),before);}
check('existing directory refused before execution',()=>refuse(out,'existing'));
check('relative directory refused before execution',()=>refuse('relative','relative'));
check('missing parent refused before execution',()=>refuse(path.join(tmp,'absent','capture'),'missing parent'));
check('input tree overlap refused before execution',()=>refuse(path.join(root,'new-capture'),'inside repo'));
check('dotdot path refused before execution',()=>refuse(tmp+'/no/../capture2','dotdot'));
const symlink=path.join(tmp,'link');fs.symlinkSync(tmp,symlink);
check('symlink ancestor refused before execution',()=>refuse(path.join(symlink,'new-capture'),'symlink'));
const dangling=path.join(tmp,'dangling');fs.symlinkSync(path.join(tmp,'none'),dangling);
check('dangling target refused before execution',()=>refuse(dangling,'dangling'));
check('filter selection unchanged',()=>{const a=run(['--filter','ok.sio']),b=run(['--filter','ok.sio','--artifacts-dir',path.join(tmp,'filtered')]);assert.equal(a.status,0);assert.equal(a.stdout,b.stdout);assert.equal(a.stderr,b.stderr);assert.equal(b.status,a.status);});
check('check-only fallback unchanged',()=>{const a=run(['--filter','rc1.sio'],{STDLIB_E2E_RUN_ALL_RUN_PASS:'0'}),b=run(['--filter','rc1.sio','--artifacts-dir',path.join(tmp,'fallback')],{STDLIB_E2E_RUN_ALL_RUN_PASS:'0'});assert.equal(a.status,0);assert.equal(a.stdout,b.stdout);assert.equal(a.stderr,b.stderr);assert.equal(b.status,a.status);});
// Direct helper controls model failures after command launch without changing production inputs.
function helper(...a){return spawnSync(process.execPath,[path.join(root,'scripts/stdlib/stdlib_e2e_diagnostics.mjs'),...a],{encoding:'utf8'});}
check('missing completed receipt is a metadata failure',()=>{assert.equal(helper('verify',out,'tests/stdlib/nested/ok.sio','run','1').status,65);});
check('absent phase receipt is metadata refusal',()=>assert.equal(helper('verify',out,'tests/stdlib/nested/ignored.sio','check','0').status,65));
check('unwritable log parent refuses before compiler execution',()=>{const locked=path.join(tmp,'locked');fs.mkdirSync(locked);fs.chmodSync(locked,0o500);try{refuse(path.join(locked,'capture'),'unwritable parent');}finally{fs.chmodSync(locked,0o700);}});
check('lost raw capture fails explicitly65 and emits no fresh results',()=>{const broken=path.join(tmp,'lost-raw'),result=path.join(tmp,'lost-raw.json');const r=run(['--filter','ok.sio','--json-out',result,'--artifacts-dir',broken],{SABOTAGE_CAPTURE_LOG:path.join(broken,'tests/stdlib/nested/ok.sio/check/combined.output')});assert.equal(r.status,65);assert.match(r.stderr,/E2E artifact capture/);assert(!fs.existsSync(result));assert(!fs.existsSync(path.join(broken,'complete.json')));});
check('raw log symlink substitution is metadata refusal',()=>{const broken=path.join(tmp,'linked-raw');const r=run(['--filter','ok.sio','--artifacts-dir',broken],{SABOTAGE_CAPTURE_LOG:path.join(broken,'tests/stdlib/nested/ok.sio/check/combined.output'),SABOTAGE_SYMLINK_TARGET:path.join(root,'tests/stdlib/nested/ok.sio')});assert.equal(r.status,65);assert.match(r.stderr,/not regular/);assert(!fs.existsSync(path.join(broken,'complete.json')));});
check('intermediate symlink refuses before outside-directory creation',()=>{const attack=path.join(tmp,'intermediate-link'),outside=path.join(tmp,'outside');fs.mkdirSync(outside);assert.equal(helper('init',attack,root,fake,path.join(root,'stdlib')).status,0);fs.symlinkSync(outside,path.join(attack,'tests'));const r=helper('begin',attack,'tests/stdlib/nested/ok.sio','check',fake,'check',path.join(root,'tests/stdlib/nested/ok.sio'));assert.equal(r.status,65);assert.match(r.stderr,/unsafe directory/);assert.deepEqual(fs.readdirSync(outside),[]);});
check('changed raw bytes refused',()=>{fs.appendFileSync(path.join(log('ok'),'combined.output'),'tampered');assert.equal(helper('verify',out,'tests/stdlib/nested/ok.sio','run','0').status,65);});
check('changed source refused at footer',()=>{fs.appendFileSync(path.join(root,'tests/stdlib/nested/ok.sio'),'// changed\n');assert.equal(helper('finalize',out).status,65);});
check('zero selected cases explicitly completes zero capture phases',()=>{const empty=path.join(tmp,'zero-selected');const r=run(['--filter','no-selected-case','--artifacts-dir',empty]);assert.equal(r.status,0);assert.equal(json(path.join(empty,'complete.json')).captured_phases,0);});
check('later selected case mutation of earlier raw output fails terminal65',()=>{const cases=path.join(root,'tests/stdlib/terminal');fs.mkdirSync(cases);for(const name of ['a_first.sio','b_late.sio'])fs.writeFileSync(path.join(cases,name),'//@ check-only\nfn main() {}\n');const capture=path.join(tmp,'terminal-rehash'),result=path.join(tmp,'terminal-rehash.json'),earlier=path.join(capture,'tests/stdlib/terminal/a_first.sio/check/combined.output');const r=run(['--filter','terminal/','--json-out',result,'--artifacts-dir',capture],{LATE_MUTATE_LOG:earlier});assert.equal(r.status,65);assert.match(r.stderr,/identity changed/);assert.match(fs.readFileSync(earlier,'utf8'),/later-case-mutation/);assert(fs.existsSync(path.join(capture,'tests/stdlib/terminal/b_late.sio/check/result.json')));assert(!fs.existsSync(result));assert(!fs.existsSync(path.join(capture,'complete.json')));fs.writeFileSync(path.join(tmp,'LATE_MUTATION_ACTUAL.json'),JSON.stringify({scope:'Operational fake later-selected-case mutation; no compiler/science execution',status:r.status,stdout:r.stdout,stderr:r.stderr,fresh_result_exists:fs.existsSync(result),complete_exists:fs.existsSync(path.join(capture,'complete.json')),earlier_receipt:json(path.join(capture,'tests/stdlib/terminal/a_first.sio/check/result.json')),later_receipt:json(path.join(capture,'tests/stdlib/terminal/b_late.sio/check/result.json')),earlier_raw_after_b64:fs.readFileSync(earlier).toString('base64')},null,2)+'\n');});
console.log(`PASS ${n} metadata controls; fixtures retained at ${tmp}`);
