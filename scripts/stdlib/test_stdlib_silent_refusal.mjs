import fs from 'node:fs';import os from 'node:os';import path from 'node:path';import assert from 'node:assert/strict';import {spawnSync} from 'node:child_process';import {fileURLToPath} from 'node:url';
const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..'),tmp=fs.mkdtempSync(path.join(os.tmpdir(),'stdlib-silent-refusal-')),root=tmp+'/root';
for(const dir of ['scripts/stdlib','scripts/lib','tests/stdlib/negative','stdlib'])fs.mkdirSync(root+'/'+dir,{recursive:true});
for(const file of ['scripts/stdlib/run_stdlib_e2e.sh','scripts/stdlib/argv_inputs.mjs','scripts/stdlib/bounded_command.mjs','scripts/stdlib/stdlib_e2e_diagnostics.mjs','scripts/stdlib/emit_stdlib_e2e_result.mjs','scripts/lib/resolve_souc.sh'])fs.copyFileSync(repo+'/'+file,root+'/'+file);
const fake=tmp+'/compiler',payload=tmp+'/payload';
fs.writeFileSync(fake,'#!/bin/bash\nprintf "%s\\n" "$1" >> "$CALLS"\nif [[ "$1" == check ]];then [[ "$VARIANT" != check-fail ]] || exit 1;exit 0;fi\n[[ "$1" == compile ]] || exit 65\n[[ "$VARIANT" != compile-fail ]] || exit 1\ncp "$PAYLOAD" "$4"\nchmod +x "$4"\n');fs.chmodSync(fake,0o700);
fs.writeFileSync(payload,'#!/bin/bash\ncase "$VARIANT" in\nnewline)printf "\\n";;\nnul)printf "\\0";;\nmessage)echo refused;;\naccept)exit 0;;\ncrash)exit 139;;\ntimeout)exit 124;;\noutput-limit)exit 122;;\nesac\nexit 1\n');fs.chmodSync(payload,0o700);
const header='//@ run-fail\n//@ expect-silent-refusal\n',fixture=root+'/tests/stdlib/negative/probe.sio';let count=0;
try{for(const variant of ['empty','newline','nul','message','accept','crash','timeout','output-limit','check-fail','compile-fail','both-contracts','no-mode','conflicting-runpass','no-contract']){
 const pre=variant==='both-contracts'?header+'//@ expect-diagnostic: refused\n':variant==='no-mode'?'//@ expect-silent-refusal\n':variant==='conflicting-runpass'?header+'//@ run-pass\n':variant==='no-contract'?'//@ run-fail\n':header;
 fs.writeFileSync(fixture,pre+'fn main() {}\n');const json=tmp+'/'+variant+'.json',calls=tmp+'/'+variant+'.calls',capture=tmp+'/'+variant+'.capture';
 const r=spawnSync('bash',[root+'/scripts/stdlib/run_stdlib_e2e.sh','--filter','negative/','--json-out',json,'--artifacts-dir',capture],{env:{...process.env,SOUC_BIN:fake,PAYLOAD:payload,CALLS:calls,VARIANT:variant,SOUNIO_STDLIB_PATH:root+'/stdlib',STDLIB_E2E_RUN_ALL_RUN_PASS:'0'},encoding:'utf8',timeout:20000});
 assert.equal(r.status,variant==='empty'?0:1,variant+' '+r.stderr);const j=JSON.parse(fs.readFileSync(json));assert.equal(j.results.length,1);assert.equal(j.totals.skip,0);assert.equal(j.results[0].status,variant==='empty'?'pass':'fail');
 if(variant==='empty'){assert.equal(fs.readFileSync(calls,'utf8'),'check\ncompile\n');assert.equal(JSON.parse(fs.readFileSync(capture+'/tests/stdlib/negative/probe.sio/native/result.json')).exit_code,1);}
 if(['newline','nul','empty'].includes(variant)){const raw=fs.readFileSync(capture+'/tests/stdlib/negative/probe.sio/native/combined.output');assert.deepEqual(raw,variant==='newline'?Buffer.from('\n'):variant==='nul'?Buffer.from([0]):Buffer.alloc(0));}
 if(['both-contracts','no-mode','conflicting-runpass','no-contract'].includes(variant))assert(!fs.existsSync(calls),variant+' must refuse before compiler');
 console.log('PASS '+variant);count++;
}console.log('STDLIB_SILENT_REFUSAL_CONTROLS_OK cases='+count);}finally{fs.rmSync(tmp,{recursive:true,force:true});}
