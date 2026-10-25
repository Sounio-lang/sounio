#!/usr/bin/env node
// Metadata-only compatibility fixtures; no compiler or numerical simulation.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {asciiJSON, digestFile, receiptInternalErrors, checkReceiptAgainstTree, parseGensTSV, buildFixedPoint} from '../dev/write_seed_receipt.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const writer=path.join(root,'scripts/dev/write_seed_receipt.mjs');
const historical=path.join(root,'bin/souc-lean-single-x86_64.SeedReceipt.json');
const historicText=fs.readFileSync(historical,'utf8'), historic=JSON.parse(historicText);
assert.deepEqual(receiptInternalErrors(historic),[]);
// Existing receipt remains exactly compatible with the prior ASCII JSON format.
assert.equal(asciiJSON(historic),historicText);
// The previous ensure_ascii format also escapes ASCII DEL in paths/labels.
assert.equal(asciiJSON({path: 'seed\x7fvalue'}), '{\n  \"path\": \"seed\\u007fvalue\"\n}\n');
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'seed-metadata-test-'));
try {
 const src=path.join(tmp,'source.txt'), inp=path.join(tmp,'input.bin'), out=path.join(tmp,'output.bin');
 fs.writeFileSync(src,'metadata source fixture\n');fs.writeFileSync(inp,'metadata input fixture\n');fs.writeFileSync(out,'metadata output fixture\n');
 const s=digestFile(src), i=digestFile(inp), o=digestFile(out);
 const generations=[{gen:'g0',gen_index:0,md5:i.md5,sha256:i.sha256,path:inp},...['g1','g2'].map((gen,j)=>({gen,gen_index:j+1,md5:o.md5,sha256:o.sha256,path:out}))];
 const valid={schema:'sounio.SeedReceipt',schema_version:1,receipt_utc:'1970-01-01T00:00:00Z',source:s,input_seed:i,generations,fixed_point:buildFixedPoint(generations,1),output_seed:o,environment:{placement:'check-only'},checks:{},limits:{provenance_note:'metadata fixture'}};
 assert.deepEqual(checkReceiptAgainstTree(valid,src,out),[]);
 const write=(name,v)=>{const p=path.join(tmp,name+'.json');fs.writeFileSync(p,asciiJSON(v));return p;};
 const cli=(p,...rest)=>spawnSync(process.execPath,[writer,'--check-against-tree',p,'--source',src,'--seed-elf',out,...rest],{encoding:'utf8'});
 let refused=0,accepted=1;
 const reject=(name,change,diagnostic)=>{const v=structuredClone(valid);change(v);const p=write(name,v),run=cli(p);assert.equal(run.status,1,name);assert.ok(run.stderr.includes(diagnostic),name+': '+run.stderr);refused++;};
 reject('wrong-source-sha',v=>v.source.sha256='a'.repeat(64),'PROVENANCE FAIL: receipt source.sha256');
 reject('wrong-output-sha',v=>v.output_seed.sha256='b'.repeat(64),'PROVENANCE FAIL: receipt output_seed.sha256');
 reject('wrong-fp-md5',v=>v.fixed_point.gk_md5='c'.repeat(32),'md5 mismatch by eye');
 reject('wrong-fp-sha',v=>v.fixed_point.gk_plus1_sha256='d'.repeat(64),'sha256 mismatch by eye');
 reject('unverified',v=>v.fixed_point.verified=false,'fixed_point.verified is not true');
 reject('missing-fixed-point',v=>delete v.fixed_point,'missing fixed_point object');
 reject('short-generations',v=>v.generations=[],'generations[] must list at least');
 reject('generation-md5-absent',v=>v.generations.forEach(g=>g.md5='e'.repeat(32)),'not present in generations[]');
 reject('missing-source',v=>v.source={},'source missing sha256/md5');
 reject('missing-output',v=>v.output_seed={},'output_seed missing sha256/md5');
 reject('missing-input',v=>v.input_seed={},'input_seed missing sha256/md5');
 reject('wrong-schema',v=>v.schema='other.Schema','schema must be sounio.SeedReceipt');
 reject('md5-fallback-wrong-source',v=>{delete v.source.sha256;v.source.md5='a'.repeat(32);},'PROVENANCE FAIL: receipt source.md5');
 reject('md5-fallback-wrong-output',v=>{delete v.output_seed.sha256;v.output_seed.md5='b'.repeat(32);},'PROVENANCE FAIL: receipt output_seed.md5');
 for(const [name,change] of [
  ['schema-optional-legacy',v=>delete v.schema],
  ['md5-fallback-legacy',v=>{delete v.source.sha256;delete v.output_seed.sha256;}],
  ['sha-precedence-legacy',v=>{v.source.md5='a'.repeat(32);v.output_seed.md5='b'.repeat(32);}],
  ['case-insensitive-legacy',v=>{for(const obj of [v.source,v.output_seed,v.fixed_point,...v.generations])for(const k of Object.keys(obj))if(k.includes('md5')||k.includes('sha256'))if(typeof obj[k]==='string')obj[k]=obj[k].toUpperCase();}],
  // Old checker judges actual hashes/verified, not these descriptive booleans.
  ['descriptive-flags-legacy',v=>{v.fixed_point.md5_equal=false;v.fixed_point.sha256_equal=false;}]
 ]){const v=structuredClone(valid);change(v);const run=cli(write(name,v));assert.equal(run.status,0,name+': '+run.stderr);accepted++;}
 const tsv=path.join(tmp,'gens.tsv');
 fs.writeFileSync(tsv,`# unsorted labels and uppercase digests\n2\t${o.md5.toUpperCase()}\t${o.sha256.toUpperCase()}\t${out}\ng0\t${i.md5}\t${i.sha256}\t${inp}\ng1\t${o.md5}\t${o.sha256}\t${out}\n`);
 assert.deepEqual(parseGensTSV(tsv),generations);
 const emit=spawnSync(process.execPath,[writer,'--out-dir',path.join(tmp,'emitted'),'--gens-tsv',tsv,'--source',src,'--input-seed',inp,'--output-seed',out,'--settle-k','1','--placement','check-only','--hostname','metadata-host','--git-commit','metadata-commit','--git-branch','metadata-branch'],{encoding:'utf8'});
 assert.equal(emit.status,0,emit.stderr);
 const emittedText=fs.readFileSync(path.join(tmp,'emitted/SeedReceipt.latest.json'),'utf8'),emitted=JSON.parse(emittedText);
 assert.deepEqual(Object.keys(emitted),['schema','schema_version','receipt_utc','git_commit','git_branch','source','input_seed','generations','fixed_point','output_seed','environment','checks','limits']);
 assert.deepEqual(emitted.source,s);assert.deepEqual(emitted.input_seed,{...i,role:'g0_start_of_chain'});assert.deepEqual(emitted.output_seed,o);assert.deepEqual(emitted.generations,generations);assert.deepEqual(emitted.fixed_point,valid.fixed_point);
 assert.ok(emittedText.includes('\\u2014'));assert.equal(asciiJSON(emitted),emittedText);
 assert.deepEqual(checkReceiptAgainstTree(emitted,src,out),[]);
 const emittedTxt=fs.readFileSync(path.join(tmp,'emitted/SeedReceipt.latest.txt'),'utf8');
 assert.ok(emittedTxt.includes(`gk_md5:       ${o.md5}\ngk_plus1_md5: ${o.md5}\nmd5_equal: true`));
 const malformed=path.join(tmp,'malformed.tsv');fs.writeFileSync(malformed,`g1x\t${o.md5}\t${o.sha256}\n`);assert.throws(()=>parseGensTSV(malformed),/bad gen/);
 assert.throws(()=>buildFixedPoint(generations,3),/requires g3 and g4/);
 const unequal=structuredClone(generations);unequal[2].sha256='f'.repeat(64);assert.equal(buildFixedPoint(unequal,1).verified,false);
 const badTsv=path.join(tmp,'unsettled.tsv');fs.writeFileSync(badTsv,unequal.map(g=>`${g.gen}\t${g.md5}\t${g.sha256}\t${g.path}`).join('\n'));
 const badEmit=spawnSync(process.execPath,[writer,'--out-dir',path.join(tmp,'refused'),'--gens-tsv',badTsv,'--source',src,'--input-seed',inp,'--output-seed',out,'--settle-k','1'],{encoding:'utf8'});
 assert.equal(badEmit.status,1);assert.ok(badEmit.stderr.includes('fixed_point not verified'));assert.equal(fs.existsSync(path.join(tmp,'refused')),false);
 console.log(`SEED_RECEIPT_METADATA_TEST_PASS accepted_legacy_cases=${accepted} rejected_provenance_cases=${refused} original_receipt_ASCII_roundtrip=byte-identical emitter_schema=preserved unsettled_writer=refused`);
} finally {fs.rmSync(tmp,{recursive:true,force:true});}
