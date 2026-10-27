#!/usr/bin/env node
// Operational, path-only fixture relocation. No numeric reference work.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
const [root,out]=process.argv.slice(2);
assert(root&&out&&path.isAbsolute(root)&&path.isAbsolute(out));
assert(/^[A-Za-z0-9_./-]+$/.test(out),'output path must be literal-safe');
const sha=s=>crypto.createHash('sha256').update(s).digest('hex');
const sourceDir=path.join(out,'src'),fileDir=path.join(out,'files');
fs.mkdirSync(sourceDir);fs.mkdirSync(fileDir);
const cases=[],global=new Map();
for(const name of ['ipc','codec','spec','ext','flight']){
 const original='tests/stdlib/arrow/test_arrow_'+name+'.sio';
 const source=fs.readFileSync(path.join(root,original),'utf8');
 const matches=[...source.matchAll(/"([/]tmp[/]sounio_(?:arrow|flight)_[A-Za-z0-9_.]+)"/g)];
 const mappings=[...new Set(matches.map(m=>m[1]))].map(from=>({from,to:path.join(fileDir,path.basename(from))}));
 for(const m of mappings){
  assert(!source.includes(m.to),'ambiguous pre-existing replacement');
  assert(!global.has(m.to),'cross-fixture output collision');
  global.set(m.to,original);
 }
 let derived=source;
 for(const m of mappings)derived=derived.split(JSON.stringify(m.from)).join(JSON.stringify(m.to));
 let reversed=derived;
 for(const m of mappings)reversed=reversed.split(JSON.stringify(m.to)).join(JSON.stringify(m.from));
 assert.equal(reversed,source,'reverse relocation differs from original');
 assert.equal((derived.match(/"([/]tmp[/]sounio_(?:arrow|flight)_[A-Za-z0-9_.]+)"/g)||[]).length,0);
 const target=path.join(sourceDir,path.basename(original));
 fs.writeFileSync(target,derived,{flag:'wx'});
 cases.push({name,original,original_sha256:sha(source),derived:target,derived_sha256:sha(derived),reverse_equal:true,mappings});
}
assert(global.size===20,'legacy fixture output set changed; audit before updating');
fs.writeFileSync(path.join(out,'FIXTURE_RELOCATION.json'),JSON.stringify({schema:'sounio.arrow.path-relocation.v1',cases,outputs:[...global.keys()].sort()},null,2)+'\n',{flag:'wx'});
console.log('ARROW_FIXTURE_RELOCATION_OK');

