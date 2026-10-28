#!/usr/bin/env node
// Text-only materialization: each algorithm body remains byte-for-byte original.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
assert.match(process.env.SLURM_JOB_ID??'',/^[1-9][0-9]*$/,'scheduled Slurm context');
assert.equal(process.argv.length,4,'usage: materialize ROOT FRESH_OUTPUT_DIRECTORY');
const root=fs.realpathSync(process.argv[2]),out=path.resolve(process.argv[3]);
assert(!fs.existsSync(out),'fresh output required');
const sha=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const sources=[
 ['unicode','tests/stdlib/wave3-core/unicode_conformance.sio'],
 ['locale','tools/unicode/probes/locale_conformance.sio'],
 ['metadata','tests/stdlib/wave3-core/unicode_locale_metadata.sio'],
 ['bidi','tests/stdlib/wave3-bidi/conformance.sio']
];
const entries=[];
fs.mkdirSync(out,{recursive:true});
for(const [family,file] of sources) {
 const original=fs.readFileSync(path.join(root,file),'utf8');
 assert.equal([...original.matchAll(/^fn main\(/gm)].length,1,'one exact original main');
 const anchor='fn main(',replacement='pub fn coherent_run_'+family+'(';
 const generated=original.replace(/^fn main\(/m,replacement);
 assert.equal(generated.replace(replacement,anchor),original,'reverse byte equality');
 const target='coherent_'+family+'.sio';
 fs.writeFileSync(path.join(out,target),generated,{flag:'wx'});
 entries.push({family,original:file,original_sha256:sha(original),generated:target,
               generated_sha256:sha(generated),only_main_name_visibility_changed:true});
}
const template='tests/stdlib/wave4-coherent/coherent.sio.in';
const main=fs.readFileSync(path.join(root,template));
fs.writeFileSync(path.join(out,'coherent.sio'),main,{flag:'wx'});
entries.push({original:template,original_sha256:sha(main),generated:'coherent.sio',
              generated_sha256:sha(main),byte_equal:true});
fs.writeFileSync(path.join(out,'MATERIALIZATION.json'),JSON.stringify({
 scope:'One ELF harness materialization, no conformance execution claim',
 argument_kinds:{1:'unicode_uca',2:'cldr_keys',3:'cldr_metadata',4:'bidi'},
 algorithm_bodies_unchanged:true,entries
},null,2)+'\n',{flag:'wx'});
console.log('COHERENT_TEXT_MATERIALIZED families=4 originals_preserved=true');
