
const fs=require('fs'),path=require('path'),crypto=require('crypto'),zlib=require('zlib'),assert=require('assert/strict');
const b='/scratch/sounio-stdlib-wave3-20261027',r=b+'/services/bidi-canonical-readback66',x=r+'/extracted',d=process.argv[2],out=d+'/compact',base=x+'/root-bidi-inputs07/results',cap=base+'/capture/tests/stdlib/wave3-bidi/conformance.sio',fixture=cap+'/runtime-fixture';
fs.mkdirSync(out);const rows=[],hash=v=>crypto.createHash('sha256').update(v).digest('hex');
function save(src,dest,kind,forceGzip=false){const data=fs.readFileSync(src),gzip=forceGzip||data.length>200000,p=dest+(gzip?'.gz':'');fs.mkdirSync(path.dirname(out+'/'+p),{recursive:true});const stored=gzip?zlib.gzipSync(data,{level:9}):data;fs.writeFileSync(out+'/'+p,stored);rows.push({file:p,source:src,provenance:kind,source_bytes:data.length,source_sha256:hash(data),compression:gzip?'gzip':'none',stored_bytes:stored.length,stored_sha256:hash(stored)});}
for(const n of ['RESULT.json','SOURCE_SCOPE','CANONICAL.resource','CANONICAL_STATUS','STATUS','canonical.stdout','canonical.stderr','RAW.sha256','RAW.before.verify','RAW.final.verify','SOURCE.sha256','SOURCE.final.verify','TOOLS.sha256','TOOLS.final.verify','SDK.sha256','SDK.final.verify','EXECUTABLES.sha256','EXECUTABLES.final.verify','REUSED_CONTROL_PHASES.tsv','CONTROL_REUSE_SCOPE','REUSED_PROTOCOL_CONTROLS.stdout','PROTOCOL_REUSE_SCOPE'])save(base+'/'+n,'canonical/'+n,'archive-member');
save(base+'/binding/COMPILER_BINDING.json','canonical/COMPILER_BINDING.json','archive-member');
for(const n of ['capture.json','complete.json'])save(base+'/capture/'+n,'capture/'+n,'archive-member',true);
for(const phase of ['check','compile','native'])for(const n of ['start.json','result.json','combined.output'])save(cap+'/'+phase+'/'+n,'capture/'+phase+'/'+n,'archive-member');
for(const n of ['ARTIFACTS.json','INPUTS.json','INPUTS.final.json','ELFS.json','RESULTS.json','CONTROLS.json','DERIVED.json','FIXTURES.json','STATUS'])save(fixture+'/'+n,'fixture/'+n,'archive-member');
save(fixture+'/corpus/COUNTS.json','fixture/COUNTS.json','archive-member');
for(const n of ['READBACK_SUMMARY.json','PHASES.tsv','ACCOUNTING','STATUS','ARCHIVE.expected.sha256','ARCHIVE.before.verify','ARCHIVE.final.verify','INVENTORY_STATUS','TOOLS.sha256','TOOLS.final.verify','metadata.stdout','metadata.stderr','EXTRACT.list'])save(r+'/'+n,'readback/'+n,'readback-job-19728');
save(r+'/input/readback.cjs','readback/readback.cjs','readback-job-19728');
for(const n of ['STATUS','ARCHIVE.before.verify','INVENTORY_STATUS','metadata.stdout','metadata.stderr','slurm-19727.err'])if(fs.existsSync(b+'/services/bidi-canonical-readback65/'+n))save(b+'/services/bidi-canonical-readback65/'+n,'failed-readback-19727/'+n,'readback-job-19727');
const defs=[
 ['19616','01',['STATUS','FIXTURE_STATUS','fixture.stderr','fixture.stdout','FIXTURE.resource']],
 ['19654','03',['STATUS','SOURCE_SCOPE']],
 ['19668','04',['STATUS','SOURCE_SCOPE','CONTROL_PHASES.tsv','test_emit_stdlib_e2e_result.mjs.stdout']],
 ['19670','05',['STATUS','SOURCE_SCOPE','CONTROL_PHASES.tsv','protocol-controls.stdout']],
 ['19672','06',['STATUS','SOURCE_SCOPE','runtime.build.stderr']]
];
for(const [job,n,names]of defs)for(const file of names)save(x+'/root-bidi-inputs'+n+'/results/'+file,'failed-attempts/'+job+'/'+file,'archive-member');
for(const file of ['root-bidi-inputs01/slurm-19616.log','root-bidi-inputs03/slurm-19654.log','bidi-canonical-19668.log','bidi-canonical-19670.log','bidi-canonical-19672.log','root-bidi-inputs07.slurm.out'])save(d+'/extra/'+file,'archived-job-logs/'+path.basename(file),'archive-member');
for(const file of fs.readdirSync(x+'/root-bidi-recipe-copies').sort())save(x+'/root-bidi-recipe-copies/'+file,'recipes/session-copies/'+file,'archived-session-copy-not-sbatch-spool');
save(x+'/root-bidi-inputs07.sh','recipes/session-copies/root-bidi-inputs07.sh','archived-session-copy-not-sbatch-spool');
for(const n of ['STATUS','collect.stderr']){const p=n==='STATUS'?b+'/services/bidi-canonical-receipt67/metadata/STATUS':b+'/services/bidi-canonical-receipt67/metadata/collect.stderr';if(fs.existsSync(p))save(p,'failed-compact-19730/'+n,'compact-job-19730');}
const o='/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/root/bidi-canonical-export07';
for(const n of fs.readdirSync(o).sort())if(n!=='bidi-complete-sources-data-elfs-tooling-attempts.tar.gz'&&n!=='ARCHIVE_CONTENTS.txt'&&fs.lstatSync(o+'/'+n).isFile())save(o+'/'+n,'export-sidecars/'+n,'export-sidecar');
const census=fs.readFileSync(r+'/ARCHIVE_CONTENTS.readback.txt');fs.writeFileSync(out+'/readback/ARCHIVE_CONTENTS.identity.json',JSON.stringify({file:'ARCHIVE_CONTENTS.readback.txt',bytes:census.length,sha256:hash(census)},null,2)+'\n');
rows.sort((a,b)=>a.file<b.file?-1:a.file>b.file?1:0);fs.writeFileSync(out+'/COMPACT_FILES.json',JSON.stringify({schema:'sounio.bidi.compact-receipt-files.v1',files:rows},null,2)+'\n');
for(const row of rows){const src=fs.readFileSync(row.source),stored=fs.readFileSync(out+'/'+row.file);assert.equal(hash(src),row.source_sha256);assert.equal(hash(stored),row.stored_sha256);assert.deepEqual(row.compression==='gzip'?zlib.gunzipSync(stored):stored,src);}
console.log('BIDI_COMPACT_RECEIPT_BYTES_VERIFIED files='+rows.length);
