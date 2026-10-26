import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
const archive=process.argv[2], checkout=process.argv[3], destination=process.argv[4];
assert(archive&&checkout&&destination,'usage: node verify_receipt.mjs ARCHIVE CHECKOUT REPORT');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
assert.equal(sha(fs.readFileSync(archive)),'fe978151463bb3aab84547cebf3394bc94b193cef077618897ab95495277bf69');
const listed=execFileSync('tar',['-tzf',archive],{encoding:'utf8'}).trim().split('\n');
for(const n of listed) assert(n.startsWith('text17651/')&&!n.split('/').includes('..')&&!n.startsWith('/'));
const tmp=fs.mkdtempSync('/tmp/SIO184-root-review-');
execFileSync('tar',['-xzf',archive,'-C',tmp]);
const root=path.join(tmp,'text17651'), r=p=>fs.readFileSync(path.join(root,p),'utf8');
const files=r('FILES.sha256').trim().split('\n');
for(const line of files) {
 const m=/^([a-f0-9]{64})  (.+)$/.exec(line);assert(m);
 const n=m[2];assert(!path.isAbsolute(n)&&!n.split('/').includes('..'));
 const p=path.join(root,n),s=fs.lstatSync(p);assert(s.isFile()&&!s.isSymbolicLink());
 assert.equal(sha(fs.readFileSync(p)),m[1],n);
}
assert.equal(files.length,444);
const header='variant\tengine\tcase\tphase\trc\texpected';
const ledger=r('results/RESULTS.tsv').trim().split('\n');assert.equal(ledger.shift(),header);
const rows=ledger.map(line=>{const a=line.split('\t');assert.equal(a.length,6);return {variant:a[0],engine:a[1],case:a[2],phase:a[3],rc:a[4],expected:a[5]};});
assert.equal(rows.length,41);assert.equal(new Set(ledger.map(x=>x.split('\t').slice(0,4).join('.'))).size,41);
for(const x of rows) {
 const key=[x.variant,x.engine,x.case,x.phase].join('.');
 assert.equal(r('results/'+key+'.status').trim(),x.rc,key);
}
const mismatch=rows.filter(x=>x.rc!==x.expected);
assert.deepEqual(mismatch.map(x=>[x.variant,x.engine,x.case,x.phase,x.rc,x.expected]),[
 ['before','lean_single','consumer','check','0','1'],
 ['before','lean_single','consumer','compile','0','1'],
 ['before','lean_single','library','check','1','0'],
 ['after','lean_single','library','check','1','0']]);
for(const engine of ['madaros','lean_single']) {
 for(const c of ['consumer','error','generated'])for(const phase of ['check','compile','native']) {
  assert.equal(rows.find(x=>x.variant==='after'&&x.engine===engine&&x.case===c&&x.phase===phase)?.rc,'0');
 }
 for(const [c,rc]of [['error','2'],['generated','201']])assert.equal(rows.find(x=>x.variant==='before'&&x.engine===engine&&x.case===c&&x.phase==='native')?.rc,rc);
 assert.match(r('results/after.'+engine+'.error.native.stdout'),/ANALYSIS_ROMBERG_ERROR_ESTIMATE_OK/);
 assert.match(r('results/after.'+engine+'.generated.native.stdout'),/ANALYSIS_CPP23_ROMBERG_CONCORDANCE_OK/);
 for(const v of ['before','after']) if(engine==='lean_single') {
  const text=r('results/'+v+'.'+engine+'.library.check.stdout')+r('results/'+v+'.'+engine+'.library.check.stderr');
  assert.match(text,/E221/);assert.match(text,/no main/);
 }
}
assert.match(r('host/ATTEMPT/sacct.terminal'),/^17651\|FAILED\|1:0\|00:00:15\|2\|4G/m);
assert.equal(r('controller-results/focused.status').trim(),'1');
assert.equal(r('results/python.runtime.nul').length,0);
const before=JSON.parse(r('PACKET_before.census.json')),after=JSON.parse(r('PACKET_after.census.json'));
assert.equal(before.length,2398);assert.equal(after.length,2398);
assert.deepEqual(before.map(x=>x.path),after.map(x=>x.path));
const changed=before.filter((x,i)=>JSON.stringify(x)!==JSON.stringify(after[i])).map(x=>x.path);
assert.deepEqual(changed.sort(),['stdlib/analysis/lib.sio','tests/stdlib/analysis/test_analysis.sio']);
for(const name of [...changed,'tests/stdlib/analysis/test_romberg_error_estimate.sio']) {
 const record=after.find(x=>x.path===name);assert(record&&record.type==='regular');
 const s=fs.statSync(path.join(checkout,name));assert.equal(s.mode&0o777,record.mode);
 assert.equal(sha(fs.readFileSync(path.join(checkout,name))),record.sha256);
 if(changed.includes(name))assert.equal(sha(execFileSync('git',['-C',checkout,'show','b22d3484c95b826955b3e232310b41a6018be751:'+name])),before.find(x=>x.path===name).sha256);
}
assert.equal(sha(fs.readFileSync(path.join(checkout,'tests/reference/analysis_romberg_reference.cpp'))),sha(fs.readFileSync(path.join(root,'PACKET_analysis_romberg_reference.cpp'))));
const verifications=['results','controller-results'].flatMap(d=>fs.readdirSync(path.join(root,d)).filter(x=>x.endsWith('.verify')).map(n=>d+'/'+n));
let verifiedLines=0;
for(const name of verifications) {const lines=r(name).trim().split('\n').filter(Boolean);for(const line of lines)assert(/: OK$|^PASS\b/.test(line),name+':'+line);verifiedLines+=lines.length;}
assert.equal(verifications.length,51);
const report={review:'Independent root metadata audit, no numerical recalculation',job:17651,state:'FAILED',exit:'1:0',elapsed_seconds:15,requested_cpus:1,allocated_logical_cpus:2,archive_sha256:sha(fs.readFileSync(archive)),archive_files_verified:files.length,phase_rows:rows.length,strict_mismatches:mismatch,after_consumer_regression_reference:'check/compile/native0 both engines; before native2 and201 reproduced',source_catalog_entries_per_variant:2398,changed_paths:changed,source_matches_current_checkout:true,verification_manifests:verifications.length,verification_lines:verifiedLines,python_runtime_bytes:0,limits:['Not a full corpus pass','Seed direct-library check E221 and accepted baseline import remain failed expectations','Heuristic last-change error estimate, not a rigorous bound','Canonical seed receipt does not establish current compiler fixed point']};
fs.writeFileSync(destination,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
