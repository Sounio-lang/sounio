#!/usr/bin/env node
// Independent operational check: GNU tar reads the packet; original Git gzip
// streams must reconstruct byte-for-byte. Does not import the packet parser.
import fs from 'node:fs';
import path from 'node:path';
import { spawn, execFileSync } from 'node:child_process';
import { createGunzip } from 'node:zlib';
import { createHash } from 'node:crypto';
const need=(ok,message)=>{if(!ok)throw Error(message);};
const sha=b=>createHash('sha256').update(b).digest('hex');
const commit='3c7c8073411a7e7254d0d9424f59f8e49eb49180';
const originals=[
 ['native-cpp-text.tar.gz','0a80794d00ba129d90e9e6043aa19026a3aacfdccd10be374cd434edd4920074','aba131479c35d1f3ccf0733762c37fc4cd0cb2eb',6527120],
 ['final-17307-text.tar.gz','0681e5ada0de62957df0d1f63df529b295bb12e6bcc6dc56af5f8b161642e723','297a6cbacb48ec8d2e4c3fc6d82a2cd48fbce0c2',8828874]
];
const run=(cmd,args,encoding='utf8')=>execFileSync(cmd,args,{encoding,maxBuffer:32*1024*1024});
function safe(name) {need(typeof name==='string'&&!path.posix.isAbsolute(name)&&!/[\\\x00-\x1f\x7f]/.test(name)&&name.replace(/\/$/,'').split('/').every(s=>s&&s!=='.'&&s!=='..'),'unsafe alias');}
try {
 const [packet]=process.argv.slice(2);need(packet&&process.argv.length===3,'usage: verifier PACKET (from repository worktree)');
 const names=run('tar',['-tzf',packet]).trimEnd().split('\n');
 need(names[0]==='INDEX.json'&&new Set(names).size===names.length,'packet index/order/duplicate');
 need(names.slice(1).every(n=>/^objects\/[0-9a-f]{64}$/.test(n)),'unexpected packet path');
 const indexBytes=run('tar',['-xzOf',packet,'INDEX.json'],null),index=JSON.parse(indexBytes);
 need(index.schema==='sounio.sio173.evidence.v1'&&index.source_commit===commit,'packet schema/source');
 need(index.archives.length===2&&index.entries.filter(e=>e.type==='0').length===1142,'original census');
 const sizes=new Map(),aliases=new Set();
 for(const e of index.entries) {
  safe(e.name);const key=e.archive+'/'+e.name;need(!aliases.has(key),'duplicate alias');aliases.add(key);
  need(e.type==='0'||e.type==='5','entry type');
  if(e.type==='0') {
   need(Number.isSafeInteger(e.size)&&e.size>=0&&e.size<=16*1024*1024,'object size');
   if(sizes.has(e.sha256))need(sizes.get(e.sha256)===e.size,'object size disagreement');
   sizes.set(e.sha256,e.size);
  }
 }
 need(sizes.size===301&&names.length===302,'object census');
 const bytes=run('tar',['--wildcards','-xzOf',packet,'objects/*'],null),objects=new Map();let offset=0;
 for(const n of names.slice(1)) {
  const h=n.slice(8);need(sizes.has(h),'unreferenced object');const b=bytes.subarray(offset,offset+sizes.get(h));offset+=sizes.get(h);
  need(b.length===sizes.get(h)&&sha(b)===h,'object bytes mismatch');objects.set(h,b);
 }
 need(offset===bytes.length&&objects.size===sizes.size,'object stream completeness');
 const results=[];
 for(const [name,digest,blob,size] of originals) {
  const a=index.archives.find(x=>x.name===name),gitpath='docs/evidence/stdlib-SIO-173/'+name;
  need(a&&a.path===gitpath&&a.sha256===digest&&a.git_blob===blob&&a.compressed_bytes===size,'original binding');
  need(run('git',['rev-parse',commit+':'+gitpath]).trim()===blob,'original Git blob differs');
  need(Number(run('git',['cat-file','-s',commit+':'+gitpath]).trim())===size,'compressed size differs');
  const segments=[],records=index.entries.filter(e=>e.archive===name);let files=0;
  for(const e of records) {
   const h=Buffer.from(e.header_base64,'base64');need(h.length===512,'header length');segments.push(h);
   if(e.type==='0') {segments.push(objects.get(e.sha256));files++;}
   segments.push(Buffer.alloc((512-e.size%512)%512));
  }
  need(Number.isSafeInteger(a.tar_bytes)&&a.tar_bytes>0&&a.tar_bytes<=128*1024*1024,'tar byte bound');
  need(Number.isSafeInteger(a.trailing_zero_bytes)&&a.trailing_zero_bytes>=1024&&a.trailing_zero_bytes<=a.tar_bytes&&a.trailing_zero_bytes<=128*1024*1024,'trailer size');
  segments.push(Buffer.alloc(a.trailing_zero_bytes));
  const proc=spawn('git',['show',commit+':'+gitpath],{stdio:['ignore','pipe','pipe']});
  let stderr='';proc.stderr.on('data',b=>{stderr+=b;});
  const exit=new Promise((resolve,reject)=>{proc.on('error',reject);proc.on('close',code=>code===0?resolve():reject(Error('git show refused '+stderr)));});
  const compressed=createHash('sha256'),tarhash=createHash('sha256');let compressedBytes=0,total=0,part=0,within=0;
  proc.stdout.on('data',b=>{compressedBytes+=b.length;compressed.update(b);});
  const stream=proc.stdout.pipe(createGunzip());
  for await(const chunk of stream) {
   tarhash.update(chunk);total+=chunk.length;let start=0;
   while(start<chunk.length) {
    while(part<segments.length&&within===segments[part].length){part++;within=0;}
    need(part<segments.length,'original has extra bytes');
    const n=Math.min(chunk.length-start,segments[part].length-within);
    need(chunk.subarray(start,start+n).equals(segments[part].subarray(within,within+n)),'original raw byte mismatch: '+name+' at '+(total-chunk.length+start));
    start+=n;within+=n;
   }
  }
  await exit;
  while(part<segments.length&&within===segments[part].length){part++;within=0;}
  need(part===segments.length,'reconstruction has extra bytes');
  need(compressedBytes===size&&compressed.digest('hex')===digest,'original compressed SHA mismatch');
  need(total===a.tar_bytes&&tarhash.digest('hex')===a.tar_sha256,'original tar digest mismatch');
  // GNU tar also independently resolves the original path census.
  const gz=run('git',['show',commit+':'+gitpath],null);
  const listed=execFileSync('tar',['-tzf','-'],{input:gz,encoding:'utf8',maxBuffer:8*1024*1024}).trimEnd().split('\n');
  need(JSON.stringify(listed)===JSON.stringify(records.map(e=>e.name)),'original GNU tar path census differs');
  results.push({archive:name,files,all_tar_bytes_compared:total,original_sha256:digest,git_blob:blob});
 }
 console.log(JSON.stringify({result:'SIO173_INDEPENDENT_ALL_RAW_BYTES_PASS',source_commit:commit,aliases:1142,objects:301,packet_sha256:sha(fs.readFileSync(packet)),index_sha256:sha(indexBytes),results},null,2));
} catch(error) {console.error('SIO173_INDEPENDENT_REFUSE: '+error.message);process.exitCode=1;}
