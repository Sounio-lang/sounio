#!/usr/bin/env node
// Operational evidence custody only. No scientific calculation or execution.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { createGunzip, createGzip, gzipSync } from 'node:zlib';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

export const COMMIT = '3c7c8073411a7e7254d0d9424f59f8e49eb49180';
export const ORIGINALS = [
 {name:'native-cpp-text.tar.gz',path:'docs/evidence/stdlib-SIO-173/native-cpp-text.tar.gz',
  sha256:'0a80794d00ba129d90e9e6043aa19026a3aacfdccd10be374cd434edd4920074',
  git_blob:'aba131479c35d1f3ccf0733762c37fc4cd0cb2eb',compressed_bytes:6527120},
 {name:'final-17307-text.tar.gz',path:'docs/evidence/stdlib-SIO-173/final-17307-text.tar.gz',
  sha256:'0681e5ada0de62957df0d1f63df529b295bb12e6bcc6dc56af5f8b161642e723',
  git_blob:'297a6cbacb48ec8d2e4c3fc6d82a2cd48fbce0c2',compressed_bytes:8828874}
];
const MAX_ENTRY = 16*1024*1024, MAX_TAR = 128*1024*1024;
export const sha = bytes => createHash('sha256').update(bytes).digest('hex');
function need(ok,message) { if(!ok)throw Error(message); }
function safe(name,directory=false) {
 need(typeof name==='string' && name.length>0 && !/[\\\x00-\x1f\x7f]/.test(name),'unsafe path');
 const clean=directory && name.endsWith('/')?name.slice(0,-1):name;
 need(!path.posix.isAbsolute(clean) && clean.split('/').every(p=>p!==''&&p!=='.'&&p!=='..'),'unsafe path');
}
function field(h,a,b) { return h.subarray(a,b).toString('utf8').split('\0')[0]; }
function oct(h,a,b) {
 const text=field(h,a,b).trim();need(/^[0-7]*$/.test(text),'unsupported tar number');
 const n=parseInt(text||'0',8);need(Number.isSafeInteger(n)&&n>=0,'tar number overflow');return n;
}
export function header(h) {
 need(h.length===512,'tar header length');
 const stored=oct(h,148,156);let sum=0;
 for(let i=0;i<512;i++)sum+=i>=148&&i<156?32:h[i];
 need(stored===sum,'tar header checksum');
 const prefix=field(h,345,500),base=field(h,0,100),name=prefix?prefix+'/'+base:base;
 const type=field(h,156,157)||'0',size=oct(h,124,136);
 need(type==='0'||type==='5','unsupported tar entry type');
 safe(name,type==='5');need(size<=MAX_ENTRY && (type!=='5'||size===0),'tar entry size');
 need(field(h,157,257)==='','unexpected tar link');
 return {name,type,size};
}
// Parse one bounded entry at a time, retaining every original 512-byte header.
// GNU/PAX/symlink/hardlink inputs are deliberately refused, never silently flattened.
export async function* readTar(filename, summary={}) {
 let pending=Buffer.alloc(0),state='header',record,left=0,padding=0,parts=[],ended=false,zeros=0,total=0;
 const digest=createHash('sha256'),names=new Set();
 const stream=fs.createReadStream(filename).pipe(createGunzip());
 for await(const chunk of stream) {
  total+=chunk.length;need(total<=MAX_TAR,'tar expanded bound');digest.update(chunk);
  pending=Buffer.concat([pending,chunk]);
  for(;;) {
   if(state==='header') {
    if(pending.length<512)break;
    const raw=Buffer.from(pending.subarray(0,512));pending=pending.subarray(512);
    if(raw.every(v=>v===0)){ended=true;zeros+=512;continue;}
    need(!ended,'tar data after terminator');
    const meta=header(raw);need(!names.has(meta.name),'duplicate tar path');names.add(meta.name);
    record={...meta,header_base64:raw.toString('base64')};left=meta.size;padding=(512-left%512)%512;parts=[];state='data';
   }
   if(state==='data') {
    if(left) {
     if(!pending.length)break;
     const n=Math.min(left,pending.length);parts.push(Buffer.from(pending.subarray(0,n)));
     pending=pending.subarray(n);left-=n;if(left)break;
    }
    state='padding';
   }
   if(state==='padding') {
    if(pending.length<padding)break;
    need(pending.subarray(0,padding).every(v=>v===0),'nonzero tar padding');
    pending=pending.subarray(padding);state='header';
    yield {...record,data:Buffer.concat(parts)};
   }
  }
 }
 need(state==='header'&&pending.length===0&&zeros>=1024,'truncated tar');
 Object.assign(summary,{tar_bytes:total,tar_sha256:digest.digest('hex'),trailing_zero_bytes:zeros});
}
function makeHeader(name,size) {
 safe(name);need(Buffer.byteLength(name)<=100,'packet name too long');
 const h=Buffer.alloc(512);h.write(name);
 const number=(v,a,n)=>h.write(v.toString(8).padStart(n-1,'0')+'\0',a,n,'ascii');
 number(0o644,100,8);number(0,108,8);number(0,116,8);number(size,124,12);number(0,136,12);
 h.fill(32,148,156);h.write('0',156);h.write('ustar\0',257);h.write('00',263);
 const sum=h.reduce((s,v)=>s+v,0);h.write(sum.toString(8).padStart(6,'0')+'\0 ',148,8);return h;
}
function* tarRecord(name,data) { yield makeHeader(name,data.length);yield data;yield Buffer.alloc((512-data.length%512)%512); }
async function hashFile(filename) { const h=createHash('sha256');for await(const b of fs.createReadStream(filename))h.update(b);return h.digest('hex'); }
export function validateRows(index,objects) {
 need(index.schema==='sounio.sio173.evidence.v1'&&Array.isArray(index.entries),'packet schema');
 const names=new Set(),used=new Set();
 for(const e of index.entries) {
  need(index.archives.some(a=>a.name===e.archive),'unknown original archive');
  safe(e.name,e.type==='5');const key=e.archive+'/'+e.name;need(!names.has(key),'duplicate alias');names.add(key);
  need(typeof e.header_base64==='string'&&/^[A-Za-z0-9+/]+={0,2}$/.test(e.header_base64),'header encoding');
  const parsed=header(Buffer.from(e.header_base64,'base64'));
  need(parsed.name===e.name&&parsed.type===e.type&&parsed.size===e.size,'original header metadata mismatch');
  if(e.type==='0') {
   need(/^[a-f0-9]{64}$/.test(e.sha256),'object digest');
   const data=objects.get(e.sha256);need(data!==undefined,'missing object');
   need(data.length===e.size&&sha(data)===e.sha256,'object bytes/digest mismatch');used.add(e.sha256);
  } else need(e.sha256===null,'directory payload');
 }
 need(used.size===objects.size,'unreferenced object');
}
export async function loadPacket(filename) {
 let index;const objects=new Map();let count=0;
 for await(const r of readTar(filename)) {
  if(count++===0) {need(r.name==='INDEX.json','index must be first');index=JSON.parse(r.data);continue;}
  need(/^objects\/[0-9a-f]{64}$/.test(r.name)&&r.type==='0','unexpected packet entry');
  const key=r.name.slice(8);need(!objects.has(key),'duplicate object');need(sha(r.data)===key,'object content tamper');objects.set(key,r.data);
 }
 need(index&&index.source_commit===COMMIT,'original commit binding');
 need(JSON.stringify(index.archives.map(({name,path,sha256,git_blob,compressed_bytes})=>({name,path,sha256,git_blob,compressed_bytes})))===JSON.stringify(ORIGINALS),'original archive binding');
 validateRows(index,objects);
 need(index.entries.filter(e=>e.type==='0').length===1142&&objects.size===301,'SIO173 census mismatch');
 for(const a of index.archives) {
  need(Number.isSafeInteger(a.tar_bytes)&&a.tar_bytes<=MAX_TAR&&a.tar_bytes>0,'tar byte bound');
  need(Number.isSafeInteger(a.trailing_zero_bytes)&&a.trailing_zero_bytes>=1024&&a.trailing_zero_bytes%512===0,'tar terminator bound');
  need(/^[a-f0-9]{64}$/.test(a.tar_sha256),'tar digest');
  const digest=createHash('sha256');let bytes=0;
  for(const e of index.entries.filter(e=>e.archive===a.name)) {
   const chunks=[Buffer.from(e.header_base64,'base64')];
   if(e.type==='0')chunks.push(objects.get(e.sha256));
   chunks.push(Buffer.alloc((512-e.size%512)%512));
   for(const chunk of chunks){digest.update(chunk);bytes+=chunk.length;}
  }
  digest.update(Buffer.alloc(a.trailing_zero_bytes));bytes+=a.trailing_zero_bytes;
  need(bytes===a.tar_bytes&&digest.digest('hex')===a.tar_sha256,'original tar reconstruction digest');
 }
 return {index,objects};
}
async function build(output) {
 need(!fs.existsSync(output),'fresh packet required');
 const index={schema:'sounio.sio173.evidence.v1',source_commit:COMMIT,archives:[],entries:[]},unique=new Set();
 for(const original of ORIGINALS) {
  need(fs.statSync(original.path).size===original.compressed_bytes&&await hashFile(original.path)===original.sha256,'original archive changed');
  const summary={};
  for await(const r of readTar(original.path,summary)) {
   const {data,...metadata}=r,digest=r.type==='0'?sha(data):null;
   index.entries.push({archive:original.name,...metadata,sha256:digest});if(digest)unique.add(digest);
  }
  index.archives.push({...original,...summary});
 }
 need(index.entries.filter(e=>e.type==='0').length===1142&&unique.size===301,'original census mismatch');
 async function* records() {
  yield* tarRecord('INDEX.json',Buffer.from(JSON.stringify(index)+'\n'));
  const emitted=new Set();
  for(const original of ORIGINALS) {
   for await(const r of readTar(original.path))if(r.type==='0') {
    const digest=sha(r.data);if(emitted.has(digest))continue;
    need(unique.has(digest),'second-pass original changed');emitted.add(digest);
    yield* tarRecord('objects/'+digest,r.data);
   }
   need(await hashFile(original.path)===original.sha256,'original archive changed after build');
  }
  need(emitted.size===unique.size,'second-pass object census');
  yield Buffer.alloc(1024);
 }
 await pipeline(Readable.from(records()),createGzip({level:9}),fs.createWriteStream(output,{flags:'wx'}));
 await loadPacket(output);
 console.log(JSON.stringify({packet:output,sha256:await hashFile(output),bytes:fs.statSync(output).size,aliases:1142,objects:301}));
}

async function testPacket(filename,output) {
 need(!fs.existsSync(output),'fresh negative-control directory required');fs.mkdirSync(output);
 const {index,objects}=await loadPacket(filename);
 const cases=[
  ['tampered-object','object content tamper',(i,o)=>{const key=[...o.keys()].find(k=>o.get(k).length>0),b=Buffer.from(o.get(key));b[0]^=1;o.set(key,b);}],
  ['missing-object','missing object',(i,o)=>o.delete(o.keys().next().value)],
  ['unsafe-alias','unsafe path',i=>{i.entries.find(e=>e.type==='0').name='../escape';}],
  ['duplicate-alias','duplicate alias',i=>{i.entries.push({...i.entries.find(e=>e.type==='0')});}],
  ['duplicate-object','duplicate tar path',()=>{}],
  ['unsafe-packet-path','unsafe path',()=>{}]
 ];
 const results=[];
 for(const [name,diagnostic,mutate] of cases) {
  const i=structuredClone(index),o=new Map(objects);mutate(i,o);
  const chunks=[...tarRecord('INDEX.json',Buffer.from(JSON.stringify(i)+'\n'))];
  for(const [key,b]of o)chunks.push(...tarRecord('objects/'+key,b));
  if(name==='duplicate-object'){const [k,b]=o.entries().next().value;chunks.push(...tarRecord('objects/'+k,b));}
  // An unsafe raw header cannot be emitted through the safe writer.
  if(name==='unsafe-packet-path'){
   const raw=makeHeader('objects/'+o.keys().next().value,0);raw.fill(0,0,100);raw.write('../escape');
   raw.fill(32,148,156);const sum=raw.reduce((n,v)=>n+v,0);raw.write(sum.toString(8).padStart(6,'0')+'\0 ',148,8);chunks.push(raw);
  }
  chunks.push(Buffer.alloc(1024));const file=path.join(output,name+'.tar.gz');fs.writeFileSync(file,gzipSync(Buffer.concat(chunks),{level:9}));
  let message='';try{await loadPacket(file);}catch(error){message=error.message;}
  need(message.includes(diagnostic),'negative control did not specifically refuse '+name+': '+message);
  results.push({name,refused:true,diagnostic:message,sha256:await hashFile(file)});
 }
 fs.writeFileSync(path.join(output,'RESULTS.json'),JSON.stringify(results,null,2)+'\n');
 console.log('SIO173_SIX_ACTUAL_PACKET_CONTROLS_PASS');
}

function selftest() {
 const data=Buffer.from('metadata fixture\n'),digest=sha(data),raw=makeHeader('case/stdout.txt',data.length);
 const index={schema:'sounio.sio173.evidence.v1',archives:[{name:'original.gz'}],
 entries:[{archive:'original.gz',name:'case/stdout.txt',type:'0',size:data.length,sha256:digest,header_base64:raw.toString('base64')}]};
 const good=new Map([[digest,data]]);validateRows(index,good);let controls=1;
 function refused(label,edit,objects=good) {const x=structuredClone(index);edit(x);assert.throws(()=>validateRows(x,objects));console.log('REFUSED '+label);controls++;}
 refused('tampered object',()=>{},new Map([[digest,Buffer.from('changed')]]));
 refused('missing object',()=>{},new Map());
 refused('unsafe parent path',x=>x.entries[0].name='../escape');
 refused('unsafe absolute path',x=>x.entries[0].name='/escape');
 refused('duplicate alias',x=>x.entries.push({...x.entries[0]}));
 refused('header metadata tamper',x=>x.entries[0].size++);
 refused('unknown archive',x=>x.entries[0].archive='other.gz');
 assert.throws(()=>header(Buffer.alloc(512)));controls++;
 console.log('SIO173_PACKET_METADATA_SELFTEST_PASS controls='+controls);
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
 try {
  const [mode,...args]=process.argv.slice(2);
  if(mode==='--test-packet'&&args.length===2)await testPacket(args[0],args[1]);
  else if(mode==='build'&&args.length===1)await build(args[0]);
  else if(mode==='verify'&&args.length===1) {const p=await loadPacket(args[0]);console.log('SIO173_PACKET_VERIFIED aliases=1142 objects='+p.objects.size);}
  else if(mode==='cat'&&args.length===3) {
   const {index,objects}=await loadPacket(args[0]);const r=index.entries.find(e=>e.archive===args[1]&&e.name===args[2]&&e.type==='0');
   need(r,'unknown file alias');process.stdout.write(objects.get(r.sha256));
  } else if(mode==='--selftest'&&args.length===0)selftest();
  else throw Error('usage: build PACKET | verify PACKET | cat PACKET ORIGINAL_ARCHIVE PATH | --selftest | --test-packet PACKET FRESH_OUT');
 } catch(error) {console.error('SIO173_CUSTODY_REFUSE: '+error.message);process.exitCode=1;}
}
