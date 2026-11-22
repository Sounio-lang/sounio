#!/usr/bin/env node
// Execute actual production Sounio lookup helpers against the original C++23
// linear byte oracle. Neither timing arithmetic nor the oracle uses JavaScript.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {execFileSync,spawnSync} from 'node:child_process';
const root=path.resolve(process.argv[2]||'.');
const source=fs.readFileSync(path.join(root,'self-hosted/compiler/lean_single.sio'),'utf8');
const workflow=fs.readFileSync(path.join(root,'.github/workflows/ci.yml'),'utf8');
const builder=fs.readFileSync(path.join(root,'scripts/ci/build_madaros_macos_arm64.sh'),'utf8');
const linux='node scripts/ci/function_lookup_index_contract.mjs "$PWD" --compiler /tmp/sounio-source-bootstrap-selfhost-linux-x86_64/artifacts/souc-stage2';
const mac='node "$ROOT_DIR/scripts/ci/function_lookup_index_contract.mjs" "$ROOT_DIR" --compiler "$SEED_BIN"';
function fn(text,name){
 const start=text.indexOf('fn '+name+'(');assert(start>=0,'missing production function '+name);
 const begin=text.indexOf('{',start);let depth=1,end=begin+1;
 while(depth&&end<text.length){if(text[end]==='{')depth++;else if(text[end]==='}')depth--;end++;}
 assert.equal(depth,0,'unclosed function '+name);return text.slice(start,end);
}
function check(w,b,s){
 assert(w.includes(linux),'Linux must execute actual helper/oracle');
 assert(w.includes('run: node scripts/ci/function_lookup_index_contract.mjs "$GITHUB_WORKSPACE" --selftest'),'fast bypass controls unwired');
 assert(b.includes(mac),'Darwin must execute actual helper/oracle');
 assert(b.indexOf(mac)<b.indexOf('if [[ -n "${SOUNIO_MACOS_NATIVE_CACHE_RESTORE:-}" ]]'),'helper proof must run before cache hit');
 assert(s.includes('    FN_COUNT = 0\n    fn_lookup_index_reset()\n'),'new compilation must reset derived names');
 for(const name of ['fn_find','fn_find_free']){
  const body=fn(s,name);assert(body.includes('fn_lookup_index_sync()'));
  assert(body.includes('FN_LOOKUP_HASH[i as usize] == h && name_eq('),'hash match cannot replace exact bytes');
 }
 assert(fn(s,'fn_find_free').includes('FN_IN_IMPL[i as usize] == 0'),'method classification must be read live');
 assert.equal((s.match(/^\s*FN_NS\[.*\]\s*=/gm)||[]).length,1,'reassess name-span mutation lifecycle');
 assert.equal((s.match(/^\s*FN_NE\[.*\]\s*=/gm)||[]).length,1,'reassess name-span mutation lifecycle');
 assert.equal((s.match(/^\s*FN_COUNT\s*=/gm)||[]).length,2,'reassess table reset/append lifecycle');
 assert(fn(s,'fn_find_linear').includes('while i < FN_COUNT'));
 assert(fn(s,'fn_find_free_linear').includes('return fn_find_linear(ns, ne)'));
}
check(workflow,builder,source);
if(process.argv.includes('--selftest')){
 for(const [w,b,s] of [[workflow.replace(linux,'true'),builder,source],[workflow,builder.replace(mac,'true'),source],[workflow,builder,source.replace('    FN_COUNT = 0\n    fn_lookup_index_reset()','    FN_COUNT = 0')],[workflow,builder,source.replaceAll('FN_LOOKUP_HASH[i as usize] == h && name_eq(','FN_LOOKUP_HASH[i as usize] == h || name_eq(')],[workflow,builder,source.replace('if FN_IN_IMPL[i as usize] == 0 { return i }','if true { return i }')]])assert.throws(()=>check(w,b,s),'accepted missing lookup obligation');
 console.log('FUNCTION_LOOKUP_INDEX_BYPASS_CONTROLS_PASS');
}
const compilerAt=process.argv.indexOf('--compiler');
const prepareAt=process.argv.indexOf('--prepare-oracle');
const externalAt=process.argv.indexOf('--oracle-binary');
if(compilerAt>=0||prepareAt>=0){
 const compiler=compilerAt>=0?path.resolve(process.argv[compilerAt+1]):null;
 const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'function-lookup-index-'));
 try{
  const cpp=String.raw`
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <utility>
constexpr std::int64_t capacity=65536;
struct Entry {std::int64_t ns,ne,method;};
std::uint64_t hash_name(const std::string& s){std::uint64_t h=2166136261;for(unsigned char c:s)h=((h^c)*16777619)&0x1FFFFFFFFFULL;return h;}
std::string hex_name(std::uint64_t n){std::string s="n";for(int k=15;k>=0;--k)s+="0123456789abcdef"[(n>>(k*4))&15];return s;}
std::pair<std::string,std::string> collision(){
 std::unordered_map<std::uint64_t,std::string> seen;seen.reserve(1200000);std::uint64_t rng=0x6e616d655f696478ULL;
 for(int i=0;i<2000000;++i){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;const auto s=hex_name(rng);const auto h=hash_name(s);const auto [it,added]=seen.emplace(h,s);if(!added&&it->second!=s)return {it->second,s};}
 throw std::runtime_error("real production-width FNV collision not found");
}
int main(int argc,char**argv){
 if(argc!=4)return 2;
 std::ofstream p(argv[1]),expected(argv[2]),facts(argv[3]);
 std::vector<unsigned char> src;std::vector<Entry> entries;
 auto sb=[&](std::int64_t i)->unsigned char{return i>=0&&i<static_cast<std::int64_t>(src.size())?src.at(i):0;};
 auto equal=[&](std::int64_t a,std::int64_t b,std::int64_t c,std::int64_t d){const auto len=b-a;if(len!=d-c)return false;for(std::int64_t i=0;i<len;++i)if(sb(a+i)!=sb(c+i))return false;return true;};
 auto append=[&](const std::string& s){auto ns=static_cast<std::int64_t>(src.size());for(unsigned char c:s){src.push_back(c);p<<"    SRC["<<src.size()-1<<" as usize] = "<<int(c)<<"\n";}p<<"    SRC_LEN = "<<src.size()<<"\n";return std::pair{ns,static_cast<std::int64_t>(src.size())};};
 auto add=[&](const std::string& s,std::int64_t method){const auto span=append(s);const auto i=entries.size();entries.push_back({span.first,span.second,method});p<<"    FN_NS["<<i<<" as usize] = "<<span.first<<"\n    FN_NE["<<i<<" as usize] = "<<span.second<<"\n    FN_IN_IMPL["<<i<<" as usize] = "<<method<<"\n    FN_COUNT = "<<entries.size()<<"\n";return i;};
 auto query_span=[&](std::int64_t ns,std::int64_t ne){std::int64_t first=-1,free=-1;for(std::size_t i=0;i<entries.size();++i)if(equal(entries[i].ns,entries[i].ne,ns,ne)){if(first<0)first=i;if(free<0&&entries[i].method==0)free=i;}if(free<0)free=first;
  expected<<first<<' '<<free<<' '<<first<<' '<<free<<'\n';p<<"    print_int(fn_find("<<ns<<", "<<ne<<")); print(\" \"); print_int(fn_find_free("<<ns<<", "<<ne<<")); print(\" \"); print_int(fn_find_linear("<<ns<<", "<<ne<<")); print(\" \"); print_int(fn_find_free_linear("<<ns<<", "<<ne<<")); print(\"\\n\")\n";};
 auto query=[&](const std::string& s){const auto span=append(s);query_span(span.first,span.second);};
 auto reset=[&]{entries.clear();src.clear();p<<"    FN_COUNT = 0\n    SRC_LEN = 0\n    fn_lookup_index_reset()\n";};
 p<<"fn main() -> i64 with IO, Mut, Panic {\n";reset();query("absent");
 const auto a=add("duplicate",1);add("other",0);const auto b=add("duplicate",0);add("duplicate",0);query("duplicate");query("other");query("missing");query("dupli");query("duplicate_more");
 // Classification is deliberately changed after the index has already synced.
 entries[a].method=0;p<<"    FN_IN_IMPL["<<a<<" as usize] = 0\n";query("duplicate");
 entries[a].method=1;entries[b].method=1;p<<"    FN_IN_IMPL["<<a<<" as usize] = 1\n    FN_IN_IMPL["<<b<<" as usize] = 1\n";query("duplicate");
 entries.back().method=1;p<<"    FN_IN_IMPL[3 as usize] = 1\n";query("duplicate");
 add("appended",0);query("appended");query("duplicate");
 // Zero-length and out-of-source spans retain original sb/name_eq behavior.
 add("",0);query("");query_span(-3,-3);add(std::string(3,'\0'),0);query_span(-3,0);query_span(20000,20003);
 // Same full production hash, not merely the same bucket; byte equality matters.
 reset();const auto pair=collision();if(hash_name(pair.first)!=hash_name(pair.second))return 3;
 add(pair.first,1);query(pair.second);add(pair.second,0);query(pair.second);query(pair.first);
 facts<<"full_hash="<<hash_name(pair.first)<<" first="<<pair.first<<" second="<<pair.second<<'\n';
 // Replace name spans/source at the same count, with the explicit reset hook.
 reset();add("replacement",0);query("replacement");query(pair.first);
 // Original registration ceiling and the last valid table slot, without a
 // giant unrolled fixture: model and executable both construct the same bytes.
 reset();p<<"    var i: i64 = 0\n    while i < 65536 {\n        let base = i * 8\n        SRC[base as usize] = 102\n        var j: i64 = 0\n        while j < 6 {\n            let digit = (i >> ((5 - j) * 4)) & 15\n            SRC[(base + j + 1) as usize] = (if digit < 10 { 48 + digit } else { 87 + digit }) as i8\n            j = j + 1\n        }\n        SRC[(base + 7) as usize] = 0\n        FN_NS[i as usize] = base\n        FN_NE[i as usize] = base + 7\n        FN_IN_IMPL[i as usize] = i & 1\n        i = i + 1\n    }\n    SRC_LEN = 524288\n    FN_COUNT = 65535\n";
 src.resize(capacity*8);entries.clear();for(std::int64_t i=0;i<capacity;++i){const auto base=i*8;src[base]='f';for(int j=0;j<6;++j)src[base+j+1]="0123456789abcdef"[(i>>((5-j)*4))&15];src[base+7]=0;if(i<capacity-1)entries.push_back({base,base+7,i&1});}
 query("f00ffff");query("f00fffe");entries.push_back({(capacity-1)*8,(capacity-1)*8+7,1});p<<"    FN_COUNT = 65536\n";query("f00ffff");query("f000000");query("not_a_registered_name");
 // Shrink forces invalidation before reusing an existing bucket/index.
 entries.resize(1);p<<"    FN_COUNT = 1\n";query("f000000");query("f00ffff");
 // Same-size source replacement needs the explicit compilation reset.
 reset();add("final",0);query("final");query("f000000");
 p<<"    return 0\n}\n";return p&&expected&&facts?0:2;
}`;
  for(const name of ['FN_NS','FN_NE','FN_IN_IMPL','FN_LOOKUP_HEAD','FN_LOOKUP_TAIL','FN_LOOKUP_NEXT','FN_LOOKUP_HASH'])assert(source.includes('var '+name+': [i64; 65536]'),'reassess actual capacity '+name);
  fs.writeFileSync(path.join(tmp,'oracle.cpp'),cpp);
  const digest=x=>crypto.createHash('sha256').update(x).digest('hex');
  let oracle=path.join(tmp,'oracle');
  if(externalAt>=0){
   assert(prepareAt<0,'cannot prepare and consume an oracle simultaneously');
   oracle=path.resolve(process.argv[externalAt+1]);
   const receipt=JSON.parse(fs.readFileSync(oracle+'.receipt.json','utf8'));
   assert.equal(receipt.source_sha256,digest(cpp),'prebuilt oracle is not bound to actual C++23 source');
   assert.equal(receipt.binary_sha256,digest(fs.readFileSync(oracle)),'prebuilt oracle binary changed');
   console.log('CPP23_PREBUILT_ORACLE_SOURCE_AND_BINARY_VERIFIED');
  }else{
   execFileSync(process.env.CXX||'c++',['-std=c++23','-O2','-Wall','-Wextra','-Werror',path.join(tmp,'oracle.cpp'),'-o',oracle],{stdio:'pipe'});
  }
  if(prepareAt>=0){
   const dest=path.resolve(process.argv[prepareAt+1]);fs.copyFileSync(oracle,dest);fs.chmodSync(dest,0o755);
   fs.writeFileSync(dest+'.receipt.json',JSON.stringify({source_sha256:digest(cpp),binary_sha256:digest(fs.readFileSync(dest)),compiler:execFileSync(process.env.CXX||'c++',['--version'],{encoding:'utf8'}).split('\n')[0],scope:'compiled C++23 fixture generator only; numerical oracle/fixture execution not performed'},null,2));
   console.log('CPP23_ORACLE_GENERATOR_COMPILED_ONLY');
  }else{
  execFileSync(oracle,[path.join(tmp,'main.sio'),path.join(tmp,'expected.txt'),path.join(tmp,'collision.txt')],{timeout:30000});
  const globals=['SRC','SRC_LEN','FN_NS','FN_NE','FN_IN_IMPL','FN_COUNT','FN_LOOKUP_HEAD','FN_LOOKUP_TAIL','FN_LOOKUP_NEXT','FN_LOOKUP_HASH','FN_LOOKUP_INDEXED'].map(name=>{const m=source.match(new RegExp('^var '+name+':.*$','m'));assert(m,'missing actual global '+name);return m[0]+'\n';}).join('');
  const names=['sb','gl_name_hash','name_eq','fn_lookup_index_reset','fn_lookup_index_sync','fn_find_linear','fn_find_free_linear','fn_find','fn_find_free'];
  const helpers=names.map(name=>fn(source,name)+'\n').join('\n');
  const main=fs.readFileSync(path.join(tmp,'main.sio'),'utf8'),expected=fs.readFileSync(path.join(tmp,'expected.txt'),'utf8');
  const darwin=process.platform==='darwin';
  function run(label,body,extra=''){
   const src=path.join(tmp,label+'.sio'),out=path.join(tmp,label);fs.writeFileSync(src,globals+extra+body+main);
   execFileSync(compiler,[src,out,...(darwin?['--target','aarch64-macos']:[])],{stdio:'pipe',timeout:120000});
   fs.chmodSync(out,0o755);if(darwin)execFileSync('codesign',['-s','-','-f',out],{stdio:'pipe'});
   const r=spawnSync(out,[],{encoding:'utf8',timeout:30000,maxBuffer:8*1024*1024});assert(!r.error,label+' execution unavailable: '+r.error);return r;
  }
  const good=run('actual',helpers);assert.equal(good.status,0,good.stderr);assert.equal(good.stdout,expected,'actual production lookup differs from independent original linear oracle');
  const chain=`        if FN_LOOKUP_HEAD[bucket as usize] == 0 {
            FN_LOOKUP_HEAD[bucket as usize] = entry
        } else {
            let tail = FN_LOOKUP_TAIL[bucket as usize] - 1
            FN_LOOKUP_NEXT[tail as usize] = entry
        }`;
  assert(helpers.includes(chain));
  const stale=helpers.replace('FN_LOOKUP_HASH[i as usize] = h','FN_LOOKUP_HASH[i as usize] = h\n        LOOKUP_CLASS_CACHE[i as usize] = FN_IN_IMPL[i as usize]').replace(fn(source,'fn_find_free'),fn(source,'fn_find_free').replaceAll('FN_IN_IMPL[','LOOKUP_CLASS_CACHE['));
  const mutants=[['hash-without-byte-equality',helpers.replaceAll(' && name_eq(FN_NS[i as usize], FN_NE[i as usize], ns, ne)','')],['reverse-registration-order',helpers.replace(chain,'        FN_LOOKUP_NEXT[i as usize] = FN_LOOKUP_HEAD[bucket as usize]\n        FN_LOOKUP_HEAD[bucket as usize] = entry')],['missing-index-reset',helpers.replace('    FN_LOOKUP_INDEXED = 0','    // reset deliberately omitted')],['stale-method-classification',stale,'var LOOKUP_CLASS_CACHE: [i64; 65536] = [0; 65536]\n']];
  for(const [label,body,extra]of mutants){assert.notEqual(body,helpers,'mutant did not alter production helper');const bad=run(label,body,extra);assert(bad.status!==0||bad.stdout!==expected,'invalid lookup mutant accepted: '+label);}
  console.log(fs.readFileSync(path.join(tmp,'collision.txt'),'utf8').trim());
  console.log('FUNCTION_LOOKUP_ACTUAL_SOUNIO_CPP23_ORACLE_AND_4_MUTANTS_PASS platform='+process.platform);
  }
 }finally{fs.rmSync(tmp,{recursive:true,force:true});}
}
console.log('FUNCTION_LOOKUP_INDEX_REQUIRED_CONTRACT_PASS');
