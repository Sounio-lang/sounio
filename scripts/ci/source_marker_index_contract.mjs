#!/usr/bin/env node
// Compile the actual production helper, compare it to the original byte-scan
// oracle in C++23, and prove wrong-boundary/missing-reset controls stay red.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFileSync,spawnSync} from 'node:child_process';
const root=path.resolve(process.argv[2]||'.');
const source=fs.readFileSync(path.join(root,'self-hosted/compiler/lean_single.sio'),'utf8');
const workflow=fs.readFileSync(path.join(root,'.github/workflows/ci.yml'),'utf8');
const builder=fs.readFileSync(path.join(root,'scripts/ci/build_madaros_macos_arm64.sh'),'utf8');
const linux='node scripts/ci/source_marker_index_contract.mjs "$PWD" --compiler /tmp/sounio-source-bootstrap-selfhost-linux-x86_64/artifacts/souc-stage2';
const mac='node "$ROOT_DIR/scripts/ci/source_marker_index_contract.mjs" "$ROOT_DIR" --compiler "$SEED_BIN"';
function check(w,b,s){
 assert(w.includes(linux),'source-bootstrap must execute actual helper');
 assert(b.includes(mac),'Darwin builder must execute actual helper');
 assert(b.indexOf(mac)<b.indexOf('if [[ -n "${SOUNIO_MACOS_NATIVE_CACHE_RESTORE:-}" ]]'),'helper proof cannot depend on cache hit');
 assert(w.includes('run: node scripts/ci/source_marker_index_contract.mjs "$GITHUB_WORKSPACE" --selftest'),'fast contract unwired');
 assert(s.includes('fn strip_extern_blocks() with IO, Mut, Panic, Div {\n    src_file_marker_index_reset()'),'in-place compaction must reset index');
 assert(s.includes('    src_file_marker_index_reset()\n    let raw = read_file(src)'),'new source must reset index');
 assert(s.includes('return src_file_marker_start(a) == src_file_marker_start(b)'),'same-file rule bypassed');
 assert(s.includes('if src_same_file_pos(TS[access_tok as usize], ST_DECL_NS[si as usize]) { return false }'),'private-field ownership bypassed');
}
check(workflow,builder,source);
if(process.argv.includes('--selftest')){
 for(const [w,b,s]of [[workflow.replace(linux,'true'),builder,source],[workflow,builder.replace(mac,'true'),source],[workflow,builder,source.replace('return src_file_marker_start(a) == src_file_marker_start(b)','return true')],[workflow,builder,source.replace('    src_file_marker_index_reset()\n    let raw = read_file(src)','    let raw = read_file(src)')]])assert.throws(()=>check(w,b,s),'accepted missing semantic obligation');
 console.log('SOURCE_MARKER_INDEX_GATE_BYPASS_CONTROLS_PASS');
}
const compilerAt=process.argv.indexOf('--compiler');
if(compilerAt>=0){
 const compiler=path.resolve(process.argv[compilerAt+1]);
 const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'source-marker-index-'));
 try{
  const oracle=String.raw`
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
int main(int argc,char**argv){
 if(argc!=3)return 2;
 constexpr long long cap=16777216;
 std::vector<unsigned char> src(cap,0);long long len=0;
 std::ofstream probe(argv[1]),expected(argv[2]);
 auto set=[&](long long p,int v){src.at(p)=v;probe<<"    SRC["<<p<<" as usize] = "<<v<<"\n";};
 auto length=[&](long long n){len=n;probe<<"    SRC_LEN = "<<n<<"\n";};
 auto reset=[&]{probe<<"    src_file_marker_index_reset()\n";};
 auto query=[&](long long p){long long start=0;for(long long i=0;i<p&&i<len;++i)if(src.at(i)==1)start=i;expected<<start<<'\n';probe<<"    print_int(src_file_marker_start("<<p<<")); print(\"\\n\")\n";};
 probe<<"fn main() -> i64 with IO, Mut, Panic {\n";
 for(auto p:{0LL,255LL,256LL,511LL,512LL,1024LL})set(p,1);
 length(1025);reset();
 // Both increasing and decreasing lookups, with strict boundaries and clamping.
 for(long long q=-2;q<=1027;++q)query(q);
 for(long long q=1027;q>=0;--q)query(q);
 // Appending changes the nearest marker, including at a checkpoint boundary.
 set(1025,1);set(1279,1);length(1280);
 for(auto q:{1025LL,1026LL,1279LL,1280LL,1281LL,1LL,256LL})query(q);
 set(1791,1);length(1800);query(1800);query(1791);query(1792);
 // Existing bytes may be replaced only with explicit index invalidation.
 set(255,0);set(256,0);set(511,0);set(512,0);set(1024,0);set(1025,0);set(1279,0);set(1791,0);
 probe<<"    // replaced-prefix-reset\n";reset();query(1800);query(1026);
 // Shrinking invalidates indexed boundaries; an append then extends correctly.
 length(300);query(300);set(300,1);length(600);query(301);query(600);
 // Last valid byte, final complete block, beyond-end clamp and backward query.
 set(cap-257,1);set(cap-256,1);set(cap-1,1);length(cap);
 for(auto q:{cap,cap+1,cap-1,cap-256,cap-255,0LL,-1LL,257LL})query(q);
 probe<<"    0\n}\n";
 return probe&&expected?0:2;
}`;
  const cap=source.match(/^var SRC_CAP: i64 = ([0-9]+)$/m)[1];
  const bufferCap=source.match(/^var SRC: \[i8; ([0-9]+)\]/m)[1];
  const prefixCap=source.match(/^var SRC_FILE_PREFIX: \[i64; ([0-9]+)\]/m)[1];
  const boundOracle=oracle.replace('constexpr long long cap=16777216;','constexpr long long cap='+cap+'; static_assert(cap=='+bufferCap+'); static_assert('+prefixCap+' == (cap >> 8)+1);');
  fs.writeFileSync(path.join(tmp,'oracle.cpp'),boundOracle);
  execFileSync(process.env.CXX||'c++',['-std=c++23','-O2','-Wall','-Wextra','-Werror',path.join(tmp,'oracle.cpp'),'-o',path.join(tmp,'oracle')]);
  execFileSync(path.join(tmp,'oracle'),[path.join(tmp,'main.sio'),path.join(tmp,'expected.txt')]);
  const begin=source.indexOf('var SRC_FILE_PREFIX:');
  const end=source.indexOf('fn src_same_file_pos(',begin);
  assert(begin>=0&&end>begin,'actual helper section missing');
  const globals=source.match(/^var SRC:.*$/m)[0]+'\n'+source.match(/^var SRC_LEN:.*$/m)[0]+'\n';
  const helper=source.slice(begin,end);
  const main=fs.readFileSync(path.join(tmp,'main.sio'),'utf8');
  const expected=fs.readFileSync(path.join(tmp,'expected.txt'),'utf8');
  const darwin=process.platform==='darwin';
  function run(label,body){
   const src=path.join(tmp,label+'.sio'),out=path.join(tmp,label);fs.writeFileSync(src,globals+body);
   execFileSync(compiler,[src,out,...(darwin?['--target','aarch64-macos']:[])],{stdio:'pipe'});
   fs.chmodSync(out,0o755);if(darwin)execFileSync('codesign',['-s','-','-f',out],{stdio:'pipe'});
   return spawnSync(out,[],{encoding:'utf8',timeout:30000,maxBuffer:8*1024*1024});
  }
  const good=run('actual',helper+main);assert.equal(good.status,0,good.stderr);assert.equal(good.stdout,expected,'actual source differs from original linear oracle');
  for(const [label,body]of [['inclusive-marker',helper.replace('while i < end {','while i <= end {')+main],['missing-reset',helper+main.replace('    // replaced-prefix-reset\n    src_file_marker_index_reset()','    // reset deliberately omitted')]]){
   const bad=run(label,body);assert(bad.status!==0||bad.stdout!==expected,'invalid boundary/reset mutant accepted: '+label);
  }
  console.log('SOURCE_MARKER_ACTUAL_SOUNIO_CPP23_ORACLE_AND_MUTANTS_PASS platform='+process.platform);
 }finally{fs.rmSync(tmp,{recursive:true,force:true});}
}
console.log('SOURCE_MARKER_INDEX_REQUIRED_CONTRACT_PASS');
