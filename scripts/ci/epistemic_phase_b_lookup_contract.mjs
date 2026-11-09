#!/usr/bin/env node
// Execute the four actual Phase-B selection/update snippets against the
// original first-name linear state contract implemented independently in C++23.
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
const linux='node scripts/ci/epistemic_phase_b_lookup_contract.mjs "$PWD" --compiler /tmp/sounio-source-bootstrap-selfhost-linux-x86_64/artifacts/souc-stage2';
const mac='node "$ROOT_DIR/scripts/ci/epistemic_phase_b_lookup_contract.mjs" "$ROOT_DIR" --compiler "$SEED_BIN"';
function fn(s,name){const a=s.indexOf('fn '+name+'('),b=s.indexOf('\nfn ',a+1);assert(a>=0&&b>a,name);return s.slice(a,b);}
function block(s,a){const o=s.indexOf('{',a);let d=1,e=o+1;while(d){assert(e<s.length,'unclosed block');if(s[e]==='{')d++;if(s[e]==='}')d--;e++;}return s.slice(a,e);}
function actual(s){
 const pass=fn(s,'epistemic_infer_pass'),phase=pass.slice(pass.indexOf('    // Phase B:'),pass.indexOf('    // Phase C:'));
 const pa=phase.indexOf('// Gen 11: fn-pointer parameter'),pb=phase.indexOf('if param_conf == 0 {',pa);assert(pa>=0&&pb>pa);
 const ra=phase.indexOf('var rfj = fn_find(rns, rne)'),rb=phase.indexOf('if rfj == FN_COUNT {',ra);assert(ra>=0&&rb>ra);
 const ca=phase.indexOf('let call_fi = fn_find(ins, ine)'),cb=phase.indexOf('if call_fi >= 0 {',ca);assert(ca>=0&&cb>ca);
 const na=phase.indexOf('let nfj = fn_find(nns, nne)'),nb=phase.indexOf('if nested_conf == 0 &&',na);assert(na>=0&&nb>na);
 const guardEnd=phase.indexOf('{',nb);assert(guardEnd>nb);
 return {parameter:block(phase,pb),rhs:phase.slice(ra,rb),rhsMiss:'rfj == FN_COUNT',call:phase.slice(ca,cb),nested:phase.slice(na,nb),measureGuard:phase.slice(nb+3,guardEnd).trim(),pass};
}
function check(s,w,b){
 const x=actual(s);
 for(const[v,args]of[['g11_fj','ptns, ptne'],['rfj','rns, rne'],['call_fi','ins, ine'],['nfj','nns, nne']])assert((x.parameter+x.rhs+x.call+x.nested).includes(v+' = fn_find('+args+')'),'first exact declaration required: '+v);
 assert(x.parameter.includes('if param_conf == 0 {')&&x.parameter.includes('if g11_fj >= 0 {'));
 assert(x.rhs.includes('if rfj < 0 { rfj = FN_COUNT }'),'preserve original builtin miss sentinel');
 assert(x.rhs.includes('if rfj < FN_COUNT {')&&x.rhs.includes('if rhs_ty == 0 { rhs_conf = 0 }'));
 assert(x.rhs.includes('rhs_conf = FN_EFF_CONF[rfj as usize]'));
 assert(x.nested.includes('if nfj >= 0 {')&&x.nested.includes('if nret != 0 {')&&x.nested.includes('nested_conf = FN_EFF_CONF[nfj as usize]'));
 assert(x.pass.includes('while efi2 < FN_COUNT {')&&x.pass.includes('ECONF_APPLIED = ECONF_APPLIED + 1'));
 assert(x.pass.includes('while refine_changed > 0 && refine_pass < 7 {'));
 assert(w.includes(linux)&&w.includes('run: node scripts/ci/epistemic_phase_b_lookup_contract.mjs "$GITHUB_WORKSPACE" --selftest'));
 assert(b.includes(mac)&&b.indexOf(mac)<b.indexOf('if [[ -n "${SOUNIO_MACOS_NATIVE_CACHE_RESTORE:-}" ]]'));
 return x;
}
const snippets=check(source,workflow,builder);
if(process.argv.includes('--selftest')){
 for(const[v,args]of[['g11_fj','ptns, ptne'],['rfj','rns, rne'],['call_fi','ins, ine'],['nfj','nns, nne']])assert.throws(()=>check(source.replace(v+' = fn_find('+args+')',v+' = fn_find_free('+args+')'),workflow,builder));
 for(const[s,w,b]of[[source.replace('if rfj < 0 { rfj = FN_COUNT }','// miss normalization omitted'),workflow,builder],[source,workflow.replace(linux,'true'),builder],[source,workflow,builder.replace(mac,'true')]])assert.throws(()=>check(s,w,b));
 console.log('EPISTEMIC_PHASE_B_LOOKUP_BYPASS_CONTROLS_PASS');
}
const cpp=String.raw`
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <utility>
struct Entry{std::string name;std::int64_t type,conf,method;};
std::uint64_t hash(const std::string&s){std::uint64_t h=2166136261;for(unsigned char c:s)h=((h^c)*16777619)&0x1FFFFFFFFFULL;return h;}
int main(int argc,char**argv){
 if(argc!=3)return 2;
 std::ofstream p(argv[1]),out(argv[2]);std::vector<Entry> entries;std::string src;
 auto append=[&](const std::string&n){auto ns=src.size();for(unsigned char c:n){p<<"    SRC["<<src.size()<<" as usize] = "<<int(c)<<"\n";src+=char(c);}p<<"    SRC_LEN = "<<src.size()<<"\n";return std::pair{ns,src.size()};};
 auto add=[&](std::string n,std::int64_t type,std::int64_t conf,std::int64_t method){auto[a,b]=append(n);auto i=entries.size();entries.push_back({n,type,conf,method});p<<"    FN_NS["<<i<<" as usize] = "<<a<<"\n    FN_NE["<<i<<" as usize] = "<<b<<"\n    FN_RET_TY["<<i<<" as usize] = "<<type<<"\n    FN_EFF_CONF["<<i<<" as usize] = "<<conf<<"\n    FN_IN_IMPL["<<i<<" as usize] = "<<method<<"\n    FN_COUNT = "<<entries.size()<<"\n";};
 auto ty=[](std::int64_t k)->std::int64_t{switch(k){case 1:return 1;case 2:return 2;case 3:return 4;case 4:return 3;case 5:return 5;case 6:case 7:return 6;default:return 0;}};
 auto query=[&](const std::string&n,std::int64_t pc){const auto input_conf=pc;auto[a,b]=append(n);std::size_t i=0;while(i<entries.size()&&entries[i].name!=n)++i;const bool found=i<entries.size();std::int64_t pt=77,rty=77,rc=17,meas=94,nested=0;
  if(found){const auto&e=entries[i];if(pc==0){pt=ty(e.type);pc=1000;}rty=ty(e.type);rc=rty==0?0:e.conf;if(e.name=="measured")meas=990;if(e.name=="asserted")meas=970;if(e.name=="constant")meas=1000;if(ty(e.type)!=0)nested=e.conf;}
  out<<pt<<' '<<pc<<' '<<i<<' '<<rty<<' '<<rc<<' '<<meas<<' '<<(!found)<<' '<<(found?static_cast<std::int64_t>(i):-1)<<' '<<nested<<' '<<(nested==0&&(n=="measure"||n=="measured"))<<'\n';
  p<<"    actual_query("<<a<<", "<<b<<", "<<input_conf<<")\n";
 };
 p<<"fn main() -> i64 with IO, Mut, Panic {\n    FN_COUNT = 0\n    SRC_LEN = 0\n    fn_lookup_index_reset()\n";
 query("missing",0);add("duplicate",3,211,1);add("duplicate",2,990,0);add("unknown",0,321,0);
 for(auto n:{"duplicate","unknown","missing","dupli","duplicate_more","measure","measured"}){query(n,0);query(n,999);query(n,1000);}
 entries[0].conf=19;p<<"    FN_EFF_CONF[0 as usize] = 19\n";query("duplicate",0);
 entries[0].type=0;p<<"    FN_RET_TY[0 as usize] = 0\n";query("duplicate",0);
 for(std::int64_t k=0;k<=8;++k){auto n="kind"+std::to_string(k);add(n,k,k*100,k&1);query(n,0);query(n,999);}
 add("measured",2,0,1);add("measured",2,999,0);add("asserted",2,417,0);add("constant",2,804,0);add("measure",0,700,0);
 for(auto n:{"measure","measured","asserted","constant"})query(n,0);
 const std::string a="n20201c6f331ffa0a",b="n6c1dc972f196cbb7";if(a==b||hash(a)!=hash(b))throw std::runtime_error("full-hash collision changed");add(a,1,117,1);query(b,0);add(b,7,814,0);query(a,0);query(b,0);add("append",4,913,0);query("append",0);
 p<<"    return 0\n}\n";return p&&out?0:2;
}
`;
const at=process.argv.indexOf('--compiler'),prep=process.argv.indexOf('--prepare-oracle'),ext=process.argv.indexOf('--oracle-binary');
if(at>=0||prep>=0){
 const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'epistemic-phase-b-'));
 try{
  const hash=x=>crypto.createHash('sha256').update(x).digest('hex');
  fs.writeFileSync(path.join(tmp,'oracle.cpp'),cpp);let oracle=path.join(tmp,'oracle');
  if(ext>=0){assert(prep<0);oracle=path.resolve(process.argv[ext+1]);const r=JSON.parse(fs.readFileSync(oracle+'.receipt.json'));assert.equal(r.source_sha256,hash(cpp));assert.equal(r.binary_sha256,hash(fs.readFileSync(oracle)));}
  else execFileSync(process.env.CXX||'c++',['-std=c++23','-O2','-Wall','-Wextra','-Werror',path.join(tmp,'oracle.cpp'),'-o',oracle],{stdio:'pipe'});
  if(prep>=0){const dest=path.resolve(process.argv[prep+1]);fs.copyFileSync(oracle,dest);fs.chmodSync(dest,0o755);fs.writeFileSync(dest+'.receipt.json',JSON.stringify({source_sha256:hash(cpp),binary_sha256:hash(fs.readFileSync(dest)),scope:'compiled C++23 generator only; no numerical execution'},null,2));console.log('EPISTEMIC_PHASE_B_CPP23_GENERATOR_COMPILED_ONLY');}
  else{
   execFileSync(oracle,[path.join(tmp,'main.sio'),path.join(tmp,'expected.txt')],{timeout:30000});
   const globals=['SRC','SRC_LEN','FN_NS','FN_NE','FN_IN_IMPL','FN_RET_TY','FN_EFF_CONF','FN_COUNT','FN_LOOKUP_HEAD','FN_LOOKUP_TAIL','FN_LOOKUP_NEXT','FN_LOOKUP_HASH','FN_LOOKUP_INDEXED'].map(n=>{const m=source.match(new RegExp('^var '+n+':.*$','m'));assert(m,n);return m[0]+'\n';}).join('');
   const helpers=['sb','gl_name_hash','name_eq','src_match','fn_lookup_index_reset','fn_lookup_index_sync','fn_find','fn_find_free','ety_from_tnk'].map(n=>fn(source,n)+'\n').join('\n');
   const main=fs.readFileSync(path.join(tmp,'main.sio'),'utf8'),expected=fs.readFileSync(path.join(tmp,'expected.txt'),'utf8'),darwin=process.platform==='darwin',compiler=path.resolve(process.argv[at+1]);
   function run(label,x,h=helpers){
    const q=`fn actual_query(ns: i64, ne: i64, initial_conf: i64) with IO, Mut, Panic {\n    let ptns = ns\n    let ptne = ne\n    var param_ty: i64 = 77\n    var param_conf: i64 = initial_conf\n${x.parameter}\n    let rns = ns\n    let rne = ne\n    var rhs_ty: i64 = 77\n    var rhs_conf: i64 = 17\n    var bind_meas_conf: i64 = 94\n${x.rhs}\n    var fallback: i64 = 0\n    if ${x.rhsMiss} { fallback = 1 }\n    let ins = ns\n    let ine = ne\n${x.call}\n    let nns = ns\n    let nne = ne\n${x.nested}\n    var measure_fallback: i64 = 0\n    if ${x.measureGuard} { measure_fallback = 1 }\n    print_int(param_ty); print(" "); print_int(param_conf); print(" "); print_int(rfj); print(" "); print_int(rhs_ty); print(" "); print_int(rhs_conf); print(" "); print_int(bind_meas_conf); print(" "); print_int(fallback); print(" "); print_int(call_fi); print(" "); print_int(nested_conf); print(" "); print_int(measure_fallback); print("\\n")\n}\n`;
    const src=path.join(tmp,label+'.sio'),bin=path.join(tmp,label);fs.writeFileSync(src,globals+h+q+main);execFileSync(compiler,[src,bin,...(darwin?['--target','aarch64-macos']:[])],{stdio:'pipe',timeout:120000});fs.chmodSync(bin,0o755);if(darwin)execFileSync('codesign',['-s','-','-f',bin],{stdio:'pipe'});const r=spawnSync(bin,[],{encoding:'utf8',timeout:30000,maxBuffer:8*1024*1024});assert(!r.error,label+' unavailable: '+r.error);return r;
   }
   const good=run('actual',snippets);assert.equal(good.status,0,good.stderr);assert.equal(good.stdout,expected,'production Phase-B state differs from independent original C++23 oracle');
   const free={...snippets};for(const k of['parameter','rhs','call','nested'])free[k]=free[k].replaceAll('fn_find(','fn_find_free(');
   const missing={...snippets,rhs:snippets.rhs.replace('if rfj < 0 { rfj = FN_COUNT }','// sentinel conversion omitted')};
   const confidence={...snippets,rhs:snippets.rhs.replace('FN_EFF_CONF[rfj as usize]','1000'),nested:snippets.nested.replace('FN_EFF_CONF[nfj as usize]','1000')};
   const exact='FN_LOOKUP_HASH[i as usize] == h && name_eq(FN_NS[i as usize], FN_NE[i as usize], ns, ne)';assert(helpers.includes(exact));
   const lostZero=helpers.replace(fn(source,'fn_find'),fn(source,'fn_find').replace(exact,exact+' && i > 0'));
   for(const[label,x,h]of[['free-preference',free,helpers],['missing-rhs-sentinel',missing,helpers],['fabricated-confidence',confidence,helpers],['lost-slot-zero',snippets,lostZero]]){const bad=run(label,x,h);assert(bad.status!==0||bad.stdout!==expected,'accepted faulty Phase-B variant '+label);}
   console.log('EPISTEMIC_PHASE_B_ACTUAL_CPP23_STATE_ORACLE_4_MUTANTS_PASS platform='+process.platform);
  }
 }finally{fs.rmSync(tmp,{recursive:true,force:true});}
}
console.log('EPISTEMIC_PHASE_B_LOOKUP_REQUIRED_CONTRACT_PASS');
