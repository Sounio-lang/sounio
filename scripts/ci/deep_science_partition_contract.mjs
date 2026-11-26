#!/usr/bin/env node
// Pin the mandatory partition inventory; this does not validate scientific values.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import cp from 'node:child_process';
import assert from 'node:assert/strict';
const root=process.argv[2];assert(root,'usage: deep_science_partition_contract.mjs REPO [--selftest]');
const workflow=fs.readFileSync(path.join(root,'.github/workflows/ci.yml'),'utf8');
const script=fs.readFileSync(path.join(root,'scripts/stdlib/deep_science_gate.sh'),'utf8');
const full=fs.readFileSync(path.join(root,'scripts/ci/madaros_full_gate.sh'),'utf8');
function check(w,s,f){
 const block=w.split('  stdlib-deep-science:\n')[1]?.split('  gate-wave-0:\n')[0];assert(block,'missing science job');
 assert(block.includes('      fail-fast: false\n      matrix:\n        group: [core, integration]\n'),'both partitions must run even if one fails');
 assert(block.includes('          STDLIB_DEEP_GROUP: ${{ matrix.group }}\n'),'partition not passed');
 assert(block.includes('name: stdlib-deep-science-${{ matrix.group }}-${{ github.sha }}-${{ github.run_attempt }}'),'partition artifact collision');
 assert(s.includes('case "$group" in all|core|integration|reference-controls) ;; *) echo "invalid STDLIB_DEEP_GROUP" >&2; exit 64 ;; esac'),'unknown groups must fail');
 assert(s.includes('if [[ "$group" == all || "$group" == core ]]; then run_core; fi'),'core disconnected');
 assert(s.includes('if [[ "$group" == all || "$group" == integration ]]; then run_integration; fi'),'integration disconnected');
 assert(s.includes('run_cpp_batch\nfor src in tests/compile-fail/stdlib_deep_'),'CPP preparation disconnected');
 assert(s.includes('(( jobs > memory_jobs )) && jobs="$memory_jobs"'),'worker memory bound removed');
 assert(s.indexOf('run_core() {')<s.indexOf('run_native_batch\n'),'native coverage outside core');
 assert(s.indexOf('run_integration() {')>s.indexOf('run_oracle_batch\n'),'oracle pairing split');
 const independent=w.split('  scientific-reference-controls:\n')[1]?.split('  current-source-madaros:\n')[0];
 assert(independent,'missing required independent reference job');
 assert(independent.includes('    needs: [impact, fast-checks]\n'),'reference controls must not wait for compiler');
 assert(!independent.includes('current-source-madaros'),'reference controls acquired a compiler dependency');
 assert(independent.includes('STDLIB_DEEP_GROUP: reference-controls'),'reference mode disconnected');
 assert(s.includes('if [[ "$group" == reference-controls ]]; then run_reference_controls; fi'),'independent dispatch disconnected');
 assert(s.includes('[[ "$group" != core || "$1" != count_dispersion_increment_reference ]]'),'core/reference partition drift');
 assert(s.includes('record arrow-gate scripts/arrow_ipc_gate.sh "$arrow_rc"'),'Arrow gate missing');
 assert(s.includes('record concordance math-pure-ln-boundary "$rc"'),'logarithm gate missing');
 assert(w.includes('      - name: Madaros full gate against the fresh build\n        env:\n          MADAROS_RAW_BIN: /tmp/madaros-ci.elf\n          SOUNIO_MADAROS_DEREF_F64_GATE_BIN: /tmp/madaros-ci.elf\n        run: bash scripts/ci/madaros_full_gate.sh'),'full gate CI must bind both compiler entrypoints');
 assert(f.includes('bash "$ROOT_DIR/scripts/ci/madaros_imported_deref_f64_array_gate.sh"'),'full gate lost dereference witness');
}
check(workflow,script,full);
if(process.argv[3]==='--selftest'){
 const cases=[
 [workflow.replace('group: [core, integration]','group: [core]'),script,full],
 [workflow.replace('fail-fast: false\n      matrix:\n        group:', 'fail-fast: true\n      matrix:\n        group:'),script,full],
 [workflow.replace('STDLIB_DEEP_GROUP: ${{ matrix.group }}','STDLIB_DEEP_GROUP: core'),script,full],
 [workflow.replace('stdlib-deep-science-${{ matrix.group }}-','stdlib-deep-science-'),script,full],
 [workflow,script.replace('then run_integration; fi','then :; fi'),full],
 [workflow.replace('  scientific-reference-controls:', '  unrelated-controls:'),script,full],
 [workflow,script.replace('then run_reference_controls; fi','then :; fi'),full],
 [workflow,script.replace('run_cpp_batch\nfor src in tests/compile-fail/stdlib_deep_',':\nfor src in tests/compile-fail/stdlib_deep_'),full],
 [workflow,script.replace('(( jobs > memory_jobs )) && jobs="$memory_jobs"',':'),full],
 [workflow.replace('SOUNIO_MADAROS_DEREF_F64_GATE_BIN: /tmp/madaros-ci.elf','IGNORED_BIN: /tmp/madaros-ci.elf'),script,full],
 ];
 for(const [i,args] of cases.entries())assert.throws(()=>check(...args),'accepted mutation '+i);
 console.log('DEEP_SCIENCE_PARTITION_MUTATIONS_PASS controls='+cases.length);
}
console.log('DEEP_SCIENCE_PARTITION_CONTRACT_PASS');

// Fixtures exercise process orchestration only, without a compiler or numerical oracle.
if(process.argv[3]==='--selftest'){
 const begin=script.indexOf('# CPP_POOL_BEGIN:');
 const end=script.indexOf('# CPP_POOL_END',begin);
 assert(begin>=0&&end>begin,'missing CPP pool fixture boundary');
 const pool=script.slice(begin,end);
 const fixture=String.raw`set -uo pipefail
out="$TEST_OUT"
jobs=2
fail=0
record() {
 printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$out/fixture.rows"
 [[ "$3" == 0 ]] || fail=1
 return 0
}
compile_cpp() {
 case "$1" in
 bad) record cpp-compile "$1" 1 ;;
 early_exit) exit 77 ;;
 *) record cpp-compile "$1" 0 ;;
 esac
}
`;
 const driver=String.raw`
queue_cpp good_a good_a ''
case "$CPP_FIXTURE_CASE" in
 success) queue_cpp good_b good_b '' ;;
 compile_error) queue_cpp bad bad '' ;;
 early_exit) queue_cpp early_exit early_exit '' ;;
 *) exit 64 ;;
esac
run_cpp_batch
grep -Fq good_a "$out/fixture.rows" || exit 1
if [[ "$CPP_FIXTURE_CASE" == success ]]; then
 [[ "$fail" == 0 ]] || exit 1
else
 [[ "$fail" == 1 ]] || exit 1
 grep -Fq failed-or-missing-status "$out/fixture.rows" || exit 1
fi
`;
 function runPool(body,kind){
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'deep-cpp-pool-'));
  try{return cp.spawnSync('bash',['-s'],{input:fixture+body+driver,env:{...process.env,TEST_OUT:dir,CPP_FIXTURE_CASE:kind},encoding:'utf8'});}
  finally{fs.rmSync(dir,{recursive:true,force:true});}
 }
 for(const kind of ['success','compile_error','early_exit']){
  const r=runPool(pool,kind);assert.equal(r.status,0,'CPP pool '+kind+' failed: '+r.stdout+r.stderr);
 }
 const mutant=pool.replace('record cpp-worker "${CPP_SOURCES[$idx]}:failed-or-missing-status" 1',':');
 assert.notEqual(mutant,pool,'CPP failure mutation did not match');
 assert.notEqual(runPool(mutant,'compile_error').status,0,'lost child failure accepted');
 assert.notEqual(runPool(mutant,'early_exit').status,0,'early child exit accepted without a status');
 console.log('DEEP_CPP_POOL_FAILURE_CONTROLS_PASS');
}

if(process.argv[3]==='--selftest'){
 // Sparse repository and stub toolchain: this proves the control job needs no native compiler.
 const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deep-independent-control-'));
 try{
  const repo=path.join(temp,'repo'), bin=path.join(temp,'toolchain');
  for(const rel of ['scripts/stdlib','tests/reference','stdlib','bin'])fs.mkdirSync(path.join(repo,rel),{recursive:true});
  fs.mkdirSync(bin);
  fs.copyFileSync(path.join(root,'scripts/stdlib/deep_science_gate.sh'),path.join(repo,'scripts/stdlib/deep_science_gate.sh'));
  fs.writeFileSync(path.join(repo,'tests/reference/count_dispersion_increment_reference.cpp'),'// orchestration fixture\n');
  fs.writeFileSync(path.join(repo,'stdlib/fixture.sio'),'// hash fixture only\n');
  fs.writeFileSync(path.join(repo,'bin/souc'),String.raw`#!/usr/bin/env bash
printf 'unexpected native execution\n' > "$FIXTURE_NATIVE_CALLS"
exit 99
`);fs.chmodSync(path.join(repo,'bin/souc'),0o755);
  fs.writeFileSync(path.join(bin,'g++'),String.raw`#!/usr/bin/env bash
set -eu
if [[ "$1" == --version ]]; then echo 'orchestration fixture toolchain'; exit 0; fi
target=''
while [[ $# != 0 ]]; do
 if [[ "$1" == -o ]]; then shift; target="$1"; fi
 shift
done
[[ -n "$target" ]] || exit 88
cat > "$target" <<'PROGRAM'
#!/usr/bin/env bash
case "$*" in --mutant-sign) exit 2 ;; *) exit 0 ;; esac
PROGRAM
chmod +x "$target"
`);fs.chmodSync(path.join(bin,'g++'),0o755);
  cp.execFileSync('git',['-C',repo,'init','-q']);
  cp.execFileSync('git',['-C',repo,'add','.']);
  cp.execFileSync('git',['-C',repo,'-c','user.name=CI Fixture','-c','user.email=ci-fixture@example.invalid','commit','-qm','fixture']);
  const output=path.join(temp,'out'), marker=path.join(temp,'native-called');
  const env={...process.env,PATH:bin+path.delimiter+process.env.PATH,STDLIB_DEEP_GROUP:'reference-controls',STDLIB_DEEP_JOBS:'1',STDLIB_DEEP_OUT:output,FIXTURE_NATIVE_CALLS:marker};
  delete env.MADAROS_RAW_BIN;
  const r=cp.spawnSync('bash',[path.join(repo,'scripts/stdlib/deep_science_gate.sh')],{env,encoding:'utf8'});
  assert.equal(r.status,0,'independent mode failed: '+r.stdout+r.stderr);
  assert.equal(fs.readFileSync(path.join(output,'STATUS'),'utf8'),'0\n');
  assert.equal(fs.readFileSync(path.join(output,'GROUP'),'utf8'),'reference-controls\n');
  assert(!fs.existsSync(marker),'independent controls executed a native compiler');
  assert(!fs.existsSync(path.join(output,'COMPILER.sha256')),'independent controls claimed a native compiler identity');
  assert(fs.readFileSync(path.join(output,'RESULTS.tsv'),'utf8').includes('oracle-worker\ttail:count_dispersion_increment_reference\t0\n'),'missing independent case receipt');
  console.log('DEEP_REFERENCE_WITHOUT_NATIVE_COMPILER_FIXTURE_PASS');
 }finally{fs.rmSync(temp,{recursive:true,force:true});}
}

if(process.argv[3]==='--selftest'){
 // Both scientific jobs install through scripts/ci/apt_ensure.sh. That script is
 // run here against isolated sudo/apt/dpkg fixtures. The Azure mirror must never
 // be selected (it failed every setup, #3547). A failed update must never reach
 // install. A failure on every mirror must propagate. One mid-sync mirror
 // (run 45803197304, 2026-11-26) must not fail the job.
 const setups=[...workflow.matchAll(/      - name: Install independent C\+\+23 reference dependencies\n        run: (.*)\n/g)].map(m=>m[1]);
 assert.equal(setups.length,2,'both scientific jobs need dependency setup coverage');
 for(const s of setups)assert.equal(s,'bash scripts/ci/apt_ensure.sh g++ libboost-dev libquadmath0 jq','scientific dependency setup must go through apt_ensure.sh with the four packages');
 const script=fs.readFileSync(path.join(root,'scripts/ci/apt_ensure.sh'),'utf8');
 assert(!/azure/i.test(script.split('\n').filter(l=>!l.trim().startsWith('#')).join('\n')),'the Azure mirror must not be selectable (#3547)');
 assert.equal((script.match(/timeout [0-9]+ \$SUDO apt-get/g)||[]).length,2,'both package-manager calls must be bounded');
 const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deep-apt-setup-'));
 try{
  const bin=path.join(temp,'bin');fs.mkdirSync(bin);
  const mirrors=path.join(temp,'mirrors'), calls=path.join(temp,'calls'), seen=path.join(temp,'seen');
  fs.writeFileSync(path.join(bin,'sudo'),String.raw`#!/usr/bin/env bash
set -euo pipefail
case "$1" in
 tee)
  [[ "$#" == 2 && "$2" == "$APT_ENSURE_MIRROR_FILE" ]] || exit 80
  cat > "$APT_ENSURE_MIRROR_FILE"; cat "$APT_ENSURE_MIRROR_FILE" >> "$APT_FIXTURE_SEEN"
  ;;
 apt-get)
  shift
  args=" $* "
  for option in Acquire::http::Timeout=30 Acquire::https::Timeout=30 Acquire::Retries=3 APT::Update::Error-Mode=any; do
   [[ "$args" == *" -o $option "* ]] || exit 82
  done
  n=$(grep -c update "$APT_FIXTURE_CALLS" || true)
  if [[ "$args" == *" update "* ]]; then
   echo update >> "$APT_FIXTURE_CALLS"
   case "$APT_FIXTURE_FAIL" in always) exit 73;; once) [[ "$n" -ge 1 ]] || exit 73;; esac
  else
   [[ "$args" == *" install -y --no-install-recommends g++ libboost-dev libquadmath0 jq "* ]] || exit 83
   echo install >> "$APT_FIXTURE_CALLS"
  fi
  ;;
 *) exit 84 ;;
esac
`);
  fs.writeFileSync(path.join(bin,'dpkg-query'),'#!/usr/bin/env bash\n[[ "${APT_FIXTURE_PRESENT:-}" == 1 ]] && { printf "install ok installed"; exit 0; }\nexit 1\n');
  for(const f of ['sudo','dpkg-query'])fs.chmodSync(path.join(bin,f),0o755);
  const cases=[
   {fail:'none',present:'',status:0,calls:'update\ninstall\n'},
   {fail:'once',present:'',status:0,calls:'update\nupdate\ninstall\n'},
   {fail:'always',present:'',status:1,calls:'update\nupdate\nupdate\nupdate\n'},
   {fail:'always',present:'1',status:0,calls:''},
  ];
  for(const c of cases){
   fs.writeFileSync(mirrors,'');fs.writeFileSync(calls,'');fs.writeFileSync(seen,'');
   const r=cp.spawnSync('bash',[path.join(root,'scripts/ci/apt_ensure.sh'),'g++','libboost-dev','libquadmath0','jq'],{encoding:'utf8',timeout:20000,env:{...process.env,PATH:bin+path.delimiter+process.env.PATH,APT_ENSURE_FORCE_SUDO:'1',APT_ENSURE_BACKOFF_SECS:'0',APT_ENSURE_MIRROR_FILE:mirrors,APT_FIXTURE_CALLS:calls,APT_FIXTURE_SEEN:seen,APT_FIXTURE_FAIL:c.fail,APT_FIXTURE_PRESENT:c.present}});
   assert.equal(r.status,c.status,`apt_ensure (${c.fail}, present=${c.present||0}): ${r.stdout}${r.stderr}`);
   assert.equal(fs.readFileSync(calls,'utf8'),c.calls,`apt_ensure (${c.fail}): a failed update must stop before installation`);
   const used=fs.readFileSync(seen,'utf8');
   assert(!/azure/i.test(used),'apt_ensure selected the Azure mirror');
   if(c.calls)assert(used.startsWith('https://archive.ubuntu.com/ubuntu/\n'),'first attempt must use archive.ubuntu.com');
  }
  console.log('DEEP_APT_SETUP_FIXTURE_PASS jobs=2 controls=4');
 }finally{fs.rmSync(temp,{recursive:true,force:true});}
}
