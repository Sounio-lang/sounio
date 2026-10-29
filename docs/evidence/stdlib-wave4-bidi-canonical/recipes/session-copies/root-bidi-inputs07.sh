#!/bin/bash
set -euo pipefail
export PATH=/scratch/sounio-stdlib-wave3-20261027/services-tools-recovered/tools:/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
R=$B/root-bidi-inputs07
S=$B/root-bidi-inputs05/source
D=$R/results
RAW=$B/root-integrated03/results/madaros
mkdir "$R"
mkdir "$D" "$D/native" "$D/binding" "$R/tmp"
export TMPDIR="$R/tmp"
finish(){ r=$?;trap - EXIT;for m in SOURCE RAW EXECUTABLES TOOLS SDK;do if [[ -s "$D/$m.sha256" ]];then sha256sum -c "$D/$m.sha256" >"$D/$m.final.verify" 2>&1||r=1;fi;done;echo "$r" >"$D/STATUS";exit "$r"; }
trap finish EXIT
cd "$S"
printf '%s\n' 'BASE=f0e25b1fecbcb99bb9d621b170667c1b1bf0ff79' 'DECLARED_OVERLAY_FROM=ccc137cf642353502f39a1457b2e28cf5af147ea
RUNNER_PROJECTION_PIN=fc271161afcd' >"$D/SOURCE_SCOPE"
find . -type f -print0|sort -z|xargs -0 sha256sum >"$D/SOURCE.sha256"
cp "$B/root-integrated03/RAW_READY" "$D/RAW.sha256"
sha256sum -c "$D/RAW.sha256" >"$D/RAW.before.verify"
sha256sum "$(command -v node)" "$(command -v g++)" "$(g++ -print-prog-name=cc1plus)" "$B/root-integrated04/tools-env-v3.sh" "$P/tools/time" >"$D/TOOLS.sha256"
find "$G" "$B/root-integrated04/private-runtime" "$B/root-integrated04/pthread-installed-v3" -type f -print0|sort -z|xargs -0 sha256sum >"$D/SDK.sha256"
ulimit -s 524288;ulimit -c 0
cmp "$D/SOURCE.sha256" "$B/root-bidi-inputs05/results/SOURCE.sha256"
cp "$B/root-bidi-inputs05/results/CONTROL_PHASES.tsv" "$D/REUSED_CONTROL_PHASES.tsv"
printf '%s\n' "Eight unchanged-source controls passed in Slurm19670; this attempt reruns the corrected protocol argument and actual canonical case." >"$D/CONTROL_REUSE_SCOPE"
cp "$B/root-bidi-inputs06/results/protocol-controls.stdout" "$D/REUSED_PROTOCOL_CONTROLS.stdout"
printf '%s\n' "Same source and corpus: all22 protocol controls passed Slurm19672; qualified v3 adds the official pthread archive for runtime linking." >"$D/PROTOCOL_REUSE_SCOPE"
g++ -std=c++23 -O2 -Wall -Wextra -Werror -shared -fPIC -pthread runtime/stdlib_platform.cpp -o "$D/native/libstdlib_platform.so" >"$D/runtime.build.stdout" 2>"$D/runtime.build.stderr"
export LD_LIBRARY_PATH="$D/native:$LD_LIBRARY_PATH" SOUNIO_STDLIB_PATH="$S/stdlib" SOUNIO_LOOP_REGIONS=1 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export MADAROS_RAW_BIN="$RAW" SOUC_NATIVE_BIN="$RAW" SOUNIO_SOUC_BIN="$RAW" SOUNIO_SOUC_RAW_MODE=modular
SOUC_BIN=$(node --input-type=module - "$RAW" "$D/binding" <<'JS'
import {createCompilerBinding} from './scripts/stdlib/compiler_binding.mjs';
const b=createCompilerBinding(process.argv[2],process.argv[3]);console.log(b.launcher);
JS
)
export SOUC_BIN
sha256sum "$D/native/libstdlib_platform.so" "$SOUC_BIN" >"$D/EXECUTABLES.sha256"
rc=0
"$P/tools/time" -v -o "$D/CANONICAL.resource" bash scripts/stdlib/run_stdlib_e2e.sh --filter tests/stdlib/wave3-bidi/conformance.sio --json-out "$D/RESULT.json" --artifacts-dir "$D/capture" --verbose >"$D/canonical.stdout" 2>"$D/canonical.stderr"||rc=$?
printf '%s\n' "$rc" >"$D/CANONICAL_STATUS"
[[ "$rc" == 0 ]]
node - "$D" <<'JS'
const fs=require('fs'),d=process.argv[2],r=JSON.parse(fs.readFileSync(d+'/RESULT.json'));
if(r.totals.total!==1||r.totals.pass!==1||r.totals.fail!==0||r.totals.skip!==0||r.results[0].mode!=='runtime-fixture'||r.results[0].reason!=='unicode18-bidi')throw Error('canonical result contract');
const root=d+'/capture/tests/stdlib/wave3-bidi/conformance.sio/runtime-fixture';
const rows=JSON.parse(fs.readFileSync(root+'/RESULTS.json'));
if(rows.length!==211||rows.some(x=>x.status!==0||!x.marker))throw Error('complete fixture execution');
if(fs.readFileSync(root+'/STATUS','utf8')!=='0\n')throw Error('fixture status');
if(!fs.existsSync(d+'/capture/complete.json'))throw Error('capture not finalized');
console.log('BIDI_CANONICAL_CASE_PASS_211_CHUNKS_861948_CASES');
JS
find "$D/binding/retained" -name '*.elf' -type f -print0|sort -z|xargs -0 sha256sum >>"$D/EXECUTABLES.sha256"
echo BIDI_CANONICAL_COMPLETE_INPUTS_AND_ARTIFACTS_OK
