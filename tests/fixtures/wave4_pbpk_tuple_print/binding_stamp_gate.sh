#!/usr/bin/env bash
# Focused native component; this does not rebuild or validate the full checker CLI.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${1:?usage: binding_stamp_gate.sh NEW_OUTPUT_DIRECTORY}
RAW=${MADAROS_RAW_BIN:?provide exact modern raw compiler}
SEED=${LEAN_SINGLE_BIN:?provide exact legacy bootstrap seed}
NODE=${NODE_BIN:-node}
test -x "$RAW"
test -x "$SEED"
test ! -e "$OUT"
mkdir -p "$OUT/source" "$OUT/results"
OUT=$(cd "$OUT" && pwd)
cp -a "$ROOT/self-hosted" "$OUT/source/self-hosted"
"$NODE" - "$ROOT" "$OUT" <<'JS'
const fs=require('fs'),crypto=require('crypto');const[root,out]=process.argv.slice(2);
const original=fs.readFileSync(root+'/self-hosted/native/worker_proof.sio','utf8');
const marker='struct WorkerLowerBinding {';const end=original.indexOf(marker);
if(end<0||original.indexOf(marker,end+1)>=0)throw Error('source profile boundary must be unique');
const declaration='module native::worker_proof\n';let prefix=original.slice(0,end);
if(!prefix.includes(declaration)||prefix.indexOf(declaration)!==prefix.lastIndexOf(declaration))throw Error('module declaration not unique');
prefix=prefix.replace(declaration,'');
let template=fs.readFileSync(root+'/tests/fixtures/wave4_pbpk_tuple_print/binding_stamp_component.sio.in','utf8');
const placeholder='@WORKER_SOURCE_PREFIX@';if(template.split(placeholder).length!==2)throw Error('component placeholder not unique');
fs.writeFileSync(out+'/source/component.sio',template.replace(placeholder,prefix));
fs.writeFileSync(out+'/WORKER_PREFIX.sha256',crypto.createHash('sha256').update(prefix).digest('hex')+'  extracted-source-prefix\n');
fs.writeFileSync(out+'/worker-profile-prefix.sio',prefix);
JS
ulimit -s 524288
ulimit -c 0
cd "$OUT"
find source -type f -print0 | sort -z | xargs -0 sha256sum > SOURCE.sha256
sha256sum "$RAW" "$SEED" "$0" "$ROOT/tests/fixtures/wave4_pbpk_tuple_print/binding_stamp_component.sio.in" > TOOLS.sha256
finish(){
  r=$?
  trap - EXIT
  set +e
  sha256sum -c SOURCE.sha256 > results/source.guard || r=1
  sha256sum -c TOOLS.sha256 > results/tools.guard || r=1
  printf '%s\n' "$r" > results/STATUS
  exit "$r"
}
trap finish EXIT
export SOUNIO_STDLIB_PATH="$OUT/source/self-hosted"
timeout 1200 "$RAW" check source/self-hosted/check/binding_stamp.sio > results/helper.check.stdout 2> results/helper.check.stderr
timeout 1200 "$RAW" check source/component.sio > results/component.check.stdout 2> results/component.check.stderr
(cd source; timeout 1200 "$SEED" component.sio "$OUT/results/component.elf") > results/component.compile.stdout 2> results/component.compile.stderr
if grep -E 'error\[E|parse error|Error:' results/component.compile.stdout results/component.compile.stderr; then exit 65; fi
chmod u+x results/component.elf
sha256sum results/component.elf > ELF.sha256
timeout 60 results/component.elf > results/component.native.stdout 2> results/component.native.stderr
test "$(grep -cxF BINDING_STAMP_COMPONENT_OK results/component.native.stdout)" = 1
sha256sum -c ELF.sha256 > results/elf.guard
printf 'BINDING_STAMP_GATE_OK\n'
