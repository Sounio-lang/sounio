#!/bin/bash
# Run only inside a Slurm allocation. Private metadata injection stays in a
# disposable source snapshot; production main never opts in to Worker issuance.
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Slurm allocation required' >&2; exit 2; }
[[ $# == 3 ]] || { echo 'usage: gate_worker_components.sh FRESH_DIR LEGACY_SEED MODERN_RAW' >&2; exit 2; }
R=$1
SEED=$(realpath "$2")
RAW=$(realpath "$3")
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
[[ ! -e "$R" && -x "$SEED" && -x "$RAW" ]] || { echo 'fresh directory and executable compilers required' >&2; exit 2; }
mkdir -p "$R/source" "$R/results"
S=$R/source
D=$R/results
cp -a "$ROOT/self-hosted" "$S/self-hosted"
cp -a "$ROOT/stdlib" "$S/stdlib"
mkdir -p "$S/tests/fixtures"
cp -a "$ROOT/tests/fixtures/wave4_worker_capability" "$S/tests/fixtures/"
C=$S/tests/fixtures/wave4_worker_capability/components
F=$S/tests/fixtures/wave4_worker_capability
cp "$S/self-hosted/native/worker_proof.sio" "$R/worker_proof.base.sio"
sha256sum "$SEED" "$RAW" "$R/worker_proof.base.sio" > "$D/INPUTS.sha256"
find "$ROOT/self-hosted" "$ROOT/stdlib" "$ROOT/tests/fixtures/wave4_worker_capability" -type f -print0 | sort -z | xargs -0 sha256sum > "$D/ORIGINAL_SOURCE.sha256"
trap 'r=$?;echo "$r" > "$D/STATUS";exit "$r"' EXIT
ulimit -s 524288
ulimit -c 0
export SOUNIO_STDLIB_PATH="$S/self-hosted" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
cd "$S"
compile_legacy() {
 local input=$1 output=$2 log=$3
 timeout 650 "$SEED" "$input" "$output" > "$log" 2>&1
 if grep -Eq '^error:|^error\[|would have dropped|cannot compile an assignment' "$log"; then echo EXPLICIT_BOOTSTRAP_ERROR >&2; exit 1; fi
 [[ -s "$output" ]]
 chmod +x "$output"
}
# Real AST/checker/lowering uses no injected signature or constructor token.
cp "$C/boundary_driver.sio.in" "$F/component_boundary_driver.sio"
for n in good_cap_roundtrip no_ctor_forward no_ctor_raw_arg bad_graph_ctor; do cp "$C/$n.sio.in" "$F/$n.sio"; done
compile_legacy "$F/component_boundary_driver.sio" "$D/boundary.elf" "$D/boundary.compile.stdout"
for n in good_i64 good_f64 good_transitive good_stack_refs good_recursive good_cap_roundtrip; do
 timeout 90 "$D/boundary.elf" "$F/$n.sio" closure > "$D/$n.stdout" 2>&1
 grep -Fxq WORKER_ACTUAL_CHECKER_LOWER_IR_CLONES_OK "$D/$n.stdout"
done
for pair in no_ctor_forward:forward no_ctor_raw_arg:raw bad_graph_ctor:bad_graph; do
 n=${pair%%:*}; mode=${pair#*:}
 timeout 90 "$D/boundary.elf" "$F/$n.sio" "$mode" > "$D/$n.stdout" 2>&1
done
grep -Fxq WORKER_NO_CONSTRUCTOR_FORWARD_CONTRACT_OK "$D/no_ctor_forward.stdout"
grep -Fxq WORKER_RAW_FUNCTION_WITHOUT_CONSTRUCTOR_REFUSED_OK "$D/no_ctor_raw_arg.stdout"
grep -Fxq WORKER_FAILED_CHECKER_GRAPH_CANNOT_ISSUE_TOKEN_OK "$D/bad_graph_ctor.stdout"
# Deliberate post-check IR mutations require private metadata injected solely
# here. This component does not claim a BFS or source-checker result.
cat "$R/worker_proof.base.sio" "$C/ir_body.sio.in" > self-hosted/native/worker_proof.sio
printf '%s\n' 'use native::worker_proof::{worker_ir_component}' 'fn main()->i32 with IO,Mut,Panic,Div,Alloc {worker_ir_component()}' > "$F/component_ir_driver.sio"
timeout 900 "$RAW" check "$F/component_ir_driver.sio" > "$D/ir.check.stdout" 2>&1
compile_legacy "$F/component_ir_driver.sio" "$D/ir.elf" "$D/ir.compile.stdout"
timeout 90 "$D/ir.elf" > "$D/ir.native.stdout" 2>&1
grep -Fxq 'WORKER_CFG_OK corruptions_refused=9 positive_merges_and_loop=3' "$D/ir.native.stdout"
grep -Fxq 'WORKER_IR_CLONES_COMPONENT_OK source_metadata=injected closures=1 original_unchanged=true mutations_refused=15 emitter_enabled=false' "$D/ir.native.stdout"
# Native bytes are tested independently of metadata admission, through the
# actual canonical runtime word-ABI thunk and with writable segments PROT_NONE.
cat "$R/worker_proof.base.sio" "$C/native_body.sio.in" > self-hosted/native/worker_proof.sio
cp "$C/native_driver.sio.in" "$F/component_native_driver.sio"
timeout 900 "$RAW" check "$F/component_native_driver.sio" > "$D/native.check.stdout" 2>&1
compile_legacy "$F/component_native_driver.sio" "$D/producer.elf" "$D/producer.compile.stdout"
timeout 90 "$D/producer.elf" "$D/worker-output.elf" good > "$D/offsets.stdout" 2>&1
grep -Fxq WORKER_NATIVE_COMPONENT_EMITTED_METADATA_INJECTED "$D/offsets.stdout"
g++ -std=c++23 -O2 -fno-omit-frame-pointer -Wall -Wextra -Werror -pedantic "$C/native_reference.cpp" -o "$D/native-reference"
timeout 60 "$D/native-reference" "$D/worker-output.elf" "$D/offsets.stdout" > "$D/native-reference.stdout" 2>&1
for mode in sentinel width unary; do
 rc=0; timeout 60 "$D/producer.elf" "$D/$mode.elf" "$mode" > "$D/$mode.stdout" 2>&1 || rc=$?
 [[ "$rc" == 1 && ! -e "$D/$mode.elf" ]]
 grep -q 'WorkerFn final IR capability or closure proof refused' "$D/$mode.stdout"
 case "$mode" in sentinel) grep -q 'closure=133' "$D/$mode.stdout";;width) grep -q 'closure=136' "$D/$mode.stdout";;unary) grep -q 'closure=134' "$D/$mode.stdout";;esac
done
timeout 60 "$D/producer.elf" "$D/empty.elf" empty > "$D/empty-offsets.stdout" 2>&1
timeout 60 "$D/native-reference" "$D/empty.elf" "$D/empty-offsets.stdout" --empty > "$D/empty-reference.stdout" 2>&1
grep -Fxq 'WORKER_EMPTY_GUARD_OK refused=4 writable_segments=PROT_NONE' "$D/empty-reference.stdout"
objdump -D -b binary -m i386:x86-64 -Mintel --start-address=4096 --stop-address=8192 "$D/worker-output.elf" > "$D/worker-output.disasm"
readelf -l "$D/worker-output.elf" > "$D/worker-output.segments"
sha256sum -c "$D/INPUTS.sha256" > "$D/INPUTS.final.verify"
sha256sum -c "$D/ORIGINAL_SOURCE.sha256" > "$D/ORIGINAL_SOURCE.final.verify"
sha256sum "$D"/*.elf > "$D/ELFS.sha256"
echo WORKER_COMPONENTS_OK_PUBLIC_MAIN_CLOSED_BFS_AND_REAL_CALLBACKS_PENDING
